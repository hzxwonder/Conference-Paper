## [200] Weisfeiler and Lehman Go Cellular: CW Networks

**Authors**: *Cristian Bodnar, Fabrizio Frasca, Nina Otter, Yuguang Wang, Pietro Liò, Guido F. Montúfar, Michael M. Bronstein*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/157792e4abb490f99dbd738483e0d2d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/157792e4abb490f99dbd738483e0d2d4-Abstract.html)

**Abstract**:

Graph Neural Networks (GNNs) are limited in their expressive power, struggle with long-range interactions and lack a principled way to model higher-order structures. These problems can be attributed to the strong coupling between the computational graph and the input graph structure. The recently proposed Message Passing Simplicial Networks naturally decouple these elements by performing message passing on the clique complex of the graph. Nevertheless, these models can be severely constrained by the rigid combinatorial structure of Simplicial Complexes (SCs). In this work, we extend recent theoretical results on SCs to regular Cell Complexes, topological objects that flexibly subsume SCs and graphs. We show that this generalisation provides a powerful set of graph "lifting" transformations, each leading to a unique hierarchical message passing procedure. The resulting methods, which we collectively call CW Networks (CWNs), are strictly more powerful than the WL test and not less powerful than the 3-WL test. In particular, we demonstrate the effectiveness of one such scheme, based on rings, when applied to molecular graph problems. The proposed architecture benefits from provably larger expressivity than commonly used GNNs, principled modelling of higher-order signals and from compressing the distances between nodes. We demonstrate that our model achieves state-of-the-art results on a variety of molecular datasets.

----

## [201] Learning Conjoint Attentions for Graph Neural Nets

**Authors**: *Tiantian He, Yew Soon Ong, Lu Bai*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1587965fb4d4b5afe8428a4a024feb0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1587965fb4d4b5afe8428a4a024feb0d-Abstract.html)

**Abstract**:

In this paper, we present Conjoint Attentions (CAs), a class of novel learning-to-attend strategies for graph neural networks (GNNs). Besides considering the layer-wise node features propagated within the GNN, CAs can additionally incorporate various structural interventions, such as node cluster embedding, and higher-order structural correlations that can be learned outside of GNN, when computing attention scores. The node features that are regarded as significant by the conjoint criteria are therefore more likely to be propagated in the GNN. Given the novel Conjoint Attention strategies, we then propose Graph conjoint attention networks (CATs) that can learn representations embedded with significant latent features deemed by the Conjoint Attentions. Besides, we theoretically validate the discriminative capacity of CATs.  CATs utilizing the proposed Conjoint Attention strategies have been extensively tested in well-established benchmarking datasets and comprehensively compared with state-of-the-art baselines. The obtained notable performance demonstrates the effectiveness of the proposed Conjoint Attentions.

----

## [202] Hybrid Regret Bounds for Combinatorial Semi-Bandits and Adversarial Linear Bandits

**Authors**: *Shinji Ito*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/15a50c8ba6a0002a2fa7e5d8c0a40bd9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/15a50c8ba6a0002a2fa7e5d8c0a40bd9-Abstract.html)

**Abstract**:

This study aims to develop bandit algorithms that automatically exploit tendencies of certain environments to improve performance, without any prior knowledge regarding the environments. We first propose an algorithm for combinatorial semi-bandits with a hybrid regret bound that includes two main features: a best-of-three-worlds guarantee and multiple data-dependent regret bounds. The former means that the algorithm will work nearly optimally in all environments in an adversarial setting, a stochastic setting, or a stochastic setting with adversarial corruptions. The latter implies that, even if the environment is far from exhibiting stochastic behavior, the algorithm will perform better as long as the environment is "easy" in terms of certain metrics. The metrics w.r.t. the easiness referred to in this paper include cumulative loss for optimal actions, total quadratic variation of losses, and path-length of a loss sequence. We also show hybrid data-dependent regret bounds for adversarial linear bandits, which include a first path-length regret bound that is tight up to logarithmic factors.

----

## [203] Pay Better Attention to Attention: Head Selection in Multilingual and Multi-Domain Sequence Modeling

**Authors**: *Hongyu Gong, Yun Tang, Juan Miguel Pino, Xian Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/15c00b5250ddedaabc203b67f8b034fd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/15c00b5250ddedaabc203b67f8b034fd-Abstract.html)

**Abstract**:

Multi-head attention has each of the attention heads collect salient information from different parts of an input sequence, making it a powerful mechanism for sequence modeling. Multilingual and multi-domain learning are common scenarios for sequence modeling, where the key challenge is to maximize positive transfer and mitigate negative interference across languages and domains. In this paper, we find that non-selective attention sharing is sub-optimal for achieving good generalization across all languages and domains. We further propose attention sharing strategies to facilitate parameter sharing and specialization in multilingual and multi-domain sequence modeling. Our approach automatically learns shared and specialized attention heads for different languages and domains. Evaluated in various tasks including speech recognition, text-to-text and speech-to-text translation, the proposed attention sharing strategies consistently bring gains to sequence models built upon multi-head attention. For speech-to-text translation, our approach yields an average of $+2.0$ BLEU over $13$ language directions in multilingual setting and $+2.0$ BLEU over $3$ domains in multi-domain setting.

----

## [204] Cardinality-Regularized Hawkes-Granger Model

**Authors**: *Tsuyoshi Idé, Georgios Kollias, Dzung T. Phan, Naoki Abe*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/15cf76466b97264765356fcc56d801d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/15cf76466b97264765356fcc56d801d1-Abstract.html)

**Abstract**:

We propose a new sparse Granger-causal learning framework for temporal event data. We focus on a specific class of point processes called the Hawkes process. We begin by pointing out that most of the existing sparse causal learning algorithms for the Hawkes process suffer from a singularity in maximum likelihood estimation. As a result, their sparse solutions can appear only as numerical artifacts. In this paper, we propose a mathematically well-defined sparse causal learning framework based on a cardinality-regularized Hawkes process, which remedies the pathological issues of existing approaches. We leverage the proposed algorithm for the task of instance-wise causal event analysis, where sparsity plays a critical role. We validate the proposed framework with two real use-cases, one from the power grid and the other from the cloud data center management domain.

----

## [205] Aligned Structured Sparsity Learning for Efficient Image Super-Resolution

**Authors**: *Yulun Zhang, Huan Wang, Can Qin, Yun Fu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/15de21c670ae7c3f6f3f1f37029303c9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/15de21c670ae7c3f6f3f1f37029303c9-Abstract.html)

**Abstract**:

Lightweight image super-resolution (SR) networks have obtained promising results with moderate model size. Many SR methods have focused on designing lightweight architectures, which neglect to further reduce the redundancy of network parameters. On the other hand, model compression techniques, like neural architecture search and knowledge distillation, typically consume considerable memory and computation resources. In contrast, network pruning is a cheap and effective model compression technique. However, it is hard to be applied to SR networks directly, because filter pruning for residual blocks is well-known tricky. To address the above issues, we propose aligned structured sparsity learning (ASSL), which introduces a weight normalization layer and applies $L_2$ regularization to the scale parameters for sparsity. To align the pruned locations across different layers, we propose a \emph{sparsity structure alignment} penalty term, which minimizes the norm of soft mask gram matrix. We apply aligned structured sparsity learning strategy to train efficient image SR network, named as ASSLN, with smaller model size and lower computation than state-of-the-art methods. We conduct extensive comparisons with lightweight SR networks. Our ASSLN achieves superior performance gains over recent methods quantitatively and visually.

----

## [206] Why Lottery Ticket Wins? A Theoretical Perspective of Sample Complexity on Sparse Neural Networks

**Authors**: *Shuai Zhang, Meng Wang, Sijia Liu, Pin-Yu Chen, Jinjun Xiong*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/15f99f2165aa8c86c9dface16fefd281-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/15f99f2165aa8c86c9dface16fefd281-Abstract.html)

**Abstract**:

The lottery ticket hypothesis (LTH) states that learning on a properly pruned network (the winning ticket) has improved test accuracy over the original unpruned network. Although LTH has been justified empirically in a broad range of deep neural network (DNN) involved applications like computer vision and natural language processing, the theoretical validation of the improved generalization of a winning ticket remains elusive. To the best of our knowledge, our work, for the first time, characterizes the performance of training a pruned neural network by analyzing the geometric structure of the objective function and the sample complexity to achieve zero generalization error. We show that the convex region near a desirable model with guaranteed generalization enlarges as the neural network model is pruned, indicating the structural importance of a winning ticket. Moreover, as the algorithm for training a pruned neural network is specified as an (accelerated) stochastic gradient descent algorithm, we theoretically show that the number of samples required for achieving zero generalization error is proportional to the number of the non-pruned weights in the hidden layer. With a fixed number of samples, training a pruned neural network enjoys a faster convergence rate to the desired model than training the original unpruned one, providing a formal justification of the improved generalization of the winning ticket. Our theoretical results are acquired from learning a pruned neural network of one hidden layer, while experimental results are further provided to justify the implications in pruning multi-layer neural networks.

----

## [207] Constrained Robust Submodular Partitioning

**Authors**: *Shengjie Wang, Tianyi Zhou, Chandrashekhar Lavania, Jeff A. Bilmes*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/161882dd2d19c716819081aee2c08b98-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/161882dd2d19c716819081aee2c08b98-Abstract.html)

**Abstract**:

In the robust submodular partitioning problem, we aim to allocate a set of items into $m$ blocks, so that the evaluation of the minimum block according to a submodular function is maximized. Robust submodular partitioning promotes the diversity of every block in the partition. It has many applications in machine learning, e.g., partitioning data for distributed training so that the gradients computed on every block are consistent. We study an extension of the robust submodular partition problem with additional constraints (e.g., cardinality, multiple matroids, and/or knapsack) on every block. For example, when partitioning data for distributed training, we can add a constraint that the number of samples of each class is the same in each partition block, ensuring data balance. We present two classes of algorithms, i.e., Min-Block Greedy based algorithms (with an $\Omega(1/m)$ bound), and Round-Robin Greedy based algorithms (with a constant bound) and show that under various constraints, they still have good approximation guarantees. Interestingly, while normally the latter runs in only weakly polynomial time, we show that using the two together yields strongly polynomial running time while preserving the approximation guarantee. Lastly, we apply the algorithms on a real-world machine learning data partitioning problem showing good results.

----

## [208] Online Knapsack with Frequency Predictions

**Authors**: *Sungjin Im, Ravi Kumar, Mahshid Montazer Qaem, Manish Purohit*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/161c5c5ad51fcc884157890511b3c8b0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/161c5c5ad51fcc884157890511b3c8b0-Abstract.html)

**Abstract**:

There has been recent interest in using machine-learned predictions to improve the worst-case guarantees of online algorithms.  In this paper we continue this line of work by studying the online knapsack problem, but with very weak predictions: in the form of knowing an upper and lower bound for the number of items of each value.  We systematically derive online algorithms that attain the best possible competitive ratio for any fixed prediction; we also extend the results to more general settings such as generalized one-way trading and two-stage online knapsack. Our work shows that even seemingly weak predictions can be utilized effectively to provably improve the performance of online algorithms.

----

## [209] On Component Interactions in Two-Stage Recommender Systems

**Authors**: *Jiri Hron, Karl Krauth, Michael I. Jordan, Niki Kilbertus*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/162d18156abe38a3b32851b72b1d44f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/162d18156abe38a3b32851b72b1d44f5-Abstract.html)

**Abstract**:

Thanks to their scalability, two-stage recommenders are used by many of today's largest online platforms, including YouTube, LinkedIn, and Pinterest. These systems produce recommendations in two steps: (i) multiple nominators—tuned for low prediction latency—preselect a small subset of candidates from the whole item pool; (ii) a slower but more accurate ranker further narrows down the nominated items, and serves to the user. Despite their popularity, the literature on two-stage recommenders is relatively scarce, and the algorithms are often treated as mere sums of their parts. Such treatment presupposes that the two-stage performance is explained by the behavior of the individual components in isolation. This is not the case: using synthetic and real-world data, we demonstrate that interactions between the ranker and the nominators substantially affect the overall performance. Motivated by these findings, we derive a generalization lower bound which shows that independent nominator training can lead to performance on par with uniformly random recommendations. We find that careful design of item pools, each assigned to a different nominator, alleviates these issues. As manual search for a good pool allocation is difficult, we propose to learn one instead using a Mixture-of-Experts based approach. This significantly improves both precision and recall at $K$.

----

## [210] Lip to Speech Synthesis with Visual Context Attentional GAN

**Authors**: *Minsu Kim, Joanna Hong, Yong Man Ro*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16437d40c29a1a7b1e78143c9c38f289-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16437d40c29a1a7b1e78143c9c38f289-Abstract.html)

**Abstract**:

In this paper, we propose a novel lip-to-speech generative adversarial network, Visual Context Attentional GAN (VCA-GAN), which can jointly model local and global lip movements during speech synthesis. Specifically, the proposed VCA-GAN synthesizes the speech from local lip visual features by finding a mapping function of viseme-to-phoneme, while global visual context is embedded into the intermediate layers of the generator to clarify the ambiguity in the mapping induced by homophene. To achieve this, a visual context attention module is proposed where it encodes global representations from the local visual features, and provides the desired global visual context corresponding to the given coarse speech representation to the generator through audio-visual attention. In addition to the explicit modelling of local and global visual representations, synchronization learning is introduced as a form of contrastive learning that guides the generator to synthesize a speech in sync with the given input lip movements. Extensive experiments demonstrate that the proposed VCA-GAN outperforms existing state-of-the-art and is able to effectively synthesize the speech from multi-speaker that has been barely handled in the previous works.

----

## [211] Non-convex Distributionally Robust Optimization: Non-asymptotic Analysis

**Authors**: *Jikai Jin, Bohang Zhang, Haiyang Wang, Liwei Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/164bf317ea19ccfd9e97853edc2389f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/164bf317ea19ccfd9e97853edc2389f4-Abstract.html)

**Abstract**:

Distributionally robust optimization (DRO) is a widely-used approach to learn models that are robust against distribution shift. Compared with the standard optimization setting, the objective function in DRO is more difficult to optimize, and most of the existing theoretical results make strong assumptions on the loss function. In this work we bridge the gap by studying DRO algorithms for general smooth non-convex losses. By carefully exploiting the specific form of the DRO objective, we are able to provide non-asymptotic convergence guarantees even though the objective function is possibly non-convex, non-smooth and has unbounded gradient noise. In particular, we prove that a special algorithm called the mini-batch normalized gradient descent with momentum, can find an $\epsilon$-first-order stationary point within $\mathcal O(\epsilon^{-4})$ gradient complexity. We also discuss the conditional value-at-risk (CVaR) setting, where we propose a penalized DRO objective based on a smoothed version of the CVaR that allows us to obtain a similar convergence guarantee. We finally verify our theoretical results in a number of tasks and find that the proposed algorithm can consistently achieve prominent acceleration.

----

## [212] Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning

**Authors**: *Kibeom Kim, Min Whoo Lee, Yoonsung Kim, Je-Hwan Ryu, Min Su Lee, Byoung-Tak Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/165a59f7cf3b5c4396ba65953d679f17-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/165a59f7cf3b5c4396ba65953d679f17-Abstract.html)

**Abstract**:

Learning in a multi-target environment without prior knowledge about the targets requires a large amount of samples and makes generalization difficult. To solve this problem, it is important to be able to discriminate targets through semantic understanding. In this paper, we propose goal-aware cross-entropy (GACE) loss, that can be utilized in a self-supervised way using auto-labeled goal states alongside reinforcement learning. Based on the loss, we then devise goal-discriminative attention networks (GDAN) which utilize the goal-relevant information to focus on the given instruction. We evaluate the proposed methods on visual navigation and robot arm manipulation tasks with multi-target environments and show that GDAN outperforms the state-of-the-art methods in terms of task success ratio, sample efficiency, and generalization. Additionally, qualitative analyses demonstrate that our proposed method can help the agent become aware of and focus on the given instruction clearly, promoting goal-directed behavior.

----

## [213] Smooth Normalizing Flows

**Authors**: *Jonas Köhler, Andreas Krämer, Frank Noé*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/167434fa6219316417cd4160c0c5e7d2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/167434fa6219316417cd4160c0c5e7d2-Abstract.html)

**Abstract**:

Normalizing flows are a promising tool for modeling probability distributions in physical systems. While state-of-the-art flows accurately approximate distributions and energies, applications in physics additionally require smooth energies to compute forces and higher-order derivatives. Furthermore, such densities are often defined on non-trivial topologies. A recent example are Boltzmann Generators for generating 3D-structures of peptides and small proteins. These generative models leverage the space of internal coordinates (dihedrals, angles, and bonds), which is a product of hypertori and compact intervals. In this work, we introduce a class of smooth mixture transformations working on both compact intervals and hypertori.Mixture transformations employ root-finding methods to invert them in practice, which has so far prevented bi-directional flow training. To this end, we show that parameter gradients and forces of such inverses can be computed from forward evaluations via the inverse function theorem.We demonstrate two advantages of such smooth flows: they allow training by force matching to simulation data and can be used as potentials in molecular dynamics simulations.

----

## [214] MetaAvatar: Learning Animatable Clothed Human Models from Few Depth Images

**Authors**: *Shaofei Wang, Marko Mihajlovic, Qianli Ma, Andreas Geiger, Siyu Tang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1680829293f2a8541efa2647a0290f88-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1680829293f2a8541efa2647a0290f88-Abstract.html)

**Abstract**:

In this paper, we aim to create generalizable and controllable neural signed distance fields (SDFs) that represent clothed humans from monocular depth observations. Recent advances in deep learning, especially neural implicit representations, have enabled human shape reconstruction and controllable avatar generation from different sensor inputs. However, to generate realistic cloth deformations from novel input poses, watertight meshes or dense full-body scans are usually needed as inputs. Furthermore, due to the difficulty of effectively modeling pose-dependent cloth deformations for diverse body shapes and cloth types, existing approaches resort to per-subject/cloth-type optimization from scratch, which is computationally expensive. In contrast, we propose an approach that can quickly generate realistic clothed human avatars, represented as controllable neural SDFs, given only monocular depth images. We achieve this by using meta-learning to learn an initialization of a hypernetwork that predicts the parameters of neural SDFs. The hypernetwork is conditioned on human poses and represents a clothed neural avatar that deforms non-rigidly according to the input poses. Meanwhile, it is meta-learned to effectively incorporate priors of diverse body shapes and cloth types and thus can be much faster to fine-tune, compared to models trained from scratch. We qualitatively and quantitatively show that our approach outperforms state-of-the-art approaches that require complete meshes as inputs while our approach requires only depth frames as inputs and runs orders of magnitudes faster. Furthermore, we demonstrate that our meta-learned hypernetwork is very robust, being the first to generate avatars with realistic dynamic cloth deformations given as few as 8 monocular depth frames.

----

## [215] Distributed Principal Component Analysis with Limited Communication

**Authors**: *Foivos Alimisis, Peter Davies, Bart Vandereycken, Dan Alistarh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1680e9fa7b4dd5d62ece800239bb53bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1680e9fa7b4dd5d62ece800239bb53bd-Abstract.html)

**Abstract**:

We study efficient distributed algorithms for the fundamental problem of principal component analysis and leading eigenvector computation on the sphere, when the data are randomly distributed among a set of computational nodes. We propose a new quantized variant of Riemannian gradient descent to solve this problem, and prove that the algorithm converges with high probability under a set of necessary spherical-convexity properties. We give bounds on the number of bits transmitted by the algorithm under common initialization schemes, and investigate the dependency on the problem dimension in each case.

----

## [216] Newton-LESS: Sparsification without Trade-offs for the Sketched Newton Update

**Authors**: *Michal Derezinski, Jonathan Lacotte, Mert Pilanci, Michael W. Mahoney*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16837163fee34175358a47e0b51485ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16837163fee34175358a47e0b51485ff-Abstract.html)

**Abstract**:

In second-order optimization, a potential bottleneck can be computing the Hessian matrix of the optimized function at every iteration. Randomized sketching has emerged as a powerful technique for constructing estimates of the Hessian which can be used to perform approximate Newton steps. This involves multiplication by a random sketching matrix, which introduces a trade-off between the computational cost of sketching and the convergence rate of the optimization. A theoretically desirable but practically much too expensive choice is to use a dense Gaussian sketching matrix, which produces unbiased estimates of the exact Newton step and offers strong problem-independent convergence guarantees. We show that the Gaussian matrix can be drastically sparsified, substantially reducing the computational cost, without affecting its convergence properties in any way. This approach, called Newton-LESS, is based on a recently introduced sketching technique: LEverage Score Sparsified (LESS) embeddings. We prove that Newton-LESS enjoys nearly the same problem-independent local convergence rate as Gaussian embeddings for a large class of functions. In particular, this leads to a new state-of-the-art convergence result for an iterative least squares solver. Finally, we substantially extend LESS embeddings to include uniformly sparsified random sign matrices which can be implemented efficiently and perform well in numerical experiments.

----

## [217] Confident Anchor-Induced Multi-Source Free Domain Adaptation

**Authors**: *Jiahua Dong, Zhen Fang, Anjin Liu, Gan Sun, Tongliang Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/168908dd3227b8358eababa07fcaf091-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/168908dd3227b8358eababa07fcaf091-Abstract.html)

**Abstract**:

Unsupervised domain adaptation has attracted appealing academic attentions by transferring knowledge from labeled source domain to unlabeled target domain. However, most existing methods assume the source data are drawn from a single domain, which cannot be successfully applied to explore complementarily transferable knowledge from multiple source domains with large distribution discrepancies. Moreover, they require access to source data during training, which are inefficient and unpractical due to privacy preservation and memory storage. To address these challenges, we develop a novel Confident-Anchor-induced multi-source-free Domain Adaptation (CAiDA) model, which is a pioneer exploration of knowledge adaptation from multiple source domains to the unlabeled target domain without any source data, but with only pre-trained source models. Specifically, a source-specific transferable perception module is proposed to automatically quantify the contributions of the complementary knowledge transferred from multi-source domains to the target domain. To generate pseudo labels for the target domain without access to the source data, we develop a confident-anchor-induced pseudo label generator by constructing a confident anchor group and assigning each unconfident target sample with a semantic-nearest confident anchor. Furthermore, a class-relationship-aware consistency loss is proposed to preserve consistent inter-class relationships by aligning soft confusion matrices across domains. Theoretical analysis answers why multi-source domains are better than a single source domain, and establishes a novel learning bound to show the effectiveness of exploiting multi-source domains. Experiments on several representative datasets illustrate the superiority of our proposed CAiDA model. The code is available at https://github.com/Learning-group123/CAiDA.

----

## [218] Word2Fun: Modelling Words as Functions for Diachronic Word Representation

**Authors**: *Benyou Wang, Emanuele Di Buccio, Massimo Melucci*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16a5cdae362b8d27a1d8f8c7b78b4330-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16a5cdae362b8d27a1d8f8c7b78b4330-Abstract.html)

**Abstract**:

Word meaning may change over time as a reflection of changes in human society. Therefore, modeling time in word representation is necessary for some diachronic tasks. Most existing diachronic word representation approaches train the embeddings separately for each pre-grouped time-stamped corpus and align these embeddings, e.g., by orthogonal projections, vector initialization, temporal referencing, and compass. However, not only does word meaning change in a short time, word meaning may also be subject to evolution over long timespans, thus resulting in a unified continuous process. A recent approach called `DiffTime'  models semantic evolution as functions parameterized by multiple-layer nonlinear neural networks over time. In this paper, we will carry on this line of work by learning explicit functions over time  for each word. Our approach, called `Word2Fun', reduces the space complexity from $\mathcal{O}(TVD)$ to $\mathcal{O}(kVD)$ where $k$  is a small constant ($k \ll T $). In particular, a specific instance based on polynomial functions could provably approximate any function modeling word evolution with a given negligible error thanks to the Weierstrass Approximation Theorem. The effectiveness of the proposed approach is evaluated in diverse tasks including time-aware word clustering, temporal analogy, and semantic change detection. Code at: {\url{https://github.com/wabyking/Word2Fun.git}}.

----

## [219] Iteratively Reweighted Least Squares for Basis Pursuit with Global Linear Convergence Rate

**Authors**: *Christian Kümmerle, Claudio Mayrink Verdun, Dominik Stöger*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16bda725ae44af3bb9316f416bd13b1b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16bda725ae44af3bb9316f416bd13b1b-Abstract.html)

**Abstract**:

The recovery of sparse data is at the core of many applications in machine learning and signal processing. While such problems can be tackled using $\ell_1$-regularization as in the LASSO estimator and in the Basis Pursuit approach, specialized algorithms are typically required to solve the corresponding high-dimensional non-smooth optimization for large instances.Iteratively Reweighted Least Squares (IRLS) is a widely used algorithm for this purpose due to its excellent numerical performance. However, while existing theory is able to guarantee convergence of this algorithm to the minimizer, it does not provide a global convergence rate. In this paper, we prove that a variant of IRLS converges \emph{with a global linear rate} to a sparse solution, i.e., with a linear error decrease occurring immediately from any initialization if the measurements fulfill the usual null space property assumption. We support our theory by numerical experiments showing that our linear rate captures the correct dimension dependence. We anticipate that our theoretical findings will lead to new insights for many other use cases of the IRLS algorithm, such as in low-rank matrix recovery.

----

## [220] Low-Rank Constraints for Fast Inference in Structured Models

**Authors**: *Justin T. Chiu, Yuntian Deng, Alexander M. Rush*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16c0d78ef6a76b5c247113a4c9514059-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16c0d78ef6a76b5c247113a4c9514059-Abstract.html)

**Abstract**:

Structured distributions, i.e. distributions over combinatorial spaces, are commonly used to learn latent probabilistic representations from observed data. However, scaling these models is bottlenecked by the high computational and memory complexity with respect to the size of the latent representations. Common models such as Hidden Markov Models (HMMs) and Probabilistic Context-Free Grammars (PCFGs) require time and space quadratic and cubic in the number of hidden states respectively. This work demonstrates a simple approach to reduce the computational and memory complexity of a large class of structured models. We show that by viewing the central inference step as a matrix-vector product and using a low-rank constraint, we can trade off model expressivity and speed via the rank.  Experiments with neural parameterized structured models for language modeling, polyphonic music modeling, unsupervised grammar induction, and video modeling show that our approach matches the accuracy of standard models at large state spaces while providing practical speedups.

----

## [221] Accumulative Poisoning Attacks on Real-time Data

**Authors**: *Tianyu Pang, Xiao Yang, Yinpeng Dong, Hang Su, Jun Zhu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16d11e9595188dbad0418a85f0351aba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16d11e9595188dbad0418a85f0351aba-Abstract.html)

**Abstract**:

Collecting training data from untrusted sources exposes machine learning services to poisoning adversaries, who maliciously manipulate training data to degrade the model accuracy. When trained on offline datasets, poisoning adversaries have to inject the poisoned data in advance before training, and the order of feeding these poisoned batches into the model is stochastic. In contrast, practical systems are more usually trained/fine-tuned on sequentially captured real-time data, in which case poisoning adversaries could dynamically poison each data batch according to the current model state. In this paper, we focus on the real-time settings and propose a new attacking strategy, which affiliates an accumulative phase with poisoning attacks to secretly (i.e., without affecting accuracy) magnify the destructive effect of a (poisoned) trigger batch. By mimicking online learning and federated learning on MNIST and CIFAR-10, we show that model accuracy significantly drops by a single update step on the trigger batch after the accumulative phase. Our work validates that a well-designed but straightforward attacking strategy can dramatically amplify the poisoning effects, with no need to explore complex techniques.

----

## [222] UCB-based Algorithms for Multinomial Logistic Regression Bandits

**Authors**: *Sanae Amani, Christos Thrampoulidis*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16f852a6d01b6065c8ff5cc11caae9c6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16f852a6d01b6065c8ff5cc11caae9c6-Abstract.html)

**Abstract**:

Out of the rich family of generalized linear bandits, perhaps the most well studied ones are logistic bandits that are used in problems with binary rewards: for instance, when the learner aims to maximize the profit over a user that can select one of two possible outcomes (e.g., `click' vs `no-click'). Despite remarkable recent progress and improved algorithms for logistic bandits, existing works do not address practical situations where the number of outcomes that can be selected by the user is larger than two (e.g., `click', `show me later', `never show again', `no click'). In this paper, we study such an extension. We use multinomial logit (MNL) to model the probability of each one of $K+1\geq 2$ possible outcomes (+1 stands for the `not click' outcome): we assume that for a learner's action $\mathbf{x}_t$, the user selects one of $K+1\geq 2$ outcomes, say outcome $i$, with a MNL probabilistic model with corresponding unknown parameter $\bar{\boldsymbol{\theta}}_{\ast i}$. Each outcome $i$ is also associated with a revenue parameter $\rho_i$ and the goal is to maximize the expected revenue. For this problem, we present MNL-UCB, an upper confidence bound (UCB)-based algorithm, that achieves regret $\tilde{\mathcal{O}}(dK\sqrt{T})$ with small dependency on problem-dependent constants that can otherwise be arbitrarily large and lead to loose regret bounds. We present numerical simulations that corroborate our theoretical results.

----

## [223] Estimating the Long-Term Effects of Novel Treatments

**Authors**: *Keith Battocchi, Eleanor Dillon, Maggie Hei, Greg Lewis, Miruna Oprescu, Vasilis Syrgkanis*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/16fa2b0294e410b2551c3bf6965c0853-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/16fa2b0294e410b2551c3bf6965c0853-Abstract.html)

**Abstract**:

Policy makers often need to estimate the long-term effects of novel treatments, while only having historical data of older treatment options. We propose a surrogate-based approach using a long-term dataset where only past treatments were administered and a short-term dataset where novel treatments have been administered. Our approach generalizes previous surrogate-style methods, allowing for continuous treatments and serially-correlated treatment policies while maintaining consistency and root-n asymptotically normal estimates under a Markovian assumption on the data and the observational policy. Using a semi-synthetic dataset on customer incentives from a major corporation, we evaluate the performance of our method and discuss solutions to practical challenges when deploying our methodology.

----

## [224] Dual Progressive Prototype Network for Generalized Zero-Shot Learning

**Authors**: *Chaoqun Wang, Shaobo Min, Xuejin Chen, Xiaoyan Sun, Houqiang Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1700002963a49da13542e0726b7bb758-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1700002963a49da13542e0726b7bb758-Abstract.html)

**Abstract**:

Generalized Zero-Shot Learning (GZSL) aims to recognize new categories with auxiliary semantic information, e.g., category attributes. In this paper, we handle the critical issue of domain shift problem, i.e., confusion between seen and unseen categories, by progressively improving cross-domain transferability and category discriminability of visual representations. Our approach, named Dual Progressive Prototype Network (DPPN), constructs two types of prototypes that record prototypical visual patterns for attributes and categories, respectively. With attribute prototypes, DPPN alternately searches attribute-related local regions and updates corresponding attribute prototypes to progressively explore accurate attribute-region correspondence. This enables DPPN to produce visual representations with accurate attribute localization ability, which benefits the semantic-visual alignment and representation transferability. Besides, along with progressive attribute localization, DPPN further projects category prototypes into multiple spaces to progressively repel visual representations from different categories, which boosts category discriminability. Both attribute and category prototypes are collaboratively learned in a unified framework, which makes visual representations of DPPN transferable and distinctive.Experiments on four benchmarks prove that DPPN effectively alleviates the domain shift problem in GZSL.

----

## [225] Derivative-Free Policy Optimization for Linear Risk-Sensitive and Robust Control Design: Implicit Regularization and Sample Complexity

**Authors**: *Kaiqing Zhang, Xiangyuan Zhang, Bin Hu, Tamer Basar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1714726c817af50457d810aae9d27a2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1714726c817af50457d810aae9d27a2e-Abstract.html)

**Abstract**:

Direct policy search serves as one of the workhorses in modern reinforcement learning (RL), and its applications in continuous control tasks have recently attracted increasing attention. In this work, we investigate the convergence theory of policy gradient (PG) methods for learning the linear risk-sensitive and robust controller. In particular, we develop PG methods that can be implemented in a derivative-free fashion by sampling system trajectories, and establish both global convergence and sample complexity results in the solutions of two fundamental settings in risk-sensitive and robust control: the finite-horizon linear exponential quadratic Gaussian, and the finite-horizon linear-quadratic disturbance attenuation problems. As a by-product, our results also provide the first sample complexity for the global convergence of PG methods on solving zero-sum linear-quadratic dynamic games, a nonconvex-nonconcave minimax optimization problem that serves as a baseline setting in multi-agent reinforcement learning (MARL) with continuous spaces. One feature of our algorithms is that during the learning phase, a certain level of robustness/risk-sensitivity of the controller is preserved, which we termed as the implicit regularization property, and is an essential requirement in safety-critical control systems.

----

## [226] G-PATE: Scalable Differentially Private Data Generator via Private Aggregation of Teacher Discriminators

**Authors**: *Yunhui Long, Boxin Wang, Zhuolin Yang, Bhavya Kailkhura, Aston Zhang, Carl A. Gunter, Bo Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/171ae1bbb81475eb96287dd78565b38b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/171ae1bbb81475eb96287dd78565b38b-Abstract.html)

**Abstract**:

Recent advances in machine learning have largely benefited from the massive accessible training data. However, large-scale data sharing has raised great privacy concerns. In this work, we propose a novel privacy-preserving data Generative model based on the PATE framework (G-PATE), aiming to train a scalable differentially private data generator that preserves high generated data utility. Our approach leverages generative adversarial nets to generate data, combined with private aggregation among different discriminators to ensure strong privacy guarantees. Compared to existing approaches, G-PATE significantly improves the use of privacy budgets. In particular, we train a student data generator with an ensemble of teacher discriminators and propose a novel private gradient aggregation mechanism to ensure differential privacy on all information that flows from teacher discriminators to the student generator. In addition, with random projection and gradient discretization, the proposed gradient aggregation mechanism is able to effectively deal with high-dimensional gradient vectors. Theoretically, we prove that G-PATE ensures differential privacy for the data generator.  Empirically, we demonstrate the superiority of G-PATE over prior work through extensive experiments. We show that G-PATE is the first work being able to generate high-dimensional image data with high data utility under limited privacy budgets ($\varepsilon \le 1$). Our code is available at https://github.com/AI-secure/G-PATE.

----

## [227] On the Existence of The Adversarial Bayes Classifier

**Authors**: *Pranjal Awasthi, Natalie Frank, Mehryar Mohri*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/172ef5a94b4dd0aa120c6878fc29f70c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/172ef5a94b4dd0aa120c6878fc29f70c-Abstract.html)

**Abstract**:

Adversarial robustness is a critical property in a variety of modern machine learning applications. While it has been the subject of several recent theoretical studies, many important questions related to adversarial robustness are still open.  In this work, we study a fundamental question regarding Bayes optimality for adversarial robustness. We provide general sufficient conditions under which the existence of a Bayes optimal classifier can be guaranteed for adversarial robustness. Our results can provide a useful tool for a subsequent study of surrogate losses in adversarial robustness and their consistency properties.

----

## [228] Convex-Concave Min-Max Stackelberg Games

**Authors**: *Denizalp Goktas, Amy Greenwald*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/174a61b0b3eab8c94e0a9e78b912307f-Abstract.html)

**Abstract**:

Min-max optimization problems (i.e., min-max games) have been attracting a great deal of attention because of their applicability to a wide range of machine learning problems. Although significant progress has been made recently, the literature to date has focused on games with independent strategy sets; little is known about solving games with dependent strategy sets, which can be characterized as min-max Stackelberg games. We introduce two first-order methods that solve a large class of convex-concave min-max Stackelberg games, and show that our methods converge in polynomial time. Min-max Stackelberg games were first studied by Wald, under the posthumous name of Waldâ€™s maximin model, a variant of which is the main paradigm used in robust optimization, which means that our methods can likewise solve many convex robust optimization problems. We observe that the computation of competitive equilibria in Fisher markets also comprises a min-max Stackelberg game.  Further, we demonstrate the efficacy and efficiency of our algorithms in practice by computing competitive equilibria in Fisher markets with varying utility structures. Our experiments suggest potential ways to extend our theoretical results, by demonstrating how different smoothness properties can affect the convergence rate of our algorithms.

----

## [229] Misspecified Gaussian Process Bandit Optimization

**Authors**: *Ilija Bogunovic, Andreas Krause*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/177db6acfe388526a4c7bff88e1feb15-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/177db6acfe388526a4c7bff88e1feb15-Abstract.html)

**Abstract**:

We consider the problem of optimizing a black-box function based on noisy bandit feedback. Kernelized bandit algorithms have shown strong empirical and theoretical performance for this problem. They heavily rely on the assumption that the model is well-specified, however, and can fail without it. Instead, we introduce and address a \emph{misspecified} kernelized bandit setting where the unknown function can be $\epsilon$--uniformly approximated by a function with a bounded norm in some Reproducing Kernel Hilbert Space (RKHS). We design efficient and practical algorithms whose performance degrades minimally in the presence of model misspecification. Specifically, we present two algorithms based on Gaussian process (GP) methods: an optimistic EC-GP-UCB algorithm that requires knowing the misspecification error, and Phased GP Uncertainty Sampling, an elimination-type algorithm that can adapt to unknown model misspecification. We provide upper bounds on their cumulative regret in terms of $\epsilon$, the time horizon, and the underlying kernel, and we show that our algorithm achieves optimal dependence on $\epsilon$ with no prior knowledge of misspecification. In addition, in a stochastic contextual setting, we show that EC-GP-UCB can be effectively combined with the regret bound balancing strategy and attain similar regret bounds despite not knowing $\epsilon$.

----

## [230] Visual Adversarial Imitation Learning using Variational Models

**Authors**: *Rafael Rafailov, Tianhe Yu, Aravind Rajeswaran, Chelsea Finn*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1796a48fa1968edd5c5d10d42c7b1813-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1796a48fa1968edd5c5d10d42c7b1813-Abstract.html)

**Abstract**:

Reward function specification, which requires considerable human effort and iteration, remains a major impediment for learning behaviors through deep reinforcement learning. In contrast, providing visual demonstrations of desired behaviors presents an easier and more natural way to teach agents. We consider a setting where an agent is provided a fixed dataset of visual demonstrations illustrating how to perform a task, and must learn to solve the task using the provided demonstrations and unsupervised environment interactions. This setting presents a number of challenges including representation learning for visual observations, sample complexity due to high dimensional spaces, and learning instability due to the lack of a fixed reward or learning signal. Towards addressing these challenges, we develop a variational model-based adversarial imitation learning (V-MAIL) algorithm. The model-based approach provides a strong signal for representation learning, enables sample efficiency, and improves the stability of adversarial training by enabling on-policy learning. Through experiments involving several vision-based locomotion and manipulation tasks, we find that V-MAIL learns successful visuomotor policies in a sample-efficient manner, has better stability compared to prior work, and also achieves higher asymptotic performance. We further find that by transferring the learned models, V-MAIL can learn new tasks from visual demonstrations without any additional environment interactions. All results including videos can be found online at https://sites.google.com/view/variational-mail

----

## [231] Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning

**Authors**: *Jongjin Park, Younggyo Seo, Chang Liu, Li Zhao, Tao Qin, Jinwoo Shin, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/17a3120e4e5fbdc3cb5b5f946809b06a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/17a3120e4e5fbdc3cb5b5f946809b06a-Abstract.html)

**Abstract**:

Behavioral cloning has proven to be effective for learning sequential decision-making policies from expert demonstrations. However, behavioral cloning often suffers from the causal confusion problem where a policy relies on the noticeable effect of expert actions due to the strong correlation but not the cause we desire. This paper presents Object-aware REgularizatiOn (OREO), a simple technique that regularizes an imitation policy in an object-aware manner. Our main idea is to encourage a policy to uniformly attend to all semantic objects, in order to prevent the policy from exploiting nuisance variables strongly correlated with expert actions. To this end, we introduce a two-stage approach: (a) we extract semantic objects from images by utilizing discrete codes from a vector-quantized variational autoencoder, and (b) we randomly drop the units that share the same discrete code together, i.e., masking out semantic objects. Our experiments demonstrate that OREO significantly improves the performance of behavioral cloning, outperforming various other regularization and causality-based methods on a variety of Atari environments and a self-driving CARLA environment. We also show that our method even outperforms inverse reinforcement learning methods trained with a considerable amount of environment interaction.

----

## [232] Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection

**Authors**: *Chunjong Park, Anas Awadalla, Tadayoshi Kohno, Shwetak N. Patel*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/17e23e50bedc63b4095e3d8204ce063b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/17e23e50bedc63b4095e3d8204ce063b-Abstract.html)

**Abstract**:

Unpredictable ML model behavior on unseen data, especially in the health domain, raises serious concerns about its safety as repercussions for mistakes can be fatal. In this paper, we explore the feasibility of using state-of-the-art out-of-distribution detectors for reliable and trustworthy diagnostic predictions. We select publicly available deep learning models relating to various health conditions (e.g., skin cancer, lung sound, and Parkinson's disease) using various input data types (e.g., image, audio, and motion data). We demonstrate that these models show unreasonable predictions on out-of-distribution datasets. We show that Mahalanobis distance- and Gram matrices-based out-of-distribution detection methods are able to detect out-of-distribution data with high accuracy for the health models that operate on different modalities. We then translate the out-of-distribution score into a human interpretable \textsc{confidence score} to investigate its effect on the users' interaction with health ML applications. Our user study shows that the \textsc{confidence score} helped the participants only trust the results with a high score to make a medical decision and disregard results with a low score. Through this work, we demonstrate that dataset shift is a critical piece of information for high-stake ML applications, such as medical diagnosis and healthcare, to provide reliable and trustworthy predictions to the users.

----

## [233] Multiclass Boosting and the Cost of Weak Learning

**Authors**: *Nataly Brukhim, Elad Hazan, Shay Moran, Indraneel Mukherjee, Robert E. Schapire*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/17f5e6db87929fb55cebeb7fd58c1d41-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/17f5e6db87929fb55cebeb7fd58c1d41-Abstract.html)

**Abstract**:

Boosting is an algorithmic approach which is based on the idea     of combining weak and moderately inaccurate hypotheses to a strong and accurate one.     In this work we study multiclass boosting with a possibly large number of classes or categories.    Multiclass boosting can be formulated in various ways.    Here, we focus on an especially natural formulation in which the weak hypotheses    are assumed to belong to an ''easy-to-learn'' base class, and    the weak learner is an agnostic PAC learner for that class    with respect to the standard classification loss.    This is in contrast with other, more complicated losses as have often been considered in the past.    The goal of the overall boosting algorithm    is then to learn a combination of weak hypotheses    by repeatedly calling the weak learner.We study the resources required for boosting, especially how theydepend on the number of classes $k$, for both the booster and weak learner.We find that the boosting algorithm itself only requires $O(\log k)$samples, as we show by analyzing a variant of AdaBoost for oursetting. In stark contrast, assuming typical limits on the number of weak-learner calls,we prove that the number of samples required by a weak learner is at least polynomial in $k$, exponentially more than thenumber of samples needed by the booster.Alternatively, we prove that the weak learner's accuracy parametermust be smaller  than an inverse polynomial in $k$, showing that the returned weakhypotheses must be nearly the best in their class when $k$ is large.We also prove a trade-off between number of oracle calls and theresources required of the weak learner, meaning that the fewer calls to theweak learner the more that is demanded on each call.

----

## [234] Partition-Based Formulations for Mixed-Integer Optimization of Trained ReLU Neural Networks

**Authors**: *Calvin Tsay, Jan Kronqvist, Alexander Thebelt, Ruth Misener*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/17f98ddf040204eda0af36a108cbdea4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/17f98ddf040204eda0af36a108cbdea4-Abstract.html)

**Abstract**:

This paper introduces a class of mixed-integer formulations for trained ReLU neural networks. The approach balances model size and tightness by partitioning node inputs into a number of groups and forming the convex hull over the partitions via disjunctive programming. At one extreme, one partition per input recovers the convex hull of a node, i.e., the tightest possible formulation for each node. For fewer partitions, we develop smaller relaxations that approximate the convex hull, and show that they outperform existing formulations. Specifically, we propose strategies for partitioning variables based on theoretical motivations and validate these strategies using extensive computational experiments. Furthermore, the proposed scheme complements known algorithmic approaches, e.g., optimization-based bound tightening captures dependencies within a partition.

----

## [235] Hyperparameter Optimization Is Deceiving Us, and How to Stop It

**Authors**: *A. Feder Cooper, Yucheng Lu, Jessica Forde, Christopher De Sa*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/17fafe5f6ce2f1904eb09d2e80a4cbf6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/17fafe5f6ce2f1904eb09d2e80a4cbf6-Abstract.html)

**Abstract**:

Recent empirical work shows that inconsistent results based on choice of hyperparameter optimization (HPO) configuration are a widespread problem in ML research. When comparing two algorithms J and K searching one subspace can yield the conclusion that J outperforms K, whereas searching another can entail the opposite. In short, the way we choose hyperparameters can deceive us. We provide a theoretical complement to this prior work, arguing that, to avoid such deception, the process of drawing conclusions from HPO should be made more rigorous. We call this process epistemic hyperparameter optimization (EHPO), and put forth a logical framework to capture its semantics and how it can lead to inconsistent conclusions about performance. Our framework enables us to prove EHPO methods that are guaranteed to be defended against deception, given bounded compute time budget t. We demonstrate our framework's utility by proving and empirically validating a defended variant of random search.

----

## [236] On the Convergence Theory of Debiased Model-Agnostic Meta-Reinforcement Learning

**Authors**: *Alireza Fallah, Kristian Georgiev, Aryan Mokhtari, Asuman E. Ozdaglar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/18085327b86002fc604c323b9a07f997-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/18085327b86002fc604c323b9a07f997-Abstract.html)

**Abstract**:

We consider Model-Agnostic Meta-Learning (MAML) methods for Reinforcement Learning (RL) problems, where the goal is to find a policy using data from several tasks represented by Markov Decision Processes (MDPs) that can be updated by one step of \textit{stochastic} policy gradient for the realized MDP. In particular, using stochastic gradients in MAML update steps is crucial for RL problems since computation of exact gradients requires access to a large number of possible trajectories. For this formulation, we propose a variant of the MAML method, named Stochastic Gradient Meta-Reinforcement Learning (SG-MRL), and study its convergence properties. We derive the iteration and sample complexity of SG-MRL to find an $\epsilon$-first-order stationary point, which, to the best of our knowledge, provides the first convergence guarantee for model-agnostic meta-reinforcement learning algorithms. We further show how our results extend to the case where more than one step of stochastic policy gradient method is used at test time. Finally, we empirically compare SG-MRL and MAML in several deep RL environments.

----

## [237] 3D Pose Transfer with Correspondence Learning and Mesh Refinement

**Authors**: *Chaoyue Song, Jiacheng Wei, Ruibo Li, Fayao Liu, Guosheng Lin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/18a411989b47ed75a60ac69d9da05aa5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/18a411989b47ed75a60ac69d9da05aa5-Abstract.html)

**Abstract**:

3D pose transfer is one of the most challenging 3D generation tasks. It aims to transfer the pose of a source mesh to a target mesh and keep the identity (e.g., body shape) of the target mesh. Some previous works require key point annotations to build reliable correspondence between the source and target meshes, while other methods do not consider any shape correspondence between sources and targets, which leads to limited generation quality. In this work, we propose a correspondence-refinement network to achieve the 3D pose transfer for both human and animal meshes. The correspondence between source and target meshes is first established by solving an optimal transport problem. Then, we warp the source mesh according to the dense correspondence and obtain a coarse warped mesh. The warped mesh will be better refined with our proposed Elastic Instance Normalization, which is a conditional normalization layer and can help to generate high-quality meshes. Extensive experimental results show that the proposed architecture can effectively transfer the poses from source to target meshes and produce better results with satisfied visual performance than state-of-the-art methods.

----

## [238] Framing RNN as a kernel method: A neural ODE approach

**Authors**: *Adeline Fermanian, Pierre Marion, Jean-Philippe Vert, Gérard Biau*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/18a9042b3fc5b02fe3d57fea87d6992f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/18a9042b3fc5b02fe3d57fea87d6992f-Abstract.html)

**Abstract**:

Building on the interpretation of a recurrent neural network (RNN) as a continuous-time neural differential equation, we show, under appropriate conditions, that the solution of a RNN can be viewed as a linear function of a specific feature set of the input sequence, known as the signature. This connection allows us to frame a RNN as a kernel method in a suitable reproducing kernel Hilbert space. As a consequence, we obtain theoretical guarantees on generalization and stability for a large class of recurrent networks. Our results are illustrated on simulated datasets.

----

## [239] Contextual Similarity Aggregation with Self-attention for Visual Re-ranking

**Authors**: *Jianbo Ouyang, Hui Wu, Min Wang, Wengang Zhou, Houqiang Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/18d10dc6e666eab6de9215ae5b3d54df-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/18d10dc6e666eab6de9215ae5b3d54df-Abstract.html)

**Abstract**:

In content-based image retrieval, the first-round retrieval result by simple visual feature comparison may be unsatisfactory, which can be refined by visual re-ranking techniques. In image retrieval, it is observed that the contextual similarity among the top-ranked images is an important clue to distinguish the semantic relevance. Inspired by this observation, in this paper, we propose a visual re-ranking method by contextual similarity aggregation with self-attention.  In our approach, for each image in the top-K ranking list, we represent it into an affinity feature vector by comparing it with a set of anchor images. Then, the affinity features of the top-K images are refined by aggregating the contextual information with a transformer encoder. Finally, the affinity features are used to recalculate the similarity scores between the query and the top-K images for re-ranking of the latter. To further improve the robustness of our re-ranking model and enhance the performance of our method, a new data augmentation scheme is designed. Since our re-ranking model is not directly involved with the visual feature used in the initial retrieval, it is ready to be applied to retrieval result lists obtained from various retrieval algorithms. We conduct comprehensive experiments on four benchmark datasets to demonstrate the generality and effectiveness of our proposed visual re-ranking method.

----

## [240] Can Information Flows Suggest Targets for Interventions in Neural Circuits?

**Authors**: *Praveen Venkatesh, Sanghamitra Dutta, Neil Mehta, Pulkit Grover*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/18de4beb01f6a17b6e1dfb9813ba6045-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/18de4beb01f6a17b6e1dfb9813ba6045-Abstract.html)

**Abstract**:

Motivated by neuroscientific and clinical applications, we empirically examine whether observational measures of information flow can suggest interventions. We do so by performing experiments on artificial neural networks in the context of fairness in machine learning, where the goal is to induce fairness in the system through interventions. Using our recently developed M-information flow framework, we measure the flow of information about the true label (responsible for accuracy, and hence desirable), and separately, the flow of information about a protected attribute (responsible for bias, and hence undesirable) on the edges of a trained neural network. We then compare the flow magnitudes against the effect of intervening on those edges by pruning. We show that pruning edges that carry larger information flows about the protected attribute reduces bias at the output to a greater extent. This demonstrates that M-information flow can meaningfully suggest targets for interventions, answering the title's question in the affirmative. We also evaluate bias-accuracy tradeoffs for different intervention strategies, to analyze how one might use estimates of desirable and undesirable information flows (here, accuracy and bias flows) to inform interventions that preserve the former while reducing the latter.

----

## [241] AutoBalance: Optimized Loss Functions for Imbalanced Data

**Authors**: *Mingchen Li, Xuechen Zhang, Christos Thrampoulidis, Jiasi Chen, Samet Oymak*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/191f8f858acda435ae0daf994e2a72c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/191f8f858acda435ae0daf994e2a72c2-Abstract.html)

**Abstract**:

Imbalanced datasets are commonplace in modern machine learning problems. The presence of under-represented classes or groups with sensitive attributes results in concerns about generalization and fairness. Such concerns are further exacerbated by the fact that large capacity deep nets can perfectly fit the training data and appear to achieve perfect accuracy and fairness during training, but perform poorly during test. To address these challenges, we propose AutoBalance, a bi-level optimization framework that automatically designs a training loss function to optimize a blend of accuracy and fairness-seeking objectives. Specifically, a lower-level problem trains the model weights, and an upper-level problem tunes the loss function by monitoring and optimizing the desired objective over the validation data. Our loss design enables personalized treatment for classes/groups by employing a parametric cross-entropy loss and individualized data augmentation schemes. We evaluate the benefits and performance of our approach for the application scenarios of imbalanced and group-sensitive classification. Extensive empirical evaluations demonstrate the benefits of AutoBalance over state-of-the-art approaches. Our experimental findings are complemented with theoretical insights on loss function design and the benefits of the train-validation split. All code is available open-source.

----

## [242] SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes

**Authors**: *Zhaozhi Qian, Yao Zhang, Ioana Bica, Angela M. Wood, Mihaela van der Schaar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/19485224d128528da1602ca47383f078-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/19485224d128528da1602ca47383f078-Abstract.html)

**Abstract**:

Most of the medical observational studies estimate the causal treatment effects using electronic health records (EHR), where a patient's covariates and outcomes are both observed longitudinally. However, previous methods focus only on adjusting for the covariates while neglecting the temporal structure in the outcomes. To bridge the gap, this paper develops a new method, SyncTwin, that learns a patient-specific time-constant representation from the pre-treatment observations. SyncTwin issues counterfactual prediction of a target patient by constructing a synthetic twin that closely matches the target in representation. The reliability of the estimated treatment effect can be assessed by comparing the observed and synthetic pre-treatment outcomes. The medical experts can interpret the estimate by examining the most important contributing individuals to the synthetic twin. In the real-data experiment, SyncTwin successfully reproduced the findings of a randomized controlled clinical trial using observational data, which demonstrates its usability in the complex real-world EHR.

----

## [243] Statistical Query Lower Bounds for List-Decodable Linear Regression

**Authors**: *Ilias Diakonikolas, Daniel Kane, Ankit Pensia, Thanasis Pittas, Alistair Stewart*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/19b1b73d63d4c9ea79f8ca57e9d67095-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/19b1b73d63d4c9ea79f8ca57e9d67095-Abstract.html)

**Abstract**:

We study the problem of list-decodable linear regression, where an adversary can corrupt a majority of the examples. Specifically, we are given a set $T$ of labeled examples $(x, y) \in \mathbb{R}^d \times \mathbb{R}$ and a parameter $0< \alpha <1/2$ such that an $\alpha$-fraction of the points in $T$ are i.i.d. samples from a linear regression model with Gaussian covariates, and the remaining $(1-\alpha)$-fraction of the points are drawn from an arbitrary noise distribution. The goal is to output a small list of hypothesis vectors such that at least one of them is close to the target regression vector. Our main result is a Statistical Query (SQ) lower bound of $d^{\mathrm{poly}(1/\alpha)}$ for this problem. Our SQ lower bound qualitatively matches the performance of previously developed algorithms, providing evidence that current upper bounds for this task are nearly best possible.

----

## [244] Unsupervised Motion Representation Learning with Capsule Autoencoders

**Authors**: *Ziwei Xu, Xudong Shen, Yongkang Wong, Mohan S. Kankanhalli*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/19ca14e7ea6328a42e0eb13d585e4c22-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/19ca14e7ea6328a42e0eb13d585e4c22-Abstract.html)

**Abstract**:

We propose the Motion Capsule Autoencoder (MCAE), which addresses a key challenge in the unsupervised learning of motion representations: transformation invariance. MCAE models motion in a two-level hierarchy. In the lower level, a spatio-temporal motion signal is divided into short, local, and semantic-agnostic snippets. In the higher level, the snippets are aggregated to form full-length semantic-aware segments. For both levels, we represent motion with a set of learned transformation invariant templates and the corresponding geometric transformations by using capsule autoencoders of a novel design. This leads to a robust and efficient encoding of viewpoint changes. MCAE is evaluated on a novel Trajectory20 motion dataset and various real-world skeleton-based human action datasets. Notably, it achieves better results than baselines on Trajectory20 with considerably fewer parameters and state-of-the-art performance on the unsupervised skeleton-based action recognition task.

----

## [245] VigDet: Knowledge Informed Neural Temporal Point Process for Coordination Detection on Social Media

**Authors**: *Yizhou Zhang, Karishma Sharma, Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1a344877f11195aaf947ccfe48ee9c89-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1a344877f11195aaf947ccfe48ee9c89-Abstract.html)

**Abstract**:

Recent years have witnessed an increasing use of coordinated accounts on social media, operated by misinformation campaigns to influence public opinion and manipulate social outcomes. Consequently, there is an urgent need to develop an effective methodology for coordinated group detection to combat the misinformation on social media. However, existing works suffer from various drawbacks, such as, either limited performance due to extreme reliance on predefined signatures of coordination, or instead an inability to address the natural sparsity of account activities on social media with useful prior domain knowledge. Therefore, in this paper, we propose a coordination detection framework incorporating neural temporal point process with prior knowledge such as temporal logic or pre-defined filtering functions. Specifically, when modeling the observed data from social media with neural temporal point process, we jointly learn a Gibbs-like distribution of group assignment based on how consistent an assignment is to (1) the account embedding space and (2) the prior knowledge. To address the challenge that the distribution is hard to be efficiently computed and sampled from, we design a theoretically guaranteed variational inference approach to learn a mean-field approximation for it. Experimental results on a real-world dataset show the effectiveness of our proposed method compared to the SOTA model in both unsupervised and semi-supervised settings. We further apply our model on a COVID-19 Vaccine Tweets dataset. The detection result suggests the presence of suspicious coordinated efforts on spreading misinformation about COVID-19 vaccines.

----

## [246] An Improved Analysis and Rates for Variance Reduction under Without-replacement Sampling Orders

**Authors**: *Xinmeng Huang, Kun Yuan, Xianghui Mao, Wotao Yin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1a3650aedfdd3a21444047ed2d89458f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1a3650aedfdd3a21444047ed2d89458f-Abstract.html)

**Abstract**:

When applying a stochastic algorithm, one must choose an order to draw samples. The practical choices are without-replacement sampling orders, which are empirically faster and more cache-friendly than uniform-iid-sampling but often have inferior theoretical guarantees. Without-replacement sampling is well understood only for SGD without variance reduction. In this paper, we will improve the convergence analysis and rates of variance reduction under without-replacement sampling orders for composite finite-sum minimization.Our results are in two-folds. First, we develop a damped variant of Finito called Prox-DFinito and  establish its convergence rates with random reshuffling, cyclic sampling, and shuffling-once, under both generally and strongly convex scenarios. These rates match full-batch gradient descent and are state-of-the-art compared to the existing results for without-replacement sampling with variance-reduction. Second, our analysis can gauge how the cyclic order will influence the rate of cyclic sampling and, thus, allows us to derive the optimal fixed ordering. In the highly data-heterogeneous scenario, Prox-DFinito with optimal cyclic sampling can attain a sample-size-independent convergence rate, which, to our knowledge, is the first result that can match with uniform-iid-sampling with variance reduction. We also propose a practical method to discover the optimal cyclic ordering numerically.

----

## [247] Exploring Forensic Dental Identification with Deep Learning

**Authors**: *Yuan Liang, Weikun Han, Liang Qiu, Chen Wu, Yiting Shao, Kun Wang, Lei He*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1a423f7c07a179ec243e82b0c017a034-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1a423f7c07a179ec243e82b0c017a034-Abstract.html)

**Abstract**:

Dental forensic identification targets to identify persons with dental traces.The task is vital for the investigation of criminal scenes and mass disasters because of the resistance of dental structures and the wide-existence of dental imaging. However, no widely accepted automated solution is available for this labour-costly task. In this work, we pioneer to study deep learning for dental forensic identification based on panoramic radiographs. We construct a comprehensive benchmark with various dental variations that can adequately reflect the difficulties of the task. By considering the task's unique challenges, we propose FoID, a deep learning method featured by: (\textit{i}) clinical-inspired attention localization, (\textit{ii}) domain-specific augmentations that enable instance discriminative learning, and (\textit{iii}) transformer-based self-attention mechanism that dynamically reasons the relative importance of attentions. We show that FoID can outperform traditional approaches by at least \textbf{22.98\%} in terms of Rank-1 accuracy, and outperform strong CNN baselines by at least \textbf{10.50\%} in terms of mean Average Precision (mAP). Moreover, extensive ablation studies verify the effectiveness of each building blocks of FoID. Our work can be a first step towards the automated system for forensic identification among large-scale multi-site databases. Also, the proposed techniques, \textit{e.g.}, self-attention mechanism, can also be meaningful for other identification tasks, \textit{e.g.}, pedestrian re-identification.Related data and codes can be found at \href{https://github.com/liangyuandg/FoID}{https://github.com/liangyuandg/FoID}.

----

## [248] Learning to Generate Realistic Noisy Images via Pixel-level Noise-aware Adversarial Training

**Authors**: *Yuanhao Cai, Xiaowan Hu, Haoqian Wang, Yulun Zhang, Hanspeter Pfister, Donglai Wei*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1a5b1e4daae265b790965a275b53ae50-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1a5b1e4daae265b790965a275b53ae50-Abstract.html)

**Abstract**:

Existing deep learning real denoising methods require a large amount of noisy-clean image pairs for supervision. Nonetheless, capturing a  real noisy-clean dataset is an unacceptable expensive and cumbersome procedure. To alleviate this problem, this work investigates how to generate realistic noisy images. Firstly, we formulate a simple yet reasonable noise model that treats each real noisy pixel as a random variable. This model splits the noisy image generation problem into two sub-problems: image domain alignment and noise domain alignment. Subsequently, we propose a novel framework, namely Pixel-level Noise-aware Generative Adversarial Network (PNGAN). PNGAN employs a pre-trained real denoiser to map the fake and real noisy images into a nearly noise-free solution space to perform image domain alignment. Simultaneously, PNGAN establishes a pixel-level adversarial training to conduct noise domain alignment. Additionally, for better noise fitting, we present an efficient architecture Simple Multi-scale Network (SMNet) as the generator. Qualitative validation shows that noise generated by PNGAN is highly similar to real noise in terms of intensity and distribution. Quantitative experiments demonstrate that a series of denoisers trained with the generated noisy images achieve state-of-the-art (SOTA) results on four real denoising benchmarks.

----

## [249] Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks

**Authors**: *Jianhong Wang, Wangkun Xu, Yunjie Gu, Wenbin Song, Tim C. Green*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1a6727711b84fd1efbb87fc565199d13-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1a6727711b84fd1efbb87fc565199d13-Abstract.html)

**Abstract**:

This paper presents a problem in power networks that creates an exciting and yet challenging real-world scenario for application of multi-agent reinforcement learning (MARL). The emerging trend of decarbonisation is placing excessive stress on power distribution networks. Active voltage control is seen as a promising solution to relieve power congestion and improve voltage quality without extra hardware investment, taking advantage of the controllable apparatuses in the network, such as roof-top photovoltaics (PVs) and static var compensators (SVCs). These controllable apparatuses appear in a vast number and are distributed in a wide geographic area, making MARL a natural candidate. This paper formulates the active voltage control problem in the framework of Dec-POMDP and establishes an open-source environment. It aims to bridge the gap between the power community and the MARL community and be a drive force towards real-world applications of MARL algorithms. Finally, we analyse the special characteristics of the active voltage control problems that cause challenges (e.g. interpretability) for state-of-the-art MARL approaches, and summarise the potential directions.

----

## [250] Looking Beyond Single Images for Contrastive Semantic Segmentation Learning

**Authors**: *Feihu Zhang, Philip H. S. Torr, René Ranftl, Stephan R. Richter*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1a68e5f4ade56ed1d4bf273e55510750-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1a68e5f4ade56ed1d4bf273e55510750-Abstract.html)

**Abstract**:

We present an approach to contrastive representation learning for semantic segmentation. Our approach leverages the representational power of existing feature extractors to find corresponding regions across images. These cross-image correspondences are used as auxiliary labels to guide the pixel-level selection of  positive and negative samples for more effective contrastive learning in semantic segmentation. We show that auxiliary labels can be generated from a variety of feature extractors, ranging from image classification networks that have been trained using unsupervised contrastive learning to segmentation models that have been trained on a small amount of labeled data. We additionally introduce a novel metric for rapidly judging the quality of a given auxiliary-labeling strategy, and empirically analyze various factors that influence the performance of contrastive learning for semantic segmentation. We demonstrate the effectiveness of our method both in the low-data as well as the high-data regime on various datasets. Our experiments show that contrastive learning with our auxiliary-labeling approach consistently boosts semantic segmentation accuracy when compared to standard ImageNet pretraining and outperforms existing approaches of contrastive and semi-supervised semantic segmentation.

----

## [251] A Constant Approximation Algorithm for Sequential Random-Order No-Substitution k-Median Clustering

**Authors**: *Tom Hess, Michal Moshkovitz, Sivan Sabato*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1aa057313c28fa4a40c5bc084b11d276-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1aa057313c28fa4a40c5bc084b11d276-Abstract.html)

**Abstract**:

We study k-median clustering under the sequential no-substitution setting. In this setting, a data stream is sequentially observed, and some of the points are selected by the algorithm as cluster centers. However, a point can be selected as a center only immediately after it is observed, before observing the next point. In addition, a selected center cannot be substituted later. We give the first algorithm for this setting that obtains a constant approximation factor on the optimal cost under a random arrival order, an exponential improvement over previous work. This is also the first constant approximation guarantee that holds without any structural assumptions on the input data. Moreover, the number of selected centers is only quasi-linear in k.  Our algorithm and analysis are based on a careful cost estimation that avoids outliers, a new concept of a linear bin division, and a multi-scale approach to center selection.

----

## [252] Dangers of Bayesian Model Averaging under Covariate Shift

**Authors**: *Pavel Izmailov, Patrick Nicholson, Sanae Lotfi, Andrew Gordon Wilson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1ab60b5e8bd4eac8a7537abb5936aadc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1ab60b5e8bd4eac8a7537abb5936aadc-Abstract.html)

**Abstract**:

Approximate Bayesian inference for neural networks is considered a robust alternative to standard training, often providing good performance on out-of-distribution data. However, Bayesian neural networks (BNNs) with high-fidelity approximate inference via full-batch Hamiltonian Monte Carlo achieve poor generalization under covariate shift, even underperforming classical estimation. We explain this surprising result, showing how a Bayesian model average can in fact be problematic under covariate shift, particularly in cases where linear dependencies in the input features cause a lack of posterior contraction. We additionally show why the same issue does not affect many approximate inference procedures, or classical maximum a-posteriori (MAP) training. Finally, we propose novel priors that improve the robustness of BNNs to many sources of covariate shift.

----

## [253] Learning Equilibria in Matching Markets from Bandit Feedback

**Authors**: *Meena Jagadeesan, Alexander Wei, Yixin Wang, Michael I. Jordan, Jacob Steinhardt*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1b89a2e980724cb8997459fadb907712-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1b89a2e980724cb8997459fadb907712-Abstract.html)

**Abstract**:

Large-scale, two-sided matching platforms must find market outcomes that align with user preferences while simultaneously learning these preferences from data. But since preferences are inherently uncertain during learning, the classical notion of stability (Gale and Shapley, 1962; Shapley and Shubik, 1971) is unattainable in these settings. To bridge this gap, we develop a framework and algorithms for learning stable market outcomes under uncertainty. Our primary setting is matching with transferable utilities, where the platform both matches agents and sets monetary transfers between them. We design an incentive-aware learning objective that captures the distance of a market outcome from equilibrium. Using this objective, we analyze the complexity of learning as a function of preference structure, casting learning as a stochastic multi-armed bandit problem. Algorithmically, we show that "optimism in the face of uncertainty," the principle underlying many bandit algorithms, applies to a primal-dual formulation of matching with transfers and leads to near-optimal regret bounds. Our work takes a first step toward elucidating when and how stable matchings arise in large, data-driven marketplaces.

----

## [254] Towards Lower Bounds on the Depth of ReLU Neural Networks

**Authors**: *Christoph Hertrich, Amitabh Basu, Marco Di Summa, Martin Skutella*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1b9812b99fe2672af746cefda86be5f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1b9812b99fe2672af746cefda86be5f9-Abstract.html)

**Abstract**:

We contribute to a better understanding of the class of functions that is represented by a neural network with ReLU activations and a given architecture. Using techniques from mixed-integer optimization, polyhedral theory, and tropical geometry, we provide a mathematical counterbalance to the universal approximation theorems which suggest that a single hidden layer is sufficient for learning tasks. In particular, we investigate whether the class of exactly representable functions strictly increases by adding more layers (with no restrictions on size). This problem has potential impact on algorithmic and statistical aspects because of the insight it provides into the class of functions represented by neural hypothesis classes. However, to the best of our knowledge, this question has not been investigated in the neural network literature. We also present upper bounds on the sizes of neural networks required to represent functions in these neural hypothesis classes.

----

## [255] The Limitations of Large Width in Neural Networks: A Deep Gaussian Process Perspective

**Authors**: *Geoff Pleiss, John P. Cunningham*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1b9f38268c50805669fd8caf8f3cc84a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1b9f38268c50805669fd8caf8f3cc84a-Abstract.html)

**Abstract**:

Large width limits have been a recent focus of deep learning research: modulo computational practicalities, do wider networks outperform narrower ones? Answering this question has been challenging, as conventional networks gain representational power with width, potentially masking any negative effects. Our analysis in this paper decouples capacity and width via the generalization of neural networks to Deep Gaussian Processes (Deep GP), a class of nonparametric hierarchical models that subsume neural nets. In doing so, we aim to understand how width affects (standard) neural networks once they have sufficient capacity for a given modeling task. Our theoretical and empirical results on Deep GP suggest that large width can be detrimental to hierarchical models. Surprisingly, we prove that even nonparametric Deep GP converge to Gaussian processes, effectively becoming shallower without any increase in representational power. The posterior, which corresponds to a mixture of data-adaptable basis functions, becomes less data-dependent with width. Our tail analysis demonstrates that width and depth have opposite effects: depth accentuates a model’s non-Gaussianity, while width makes models increasingly Gaussian. We find there is a “sweet spot” that maximizes test performance before the limiting GP behavior prevents adaptability, occurring at width = 1 or width = 2 for nonparametric Deep GP. These results make strong predictions about the same phenomenon in conventional neural networks trained with L2 regularization (analogous to a Gaussian prior on parameters): we show that such neural networks may need up to 500 − 1000 hidden units for sufficient capacity - depending on the dataset - but further width degrades performance.

----

## [256] Exact marginal prior distributions of finite Bayesian neural networks

**Authors**: *Jacob A. Zavatone-Veth, Cengiz Pehlevan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1baff70e2669e8376347efd3a874a341-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1baff70e2669e8376347efd3a874a341-Abstract.html)

**Abstract**:

Bayesian neural networks are theoretically well-understood only in the infinite-width limit, where Gaussian priors over network weights yield Gaussian priors over network outputs. Recent work has suggested that finite Bayesian networks may outperform their infinite counterparts, but their non-Gaussian output priors have been characterized only though perturbative approaches. Here, we derive exact solutions for the function space priors for individual input examples of a class of finite fully-connected feedforward Bayesian neural networks. For deep linear networks, the prior has a simple expression in terms of the Meijer $G$-function. The prior of a finite ReLU network is a mixture of the priors of linear networks of smaller widths, corresponding to different numbers of active units in each layer. Our results unify previous descriptions of finite network priors in terms of their tail decay and large-width behavior.

----

## [257] Spatiotemporal Joint Filter Decomposition in 3D Convolutional Neural Networks

**Authors**: *Zichen Miao, Ze Wang, Xiuyuan Cheng, Qiang Qiu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1bb91f73e9d31ea2830a5e73ce3ed328-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1bb91f73e9d31ea2830a5e73ce3ed328-Abstract.html)

**Abstract**:

In this paper, we introduce spatiotemporal joint filter decomposition to decouple spatial and temporal learning, while preserving spatiotemporal dependency in a video. A 3D convolutional filter is now jointly decomposed over a set of spatial and temporal filter atoms respectively. In this way, a 3D convolutional layer becomes three: a temporal atom layer, a spatial atom layer, and a joint coefficient layer, all three remaining convolutional. One obvious arithmetic manipulation allowed in our joint decomposition is to swap spatial or temporal atoms with a set of atoms that have the same number but different sizes, while keeping the remaining unchanged. For example, as shown later, we can now achieve tempo-invariance by simply dilating temporal atoms only. To illustrate this useful atom-swapping property, we further demonstrate how such a decomposition permits the direct learning of 3D CNNs with full-size videos through iterations of two consecutive sub-stages of learning: In the temporal stage, full-temporal downsampled-spatial data are used to learn temporal atoms and joint coefficients while fixing spatial atoms. In the spatial stage, full-spatial downsampled-temporal data are used for spatial atoms and joint coefficients while fixing temporal atoms. We show empirically on multiple action recognition datasets that, the decoupled spatiotemporal learning significantly reduces the model memory footprints, and allows deep 3D CNNs to model high-spatial long-temporal dependency with limited computational resources while delivering comparable performance.

----

## [258] Pooling by Sliced-Wasserstein Embedding

**Authors**: *Navid Naderializadeh, Joseph F. Comer, Reed W. Andrews, Heiko Hoffmann, Soheil Kolouri*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1bc2029a8851ad344a8d503930dfd7f7-Abstract.html)

**Abstract**:

Learning representations from sets has become increasingly important with many applications in point cloud processing, graph learning, image/video recognition, and object detection. We introduce a geometrically-interpretable and generic pooling mechanism for aggregating a set of features into a fixed-dimensional representation. In particular, we treat elements of a set as samples from a probability distribution and propose an end-to-end trainable Euclidean embedding for sliced-Wasserstein distance to learn from set-structured data effectively. We evaluate our proposed pooling method on a wide variety of set-structured data, including point-cloud, graph, and image classification tasks, and demonstrate that our proposed method provides superior performance over existing set representation learning approaches. Our code is available at https://github.com/navid-naderi/PSWE.

----

## [259] On the Theory of Reinforcement Learning with Once-per-Episode Feedback

**Authors**: *Niladri S. Chatterji, Aldo Pacchiano, Peter L. Bartlett, Michael I. Jordan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1bf2efbbe0c49b9f567c2e40f645279a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1bf2efbbe0c49b9f567c2e40f645279a-Abstract.html)

**Abstract**:

We study a theory of reinforcement learning (RL) in which the learner receives binary feedback only once at the end of an episode. While this is an extreme test case for theory, it is also arguably more representative of real-world applications than the traditional requirement in RL practice that the learner receive feedback at every time step. Indeed, in many real-world applications of reinforcement learning, such as self-driving cars and robotics, it is easier to evaluate whether a learner's complete trajectory was either good'' orbad,'' but harder to provide a reward signal at each step. To show that learning is possible in this more challenging setting, we study the case where trajectory labels are generated by an unknown parametric model, and provide a statistically and computationally efficient algorithm that achieves sublinear regret.

----

## [260] ResNEsts and DenseNEsts: Block-based DNN Models with Improved Representation Guarantees

**Authors**: *Kuan-Lin Chen, Ching Hua Lee, Harinath Garudadri, Bhaskar D. Rao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1bf50aaf147b3b0ddd26a820d2ed394d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1bf50aaf147b3b0ddd26a820d2ed394d-Abstract.html)

**Abstract**:

Models recently used in the literature proving residual networks (ResNets) are better than linear predictors are actually different from standard ResNets that have been widely used in computer vision. In addition to the assumptions such as scalar-valued output or single residual block, the models fundamentally considered in the literature have no nonlinearities at the final residual representation that feeds into the final affine layer. To codify such a difference in nonlinearities and reveal a linear estimation property, we define ResNEsts, i.e., Residual Nonlinear Estimators, by simply dropping nonlinearities at the last residual representation from standard ResNets. We show that wide ResNEsts with bottleneck blocks can always guarantee a very desirable training property that standard ResNets aim to achieve, i.e., adding more blocks does not decrease performance given the same set of basis elements. To prove that, we first recognize ResNEsts are basis function models that are limited by a coupling problem in basis learning and linear prediction. Then, to decouple prediction weights from basis learning, we construct a special architecture termed augmented ResNEst (A-ResNEst) that always guarantees no worse performance with the addition of a block. As a result, such an A-ResNEst establishes empirical risk lower bounds for a ResNEst using corresponding bases. Our results demonstrate ResNEsts indeed have a problem of diminishing feature reuse; however, it can be avoided by sufficiently expanding or widening the input space, leading to the above-mentioned desirable property. Inspired by the densely connected networks (DenseNets) that have been shown to outperform ResNets, we also propose a corresponding new model called Densely connected Nonlinear Estimator (DenseNEst). We show that any DenseNEst can be represented as a wide ResNEst with bottleneck blocks. Unlike ResNEsts, DenseNEsts exhibit the desirable property without any special architectural re-design.

----

## [261] Locally private online change point detection

**Authors**: *Thomas Berrett, Yi Yu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1c1d4df596d01da60385f0bb17a4a9e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1c1d4df596d01da60385f0bb17a4a9e0-Abstract.html)

**Abstract**:

We study online change point detection problems under the constraint of local differential privacy (LDP) where, in particular, the statistician does not have access to the raw data.  As a concrete problem, we study a multivariate nonparametric regression problem.  At each time point $t$, the raw data are assumed to be of the form $(X_t, Y_t)$, where $X_t$ is a $d$-dimensional feature vector and $Y_t$ is a response variable. Our primary aim is to detect changes in the regression function $m_t(x)=\mathbb{E}(Y_t |X_t=x)$ as soon as the change occurs.  We provide algorithms which respect the LDP constraint, which control the false alarm probability, and which detect changes with a minimal (minimax rate-optimal) delay.  To quantify the cost of privacy, we also present the optimal rate in the benchmark, non-private setting.  These non-private results are also new to the literature and thus are interesting \emph{per se}.  In addition, we study the univariate mean online change point detection problem, under privacy constraints.  This serves as the blueprint of studying more complicated private change point detection problems.

----

## [262] Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization

**Authors**: *Kartik Ahuja, Ethan Caballero, Dinghuai Zhang, Jean-Christophe Gagnon-Audet, Yoshua Bengio, Ioannis Mitliagkas, Irina Rish*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html)

**Abstract**:

The invariance principle from causality is at the heart of notable approaches such as invariant risk minimization (IRM) that seek to address out-of-distribution (OOD) generalization failures. Despite the promising theory, invariance principle-based approaches fail in common classification tasks, where invariant (causal) features capture all the information about the label.  Are these failures due to the methods failing to capture the invariance? Or is the invariance principle itself insufficient? To answer these questions, we revisit the fundamental assumptions in linear regression tasks, where invariance-based approaches were shown to provably generalize OOD. In contrast to the linear regression tasks, we show that for linear classification tasks we need much stronger restrictions on the distribution shifts, or otherwise OOD generalization is impossible.  Furthermore, even with appropriate restrictions on distribution shifts in place, we show that the invariance principle alone is insufficient. We prove that a form of the information bottleneck constraint along with invariance helps address the key failures when invariant features capture all the information about the label and also retains the existing success when they do not. We propose an approach that incorporates both of these principles and demonstrate its effectiveness in several experiments.

----

## [263] Repulsive Deep Ensembles are Bayesian

**Authors**: *Francesco D'Angelo, Vincent Fortuin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1c63926ebcabda26b5cdb31b5cc91efb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1c63926ebcabda26b5cdb31b5cc91efb-Abstract.html)

**Abstract**:

Deep ensembles have recently gained popularity in the deep learning community for their conceptual simplicity and efficiency. However, maintaining functional diversity between ensemble members that are independently trained with gradient descent is challenging. This can lead to pathologies when adding more ensemble members, such as a saturation of the ensemble performance, which converges to the performance of a single model. Moreover, this does not only affect the quality of its predictions, but even more so the uncertainty estimates of the ensemble, and thus its performance on out-of-distribution data. We hypothesize that this limitation can be overcome by discouraging different ensemble members from collapsing to the same function. To this end, we introduce a kernelized repulsive term in the update rule of the deep ensembles. We show that this simple modification not only enforces and maintains diversity among the members but, even more importantly, transforms the maximum a posteriori inference into proper Bayesian inference. Namely, we show that the training dynamics of our proposed repulsive ensembles follow a Wasserstein gradient flow of the KL divergence with the true posterior. We study repulsive terms in weight and function space and empirically compare their performance to standard ensembles and Bayesian baselines on synthetic and real-world prediction tasks.

----

## [264] BayesIMP: Uncertainty Quantification for Causal Data Fusion

**Authors**: *Siu Lun Chau, Jean-Francois Ton, Javier González, Yee Whye Teh, Dino Sejdinovic*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1ca5c750a30312d1919ae6a4d636dcc4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1ca5c750a30312d1919ae6a4d636dcc4-Abstract.html)

**Abstract**:

While causal models are becoming one of the mainstays of machine learning, the problem of uncertainty quantification in causal inference remains challenging. In this paper, we study the causal data fusion problem, where data arising from multiple causal graphs are combined to estimate the average treatment effect of a target variable. As data arises from multiple sources and can vary in quality and sample size, principled uncertainty quantification becomes essential. To that end, we introduce \emph{Bayesian Causal Mean Processes}, the framework which combines ideas from probabilistic integration and kernel mean embeddings to represent interventional distributions in the reproducing kernel Hilbert space, while taking into account the uncertainty within each causal graph. To demonstrate the informativeness of our uncertainty estimation, we apply our method to the Causal Bayesian Optimisation task and show improvements over state-of-the-art methods.

----

## [265] RMM: Reinforced Memory Management for Class-Incremental Learning

**Authors**: *Yaoyao Liu, Bernt Schiele, Qianru Sun*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1cbcaa5abbb6b70f378a3a03d0c26386-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1cbcaa5abbb6b70f378a3a03d0c26386-Abstract.html)

**Abstract**:

Class-Incremental Learning (CIL) [38] trains classifiers under a strict memory budget: in each incremental phase, learning is done for new data, most of which is abandoned to free space for the next phase. The preserved data are exemplars used for replaying. However, existing methods use a static and ad hoc strategy for memory allocation, which is often sub-optimal. In this work, we propose a dynamic memory management strategy that is optimized for the incremental phases and different object classes. We call our method reinforced memory management (RMM), leveraging reinforcement learning. RMM training is not naturally compatible with CIL as the past, and future data are strictly non-accessible during the incremental phases. We solve this by training the policy function of RMM on pseudo CIL tasks, e.g., the tasks built on the data of the zeroth phase, and then applying it to target tasks. RMM propagates two levels of actions: Level-1 determines how to split the memory between old and new classes, and Level-2 allocates memory for each specific class. In essence, it is an optimizable and general method for memory management that can be used in any replaying-based CIL method. For evaluation, we plug RMM into two top-performing baselines (LUCIR+AANets and POD+AANets [28]) and conduct experiments on three benchmarks (CIFAR-100, ImageNet-Subset, and ImageNet-Full). Our results show clear improvements, e.g., boosting POD+AANets by 3.6%, 4.4%, and 1.9% in the 25-Phase settings of the above benchmarks, respectively.  The code is available at https://class-il.mpi-inf.mpg.de/rmm/.

----

## [266] Learning Compact Representations of Neural Networks using DiscriminAtive Masking (DAM)

**Authors**: *Jie Bu, Arka Daw, M. Maruf, Anuj Karpatne*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1cc8a8ea51cd0adddf5dab504a285915-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1cc8a8ea51cd0adddf5dab504a285915-Abstract.html)

**Abstract**:

A central goal in deep learning is to learn  compact representations of features at every layer of a neural network, which is useful for both unsupervised representation learning  and structured network pruning. While there is a growing body of work in structured pruning, current state-of-the-art methods  suffer from two key limitations: (i) instability during training, and (ii) need for an additional step of fine-tuning, which is resource-intensive. At the core of these limitations is the lack of a systematic approach that jointly prunes and refines weights during training in a single stage, and does not require any fine-tuning upon convergence to achieve state-of-the-art performance. We present a novel single-stage structured pruning method termed DiscriminAtive Masking (DAM). The key intuition behind DAM is to discriminatively prefer some of the neurons to be refined during the training process, while gradually masking out other neurons. We show that our proposed DAM approach has remarkably good performance over a diverse range of applications in representation learning and structured pruning, including dimensionality reduction, recommendation system, graph representation learning, and structured pruning for image classification. We also theoretically show that the learning objective of DAM is directly related to minimizing the L_0 norm of the masking layer. All of our codes and datasets are available https://github.com/jayroxis/dam-pytorch.

----

## [267] Neural Auto-Curricula in Two-Player Zero-Sum Games

**Authors**: *Xidong Feng, Oliver Slumbers, Ziyu Wan, Bo Liu, Stephen McAleer, Ying Wen, Jun Wang, Yaodong Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1cd73be1e256a7405516501e94e892ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1cd73be1e256a7405516501e94e892ac-Abstract.html)

**Abstract**:

When solving two-player zero-sum games, multi-agent reinforcement learning (MARL) algorithms often create populations of agents where, at each iteration, a new agent is discovered as the best response to a mixture over the opponent population. Within such a process, the update rules of "who to compete with" (i.e., the opponent mixture) and "how to beat them" (i.e., finding best responses) are underpinned by manually developed game theoretical principles such as fictitious play and Double Oracle. In this paper, we introduce a novel framework—Neural Auto-Curricula (NAC)—that leverages meta-gradient descent to automate the discovery of the learning update rule without explicit human design. Specifically, we parameterise the opponent selection module by neural networks and the best-response module by optimisation subroutines, and update their parameters solely via interaction with the game engine, where both players aim to minimise their exploitability. Surprisingly, even without human design, the discovered MARL algorithms achieve competitive or even better performance with the state-of-the-art population-based game solvers (e.g., PSRO) on Games of Skill, differentiable Lotto, non-transitive Mixture Games, Iterated Matching Pennies, and Kuhn Poker. Additionally, we show that NAC is able to generalise from small games to large games, for example training on Kuhn Poker and outperforming PSRO on Leduc Poker. Our work inspires a promising future direction to discover general MARL algorithms solely from data.

----

## [268] ImageBART: Bidirectional Context with Multinomial Diffusion for Autoregressive Image Synthesis

**Authors**: *Patrick Esser, Robin Rombach, Andreas Blattmann, Björn Ommer*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1cdf14d1e3699d61d237cf76ce1c2dca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1cdf14d1e3699d61d237cf76ce1c2dca-Abstract.html)

**Abstract**:

Autoregressive models and their sequential factorization of the data likelihood have recently demonstrated great potential for image representation and synthesis. Nevertheless, they incorporate image context in a linear 1D order by attending only to previously synthesized image patches above or to the left. Not only is this unidirectional, sequential bias of attention unnatural for images as it disregards large parts of a scene until synthesis is almost complete. It also processes the entire image on a single scale, thus ignoring more global contextual information up to the gist of the entire scene. As a remedy we incorporate a coarse-to-fine hierarchy of context by combining the autoregressive formulation with a multinomial diffusion process: Whereas a multistage diffusion process successively compresses and removes information to coarsen an image, we train a Markov chain to invert this process. In each stage, the resulting autoregressive ImageBART model progressively incorporates context from previous stages in a coarse-to-fine manner. Experiments demonstrate the gain over current autoregressive models, continuous diffusion probabilistic models, and latent variable models. Moreover, the approach enables to control the synthesis process and to trade compression rate against reconstruction accuracy, while still guaranteeing visually plausible results.

----

## [269] From global to local MDI variable importances for random forests and when they are Shapley values

**Authors**: *Antonio Sutera, Gilles Louppe, Vân Anh Huynh-Thu, Louis Wehenkel, Pierre Geurts*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1cfa81af29c6f2d8cacb44921722e753-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1cfa81af29c6f2d8cacb44921722e753-Abstract.html)

**Abstract**:

Random forests have been widely used for their ability to provide so-called importance measures, which give insight at a global (per dataset) level on the relevance of input variables to predict a certain output. On the other hand, methods based on Shapley values have been introduced to refine the analysis of feature relevance in tree-based models to a local (per instance) level. In this context, we first show that the global Mean Decrease of Impurity (MDI) variable importance scores correspond to Shapley values under some conditions. Then, we derive a local MDI importance measure of variable relevance, which has a very natural connection with the global MDI measure and can be related to a new notion of local feature relevance. We further link local MDI importances with Shapley values and discuss them in the light of related measures from the literature. The measures are illustrated through experiments on several classification and regression problems.

----

## [270] Adversarial Robustness of Streaming Algorithms through Importance Sampling

**Authors**: *Vladimir Braverman, Avinatan Hassidim, Yossi Matias, Mariano Schain, Sandeep Silwal, Samson Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1d01bd2e16f57892f0954902899f0692-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1d01bd2e16f57892f0954902899f0692-Abstract.html)

**Abstract**:

Robustness against adversarial attacks has recently been at the forefront of algorithmic design for machine learning tasks. In the adversarial streaming model, an adversary gives an algorithm a sequence of adaptively chosen updates $u_1,\ldots,u_n$ as a data stream. The goal of the algorithm is to compute or approximate some predetermined function for every prefix of the adversarial stream, but the adversary may generate future updates based on previous outputs of the algorithm. In particular, the adversary may gradually learn the random bits internally used by an algorithm to manipulate dependencies in the input. This is especially problematic as many important problems in the streaming model require randomized algorithms, as they are known to not admit any deterministic algorithms that use sublinear space. In this paper, we introduce adversarially robust streaming algorithms for central machine learning and algorithmic tasks, such as regression and clustering, as well as their more general counterparts, subspace embedding, low-rank approximation, and coreset construction. For regression and other numerical linear algebra related tasks, we consider the row arrival streaming model. Our results are based on a simple, but powerful, observation that many importance sampling-based algorithms give rise to adversarial robustness which is in contrast to sketching based algorithms, which are very prevalent in the streaming literature but suffer from adversarial attacks. In addition, we show that the well-known merge and reduce paradigm in streaming is adversarially robust. Since the merge and reduce paradigm allows coreset constructions in the streaming setting, we thus obtain robust algorithms for $k$-means, $k$-median, $k$-center, Bregman clustering, projective clustering, principal component analysis (PCA) and non-negative matrix factorization. To the best of our knowledge, these are the first adversarially robust results for these problems yet require no new algorithmic implementations. Finally, we empirically confirm the robustness of our algorithms on various adversarial attacks and demonstrate that by contrast, some common existing algorithms are not robust.

----

## [271] Tractable Regularization of Probabilistic Circuits

**Authors**: *Anji Liu, Guy Van den Broeck*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1d0832c4969f6a4cc8e8a8fffe083efb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1d0832c4969f6a4cc8e8a8fffe083efb-Abstract.html)

**Abstract**:

Probabilistic Circuits (PCs) are a promising avenue for probabilistic modeling. They combine advantages of probabilistic graphical models (PGMs) with those of neural networks (NNs). Crucially, however, they are tractable probabilistic models, supporting efficient and exact computation of many probabilistic inference queries, such as marginals and MAP. Further, since PCs are structured computation graphs, they can take advantage of deep-learning-style parameter updates, which greatly improves their scalability. However, this innovation also makes PCs prone to overfitting, which has been observed in many standard benchmarks. Despite the existence of abundant regularization techniques for both PGMs and NNs, they are not effective enough when applied to PCs. Instead, we re-think regularization for PCs and propose two intuitive techniques, data softening and entropy regularization, that both take advantage of PCs' tractability and still have an efficient implementation as a computation graph. Specifically, data softening provides a principled way to add uncertainty in datasets in closed form, which implicitly regularizes PC parameters. To learn parameters from a softened dataset, PCs only need linear time by virtue of their tractability. In entropy regularization, the exact entropy of the distribution encoded by a PC can be regularized directly, which is again infeasible for most other density estimation models. We show that both methods consistently improve the generalization performance of a wide variety of PCs. Moreover, when paired with a simple PC structure, we achieved state-of-the-art results on 10 out of 20 standard discrete density estimation benchmarks. Open-source code and experiments are available at https://github.com/UCLA-StarAI/Tractable-PC-Regularization.

----

## [272] On Interaction Between Augmentations and Corruptions in Natural Corruption Robustness

**Authors**: *Eric Mintun, Alexander Kirillov, Saining Xie*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1d49780520898fe37f0cd6b41c5311bf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1d49780520898fe37f0cd6b41c5311bf-Abstract.html)

**Abstract**:

Invariance to a broad array of image corruptions, such as warping, noise, or color shifts, is an important aspect of building robust models in computer vision. Recently, several new data augmentations have been proposed that significantly improve performance on ImageNet-C, a benchmark of such corruptions. However, there is still a lack of basic understanding on the relationship between data augmentations and test-time corruptions. To this end, we develop a feature space for image transforms, and then use a new measure in this space between augmentations and corruptions called the Minimal Sample Distance to demonstrate there is a strong correlation between similarity and performance. We then investigate recent data augmentations and observe a significant degradation in corruption robustness when the test-time corruptions are sampled to be perceptually dissimilar from ImageNet-C in this feature space. Our results suggest that test error can be improved by training on perceptually similar augmentations, and data augmentations may not generalize well beyond the existing benchmark. We hope our results and tools will allow for more robust progress towards improving robustness to image corruptions. We provide code at https://github.com/facebookresearch/augmentation-corruption.

----

## [273] Dynamic Distillation Network for Cross-Domain Few-Shot Recognition with Unlabeled Data

**Authors**: *Ashraful Islam, Chun-Fu (Richard) Chen, Rameswar Panda, Leonid Karlinsky, Rogério Feris, Richard J. Radke*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1d6408264d31d453d556c60fe7d0459e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1d6408264d31d453d556c60fe7d0459e-Abstract.html)

**Abstract**:

Most existing works in few-shot learning rely on meta-learning the network on a large base dataset which is typically from the same domain as the target dataset. We tackle the problem of cross-domain few-shot learning where there is a large shift between the base and target domain. The problem of cross-domain few-shot recognition with unlabeled target data is largely unaddressed in the literature. STARTUP was the first method that tackles this problem using self-training. However, it uses a fixed teacher pretrained on a labeled base dataset to create soft labels for the unlabeled target samples. As the base dataset and unlabeled dataset are from different domains, projecting the target images in the class-domain of the base dataset with a fixed pretrained model might be sub-optimal. We propose a simple dynamic distillation-based approach to facilitate unlabeled images from the novel/base dataset. We impose consistency regularization by calculating predictions from the weakly-augmented versions of the unlabeled images from a teacher network and matching it with the strongly augmented versions of the same images from a student network. The parameters of the teacher network are updated as exponential moving average of the parameters of the student network. We show that the proposed network learns representation that can be easily adapted to the target domain even though it has not been trained with target-specific classes during the pretraining phase. Our model outperforms the current state-of-the art method by 4.4% for 1-shot and 3.6% for 5-shot classification in the BSCD-FSL benchmark, and also shows competitive performance on traditional in-domain few-shot learning task.

----

## [274] Hypergraph Propagation and Community Selection for Objects Retrieval

**Authors**: *Guoyuan An, Yuchi Huo, Sung Eui Yoon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1da546f25222c1ee710cf7e2f7a3ff0c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1da546f25222c1ee710cf7e2f7a3ff0c-Abstract.html)

**Abstract**:

Spatial verification is a crucial technique for particular object retrieval. It utilizes spatial information for the accurate detection of true positive images. However, existing query expansion and diffusion methods cannot efficiently propagate the spatial information in an ordinary graph with scalar edge weights, resulting in low recall or precision. To tackle these problems, we propose a novel hypergraph-based framework that efficiently propagates spatial information in query time and retrieves an object in the database accurately. Additionally, we propose using the image graph's structure information through community selection technique, to measure the accuracy of the initial search result and to provide correct starting points for hypergraph propagation without heavy spatial verification computations. Experiment results on ROxford and RParis show that our method  significantly outperforms the existing query expansion and diffusion methods.

----

## [275] Deep learning is adaptive to intrinsic dimensionality of model smoothness in anisotropic Besov space

**Authors**: *Taiji Suzuki, Atsushi Nitanda*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1dacb10f0623c67cb7dbb37587d8b38a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1dacb10f0623c67cb7dbb37587d8b38a-Abstract.html)

**Abstract**:

Deep learning has exhibited superior performance for various tasks, especially for high-dimensional datasets, such as images. To understand this property, we investigate the approximation and estimation ability of deep learning on {\it anisotropic Besov spaces}.The anisotropic Besov space is characterized by direction-dependent smoothness and includes several function classes that have been investigated thus far.We demonstrate that the approximation error and estimation error of deep learning only depend on the average value of the smoothness parameters in all directions. Consequently, the curse of dimensionality can be avoided if the smoothness of the target function is highly anisotropic.Unlike existing studies, our analysis does not require a low-dimensional structure of the input data.We also investigate the minimax optimality of deep learning and compare its performance with that of the kernel method (more generally, linear estimators).The results show that deep learning has better dependence on the input dimensionality if the target function possesses anisotropic smoothness, and it achieves an adaptive rate for functions with spatially inhomogeneous smoothness.

----

## [276] QuPeD: Quantized Personalization via Distillation with Applications to Federated Learning

**Authors**: *Kaan Ozkara, Navjot Singh, Deepesh Data, Suhas N. Diggavi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1dba3025b159cd9354da65e2d0436a31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1dba3025b159cd9354da65e2d0436a31-Abstract.html)

**Abstract**:

Traditionally, federated learning (FL) aims to train a single global model while collaboratively using multiple clients and a server. Two natural challenges that FL algorithms face are heterogeneity in data across clients and collaboration of clients with diverse resources. In this work, we introduce a quantized and personalized FL algorithm QuPeD that facilitates collective (personalized model compression) training via knowledge distillation (KD)  among clients who have access to heterogeneous data and resources. For personalization, we allow clients to learn compressed personalized models with different quantization parameters and model dimensions/structures. Towards this, first we propose an algorithm for learning quantized models through a relaxed optimization problem, where quantization values are also optimized over. When each client participating in the (federated) learning process has different requirements of the compressed model (both in model dimension and precision), we formulate a compressed personalization framework by introducing knowledge distillation loss for local client objectives collaborating through a global model. We develop an alternating proximal gradient update for solving this compressed personalization problem, and analyze its convergence properties. Numerically, we validate that QuPeD outperforms competing personalized FL methods, FedAvg, and local training of clients in various heterogeneous settings.

----

## [277] Model Adaptation: Historical Contrastive Learning for Unsupervised Domain Adaptation without Source Data

**Authors**: *Jiaxing Huang, Dayan Guan, Aoran Xiao, Shijian Lu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1dba5eed8838571e1c80af145184e515-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1dba5eed8838571e1c80af145184e515-Abstract.html)

**Abstract**:

Unsupervised domain adaptation aims to align a labeled source domain and an unlabeled target domain, but it requires to access the source data which often raises concerns in data privacy, data portability and data transmission efficiency. We study unsupervised model adaptation (UMA), or called Unsupervised Domain Adaptation without Source Data, an alternative setting that aims to adapt source-trained models towards target distributions without accessing source data. To this end, we design an innovative historical contrastive learning (HCL) technique that exploits historical source hypothesis to make up for the absence of source data in UMA. HCL addresses the UMA challenge from two perspectives. First, it introduces historical contrastive instance discrimination (HCID) that learns from target samples by contrasting their embeddings which are generated by the currently adapted model and the historical models. With the historical models, HCID encourages UMA to learn instance-discriminative target representations while preserving the source hypothesis. Second, it introduces historical contrastive category discrimination (HCCD) that pseudo-labels target samples to learn category-discriminative target representations. Specifically, HCCD re-weights pseudo labels according to their prediction consistency across the current and historical models. Extensive experiments show that HCL outperforms and state-of-the-art methods consistently across a variety of visual tasks and setups.

----

## [278] The Out-of-Distribution Problem in Explainability and Search Methods for Feature Importance Explanations

**Authors**: *Peter Hase, Harry Xie, Mohit Bansal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1def1713ebf17722cbe300cfc1c88558-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1def1713ebf17722cbe300cfc1c88558-Abstract.html)

**Abstract**:

Feature importance (FI) estimates are a popular form of explanation, and they are commonly created and evaluated by computing the change in model confidence caused by removing certain input features at test time. For example, in the standard Sufficiency metric, only the top-k most important tokens are kept. In this paper, we study several under-explored dimensions of FI explanations, providing conceptual and empirical improvements for this form of explanation. First, we advance a new argument for why it can be problematic to remove features from an input when creating or evaluating explanations: the fact that these counterfactual inputs are out-of-distribution (OOD) to models implies that the resulting explanations are socially misaligned. The crux of the problem is that the model prior and random weight initialization influence the explanations (and explanation metrics) in unintended ways. To resolve this issue, we propose a simple alteration to the model training process, which results in more socially aligned explanations and metrics. Second, we compare among five approaches for removing features from model inputs. We find that some methods produce more OOD counterfactuals than others, and we make recommendations for selecting a feature-replacement function. Finally, we introduce four search-based methods for identifying FI explanations and compare them to strong baselines, including LIME, Anchors, and Integrated Gradients. Through experiments with six diverse text classification datasets, we find that the only method that consistently outperforms random search is a Parallel Local Search (PLS) that we introduce. Improvements over the second best method are as large as 5.4 points for Sufficiency and 17 points for Comprehensiveness.

----

## [279] Control Variates for Slate Off-Policy Evaluation

**Authors**: *Nikos Vlassis, Ashok Chandrashekar, Fernando Amat Gil, Nathan Kallus*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e0b802d5c0e1e8434a771ba7ff2c301-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e0b802d5c0e1e8434a771ba7ff2c301-Abstract.html)

**Abstract**:

We study the problem of off-policy evaluation from batched contextual bandit data with multidimensional actions, often termed slates. The problem is common to recommender systems and user-interface optimization, and it is particularly challenging because of the combinatorially-sized action space. Swaminathan et al. (2017) have proposed the pseudoinverse (PI) estimator under the assumption that the conditional mean rewards are additive in actions. Using control variates, we consider a large class of unbiased estimators that includes as specific cases the PI estimator and (asymptotically) its self-normalized variant. By optimizing over this class, we obtain new estimators with risk improvement guarantees over both the PI and the self-normalized PI estimators. Experiments with real-world recommender data as well as synthetic data validate these improvements in practice.

----

## [280] Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation

**Authors**: *Nicklas Hansen, Hao Su, Xiaolong Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e0f65eb20acbfb27ee05ddc000b50ec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e0f65eb20acbfb27ee05ddc000b50ec-Abstract.html)

**Abstract**:

While agents trained by Reinforcement Learning (RL) can solve increasingly challenging tasks directly from visual observations, generalizing learned skills to novel environments remains very challenging. Extensive use of data augmentation is a promising technique for improving generalization in RL, but it is often found to decrease sample efficiency and can even lead to divergence. In this paper, we investigate causes of instability when using data augmentation in common off-policy RL algorithms. We identify two problems, both rooted in high-variance Q-targets. Based on our findings, we propose a simple yet effective technique for stabilizing this class of algorithms under augmentation. We perform extensive empirical evaluation of image-based RL using both ConvNets and Vision Transformers (ViT) on a family of benchmarks based on DeepMind Control Suite, as well as in robotic manipulation tasks. Our method greatly improves stability and sample efficiency of ConvNets under augmentation, and achieves generalization results competitive with state-of-the-art methods for image-based RL in environments with unseen visuals. We further show that our method scales to RL with ViT-based architectures, and that data augmentation may be especially important in this setting.

----

## [281] On Effective Scheduling of Model-based Reinforcement Learning

**Authors**: *Hang Lai, Jian Shen, Weinan Zhang, Yimin Huang, Xing Zhang, Ruiming Tang, Yong Yu, Zhenguo Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html)

**Abstract**:

Model-based reinforcement learning has attracted wide attention due to its superior sample efficiency. Despite its impressive success so far, it is still unclear how to appropriately schedule the important hyperparameters to achieve adequate performance, such as the real data ratio for policy optimization in Dyna-style model-based algorithms. In this paper, we first theoretically analyze the role of real data in policy training, which suggests that gradually increasing the ratio of real data yields better performance. Inspired by the analysis, we propose a framework named AutoMBPO to automatically schedule the real data ratio as well as other hyperparameters in training model-based policy optimization (MBPO) algorithm, a representative running case of model-based methods. On several continuous control tasks, the MBPO instance trained with hyperparameters scheduled by AutoMBPO can significantly surpass the original one, and the real data ratio schedule found by AutoMBPO shows consistency with our theoretical analysis.

----

## [282] Removing Inter-Experimental Variability from Functional Data in Systems Neuroscience

**Authors**: *Dominic Gonschorek, Larissa Höfling, Klaudia P. Szatko, Katrin Franke, Timm Schubert, Benjamin A. Dunn, Philipp Berens, David A. Klindt, Thomas Euler*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e5eeb40a3fce716b244599862fd2200-Abstract.html)

**Abstract**:

Integrating data from multiple experiments is common practice in systems neuroscience but it requires inter-experimental variability to be negligible compared to the biological signal of interest. This requirement is rarely fulfilled; systematic changes between experiments can drastically affect the outcome of complex analysis pipelines. Modern machine learning approaches designed to adapt models across multiple data domains offer flexible ways of removing inter-experimental variability where classical statistical methods often fail. While applications of these methods have been mostly limited to single-cell genomics, in this work, we develop a theoretical framework for domain adaptation in systems neuroscience. We implement this in an adversarial optimization scheme that removes inter-experimental variability while preserving the biological signal. We compare our method to previous approaches on a large-scale dataset of two-photon imaging recordings of retinal bipolar cell responses to visual stimuli. This dataset provides a unique benchmark as it contains biological signal from well-defined cell types that is obscured by large inter-experimental variability. In a supervised setting, we compare the generalization performance of cell type classifiers across experiments, which we validate with anatomical cell type distributions from electron microscopy data. In an unsupervised setting, we remove inter-experimental variability from the data which can then be fed into arbitrary downstream analyses. In both settings, we find that our method achieves the best trade-off between removing inter-experimental variability and preserving biological signal. Thus, we offer a flexible approach to remove inter-experimental variability and integrate datasets across experiments in systems neuroscience. Code available at https://github.com/eulerlab/rave.

----

## [283] Learning Knowledge Graph-based World Models of Textual Environments

**Authors**: *Prithviraj Ammanabrolu, Mark O. Riedl*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e747ddbea997a1b933aaf58a7953c3c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e747ddbea997a1b933aaf58a7953c3c-Abstract.html)

**Abstract**:

World models improve a learning agent's ability to efficiently operate in interactive and situated environments. This work focuses on the task of building world models of text-based game environments. Text-based games, or interactive narratives, are reinforcement learning environments in which agents perceive and interact with the world using textual natural language. These environments contain long, multi-step puzzles or quests woven through a world that is filled with hundreds of characters, locations, and objects. Our world model learns to simultaneously: (1) predict changes in the world caused by an agent's actions when representing the world as a knowledge graph; and (2) generate the set of contextually relevant natural language actions required to operate in the world. We frame this task as a Set of Sequences generation problem by exploiting the inherent structure of knowledge graphs and actions and introduce both a transformer-based multi-task architecture and a loss function to train it. A zero-shot ablation study on never-before-seen textual worlds shows that our methodology significantly outperforms existing textual world modeling techniques as well as the importance of each of our contributions.

----

## [284] Damped Anderson Mixing for Deep Reinforcement Learning: Acceleration, Convergence, and Stabilization

**Authors**: *Ke Sun, Yafei Wang, Yi Liu, Yingnan Zhao, Bo Pan, Shangling Jui, Bei Jiang, Linglong Kong*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html)

**Abstract**:

Anderson mixing has been heuristically applied to reinforcement learning (RL) algorithms for accelerating convergence and improving the sampling efficiency of deep RL. Despite its heuristic improvement of convergence, a rigorous mathematical justification for the benefits of Anderson mixing in RL has not yet been put forward. In this paper, we provide deeper insights into a class of acceleration schemes built on Anderson mixing that improve the convergence of deep RL algorithms. Our main results establish a connection between Anderson mixing and quasi-Newton methods and prove that Anderson mixing increases the convergence radius of policy iteration schemes by an extra contraction factor. The key focus of the analysis roots in the fixed-point iteration nature of RL. We further propose a stabilization strategy by introducing a stable regularization term in Anderson mixing and a differentiable, non-expansive MellowMax operator that can allow both faster convergence and more stable behavior. Extensive experiments demonstrate that our proposed method enhances the convergence, stability, and performance of RL algorithms.

----

## [285] Approximate Decomposable Submodular Function Minimization for Cardinality-Based Components

**Authors**: *Nate Veldt, Austin R. Benson, Jon M. Kleinberg*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)

**Abstract**:

Minimizing a sum of simple submodular functions of limited support is a special case of general submodular function minimization that has seen numerous applications in machine learning. We develop faster techniques for instances where components in the sum are cardinality-based, meaning they depend only on the size of the input set. This variant is one of the most widely applied in practice, encompassing, e.g., common energy functions arising in image segmentation and recent generalized hypergraph cut functions. We develop the first approximation algorithms for this problem, where the approximations can be quickly computed via reduction to a sparse graph cut problem, with graph sparsity controlled by the desired approximation factor. Our method relies on a new connection between sparse graph reduction techniques and piecewise linear approximations to concave functions. Our sparse reduction technique leads to significant improvements in theoretical runtimes, as well as substantial practical gains in problems ranging from benchmark image segmentation tasks to hypergraph clustering problems.

----

## [286] Episodic Multi-agent Reinforcement Learning with Curiosity-driven Exploration

**Authors**: *Lulu Zheng, Jiarui Chen, Jianhao Wang, Jiamin He, Yujing Hu, Yingfeng Chen, Changjie Fan, Yang Gao, Chongjie Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e8ca836c962598551882e689265c1c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e8ca836c962598551882e689265c1c5-Abstract.html)

**Abstract**:

Efficient exploration in deep cooperative multi-agent reinforcement learning (MARL) still remains challenging in complex coordination problems. In this paper, we introduce a novel Episodic Multi-agent reinforcement learning with Curiosity-driven exploration, called EMC. We leverage an insight of popular factorized MARL algorithms that the ``induced" individual Q-values, i.e., the individual utility functions  used for local execution,  are the embeddings of local action-observation histories, and can capture the interaction between agents due to reward backpropagation during centralized training. Therefore, we use prediction errors of individual Q-values as intrinsic rewards for coordinated exploration and utilize episodic memory to exploit explored informative experience to boost policy training. As the dynamics of an agent's individual Q-value function captures the novelty of states and the influence from other agents, our intrinsic reward can induce coordinated exploration to new or promising states. We illustrate the advantages of our method by didactic examples, and demonstrate its significant outperformance over state-of-the-art MARL baselines on challenging tasks in the StarCraft II micromanagement benchmark.

----

## [287] Two Sides of Meta-Learning Evaluation: In vs Out of Distribution

**Authors**: *Amrith Setlur, Oscar Li, Virginia Smith*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1e932f24dc0aa4e7a6ac2beec387416d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1e932f24dc0aa4e7a6ac2beec387416d-Abstract.html)

**Abstract**:

We categorize meta-learning evaluation into two settings: $\textit{in-distribution}$ [ID], in which the train and test tasks are sampled $\textit{iid}$ from the same underlying task distribution, and $\textit{out-of-distribution}$ [OOD], in which they are not. While most meta-learning theory and some FSL applications follow the ID setting, we identify that most existing few-shot classification benchmarks instead reflect OOD evaluation, as they use disjoint sets of train (base) and test (novel) classes for task generation. This discrepancy is problematic because -- as we show on numerous benchmarks -- meta-learning methods that perform better on existing OOD datasets may perform significantly worse in the ID setting. In addition, in the OOD setting, even though current FSL benchmarks seem befitting, our study highlights concerns in 1) reliably performing model selection for a given meta-learning method, and 2) consistently comparing the performance of different methods. To address these concerns, we provide suggestions on how to construct FSL benchmarks to allow for ID evaluation as well as more reliable OOD evaluation. Our work aims to inform the meta-learning community about the importance and distinction of ID vs. OOD evaluation, as well as the subtleties of OOD evaluation with current benchmarks.

----

## [288] Debiased Visual Question Answering from Feature and Sample Perspectives

**Authors**: *Zhiquan Wen, Guanghui Xu, Mingkui Tan, Qingyao Wu, Qi Wu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1f4477bad7af3616c1f933a02bfabe4e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1f4477bad7af3616c1f933a02bfabe4e-Abstract.html)

**Abstract**:

Visual question answering (VQA) is designed to examine the visual-textual reasoning ability of an intelligent agent. However, recent observations show that many VQA models may only capture the biases between questions and answers in a dataset rather than showing real reasoning abilities. For example, given a question, some VQA models tend to output the answer that occurs frequently in the dataset and ignore the images. To reduce this tendency, existing methods focus on weakening the language bias. Meanwhile, only a few works also consider vision bias implicitly. However, these methods introduce additional annotations or show unsatisfactory performance. Moreover, not all biases are harmful to the models. Some “biases” learnt from datasets represent natural rules of the world and can help limit the range of answers. Thus, how to filter and remove the true negative biases in language and vision modalities remain a major challenge. In this paper, we propose a method named D-VQA to alleviate the above challenges from the feature and sample perspectives. Specifically, from the feature perspective, we build a question-to-answer and vision-to-answer branch to capture the language and vision biases, respectively. Next, we apply two unimodal bias detection modules to explicitly recognise and remove the negative biases. From the sample perspective, we construct two types of negative samples to assist the training of the models, without introducing additional annotations. Extensive experiments on the VQA-CP v2 and VQA v2 datasets demonstrate the effectiveness of our D-VQA method.

----

## [289] Towards a Unified Game-Theoretic View of Adversarial Perturbations and Robustness

**Authors**: *Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1f4fe6a4411edc2ff625888b4093e917-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1f4fe6a4411edc2ff625888b4093e917-Abstract.html)

**Abstract**:

This paper provides a unified view to explain different adversarial attacks and defense methods, i.e. the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing robustness-boosting methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features. Our code is available online at https://github.com/Jie-Ren/A-Unified-Game-Theoretic-Interpretation-of-Adversarial-Robustness.

----

## [290] On the Out-of-distribution Generalization of Probabilistic Image Modelling

**Authors**: *Mingtian Zhang, Andi Zhang, Steven McDonagh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1f88c7c5d7d94ae08bd752aa3d82108b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1f88c7c5d7d94ae08bd752aa3d82108b-Abstract.html)

**Abstract**:

Out-of-distribution (OOD) detection and lossless compression constitute two problems that can be solved by the training of probabilistic models on a first dataset with subsequent likelihood evaluation on a second dataset, where data distributions differ. By defining the generalization of probabilistic models in terms of likelihood we show that, in the case of image models, the OOD generalization ability is dominated by local features. This motivates our proposal of a Local Autoregressive model that exclusively models local image features towards improving OOD performance. We apply the proposed model to OOD detection tasks and achieve state-of-the-art unsupervised OOD detection performance without the introduction of additional data. Additionally, we employ our model to build a new lossless image compressor: NeLLoC (Neural Local Lossless Compressor) and report state-of-the-art compression rates and model size.

----

## [291] Exploiting Local Convergence of Quasi-Newton Methods Globally: Adaptive Sample Size Approach

**Authors**: *Qiujiang Jin, Aryan Mokhtari*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1f9b616faddedc02339603f3b37d196c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1f9b616faddedc02339603f3b37d196c-Abstract.html)

**Abstract**:

In this paper, we study the application of quasi-Newton methods for solving empirical risk minimization (ERM) problems defined over a large dataset. Traditional deterministic and stochastic quasi-Newton methods can be executed to solve such problems; however, it is known that their global convergence rate may not be better than first-order methods, and their local superlinear convergence only appears towards the end of the learning process. In this paper, we use an adaptive sample size scheme that exploits the superlinear convergence of quasi-Newton methods globally and throughout the entire learning process. The main idea of the proposed adaptive sample size algorithms is to start with a small subset of data points and solve their corresponding ERM problem within its statistical accuracy, and then enlarge the sample size geometrically and use the optimal solution of the problem corresponding to the smaller set as an initial point for solving the subsequent ERM problem with more samples. We show that if the initial sample size is sufficiently large and we use quasi-Newton methods to solve each subproblem, the subproblems can be solved superlinearly fast (after at most three iterations), as we guarantee that the iterates always stay within a neighborhood that quasi-Newton methods converge superlinearly. Numerical experiments on various datasets confirm our theoretical results and demonstrate the computational advantages of our method.

----

## [292] PDE-GCN: Novel Architectures for Graph Neural Networks Motivated by Partial Differential Equations

**Authors**: *Moshe Eliasof, Eldad Haber, Eran Treister*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1f9f9d8ff75205aa73ec83e543d8b571-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1f9f9d8ff75205aa73ec83e543d8b571-Abstract.html)

**Abstract**:

Graph neural networks are increasingly becoming the go-to approach in various fields such as computer vision, computational biology and chemistry, where data are naturally explained by graphs. However, unlike traditional convolutional neural networks, deep graph networks do not necessarily yield better performance than shallow graph networks. This behavior usually stems from the over-smoothing phenomenon. In this work, we propose a family of architecturesto control this behavior by design. Our networks are motivated by numerical methods for solving Partial Differential Equations (PDEs) on manifolds, and as such, their behavior can be explained by similar analysis. Moreover, as we demonstrate using an extensive set of experiments, our PDE-motivated networks can generalize and be effective for various types of problems from different fields. Our architectures obtain better or on par with the current state-of-the-art results for problems that are typically approached using different architectures.

----

## [293] Information Directed Reward Learning for Reinforcement Learning

**Authors**: *David Lindner, Matteo Turchetta, Sebastian Tschiatschek, Kamil Ciosek, Andreas Krause*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1fa6269f58898f0e809575c9a48747ef-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1fa6269f58898f0e809575c9a48747ef-Abstract.html)

**Abstract**:

For many reinforcement learning (RL) applications, specifying a reward is difficult. In this paper, we consider an RL setting where the agent can obtain information about the reward only by querying an expert that can, for example, evaluate individual states or provide binary preferences over trajectories. From such expensive feedback, we aim to learn a model of the reward function that allows standard RL algorithms to achieve high expected return with as few expert queries as possible. For this purpose, we propose Information Directed Reward Learning (IDRL), which uses a Bayesian model of the reward function and selects queries that maximize the information gain about the difference in return between potentially optimal policies. In contrast to prior active reward learning methods designed for specific types of queries, IDRL naturally accommodates different query types. Moreover, by shifting the focus from reducing the reward approximation error to improving the policy induced by the reward model, it achieves similar or better performance with significantly fewer queries. We support our findings with extensive evaluations  in multiple environments and with different types of queries.

----

## [294] SSMF: Shifting Seasonal Matrix Factorization

**Authors**: *Koki Kawabata, Siddharth Bhatia, Rui Liu, Mohit Wadhwa, Bryan Hooi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1fb2a1c37b18aa4611c3949d6148d0f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1fb2a1c37b18aa4611c3949d6148d0f8-Abstract.html)

**Abstract**:

Given taxi-ride counts information between departure and destination locations, how can we forecast their future demands? In general, given a data stream of events with seasonal patterns that innovate over time, how can we effectively and efficiently forecast future events? In this paper, we propose Shifting Seasonal Matrix Factorization approach, namely SSMF, that can adaptively learn multiple seasonal patterns (called regimes), as well as switching between them. Our proposed method has the following properties: (a) it accurately forecasts future events by detecting regime shifts in seasonal patterns as the data stream evolves; (b) it works in an online setting, i.e., processes each observation in constant time and memory; (c) it effectively realizes regime shifts without human intervention by using a lossless data compression scheme. We demonstrate that our algorithm outperforms state-of-the-art baseline methods by accurately forecasting upcoming events on three real-world data streams.

----

## [295] Associative Memories via Predictive Coding

**Authors**: *Tommaso Salvatori, Yuhang Song, Yujian Hong, Lei Sha, Simon Frieder, Zhenghua Xu, Rafal Bogacz, Thomas Lukasiewicz*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1fb36c4ccf88f7e67ead155496f02338-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1fb36c4ccf88f7e67ead155496f02338-Abstract.html)

**Abstract**:

Associative memories in the brain receive and store patterns of activity registered by the sensory neurons, and are able to retrieve them when necessary. Due to their importance in human intelligence, computational models of associative memories have been developed for several decades now. In this paper, we present a novel neural model for realizing associative memories, which is based on a hierarchical generative network that receives external stimuli via sensory neurons. It is trained using predictive coding, an error-based learning algorithm inspired by information processing in the cortex. To test the model's capabilities, we perform multiple retrieval experiments from both corrupted and incomplete data points. In an extensive comparison, we show that this new model outperforms in retrieval accuracy and robustness popular associative memory models, such as  autoencoders trained via backpropagation, and modern Hopfield networks. In particular, in completing partial data points, our model achieves remarkable results on natural image datasets, such as ImageNet, with a surprisingly high accuracy, even when only a tiny fraction of pixels of the original images is presented. Our model provides a plausible framework to study learning and retrieval of memories in the brain, as it closely mimics the behavior of the hippocampus as a memory index and generative model.

----

## [296] Robust and differentially private mean estimation

**Authors**: *Xiyang Liu, Weihao Kong, Sham M. Kakade, Sewoong Oh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1fc5309ccc651bf6b5d22470f67561ea-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1fc5309ccc651bf6b5d22470f67561ea-Abstract.html)

**Abstract**:

In statistical learning and analysis from shared data, which is increasingly widely adopted in platforms such as federated learning and meta-learning, there are two major concerns: privacy and robustness. Each participating individual should be able to contribute without the fear of leaking one's sensitive information.  At the same time, the system should be robust in the presence of malicious participants inserting corrupted data. Recent algorithmic advances in learning from shared data focus on either one of these threats, leaving the system vulnerable to the other. We bridge this gap for the canonical problem of estimating the mean from i.i.d.~samples. We introduce PRIME, which is the first efficient algorithm that achieves both privacy and robustness for a wide range of distributions. We further complement this result with a novel exponential time algorithm that improves the sample complexity of PRIME, achieving a near-optimal guarantee and matching that of a known lower bound for (non-robust) private mean estimation. This proves that there is no extra statistical cost to simultaneously guaranteeing privacy and robustness.

----

## [297] Adaptable Agent Populations via a Generative Model of Policies

**Authors**: *Kenneth Derek, Phillip Isola*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/1fc8c3d03b0021478a8c9ebdcd457c67-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1fc8c3d03b0021478a8c9ebdcd457c67-Abstract.html)

**Abstract**:

In the natural world, life has found innumerable ways to survive and often thrive. Between and even within species, each individual is in some manner unique, and this diversity lends adaptability and robustness to life. In this work, we aim to learn a space of diverse and high-reward policies in a given environment. To this end, we introduce a generative model of policies for reinforcement learning, which maps a low-dimensional latent space to an agent policy space. Our method enables learning an entire population of agent policies, without requiring the use of separate policy parameters. Just as real world populations can adapt and evolve via natural selection, our method is able to adapt to changes in our environment solely by selecting for policies in latent space. We test our generative modelâ€™s capabilities in a variety of environments, including an open-ended grid-world and a two-player soccer environment. Code, visualizations, and additional experiments can be found at https://kennyderek.github.io/adap/.

----

## [298] A No-go Theorem for Robust Acceleration in the Hyperbolic Plane

**Authors**: *Linus Hamilton, Ankur Moitra*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/201d546992726352471cfea6b0df0a48-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/201d546992726352471cfea6b0df0a48-Abstract.html)

**Abstract**:

In recent years there has been significant effort to adapt the key tools and ideas in convex optimization to the Riemannian setting. One key challenge has remained: Is there a Nesterov-like accelerated gradient method for geodesically convex functions on a Riemannian manifold? Recent work has given partial answers and the hope was that this ought to be possible. Here we prove that in a noisy setting, there is no analogue of accelerated gradient descent for geodesically convex functions on the hyperbolic plane. Our results apply even when the noise is exponentially small. The key intuition behind our proof is short and simple: In negatively curved spaces, the volume of a ball grows so fast that information about the past gradients is not useful in the future.

----

## [299] Privately Learning Mixtures of Axis-Aligned Gaussians

**Authors**: *Ishaq Aden-Ali, Hassan Ashtiani, Christopher Liaw*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/201d7288b4c18a679e48b31c72c30ded-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/201d7288b4c18a679e48b31c72c30ded-Abstract.html)

**Abstract**:

We consider the problem of learning multivariate Gaussians under the constraint of approximate differential privacy. We prove that $\widetilde{O}(k^2 d \log^{3/2}(1/\delta) / \alpha^2 \varepsilon)$ samples are sufficient to learn a mixture of $k$ axis-aligned Gaussians in $\mathbb{R}^d$ to within total variation distance $\alpha$ while satisfying $(\varepsilon, \delta)$-differential privacy. This is the first result for privately learning mixtures of unbounded axis-aligned (or even unbounded univariate) Gaussians. If the covariance matrices of each of the Gaussians is the identity matrix, we show that $\widetilde{O}(kd/\alpha^2 + kd \log(1/\delta) / \alpha \varepsilon)$ samples are sufficient.To prove our results, we design a new technique for privately learning mixture distributions.  A class of distributions $\mathcal{F}$ is said to be list-decodable if there is an algorithm that, given "heavily corrupted" samples from $f \in \mathcal{F}$, outputs a list of distributions one of which approximates $f$. We show that if $\mathcal{F}$ is privately list-decodable then we can learn mixtures of distributions in $\mathcal{F}$. Finally, we show axis-aligned Gaussian distributions are privately list-decodable, thereby proving mixtures of such distributions are privately learnable.

----

## [300] Deep Self-Dissimilarities as Powerful Visual Fingerprints

**Authors**: *Idan Kligvasser, Tamar Rott Shaham, Yuval Bahat, Tomer Michaeli*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/20479c788fb27378c2c99eadcf207e7f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/20479c788fb27378c2c99eadcf207e7f-Abstract.html)

**Abstract**:

Features extracted from deep layers of classification networks are widely used as image descriptors. Here, we exploit an unexplored property of these features: their internal dissimilarity. While small image patches are known to have similar statistics across image scales, it turns out that the internal distribution of deep features varies distinctively between scales. We show how this deep self dissimilarity (DSD) property can be used as a powerful visual fingerprint. Particularly, we illustrate that full-reference and no-reference image quality measures derived from DSD are highly correlated with human preference. In addition, incorporating DSD as a loss function in training of image restoration networks, leads to results that are at least as photo-realistic as those obtained by GAN based methods, while not requiring adversarial training.

----

## [301] Invariant Causal Imitation Learning for Generalizable Policies

**Authors**: *Ioana Bica, Daniel Jarrett, Mihaela van der Schaar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/204904e461002b28511d5880e1c36a0f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/204904e461002b28511d5880e1c36a0f-Abstract.html)

**Abstract**:

Consider learning an imitation policy on the basis of demonstrated behavior from multiple environments, with an eye towards deployment in an unseen environment. Since the observable features from each setting may be different, directly learning individual policies as mappings from features to actions is prone to spurious correlations---and may not generalize well. However, the expertâ€™s policy is often a function of a shared latent structure underlying those observable features that is invariant across settings. By leveraging data from multiple environments, we propose Invariant Causal Imitation Learning (ICIL), a novel technique in which we learn a feature representation that is invariant across domains, on the basis of which we learn an imitation policy that matches expert behavior. To cope with transition dynamics mismatch, ICIL learns a shared representation of causal features (for all training environments), that is disentangled from the specific representations of noise variables (for each of those environments). Moreover, to ensure that the learned policy matches the observation distribution of the expert's policy, ICIL estimates the energy of the expert's observations and uses a regularization term that minimizes the imitator policy's next state energy. Experimentally, we compare our methods against several benchmarks in control and healthcare tasks and show its effectiveness in learning imitation policies capable of generalizing to unseen environments.

----

## [302] CoAtNet: Marrying Convolution and Attention for All Data Sizes

**Authors**: *Zihang Dai, Hanxiao Liu, Quoc V. Le, Mingxing Tan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html)

**Abstract**:

Transformers have attracted increasing interests in computer vision, but they still fall behind state-of-the-art convolutional networks. In this work, we show that while Transformers tend to have larger model capacity, their generalization can be worse than convolutional networks due to the lack of the right inductive bias. To effectively combine the strengths from both architectures, we present CoAtNets(pronounced "coat" nets), a family of hybrid models built from two key insights: (1) depthwise Convolution and self-Attention can be naturally unified via simple relative attention; (2) vertically stacking convolution layers and attention layers in a principled way is surprisingly effective in improving generalization, capacity and efficiency. Experiments show that our CoAtNets achieve state-of-the-art performance under different resource constraints across various datasets: Without extra data, CoAtNet achieves 86.0% ImageNet top-1 accuracy; When pre-trained with 13M images from ImageNet-21K, our CoAtNet achieves 88.56% top-1 accuracy, matching ViT-huge pre-trained with 300M images from JFT-300M while using 23x less data; Notably, when we further scale up CoAtNet with JFT-3B, it achieves 90.88% top-1 accuracy on ImageNet, establishing a new state-of-the-art result.

----

## [303] Mixed Supervised Object Detection by Transferring Mask Prior and Semantic Similarity

**Authors**: *Yan Liu, Zhijie Zhang, Li Niu, Junjie Chen, Liqing Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/20885c72ca35d75619d6a378edea9f76-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/20885c72ca35d75619d6a378edea9f76-Abstract.html)

**Abstract**:

Object detection has achieved promising success, but requires large-scale fully-annotated data, which is time-consuming and labor-extensive. Therefore, we consider object detection with mixed supervision, which learns novel object categories using weak annotations with the help of full annotations of existing base object categories. Previous works using mixed supervision mainly learn the class-agnostic objectness from fully-annotated categories, which can be transferred to upgrade the weak annotations to pseudo full annotations for novel categories. In this paper, we further transfer mask prior and semantic similarity to bridge the gap between novel categories and base categories. Specifically, the ability of using mask prior to help detect objects is learned from base categories and transferred to novel categories. Moreover, the semantic similarity between objects learned from base categories is transferred to denoise the pseudo full annotations for novel categories. Experimental results on three benchmark datasets demonstrate the effectiveness of our method over existing methods. Codes are available at https://github.com/bcmi/TraMaS-Weak-Shot-Object-Detection.

----

## [304] Celebrating Diversity in Shared Multi-Agent Reinforcement Learning

**Authors**: *Chenghao Li, Tonghan Wang, Chengjie Wu, Qianchuan Zhao, Jun Yang, Chongjie Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/20aee3a5f4643755a79ee5f6a73050ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/20aee3a5f4643755a79ee5f6a73050ac-Abstract.html)

**Abstract**:

Recently, deep multi-agent reinforcement learning (MARL) has shown the promise to solve complex cooperative tasks. Its success is partly because of parameter sharing among agents. However, such sharing may lead agents to behave similarly and limit their coordination capacity. In this paper, we aim to introduce diversity in both optimization and representation of shared multi-agent reinforcement learning. Specifically, we propose an information-theoretical regularization to maximize the mutual information between agents' identities and their trajectories, encouraging extensive exploration and diverse individualized behaviors. In representation, we incorporate agent-specific modules in the shared neural network architecture, which are regularized by L1-norm to promote learning sharing among agents while keeping necessary diversity. Empirical results show that our method achieves state-of-the-art performance on Google Research Football and super hard StarCraft II micromanagement tasks.

----

## [305] Rebounding Bandits for Modeling Satiation Effects

**Authors**: *Liu Leqi, Fatma Kilinç-Karzan, Zachary C. Lipton, Alan L. Montgomery*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2109737282d2c2de4fc5534be26c9bb6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2109737282d2c2de4fc5534be26c9bb6-Abstract.html)

**Abstract**:

Psychological research shows that enjoyment of many goods is subject to satiation, with short-term satisfaction declining after repeated exposures to the same item. Nevertheless, proposed algorithms for powering recommender systems seldom model these dynamics, instead proceeding as though user preferences were fixed in time. In this work, we introduce rebounding bandits, a multi-armed bandit setup, where satiation dynamics are modeled as time-invariant linear dynamical systems. Expected rewards for each arm decline monotonically with consecutive exposures and rebound towards the initial reward whenever that arm is not pulled. Unlike classical bandit algorithms, methods for tackling rebounding bandits must plan ahead and model-based methods rely on estimating the parameters of the satiation dynamics. We characterize the planning problem, showing that the greedy policy is optimal when the arms exhibit identical deterministic dynamics. To address stochastic satiation dynamics with unknown parameters, we propose Explore-Estimate-Plan, an algorithm that pulls arms methodically, estimates the system dynamics, and then plans accordingly.

----

## [306] Sample Complexity of Tree Search Configuration: Cutting Planes and Beyond

**Authors**: *Maria-Florina Balcan, Siddharth Prasad, Tuomas Sandholm, Ellen Vitercik*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/210b7ec74fc9cec6fb8388dbbdaf23f7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/210b7ec74fc9cec6fb8388dbbdaf23f7-Abstract.html)

**Abstract**:

Cutting-plane methods have enabled remarkable successes in integer programming over the last few decades. State-of-the-art solvers integrate a myriad of cutting-plane techniques to speed up the underlying tree-search algorithm used to find optimal solutions. In this paper we provide sample complexity bounds for cut-selection in branch-and-cut (B&C). Given a training set of integer programs sampled from an application-specific input distribution and a family of cut selection policies, these guarantees bound the number of samples sufficient to ensure that using any policy in the family, the size of the tree B&C builds on average over the training set is close to the expected size of the tree B&C builds. We first bound the sample complexity of learning cutting planes from the canonical family of Chv√°tal-Gomory cuts. Our bounds handle any number of waves of any number of cuts and are fine tuned to the magnitudes of the constraint coefficients. Next, we prove sample complexity bounds for more sophisticated cut selection policies that use a combination of scoring rules to choose from a family of cuts. Finally, beyond the realm of cutting planes for integer programming, we develop a general abstraction of tree search that captures key components such as node selection and variable selection. For this abstraction, we bound the sample complexity of learning a good policy for building the search tree.

----

## [307] IQ-Learn: Inverse soft-Q Learning for Imitation

**Authors**: *Divyansh Garg, Shuvam Chakraborty, Chris Cundy, Jiaming Song, Stefano Ermon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/210f760a89db30aa72ca258a3483cc7f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/210f760a89db30aa72ca258a3483cc7f-Abstract.html)

**Abstract**:

In many sequential decision-making problems (e.g., robotics control, game playing, sequential prediction), human or expert data is available containing useful information about the task. However, imitation learning (IL) from a small amount of expert data can be challenging in high-dimensional environments with complex dynamics. Behavioral cloning is a simple method that is widely used due to its simplicity of implementation and stable convergence but doesn't utilize any information involving the environmentâ€™s dynamics. Many existing methods that exploit dynamics information are difficult to train in practice due to an adversarial optimization process over reward and policy approximators or biased, high variance gradient estimators. We introduce a method for dynamics-aware IL which avoids adversarial training by learning a single Q-function, implicitly representing both reward and policy. On standard benchmarks, the implicitly learned rewards show a high positive correlation with the ground-truth rewards, illustrating our method can also be used for inverse reinforcement learning (IRL). Our method, Inverse soft-Q learning (IQ-Learn) obtains state-of-the-art results in offline and online imitation learning settings, significantly outperforming existing methods both in the number of required environment interactions and scalability in high-dimensional spaces, often by more than 3x.

----

## [308] Task-Agnostic Undesirable Feature Deactivation Using Out-of-Distribution Data

**Authors**: *Dongmin Park, Hwanjun Song, Minseok Kim, Jae-Gil Lee*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21186d7b1482412ab14f0332b8aee119-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21186d7b1482412ab14f0332b8aee119-Abstract.html)

**Abstract**:

A deep neural network (DNN) has achieved great success in many machine learning tasks by virtue of its high expressive power. However, its prediction can be easily biased to undesirable features, which are not essential for solving the target task and are even imperceptible to a human, thereby resulting in poor generalization. Leveraging plenty of undesirable features in out-of-distribution (OOD) examples has emerged as a potential solution for de-biasing such features, and a recent study shows that softmax-level calibration of OOD examples can successfully remove the contribution of undesirable features to the last fully-connected layer of a classifier. However, its applicability is confined to the classification task, and its impact on a DNN feature extractor is not properly investigated. In this paper, we propose Taufe, a novel regularizer that deactivates many undesirable features using OOD examples in the feature extraction layer and thus removes the dependency on the task-specific softmax layer. To show the task-agnostic nature of Taufe, we rigorously validate its performance on three tasks, classification, regression, and a mix of them, on CIFAR-10, CIFAR-100, ImageNet, CUB200, and CAR datasets. The results demonstrate that Taufe consistently outperforms the state-of-the-art method as well as the baselines without regularization.

----

## [309] Private Non-smooth ERM and SCO in Subquadratic Steps

**Authors**: *Janardhan Kulkarni, Yin Tat Lee, Daogao Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/211c1e0b83b9c69fa9c4bdede203c1e3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/211c1e0b83b9c69fa9c4bdede203c1e3-Abstract.html)

**Abstract**:

We study the differentially private Empirical Risk Minimization (ERM) and Stochastic Convex Optimization (SCO) problems for non-smooth convex functions. We get a (nearly) optimal bound on the excess empirical risk for ERM with $O(\frac{N^{3/2}}{d^{1/8}}+ \frac{N^2}{d})$ gradient queries, which is achieved with the help of subsampling and smoothing the function via convolution. Combining this result with the iterative localization technique of Feldman et al. \cite{fkt20}, we achieve the optimal excess population loss for the SCO problem with $O(\min\{N^{5/4}d^{1/8},\frac{ N^{3/2}}{d^{1/8}}\})$ gradient queries.Our work makes progress towards resolving a question raised by Bassily et al. \cite{bfgt20}, giving first algorithms for private SCO with subquadratic steps. In a concurrent work, Asi et al. \cite{afkt21} gave other algorithms for private ERM and SCO with subquadratic steps.

----

## [310] Towards Instance-Optimal Offline Reinforcement Learning with Pessimism

**Authors**: *Ming Yin, Yu-Xiang Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/212ab20dbdf4191cbcdcf015511783f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/212ab20dbdf4191cbcdcf015511783f4-Abstract.html)

**Abstract**:

We study the \emph{offline reinforcement learning}  (offline RL) problem, where the goal is to learn a reward-maximizing policy in an unknown \emph{Markov Decision Process} (MDP) using the data coming from a policy $\mu$. In particular, we consider the sample complexity problems of offline RL for the finite horizon MDPs. Prior works derive the information-theoretical lower bounds based on different data-coverage assumptions and their upper bounds are expressed by the covering coefficients which lack the explicit characterization of system quantities. In this work, we analyze the \emph{Adaptive Pessimistic Value Iteration} (APVI) algorithm and derive the suboptimality upper bound that nearly matches\[O\left(\sum_{h=1}^H\sum_{s_h,a_h}d^{\pi^\star}_h(s_h,a_h)\sqrt{\frac{\mathrm{Var}_{P_{s_h,a_h}}{(V^\star_{h+1}+r_h)}}{d^\mu_h(s_h,a_h)}}\sqrt{\frac{1}{n}}\right).\]We also prove an information-theoretical lower bound to show this quantity is required under the weak assumption that $d^\mu_h(s_h,a_h)>0$ if $d^{\pi^\star}_h(s_h,a_h)>0$. Here $\pi^\star$ is a optimal policy, $\mu$ is the behavior policy and $d(s_h,a_h)$ is the marginal state-action probability. We call this adaptive bound the \emph{intrinsic offline reinforcement learning bound} since it directly implies all the existing optimal results: minimax rate under uniform data-coverage assumption, horizon-free setting, single policy concentrability, and the tight problem-dependent results. Later, we extend the result to the \emph{assumption-free} regime (where we make no assumption on $\mu$) and obtain the assumption-free intrinsic bound. Due to its generic form, we believe the intrinsic bound could help illuminate what makes a specific problem hard and reveal the fundamental challenges in offline RL.

----

## [311] Speedy Performance Estimation for Neural Architecture Search

**Authors**: *Robin Ru, Clare Lyle, Lisa Schut, Miroslav Fil, Mark van der Wilk, Yarin Gal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2130eb640e0a272898a51da41363542d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2130eb640e0a272898a51da41363542d-Abstract.html)

**Abstract**:

Reliable yet efficient evaluation of generalisation performance of a proposed architecture is crucial to the success of neural architecture search (NAS). Traditional approaches face a variety of limitations: training each architecture to completion is prohibitively expensive, early stopped validation accuracy may correlate poorly with fully trained performance, and model-based estimators require large training sets. We instead propose to estimate the final test performance based on a simple measure of training speed. Our estimator is theoretically motivated by the connection between generalisation and training speed, and is also inspired by the reformulation of a PAC-Bayes bound under the Bayesian setting. Our model-free estimator is simple, efficient, and cheap to implement, and does not require hyperparameter-tuning or surrogate training before deployment. We demonstrate on various NAS search spaces that our estimator consistently outperforms other alternatives in achieving better correlation with the true test performance rankings. We further show that our estimator can be easily incorporated into both query-based and one-shot NAS methods to improve the speed or quality of the search.

----

## [312] How Tight Can PAC-Bayes be in the Small Data Regime?

**Authors**: *Andrew Y. K. Foong, Wessel P. Bruinsma, David R. Burt, Richard E. Turner*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/214cfbe603b7f9f9bc005d5f53f7a1d3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/214cfbe603b7f9f9bc005d5f53f7a1d3-Abstract.html)

**Abstract**:

In this paper, we investigate the question: _Given a small number of datapoints, for example $N = 30$, how tight can PAC-Bayes and test set bounds be made?_ For such small datasets, test set bounds adversely affect generalisation performance by withholding data from the training procedure. In this setting, PAC-Bayes bounds are especially attractive, due to their ability to use all the data to simultaneously learn a posterior and bound its generalisation risk. We focus on the case of i.i.d. data with a bounded loss and consider the generic PAC-Bayes theorem of Germain et al. While their theorem is known to recover many existing PAC-Bayes bounds, it is unclear what the tightest bound derivable from their framework is. For a fixed learning algorithm and dataset, we show that the tightest possible bound coincides with a bound considered by Catoni; and, in the more natural case of distributions over datasets, we establish a lower bound on the best bound achievable in expectation. Interestingly, this lower bound recovers the Chernoff test set bound if the posterior is equal to the prior. Moreover, to illustrate how tight these bounds can be, we study synthetic one-dimensional classification tasks in which it is feasible to meta-learn both the prior and the form of the bound to numerically optimise for the tightest bounds possible. We find that in this simple, controlled scenario, PAC-Bayes bounds are competitive with comparable, commonly used Chernoff test set bounds. However, the sharpest test set bounds still lead to better guarantees on the generalisation error than the PAC-Bayes bounds we consider.

----

## [313] Deep Synoptic Monte-Carlo Planning in Reconnaissance Blind Chess

**Authors**: *Gregory Clark*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/215a71a12769b056c3c32e7299f1c5ed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/215a71a12769b056c3c32e7299f1c5ed-Abstract.html)

**Abstract**:

This paper introduces deep synoptic Monte Carlo planning (DSMCP) for large imperfect information games. The algorithm constructs a belief state with an unweighted particle filter and plans via playouts that start at samples drawn from the belief state. The algorithm accounts for uncertainty by performing inference on "synopses," a novel stochastic abstraction of information states. DSMCP is the basis of the program Penumbra, which won the official 2020 reconnaissance blind chess competition versus 33 other programs. This paper also evaluates algorithm variants that incorporate caution, paranoia, and a novel bandit algorithm. Furthermore, it audits the synopsis features used in Penumbra with per-bit saliency statistics.

----

## [314] Dynamic Analysis of Higher-Order Coordination in Neuronal Assemblies via De-Sparsified Orthogonal Matching Pursuit

**Authors**: *Shoutik Mukherjee, Behtash Babadi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2172fde49301047270b2897085e4319d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2172fde49301047270b2897085e4319d-Abstract.html)

**Abstract**:

Coordinated ensemble spiking activity is widely observable in neural recordings and central in the study of population codes, with hypothesized roles including robust stimulus representation, interareal communication of neural information, and learning and memory formation. Model-free measures of synchrony characterize the coherence of pairwise activity, but not higher-order interactions; this limitation is transcended by statistical models of ensemble spiking activity. However, existing model-based analyses often impose assumptions about the relevance of higher-order interactions and require multiple repeated trials in order to characterize dynamics in the correlational structure of ensemble activity. To address these shortcomings, we propose an adaptive greedy filtering algorithm based on a discretized mark point-process model of ensemble spiking and a corresponding precise statistical inference framework to identify significant coordinated higher-order spiking activity. In the course of developing the statistical inference procedures, we also show that confidence intervals can be constructed for greedily estimated parameters. We demonstrate the utility of our proposed methods on simulated neuronal assemblies. Applied to multi-electrode recordings of human cortical ensembles, our proposed methods provide new insights into the dynamics underlying localized population activity during transitions between brain states.

----

## [315] Efficient Training of Retrieval Models using Negative Cache

**Authors**: *Erik Lindgren, Sashank J. Reddi, Ruiqi Guo, Sanjiv Kumar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2175f8c5cd9604f6b1e576b252d4c86e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2175f8c5cd9604f6b1e576b252d4c86e-Abstract.html)

**Abstract**:

Factorized models, such as two tower neural network models, are widely used for scoring (query, document) pairs in information retrieval tasks. These models are typically trained by optimizing the model parameters to score relevant positive" pairs higher than the irrelevantnegative" ones. While a large set of negatives typically improves the model performance, limited computation and memory budgets place constraints on the number of negatives used during training. In this paper, we develop a novel negative sampling technique for accelerating training with softmax cross-entropy loss. By using cached (possibly stale) item embeddings, our technique enables training with a large pool of negatives with reduced memory and computation. We also develop a streaming variant of our algorithm geared towards very large datasets. Furthermore, we establish a theoretical basis for our approach by showing that updating a very small fraction of the cache at each iteration can still ensure fast convergence. Finally, we experimentally validate our approach and show that it is efficient and compares favorably with more complex, state-of-the-art approaches.

----

## [316] Understanding Partial Multi-Label Learning via Mutual Information

**Authors**: *Xiuwen Gong, Dong Yuan, Wei Bao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/217c0e01c1828e7279051f1b6675745d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/217c0e01c1828e7279051f1b6675745d-Abstract.html)

**Abstract**:

To deal with ambiguities in partial multilabel learning (PML), state-of-the-art methods perform disambiguation by identifying ground-truth labels directly. However, there is an essential question:“Can the ground-truth labels be identified precisely?". If yes, “How can the ground-truth labels be found?". This paper provides affirmative answers to these questions. Instead of adopting hand-made heuristic strategy, we propose a novel Mutual Information Label Identification for Partial Multilabel Learning (MILI-PML), which is derived from a clear probabilistic formulation and could be easily interpreted theoretically from the mutual information perspective, as well as naturally incorporates the feature/label relevancy considerations. Extensive experiments on synthetic and real-world datasets clearly demonstrate the superiorities of the proposed MILI-PML.

----

## [317] Environment Generation for Zero-Shot Compositional Reinforcement Learning

**Authors**: *Izzeddin Gur, Natasha Jaques, Yingjie Miao, Jongwook Choi, Manoj Tiwari, Honglak Lee, Aleksandra Faust*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/218344619d8fb95d504ccfa11804073f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/218344619d8fb95d504ccfa11804073f-Abstract.html)

**Abstract**:

Many real-world problems are compositional – solving them requires completing interdependent sub-tasks, either in series or in parallel, that can be represented as a dependency graph. Deep reinforcement learning (RL) agents often struggle to learn such complex tasks due to the long time horizons and sparse rewards. To address this problem, we present Compositional Design of Environments (CoDE), which trains a Generator agent to automatically build a series of compositional tasks tailored to the RL agent’s current skill level. This automatic curriculum not only enables the agent to learn more complex tasks than it could have otherwise, but also selects tasks where the agent’s performance is weak, enhancing its robustness and ability to generalize zero-shot to unseen tasks at test-time. We analyze why current environment generation techniques are insufficient for the problem of generating compositional tasks, and propose a new algorithm that addresses these issues. Our results assess learning and generalization across multiple compositional tasks, including the real-world problem of learning to navigate and interact with web pages. We learn to generate environments composed of multiple pages or rooms, and train RL agents capable of completing wide-range of complex tasks in those environments. We contribute two new benchmark frameworks for generating compositional tasks, compositional MiniGrid and gMiniWoB for web navigation. CoDE yields 4x higher success rate than the strongest baseline, and demonstrates strong performance of real websites learned on 3500 primitive tasks.

----

## [318] Optimizing Conditional Value-At-Risk of Black-Box Functions

**Authors**: *Quoc Phong Nguyen, Zhongxiang Dai, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/219ece62fae865562d4510ea501cf349-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/219ece62fae865562d4510ea501cf349-Abstract.html)

**Abstract**:

This paper presents two Bayesian optimization (BO) algorithms with theoretical performance guarantee to maximize the conditional value-at-risk (CVaR) of a black-box function: CV-UCB and CV-TS which are based on the well-established principle of optimism in the face of uncertainty and Thompson sampling, respectively. To achieve this, we develop an upper confidence bound of CVaR and prove the no-regret guarantee of CV-UCB by utilizing an interesting connection between CVaR and value-at-risk (VaR). For CV-TS, though it is straightforwardly performed with Thompson sampling, bounding its Bayesian regret is non-trivial because it requires a tail expectation bound for the distribution of CVaR of a black-box function, which has not been shown in the literature. The performances of both CV-UCB and CV-TS are empirically evaluated in optimizing CVaR of synthetic benchmark functions and simulated real-world optimization problems.

----

## [319] E(n) Equivariant Normalizing Flows

**Authors**: *Victor Garcia Satorras, Emiel Hoogeboom, Fabian Fuchs, Ingmar Posner, Max Welling*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21b5680d80f75a616096f2e791affac6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21b5680d80f75a616096f2e791affac6-Abstract.html)

**Abstract**:

This paper introduces a generative model equivariant to Euclidean symmetries: E(n) Equivariant Normalizing Flows (E-NFs). To construct E-NFs, we take the discriminative E(n) graph neural networks and integrate them as a differential equation to obtain an invertible equivariant function: a continuous-time normalizing flow. We demonstrate that E-NFs considerably outperform baselines and existing methods from the literature on particle systems such as DW4 and LJ13, and on molecules from QM9 in terms of log-likelihood. To the best of our knowledge, this is the first flow that jointly generates molecule features and positions in 3D.

----

## [320] Revitalizing CNN Attention via Transformers in Self-Supervised Visual Representation Learning

**Authors**: *Chongjian Ge, Youwei Liang, Yibing Song, Jianbo Jiao, Jue Wang, Ping Luo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21be992eb8016e541a15953eee90760e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21be992eb8016e541a15953eee90760e-Abstract.html)

**Abstract**:

Studies on self-supervised visual representation learning (SSL) improve encoder backbones to discriminate training samples without labels. While CNN encoders via SSL achieve comparable recognition performance to those via supervised learning, their network attention is under-explored for further improvement. Motivated by the transformers that explore visual attention effectively in recognition scenarios, we propose a CNN Attention REvitalization (CARE) framework to train attentive CNN encoders guided by transformers in SSL. The proposed CARE framework consists of a CNN stream (C-stream) and a transformer stream (T-stream), where each stream contains two branches. C-stream follows an existing SSL framework with two CNN encoders, two projectors, and a predictor. T-stream contains two transformers, two projectors, and a predictor. T-stream connects to CNN encoders and is in parallel to the remaining C-Stream. During training, we perform SSL in both streams simultaneously and use the T-stream output to supervise C-stream. The features from CNN encoders are modulated in T-stream for visual attention enhancement and become suitable for the SSL scenario. We use these modulated features to supervise C-stream for learning attentive CNN encoders. To this end, we revitalize CNN attention by using transformers as guidance. Experiments on several standard visual recognition benchmarks, including image classification, object detection, and semantic segmentation, show that the proposed CARE framework improves CNN encoder backbones to the state-of-the-art performance.

----

## [321] A Critical Look at the Consistency of Causal Estimation with Deep Latent Variable Models

**Authors**: *Severi Rissanen, Pekka Marttinen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21c5bba1dd6aed9ab48c2b34c1a0adde-Abstract.html)

**Abstract**:

Using deep latent variable models in causal inference has attracted considerable interest recently, but an essential open question is their ability to yield consistent causal estimates. While they have demonstrated promising results and theory exists on some simple model formulations, we also know that causal effects are not even identifiable in general with latent variables. We investigate this gap between theory and empirical results with analytical considerations and extensive experiments under multiple synthetic and real-world data sets, using the causal effect variational autoencoder (CEVAE) as a case study. While CEVAE seems to work reliably under some simple scenarios, it does not estimate the causal effect correctly with a misspecified latent variable or a complex data distribution, as opposed to its original motivation. Hence, our results show that more attention should be paid to ensuring the correctness of causal estimates with deep latent variable models.

----

## [322] Improving Robustness using Generated Data

**Authors**: *Sven Gowal, Sylvestre-Alvise Rebuffi, Olivia Wiles, Florian Stimberg, Dan Andrei Calian, Timothy A. Mann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21ca6d0cf2f25c4dbb35d8dc0b679c3f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21ca6d0cf2f25c4dbb35d8dc0b679c3f-Abstract.html)

**Abstract**:

Recent work argues that robust training requires substantially larger datasets than those required for standard classification. On CIFAR-10 and CIFAR-100, this translates into a sizable robust-accuracy gap between models trained solely on data from the original training set and those trained with additional data extracted from the "80 Million Tiny Images" dataset (TI-80M). In this paper, we explore how generative models trained solely on the original training set can be leveraged to artificially increase the size of the original training set and improve adversarial robustness to $\ell_p$ norm-bounded perturbations. We identify the sufficient conditions under which incorporating additional generated data can improve robustness, and demonstrate that it is possible to significantly reduce the robust-accuracy gap to models trained with additional real data. Surprisingly, we even show that even the addition of non-realistic random data (generated by Gaussian sampling) can improve robustness. We evaluate our approach on CIFAR-10, CIFAR-100, SVHN and TinyImageNet against $\ell_\infty$ and $\ell_2$ norm-bounded perturbations of size $\epsilon = 8/255$ and $\epsilon = 128/255$, respectively. We show large absolute improvements in robust accuracy compared to previous state-of-the-art methods. Against $\ell_\infty$ norm-bounded perturbations of size $\epsilon = 8/255$, our models achieve 66.10% and 33.49% robust accuracy on CIFAR-10 and CIFAR-100, respectively (improving upon the state-of-the-art by +8.96% and +3.29%). Against $\ell_2$ norm-bounded perturbations of size $\epsilon = 128/255$, our model achieves 78.31% on CIFAR-10 (+3.81%). These results beat most prior works that use external data.

----

## [323] An Analysis of Constant Step Size SGD in the Non-convex Regime: Asymptotic Normality and Bias

**Authors**: *Lu Yu, Krishnakumar Balasubramanian, Stanislav Volgushev, Murat A. Erdogdu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21ce689121e39821d07d04faab328370-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21ce689121e39821d07d04faab328370-Abstract.html)

**Abstract**:

Structured non-convex learning problems, for which critical points have favorable statistical properties, arise frequently in statistical machine learning. Algorithmic convergence and statistical estimation rates are well-understood for such problems. However, quantifying the uncertainty associated with the underlying training algorithm is not well-studied in the non-convex setting. In order to address this shortcoming, in this work, we establish an asymptotic normality result for the constant step size stochastic gradient descent (SGD)  algorithm---a widely used algorithm in practice. Specifically, based on the relationship between SGD and Markov Chains  [DDB19], we show that the average of SGD iterates is asymptotically normally distributed around the expected value of their unique invariant distribution, as long as the non-convex and non-smooth objective function satisfies a dissipativity property. We also characterize the bias between this expected value and the critical points of the objective function under various local regularity conditions. Together, the above two results could be leveraged to construct confidence intervals for non-convex problems that are trained using the SGD algorithm.

----

## [324] Learning to Learn Graph Topologies

**Authors**: *Xingyue Pu, Tianyue Cao, Xiaoyun Zhang, Xiaowen Dong, Siheng Chen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/21e4ef94f2a6b23597efabaec584b504-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/21e4ef94f2a6b23597efabaec584b504-Abstract.html)

**Abstract**:

Learning a graph topology to reveal the underlying relationship between data entities plays an important role in various machine learning and data analysis tasks. Under the assumption that structured data vary smoothly over a graph, the problem can be formulated as a regularised convex optimisation over a positive semidefinite cone and solved by iterative algorithms. Classic methods require an explicit convex function to reflect generic topological priors, e.g. the $\ell_1$ penalty for enforcing sparsity, which limits the flexibility and expressiveness in learning rich topological structures. We propose to learn a mapping from node data to the graph structure based on the idea of learning to optimise (L2O). Specifically, our model first unrolls an iterative primal-dual splitting algorithm into a neural network. The key structural proximal projection is replaced with a variational autoencoder that refines the estimated graph with enhanced topological properties. The model is trained in an end-to-end fashion with pairs of node data and graph samples. Experiments on both synthetic and real-world data demonstrate that our model is more efficient than classic iterative algorithms in learning a graph with specific topological properties.

----

## [325] Invertible Tabular GANs: Killing Two Birds with One Stone for Tabular Data Synthesis

**Authors**: *Jaehoon Lee, Jihyeon Hyeong, Jinsung Jeon, Noseong Park, Jihoon Cho*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/22456f4b545572855c766df5eefc9832-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/22456f4b545572855c766df5eefc9832-Abstract.html)

**Abstract**:

Tabular data synthesis has received wide attention in the literature. This is because available data is often limited, incomplete, or cannot be obtained easily, and data privacy is becoming increasingly important. In this work, we present a generalized GAN framework for tabular synthesis, which combines the adversarial training of GANs and the negative log-density regularization of invertible neural networks. The proposed framework can be used for two distinctive objectives. First,  we can further improve the synthesis quality, by decreasing the negative log-density of real records in the process of adversarial training. On the other hand, by increasing the negative log-density of real records, realistic fake records can be synthesized in a way that they are not too much close to real records and reduce the chance of potential information leakage. We conduct experiments with real-world datasets for classification, regression, and privacy attacks. In general, the proposed method demonstrates the best synthesis quality (in terms of task-oriented evaluation metrics, e.g., F1) when decreasing the negative log-density during the adversarial training. If increasing the negative log-density, our experimental results show that the distance between real and fake records increases, enhancing robustness against privacy attacks.

----

## [326] Reducing Collision Checking for Sampling-Based Motion Planning Using Graph Neural Networks

**Authors**: *Chenning Yu, Sicun Gao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/224e5e49814ca908e58c02e28a0462c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/224e5e49814ca908e58c02e28a0462c1-Abstract.html)

**Abstract**:

Sampling-based motion planning is a popular approach in robotics for finding paths in continuous configuration spaces. Checking collision with obstacles is the major computational bottleneck in this process. We propose new learning-based methods for reducing collision checking to accelerate motion planning by training graph neural networks (GNNs) that perform path exploration and path smoothing. Given random geometric graphs (RGGs) generated from batch sampling, the path exploration component iteratively predicts collision-free edges to prioritize their exploration. The path smoothing component then optimizes paths obtained from the exploration stage. The methods benefit from the ability of GNNs of capturing geometric patterns from RGGs through batch sampling and generalize better to unseen environments. Experimental results show that the learned components can significantly reduce collision checking and improve overall planning efficiency in challenging high-dimensional motion planning tasks.

----

## [327] Sample Complexity Bounds for Active Ranking from Multi-wise Comparisons

**Authors**: *Wenbo Ren, Jia Liu, Ness B. Shroff*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/22508552d3fc22f867e33e6c56b30b16-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/22508552d3fc22f867e33e6c56b30b16-Abstract.html)

**Abstract**:

We study the sample complexity (i.e., the number of comparisons needed) bounds for actively ranking a set of $n$ items from multi-wise comparisons. Here, a multi-wise comparison takes $m$ items as input and returns a (noisy) result about the best item (the winner feedback) or the order of these items (the full-ranking feedback). We consider two basic ranking problems: top-$k$ items selection and full ranking. Unlike previous works that study ranking from multi-wise comparisons, in this paper, we do not require any parametric model or assumption and work on the fundamental setting where each comparison returns the correct result with probability $1$ or a certain probability larger than $\frac{1}{2}$. This paper helps understand whether and to what degree utilizing multi-wise comparisons can reduce the sample complexity for the ranking problems compared to ranking from pairwise comparisons. Specifically, under the winner feedback setting, one can reduce the sample complexity for top-$k$ selection up to an $m$ factor and that for full ranking up to a $\log{m}$ factor. Under the full-ranking feedback setting, one can reduce the sample complexity for top-$k$ selection up to an $m$ factor and that for full ranking up to an $m\log{m}$ factor. We also conduct numerical simulations to confirm our theoretical results.

----

## [328] Efficient Bayesian network structure learning via local Markov boundary search

**Authors**: *Ming Gao, Bryon Aragam*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/22722a343513ed45f14905eb07621686-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/22722a343513ed45f14905eb07621686-Abstract.html)

**Abstract**:

We analyze the complexity of learning directed acyclic graphical models from observational data in general settings without specific distributional assumptions. Our approach is information-theoretic and uses a local Markov boundary search procedure in order to recursively construct ancestral sets in the underlying graphical model. Perhaps surprisingly, we show that for certain graph ensembles, a simple forward greedy search algorithm (i.e. without a backward pruning phase) suffices to learn the Markov boundary of each node. This substantially improves the sample complexity, which we show is at most polynomial in the number of nodes. This is then applied to learn the entire graph under a novel identifiability condition that generalizes existing conditions from the literature. As a matter of independent interest, we establish finite-sample guarantees for the problem of recovering Markov boundaries from data. Moreover, we apply our results to the special case of polytrees, for which the assumptions simplify, and provide explicit conditions under which polytrees are identifiable and learnable in polynomial time. We further illustrate the performance of the algorithm, which is easy to implement, in a simulation study. Our approach is general, works for discrete or continuous distributions without distributional assumptions, and as such sheds light on the minimal assumptions required to efficiently learn the structure of directed graphical models from data.

----

## [329] Learning Dynamic Graph Representation of Brain Connectome with Spatio-Temporal Attention

**Authors**: *Byung-Hoon Kim, Jong Chul Ye, Jae-Jin Kim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/22785dd2577be2ce28ef79febe80db10-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/22785dd2577be2ce28ef79febe80db10-Abstract.html)

**Abstract**:

Functional connectivity (FC) between regions of the brain can be assessed by the degree of temporal correlation measured with functional neuroimaging modalities. Based on the fact that these connectivities build a network, graph-based approaches for analyzing the brain connectome have provided insights into the functions of the human brain. The development of graph neural networks (GNNs) capable of learning representation from graph structured data has led to increased interest in learning the graph representation of the brain connectome. Although recent attempts to apply GNN to the FC network have shown promising results, there is still a common limitation that they usually do not incorporate the dynamic characteristics of the FC network which fluctuates over time. In addition, a few studies that have attempted to use dynamic FC as an input for the GNN reported a reduction in performance compared to static FC methods, and did not provide temporal explainability. Here, we propose STAGIN, a method for learning dynamic graph representation of the brain connectome with spatio-temporal attention. Specifically, a temporal sequence of brain graphs is input to the STAGIN to obtain the dynamic graph representation, while novel READOUT functions and the Transformer encoder provide spatial and temporal explainability with attention, respectively. Experiments on the HCP-Rest and the HCP-Task datasets demonstrate exceptional performance of our proposed method. Analysis of the spatio-temporal attention also provide concurrent interpretation with the neuroscientific knowledge, which further validates our method. Code is available at https://github.com/egyptdj/stagin

----

## [330] Understanding the Generalization Benefit of Model Invariance from a Data Perspective

**Authors**: *Sicheng Zhu, Bang An, Furong Huang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2287c6b8641dd2d21ab050eb9ff795f3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2287c6b8641dd2d21ab050eb9ff795f3-Abstract.html)

**Abstract**:

Machine learning models that are developed to be invariant under certain types of data transformations have shown improved generalization in practice. However, a principled understanding of why invariance benefits generalization is limited. Given a dataset, there is often no principled way to select "suitable" data transformations under which model invariance guarantees better generalization. This paper studies the generalization benefit of model invariance by introducing the sample cover induced by transformations, i.e., a representative subset of a dataset that can approximately recover the whole dataset using transformations. For any data transformations, we provide refined generalization bounds for invariant models based on the sample cover. We also characterize the "suitability" of a set of data transformations by the sample covering number induced by transformations, i.e., the smallest size of its induced sample covers. We show that we may tighten the generalization bounds for "suitable" transformations that have a small sample covering number. In addition, our proposed sample covering number can be empirically evaluated and thus provides a guidance for selecting transformations to develop model invariance for better generalization. In experiments on multiple datasets, we evaluate sample covering numbers for some commonly used transformations and show that the smaller sample covering number for a set of transformations (e.g., the 3D-view transformation) indicates a smaller gap between the test and training error for invariant models, which verifies our propositions.

----

## [331] Improved Variance-Aware Confidence Sets for Linear Bandits and Linear Mixture MDP

**Authors**: *Zihan Zhang, Jiaqi Yang, Xiangyang Ji, Simon S. Du*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/228bbc2f87caeb21bb7f6949fddcb91d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/228bbc2f87caeb21bb7f6949fddcb91d-Abstract.html)

**Abstract**:

This paper presents new \emph{variance-aware} confidence sets for linear bandits and linear mixture Markov Decision Processes (MDPs).With the new confidence sets, we obtain the follow regret bounds:For linear bandits, we obtain an $\widetilde{O}(\mathrm{poly}(d)\sqrt{1 + \sum_{k=1}^{K}\sigma_k^2})$ data-dependent regret bound, where $d$ is the feature dimension, $K$ is the number of rounds, and $\sigma_k^2$ is the \emph{unknown} variance of the reward at the $k$-th round. This is the first regret bound that only scales with the variance and the dimension but \emph{no explicit polynomial dependency on $K$}.When variances are small, this bound can be significantly smaller than the $\widetilde{\Theta}\left(d\sqrt{K}\right)$ worst-case regret bound.For linear mixture MDPs, we obtain an $\widetilde{O}(\mathrm{poly}(d, \log H)\sqrt{K})$ regret bound, where $d$ is the number of base models, $K$ is the number of episodes, and $H$ is the planning horizon. This is the first regret bound that only scales \emph{logarithmically} with $H$ in the reinforcement learning with linear function approximation setting, thus \emph{exponentially improving} existing results, and resolving an open problem in \citep{zhou2020nearly}.We develop three technical ideas that may be of independent interest:1) applications of the peeling technique to both the input norm and the variance magnitude, 2) a recursion-based estimator for the variance, and 3) a new convex potential lemma that generalizes the seminal elliptical potential lemma.

----

## [332] How Should Pre-Trained Language Models Be Fine-Tuned Towards Adversarial Robustness?

**Authors**: *Xinshuai Dong, Anh Tuan Luu, Min Lin, Shuicheng Yan, Hanwang Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/22b1f2e0983160db6f7bb9f62f4dbb39-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/22b1f2e0983160db6f7bb9f62f4dbb39-Abstract.html)

**Abstract**:

The fine-tuning of pre-trained language models has a great success in many NLP fields. Yet, it is strikingly vulnerable to adversarial examples, e.g., word substitution attacks using only synonyms can easily fool a BERT-based sentiment analysis model. In this paper, we demonstrate that adversarial training, the prevalent defense technique, does not directly fit a conventional fine-tuning scenario, because it suffers severely from catastrophic forgetting: failing to retain the generic and robust linguistic features that have already been captured by the pre-trained model. In this light, we propose Robust Informative Fine-Tuning (RIFT), a novel adversarial fine-tuning method from an information-theoretical perspective. In particular, RIFT encourages an objective model to retain the features learned from the pre-trained model throughout the entire fine-tuning process, whereas a conventional one only uses the pre-trained weights for initialization. Experimental results show that RIFT consistently outperforms the state-of-the-arts on two popular NLP tasks: sentiment analysis and natural language inference, under different attacks across various pre-trained language models.

----

## [333] Recursive Bayesian Networks: Generalising and Unifying Probabilistic Context-Free Grammars and Dynamic Bayesian Networks

**Authors**: *Robert Lieck, Martin Rohrmeier*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/22fb0cee7e1f3bde58293de743871417-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/22fb0cee7e1f3bde58293de743871417-Abstract.html)

**Abstract**:

Probabilistic context-free grammars (PCFGs) and dynamic Bayesian networks (DBNs) are widely used sequence models with complementary strengths and limitations. While PCFGs allow for nested hierarchical dependencies (tree structures), their latent variables (non-terminal symbols) have to be discrete. In contrast, DBNs allow for continuous latent variables, but the dependencies are strictly sequential (chain structure). Therefore, neither can be applied if the latent variables are assumed to be continuous and also to have a nested hierarchical dependency structure. In this paper, we present Recursive Bayesian Networks (RBNs), which generalise and unify PCFGs and DBNs, combining their strengths and containing both as special cases. RBNs define a joint distribution over tree-structured Bayesian networks with discrete or continuous latent variables. The main challenge lies in performing joint inference over the exponential number of possible structures and the continuous variables. We provide two solutions: 1) For arbitrary RBNs, we generalise inside and outside probabilities from PCFGs to the mixed discrete-continuous case, which allows for maximum posterior estimates of the continuous latent variables via gradient descent, while marginalising over network structures. 2) For Gaussian RBNs, we additionally derive an analytic approximation of the marginal data likelihood (evidence) and marginal posterior distribution, allowing for robust parameter optimisation and Bayesian inference. The capacity and diverse applications of RBNs are illustrated on two examples: In a quantitative evaluation on synthetic data, we demonstrate and discuss the advantage of RBNs for segmentation and tree induction from noisy sequences, compared to change point detection and hierarchical clustering. In an application to musical data, we approach the unsolved problem of hierarchical music analysis from the raw note level and compare our results to expert annotations.

----

## [334] EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback

**Authors**: *Peter Richtárik, Igor Sokolov, Ilyas Fatkhullin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)

**Abstract**:

Error feedback (EF), also known as error compensation, is an immensely popular convergence stabilization mechanism in the context of distributed training of supervised machine learning models enhanced by the use of contractive communication compression mechanisms, such as Top-$k$. First proposed by Seide et al [2014] as a heuristic, EF resisted any theoretical understanding until recently [Stich et al., 2018, Alistarh et al., 2018]. While these early breakthroughs were followed by a steady stream of works offering various improvements and generalizations, the current theoretical understanding of EF is still very limited. Indeed, to the best of our knowledge, all existing analyses either i) apply to the single node setting only, ii) rely on very strong and often unreasonable assumptions, such as global boundedness of the gradients, or iterate-dependent assumptions that cannot be checked a-priori and may not hold in practice, or iii) circumvent these issues via the introduction of additional unbiased compressors, which increase the communication cost. In this work we fix all these deficiencies by proposing and analyzing a new EF mechanism, which we call EF21, which consistently and substantially outperforms EF in practice. Moreover, our theoretical analysis relies on standard assumptions only, works in the distributed heterogeneous data setting, and leads to better and more meaningful rates. In particular, we prove that EF21 enjoys a fast $\mathcal{O}(1/T)$  convergence rate for smooth nonconvex problems, beating the previous bound of $\mathcal{O}(1/T^{2/3})$, which was shown under a strong bounded gradients assumption. We further improve this to a fast linear rate for Polyak-Lojasiewicz functions, which is the first linear convergence result for an error feedback method not relying on unbiased compressors. Since EF has a large number of applications where it reigns supreme, we believe that our 2021 variant, EF21, will have a large impact on the practice of communication efficient distributed learning.

----

## [335] Mixture weights optimisation for Alpha-Divergence Variational Inference

**Authors**: *Kamélia Daudel, Randal Douc*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/233f1dd0f3f537bcb7a338ea74d63483-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/233f1dd0f3f537bcb7a338ea74d63483-Abstract.html)

**Abstract**:

This paper focuses on $\alpha$-divergence minimisation methods for Variational Inference. More precisely, we are interested in algorithms optimising the mixture weights of any given mixture model, without any information on the underlying distribution of its mixture components parameters. The Power Descent, defined for all $\alpha \neq 1$, is one such algorithm and we establish in our work the full proof of its convergence towards the optimal mixture weights when $\alpha <1$. Since the $\alpha$-divergence recovers the widely-used forward Kullback-Leibler when $\alpha \to 1$, we then extend the Power Descent to the case $\alpha = 1$ and show that we obtain an Entropic Mirror Descent. This leads us to investigate the link between Power Descent and Entropic Mirror Descent: first-order approximations allow us to introduce the R\'{e}nyi Descent, a novel algorithm for which we prove an $O(1/N)$ convergence rate. Lastly, we compare numerically the behavior of the unbiased Power Descent and of the biased R\'{e}nyi Descent and we discuss the potential advantages of one algorithm over the other.

----

## [336] Instance-dependent Label-noise Learning under a Structural Causal Model

**Authors**: *Yu Yao, Tongliang Liu, Mingming Gong, Bo Han, Gang Niu, Kun Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/23451391cd1399019fa0421129066bc6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/23451391cd1399019fa0421129066bc6-Abstract.html)

**Abstract**:

Label noise generally degenerates the performance of deep learning algorithms because deep neural networks easily overfit label errors. Let $X$ and $Y$ denote the instance and clean label, respectively. When $Y$ is a cause of $X$, according to which many datasets have been constructed, e.g., \textit{SVHN} and \textit{CIFAR}, the distributions of $P(X)$ and $P(Y|X)$ are generally entangled. This means that the unsupervised instances are helpful to learn the classifier and thus reduce the side effect of label noise. However, it remains elusive on how to exploit the causal information to handle the label-noise problem. We propose to model and make use of the causal process in order to correct the label-noise effect.Empirically, the proposed method outperforms all state-of-the-art methods on both synthetic and real-world label-noise datasets.

----

## [337] Combining Human Predictions with Model Probabilities via Confusion Matrices and Calibration

**Authors**: *Gavin Kerrigan, Padhraic Smyth, Mark Steyvers*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/234b941e88b755b7a72a1c1dd5022f30-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/234b941e88b755b7a72a1c1dd5022f30-Abstract.html)

**Abstract**:

An increasingly common use case for machine learning models is augmenting the abilities of human decision makers. For classification tasks where neither the human nor model are perfectly accurate, a key step in obtaining high performance is combining their individual predictions in a manner that leverages their relative strengths. In this work, we develop a set of algorithms that combine the probabilistic output of a model with the class-level output of a human. We show theoretically that the accuracy of our combination model is driven not only by the individual human and model accuracies, but also by the model's confidence.  Empirical results on image classification with CIFAR-10 and a subset of ImageNet demonstrate that such human-model combinations consistently have higher accuracies than the model or human alone, and that the parameters of the combination method can be estimated effectively with as few as ten labeled datapoints.

----

## [338] $\texttt{LeadCache}$: Regret-Optimal Caching in Networks

**Authors**: *Debjit Paria, Abhishek Sinha*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2387337ba1e0b0249ba90f55b2ba2521-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2387337ba1e0b0249ba90f55b2ba2521-Abstract.html)

**Abstract**:

We consider an online prediction problem in the context of network caching. Assume that multiple users are connected to several caches via a bipartite network. At any time slot, each user may request an arbitrary file chosen from a large catalog. A user's request at a slot is met if the requested file is cached in at least one of the caches connected to the user. Our objective is to predict, prefetch, and optimally distribute the files on the caches at each slot to maximize the total number of cache hits. The problem is non-trivial due to the non-convex and non-smooth nature of the objective function. In this paper, we propose $\texttt{LeadCache}$ - an efficient online caching policy based on the Follow-the-Perturbed-Leader paradigm. We show that $\texttt{LeadCache}$ is regret-optimal up to a factor of $\tilde{O}(n^{3/8}),$ where $n$ is the number of users. We design two efficient implementations of the $\texttt{LeadCache}$ policy, one based on Pipage rounding and the other based on Madow's sampling, each of which makes precisely one call to an LP-solver per iteration. Furthermore, with a Strong-Law-type assumption, we show that the total number of file fetches under $\texttt{LeadCache}$ remains almost surely finite over an infinite horizon. Finally, we derive an approximately tight regret lower bound using results from graph coloring. We conclude that the learning-based $\texttt{LeadCache}$ policy decisively outperforms the state-of-the-art caching policies both theoretically and empirically.

----

## [339] Probabilistic Attention for Interactive Segmentation

**Authors**: *Prasad Gabbur, Manjot Bilkhu, Javier R. Movellan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/23937b42f9273974570fb5a56a6652ee-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/23937b42f9273974570fb5a56a6652ee-Abstract.html)

**Abstract**:

We provide a probabilistic interpretation of attention and show that the standard dot-product attention in transformers is a special case of Maximum A Posteriori (MAP) inference. The proposed approach suggests the use of Expectation Maximization algorithms for on-line adaptation of key and value model parameters. This approach is  useful for cases in which external agents, e.g., annotators, provide inference-time information about the correct values of some tokens, e.g., the semantic category of some pixels,  and we need for this new information to propagate to other tokens in a principled manner. We illustrate the approach on an interactive semantic segmentation task in which annotators and models collaborate online to improve annotation efficiency. Using standard benchmarks, we observe that key adaptation boosts model performance ($\sim10\%$ mIoU) in the low feedback regime and value propagation improves model responsiveness in the high feedback regime. A PyTorch layer implementation of our probabilistic attention model is available here: https://github.com/apple/ml-probabilistic-attention.

----

## [340] Influence Patterns for Explaining Information Flow in BERT

**Authors**: *Kaiji Lu, Zifan Wang, Piotr Mardziel, Anupam Datta*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/239f914f30ea3c948fce2ea07a9efb33-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/239f914f30ea3c948fce2ea07a9efb33-Abstract.html)

**Abstract**:

While attention is all you need may be proving true, we do not know why: attention-based transformer models such as BERT are superior but how information flows from input tokens to output predictions are unclear.  We introduce influence patterns,  abstractions of sets of paths  through a transformer model. Patterns quantify and localize the flow of  information to paths passing through a sequence of model nodes. Experimentally, we find that significant portion of information flow in BERT goes through skip connections instead of attention heads. We further show that consistency of patterns across instances is an indicator of BERTâ€™s performance. Finally, we demonstrate that patterns account for far more model performance than previous attention-based and layer-based methods.

----

## [341] Robust Regression Revisited: Acceleration and Improved Estimation Rates

**Authors**: *Arun Jambulapati, Jerry Li, Tselil Schramm, Kevin Tian*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/23b023b22d0bf47626029d5961328028-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/23b023b22d0bf47626029d5961328028-Abstract.html)

**Abstract**:

We study fast algorithms for statistical regression problems under the strong contamination model, where the goal is to approximately optimize a generalized linear model (GLM) given adversarially corrupted samples. Prior works in this line of research were based on the \emph{robust gradient descent} framework of \cite{PrasadSBR20}, a first-order method using biased gradient queries, or the \emph{Sever} framework of \cite{DiakonikolasKK019}, an iterative outlier-removal method calling a stationary point finder. We present nearly-linear time algorithms for robust regression problems with improved runtime or estimation guarantees compared to the state-of-the-art. For the general case of smooth GLMs (e.g.\ logistic regression), we show that the robust gradient descent framework of \cite{PrasadSBR20} can be \emph{accelerated}, and show our algorithm extends to optimizing the Moreau envelopes of Lipschitz GLMs (e.g.\ support vector machines), answering several open questions in the literature. For the well-studied case of robust linear regression, we present an alternative approach obtaining improved estimation rates over prior nearly-linear time algorithms. Interestingly, our algorithm starts with an identifiability proof introduced in the context of the sum-of-squares algorithm of \cite{BakshiP21}, which achieved optimal error rates while requiring large polynomial runtime and sample complexity. We reinterpret their proof within the Sever framework and obtain a dramatically faster and more sample-efficient algorithm under fewer distributional assumptions.

----

## [342] Automatic Unsupervised Outlier Model Selection

**Authors**: *Yue Zhao, Ryan A. Rossi, Leman Akoglu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html)

**Abstract**:

Given an unsupervised outlier detection task on a new dataset, how can we automatically select a good outlier detection algorithm and its hyperparameter(s) (collectively called a model)? In this work, we tackle the unsupervised outlier model selection (UOMS) problem, and propose MetaOD, a principled, data-driven approach to UOMS based on meta-learning. The UOMS problem is notoriously challenging, as compared to model selection for classification and clustering, since (i) model evaluation is infeasible due to the lack of hold-out data with labels, and (ii) model comparison is infeasible due to the lack of a universal objective function. MetaOD capitalizes on the performances of a large body of detection models on historical outlier detection benchmark datasets, and carries over this prior experience to automatically select an effective model to be employed on a new dataset without any labels, model evaluations or model comparisons. To capture task similarity within our meta-learning framework, we introduce specialized meta-features that quantify outlying characteristics of a dataset. Extensive experiments show that selecting a model by MetaOD significantly outperforms no model selection (e.g. always using the same popular model or the ensemble of many) as well as other meta-learning techniques that we tailored for UOMS. Moreover upon (meta-)training, MetaOD is extremely efficient at test time; selecting from a large pool of 300+ models takes less than 1 second for a new task. We open-source MetaOD and our meta-learning database for practical use and to foster further research on the UOMS problem.

----

## [343] Pruning Randomly Initialized Neural Networks with Iterative Randomization

**Authors**: *Daiki Chijiwa, Shin'ya Yamaguchi, Yasutoshi Ida, Kenji Umakoshi, Tomohiro Inoue*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/23e582ad8087f2c03a5a31c125123f9a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/23e582ad8087f2c03a5a31c125123f9a-Abstract.html)

**Abstract**:

Pruning the weights of randomly initialized neural networks plays an important role in the context of lottery ticket hypothesis. Ramanujan et al. (2020) empirically showed that only pruning the weights can achieve remarkable performance instead of optimizing the weight values. However, to achieve the same level of performance as the weight optimization, the pruning approach requires more parameters in the networks before pruning and thus more memory space. To overcome this parameter inefficiency, we introduce a novel framework to prune randomly initialized neural networks with iteratively randomizing weight values (IteRand). Theoretically, we prove an approximation theorem in our framework, which indicates that the randomizing operations are provably effective to reduce the required number of the parameters. We also empirically demonstrate the parameter efficiency in multiple experiments on CIFAR-10 and ImageNet.

----

## [344] Probing Inter-modality: Visual Parsing with Self-Attention for Vision-and-Language Pre-training

**Authors**: *Hongwei Xue, Yupan Huang, Bei Liu, Houwen Peng, Jianlong Fu, Houqiang Li, Jiebo Luo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/23fa71cc32babb7b91130824466d25a5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/23fa71cc32babb7b91130824466d25a5-Abstract.html)

**Abstract**:

Vision-Language Pre-training (VLP) aims to learn multi-modal representations from image-text pairs and serves for downstream vision-language tasks in a fine-tuning fashion. The dominant VLP models adopt a CNN-Transformer architecture, which embeds images with a CNN, and then aligns images and text with a Transformer.  Visual relationship between visual contents plays an important role in image understanding and is the basic for inter-modal alignment learning. However, CNNs have limitations in visual relation learning due to local receptive field's weakness in modeling long-range dependencies. Thus the two objectives of learning visual relation and inter-modal alignment are encapsulated in the same Transformer network. Such design might restrict the inter-modal alignment learning in the Transformer by ignoring the specialized characteristic of each objective. To tackle this, we propose a fully Transformer visual embedding for VLP to better learn visual relation and further promote inter-modal alignment. Specifically, we propose a metric named Inter-Modality Flow (IMF) to measure the interaction between vision and language modalities (i.e., inter-modality). We also design a novel masking optimization mechanism named Masked Feature Regression (MFR) in Transformer to further promote the inter-modality learning. To the best of our knowledge, this is the first study to explore the benefit of Transformer for visual feature learning in VLP.  We verify our method on a wide range of vision-language tasks, including Visual Question Answering (VQA), Visual Entailment and Visual Reasoning. Our approach not only outperforms the state-of-the-art VLP performance, but also shows benefits on the IMF metric.

----

## [345] Stability and Generalization of Bilevel Programming in Hyperparameter Optimization

**Authors**: *Fan Bao, Guoqiang Wu, Chongxuan Li, Jun Zhu, Bo Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2406a0a94c80406914ff2f6c9fdd67d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2406a0a94c80406914ff2f6c9fdd67d5-Abstract.html)

**Abstract**:

The (gradient-based) bilevel programming framework is widely used in hyperparameter optimization and has achieved excellent performance empirically. Previous theoretical work mainly focuses on its optimization properties, while leaving the analysis on generalization largely open. This paper attempts to address the issue by presenting an expectation bound w.r.t. the validation set based on uniform stability. Our results can explain some mysterious behaviours of the bilevel programming in practice, for instance, overfitting to the validation set. We also present an expectation bound for the classical cross-validation algorithm. Our results suggest that gradient-based algorithms can be better than cross-validation under certain conditions in a theoretical perspective. Furthermore, we prove that regularization terms in both the outer and inner levels can relieve the overfitting problem in  gradient-based algorithms. In experiments on feature learning and data reweighting for noisy labels, we corroborate our theoretical findings.

----

## [346] Regime Switching Bandits

**Authors**: *Xiang Zhou, Yi Xiong, Ningyuan Chen, Xuefeng Gao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/240ac9371ec2671ae99847c3ae2e6384-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/240ac9371ec2671ae99847c3ae2e6384-Abstract.html)

**Abstract**:

We study a multi-armed bandit problem where the rewards exhibit regime switching. Specifically, the distributions of the random rewards generated from all arms are modulated by a common underlying state modeled as a finite-state Markov chain. The agent does not observe the underlying state and has to learn the transition matrix and the reward distributions. We propose a learning algorithm for this problem, building on spectral method-of-moments estimations for hidden Markov models, belief error control in partially observable Markov decision processes and upper-confidence-bound methods for online learning. We also establish an upper bound $O(T^{2/3}\sqrt{\log T})$ for the proposed learning algorithm where $T$ is the learning horizon. Finally, we conduct proof-of-concept experiments to illustrate the performance of the learning algorithm.

----

## [347] MixACM: Mixup-Based Robustness Transfer via Distillation of Activated Channel Maps

**Authors**: *Muhammad Awais, Fengwei Zhou, Chuanlong Xie, Jiawei Li, Sung-Ho Bae, Zhenguo Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/240c945bb72980130446fc2b40fbb8e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/240c945bb72980130446fc2b40fbb8e0-Abstract.html)

**Abstract**:

Deep neural networks are susceptible to adversarially crafted, small, and imperceptible changes in the natural inputs. The most effective defense mechanism against these examples is adversarial training which constructs adversarial examples during training by iterative maximization of loss. The model is then trained to minimize the loss on these constructed examples. This min-max optimization requires more data, larger capacity models, and additional computing resources. It also degrades the standard generalization performance of a model. Can we achieve robustness more efficiently? In this work, we explore this question from the perspective of knowledge transfer. First, we theoretically show the transferability of robustness from an adversarially trained teacher model to a student model with the help of mixup augmentation. Second, we propose a novel robustness transfer method called Mixup-Based Activated Channel Maps (MixACM) Transfer. MixACM transfers robustness from a robust teacher to a student by matching activated channel maps generated without expensive adversarial perturbations. Finally, extensive experiments on multiple datasets and different learning scenarios show our method can transfer robustness while also improving generalization on natural images.

----

## [348] Localization, Convexity, and Star Aggregation

**Authors**: *Suhas Vijaykumar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2417dc8af8570f274e6775d4d60496da-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2417dc8af8570f274e6775d4d60496da-Abstract.html)

**Abstract**:

Offset Rademacher complexities have been shown to provide tight upper bounds for the square loss in a broad class of problems including improper statistical learning and online learning. We show that the offset complexity can be generalized to any loss that satisfies a certain general convexity condition. Further, we show that this condition is closely related to both exponential concavity and self-concordance, unifying apparently disparate results. By a novel geometric argument, many of our bounds translate to improper learning in a non-convex class with Audibert's star algorithm. Thus, the offset complexity provides a versatile analytic tool that covers both convex empirical risk minimization and improper learning under entropy conditions. Applying the method, we recover the optimal rates for proper and improper learning with the $p$-loss for $1 < p < \infty$, and show that improper variants of empirical risk minimization can attain fast rates for logistic regression and other generalized linear models.

----

## [349] Aligning Silhouette Topology for Self-Adaptive 3D Human Pose Recovery

**Authors**: *Mugalodi Rakesh, Jogendra Nath Kundu, Varun Jampani, Venkatesh Babu R.*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/242c100dc94f871b6d7215b868a875f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/242c100dc94f871b6d7215b868a875f8-Abstract.html)

**Abstract**:

Articulation-centric 2D/3D pose supervision forms the core training objective in most existing 3D human pose estimation techniques. Except for synthetic source environments, acquiring such rich supervision for each real target domain at deployment is highly inconvenient. However, we realize that standard foreground silhouette estimation techniques (on static camera feeds) remain unaffected by domain-shifts. Motivated by this, we propose a novel target adaptation framework that relies only on silhouette supervision to adapt a source-trained model-based regressor. However, in the absence of any auxiliary cue (multi-view, depth, or 2D pose), an isolated silhouette loss fails to provide a reliable pose-specific gradient and requires to be employed in tandem with a topology-centric loss. To this end, we develop a series of convolution-friendly spatial transformations in order to disentangle a topological-skeleton representation from the raw silhouette. Such a design paves the way to devise a Chamfer-inspired spatial topological-alignment loss via distance field computation, while effectively avoiding any gradient hindering spatial-to-pointset mapping. Experimental results demonstrate our superiority against prior-arts in self-adapting a source trained model to diverse unlabeled target domains, such as a) in-the-wild datasets, b) low-resolution image domains, and c) adversarially perturbed image domains (via UAP).

----

## [350] Self-Adaptable Point Processes with Nonparametric Time Decays

**Authors**: *Zhimeng Pan, Zheng Wang, Jeff M. Phillips, Shandian Zhe*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/243facb29564e7b448834a7c9d901201-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/243facb29564e7b448834a7c9d901201-Abstract.html)

**Abstract**:

Many applications involve multi-type event data. Understanding the complex influences of the events on each other is critical to discover useful knowledge and to predict future events and their types. Existing methods either ignore or partially account for these influences. Recent works use recurrent neural networks to model the event rate. While being highly expressive, they couple all the temporal dependencies in a black-box and can hardly extract meaningful knowledge. More important, most methods assume an exponential time decay of the influence strength, which is over-simplified and can miss many important strength varying patterns.  To overcome these limitations, we propose SPRITE, a $\underline{S}$elf-adaptable  $\underline{P}$oint p$\underline{R}$ocess w$\underline{I}$th nonparametric $\underline{T}$ime d$\underline{E}$cays, which can decouple the influences between every pair of the events and capture various time decays of the influence strengths. Specifically, we use an embedding to represent each event type and model the event influence as an unknown function of the embeddings and time span. We derive a general construction that can cover all possible time decaying functions. By placing Gaussian process (GP) priors over the latent functions and using Gauss-Legendre quadrature to obtain the integral in the construction, we can flexibly estimate all kinds of time-decaying influences, without restricting to any specific form or imposing derivative constraints that bring learning difficulties.  We then use weight space augmentation of GPs to develop an efficient stochastic variational learning algorithm. We show the advantages of our approach in both the ablation study and real-world applications.

----

## [351] Offline Meta Reinforcement Learning - Identifiability Challenges and Effective Data Collection Strategies

**Authors**: *Ron Dorfman, Idan Shenfeld, Aviv Tamar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/248024541dbda1d3fd75fe49d1a4df4d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/248024541dbda1d3fd75fe49d1a4df4d-Abstract.html)

**Abstract**:

Consider the following instance of the Offline Meta Reinforcement Learning (OMRL) problem: given the complete training logs of $N$ conventional RL agents, trained on $N$ different tasks, design a meta-agent that can quickly maximize reward in a new, unseen task from the same task distribution. In particular, while each conventional RL agent explored and exploited its own different task, the meta-agent must identify regularities in the data that lead to effective exploration/exploitation in the unseen task. Here, we take a Bayesian RL (BRL) view, and seek to learn a Bayes-optimal policy from the offline data. Building on the recent VariBAD BRL approach, we develop an off-policy BRL method that learns to plan an  exploration strategy based on an adaptive neural belief estimate. However, learning to infer such a belief from offline data brings a new identifiability issue we term MDP ambiguity. We characterize the problem, and suggest resolutions via data collection and modification procedures.Finally, we evaluate our framework on a diverse set of domains, including difficult sparse reward tasks, and demonstrate learning of effective exploration behavior that is qualitatively different from the exploration used by any RL agent in the data. Our code is available online at \url{https://github.com/Rondorf/BOReL}.

----

## [352] RoMA: Robust Model Adaptation for Offline Model-based Optimization

**Authors**: *Sihyun Yu, Sungsoo Ahn, Le Song, Jinwoo Shin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/24b43fb034a10d78bec71274033b4096-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/24b43fb034a10d78bec71274033b4096-Abstract.html)

**Abstract**:

We consider the problem of searching an input maximizing a black-box objective function given a static dataset of input-output queries. A popular approach to solving this problem is maintaining a proxy model, e.g., a deep neural network (DNN), that approximates the true objective function. Here, the main challenge is how to avoid adversarially optimized inputs during the search, i.e., the inputs where the DNN highly overestimates the true objective function. To handle the issue, we propose a new framework, coined robust model adaptation (RoMA), based on gradient-based optimization of inputs over the DNN. Specifically, it consists of two steps: (a) a pre-training strategy to robustly train the proxy model and (b) a novel adaptation procedure of the proxy model to have robust estimates for a specific set of candidate solutions. At a high level, our scheme utilizes the local smoothness prior to overcome the brittleness of the DNN. Experiments under various tasks show the effectiveness of RoMA compared with previous methods, obtaining state-of-the-art results, e.g., RoMA outperforms all at 4 out of 6 tasks and achieves runner-up results at the remaining tasks.

----

## [353] Flexible Option Learning

**Authors**: *Martin Klissarov, Doina Precup*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/24cceab7ffc1118f5daaace13c670885-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/24cceab7ffc1118f5daaace13c670885-Abstract.html)

**Abstract**:

Temporal abstraction in reinforcement learning (RL), offers the promise of improving generalization and knowledge transfer in complex environments, by propagating information more efficiently over time. Although option learning was initially formulated in a way that allows updating many options simultaneously, using off-policy, intra-option learning (Sutton, Precup & Singh, 1999) , many of the recent hierarchical reinforcement learning approaches only update a single option at a time: the option currently executing. We revisit and extend intra-option learning in the context of deep reinforcement learning, in order to enable updating all options consistent with current primitive action choices, without introducing any additional estimates. Our method can therefore be naturally adopted in most hierarchical RL frameworks. When we combine our approach with the option-critic algorithm for option discovery, we obtain significant improvements in performance and data-efficiency across a wide variety of domains.

----

## [354] Faster Directional Convergence of Linear Neural Networks under Spherically Symmetric Data

**Authors**: *Dachao Lin, Ruoyu Sun, Zhihua Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/24ec8468b67314c2013d215b77034476-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/24ec8468b67314c2013d215b77034476-Abstract.html)

**Abstract**:

In this paper, we study gradient methods for training deep linear neural networks with binary cross-entropy loss. In particular, we show global directional convergence guarantees from a polynomial rate to a linear rate for (deep) linear networks with spherically symmetric data distribution, which can be viewed as a specific zero-margin dataset. Our results do not require the assumptions in other works such as small initial loss, presumed convergence of weight direction, or overparameterization. We also characterize our findings in experiments.

----

## [355] Online Facility Location with Multiple Advice

**Authors**: *Matteo Almanza, Flavio Chierichetti, Silvio Lattanzi, Alessandro Panconesi, Giuseppe Re*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/250473494b245120a7eaf8b2e6b1f17c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/250473494b245120a7eaf8b2e6b1f17c-Abstract.html)

**Abstract**:

Clustering is a central topic in unsupervised learning and its online formulation has received a lot of attention in recent years. In this paper, we study the classic facility location problem in the presence of multiple machine-learned advice. We design an algorithm with provable performance guarantees such that, if the advice is good, it outperforms the best-known online algorithms for the problem, and if it is bad it still matches their performance.We complement our theoretical analysis with an in-depth study of the performance of our algorithm, showing its effectiveness on synthetic and real-world data sets.

----

## [356] Credit Assignment in Neural Networks through Deep Feedback Control

**Authors**: *Alexander Meulemans, Matilde Tristany Farinha, Javier García Ordóñez, Pau Vilimelis Aceituno, João Sacramento, Benjamin F. Grewe*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/25048eb6a33209cb5a815bff0cf6887c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/25048eb6a33209cb5a815bff0cf6887c-Abstract.html)

**Abstract**:

The success of deep learning sparked interest in whether the brain learns by using similar techniques for assigning credit to each synaptic weight for its contribution to the network output. However, the majority of current attempts at biologically-plausible learning methods are either non-local in time, require highly specific connectivity motifs, or have no clear link to any known mathematical optimization method. Here, we introduce Deep Feedback Control (DFC), a new learning method that uses a feedback controller to drive a deep neural network to match a desired output target and whose control signal can be used for credit assignment. The resulting learning rule is fully local in space and time and approximates Gauss-Newton optimization for a wide range of feedback connectivity patterns. To further underline its biological plausibility, we relate DFC to a multi-compartment model of cortical pyramidal neurons with a local voltage-dependent synaptic plasticity rule, consistent with recent theories of dendritic processing. By combining dynamical system theory with mathematical optimization theory, we provide a strong theoretical foundation for DFC that we corroborate with detailed results on toy experiments and standard computer-vision benchmarks.

----

## [357] Robust Online Correlation Clustering

**Authors**: *Silvio Lattanzi, Benjamin Moseley, Sergei Vassilvitskii, Yuyan Wang, Rudy Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/250dd56814ad7c50971ee4020519c6f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/250dd56814ad7c50971ee4020519c6f5-Abstract.html)

**Abstract**:

In correlation clustering we are given a set of points along with recommendations whether each pair of points should be placed in the same cluster or into separate clusters. The goal cluster the points to minimize disagreements from the recommendations. We study the correlation clustering problem in the online setting., where points arrive one at a time, and upon arrival the algorithm must make an irrevocable cluster assignment decision. While the online version is natural, there is a simple lower bound that rules out any algorithm with a non-trivial competitive ratio. In this work we go beyond worst case analysis, and show that the celebrated Pivot algorithm performs well when given access to a small number of random samples from the input. Moreover, we prove that Pivot is robust to additional adversarial perturbations of the sample set in this setting. We conclude with an empirical analysis validating our theoretical findings.

----

## [358] Neural Additive Models: Interpretable Machine Learning with Neural Nets

**Authors**: *Rishabh Agarwal, Levi Melnick, Nicholas Frosst, Xuezhou Zhang, Benjamin J. Lengerich, Rich Caruana, Geoffrey E. Hinton*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/251bd0442dfcc53b5a761e050f8022b8-Abstract.html)

**Abstract**:

Deep neural networks (DNNs) are powerful black-box predictors that have achieved impressive performance on a wide variety of tasks. However, their accuracy comes at the cost of intelligibility: it is usually unclear how they make their decisions. This hinders their applicability to high stakes decision-making domains such as healthcare. We propose Neural Additive Models (NAMs) which combine some of the expressivity of DNNs with the inherent intelligibility of generalized additive models. NAMs learn a linear combination of neural networks that each attend to a single input feature. These networks are trained jointly and can learn arbitrarily complex relationships between their input feature and the output. Our experiments on regression and classification datasets show that NAMs are more accurate than widely used intelligible models such as logistic regression and shallow decision trees. They perform similarly to existing state-of-the-art generalized additive models in accuracy, but are more flexible because they are based on neural nets instead of boosted trees. To demonstrate this, we show how NAMs can be used for multitask learning on synthetic data and on the COMPAS recidivism data due to their composability, and demonstrate that the differentiability of NAMs allows them to train more complex interpretable models for COVID-19.

----

## [359] Representation Learning for Event-based Visuomotor Policies

**Authors**: *Sai Vemprala, Sami Mian, Ashish Kapoor*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/251c5ffd6b62cc21c446c963c76cf214-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/251c5ffd6b62cc21c446c963c76cf214-Abstract.html)

**Abstract**:

Event-based cameras are dynamic vision sensors that provide asynchronous measurements of changes in per-pixel brightness at a microsecond level. This makes them significantly faster than conventional frame-based cameras, and an appealing choice for high-speed robot navigation. While an interesting sensor modality, this asynchronously streamed event data poses a challenge for machine learning based computer vision techniques that are more suited for synchronous, frame-based data. In this paper, we present an event variational autoencoder through which compact representations can be learnt directly from asynchronous spatiotemporal event data. Furthermore, we show that such pretrained representations can be used for event-based reinforcement learning instead of end-to-end reward driven perception. We validate this framework of learning event-based visuomotor policies by applying it to an obstacle avoidance scenario in simulation. Compared to techniques that treat event data as images, we show that representations learnt from event streams result in faster policy training, adapt to different control capacities, and demonstrate a higher degree of robustness to environmental changes and sensor noise.

----

## [360] Kernel Functional Optimisation

**Authors**: *Arun Kumar Anjanapura Venkatesh, Alistair Shilton, Santu Rana, Sunil Gupta, Svetha Venkatesh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/251e16a2aac0ca4847adf561483381bf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/251e16a2aac0ca4847adf561483381bf-Abstract.html)

**Abstract**:

Traditional methods for kernel selection rely on parametric kernel functions or a combination thereof and although the kernel hyperparameters are tuned, these methods often provide sub-optimal results due to the limitations induced by the parametric forms. In this paper, we propose a novel formulation for kernel selection using efficient Bayesian optimisation to find the best fitting non-parametric kernel. The kernel is expressed using a linear combination of functions sampled from a prior Gaussian Process (GP) defined by a hyperkernel. We also provide a mechanism to ensure the positive definiteness of the Gram matrix constructed using the resultant kernels. Our experimental results on GP regression and Support Vector Machine (SVM) classification tasks involving both synthetic functions and several real-world datasets show the superiority of our approach over the state-of-the-art.

----

## [361] Generalized Shape Metrics on Neural Representations

**Authors**: *Alex H. Williams, Erin Kunz, Simon Kornblith, Scott W. Linderman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/252a3dbaeb32e7690242ad3b556e626b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/252a3dbaeb32e7690242ad3b556e626b-Abstract.html)

**Abstract**:

Understanding the operation of biological and artificial networks remains a difficult and important challenge. To identify general principles, researchers are increasingly interested in surveying large collections of networks that are trained on, or biologically adapted to, similar tasks. A standardized set of analysis tools is now needed to identify how network-level covariates---such as architecture, anatomical brain region, and model organism---impact neural representations (hidden layer activations). Here, we provide a rigorous foundation for these analyses by defining a broad family of metric spaces that quantify representational dissimilarity. Using this framework, we modify existing representational similarity measures based on canonical correlation analysis and centered kernel alignment to satisfy the triangle inequality, formulate a novel metric that respects the inductive biases in convolutional layers, and identify approximate Euclidean embeddings that enable network representations to be incorporated into essentially any off-the-shelf machine learning method. We demonstrate these methods on large-scale datasets from biology (Allen Institute Brain Observatory) and deep learning (NAS-Bench-101). In doing so, we identify relationships between neural representations that are interpretable in terms of anatomical features and model performance.

----

## [362] Diverse Message Passing for Attribute with Heterophily

**Authors**: *Liang Yang, Mengzhe Li, Liyang Liu, Bingxin Niu, Chuan Wang, Xiaochun Cao, Yuanfang Guo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/253614bbac999b38b5b60cae531c4969-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/253614bbac999b38b5b60cae531c4969-Abstract.html)

**Abstract**:

Most of the existing GNNs can be modeled via the Uniform Message Passing framework. This framework considers all the attributes of each node in its entirety, shares the uniform propagation weights along each edge,  and focuses on the uniform weight learning. The design of this framework possesses two prerequisites, the simplification of homophily and heterophily to the node-level property and the ignorance of attribute differences. Unfortunately, different attributes possess diverse characteristics. In this paper, the network homophily rate defined with  respect to the node labels is extended to attribute homophily rate by taking the attributes as weak labels. Based on this attribute homophily rate, we propose a Diverse Message Passing (DMP) framework, which specifies every attribute propagation weight on each edge. Besides, we propose two specific strategies to significantly reduce the computational complexity of DMP to prevent the overfitting issue.  By investigating the spectral characteristics, existing spectral GNNs are actually equivalent to a degenerated version of DMP.  From the perspective of numerical optimization, we provide a theoretical analysis to demonstrate DMP's powerful representation ability and the ability of alleviating the over-smoothing issue.  Evaluations on various  real networks demonstrate the superiority of our DMP on  handling the networks with heterophily  and alleviating the over-smoothing issue, compared to the existing state-of-the-arts.

----

## [363] Towards Robust Bisimulation Metric Learning

**Authors**: *Mete Kemertas, Tristan Aumentado-Armstrong*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/256bf8e6923a52fda8ddf7dc050a1148-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/256bf8e6923a52fda8ddf7dc050a1148-Abstract.html)

**Abstract**:

Learned representations in deep reinforcement learning (DRL) have to extract task-relevant information from complex observations, balancing between robustness to distraction and informativeness to the policy. Such stable and rich representations, often learned via modern function approximation techniques, can enable practical application of the policy improvement theorem, even in high-dimensional continuous state-action spaces. Bisimulation metrics offer one solution to this representation learning problem, by collapsing functionally similar states together in representation space, which promotes invariance to noise and distractors. In this work, we generalize value function approximation bounds for on-policy bisimulation metrics to non-optimal policies and approximate environment dynamics. Our theoretical results help us identify embedding pathologies that may occur in practical use. In particular, we find that these issues stem from an underconstrained dynamics model and an unstable dependence of the embedding norm on the reward signal in environments with sparse rewards. Further, we propose a set of practical remedies: (i) a norm constraint on the representation space, and (ii) an extension of prior approaches with intrinsic rewards and latent space regularization. Finally, we provide evidence that the resulting method is not only more robust to sparse reward functions, but also able to solve challenging continuous control tasks with observational distractions, where prior methods fail.

----

## [364] Beyond BatchNorm: Towards a Unified Understanding of Normalization in Deep Learning

**Authors**: *Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2578eb9cdf020730f77793e8b58e165a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2578eb9cdf020730f77793e8b58e165a-Abstract.html)

**Abstract**:

Inspired by BatchNorm, there has been an explosion of normalization layers in deep learning. Recent works have identified a multitude of beneficial properties in BatchNorm to explain its success. However, given the pursuit of alternative normalization layers, these properties need to be generalized so that any given layer's success/failure can be accurately predicted. In this work, we take a first step towards this goal by extending known properties of BatchNorm in randomly initialized deep neural networks (DNNs) to several recently proposed normalization layers. Our primary findings follow: (i) similar to BatchNorm, activations-based normalization layers can prevent exponential growth of activations in ResNets, but parametric techniques require explicit remedies; (ii) use of GroupNorm can ensure an informative forward propagation, with different samples being assigned dissimilar activations, but increasing group size results in increasingly indistinguishable activations for different samples, explaining slow convergence speed in models with LayerNorm; and (iii) small group sizes result in large gradient norm in earlier layers, hence explaining training instability issues in Instance Normalization and illustrating a speed-stability tradeoff in GroupNorm. Overall, our analysis reveals a unified set of mechanisms that underpin the success of normalization methods in deep learning, providing us with a compass to systematically explore the vast design space of DNN normalization layers.

----

## [365] Representation Learning Beyond Linear Prediction Functions

**Authors**: *Ziping Xu, Ambuj Tewari*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/258be18e31c8188555c2ff05b4d542c3-Abstract.html)

**Abstract**:

Recent papers on the theory of representation learning has shown the importance of a quantity called diversity when generalizing from a set of source tasks to a target task. Most of these papers assume that the function mapping shared representations to predictions is linear, for both source and target tasks. In practice, researchers in deep learning use different numbers of extra layers following the pretrained model based on the difficulty of the new task. This motivates us to ask whether diversity can be achieved when source tasks and the target task use different prediction function spaces beyond linear functions. We show that diversity holds even if the target task uses a neural network with multiple layers, as long as source tasks use linear functions. If source tasks use nonlinear prediction functions, we provide a negative result by showing that depth-1 neural networks with ReLu activation function need exponentially many source tasks to achieve diversity. For a general function class, we find that eluder dimension gives a lower bound on the number of tasks required for diversity. Our theoretical results imply that simpler tasks generalize better. Though our theoretical results are shown for the global minimizer of empirical risks, their qualitative predictions still hold true for gradient-based optimization algorithms as verified by our simulations on deep neural networks.

----

## [366] Volume Rendering of Neural Implicit Surfaces

**Authors**: *Lior Yariv, Jiatao Gu, Yoni Kasten, Yaron Lipman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/25e2a30f44898b9f3e978b1786dcd85c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/25e2a30f44898b9f3e978b1786dcd85c-Abstract.html)

**Abstract**:

Neural volume rendering became increasingly popular recently due to its success in synthesizing novel views of a scene from a sparse set of input images. So far, the geometry learned by neural volume rendering techniques was modeled using a generic density function. Furthermore, the geometry itself was extracted using an arbitrary level set of the density function leading to a noisy, often low fidelity reconstruction.The goal of this paper is to improve geometry representation and reconstruction in neural volume rendering. We achieve that by modeling the volume density as a function of the geometry. This is in contrast to previous work modeling the geometry as a function of the volume density. In more detail, we define the volume density function as Laplace's cumulative distribution function (CDF) applied to a signed distance function (SDF) representation. This simple density representation has three benefits: (i) it provides a useful inductive bias to the geometry learned in the neural volume rendering process; (ii) it facilitates a bound on the opacity approximation error, leading to an accurate sampling of the viewing ray. Accurate sampling is important to provide a precise coupling of geometry and radiance; and (iii) it allows efficient unsupervised disentanglement of shape and appearance in volume rendering.Applying this new density representation to challenging scene multiview datasets produced high quality geometry reconstructions, outperforming relevant baselines. Furthermore, switching shape and appearance between scenes is possible due to the disentanglement of the two.

----

## [367] MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers

**Authors**: *Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, Zaïd Harchaoui*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/260c2432a0eecc28ce03c10dadc078a4-Abstract.html)

**Abstract**:

As major progress is made in open-ended text generation, measuring how close machine-generated text is to human language remains a critical open problem. We introduce Mauve, a comparison measure for open-ended text generation, which directly compares the learnt distribution from a text generation model to the distribution of human-written text using divergence frontiers. Mauve scales up to modern text generation models by computing information divergences in a quantized embedding space. Through an extensive empirical study on three open-ended generation tasks, we find that Mauve identifies known properties of generated text, scales naturally with model size, and correlates with human judgments, with fewer restrictions than existing distributional evaluation metrics.

----

## [368] Accurately Solving Rod Dynamics with Graph Learning

**Authors**: *Han Shao, Tassilo Kugelstadt, Torsten Hädrich, Wojtek Palubicki, Jan Bender, Sören Pirk, Dominik L. Michels*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26337353b7962f533d78c762373b3318-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26337353b7962f533d78c762373b3318-Abstract.html)

**Abstract**:

Iterative solvers are widely used to accurately simulate physical systems. These solvers require initial guesses to generate a sequence of improving approximate solutions. In this contribution, we introduce a novel method to accelerate iterative solvers for rod dynamics with graph networks (GNs) by predicting the initial guesses to reduce the number of iterations. Unlike existing methods that aim to learn physical systems in an end-to-end manner, our approach guarantees long-term stability and therefore leads to more accurate solutions. Furthermore, our method improves the run time performance of traditional iterative solvers for rod dynamics. To explore our method we make use of position-based dynamics (PBD) as a common solver for physical systems and evaluate it by simulating the dynamics of elastic rods. Our approach is able to generalize across different initial conditions, discretizations, and realistic material properties. We demonstrate that it also performs well when taking discontinuous effects into account such as collisions between individual rods. Finally, to illustrate the scalability of our approach, we simulate complex 3D tree models composed of over a thousand individual branch segments swaying in wind fields.

----

## [369] Limiting fluctuation and trajectorial stability of multilayer neural networks with mean field training

**Authors**: *Huy Tuan Pham, Phan-Minh Nguyen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2639ba2137371773aa1e64e7735cdb30-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2639ba2137371773aa1e64e7735cdb30-Abstract.html)

**Abstract**:

The mean field theory of multilayer neural networks centers around a particular infinite-width scaling, in which the learning dynamics is shown to be closely tracked by the mean field limit. A random fluctuation around this infinite-width limit is expected from a large-width expansion to the next order. This fluctuation has been studied only in the case of shallow networks, where previous works employ heavily technical notions or additional formulation ideas amenable only to that case. Treatment of the multilayer case has been missing, with the chief difficulty in finding a formulation that must capture the stochastic dependency across not only time but also depth.In this work, we initiate the study of the fluctuation in the case of multilayer networks, at any network depth. Leveraging on the neuronal embedding framework recently introduced by Nguyen and Pham, we systematically derive a system of dynamical equations, called the second-order mean field limit, that captures the limiting fluctuation distribution. We demonstrate through the framework the complex interaction among neurons in this second-order mean field limit, the stochasticity with cross-layer dependency and the nonlinear time evolution inherent in the limiting fluctuation. A limit theorem is proven to relate quantitatively this limit to the fluctuation realized by large-width networks.We apply the result to show a stability property of gradient descent mean field training: in the large-width regime, along the training trajectory, it progressively biases towards a solution with "minimal fluctuation" (in fact, vanishing fluctuation) in the learned output function, even after the network has been initialized at or has converged (sufficiently fast) to a global optimum. This extends a similar phenomenon previously shown only for shallow networks with a squared loss in the empirical risk minimization setting, to multilayer networks with a loss function that is not necessarily convex in a more general setting.

----

## [370] Medical Dead-ends and Learning to Identify High-Risk States and Treatments

**Authors**: *Mehdi Fatemi, Taylor W. Killian, Jayakumar Subramanian, Marzyeh Ghassemi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26405399c51ad7b13b504e74eb7c696c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26405399c51ad7b13b504e74eb7c696c-Abstract.html)

**Abstract**:

Machine learning has successfully framed many sequential decision making problems as either supervised prediction, or optimal decision-making policy identification via reinforcement learning. In data-constrained offline settings, both approaches may fail as they assume fully optimal behavior or rely on exploring alternatives that may not exist. We introduce an inherently different approach that identifies "dead-ends" of a state space. We focus on patient condition in the intensive care unit, where a "medical dead-end" indicates that a patient will expire, regardless of all potential future treatment sequences. We postulate "treatment security" as avoiding treatments with probability proportional to their chance of leading to dead-ends, present a formal proof, and frame discovery as an RL problem. We then train three independent deep neural models for automated state construction, dead-end discovery and confirmation. Our empirical results discover that dead-ends exist in real clinical data among septic patients, and further reveal gaps between secure treatments and those administered.

----

## [371] Overcoming the Convex Barrier for Simplex Inputs

**Authors**: *Harkirat Singh Behl, M. Pawan Kumar, Philip H. S. Torr, Krishnamurthy Dvijotham*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26657d5ff9020d2abefe558796b99584-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26657d5ff9020d2abefe558796b99584-Abstract.html)

**Abstract**:

Recent progress in neural network verification has challenged the notion of a convex barrier, that is, an inherent weakness in the convex relaxation of the output of a neural network. Specifically, there now exists a tight relaxation for verifying the robustness of a neural network to $\ell_\infty$ input perturbations, as well as efficient primal and dual solvers for the relaxation. Buoyed by this success, we consider the problem of developing similar techniques for verifying robustness to input perturbations within the probability simplex. We prove a somewhat surprising result that, in this case, not only can one design a tight relaxation that overcomes the convex barrier, but the size of the relaxation remains linear in the number of neurons, thereby leading to simpler and more efficient algorithms. We establish the scalability of our overall approach via the specification of $\ell_1$ robustness for CIFAR-10 and MNIST classification, where our approach improves the state of the art verified accuracy by up to $14.4\%$. Furthermore, we establish its accuracy on a novel and highly challenging task of verifying the robustness of a multi-modal (text and image) classifier to arbitrary changes in its textual input.

----

## [372] High-probability Bounds for Non-Convex Stochastic Optimization with Heavy Tails

**Authors**: *Ashok Cutkosky, Harsh Mehta*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26901debb30ea03f0aa833c9de6b81e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26901debb30ea03f0aa833c9de6b81e9-Abstract.html)

**Abstract**:

We consider non-convex stochastic optimization using first-order algorithms for which the gradient estimates may have heavy tails. We show that a combination of gradient clipping, momentum, and normalized gradient descent yields convergence to critical points in high-probability with best-known rates for smooth losses when the gradients only have bounded $\mathfrak{p}$th moments for some $\mathfrak{p}\in(1,2]$. We then consider the case of second-order smooth losses, which to our knowledge have not been studied in this setting, and again obtain high-probability bounds for any $\mathfrak{p}$. Moreover, our results hold for arbitrary smooth norms, in contrast to the typical SGD analysis which requires a Hilbert space norm. Further, we show that after a suitable "burn-in" period, the objective value will monotonically decrease for every iteration until a critical point is identified, which provides intuition behind the popular practice of learning rate "warm-up'' and also yields a last-iterate guarantee.

----

## [373] Batch Normalization Orthogonalizes Representations in Deep Random Networks

**Authors**: *Hadi Daneshmand, Amir Joudaki, Francis R. Bach*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html)

**Abstract**:

This paper underlines an elegant property of batch-normalization (BN): Successive batch normalizations with random linear updates make samples increasingly orthogonal. We establish a non-asymptotic characterization of the interplay between depth, width, and the orthogonality of deep representations. More precisely, we prove, under a mild assumption, the deviation of the representations from orthogonality rapidly decays with depth up to a term inversely proportional to the network width. This result has two main theoretical and practical implications: 1) Theoretically, as the depth grows, the distribution of the outputs contracts to a Wasserstein-2 ball around an isotropic normal distribution. Furthermore, the radius of this Wasserstein ball shrinks with the width of the network. 2) Practically, the orthogonality of the representations directly influences the performance of stochastic gradient descent (SGD). When representations are initially aligned, we observe SGD wastes many iterations to disentangle representations before the classification. Nevertheless, we experimentally show that starting optimization from orthogonal representations is sufficient to accelerate SGD, with no need for BN.

----

## [374] Support vector machines and linear regression coincide with very high-dimensional features

**Authors**: *Navid Ardeshir, Clayton Sanford, Daniel J. Hsu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26d4b4313a7e5828856bc0791fca39a2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26d4b4313a7e5828856bc0791fca39a2-Abstract.html)

**Abstract**:

The support vector machine (SVM) and minimum Euclidean norm least squares regression are two fundamentally different approaches to fitting linear models, but they have recently been connected in models for very high-dimensional data through a phenomenon of support vector proliferation, where every training example used to fit an SVM becomes a support vector. In this paper, we explore the generality of this phenomenon and make the following contributions. First, we prove a super-linear lower bound on the dimension (in terms of sample size) required for support vector proliferation in independent feature models, matching the upper bounds from previous works. We further identify a sharp phase transition in Gaussian feature models, bound the width of this transition, and give experimental support for its universality. Finally, we hypothesize that this phase transition occurs only in much higher-dimensional settings in the $\ell_1$ variant of the SVM, and we present a new geometric characterization of the problem that may elucidate this phenomenon for the general $\ell_p$ case.

----

## [375] Coupled Segmentation and Edge Learning via Dynamic Graph Propagation

**Authors**: *Zhiding Yu, Rui Huang, Wonmin Byeon, Sifei Liu, Guilin Liu, Thomas M. Breuel, Anima Anandkumar, Jan Kautz*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/26ddd45b02859e836d13d4b9fde34281-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/26ddd45b02859e836d13d4b9fde34281-Abstract.html)

**Abstract**:

Image segmentation and edge detection are both central problems in perceptual grouping. It is therefore interesting to study how these two tasks can be coupled to benefit each other. Indeed, segmentation can be easily transformed into contour edges to guide edge learning. However, the converse is nontrivial since general edges may not always form closed contours. In this paper, we propose a principled end-to-end framework for coupled edge and segmentation learning, where edges are leveraged as pairwise similarity cues to guide segmentation. At the core of our framework is a recurrent module termed as dynamic graph propagation (DGP) layer that performs message passing on dynamically constructed graphs. The layer uses learned gating to dynamically select neighbors for message passing using max-pooling. The output from message passing is further gated with an edge signal to refine segmentation. Experiments demonstrate that the proposed framework is able to let both tasks mutually improve each other. On Cityscapes validation, our best model achieves 83.7% mIoU in semantic segmentation and 78.7% maximum F-score in semantic edge detection. Our method also leads to improved zero-shot robustness on Cityscapes with natural corruptions (Cityscapes-C).

----

## [376] Offline RL Without Off-Policy Evaluation

**Authors**: *David Brandfonbrener, Will Whitney, Rajesh Ranganath, Joan Bruna*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/274a10ffa06e434f2a94df765cac6bf4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/274a10ffa06e434f2a94df765cac6bf4-Abstract.html)

**Abstract**:

Most prior approaches to offline reinforcement learning (RL) have taken an iterative actor-critic approach involving off-policy evaluation. In this paper we show that simply doing one step of constrained/regularized policy improvement using an on-policy Q estimate of the behavior policy performs surprisingly well. This one-step algorithm beats the previously reported results of iterative algorithms on a large portion of the D4RL benchmark. The one-step baseline achieves this strong performance while being notably simpler and more robust to hyperparameters than previously proposed iterative algorithms. We argue that the relatively poor performance of iterative approaches is a result of the high variance inherent in doing off-policy evaluation and magnified by the repeated optimization of policies against those estimates. In addition, we hypothesize that the strong performance of the one-step algorithm is due to a combination of favorable structure in the environment and behavior policy.

----

## [377] Continuous vs Discrete Optimization of Deep Neural Networks

**Authors**: *Omer Elkabetz, Nadav Cohen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/274ad4786c3abca69fa097b85867d9a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/274ad4786c3abca69fa097b85867d9a4-Abstract.html)

**Abstract**:

Existing analyses of optimization in deep learning are either continuous, focusing on (variants of) gradient flow, or discrete, directly treating (variants of) gradient descent.  Gradient flow is amenable to theoretical analysis, but is stylized and disregards computational efficiency.  The extent to which it represents gradient descent is an open question in the theory of deep learning.  The current paper studies this question.  Viewing gradient descent as an approximate numerical solution to the initial value problem of gradient flow, we find that the degree of approximation depends on the curvature around the gradient flow trajectory.  We then show that over deep neural networks with homogeneous activations, gradient flow trajectories enjoy favorable curvature, suggesting they are well approximated by gradient descent.  This finding allows us to translate an analysis of gradient flow over deep linear neural networks into a guarantee that gradient descent efficiently converges to global minimum almost surely under random initialization.  Experiments suggest that over simple deep neural networks, gradient descent with conventional step size is indeed close to gradient flow.  We hypothesize that the theory of gradient flows will unravel mysteries behind deep learning.

----

## [378] CrypTen: Secure Multi-Party Computation Meets Machine Learning

**Authors**: *Brian Knott, Shobha Venkataraman, Awni Y. Hannun, Shubho Sengupta, Mark Ibrahim, Laurens van der Maaten*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2754518221cfbc8d25c13a06a4cb8421-Abstract.html)

**Abstract**:

Secure multi-party computation (MPC) allows parties to perform computations on data while keeping that data private. This capability has great potential for machine-learning applications: it facilitates training of machine-learning models on private data sets owned by different parties, evaluation of one party's private model using another party's private data, etc. Although a range of studies implement machine-learning models via secure MPC, such implementations are not yet mainstream. Adoption of secure MPC is hampered by the absence of flexible software frameworks that `"speak the language" of machine-learning researchers and engineers. To foster adoption of secure MPC in machine learning, we present CrypTen: a software framework that exposes popular secure MPC primitives via abstractions that are common in modern machine-learning frameworks, such as tensor computations, automatic differentiation, and modular neural networks. This paper describes the design of CrypTen and measure its performance on state-of-the-art models for text classification, speech recognition, and image classification. Our benchmarks show that CrypTen's GPU support and high-performance communication between (an arbitrary number of) parties allows it to perform efficient private evaluation of modern machine-learning models under a semi-honest threat model. For example, two parties using CrypTen can securely predict phonemes in speech recordings using Wav2Letter faster than real-time. We hope that CrypTen will spur adoption of secure MPC in the machine-learning community.

----

## [379] Can contrastive learning avoid shortcut solutions?

**Authors**: *Joshua Robinson, Li Sun, Ke Yu, Kayhan Batmanghelich, Stefanie Jegelka, Suvrit Sra*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/27934a1f19d678a1377c257b9a780e80-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/27934a1f19d678a1377c257b9a780e80-Abstract.html)

**Abstract**:

The generalization of representations learned via contrastive learning depends crucially on what features of the data are extracted. However, we observe that the contrastive loss does not always sufficiently guide which features are extracted, a behavior that can negatively impact the performance on downstream tasks via â€œshortcuts", i.e., by inadvertently suppressing important predictive features.  We find that feature extraction is influenced by  the difficulty of the so-called instance discrimination task (i.e., the task of discriminating pairs of similar points from pairs of dissimilar ones). Although harder pairs improve the representation of some features, the improvement comes at the cost of suppressing previously well represented features. In response, we propose implicit feature modification (IFM), a method for altering positive and negative samples in order to guide contrastive models towards capturing a wider variety of predictive features. Empirically, we observe that IFM reduces feature suppression, and as a result improves performance on vision and medical imaging tasks.

----

## [380] See More for Scene: Pairwise Consistency Learning for Scene Classification

**Authors**: *Gongwei Chen, Xinhang Song, Bohan Wang, Shuqiang Jiang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/27d52bcb3580724eb4cbe9f2718a9365-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/27d52bcb3580724eb4cbe9f2718a9365-Abstract.html)

**Abstract**:

Scene classification is a valuable classification subtask and has its own characteristics which still needs more in-depth studies. Basically, scene characteristics are distributed over the whole image, which cause the need of “seeing” comprehensive and informative regions. Previous works mainly focus on region discovery and aggregation, while rarely involves the inherent properties of CNN along with its potential ability to satisfy the requirements of scene classification. In this paper, we propose to understand scene images and the scene classification CNN models in terms of the focus area. From this new perspective, we find that large focus area is preferred in scene classification CNN models as a consequence of learning scene characteristics. Meanwhile, the analysis about existing training schemes helps us to understand the effects of focus area, and also raises the question about optimal training method for scene classification. Pursuing the better usage of scene characteristics, we propose a new learning scheme with a tailored loss in the goal of activating larger focus area on scene images. Since the supervision of the target regions to be enlarged is usually lacked, our alternative learning scheme is to erase already activated area, and allow the CNN models to activate more area during training. The proposed scheme is implemented by keeping the pairwise consistency between the output of  the erased image and its original one. In particular, a tailored loss is proposed to keep such pairwise consistency by leveraging category-relevance information. Experiments on Places365 show the significant improvements of our method with various CNNs. Our method shows an inferior result on the object-centric dataset, ImageNet, which experimentally indicates that it captures the unique characteristics of scenes.

----

## [381] Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss

**Authors**: *Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/27debb435021eb68b3965290b5e24c49-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/27debb435021eb68b3965290b5e24c49-Abstract.html)

**Abstract**:

Recent works in self-supervised learning have advanced the state-of-the-art by relying on the contrastive learning paradigm, which learns representations by pushing positive pairs, or similar examples from the same class, closer together while keeping negative pairs far apart. Despite the empirical successes, theoretical foundations are limited -- prior analyses assume conditional independence of the positive pairs given the same class label, but recent empirical applications use heavily correlated positive pairs (i.e., data augmentations of the same image). Our work analyzes contrastive learning without assuming conditional independence of positive pairs using a novel concept of the augmentation graph on data.  Edges in this graph connect augmentations of the same data, and ground-truth classes naturally form connected sub-graphs. We propose a loss that performs spectral decomposition on the population augmentation graph and can be succinctly written as a contrastive learning objective on neural net representations. Minimizing this objective leads to features with provable accuracy guarantees under linear probe evaluation. By standard generalization bounds, these accuracy guarantees also hold when minimizing the training contrastive loss. In all, this work provides the first provable analysis for contrastive learning where the guarantees can apply to realistic empirical settings.

----

## [382] Greedy Approximation Algorithms for Active Sequential Hypothesis Testing

**Authors**: *Kyra Gan, Su Jia, Andrew A. Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/27e9661e033a73a6ad8cefcde965c54d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/27e9661e033a73a6ad8cefcde965c54d-Abstract.html)

**Abstract**:

In the problem of \emph{active sequential hypothesis testing} (ASHT), a learner seeks to identify the \emph{true} hypothesis from among a known set of hypotheses. The learner is given a set of actions and knows the random distribution of the outcome of any action under any true hypothesis. Given a target error $\delta>0$, the goal is to sequentially select the fewest number of actions so as to identify the true hypothesis with probability at least $1 - \delta$. Motivated by applications in which the number of hypotheses or actions is massive (e.g., genomics-based cancer detection), we propose efficient (greedy, in fact) algorithms and provide the first approximation guarantees for ASHT, under two types of adaptivity. Both of our guarantees are independent of the number of actions and logarithmic in the number of hypotheses. We numerically evaluate the performance of our algorithms using both synthetic and real-world DNA mutation data, demonstrating that our algorithms outperform previously proposed heuristic policies by large margins.

----

## [383] When False Positive is Intolerant: End-to-End Optimization with Low FPR for Multipartite Ranking

**Authors**: *Peisong Wen, Qianqian Xu, Zhiyong Yang, Yuan He, Qingming Huang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/28267ab848bcf807b2ed53c3a8f8fc8a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/28267ab848bcf807b2ed53c3a8f8fc8a-Abstract.html)

**Abstract**:

Multipartite ranking is a basic task in machine learning, where the Area Under the receiver operating characteristics Curve (AUC) is generally applied as the evaluation metric. Despite that AUC reflects the overall performance of the model, it is inconsistent with the expected performance in some application scenarios, where only a low False Positive Rate (FPR) is meaningful. To leverage high performance under low FPRs, we consider an alternative metric for multipartite ranking evaluating the True Positive Rate (TPR) at a given FPR, denoted as TPR@FPR. Unfortunately, the key challenge of direct  TPR@FPR optimization is two-fold: \textbf{a)} the original objective function is not differentiable, making gradient backpropagation impossible; \textbf{b)} the loss function could not be written as a sum of independent instance-wise terms, making mini-batch based optimization infeasible.      To address these issues, we propose a novel framework on top of the deep learning framework named \textit{Cross-Batch Approximation for Multipartite Ranking (CBA-MR)}. In face of \textbf{a)},  we propose a differentiable surrogate optimization problem where the instances having a short-time effect on FPR are rendered with different weights based on the random walk hypothesis. To tackle \textbf{b)}, we propose a fast ranking estimation method, where the full-batch loss evaluation is replaced by a delayed update scheme with the help of an embedding cache. Finally, experimental results on four real-world benchmarks are provided to demonstrate the effectiveness of the proposed method.

----

## [384] Convex Polytope Trees and its Application to VAE

**Authors**: *Mohammadreza Armandpour, Ali Sadeghian, Mingyuan Zhou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/285a25c17f351708754cdb6d56f3962e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/285a25c17f351708754cdb6d56f3962e-Abstract.html)

**Abstract**:

A decision tree is commonly restricted to use a single hyperplane to split the covariate space at each of its internal nodes. It often requires a large number of nodes to achieve high accuracy. In this paper, we propose convex polytope trees (CPT) to expand the family of decision trees by an interpretable generalization of their decision boundary. The splitting function at each node of CPT is based on the logical disjunction of a community of differently weighted probabilistic linear decision-makers, which also geometrically corresponds to a convex polytope in the covariate space. We use a nonparametric Bayesian prior at each node to infer the community's size, encouraging simpler decision boundaries by shrinking the number of polytope facets. We develop a greedy method to efficiently construct CPT and scalable end-to-end training algorithms for the tree parameters when the tree structure is given. We empirically demonstrate the efficiency of CPT over existing state-of-the-art decision trees in several real-world classification and regression tasks from diverse domains.

----

## [385] The Skellam Mechanism for Differentially Private Federated Learning

**Authors**: *Naman Agarwal, Peter Kairouz, Ziyu Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/285baacbdf8fda1de94b19282acd23e2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/285baacbdf8fda1de94b19282acd23e2-Abstract.html)

**Abstract**:

We introduce the multi-dimensional Skellam mechanism, a discrete differential privacy mechanism based on the difference of two independent Poisson random variables. To quantify its privacy guarantees, we analyze the privacy loss distribution via a numerical evaluation and provide a sharp bound on the RÃ©nyi divergence between two shifted Skellam distributions. While useful in both centralized and distributed privacy applications, we investigate how it can be applied in the context of federated learning with secure aggregation under communication constraints. Our theoretical findings and extensive experimental evaluations demonstrate that the Skellam mechanism provides the same privacy-accuracy trade-offs as the continuous Gaussian mechanism, even when the precision is low. More importantly, Skellam is closed under summation and sampling from it only requires sampling from a Poisson distribution -- an efficient routine that ships with all machine learning and data analysis software packages. These features, along with its discrete nature and competitive privacy-accuracy trade-offs, make it an attractive practical alternative to the newly introduced discrete Gaussian mechanism.

----

## [386] Stability and Deviation Optimal Risk Bounds with Convergence Rate $O(1/n)$

**Authors**: *Yegor Klochkov, Nikita Zhivotovskiy*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/286674e3082feb7e5afb92777e48821f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/286674e3082feb7e5afb92777e48821f-Abstract.html)

**Abstract**:

The sharpest known high probability generalization bounds for uniformly stable algorithms (Feldman, Vondrak, NeurIPS 2018, COLT, 2019), (Bousquet, Klochkov, Zhivotovskiy, COLT, 2020) contain a generally inevitable sampling error term of order $\Theta(1/\sqrt{n})$. When applied to excess risk bounds, this leads to suboptimal results in several standard stochastic convex optimization problems. We show that if the so-called Bernstein condition is satisfied, the term $\Theta(1/\sqrt{n})$ can be avoided, and high probability excess risk bounds of order up to $O(1/n)$ are possible via uniform stability. Using this result, we show a high probability excess risk bound with the rate $O(\log n/n)$ for strongly convex and Lipschitz losses valid for \emph{any} empirical risk minimization method. This resolves a question of Shalev-Shwartz, Shamir, Srebro, and Sridharan (COLT, 2009). We discuss how $O(\log n/n)$ high probability excess risk bounds are possible for projected gradient descent in the case of strongly convex and Lipschitz losses without the usual smoothness assumption.

----

## [387] SketchGen: Generating Constrained CAD Sketches

**Authors**: *Wamiq Reyaz Para, Shariq Farooq Bhat, Paul Guerrero, Tom Kelly, Niloy J. Mitra, Leonidas J. Guibas, Peter Wonka*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/28891cb4ab421830acc36b1f5fd6c91e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/28891cb4ab421830acc36b1f5fd6c91e-Abstract.html)

**Abstract**:

Computer-aided design (CAD) is the most widely used modeling approach for technical design. The typical starting point in these designs is 2D sketches which can later be extruded and combined to obtain complex three-dimensional assemblies. Such sketches are typically composed of parametric primitives, such as points, lines, and circular arcs, augmented with geometric constraints linking the primitives, such as coincidence, parallelism, or orthogonality. Sketches can be represented as graphs, with the primitives as nodes and the constraints as edges. Training a model to automatically generate CAD sketches can enable several novel workflows, but is challenging due to the complexity of the graphs and the heterogeneity of the primitives and constraints. In particular, each type of primitive and constraint may require a record of different size and parameter types.We propose SketchGen as a generative model based on a transformer architecture to address the heterogeneity problem by carefully designing a sequential language for the primitives and constraints that allows distinguishing between different primitive or constraint types and their parameters, while encouraging our model to re-use information across related parameters, encoding shared structure. A particular highlight of our work is the ability to produce primitives linked via constraints that enables the final output to be further regularized via a constraint solver. We evaluate our model by demonstrating constraint prediction for given sets of primitives and full sketch generation from scratch, showing that our approach significantly out performs the state-of-the-art in CAD sketch generation.

----

## [388] CLDA: Contrastive Learning for Semi-Supervised Domain Adaptation

**Authors**: *Ankit Singh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/288cd2567953f06e460a33951f55daaf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/288cd2567953f06e460a33951f55daaf-Abstract.html)

**Abstract**:

Unsupervised Domain Adaptation (UDA) aims to align the labeled source distribution with the unlabeled target distribution to obtain domain invariant predictive models. However, the application of well-known UDA approaches does not generalize well in Semi-Supervised Domain Adaptation (SSDA) scenarios where few labeled samples from the target domain are available.This paper proposes a simple Contrastive Learning framework for semi-supervised Domain Adaptation (CLDA) that attempts to bridge the intra-domain gap between the labeled and unlabeled target distributions and the inter-domain gap between source and unlabeled target distribution in SSDA. We suggest employing class-wise contrastive learning to reduce the inter-domain gap and instance-level contrastive alignment between the original(input image) and strongly augmented unlabeled target images to minimize the intra-domain discrepancy. We have empirically shown that both of these modules complement each other to achieve superior performance. Experiments on three well-known domain adaptation benchmark datasets, namely DomainNet, Office-Home, and Office31, demonstrate the effectiveness of our approach. CLDA achieves state-of-the-art results on all the above datasets.

----

## [389] Differentially Private n-gram Extraction

**Authors**: *Kunho Kim, Sivakanth Gopi, Janardhan Kulkarni, Sergey Yekhanin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/28ce9bc954876829eeb56ff46da8e1ab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/28ce9bc954876829eeb56ff46da8e1ab-Abstract.html)

**Abstract**:

We revisit the problem of $n$-gram extraction in the differential privacy setting. In this problem, given a corpus of private text data, the goal is to release as many $n$-grams as possible while preserving user level privacy. Extracting $n$-grams is a fundamental subroutine in many NLP applications such as sentence completion, auto response generation for emails, etc. The problem also arises in other applications such as sequence mining, trajectory analysis, etc., and is a generalization of recently studied differentially private set union (DPSU) by Gopi et al. (2020). In this paper, we develop a new differentially private algorithm for this problem which, in our experiments, significantly outperforms the state-of-the-art. Our improvements stem from combining recent advances in DPSU, privacy accounting, and new heuristics for pruning in the tree-based approach initiated by Chen et al. (2012).

----

## [390] Capturing implicit hierarchical structure in 3D biomedical images with self-supervised hyperbolic representations

**Authors**: *Joy Hsu, Jeffrey Gu, Gong Her Wu, Wah Chiu, Serena Yeung*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/291d43c696d8c3704cdbe0a72ade5f6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/291d43c696d8c3704cdbe0a72ade5f6c-Abstract.html)

**Abstract**:

We consider the task of representation learning for unsupervised segmentation of 3D voxel-grid biomedical images. We show that models that capture implicit hierarchical relationships between subvolumes are better suited for this task. To that end, we consider encoder-decoder architectures with a hyperbolic latent space, to explicitly capture hierarchical relationships present in subvolumes of the data. We propose utilizing a 3D hyperbolic variational autoencoder with a novel gyroplane convolutional layer to map from the embedding space back to 3D images. To capture these relationships, we introduce an essential self-supervised loss---in addition to the standard VAE loss---which infers approximate hierarchies and encourages implicitly related subvolumes to be mapped closer in the embedding space. We present experiments on synthetic datasets along with a dataset from the medical domain to validate our hypothesis.

----

## [391] Noisy Recurrent Neural Networks

**Authors**: *Soon Hoe Lim, N. Benjamin Erichson, Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/29301521774ff3cbd26652b2d5c95996-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/29301521774ff3cbd26652b2d5c95996-Abstract.html)

**Abstract**:

We provide a general framework for studying recurrent neural networks (RNNs) trained by injecting noise into hidden states. Specifically, we consider RNNs that can be viewed as discretizations of stochastic differential equations driven by input data. This framework allows us to study the implicit regularization effect of general noise injection schemes by deriving an approximate explicit regularizer in the small noise regime. We find that, under reasonable assumptions, this implicit regularization promotes flatter minima; it biases towards models with more stable dynamics; and, in classification tasks, it favors models with larger classification margin. Sufficient conditions for global stability are obtained, highlighting the phenomenon of stochastic stabilization, where noise injection can improve stability during training. Our theory is supported by empirical results which demonstrate that the RNNs have improved robustness with respect to various input perturbations.

----

## [392] Matrix encoding networks for neural combinatorial optimization

**Authors**: *Yeong-Dae Kwon, Jinho Choo, Iljoo Yoon, Minah Park, Duwon Park, Youngjune Gwon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/29539ed932d32f1c56324cded92c07c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/29539ed932d32f1c56324cded92c07c2-Abstract.html)

**Abstract**:

Machine Learning (ML) can help solve combinatorial optimization (CO) problems better. A popular approach is to use a neural net to compute on the parameters of a given CO problem and extract useful information that guides the search for good solutions. Many CO problems of practical importance can be specified in a matrix form of parameters quantifying the relationship between two groups of items. There is currently no neural net model, however, that takes in such matrix-style relationship data as an input. Consequently, these types of CO problems have been out of reach for ML engineers. In this paper, we introduce Matrix Encoding Network (MatNet) and show how conveniently it takes in and processes parameters of such complex CO problems. Using an end-to-end model based on MatNet, we solve asymmetric traveling salesman (ATSP) and flexible flow shop (FFSP) problems as the earliest neural approach. In particular, for a class of FFSP we have tested MatNet on, we demonstrate a far superior empirical performance to any methods (neural or not) known to date.

----

## [393] When Is Unsupervised Disentanglement Possible?

**Authors**: *Daniella Horan, Eitan Richardson, Yair Weiss*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/29586cb449c90e249f1f09a0a4ee245a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/29586cb449c90e249f1f09a0a4ee245a-Abstract.html)

**Abstract**:

A common assumption in many domains is that high dimensional data are a smooth nonlinear function of a small number of independent factors. When is it possible to recover the factors from unlabeled data? In the context of deep models this problem is called “disentanglement” and was recently shown to be impossible without additional strong assumptions [17, 19]. In this paper, we show that the assumption of local isometry together with non-Gaussianity of the factors, is sufficient to provably recover disentangled representations from data. We leverage recent advances in deep generative models to construct manifolds of highly realistic images for which the ground truth latent representation is known, and test whether modern and classical methods succeed in recovering the latent factors. For many different manifolds, we find that a spectral method that explicitly optimizes local isometry and non-Gaussianity consistently finds the correct latent factors, while baseline deep autoencoders do not. We propose how to encourage deep autoencoders to find encodings that satisfy local isometry and show that this helps them discover disentangled representations. Overall, our results suggest that in some realistic settings, unsupervised disentanglement is provably possible, without any domain-specific assumptions.

----

## [394] Continuous Latent Process Flows

**Authors**: *Ruizhi Deng, Marcus A. Brubaker, Greg Mori, Andreas M. Lehrmann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2983e3047c0c730d3b7c022584717f3f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2983e3047c0c730d3b7c022584717f3f-Abstract.html)

**Abstract**:

Partial observations of continuous time-series dynamics at arbitrary time stamps exist in many disciplines. Fitting this type of data using statistical models with continuous dynamics is not only promising at an intuitive level but also has practical benefits, including the ability to generate continuous trajectories and to perform inference on previously unseen time stamps. Despite exciting progress in this area, the existing models still face challenges in terms of their representational power and the quality of their variational approximations. We tackle these challenges with continuous latent process flows (CLPF), a principled architecture decoding continuous latent processes into continuous observable processes using a time-dependent normalizing flow driven by a stochastic differential equation. To optimize our model using maximum likelihood, we propose a novel piecewise construction of a variational posterior process and derive the corresponding variational lower bound using trajectory re-weighting. Our ablation studies demonstrate the effectiveness of our contributions in various inference tasks on irregular time grids. Comparisons to state-of-the-art baselines show our model's favourable performance on both synthetic and real-world time-series data.

----

## [395] Perturbation-based Regret Analysis of Predictive Control in Linear Time Varying Systems

**Authors**: *Yiheng Lin, Yang Hu, Guanya Shi, Haoyuan Sun, Guannan Qu, Adam Wierman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/298f587406c914fad5373bb689300433-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/298f587406c914fad5373bb689300433-Abstract.html)

**Abstract**:

We study predictive control in a setting where the dynamics are time-varying and linear, and the costs are time-varying and well-conditioned. At each time step, the controller receives the exact predictions of costs, dynamics, and disturbances for the future $k$ time steps. We show that when the prediction window $k$ is sufficiently large, predictive control is input-to-state stable and achieves a dynamic regret of $O(\lambda^k T)$, where $\lambda < 1$ is a positive constant. This is the first dynamic regret bound on the predictive control of linear time-varying systems. We also show a variation of predictive control obtains the first competitive bound for the control of linear time-varying systems:  $1 + O(\lambda^k)$. Our results are derived using a novel proof framework based on a perturbation bound that characterizes how a small change to the system parameters impacts the optimal trajectory.

----

## [396] Dataset Distillation with Infinitely Wide Convolutional Networks

**Authors**: *Timothy Nguyen, Roman Novak, Lechao Xiao, Jaehoon Lee*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/299a23a2291e2126b91d54f3601ec162-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/299a23a2291e2126b91d54f3601ec162-Abstract.html)

**Abstract**:

The effectiveness of machine learning algorithms arises from being able to extract useful features from large amounts of data. As model and dataset sizes increase, dataset distillation methods that compress large datasets into significantly smaller yet highly performant ones will become valuable in terms of training efficiency and useful feature extraction. To that end, we apply a novel distributed kernel-based meta-learning framework to achieve state-of-the-art results for dataset distillation using infinitely wide convolutional neural networks. For instance, using only 10  datapoints (0.02% of original dataset), we obtain over 65% test accuracy on CIFAR-10 image classification task, a dramatic improvement over the previous best test accuracy of 40%. Our state-of-the-art results extend across many other settings for MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and SVHN. Furthermore, we perform some preliminary analyses of our distilled datasets to shed light on how they differ from naturally occurring data.

----

## [397] SPANN: Highly-efficient Billion-scale Approximate Nearest Neighborhood Search

**Authors**: *Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li, Mao Yang, Jingdong Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/299dc35e747eb77177d9cea10a802da2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/299dc35e747eb77177d9cea10a802da2-Abstract.html)

**Abstract**:

The in-memory algorithms for approximate nearest neighbor search (ANNS) have achieved great success for fast high-recall search, but are extremely expensive when handling very large scale database. Thus, there is an increasing request for the hybrid ANNS solutions with small memory and inexpensive solid-state drive (SSD). In this paper, we present a simple but efficient memory-disk hybrid indexing and search system, named SPANN, that follows the inverted index methodology. It stores the centroid points of the posting lists in the memory and the large posting lists in the disk. We guarantee both disk-access efficiency (low  latency) and high recall by effectively reducing the disk-access number and retrieving high-quality posting lists. In the index-building stage, we adopt a hierarchical balanced clustering algorithm to balance the length of posting lists and augment the posting list by adding the points in the closure of the corresponding clusters. In the search stage, we use a query-aware scheme to dynamically prune the access of unnecessary posting lists.  Experiment results demonstrate that SPANN is 2X faster than the state-of-the-art ANNS solution DiskANN to reach the same recall quality 90% with same memory cost in three billion-scale datasets. It can reach 90% recall@1 and recall@10 in just around one millisecond with only about 10% of original memory cost.  Code is available at: https://github.com/microsoft/SPTAG.

----

## [398] Distilling Object Detectors with Feature Richness

**Authors**: *Zhixing Du, Rui Zhang, Ming Chang, Xishan Zhang, Shaoli Liu, Tianshi Chen, Yunji Chen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/29c0c0ee223856f336d7ea8052057753-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/29c0c0ee223856f336d7ea8052057753-Abstract.html)

**Abstract**:

In recent years, large-scale deep models have achieved great success, but the huge computational complexity and massive storage requirements make it a great challenge to deploy them in resource-limited devices. As a model compression and acceleration method, knowledge distillation effectively improves the performance of small models by transferring the dark knowledge from the teacher detector. However, most of the existing distillation-based detection methods mainly imitating features near bounding boxes, which suffer from two limitations. First, they ignore the beneficial features outside the bounding boxes. Second, these methods imitate some features which are mistakenly regarded as the background by the teacher detector. To address the above issues, we propose a novel Feature-Richness Score (FRS) method to choose important features that improve generalized detectability during distilling. The proposed method effectively retrieves the important features outside the bounding boxes and removes the detrimental features within the bounding boxes. Extensive experiments show that our methods achieve excellent performance on both anchor-based and anchor-free detectors. For example, RetinaNet with ResNet-50 achieves 39.7% in mAP on the COCO2017 dataset, which even surpasses the ResNet-101 based teacher detector 38.9% by 0.8%. Our implementation is available at https://github.com/duzhixing/FRS.

----

## [399] Analysis of one-hidden-layer neural networks via the resolvent method

**Authors**: *Vanessa Piccolo, Dominik Schröder*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/29d74915e1b323676bfc28f91b3c4802-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/29d74915e1b323676bfc28f91b3c4802-Abstract.html)

**Abstract**:

In this work, we investigate the asymptotic spectral density of the random feature matrix $M = Y Y^*$ with $Y = f(WX)$ generated by a single-hidden-layer neural network, where $W$ and $X$ are random rectangular matrices with i.i.d. centred entries and $f$ is a non-linear smooth function which is applied entry-wise. We prove that the Stieltjes transform of the limiting spectral distribution approximately satisfies a quartic self-consistent equation, which is exactly the equation obtained by [Pennington, Worah 2017] and [Benigni, Péché 2019] with the moment method. We extend the previous results to the case of additive bias $Y=f(WX+B)$ with $B$ being an independent rank-one Gaussian random matrix, closer modelling the neural network infrastructures encountered in practice. Our key finding is that in the case of additive bias it is impossible to choose an activation function preserving the layer-to-layer singular value distribution, in sharp contrast to the bias-free case where a simple integral constraint is sufficient to achieve isospectrality. To obtain the asymptotics for the empirical spectral density we follow the resolvent method from random matrix theory via the cumulant expansion. We find that this approach is more robust and less combinatorial than the moment method and expect that it will apply also for models where the combinatorics of the former become intractable. The resolvent method has been widely employed, but compared to previous works, it is applied here to non-linear random matrices.

----



[Go to the previous page](NIPS-2021-list01.md)

[Go to the next page](NIPS-2021-list03.md)

[Go to the catalog section](README.md)