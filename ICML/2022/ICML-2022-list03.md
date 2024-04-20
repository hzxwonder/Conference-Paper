## [400] Transformer Quality in Linear Time

**Authors**: *Weizhe Hua, Zihang Dai, Hanxiao Liu, Quoc V. Le*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/hua22a.html](https://proceedings.mlr.press/v162/hua22a.html)

**Abstract**:

We revisit the design choices in Transformers, and propose methods to address their weaknesses in handling long sequences. First, we propose a simple layer named gated attention unit, which allows the use of a weaker single-head attention with minimal quality loss. We then propose a linear approximation method complementary to this new layer, which is accelerator-friendly and highly competitive in quality. The resulting model, named FLASH, matches the perplexity of improved Transformers over both short (512) and long (8K) context lengths, achieving training speedups of up to 4.9x on Wiki-40B and 12.1x on PG-19 for auto-regressive language modeling, and 4.8x on C4 for masked language modeling.

----

## [401] Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents

**Authors**: *Wenlong Huang, Pieter Abbeel, Deepak Pathak, Igor Mordatch*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22a.html](https://proceedings.mlr.press/v162/huang22a.html)

**Abstract**:

Can world knowledge learned by large language models (LLMs) be used to act in interactive environments? In this paper, we investigate the possibility of grounding high-level tasks, expressed in natural language (e.g. “make breakfast”), to a chosen set of actionable steps (e.g. “open fridge”). While prior work focused on learning from explicit step-by-step examples of how to act, we surprisingly find that if pre-trained LMs are large enough and prompted appropriately, they can effectively decompose high-level tasks into mid-level plans without any further training. However, the plans produced naively by LLMs often cannot map precisely to admissible actions. We propose a procedure that conditions on existing demonstrations and semantically translates the plans to admissible actions. Our evaluation in the recent VirtualHome environment shows that the resulting method substantially improves executability over the LLM baseline. The conducted human evaluation reveals a trade-off between executability and correctness but shows a promising sign towards extracting actionable knowledge from language models.

----

## [402] Forward Operator Estimation in Generative Models with Kernel Transfer Operators

**Authors**: *Zhichun Huang, Rudrasis Chakraborty, Vikas Singh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22b.html](https://proceedings.mlr.press/v162/huang22b.html)

**Abstract**:

Generative models which use explicit density modeling (e.g., variational autoencoders, flow-based generative models) involve finding a mapping from a known distribution, e.g. Gaussian, to the unknown input distribution. This often requires searching over a class of non-linear functions (e.g., representable by a deep neural network). While effective in practice, the associated runtime/memory costs can increase rapidly, usually as a function of the performance desired in an application. We propose a substantially cheaper (and simpler) forward operator estimation strategy based on adapting known results on kernel transfer operators. We show that our formulation enables highly efficient distribution approximation and sampling, and offers surprisingly good empirical performance that compares favorably with powerful baselines, but with significant runtime savings. We show that the algorithm also performs well in small sample size settings (in brain imaging).

----

## [403] Adaptive Best-of-Both-Worlds Algorithm for Heavy-Tailed Multi-Armed Bandits

**Authors**: *Jiatai Huang, Yan Dai, Longbo Huang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22c.html](https://proceedings.mlr.press/v162/huang22c.html)

**Abstract**:

In this paper, we generalize the concept of heavy-tailed multi-armed bandits to adversarial environments, and develop robust best-of-both-worlds algorithms for heavy-tailed multi-armed bandits (MAB), where losses have $\alpha$-th ($1<\alpha\le 2$) moments bounded by $\sigma^\alpha$, while the variances may not exist. Specifically, we design an algorithm \texttt{HTINF}, when the heavy-tail parameters $\alpha$ and $\sigma$ are known to the agent, \texttt{HTINF} simultaneously achieves the optimal regret for both stochastic and adversarial environments, without knowing the actual environment type a-priori. When $\alpha,\sigma$ are unknown, \texttt{HTINF} achieves a $\log T$-style instance-dependent regret in stochastic cases and $o(T)$ no-regret guarantee in adversarial cases. We further develop an algorithm \texttt{AdaTINF}, achieving $\mathcal O(\sigma K^{1-\nicefrac 1\alpha}T^{\nicefrac{1}{\alpha}})$ minimax optimal regret even in adversarial settings, without prior knowledge on $\alpha$ and $\sigma$. This result matches the known regret lower-bound (Bubeck et al., 2013), which assumed a stochastic environment and $\alpha$ and $\sigma$ are both known. To our knowledge, the proposed \texttt{HTINF} algorithm is the first to enjoy a best-of-both-worlds regret guarantee, and \texttt{AdaTINF} is the first algorithm that can adapt to both $\alpha$ and $\sigma$ to achieve optimal gap-indepedent regret bound in classical heavy-tailed stochastic MAB setting and our novel adversarial formulation.

----

## [404] Frustratingly Easy Transferability Estimation

**Authors**: *Long-Kai Huang, Junzhou Huang, Yu Rong, Qiang Yang, Ying Wei*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22d.html](https://proceedings.mlr.press/v162/huang22d.html)

**Abstract**:

Transferability estimation has been an essential tool in selecting a pre-trained model and the layers in it for transfer learning, to transfer, so as to maximize the performance on a target task and prevent negative transfer. Existing estimation algorithms either require intensive training on target tasks or have difficulties in evaluating the transferability between layers. To this end, we propose a simple, efficient, and effective transferability measure named TransRate. Through a single pass over examples of a target task, TransRate measures the transferability as the mutual information between features of target examples extracted by a pre-trained model and their labels. We overcome the challenge of efficient mutual information estimation by resorting to coding rate that serves as an effective alternative to entropy. From the perspective of feature representation, the resulting TransRate evaluates both completeness (whether features contain sufficient information of a target task) and compactness (whether features of each class are compact enough for good generalization) of pre-trained features. Theoretically, we have analyzed the close connection of TransRate to the performance after transfer learning. Despite its extraordinary simplicity in 10 lines of codes, TransRate performs remarkably well in extensive evaluations on 35 pre-trained models and 16 downstream tasks.

----

## [405] Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning? (Provably)

**Authors**: *Yu Huang, Junyang Lin, Chang Zhou, Hongxia Yang, Longbo Huang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22e.html](https://proceedings.mlr.press/v162/huang22e.html)

**Abstract**:

Despite the remarkable success of deep multi-modal learning in practice, it has not been well-explained in theory. Recently, it has been observed that the best uni-modal network outperforms the jointly trained multi-modal network across different combinations of modalities on various tasks, which is counter-intuitive since multiple signals would bring more information (Wang et al., 2020). This work provides a theoretical explanation for the emergence of such performance gap in neural networks for the prevalent joint training framework. Based on a simplified data distribution that captures the realistic property of multi-modal data, we prove that for multi-modal late-fusion network with (smoothed) ReLU activation trained jointly by gradient descent, different modalities will compete with each other and only a subset of modalities will be learned by its corresponding encoder networks. We refer to this phenomenon as modality competition, and the losing modalities, which fail to be discovered, are the origins where the sub-optimality of joint training comes from. In contrast, for uni-modal networks with similar learning settings, we provably show that the networks will focus on learning modality-associated features. Experimentally, we illustrate that modality competition matches the intrinsic behavior of late-fusion joint training to supplement our theoretical results. To the best of our knowledge, our work is the first theoretical treatment towards the degenerating aspect of multi-modal learning in neural networks.

----

## [406] Action-Sufficient State Representation Learning for Control with Structural Constraints

**Authors**: *Biwei Huang, Chaochao Lu, Liu Leqi, José Miguel Hernández-Lobato, Clark Glymour, Bernhard Schölkopf, Kun Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22f.html](https://proceedings.mlr.press/v162/huang22f.html)

**Abstract**:

Perceived signals in real-world scenarios are usually high-dimensional and noisy, and finding and using their representation that contains essential and sufficient information required by downstream decision-making tasks will help improve computational efficiency and generalization ability in the tasks. In this paper, we focus on partially observable environments and propose to learn a minimal set of state representations that capture sufficient information for decision-making, termed Action-Sufficient state Representations (ASRs). We build a generative environment model for the structural relationships among variables in the system and present a principled way to characterize ASRs based on structural constraints and the goal of maximizing cumulative reward in policy learning. We then develop a structured sequential Variational Auto-Encoder to estimate the environment model and extract ASRs. Our empirical results on CarRacing and VizDoom demonstrate a clear advantage of learning and using ASRs for policy learning. Moreover, the estimated environment model and ASRs allow learning behaviors from imagined outcomes in the compact latent space to improve sample efficiency.

----

## [407] 3DLinker: An E(3) Equivariant Variational Autoencoder for Molecular Linker Design

**Authors**: *Yinan Huang, Xingang Peng, Jianzhu Ma, Muhan Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22g.html](https://proceedings.mlr.press/v162/huang22g.html)

**Abstract**:

Deep learning has achieved tremendous success in designing novel chemical compounds with desirable pharmaceutical properties. In this work, we focus on a new type of drug design problem — generating a small “linker” to physically attach two independent molecules with their distinct functions. The main computational challenges include: 1) the generation of linkers is conditional on the two given molecules, in contrast to generating complete molecules from scratch in previous works; 2) linkers heavily depend on the anchor atoms of the two molecules to be connected, which are not known beforehand; 3) 3D structures and orientations of the molecules need to be considered to avoid atom clashes, for which equivariance to E(3) group are necessary. To address these problems, we propose a conditional generative model, named 3DLinker, which is able to predict anchor atoms and jointly generate linker graphs and their 3D structures based on an E(3) equivariant graph variational autoencoder. So far as we know, no previous models could achieve this task. We compare our model with multiple conditional generative models modified from other molecular design tasks and find that our model has a significantly higher rate in recovering molecular graphs, and more importantly, accurately predicting the 3D coordinates of all the atoms.

----

## [408] SDQ: Stochastic Differentiable Quantization with Mixed Precision

**Authors**: *Xijie Huang, Zhiqiang Shen, Shichao Li, Zechun Liu, Xianghong Hu, Jeffry Wicaksana, Eric P. Xing, Kwang-Ting Cheng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22h.html](https://proceedings.mlr.press/v162/huang22h.html)

**Abstract**:

In order to deploy deep models in a computationally efficient manner, model quantization approaches have been frequently used. In addition, as new hardware that supports various-bit arithmetic operations, recent research on mixed precision quantization (MPQ) begins to fully leverage the capacity of representation by searching various bitwidths for different layers and modules in a network. However, previous studies mainly search the MPQ strategy in a costly scheme using reinforcement learning, neural architecture search, etc., or simply utilize partial prior knowledge for bitwidth distribution, which might be biased and sub-optimal. In this work, we present a novel Stochastic Differentiable Quantization (SDQ) method that can automatically learn the MPQ strategy in a more flexible and globally-optimized space with a smoother gradient approximation. Particularly, Differentiable Bitwidth Parameters (DBPs) are employed as the probability factors in stochastic quantization between adjacent bitwidth. After the optimal MPQ strategy is acquired, we further train our network with the entropy-aware bin regularization and knowledge distillation. We extensively evaluate our method on different networks, hardwares (GPUs and FPGA), and datasets. SDQ outperforms all other state-of-the-art mixed or single precision quantization with less bitwidth, and are even better than the original full-precision counterparts across various ResNet and MobileNet families, demonstrating the effectiveness and superiority of our method. Code will be publicly available.

----

## [409] Tackling Data Heterogeneity: A New Unified Framework for Decentralized SGD with Sample-induced Topology

**Authors**: *Yan Huang, Ying Sun, Zehan Zhu, Changzhi Yan, Jinming Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22i.html](https://proceedings.mlr.press/v162/huang22i.html)

**Abstract**:

We develop a general framework unifying several gradient-based stochastic optimization methods for empirical risk minimization problems both in centralized and distributed scenarios. The framework hinges on the introduction of an augmented graph consisting of nodes modeling the samples and edges modeling both the inter-device communication and intra-device stochastic gradient computation. By designing properly the topology of the augmented graph, we are able to recover as special cases the renowned Local-SGD and DSGD algorithms, and provide a unified perspective for variance-reduction (VR) and gradient-tracking (GT) methods such as SAGA, Local-SVRG and GT-SAGA. We also provide a unified convergence analysis for smooth and (strongly) convex objectives relying on a proper structured Lyapunov function, and the obtained rate can recover the best known results for many existing algorithms. The rate results further reveal that VR and GT methods can effectively eliminate data heterogeneity within and across devices, respectively, enabling the exact convergence of the algorithm to the optimal solution. Numerical experiments confirm the findings in this paper.

----

## [410] Efficient Representation Learning via Adaptive Context Pooling

**Authors**: *Chen Huang, Walter Talbott, Navdeep Jaitly, Joshua M. Susskind*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22j.html](https://proceedings.mlr.press/v162/huang22j.html)

**Abstract**:

Self-attention mechanisms model long-range context by using pairwise attention between all input tokens. In doing so, they assume a fixed attention granularity defined by the individual tokens (e.g., text characters or image pixels), which may not be optimal for modeling complex dependencies at higher levels. In this paper, we propose ContextPool to address this problem by adapting the attention granularity for each token. Inspired by the success of ConvNets that are combined with pooling to capture long-range dependencies, we learn to pool neighboring features for each token before computing attention in a given attention layer. The pooling weights and support size are adaptively determined, allowing the pooled features to encode meaningful context with varying scale. We show that ContextPool makes attention models more expressive, achieving strong performance often with fewer layers and thus significantly reduced cost. Experiments validate that our ContextPool module, when plugged into transformer models, matches or surpasses state-of-the-art performance using less compute on several language and image benchmarks, outperforms recent works with learned context sizes or sparse attention patterns, and is also applicable to ConvNets for efficient feature learning.

----

## [411] On the Learning of Non-Autoregressive Transformers

**Authors**: *Fei Huang, Tianhua Tao, Hao Zhou, Lei Li, Minlie Huang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22k.html](https://proceedings.mlr.press/v162/huang22k.html)

**Abstract**:

Non-autoregressive Transformer (NAT) is a family of text generation models, which aims to reduce the decoding latency by predicting the whole sentences in parallel. However, such latency reduction sacrifices the ability to capture left-to-right dependencies, thereby making NAT learning very challenging. In this paper, we present theoretical and empirical analyses to reveal the challenges of NAT learning and propose a unified perspective to understand existing successes. First, we show that simply training NAT by maximizing the likelihood can lead to an approximation of marginal distributions but drops all dependencies between tokens, where the dropped information can be measured by the dataset’s conditional total correlation. Second, we formalize many previous objectives in a unified framework and show that their success can be concluded as maximizing the likelihood on a proxy distribution, leading to a reduced information loss. Empirical studies show that our perspective can explain the phenomena in NAT learning and guide the design of new training methods.

----

## [412] Going Deeper into Permutation-Sensitive Graph Neural Networks

**Authors**: *Zhongyu Huang, Yingheng Wang, Chaozhuo Li, Huiguang He*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22l.html](https://proceedings.mlr.press/v162/huang22l.html)

**Abstract**:

The invariance to permutations of the adjacency matrix, i.e., graph isomorphism, is an overarching requirement for Graph Neural Networks (GNNs). Conventionally, this prerequisite can be satisfied by the invariant operations over node permutations when aggregating messages. However, such an invariant manner may ignore the relationships among neighboring nodes, thereby hindering the expressivity of GNNs. In this work, we devise an efficient permutation-sensitive aggregation mechanism via permutation groups, capturing pairwise correlations between neighboring nodes. We prove that our approach is strictly more powerful than the 2-dimensional Weisfeiler-Lehman (2-WL) graph isomorphism test and not less powerful than the 3-WL test. Moreover, we prove that our approach achieves the linear sampling complexity. Comprehensive experiments on multiple synthetic and real-world datasets demonstrate the superiority of our model.

----

## [413] Directed Acyclic Transformer for Non-Autoregressive Machine Translation

**Authors**: *Fei Huang, Hao Zhou, Yang Liu, Hang Li, Minlie Huang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huang22m.html](https://proceedings.mlr.press/v162/huang22m.html)

**Abstract**:

Non-autoregressive Transformers (NATs) significantly reduce the decoding latency by generating all tokens in parallel. However, such independent predictions prevent NATs from capturing the dependencies between the tokens for generating multiple possible translations. In this paper, we propose Directed Acyclic Transfomer (DA-Transformer), which represents the hidden states in a Directed Acyclic Graph (DAG), where each path of the DAG corresponds to a specific translation. The whole DAG simultaneously captures multiple translations and facilitates fast predictions in a non-autoregressive fashion. Experiments on the raw training data of WMT benchmark show that DA-Transformer substantially outperforms previous NATs by about 3 BLEU on average, which is the first NAT model that achieves competitive results with autoregressive Transformers without relying on knowledge distillation.

----

## [414] Unsupervised Ground Metric Learning Using Wasserstein Singular Vectors

**Authors**: *Geert-Jan Huizing, Laura Cantini, Gabriel Peyré*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huizing22a.html](https://proceedings.mlr.press/v162/huizing22a.html)

**Abstract**:

Defining meaningful distances between samples in a dataset is a fundamental problem in machine learning. Optimal Transport (OT) lifts a distance between features (the "ground metric") to a geometrically meaningful distance between samples. However, there is usually no straightforward choice of ground metric. Supervised ground metric learning approaches exist but require labeled data. In absence of labels, only ad-hoc ground metrics remain. Unsupervised ground metric learning is thus a fundamental problem to enable data-driven applications of OT. In this paper, we propose for the first time a canonical answer by simultaneously computing an OT distance between samples and between features of a dataset. These distance matrices emerge naturally as positive singular vectors of the function mapping ground metrics to OT distances. We provide criteria to ensure the existence and uniqueness of these singular vectors. We then introduce scalable computational methods to approximate them in high-dimensional settings, using stochastic approximation and entropic regularization. Finally, we showcase Wasserstein Singular Vectors on a single-cell RNA-sequencing dataset.

----

## [415] Robust Kernel Density Estimation with Median-of-Means principle

**Authors**: *Pierre Humbert, Batiste Le Bars, Ludovic Minvielle*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/humbert22a.html](https://proceedings.mlr.press/v162/humbert22a.html)

**Abstract**:

In this paper, we introduce a robust non-parametric density estimator combining the popular Kernel Density Estimation method and the Median-of-Means principle (MoM-KDE). This estimator is shown to achieve robustness for a large class of anomalous data, potentially adversarial. In particular, while previous works only prove consistency results under very specific contamination models, this work provides finite-sample high-probability error-bounds without any prior knowledge on the outliers. To highlight the robustness of our method, we introduce an influence function adapted to the considered OUI framework. Finally, we show that MoM-KDE achieves competitive results when compared with other robust kernel estimators, while having significantly lower computational complexity.

----

## [416] A data-driven approach for learning to control computers

**Authors**: *Peter C. Humphreys, David Raposo, Tobias Pohlen, Gregory Thornton, Rachita Chhaparia, Alistair Muldal, Josh Abramson, Petko Georgiev, Adam Santoro, Timothy P. Lillicrap*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/humphreys22a.html](https://proceedings.mlr.press/v162/humphreys22a.html)

**Abstract**:

It would be useful for machines to use computers as humans do so that they can aid us in everyday tasks. This is a setting in which there is also the potential to leverage large-scale expert demonstrations and human judgements of interactive behaviour, which are two ingredients that have driven much recent success in AI. Here we investigate the setting of computer control using keyboard and mouse, with goals specified via natural language. Instead of focusing on hand-designed curricula and specialized action spaces, we focus on developing a scalable method centered on reinforcement learning combined with behavioural priors informed by actual human-computer interactions. We achieve state-of-the-art and human-level mean performance across all tasks within the MiniWob++ benchmark, a challenging suite of computer control problems, and find strong evidence of cross-task transfer. These results demonstrate the usefulness of a unified human-agent interface when training machines to use computers. Altogether our results suggest a formula for achieving competency beyond MiniWob++ and towards controlling computers, in general, as a human would.

----

## [417] Proximal Denoiser for Convergent Plug-and-Play Optimization with Nonconvex Regularization

**Authors**: *Samuel Hurault, Arthur Leclaire, Nicolas Papadakis*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/hurault22a.html](https://proceedings.mlr.press/v162/hurault22a.html)

**Abstract**:

Plug-and-Play (PnP) methods solve ill-posed inverse problems through iterative proximal algorithms by replacing a proximal operator by a denoising operation. When applied with deep neural network denoisers, these methods have shown state-of-the-art visual performance for image restoration problems. However, their theoretical convergence analysis is still incomplete. Most of the existing convergence results consider nonexpansive denoisers, which is non-realistic, or limit their analysis to strongly convex data-fidelity terms in the inverse problem to solve. Recently, it was proposed to train the denoiser as a gradient descent step on a functional parameterized by a deep neural network. Using such a denoiser guarantees the convergence of the PnP version of the Half-Quadratic-Splitting (PnP-HQS) iterative algorithm. In this paper, we show that this gradient denoiser can actually correspond to the proximal operator of another scalar function. Given this new result, we exploit the convergence theory of proximal algorithms in the nonconvex setting to obtain convergence results for PnP-PGD (Proximal Gradient Descent) and PnP-ADMM (Alternating Direction Method of Multipliers). When built on top of a smooth gradient denoiser, we show that PnP-PGD and PnP-ADMM are convergent and target stationary points of an explicit functional. These convergence results are confirmed with numerical experiments on deblurring, super-resolution and inpainting.

----

## [418] Inverse Contextual Bandits: Learning How Behavior Evolves over Time

**Authors**: *Alihan Hüyük, Daniel Jarrett, Mihaela van der Schaar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/huyuk22a.html](https://proceedings.mlr.press/v162/huyuk22a.html)

**Abstract**:

Understanding a decision-maker’s priorities by observing their behavior is critical for transparency and accountability in decision processes{—}such as in healthcare. Though conventional approaches to policy learning almost invariably assume stationarity in behavior, this is hardly true in practice: Medical practice is constantly evolving as clinical professionals fine-tune their knowledge over time. For instance, as the medical community’s understanding of organ transplantations has progressed over the years, a pertinent question is: How have actual organ allocation policies been evolving? To give an answer, we desire a policy learning method that provides interpretable representations of decision-making, in particular capturing an agent’s non-stationary knowledge of the world, as well as operating in an offline manner. First, we model the evolving behavior of decision-makers in terms of contextual bandits, and formalize the problem of Inverse Contextual Bandits ("ICB"). Second, we propose two concrete algorithms as solutions, learning parametric and non-parametric representations of an agent’s behavior. Finally, using both real and simulated data for liver transplantations, we illustrate the applicability and explainability of our method, as well as benchmarking and validating the accuracy of our algorithms.

----

## [419] Datamodels: Understanding Predictions with Data and Data with Predictions

**Authors**: *Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, Aleksander Madry*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ilyas22a.html](https://proceedings.mlr.press/v162/ilyas22a.html)

**Abstract**:

We present a conceptual framework, datamodeling, for analyzing the behavior of a model class in terms of the training data. For any fixed “target” example $x$, training set $S$, and learning algorithm, a datamodel is a parameterized function $2^S \to \mathbb{R}$ that for any subset of $S’ \subset S$—using only information about which examples of $S$ are contained in $S’$—predicts the outcome of training a model on $S’$ and evaluating on $x$. Despite the complexity of the underlying process being approximated (e.g. end-to-end training and evaluation of deep neural networks), we show that even simple linear datamodels successfully predict model outputs. We then demonstrate that datamodels give rise to a variety of applications, such as: accurately predicting the effect of dataset counterfactuals; identifying brittle predictions; finding semantically similar examples; quantifying train-test leakage; and embedding data into a well-behaved and feature-rich representation space.

----

## [420] Parsimonious Learning-Augmented Caching

**Authors**: *Sungjin Im, Ravi Kumar, Aditya Petety, Manish Purohit*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/im22a.html](https://proceedings.mlr.press/v162/im22a.html)

**Abstract**:

Learning-augmented algorithms—in which, traditional algorithms are augmented with machine-learned predictions—have emerged as a framework to go beyond worst-case analysis. The overarching goal is to design algorithms that perform near-optimally when the predictions are accurate yet retain certain worst-case guarantees irrespective of the accuracy of the predictions. This framework has been successfully applied to online problems such as caching where the predictions can be used to alleviate uncertainties. In this paper we introduce and study the setting in which the learning-augmented algorithm can utilize the predictions parsimoniously. We consider the caching problem—which has been extensively studied in the learning-augmented setting—and show that one can achieve quantitatively similar results but only using a sublinear number of predictions.

----

## [421] Bayesian Optimization for Distributionally Robust Chance-constrained Problem

**Authors**: *Yu Inatsu, Shion Takeno, Masayuki Karasuyama, Ichiro Takeuchi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/inatsu22a.html](https://proceedings.mlr.press/v162/inatsu22a.html)

**Abstract**:

In black-box function optimization, we need to consider not only controllable design variables but also uncontrollable stochastic environment variables. In such cases, it is necessary to solve the optimization problem by taking into account the uncertainty of the environmental variables. Chance-constrained (CC) problem, the problem of maximizing the expected value under a certain level of constraint satisfaction probability, is one of the practically important problems in the presence of environmental variables. In this study, we consider distributionally robust CC (DRCC) problem and propose a novel DRCC Bayesian optimization method for the case where the distribution of the environmental variables cannot be precisely specified. We show that the proposed method can find an arbitrary accurate solution with high probability in a finite number of trials, and confirm the usefulness of the proposed method through numerical experiments.

----

## [422] LeNSE: Learning To Navigate Subgraph Embeddings for Large-Scale Combinatorial Optimisation

**Authors**: *David Ireland, Giovanni Montana*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ireland22a.html](https://proceedings.mlr.press/v162/ireland22a.html)

**Abstract**:

Combinatorial Optimisation problems arise in several application domains and are often formulated in terms of graphs. Many of these problems are NP-hard, but exact solutions are not always needed. Several heuristics have been developed to provide near-optimal solutions; however, they do not typically scale well with the size of the graph. We propose a low-complexity approach for identifying a (possibly much smaller) subgraph of the original graph where the heuristics can be run in reasonable time and with a high likelihood of finding a global near-optimal solution. The core component of our approach is LeNSE, a reinforcement learning algorithm that learns how to navigate the space of possible subgraphs using an Euclidean subgraph embedding as its map. To solve CO problems, LeNSE is provided with a discriminative embedding trained using any existing heuristics using only on a small portion of the original graph. When tested on three problems (vertex cover, max-cut and influence maximisation) using real graphs with up to $10$ million edges, LeNSE identifies small subgraphs yielding solutions comparable to those found by running the heuristics on the entire graph, but at a fraction of the total run time. Code for the experiments is available in the public GitHub repo at https://github.com/davidireland3/LeNSE.

----

## [423] The Dual Form of Neural Networks Revisited: Connecting Test Time Predictions to Training Patterns via Spotlights of Attention

**Authors**: *Kazuki Irie, Róbert Csordás, Jürgen Schmidhuber*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/irie22a.html](https://proceedings.mlr.press/v162/irie22a.html)

**Abstract**:

Linear layers in neural networks (NNs) trained by gradient descent can be expressed as a key-value memory system which stores all training datapoints and the initial weights, and produces outputs using unnormalised dot attention over the entire training experience. While this has been technically known since the 1960s, no prior work has effectively studied the operations of NNs in such a form, presumably due to prohibitive time and space complexities and impractical model sizes, all of them growing linearly with the number of training patterns which may get very large. However, this dual formulation offers a possibility of directly visualising how an NN makes use of training patterns at test time, by examining the corresponding attention weights. We conduct experiments on small scale supervised image classification tasks in single-task, multi-task, and continual learning settings, as well as language modelling, and discuss potentials and limits of this view for better understanding and interpreting how NNs exploit training patterns. Our code is public.

----

## [424] A Modern Self-Referential Weight Matrix That Learns to Modify Itself

**Authors**: *Kazuki Irie, Imanol Schlag, Róbert Csordás, Jürgen Schmidhuber*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/irie22b.html](https://proceedings.mlr.press/v162/irie22b.html)

**Abstract**:

The weight matrix (WM) of a neural network (NN) is its program. The programs of many traditional NNs are learned through gradient descent in some error function, then remain fixed. The WM of a self-referential NN, however, can keep rapidly modifying all of itself during runtime. In principle, such NNs can meta-learn to learn, and meta-meta-learn to meta-learn to learn, and so on, in the sense of recursive self-improvement. While NN architectures potentially capable of implementing such behaviour have been proposed since the ’90s, there have been few if any practical studies. Here we revisit such NNs, building upon recent successes of fast weight programmers and closely related linear Transformers. We propose a scalable self-referential WM (SRWM) that learns to use outer products and the delta update rule to modify itself. We evaluate our SRWM in supervised few-shot learning and in multi-task reinforcement learning with procedurally generated game environments. Our experiments demonstrate both practical applicability and competitive performance of the proposed SRWM. Our code is public.

----

## [425] Revisiting Online Submodular Minimization: Gap-Dependent Regret Bounds, Best of Both Worlds and Adversarial Robustness

**Authors**: *Shinji Ito*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ito22a.html](https://proceedings.mlr.press/v162/ito22a.html)

**Abstract**:

In this paper, we consider online decision problems with submodular loss functions. For such problems, existing studies have only dealt with worst-case analysis. This study goes beyond worst-case analysis to show instance-dependent regret bounds. More precisely, for each of the full-information and bandit-feedback settings, we propose an algorithm that achieves a gap-dependent O(log T)-regret bound in the stochastic environment and is comparable to the best existing algorithm in the adversarial environment. The proposed algorithms also work well in the stochastic environment with adversarial corruptions, which is an intermediate setting between the stochastic and adversarial environments.

----

## [426] Modeling Strong and Human-Like Gameplay with KL-Regularized Search

**Authors**: *Athul Paul Jacob, David J. Wu, Gabriele Farina, Adam Lerer, Hengyuan Hu, Anton Bakhtin, Jacob Andreas, Noam Brown*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jacob22a.html](https://proceedings.mlr.press/v162/jacob22a.html)

**Abstract**:

We consider the task of accurately modeling strong human policies in multi-agent decision-making problems, given examples of human behavior. Imitation learning is effective at predicting human actions but may not match the strength of expert humans (e.g., by sometimes committing blunders), while self-play learning and search techniques such as AlphaZero lead to strong performance but may produce policies that differ markedly from human behavior. In chess and Go, we show that regularized search algorithms that penalize KL divergence from an imitation-learned policy yield higher prediction accuracy of strong humans and better performance than imitation learning alone. We then introduce a novel regret minimization algorithm that is regularized based on the KL divergence from an imitation-learned policy, and show that using this algorithm for search in no-press Diplomacy yields a policy that matches the human prediction accuracy of imitation learning while being substantially stronger.

----

## [427] A deep convolutional neural network that is invariant to time rescaling

**Authors**: *Brandon G. Jacques, Zoran Tiganj, Aakash Sarkar, Marc W. Howard, Per B. Sederberg*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jacques22a.html](https://proceedings.mlr.press/v162/jacques22a.html)

**Abstract**:

Human learners can readily understand speech, or a melody, when it is presented slower or faster than usual. This paper presents a deep CNN (SITHCon) that uses a logarithmically compressed temporal representation at each level. Because rescaling the time of the input results in a translation of $\log$ time, and because the output of the convolution is invariant to translations, this network can generalize to out-of-sample data that are temporal rescalings of a learned pattern. We compare the performance of SITHCon to a Temporal Convolution Network (TCN) on classification and regression problems with both univariate and multivariate time series. We find that SITHCon, unlike TCN, generalizes robustly over rescalings of about an order of magnitude. Moreover, we show that the network can generalize over exponentially large scales without retraining the weights simply by extending the range of the logarithmically-compressed temporal memory.

----

## [428] Input Dependent Sparse Gaussian Processes

**Authors**: *Bahram Jafrasteh, Carlos Villacampa-Calvo, Daniel Hernández-Lobato*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jafrasteh22a.html](https://proceedings.mlr.press/v162/jafrasteh22a.html)

**Abstract**:

Gaussian Processes (GPs) are non-parametric models that provide accurate uncertainty estimates. Nevertheless, they have a cubic cost in the number of data instances $N$. To overcome this, sparse GP approximations are used, in which a set of $M \ll N$ inducing points is introduced. The location of the inducing points is learned by considering them parameters of an approximate posterior distribution $q$. Sparse GPs, combined with stochastic variational inference for inferring $q$ have a cost per iteration in $\mathcal{O}(M^3)$. Critically, the inducing points determine the flexibility of the model and they are often located in regions where the latent function changes. A limitation is, however, that in some tasks a large number of inducing points may be required to obtain good results. To alleviate this, we propose here to amortize the computation of the inducing points locations, as well as the parameters of $q$. For this, we use a neural network that receives a data instance as an input and outputs the corresponding inducing points locations and the parameters of $q$. We evaluate our method in several experiments, showing that it performs similar or better than other state-of-the-art sparse variational GPs. However, in our method the number of inducing points is reduced drastically since they depend on the input data. This makes our method scale to larger datasets and have faster training and prediction times.

----

## [429] Regret Minimization with Performative Feedback

**Authors**: *Meena Jagadeesan, Tijana Zrnic, Celestine Mendler-Dünner*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jagadeesan22a.html](https://proceedings.mlr.press/v162/jagadeesan22a.html)

**Abstract**:

In performative prediction, the deployment of a predictive model triggers a shift in the data distribution. As these shifts are typically unknown ahead of time, the learner needs to deploy a model to get feedback about the distribution it induces. We study the problem of finding near-optimal models under performativity while maintaining low regret. On the surface, this problem might seem equivalent to a bandit problem. However, it exhibits a fundamentally richer feedback structure that we refer to as performative feedback: after every deployment, the learner receives samples from the shifted distribution rather than bandit feedback about the reward. Our main contribution is regret bounds that scale only with the complexity of the distribution shifts and not that of the reward function. The key algorithmic idea is careful exploration of the distribution shifts that informs a novel construction of confidence bounds on the risk of unexplored models. The construction only relies on smoothness of the shifts and does not assume convexity. More broadly, our work establishes a conceptual approach for leveraging tools from the bandits literature for the purpose of regret minimization with performative feedback.

----

## [430] Biological Sequence Design with GFlowNets

**Authors**: *Moksh Jain, Emmanuel Bengio, Alex Hernández-García, Jarrid Rector-Brooks, Bonaventure F. P. Dossou, Chanakya Ajit Ekbote, Jie Fu, Tianyu Zhang, Michael Kilgour, Dinghuai Zhang, Lena Simine, Payel Das, Yoshua Bengio*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jain22a.html](https://proceedings.mlr.press/v162/jain22a.html)

**Abstract**:

Design of de novo biological sequences with desired properties, like protein and DNA sequences, often involves an active loop with several rounds of molecule ideation and expensive wet-lab evaluations. These experiments can consist of multiple stages, with increasing levels of precision and cost of evaluation, where candidates are filtered. This makes the diversity of proposed candidates a key consideration in the ideation phase. In this work, we propose an active learning algorithm leveraging epistemic uncertainty estimation and the recently proposed GFlowNets as a generator of diverse candidate solutions, with the objective to obtain a diverse batch of useful (as defined by some utility function, for example, the predicted anti-microbial activity of a peptide) and informative candidates after each round. We also propose a scheme to incorporate existing labeled datasets of candidates, in addition to a reward function, to speed up learning in GFlowNets. We present empirical results on several biological sequence design tasks, and we find that our method generates more diverse and novel batches with high scoring candidates compared to existing approaches.

----

## [431] Combining Diverse Feature Priors

**Authors**: *Saachi Jain, Dimitris Tsipras, Aleksander Madry*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jain22b.html](https://proceedings.mlr.press/v162/jain22b.html)

**Abstract**:

To improve model generalization, model designers often restrict the features that their models use, either implicitly or explicitly. In this work, we explore the design space of leveraging such feature priors by viewing them as distinct perspectives on the data. Specifically, we find that models trained with diverse sets of explicit feature priors have less overlapping failure modes, and can thus be combined more effectively. Moreover, we demonstrate that jointly training such models on additional (unlabeled) data allows them to correct each other’s mistakes, which, in turn, leads to better generalization and resilience to spurious correlations.

----

## [432] Training Your Sparse Neural Network Better with Any Mask

**Authors**: *Ajay Kumar Jaiswal, Haoyu Ma, Tianlong Chen, Ying Ding, Zhangyang Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jaiswal22a.html](https://proceedings.mlr.press/v162/jaiswal22a.html)

**Abstract**:

Pruning large neural networks to create high-quality, independently trainable sparse masks, which can maintain similar performance to their dense counterparts, is very desirable due to the reduced space and time complexity. As research effort is focused on increasingly sophisticated pruning methods that leads to sparse subnetworks trainable from the scratch, we argue for an orthogonal, under-explored theme: improving training techniques for pruned sub-networks, i.e. sparse training. Apart from the popular belief that only the quality of sparse masks matters for sparse training, in this paper we demonstrate an alternative opportunity: one can carefully customize the sparse training techniques to deviate from the default dense network training protocols, consisting of introducing “ghost" neurons and skip connections at the early stage of training, and strategically modifying the initialization as well as labels. Our new sparse training recipe is generally applicable to improving training from scratch with various sparse masks. By adopting our newly curated techniques, we demonstrate significant performance gains across various popular datasets (CIFAR-10, CIFAR-100, TinyImageNet), architectures (ResNet-18/32/104, Vgg16, MobileNet), and sparse mask options (lottery ticket, SNIP/GRASP, SynFlow, or even randomly pruning), compared to the default training protocols, especially at high sparsity levels. Codes will be publicly available.

----

## [433] Sequential Covariate Shift Detection Using Classifier Two-Sample Tests

**Authors**: *Sooyong Jang, Sangdon Park, Insup Lee, Osbert Bastani*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jang22a.html](https://proceedings.mlr.press/v162/jang22a.html)

**Abstract**:

A standard assumption in supervised learning is that the training data and test data are from the same distribution. However, this assumption often fails to hold in practice, which can cause the learned model to perform poorly. We consider the problem of detecting covariate shift, where the covariate distribution shifts but the conditional distribution of labels given covariates remains the same. This problem can naturally be solved using a two-sample test{—}i.e., test whether the current test distribution of covariates equals the training distribution of covariates. Our algorithm builds on classifier tests, which train a discriminator to distinguish train and test covariates, and then use the accuracy of this discriminator as a test statistic. A key challenge is that classifier tests assume given a fixed set of test covariates. In practice, test covariates often arrive sequentially over time{—}e.g., a self-driving car observes a stream of images while driving. Furthermore, covariate shift can occur multiple times{—}i.e., shift and then shift back later or gradually shift over time. To address these challenges, our algorithm trains the discriminator online. Additionally, it evaluates test accuracy using each new covariate before taking a gradient step; this strategy avoids constructing a held-out test set, which can improve sample efficiency. We prove that this optimization preserves the correctness{—}i.e., our algorithm achieves a desired bound on the false positive rate. In our experiments, we show that our algorithm efficiently detects covariate shifts on multiple datasets{—}ImageNet, IWildCam, and Py150.

----

## [434] Surrogate Likelihoods for Variational Annealed Importance Sampling

**Authors**: *Martin Jankowiak, Du Phan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jankowiak22a.html](https://proceedings.mlr.press/v162/jankowiak22a.html)

**Abstract**:

Variational inference is a powerful paradigm for approximate Bayesian inference with a number of appealing properties, including support for model learning and data subsampling. By contrast MCMC methods like Hamiltonian Monte Carlo do not share these properties but remain attractive since, contrary to parametric methods, MCMC is asymptotically unbiased. For these reasons researchers have sought to combine the strengths of both classes of algorithms, with recent approaches coming closer to realizing this vision in practice. However, supporting data subsampling in these hybrid methods can be a challenge, a shortcoming that we address by introducing a surrogate likelihood that can be learned jointly with other variational parameters. We argue theoretically that the resulting algorithm allows an intuitive trade-off between inference fidelity and computational cost. In an extensive empirical comparison we show that our method performs well in practice and that it is well-suited for black-box inference in probabilistic programming frameworks.

----

## [435] Planning with Diffusion for Flexible Behavior Synthesis

**Authors**: *Michael Janner, Yilun Du, Joshua B. Tenenbaum, Sergey Levine*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/janner22a.html](https://proceedings.mlr.press/v162/janner22a.html)

**Abstract**:

Model-based reinforcement learning methods often use learning only for the purpose of recovering an approximate dynamics model, offloading the rest of the decision-making work to classical trajectory optimizers. While conceptually simple, this combination has a number of empirical shortcomings, suggesting that learned models may not be well-suited to standard trajectory optimization. In this paper, we consider what it would look like to fold as much of the trajectory optimization pipeline as possible into the modeling problem, such that sampling from the model and planning with it become nearly identical. The core of our technical approach lies in a diffusion probabilistic model that plans by iteratively denoising trajectories. We show how classifier-guided sampling and image inpainting can be reinterpreted as coherent planning strategies, explore the unusual and useful properties of diffusion-based planning methods, and demonstrate the effectiveness of our framework in control settings that emphasize long-horizon decision-making and test-time flexibility.

----

## [436] HyperImpute: Generalized Iterative Imputation with Automatic Model Selection

**Authors**: *Daniel Jarrett, Bogdan Cebere, Tennison Liu, Alicia Curth, Mihaela van der Schaar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jarrett22a.html](https://proceedings.mlr.press/v162/jarrett22a.html)

**Abstract**:

Consider the problem of imputing missing values in a dataset. One the one hand, conventional approaches using iterative imputation benefit from the simplicity and customizability of learning conditional distributions directly, but suffer from the practical requirement for appropriate model specification of each and every variable. On the other hand, recent methods using deep generative modeling benefit from the capacity and efficiency of learning with neural network function approximators, but are often difficult to optimize and rely on stronger data assumptions. In this work, we study an approach that marries the advantages of both: We propose *HyperImpute*, a generalized iterative imputation framework for adaptively and automatically configuring column-wise models and their hyperparameters. Practically, we provide a concrete implementation with out-of-the-box learners, optimizers, simulators, and extensible interfaces. Empirically, we investigate this framework via comprehensive experiments and sensitivities on a variety of public datasets, and demonstrate its ability to generate accurate imputations relative to a strong suite of benchmarks. Contrary to recent work, we believe our findings constitute a strong defense of the iterative imputation paradigm.

----

## [437] Mitigating Modality Collapse in Multimodal VAEs via Impartial Optimization

**Authors**: *Adrián Javaloy, Maryam Meghdadi, Isabel Valera*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/javaloy22a.html](https://proceedings.mlr.press/v162/javaloy22a.html)

**Abstract**:

A number of variational autoencoders (VAEs) have recently emerged with the aim of modeling multimodal data, e.g., to jointly model images and their corresponding captions. Still, multimodal VAEs tend to focus solely on a subset of the modalities, e.g., by fitting the image while neglecting the caption. We refer to this limitation as modality collapse. In this work, we argue that this effect is a consequence of conflicting gradients during multimodal VAE training. We show how to detect the sub-graphs in the computational graphs where gradients conflict (impartiality blocks), as well as how to leverage existing gradient-conflict solutions from multitask learning to mitigate modality collapse. That is, to ensure impartial optimization across modalities. We apply our training framework to several multimodal VAE models, losses and datasets from the literature, and empirically show that our framework significantly improves the reconstruction performance, conditional generation, and coherence of the latent space across modalities.

----

## [438] Towards understanding how momentum improves generalization in deep learning

**Authors**: *Samy Jelassi, Yuanzhi Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jelassi22a.html](https://proceedings.mlr.press/v162/jelassi22a.html)

**Abstract**:

Stochastic gradient descent (SGD) with momentum is widely used for training modern deep learning architectures. While it is well-understood that using momentum can lead to faster convergence rate in various settings, it has also been observed that momentum yields higher generalization. Prior work argue that momentum stabilizes the SGD noise during training and this leads to higher generalization. In this paper, we adopt another perspective and first empirically show that gradient descent with momentum (GD+M) significantly improves generalization compared to gradient descent (GD) in some deep learning problems. From this observation, we formally study how momentum improves generalization. We devise a binary classification setting where a one-hidden layer (over-parameterized) convolutional neural network trained with GD+M provably generalizes better than the same network trained with GD, when both algorithms are similarly initialized. The key insight in our analysis is that momentum is beneficial in datasets where the examples share some feature but differ in their margin. Contrary to GD that memorizes the small margin data, GD+M still learns the feature in these data thanks to its historical gradients. Lastly, we empirically validate our theoretical findings.

----

## [439] MASER: Multi-Agent Reinforcement Learning with Subgoals Generated from Experience Replay Buffer

**Authors**: *Jeewon Jeon, Woojun Kim, Whiyoung Jung, Youngchul Sung*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jeon22a.html](https://proceedings.mlr.press/v162/jeon22a.html)

**Abstract**:

In this paper, we consider cooperative multi-agent reinforcement learning (MARL) with sparse reward. To tackle this problem, we propose a novel method named MASER: MARL with subgoals generated from experience replay buffer. Under the widely-used assumption of centralized training with decentralized execution and consistent Q-value decomposition for MARL, MASER automatically generates proper subgoals for multiple agents from the experience replay buffer by considering both individual Q-value and total Q-value. Then, MASER designs individual intrinsic reward for each agent based on actionable representation relevant to Q-learning so that the agents reach their subgoals while maximizing the joint action value. Numerical results show that MASER significantly outperforms StarCraft II micromanagement benchmark compared to other state-of-the-art MARL algorithms.

----

## [440] An Exact Symbolic Reduction of Linear Smart Predict+Optimize to Mixed Integer Linear Programming

**Authors**: *Jihwan Jeong, Parth Jaggi, Andrew Butler, Scott Sanner*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jeong22a.html](https://proceedings.mlr.press/v162/jeong22a.html)

**Abstract**:

Predictive models are traditionally optimized independently of their use in downstream decision-based optimization. The ‘smart, predict then optimize’ (SPO) framework addresses this shortcoming by optimizing predictive models in order to minimize the final downstream decision loss. To date, several local first-order methods and convex approximations have been proposed. These methods have proven to be effective in practice, however, it remains generally unclear as to how close these local solutions are to global optimality. In this paper, we cast the SPO problem as a bi-level program and apply Symbolic Variable Elimination (SVE) to analytically solve the lower optimization. The resulting program can then be formulated as a mixed-integer linear program (MILP) which is solved to global optimality using standard off-the-shelf solvers. To our knowledge, our framework is the first to provide a globally optimal solution to the linear SPO problem. Experimental results comparing with state-of-the-art local SPO solvers show that the globally optimal solution obtains up to two orders of magnitude reduction in decision regret.

----

## [441] Agnostic Learnability of Halfspaces via Logistic Loss

**Authors**: *Ziwei Ji, Kwangjun Ahn, Pranjal Awasthi, Satyen Kale, Stefani Karp*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ji22a.html](https://proceedings.mlr.press/v162/ji22a.html)

**Abstract**:

We investigate approximation guarantees provided by logistic regression for the fundamental problem of agnostic learning of homogeneous halfspaces. Previously, for a certain broad class of “well-behaved” distributions on the examples, Diakonikolas et al. (2020) proved an tilde{Omega}(OPT) lower bound, while Frei et al. (2021) proved an tilde{O}(sqrt{OPT}) upper bound, where OPT denotes the best zero-one/misclassification risk of a homogeneous halfspace. In this paper, we close this gap by constructing a well-behaved distribution such that the global minimizer of the logistic risk over this distribution only achieves Omega(sqrt{OPT}) misclassification risk, matching the upper bound in (Frei et al., 2021). On the other hand, we also show that if we impose a radial-Lipschitzness condition in addition to well-behaved-ness on the distribution, logistic regression on a ball of bounded radius reaches tilde{O}(OPT) misclassification risk. Our techniques also show for any well-behaved distribution, regardless of radial Lipschitzness, we can overcome the Omega(sqrt{OPT}) lower bound for logistic loss simply at the cost of one additional convex optimization step involving the hinge loss and attain tilde{O}(OPT) misclassification risk. This two-step convex optimization algorithm is simpler than previous methods obtaining this guarantee, all of which require solving O(log(1/OPT)) minimization problems.

----

## [442] Improving Policy Optimization with Generalist-Specialist Learning

**Authors**: *Zhiwei Jia, Xuanlin Li, Zhan Ling, Shuang Liu, Yiran Wu, Hao Su*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jia22a.html](https://proceedings.mlr.press/v162/jia22a.html)

**Abstract**:

Generalization in deep reinforcement learning over unseen environment variations usually requires policy learning over a large set of diverse training variations. We empirically observe that an agent trained on many variations (a generalist) tends to learn faster at the beginning, yet its performance plateaus at a less optimal level for a long time. In contrast, an agent trained only on a few variations (a specialist) can often achieve high returns under a limited computational budget. To have the best of both worlds, we propose a novel generalist-specialist training framework. Specifically, we first train a generalist on all environment variations; when it fails to improve, we launch a large population of specialists with weights cloned from the generalist, each trained to master a selected small subset of variations. We finally resume the training of the generalist with auxiliary rewards induced by demonstrations of all specialists. In particular, we investigate the timing to start specialist training and compare strategies to learn generalists with assistance from specialists. We show that this framework pushes the envelope of policy learning on several challenging and popular benchmarks including Procgen, Meta-World and ManiSkill.

----

## [443] Translatotron 2: High-quality direct speech-to-speech translation with voice preservation

**Authors**: *Ye Jia, Michelle Tadmor Ramanovich, Tal Remez, Roi Pomerantz*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jia22b.html](https://proceedings.mlr.press/v162/jia22b.html)

**Abstract**:

We present Translatotron 2, a neural direct speech-to-speech translation model that can be trained end-to-end. Translatotron 2 consists of a speech encoder, a linguistic decoder, an acoustic synthesizer, and a single attention module that connects them together. Experimental results on three datasets consistently show that Translatotron 2 outperforms the original Translatotron by a large margin on both translation quality (up to +15.5 BLEU) and speech generation quality, and approaches the same of cascade systems. In addition, we propose a simple method for preserving speakers’ voices from the source speech to the translation speech in a different language. Unlike existing approaches, the proposed method is able to preserve each speaker’s voice on speaker turns without requiring for speaker segmentation. Furthermore, compared to existing approaches, it better preserves speaker’s privacy and mitigates potential misuse of voice cloning for creating spoofing audio artifacts.

----

## [444] Online Learning and Pricing with Reusable Resources: Linear Bandits with Sub-Exponential Rewards

**Authors**: *Huiwen Jia, Cong Shi, Siqian Shen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jia22c.html](https://proceedings.mlr.press/v162/jia22c.html)

**Abstract**:

We consider a price-based revenue management problem with reusable resources over a finite time horizon $T$. The problem finds important applications in car/bicycle rental, ridesharing, cloud computing, and hospitality management. Customers arrive following a price-dependent Poisson process and each customer requests one unit of $c$ homogeneous reusable resources. If there is an available unit, the customer gets served within a price-dependent exponentially distributed service time; otherwise, she waits in a queue until the next available unit. The decision maker assumes that the inter-arrival and service intervals have an unknown linear dependence on a $d_f$-dimensional feature vector associated with the posted price. We propose a rate-optimal online learning and pricing algorithm, termed Batch Linear Confidence Bound (BLinUCB), and prove that the cumulative regret is $\tilde{O}( d_f \sqrt{T } )$. In establishing the regret, we bound the transient system performance upon price changes via a coupling argument, and also generalize linear bandits to accommodate sub-exponential rewards.

----

## [445] The Role of Deconfounding in Meta-learning

**Authors**: *Yinjie Jiang, Zhengyu Chen, Kun Kuang, Luotian Yuan, Xinhai Ye, Zhihua Wang, Fei Wu, Ying Wei*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jiang22a.html](https://proceedings.mlr.press/v162/jiang22a.html)

**Abstract**:

Meta-learning has emerged as a potent paradigm for quick learning of few-shot tasks, by leveraging the meta-knowledge learned from meta-training tasks. Well-generalized meta-knowledge that facilitates fast adaptation in each task is preferred; however, recent evidence suggests the undesirable memorization effect where the meta-knowledge simply memorizing all meta-training tasks discourages task-specific adaptation and poorly generalizes. There have been several solutions to mitigating the effect, including both regularizer-based and augmentation-based methods, while a systematic understanding of these methods in a single framework is still lacking. In this paper, we offer a novel causal perspective of meta-learning. Through the lens of causality, we conclude the universal label space as a confounder to be the causing factor of memorization and frame the two lines of prevailing methods as different deconfounder approaches. Remarkably, derived from the causal inference principle of front-door adjustment, we propose two frustratingly easy but effective deconfounder algorithms, i.e., sampling multiple versions of the meta-knowledge via Dropout and grouping the meta-knowledge into multiple bins. The proposed causal perspective not only brings in the two deconfounder algorithms that surpass previous works in four benchmark datasets towards combating memorization, but also opens a promising direction for meta-learning.

----

## [446] Subspace Learning for Effective Meta-Learning

**Authors**: *Weisen Jiang, James T. Kwok, Yu Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jiang22b.html](https://proceedings.mlr.press/v162/jiang22b.html)

**Abstract**:

Meta-learning aims to extract meta-knowledge from historical tasks to accelerate learning on new tasks. Typical meta-learning algorithms like MAML learn a globally-shared meta-model for all tasks. However, when the task environments are complex, task model parameters are diverse and a common meta-model is insufficient to capture all the meta-knowledge. To address this challenge, in this paper, task model parameters are structured into multiple subspaces, and each subspace represents one type of meta-knowledge. We propose an algorithm to learn the meta-parameters (\ie, subspace bases). We theoretically study the generalization properties of the learned subspaces. Experiments on regression and classification meta-learning datasets verify the effectiveness of the proposed algorithm.

----

## [447] Optimal Algorithms for Stochastic Multi-Level Compositional Optimization

**Authors**: *Wei Jiang, Bokun Wang, Yibo Wang, Lijun Zhang, Tianbao Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jiang22c.html](https://proceedings.mlr.press/v162/jiang22c.html)

**Abstract**:

In this paper, we investigate the problem of stochastic multi-level compositional optimization, where the objective function is a composition of multiple smooth but possibly non-convex functions. Existing methods for solving this problem either suffer from sub-optimal sample complexities or need a huge batch size. To address this limitation, we propose a Stochastic Multi-level Variance Reduction method (SMVR), which achieves the optimal sample complexity of $\mathcal{O}\left(1 / \epsilon^{3}\right)$ to find an $\epsilon$-stationary point for non-convex objectives. Furthermore, when the objective function satisfies the convexity or Polyak-{Ł}ojasiewicz (PL) condition, we propose a stage-wise variant of SMVR and improve the sample complexity to $\mathcal{O}\left(1 / \epsilon^{2}\right)$ for convex functions or $\mathcal{O}\left(1 /(\mu\epsilon)\right)$ for non-convex functions satisfying the $\mu$-PL condition. The latter result implies the same complexity for $\mu$-strongly convex functions. To make use of adaptive learning rates, we also develop Adaptive SMVR, which achieves the same optimal complexities but converges faster in practice. All our complexities match the lower bounds not only in terms of $\epsilon$ but also in terms of $\mu$ (for PL or strongly convex functions), without using a large batch size in each iteration.

----

## [448] Antibody-Antigen Docking and Design via Hierarchical Structure Refinement

**Authors**: *Wengong Jin, Regina Barzilay, Tommi S. Jaakkola*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22a.html](https://proceedings.mlr.press/v162/jin22a.html)

**Abstract**:

Computational antibody design seeks to automatically create an antibody that binds to an antigen. The binding affinity is governed by the 3D binding interface where antibody residues (paratope) closely interact with antigen residues (epitope). Thus, the key question of antibody design is how to predict the 3D paratope-epitope complex (i.e., docking) for paratope generation. In this paper, we propose a new model called Hierarchical Structure Refinement Network (HSRN) for paratope docking and design. During docking, HSRN employs a hierarchical message passing network to predict atomic forces and use them to refine a binding complex in an iterative, equivariant manner. During generation, its autoregressive decoder progressively docks generated paratopes and builds a geometric representation of the binding interface to guide the next residue choice. Our results show that HSRN significantly outperforms prior state-of-the-art on paratope docking and design benchmarks.

----

## [449] Sharpened Quasi-Newton Methods: Faster Superlinear Rate and Larger Local Convergence Neighborhood

**Authors**: *Qiujiang Jin, Alec Koppel, Ketan Rajawat, Aryan Mokhtari*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22b.html](https://proceedings.mlr.press/v162/jin22b.html)

**Abstract**:

Non-asymptotic analysis of quasi-Newton methods have received a lot of attention recently. In particular, several works have established a non-asymptotic superlinear rate of $$\mathcal{O}((1/\sqrt{t})^t)$$ for the (classic) BFGS method by exploiting the fact that its error of Newton direction approximation approaches zero. Moreover, a greedy variant of the BFGS method was recently proposed which accelerates the convergence of BFGS by directly approximating the Hessian matrix, instead of Newton direction, and achieves a fast local quadratic convergence rate. Alas, the local quadratic convergence of Greedy-BFGS requires way more updates compared to the number of iterations that BFGS requires for a local superlinear rate. This is due to the fact that in Greedy-BFGS the Hessian is directly approximated and the Newton direction approximation may not be as accurate as the one for BFGS. In this paper, we close this gap and present a novel BFGS method that has the best of two worlds. More precisely, it leverages the approximation ideas of both BFGS and Greedy-BFGS to properly approximate both the Newton direction and the Hessian matrix. Our theoretical results show that our method out-performs both BFGS and Greedy-BFGS in terms of convergence rate, while it reaches its quadratic convergence rate with fewer steps compared to Greedy-BFGS. Numerical experiments on various datasets also confirm our theoretical findings.

----

## [450] The Power of Exploiter: Provable Multi-Agent RL in Large State Spaces

**Authors**: *Chi Jin, Qinghua Liu, Tiancheng Yu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22c.html](https://proceedings.mlr.press/v162/jin22c.html)

**Abstract**:

Modern reinforcement learning (RL) commonly engages practical problems with large state spaces, where function approximation must be deployed to approximate either the value function or the policy. While recent progresses in RL theory address a rich set of RL problems with general function approximation, such successes are mostly restricted to the single-agent setting. It remains elusive how to extend these results to multi-agent RL, especially in the face of new game-theoretical challenges. This paper considers two-player zero-sum Markov Games (MGs). We propose a new algorithm that can provably find the Nash equilibrium policy using a polynomial number of samples, for any MG with low multi-agent Bellman-Eluder dimension—a new complexity measure adapted from its single-agent version (Jin et al., 2021). A key component of our new algorithm is the exploiter, which facilitates the learning of the main player by deliberately exploiting her weakness. Our theoretical framework is generic, which applies to a wide range of models including but not limited to tabular MGs, MGs with linear or kernel function approximation, and MGs with rich observations.

----

## [451] Domain Adaptation for Time Series Forecasting via Attention Sharing

**Authors**: *Xiaoyong Jin, Youngsuk Park, Danielle C. Maddix, Hao Wang, Yuyang Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22d.html](https://proceedings.mlr.press/v162/jin22d.html)

**Abstract**:

Recently, deep neural networks have gained increasing popularity in the field of time series forecasting. A primary reason for their success is their ability to effectively capture complex temporal dynamics across multiple related time series. The advantages of these deep forecasters only start to emerge in the presence of a sufficient amount of data. This poses a challenge for typical forecasting problems in practice, where there is a limited number of time series or observations per time series, or both. To cope with this data scarcity issue, we propose a novel domain adaptation framework, Domain Adaptation Forecaster (DAF). DAF leverages statistical strengths from a relevant domain with abundant data samples (source) to improve the performance on the domain of interest with limited data (target). In particular, we use an attention-based shared module with a domain discriminator across domains and private modules for individual domains. We induce domain-invariant latent features (queries and keys) and retrain domain-specific features (values) simultaneously to enable joint training of forecasters on source and target domains. A main insight is that our design of aligning keys allows the target domain to leverage source time series even with different characteristics. Extensive experiments on various domains demonstrate that our proposed method outperforms state-of-the-art baselines on synthetic and real-world datasets, and ablation studies verify the effectiveness of our design choices.

----

## [452] Accelerated Federated Learning with Decoupled Adaptive Optimization

**Authors**: *Jiayin Jin, Jiaxiang Ren, Yang Zhou, Lingjuan Lyu, Ji Liu, Dejing Dou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22e.html](https://proceedings.mlr.press/v162/jin22e.html)

**Abstract**:

The federated learning (FL) framework enables edge clients to collaboratively learn a shared inference model while keeping privacy of training data on clients. Recently, many heuristics efforts have been made to generalize centralized adaptive optimization methods, such as SGDM, Adam, AdaGrad, etc., to federated settings for improving convergence and accuracy. However, there is still a paucity of theoretical principles on where to and how to design and utilize adaptive optimization methods in federated settings. This work aims to develop novel adaptive optimization methods for FL from the perspective of dynamics of ordinary differential equations (ODEs). First, an analytic framework is established to build a connection between federated optimization methods and decompositions of ODEs of corresponding centralized optimizers. Second, based on this analytic framework, a momentum decoupling adaptive optimization method, FedDA, is developed to fully utilize the global momentum on each local iteration and accelerate the training convergence. Last but not least, full batch gradients are utilized to mimic centralized optimization in the end of the training process to ensure the convergence and overcome the possible inconsistency caused by adaptive optimization methods.

----

## [453] Supervised Off-Policy Ranking

**Authors**: *Yue Jin, Yue Zhang, Tao Qin, Xudong Zhang, Jian Yuan, Houqiang Li, Tie-Yan Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22f.html](https://proceedings.mlr.press/v162/jin22f.html)

**Abstract**:

Off-policy evaluation (OPE) is to evaluate a target policy with data generated by other policies. Most previous OPE methods focus on precisely estimating the true performance of a policy. We observe that in many applications, (1) the end goal of OPE is to compare two or multiple candidate policies and choose a good one, which is a much simpler task than precisely evaluating their true performance; and (2) there are usually multiple policies that have been deployed to serve users in real-world systems and thus the true performance of these policies can be known. Inspired by the two observations, in this work, we study a new problem, supervised off-policy ranking (SOPR), which aims to rank a set of target policies based on supervised learning by leveraging off-policy data and policies with known performance. We propose a method to solve SOPR, which learns a policy scoring model by minimizing a ranking loss of the training policies rather than estimating the precise policy performance. The scoring model in our method, a hierarchical Transformer based model, maps a set of state-action pairs to a score, where the state of each pair comes from the off-policy data and the action is taken by a target policy on the state in an offline manner. Extensive experiments on public datasets show that our method outperforms baseline methods in terms of rank correlation, regret value, and stability. Our code is publicly available at GitHub.

----

## [454] Input-agnostic Certified Group Fairness via Gaussian Parameter Smoothing

**Authors**: *Jiayin Jin, Zeru Zhang, Yang Zhou, Lingfei Wu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jin22g.html](https://proceedings.mlr.press/v162/jin22g.html)

**Abstract**:

Only recently, researchers attempt to provide classification algorithms with provable group fairness guarantees. Most of these algorithms suffer from harassment caused by the requirement that the training and deployment data follow the same distribution. This paper proposes an input-agnostic certified group fairness algorithm, FairSmooth, for improving the fairness of classification models while maintaining the remarkable prediction accuracy. A Gaussian parameter smoothing method is developed to transform base classifiers into their smooth versions. An optimal individual smooth classifier is learnt for each group with only the data regarding the group and an overall smooth classifier for all groups is generated by averaging the parameters of all the individual smooth ones. By leveraging the theory of nonlinear functional analysis, the smooth classifiers are reformulated as output functions of a Nemytskii operator. Theoretical analysis is conducted to derive that the Nemytskii operator is smooth and induces a Frechet differentiable smooth manifold. We theoretically demonstrate that the smooth manifold has a global Lipschitz constant that is independent of the domain of the input data, which derives the input-agnostic certified group fairness.

----

## [455] Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations

**Authors**: *Jaehyeong Jo, Seul Lee, Sung Ju Hwang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jo22a.html](https://proceedings.mlr.press/v162/jo22a.html)

**Abstract**:

Generating graph-structured data requires learning the underlying distribution of graphs. Yet, this is a challenging problem, and the previous graph generative methods either fail to capture the permutation-invariance property of graphs or cannot sufficiently model the complex dependency between nodes and edges, which is crucial for generating real-world graphs such as molecules. To overcome such limitations, we propose a novel score-based generative model for graphs with a continuous-time framework. Specifically, we propose a new graph diffusion process that models the joint distribution of the nodes and edges through a system of stochastic differential equations (SDEs). Then, we derive novel score matching objectives tailored for the proposed diffusion process to estimate the gradient of the joint log-density with respect to each component, and introduce a new solver for the system of SDEs to efficiently sample from the reverse diffusion process. We validate our graph generation method on diverse datasets, on which it either achieves significantly superior or competitive performance to the baselines. Further analysis shows that our method is able to generate molecules that lie close to the training distribution yet do not violate the chemical valency rule, demonstrating the effectiveness of the system of SDEs in modeling the node-edge relationships.

----

## [456] Choosing Answers in Epsilon-Best-Answer Identification for Linear Bandits

**Authors**: *Marc Jourdan, Rémy Degenne*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jourdan22a.html](https://proceedings.mlr.press/v162/jourdan22a.html)

**Abstract**:

In pure-exploration problems, information is gathered sequentially to answer a question on the stochastic environment. While best-arm identification for linear bandits has been extensively studied in recent years, few works have been dedicated to identifying one arm that is $\varepsilon$-close to the best one (and not exactly the best one). In this problem with several correct answers, an identification algorithm should focus on one candidate among those answers and verify that it is correct. We demonstrate that picking the answer with highest mean does not allow an algorithm to reach asymptotic optimality in terms of expected sample complexity. Instead, a furthest answer should be identified.		 Using that insight to choose the candidate answer carefully, we develop a simple procedure to adapt best-arm identification algorithms to tackle $\varepsilon$-best-answer identification in transductive linear stochastic bandits. Finally, we propose an asymptotically optimal algorithm for this setting, which is shown to achieve competitive empirical performance against existing modified best-arm identification algorithms.

----

## [457] Robust Fine-Tuning of Deep Neural Networks with Hessian-based Generalization Guarantees

**Authors**: *Haotian Ju, Dongyue Li, Hongyang R. Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ju22a.html](https://proceedings.mlr.press/v162/ju22a.html)

**Abstract**:

We consider transfer learning approaches that fine-tune a pretrained deep neural network on a target task. We investigate generalization properties of fine-tuning to understand the problem of overfitting, which often happens in practice. Previous works have shown that constraining the distance from the initialization of fine-tuning improves generalization. Using a PAC-Bayesian analysis, we observe that besides distance from initialization, Hessians affect generalization through the noise stability of deep neural networks against noise injections. Motivated by the observation, we develop Hessian distance-based generalization bounds for a wide range of fine-tuning methods. Next, we investigate the robustness of fine-tuning with noisy labels. We design an algorithm that incorporates consistent losses and distance-based regularization for fine-tuning. Additionally, we prove a generalization error bound of our algorithm under class conditional independent noise in the training dataset labels. We perform a detailed empirical study of our algorithm on various noisy environments and architectures. For example, on six image classification tasks whose training labels are generated with programmatic labeling, we show a 3.26% accuracy improvement over prior methods. Meanwhile, the Hessian distance measure of the fine-tuned network using our algorithm decreases by six times more than existing approaches.

----

## [458] Robust alignment of cross-session recordings of neural population activity by behaviour via unsupervised domain adaptation

**Authors**: *Justin Jude, Matthew G. Perich, Lee E. Miller, Matthias H. Hennig*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jude22a.html](https://proceedings.mlr.press/v162/jude22a.html)

**Abstract**:

Neural population activity relating to behaviour is assumed to be inherently low-dimensional despite the observed high dimensionality of data recorded using multi-electrode arrays. Therefore, predicting behaviour from neural population recordings has been shown to be most effective when using latent variable models. Over time however, the activity of single neurons can drift, and different neurons will be recorded due to movement of implanted neural probes. This means that a decoder trained to predict behaviour on one day performs worse when tested on a different day. On the other hand, evidence suggests that the latent dynamics underlying behaviour may be stable even over months and years. Based on this idea, we introduce a model capable of inferring behaviourally relevant latent dynamics from previously unseen data recorded from the same animal, without any need for decoder recalibration. We show that unsupervised domain adaptation combined with a sequential variational autoencoder, trained on several sessions, can achieve good generalisation to unseen data and correctly predict behaviour where conventional methods fail. Our results further support the hypothesis that behaviour-related neural dynamics are low-dimensional and stable over time, and will enable more effective and flexible use of brain computer interface technologies.

----

## [459] On Measuring Causal Contributions via do-interventions

**Authors**: *Yonghan Jung, Shiva Prasad Kasiviswanathan, Jin Tian, Dominik Janzing, Patrick Blöbaum, Elias Bareinboim*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jung22a.html](https://proceedings.mlr.press/v162/jung22a.html)

**Abstract**:

Causal contributions measure the strengths of different causes to a target quantity. Understanding causal contributions is important in empirical sciences and data-driven disciplines since it allows to answer practical queries like “what are the contributions of each cause to the effect?” In this paper, we develop a principled method for quantifying causal contributions. First, we provide desiderata of properties axioms that causal contribution measures should satisfy and propose the do-Shapley values (inspired by do-interventions [Pearl, 2000]) as a unique method satisfying these properties. Next, we develop a criterion under which the do-Shapley values can be efficiently inferred from non-experimental data. Finally, we provide do-Shapley estimators exhibiting consistency, computational feasibility, and statistical robustness. Simulation results corroborate with the theory.

----

## [460] Efficient Approximate Inference for Stationary Kernel on Frequency Domain

**Authors**: *Yohan Jung, Kyungwoo Song, Jinkyoo Park*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/jung22b.html](https://proceedings.mlr.press/v162/jung22b.html)

**Abstract**:

Based on the Fourier duality between a stationary kernel and its spectral density, modeling the spectral density using a Gaussian mixture density enables one to construct a flexible kernel, known as a Spectral Mixture kernel, that can model any stationary kernel. However, despite its expressive power, training this kernel is typically difficult because scalability and overfitting issues often arise due to a large number of training parameters. To resolve these issues, we propose an approximate inference method for estimating the Spectral mixture kernel hyperparameters. Specifically, we approximate this kernel by using the finite random spectral points based on Random Fourier Feature and optimize the parameters for the distribution of spectral points by sampling-based variational inference. To improve this inference procedure, we analyze the training loss and propose two special methods: a sampling method of spectral points to reduce the error of the approximate kernel in training, and an approximate natural gradient to accelerate the convergence of parameter inference.

----

## [461] Sketching Algorithms and Lower Bounds for Ridge Regression

**Authors**: *Praneeth Kacham, David P. Woodruff*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kacham22a.html](https://proceedings.mlr.press/v162/kacham22a.html)

**Abstract**:

We give a sketching-based iterative algorithm that computes a $1+\varepsilon$ approximate solution for the ridge regression problem $\min_x \|Ax-b\|_2^2 +\lambda\|x\|_2^2$ where $A \in R^{n \times d}$ with $d \ge n$. Our algorithm, for a constant number of iterations (requiring a constant number of passes over the input), improves upon earlier work (Chowdhury et al.) by requiring that the sketching matrix only has a weaker Approximate Matrix Multiplication (AMM) guarantee that depends on $\varepsilon$, along with a constant subspace embedding guarantee. The earlier work instead requires that the sketching matrix has a subspace embedding guarantee that depends on $\varepsilon$. For example, to produce a $1+\varepsilon$ approximate solution in $1$ iteration, which requires $2$ passes over the input, our algorithm requires the OSNAP embedding to have $m= O(n\sigma^2/\lambda\varepsilon)$ rows with a sparsity parameter $s = O(\log(n))$, whereas the earlier algorithm of Chowdhury et al. with the same number of rows of OSNAP requires a sparsity $s = O(\sqrt{\sigma^2/\lambda\varepsilon} \cdot \log(n))$, where $\sigma = \opnorm{A}$ is the spectral norm of the matrix $A$. We also show that this algorithm can be used to give faster algorithms for kernel ridge regression. Finally, we show that the sketch size required for our algorithm is essentially optimal for a natural framework of algorithms for ridge regression by proving lower bounds on oblivious sketching matrices for AMM. The sketch size lower bounds for AMM may be of independent interest.

----

## [462] Flashlight: Enabling Innovation in Tools for Machine Learning

**Authors**: *Jacob D. Kahn, Vineel Pratap, Tatiana Likhomanenko, Qiantong Xu, Awni Y. Hannun, Jeff Cai, Paden Tomasello, Ann Lee, Edouard Grave, Gilad Avidov, Benoit Steiner, Vitaliy Liptchinsky, Gabriel Synnaeve, Ronan Collobert*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kahn22a.html](https://proceedings.mlr.press/v162/kahn22a.html)

**Abstract**:

As the computational requirements for machine learning systems and the size and complexity of machine learning frameworks increases, essential framework innovation has become challenging. While computational needs have driven recent compiler, networking, and hardware advancements, utilization of those advancements by machine learning tools is occurring at a slower pace. This is in part due to the difficulties involved in prototyping new computational paradigms with existing frameworks. Large frameworks prioritize machine learning researchers and practitioners as end users and pay comparatively little attention to systems researchers who can push frameworks forward — we argue that both are equally important stakeholders. We introduce Flashlight, an open-source library built to spur innovation in machine learning tools and systems by prioritizing open, modular, customizable internals and state-of-the-art, research-ready models and training setups across a variety of domains. Flashlight allows systems researchers to rapidly prototype and experiment with novel ideas in machine learning computation and has low overhead, competing with and often outperforming other popular machine learning frameworks. We see Flashlight as a tool enabling research that can benefit widely used libraries downstream and bring machine learning and systems researchers closer together.

----

## [463] Learning-based Optimisation of Particle Accelerators Under Partial Observability Without Real-World Training

**Authors**: *Jan Kaiser, Oliver Stein, Annika Eichler*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kaiser22a.html](https://proceedings.mlr.press/v162/kaiser22a.html)

**Abstract**:

In recent work, it has been shown that reinforcement learning (RL) is capable of solving a variety of problems at sometimes super-human performance levels. But despite continued advances in the field, applying RL to complex real-world control and optimisation problems has proven difficult. In this contribution, we demonstrate how to successfully apply RL to the optimisation of a highly complex real-world machine {–} specifically a linear particle accelerator {–} in an only partially observable setting and without requiring training on the real machine. Our method outperforms conventional optimisation algorithms in both the achieved result and time taken as well as already achieving close to human-level performance. We expect that such automation of machine optimisation will push the limits of operability, increase machine availability and lead to a paradigm shift in how such machines are operated, ultimately facilitating advances in a variety of fields, such as science and medicine among many others.

----

## [464] Stochastic Deep Networks with Linear Competing Units for Model-Agnostic Meta-Learning

**Authors**: *Konstantinos Kalais, Sotirios Chatzis*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kalais22a.html](https://proceedings.mlr.press/v162/kalais22a.html)

**Abstract**:

This work addresses meta-learning (ML) by considering deep networks with stochastic local winner-takes-all (LWTA) activations. This type of network units results in sparse representations from each model layer, as the units are organized into blocks where only one unit generates a non-zero output. The main operating principle of the introduced units rely on stochastic principles, as the network performs posterior sampling over competing units to select the winner. Therefore, the proposed networks are explicitly designed to extract input data representations of sparse stochastic nature, as opposed to the currently standard deterministic representation paradigm. Our approach produces state-of-the-art predictive accuracy on few-shot image classification and regression experiments, as well as reduced predictive error on an active learning setting; these improvements come with an immensely reduced computational cost. Code is available at: https://github.com/Kkalais/StochLWTA-ML

----

## [465] Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning

**Authors**: *Nathan Kallus, Xiaojie Mao, Kaiwen Wang, Zhengyuan Zhou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kallus22a.html](https://proceedings.mlr.press/v162/kallus22a.html)

**Abstract**:

Off-policy evaluation and learning (OPE/L) use offline observational data to make better decisions, which is crucial in applications where online experimentation is limited. However, depending entirely on logged data, OPE/L is sensitive to environment distribution shifts — discrepancies between the data-generating environment and that where policies are deployed. Si et al., (2020) proposed distributionally robust OPE/L (DROPE/L) to address this, but the proposal relies on inverse-propensity weighting, whose estimation error and regret will deteriorate if propensities are nonparametrically estimated and whose variance is suboptimal even if not. For standard, non-robust, OPE/L, this is solved by doubly robust (DR) methods, but they do not naturally extend to the more complex DROPE/L, which involves a worst-case expectation. In this paper, we propose the first DR algorithms for DROPE/L with KL-divergence uncertainty sets. For evaluation, we propose Localized Doubly Robust DROPE (LDR$^2$OPE) and show that it achieves semiparametric efficiency under weak product rates conditions. Thanks to a localization technique, LDR$^2$OPE only requires fitting a small number of regressions, just like DR methods for standard OPE. For learning, we propose Continuum Doubly Robust DROPL (CDR$^2$OPL) and show that, under a product rate condition involving a continuum of regressions, it enjoys a fast regret rate of $O(N^{-1/2})$ even when unknown propensities are nonparametrically estimated. We empirically validate our algorithms in simulations and further extend our results to general $f$-divergence uncertainty sets.

----

## [466] Improved Rates for Differentially Private Stochastic Convex Optimization with Heavy-Tailed Data

**Authors**: *Gautam Kamath, Xingtu Liu, Huanyu Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kamath22a.html](https://proceedings.mlr.press/v162/kamath22a.html)

**Abstract**:

We study stochastic convex optimization with heavy-tailed data under the constraint of differential privacy (DP). Most prior work on this problem is restricted to the case where the loss function is Lipschitz. Instead, as introduced by Wang, Xiao, Devadas, and Xu \cite{WangXDX20}, we study general convex loss functions with the assumption that the distribution of gradients has bounded $k$-th moments. We provide improved upper bounds on the excess population risk under concentrated DP for convex and strongly convex loss functions. Along the way, we derive new algorithms for private mean estimation of heavy-tailed distributions, under both pure and concentrated DP. Finally, we prove nearly-matching lower bounds for private stochastic convex optimization with strongly convex losses and mean estimation, showing new separations between pure and concentrated DP.

----

## [467] Comprehensive Analysis of Negative Sampling in Knowledge Graph Representation Learning

**Authors**: *Hidetaka Kamigaito, Katsuhiko Hayashi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kamigaito22a.html](https://proceedings.mlr.press/v162/kamigaito22a.html)

**Abstract**:

Negative sampling (NS) loss plays an important role in learning knowledge graph embedding (KGE) to handle a huge number of entities. However, the performance of KGE degrades without hyperparameters such as the margin term and number of negative samples in NS loss being appropriately selected. Currently, empirical hyperparameter tuning addresses this problem at the cost of computational time. To solve this problem, we theoretically analyzed NS loss to assist hyperparameter tuning and understand the better use of the NS loss in KGE learning. Our theoretical analysis showed that scoring methods with restricted value ranges, such as TransE and RotatE, require appropriate adjustment of the margin term or the number of negative samples different from those without restricted value ranges, such as RESCAL, ComplEx, and DistMult. We also propose subsampling methods specialized for the NS loss in KGE studied from a theoretical aspect. Our empirical analysis on the FB15k-237, WN18RR, and YAGO3-10 datasets showed that the results of actually trained models agree with our theoretical findings.

----

## [468] Matching Learned Causal Effects of Neural Networks with Domain Priors

**Authors**: *Sai Srinivas Kancheti, Abbavaram Gowtham Reddy, Vineeth N. Balasubramanian, Amit Sharma*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kancheti22a.html](https://proceedings.mlr.press/v162/kancheti22a.html)

**Abstract**:

A trained neural network can be interpreted as a structural causal model (SCM) that provides the effect of changing input variables on the model’s output. However, if training data contains both causal and correlational relationships, a model that optimizes prediction accuracy may not necessarily learn the true causal relationships between input and output variables. On the other hand, expert users often have prior knowledge of the causal relationship between certain input variables and output from domain knowledge. Therefore, we propose a regularization method that aligns the learned causal effects of a neural network with domain priors, including both direct and total causal effects. We show that this approach can generalize to different kinds of domain priors, including monotonicity of causal effect of an input variable on output or zero causal effect of a variable on output for purposes of fairness. Our experiments on twelve benchmark datasets show its utility in regularizing a neural network model to maintain desired causal effects, without compromising on accuracy. Importantly, we also show that a model thus trained is robust and gets improved accuracy on noisy inputs.

----

## [469] Deduplicating Training Data Mitigates Privacy Risks in Language Models

**Authors**: *Nikhil Kandpal, Eric Wallace, Colin Raffel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kandpal22a.html](https://proceedings.mlr.press/v162/kandpal22a.html)

**Abstract**:

Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence’s count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated 1000x more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.

----

## [470] Lyapunov Density Models: Constraining Distribution Shift in Learning-Based Control

**Authors**: *Katie Kang, Paula Gradu, Jason J. Choi, Michael Janner, Claire J. Tomlin, Sergey Levine*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kang22a.html](https://proceedings.mlr.press/v162/kang22a.html)

**Abstract**:

Learned models and policies can generalize effectively when evaluated within the distribution of the training data, but can produce unpredictable and erroneous outputs on out-of-distribution inputs. In order to avoid distribution shift when deploying learning-based control algorithms, we seek a mechanism to constrain the agent to states and actions that resemble those that the method was trained on. In control theory, Lyapunov stability and control-invariant sets allow us to make guarantees about controllers that stabilize the system around specific states, while in machine learning, density models allow us to estimate the training data distribution. Can we combine these two concepts, producing learning-based control algorithms that constrain the system to in-distribution states using only in-distribution actions? In this paper, we propose to do this by combining concepts from Lyapunov stability and density estimation, introducing Lyapunov density models: a generalization of control Lyapunov functions and density models that provides guarantees about an agent’s ability to stay in-distribution over its entire trajectory.

----

## [471] Forget-free Continual Learning with Winning Subnetworks

**Authors**: *Haeyong Kang, Rusty John Lloyd Mina, Sultan Rizky Hikmawan Madjid, Jaehong Yoon, Mark Hasegawa-Johnson, Sung Ju Hwang, Chang D. Yoo*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kang22b.html](https://proceedings.mlr.press/v162/kang22b.html)

**Abstract**:

Inspired by Lottery Ticket Hypothesis that competitive subnetworks exist within a dense network, we propose a continual learning method referred to as Winning SubNetworks (WSN), which sequentially learns and selects an optimal subnetwork for each task. Specifically, WSN jointly learns the model weights and task-adaptive binary masks pertaining to subnetworks associated with each task whilst attempting to select a small set of weights to be activated (winning ticket) by reusing weights of the prior subnetworks. The proposed method is inherently immune to catastrophic forgetting as each selected subnetwork model does not infringe upon other subnetworks. Binary masks spawned per winning ticket are encoded into one N-bit binary digit mask, then compressed using Huffman coding for a sub-linear increase in network capacity with respect to the number of tasks.

----

## [472] Differentially Private Approximate Quantiles

**Authors**: *Haim Kaplan, Shachar Schnapp, Uri Stemmer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kaplan22a.html](https://proceedings.mlr.press/v162/kaplan22a.html)

**Abstract**:

In this work we study the problem of differentially private (DP) quantiles, in which given dataset $X$ and quantiles $q_1, ..., q_m \in [0,1]$, we want to output $m$ quantile estimations which are as close as possible to the true quantiles and preserve DP. We describe a simple recursive DP algorithm, which we call Approximate Quantiles (AQ), for this task. We give a worst case upper bound on its error, and show that its error is much lower than of previous implementations on several different datasets. Furthermore, it gets this low error while running time two orders of magnitude faster that the best previous implementation.

----

## [473] Simultaneous Graph Signal Clustering and Graph Learning

**Authors**: *Abdullah Karaaslanli, Selin Aviyente*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/karaaslanli22a.html](https://proceedings.mlr.press/v162/karaaslanli22a.html)

**Abstract**:

Graph learning (GL) aims to infer the topology of an unknown graph from a set of observations on its nodes, i.e., graph signals. While most of the existing GL approaches focus on homogeneous datasets, in many real world applications, data is heterogeneous, where graph signals are clustered and each cluster is associated with a different graph. In this paper, we address the problem of learning multiple graphs from heterogeneous data by formulating an optimization problem for joint graph signal clustering and graph topology inference. In particular, our approach extends spectral clustering by partitioning the graph signals not only based on their pairwise similarities but also their smoothness with respect to the graphs associated with the clusters. The proposed method also learns the representative graph for each cluster using the smoothness of the graph signals with respect to the graph topology. The resulting optimization problem is solved with an efficient block-coordinate descent algorithm and results on simulated and real data indicate the effectiveness of the proposed method.

----

## [474] Composing Partial Differential Equations with Physics-Aware Neural Networks

**Authors**: *Matthias Karlbauer, Timothy Praditia, Sebastian Otte, Sergey Oladyshkin, Wolfgang Nowak, Martin V. Butz*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/karlbauer22a.html](https://proceedings.mlr.press/v162/karlbauer22a.html)

**Abstract**:

We introduce a compositional physics-aware FInite volume Neural Network (FINN) for learning spatiotemporal advection-diffusion processes. FINN implements a new way of combining the learning abilities of artificial neural networks with physical and structural knowledge from numerical simulation by modeling the constituents of partial differential equations (PDEs) in a compositional manner. Results on both one- and two-dimensional PDEs (Burgers’, diffusion-sorption, diffusion-reaction, Allen{–}Cahn) demonstrate FINN’s superior modeling accuracy and excellent out-of-distribution generalization ability beyond initial and boundary conditions. With only one tenth of the number of parameters on average, FINN outperforms pure machine learning and other state-of-the-art physics-aware models in all cases{—}often even by multiple orders of magnitude. Moreover, FINN outperforms a calibrated physical model when approximating sparse real-world data in a diffusion-sorption scenario, confirming its generalization abilities and showing explanatory potential by revealing the unknown retardation factor of the observed process.

----

## [475] Meta-Learning Hypothesis Spaces for Sequential Decision-making

**Authors**: *Parnian Kassraie, Jonas Rothfuss, Andreas Krause*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kassraie22a.html](https://proceedings.mlr.press/v162/kassraie22a.html)

**Abstract**:

Obtaining reliable, adaptive confidence sets for prediction functions (hypotheses) is a central challenge in sequential decision-making tasks, such as bandits and model-based reinforcement learning. These confidence sets typically rely on prior assumptions on the hypothesis space, e.g., the known kernel of a Reproducing Kernel Hilbert Space (RKHS). Hand-designing such kernels is error prone, and misspecification may lead to poor or unsafe performance. In this work, we propose to meta-learn a kernel from offline data (Meta-KeL). For the case where the unknown kernel is a combination of known base kernels, we develop an estimator based on structured sparsity. Under mild conditions, we guarantee that our estimated RKHS yields valid confidence sets that, with increasing amounts of offline data, become as tight as those given the true unknown kernel. We demonstrate our approach on the kernelized bandits problem (a.k.a. Bayesian optimization), where we establish regret bounds competitive with those given the true kernel. We also empirically evaluate the effectiveness of our approach on a Bayesian optimization task.

----

## [476] FOCUS: Familiar Objects in Common and Uncommon Settings

**Authors**: *Priyatham Kattakinda, Soheil Feizi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kattakinda22a.html](https://proceedings.mlr.press/v162/kattakinda22a.html)

**Abstract**:

Standard training datasets for deep learning often do not contain objects in uncommon and rare settings (e.g., “a plane on water”, “a car in snowy weather”). This can cause models trained on these datasets to incorrectly predict objects that are typical for the context in the image, rather than identifying the objects that are actually present. In this paper, we introduce FOCUS (Familiar Objects in Common and Uncommon Settings), a dataset for stress-testing the generalization power of deep image classifiers. By leveraging the power of modern search engines, we deliberately gather data containing objects in common and uncommon settings; in a wide range of locations, weather conditions, and time of day. We present a detailed analysis of the performance of various popular image classifiers on our dataset and demonstrate a clear drop in accuracy when classifying images in uncommon settings. We also show that finetuning a model on our dataset drastically improves its ability to focus on the object of interest leading to better generalization. Lastly, we leverage FOCUS to machine annotate additional visual attributes for the entirety of ImageNet. We believe that our dataset will aid researchers in understanding the inability of deep models to generalize well to uncommon settings and drive future work on improving their distributional robustness.

----

## [477] Training OOD Detectors in their Natural Habitats

**Authors**: *Julian Katz-Samuels, Julia B. Nakhleh, Robert D. Nowak, Yixuan Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/katz-samuels22a.html](https://proceedings.mlr.press/v162/katz-samuels22a.html)

**Abstract**:

Out-of-distribution (OOD) detection is important for machine learning models deployed in the wild. Recent methods use auxiliary outlier data to regularize the model for improved OOD detection. However, these approaches make a strong distributional assumption that the auxiliary outlier data is completely separable from the in-distribution (ID) data. In this paper, we propose a novel framework that leverages wild mixture data—that naturally consists of both ID and OOD samples. Such wild data is abundant and arises freely upon deploying a machine learning classifier in their natural habitats. Our key idea is to formulate a constrained optimization problem and to show how to tractably solve it. Our learning objective maximizes the OOD detection rate, subject to constraints on the classification error of ID data and on the OOD error rate of ID examples. We extensively evaluate our approach on common OOD detection tasks and demonstrate superior performance. Code is available at https://github.com/jkatzsam/woods_ood.

----

## [478] Robustness Implies Generalization via Data-Dependent Generalization Bounds

**Authors**: *Kenji Kawaguchi, Zhun Deng, Kyle Luh, Jiaoyang Huang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kawaguchi22a.html](https://proceedings.mlr.press/v162/kawaguchi22a.html)

**Abstract**:

This paper proves that robustness implies generalization via data-dependent generalization bounds. As a result, robustness and generalization are shown to be connected closely in a data-dependent manner. Our bounds improve previous bounds in two directions, to solve an open problem that has seen little development since 2010. The first is to reduce the dependence on the covering number. The second is to remove the dependence on the hypothesis space. We present several examples, including ones for lasso and deep learning, in which our bounds are provably preferable. The experiments on real-world data and theoretical models demonstrate near-exponential improvements in various situations. To achieve these improvements, we do not require additional assumptions on the unknown distribution; instead, we only incorporate an observable and computable property of the training samples. A key technical innovation is an improved concentration bound for multinomial random variables that is of independent interest beyond robustness and generalization.

----

## [479] Generating Distributional Adversarial Examples to Evade Statistical Detectors

**Authors**: *Yigitcan Kaya, Muhammad Bilal Zafar, Sergül Aydöre, Nathalie Rauschmayr, Krishnaram Kenthapadi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kaya22a.html](https://proceedings.mlr.press/v162/kaya22a.html)

**Abstract**:

Deep neural networks (DNNs) are known to be highly vulnerable to adversarial examples (AEs) that include malicious perturbations. Assumptions about the statistical differences between natural and adversarial inputs are commonplace in many detection techniques. As a best practice, AE detectors are evaluated against ’adaptive’ attackers who actively perturb their inputs to avoid detection. Due to the difficulties in designing adaptive attacks, however, recent work suggests that most detectors have incomplete evaluation. We aim to fill this gap by designing a generic adaptive attack against detectors: the ’statistical indistinguishability attack’ (SIA). SIA optimizes a novel objective to craft adversarial examples (AEs) that follow the same distribution as the natural inputs with respect to DNN representations. Our objective targets all DNN layers simultaneously as we show that AEs being indistinguishable at one layer might fail to be so at other layers. SIA is formulated around evading distributional detectors that inspect a set of AEs as a whole and is also effective against four individual AE detectors, two dataset shift detectors, and an out-of-distribution sample detector, curated from published works. This suggests that SIA can be a reliable tool for evaluating the security of a range of detectors.

----

## [480] Secure Quantized Training for Deep Learning

**Authors**: *Marcel Keller, Ke Sun*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/keller22a.html](https://proceedings.mlr.press/v162/keller22a.html)

**Abstract**:

We implement training of neural networks in secure multi-party computation (MPC) using quantization commonly used in said setting. We are the first to present an MNIST classifier purely trained in MPC that comes within 0.2 percent of the accuracy of the same convolutional neural network trained via plaintext computation. More concretely, we have trained a network with two convolutional and two dense layers to 99.2% accuracy in 3.5 hours (under one hour for 99% accuracy). We have also implemented AlexNet for CIFAR-10, which converges in a few hours. We develop novel protocols for exponentiation and inverse square root. Finally, we present experiments in a range of MPC security models for up to ten parties, both with honest and dishonest majority as well as semi-honest and malicious security.

----

## [481] A Convergent and Dimension-Independent Min-Max Optimization Algorithm

**Authors**: *Vijay Keswani, Oren Mangoubi, Sushant Sachdeva, Nisheeth K. Vishnoi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/keswani22a.html](https://proceedings.mlr.press/v162/keswani22a.html)

**Abstract**:

We study a variant of a recently introduced min-max optimization framework where the max-player is constrained to update its parameters in a greedy manner until it reaches a first-order stationary point. Our equilibrium definition for this framework depends on a proposal distribution which the min-player uses to choose directions in which to update its parameters. We show that, given a smooth and bounded nonconvex-nonconcave objective function, access to any proposal distribution for the min-player’s updates, and stochastic gradient oracle for the max-player, our algorithm converges to the aforementioned approximate local equilibrium in a number of iterations that does not depend on the dimension. The equilibrium point found by our algorithm depends on the proposal distribution, and when applying our algorithm to train GANs we choose the proposal distribution to be a distribution of stochastic gradients. We empirically evaluate our algorithm on challenging nonconvex-nonconcave test-functions and loss functions arising in GAN training. Our algorithm converges on these test functions and, when used to train GANs, trains stably on synthetic and real-world datasets and avoids mode collapse.

----

## [482] Neural Network Poisson Models for Behavioural and Neural Spike Train Data

**Authors**: *Moein Khajehnejad, Forough Habibollahi, Richard Nock, Ehsan Arabzadeh, Peter Dayan, Amir Dezfouli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/khajehnejad22a.html](https://proceedings.mlr.press/v162/khajehnejad22a.html)

**Abstract**:

One of the most important and challenging application areas for complex machine learning methods is to predict, characterize and model rich, multi-dimensional, neural data. Recent advances in neural recording techniques have made it possible to monitor the activity of a large number of neurons across different brain regions as animals perform behavioural tasks. This poses the critical challenge of establishing links between neural activity at a microscopic scale, which might for instance represent sensory input, and at a macroscopic scale, which then generates behaviour. Predominant modeling methods apply rather disjoint techniques to these scales; by contrast, we suggest an end-to-end model which exploits recent developments of flexible, but tractable, neural network point-process models to characterize dependencies between stimuli, actions, and neural data. We apply this model to a public dataset collected using Neuropixel probes in mice performing a visually-guided behavioural task as well as a synthetic dataset produced from a hierarchical network model with reciprocally connected sensory and integration circuits intended to characterize animal behaviour in a fixed-duration motion discrimination task. We show that our model outperforms previous approaches and contributes novel insights into the relationships between neural activity and behaviour.

----

## [483] Federated Reinforcement Learning: Linear Speedup Under Markovian Sampling

**Authors**: *Sajad Khodadadian, Pranay Sharma, Gauri Joshi, Siva Theja Maguluri*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/khodadadian22a.html](https://proceedings.mlr.press/v162/khodadadian22a.html)

**Abstract**:

Since reinforcement learning algorithms are notoriously data-intensive, the task of sampling observations from the environment is usually split across multiple agents. However, transferring these observations from the agents to a central location can be prohibitively expensive in terms of the communication cost, and it can also compromise the privacy of each agent’s local behavior policy. In this paper, we consider a federated reinforcement learning framework where multiple agents collaboratively learn a global model, without sharing their individual data and policies. Each agent maintains a local copy of the model and updates it using locally sampled data. Although having N agents enables the sampling of N times more data, it is not clear if it leads to proportional convergence speedup. We propose federated versions of on-policy TD, off-policy TD and Q-learning, and analyze their convergence. For all these algorithms, to the best of our knowledge, we are the first to consider Markovian noise and multiple local updates, and prove a linear convergence speedup with respect to the number of agents. To obtain these results, we show that federated TD and Q-learning are special cases of a general framework for federated stochastic approximation with Markovian noise, and we leverage this framework to provide a unified convergence analysis that applies to all the algorithms.

----

## [484] Multi-Level Branched Regularization for Federated Learning

**Authors**: *Jinkyu Kim, Geeho Kim, Bohyung Han*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22a.html](https://proceedings.mlr.press/v162/kim22a.html)

**Abstract**:

A critical challenge of federated learning is data heterogeneity and imbalance across clients, which leads to inconsistency between local networks and unstable convergence of global models. To alleviate the limitations, we propose a novel architectural regularization technique that constructs multiple auxiliary branches in each local model by grafting local and global subnetworks at several different levels and that learns the representations of the main pathway in the local model congruent to the auxiliary hybrid pathways via online knowledge distillation. The proposed technique is effective to robustify the global model even in the non-iid setting and is applicable to various federated learning frameworks conveniently without incurring extra communication costs. We perform comprehensive empirical studies and demonstrate remarkable performance gains in terms of accuracy and efficiency compared to existing methods. The source code is available at our project page.

----

## [485] Learning fair representation with a parametric integral probability metric

**Authors**: *Dongha Kim, Kunwoong Kim, Insung Kong, Ilsang Ohn, Yongdai Kim*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22b.html](https://proceedings.mlr.press/v162/kim22b.html)

**Abstract**:

As they have a vital effect on social decision-making, AI algorithms should be not only accurate but also fair. Among various algorithms for fairness AI, learning fair representation (LFR), whose goal is to find a fair representation with respect to sensitive variables such as gender and race, has received much attention. For LFR, the adversarial training scheme is popularly employed as is done in the generative adversarial network type algorithms. The choice of a discriminator, however, is done heuristically without justification. In this paper, we propose a new adversarial training scheme for LFR, where the integral probability metric (IPM) with a specific parametric family of discriminators is used. The most notable result of the proposed LFR algorithm is its theoretical guarantee about the fairness of the final prediction model, which has not been considered yet. That is, we derive theoretical relations between the fairness of representation and the fairness of the prediction model built on the top of the representation (i.e., using the representation as the input). Moreover, by numerical experiments, we show that our proposed LFR algorithm is computationally lighter and more stable, and the final prediction model is competitive or superior to other LFR algorithms using more complex discriminators.

----

## [486] Dataset Condensation via Efficient Synthetic-Data Parameterization

**Authors**: *Jang-Hyun Kim, Jinuk Kim, Seong Joon Oh, Sangdoo Yun, Hwanjun Song, Joonhyun Jeong, Jung-Woo Ha, Hyun Oh Song*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22c.html](https://proceedings.mlr.press/v162/kim22c.html)

**Abstract**:

The great success of machine learning with massive amounts of data comes at a price of huge computation costs and storage for training and tuning. Recent studies on dataset condensation attempt to reduce the dependence on such massive data by synthesizing a compact training dataset. However, the existing approaches have fundamental limitations in optimization due to the limited representability of synthetic datasets without considering any data regularity characteristics. To this end, we propose a novel condensation framework that generates multiple synthetic data with a limited storage budget via efficient parameterization considering data regularity. We further analyze the shortcomings of the existing gradient matching-based condensation methods and develop an effective optimization technique for improving the condensation of training data information. We propose a unified algorithm that drastically improves the quality of condensed data against the current state-of-the-art on CIFAR-10, ImageNet, and Speech Commands.

----

## [487] Guided-TTS: A Diffusion Model for Text-to-Speech via Classifier Guidance

**Authors**: *Heeseung Kim, Sungwon Kim, Sungroh Yoon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22d.html](https://proceedings.mlr.press/v162/kim22d.html)

**Abstract**:

We propose Guided-TTS, a high-quality text-to-speech (TTS) model that does not require any transcript of target speaker using classifier guidance. Guided-TTS combines an unconditional diffusion probabilistic model with a separately trained phoneme classifier for classifier guidance. Our unconditional diffusion model learns to generate speech without any context from untranscribed speech data. For TTS synthesis, we guide the generative process of the diffusion model with a phoneme classifier trained on a large-scale speech recognition dataset. We present a norm-based scaling method that reduces the pronunciation errors of classifier guidance in Guided-TTS. We show that Guided-TTS achieves a performance comparable to that of the state-of-the-art TTS model, Grad-TTS, without any transcript for LJSpeech. We further demonstrate that Guided-TTS performs well on diverse datasets including a long-form untranscribed dataset.

----

## [488] Variational On-the-Fly Personalization

**Authors**: *Jangho Kim, Juntae Lee, Simyung Chang, Nojun Kwak*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22e.html](https://proceedings.mlr.press/v162/kim22e.html)

**Abstract**:

With the development of deep learning (DL) technologies, the demand for DL-based services on personal devices, such as mobile phones, also increases rapidly. In this paper, we propose a novel personalization method, Variational On-the-Fly Personalization. Compared to the conventional personalization methods that require additional fine-tuning with personal data, the proposed method only requires forwarding a handful of personal data on-the-fly. Assuming even a single personal data can convey the characteristics of a target person, we develop the variational hyper-personalizer to capture the weight distribution of layers that fits the target person. In the testing phase, the hyper-personalizer estimates the model’s weights on-the-fly based on personality by forwarding only a small amount of (even a single) personal enrollment data. Hence, the proposed method can perform the personalization without any training software platform and additional cost in the edge device. In experiments, we show our approach can effectively generate reliable personalized models via forwarding (not back-propagating) a handful of samples.

----

## [489] Fisher SAM: Information Geometry and Sharpness Aware Minimisation

**Authors**: *Minyoung Kim, Da Li, Shell Xu Hu, Timothy M. Hospedales*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22f.html](https://proceedings.mlr.press/v162/kim22f.html)

**Abstract**:

Recent sharpness-aware minimisation (SAM) is known to find flat minima which is beneficial for better generalisation with improved robustness. SAM essentially modifies the loss function by the maximum loss value within the small neighborhood around the current iterate. However, it uses the Euclidean ball to define the neighborhood, which can be less accurate since loss functions for neural networks are typically defined over probability distributions (e.g., class predictive probabilities), rendering the parameter space no more Euclidean. In this paper we consider the information geometry of the model parameter space when defining the neighborhood, namely replacing SAM’s Euclidean balls with ellipsoids induced by the Fisher information. Our approach, dubbed Fisher SAM, defines more accurate neighborhood structures that conform to the intrinsic metric of the underlying statistical manifold. For instance, SAM may probe the worst-case loss value at either a too nearby or inappropriately distant point due to the ignorance of the parameter space geometry, which is avoided by our Fisher SAM. Another recent Adaptive SAM approach that stretches/shrinks the Euclidean ball in accordance with the scales of the parameter magnitudes, might be dangerous, potentially destroying the neighborhood structure even severely. We demonstrate the improved performance of the proposed Fisher SAM on several benchmark datasets/tasks.

----

## [490] ViT-NeT: Interpretable Vision Transformers with Neural Tree Decoder

**Authors**: *Sangwon Kim, Jae-Yeal Nam, ByoungChul Ko*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22g.html](https://proceedings.mlr.press/v162/kim22g.html)

**Abstract**:

Vision transformers (ViTs), which have demonstrated a state-of-the-art performance in image classification, can also visualize global interpretations through attention-based contributions. However, the complexity of the model makes it difficult to interpret the decision-making process, and the ambiguity of the attention maps can cause incorrect correlations between image patches. In this study, we propose a new ViT neural tree decoder (ViT-NeT). A ViT acts as a backbone, and to solve its limitations, the output contextual image patches are applied to the proposed NeT. The NeT aims to accurately classify fine-grained objects with similar inter-class correlations and different intra-class correlations. In addition, it describes the decision-making process through a tree structure and prototype and enables a visual interpretation of the results. The proposed ViT-NeT is designed to not only improve the classification performance but also provide a human-friendly interpretation, which is effective in resolving the trade-off between performance and interpretability. We compared the performance of ViT-NeT with other state-of-art methods using widely used fine-grained visual categorization benchmark datasets and experimentally proved that the proposed method is superior in terms of the classification performance and interpretability. The code and models are publicly available at https://github.com/jumpsnack/ViT-NeT.

----

## [491] Sanity Simulations for Saliency Methods

**Authors**: *Joon Sik Kim, Gregory Plumb, Ameet Talwalkar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22h.html](https://proceedings.mlr.press/v162/kim22h.html)

**Abstract**:

Saliency methods are a popular class of feature attribution explanation methods that aim to capture a model’s predictive reasoning by identifying "important" pixels in an input image. However, the development and adoption of these methods are hindered by the lack of access to ground-truth model reasoning, which prevents accurate evaluation. In this work, we design a synthetic benchmarking framework, SMERF, that allows us to perform ground-truth-based evaluation while controlling the complexity of the model’s reasoning. Experimentally, SMERF reveals significant limitations in existing saliency methods and, as a result, represents a useful tool for the development of new saliency methods.

----

## [492] Soft Truncation: A Universal Training Technique of Score-based Diffusion Model for High Precision Score Estimation

**Authors**: *Dongjun Kim, Seungjae Shin, Kyungwoo Song, Wanmo Kang, Il-Chul Moon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22i.html](https://proceedings.mlr.press/v162/kim22i.html)

**Abstract**:

Recent advances in diffusion models bring state-of-the-art performance on image generation tasks. However, empirical results from previous research in diffusion models imply an inverse correlation between density estimation and sample generation performances. This paper investigates with sufficient empirical evidence that such inverse correlation happens because density estimation is significantly contributed by small diffusion time, whereas sample generation mainly depends on large diffusion time. However, training a score network well across the entire diffusion time is demanding because the loss scale is significantly imbalanced at each diffusion time. For successful training, therefore, we introduce Soft Truncation, a universally applicable training technique for diffusion models, that softens the fixed and static truncation hyperparameter into a random variable. In experiments, Soft Truncation achieves state-of-the-art performance on CIFAR-10, CelebA, CelebA-HQ $256\times 256$, and STL-10 datasets.

----

## [493] Rotting Infinitely Many-Armed Bandits

**Authors**: *Jung-Hun Kim, Milan Vojnovic, Se-Young Yun*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22j.html](https://proceedings.mlr.press/v162/kim22j.html)

**Abstract**:

We consider the infinitely many-armed bandit problem with rotting rewards, where the mean reward of an arm decreases at each pull of the arm according to an arbitrary trend with maximum rotting rate $\varrho=o(1)$. We show that this learning problem has an $\Omega(\max\{\varrho^{1/3}T, \sqrt{T}\})$ worst-case regret lower bound where $T$ is the time horizon. We show that a matching upper bound $\tilde{O}(\max\{\varrho^{1/3}T, \sqrt{T}\})$, up to a poly-logarithmic factor, can be achieved by an algorithm that uses a UCB index for each arm and a threshold value to decide whether to continue pulling an arm or remove the arm from further consideration, when the algorithm knows the value of the maximum rotting rate $\varrho$. We also show that an $\tilde{O}(\max\{\varrho^{1/3}T, T^{3/4}\})$ regret upper bound can be achieved by an algorithm that does not know the value of $\varrho$, by using an adaptive UCB index along with an adaptive threshold value.

----

## [494] Accelerated Gradient Methods for Geodesically Convex Optimization: Tractable Algorithms and Convergence Analysis

**Authors**: *Jungbin Kim, Insoon Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kim22k.html](https://proceedings.mlr.press/v162/kim22k.html)

**Abstract**:

We propose computationally tractable accelerated first-order methods for Riemannian optimization, extending the Nesterov accelerated gradient (NAG) method. For both geodesically convex and geodesically strongly convex objective functions, our algorithms are shown to have the same iteration complexities as those for the NAG method on Euclidean spaces, under only standard assumptions. To the best of our knowledge, the proposed scheme is the first fully accelerated method for geodesically convex optimization problems. Our convergence analysis makes use of novel metric distortion lemmas as well as carefully designed potential functions. A connection with the continuous-time dynamics for modeling Riemannian acceleration in (Alimisis et al., 2020) is also identified by letting the stepsize tend to zero. We validate our theoretical results through numerical experiments.

----

## [495] Generalizing to New Physical Systems via Context-Informed Dynamics Model

**Authors**: *Matthieu Kirchmeyer, Yuan Yin, Jérémie Donà, Nicolas Baskiotis, Alain Rakotomamonjy, Patrick Gallinari*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kirchmeyer22a.html](https://proceedings.mlr.press/v162/kirchmeyer22a.html)

**Abstract**:

Data-driven approaches to modeling physical systems fail to generalize to unseen systems that share the same general dynamics with the learning domain, but correspond to different physical contexts. We propose a new framework for this key problem, context-informed dynamics adaptation (CoDA), which takes into account the distributional shift across systems for fast and efficient adaptation to new dynamics. CoDA leverages multiple environments, each associated to a different dynamic, and learns to condition the dynamics model on contextual parameters, specific to each environment. The conditioning is performed via a hypernetwork, learned jointly with a context vector from observed data. The proposed formulation constrains the search hypothesis space for fast adaptation and better generalization across environments with few samples. We theoretically motivate our approach and show state-of-the-art generalization results on a set of nonlinear dynamics, representative of a variety of application domains. We also show, on these systems, that new system parameters can be inferred from context vectors with minimal supervision.

----

## [496] SoQal: Selective Oracle Questioning for Consistency Based Active Learning of Cardiac Signals

**Authors**: *Dani Kiyasseh, Tingting Zhu, David A. Clifton*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kiyasseh22a.html](https://proceedings.mlr.press/v162/kiyasseh22a.html)

**Abstract**:

Clinical settings are often characterized by abundant unlabelled data and limited labelled data. This is typically driven by the high burden placed on oracles (e.g., physicians) to provide annotations. One way to mitigate this burden is via active learning (AL) which involves the (a) acquisition and (b) annotation of informative unlabelled instances. Whereas previous work addresses either one of these elements independently, we propose an AL framework that addresses both. For acquisition, we propose Bayesian Active Learning by Consistency (BALC), a sub-framework which perturbs both instances and network parameters and quantifies changes in the network output probability distribution. For annotation, we propose SoQal, a sub-framework that dynamically determines whether, for each acquired unlabelled instance, to request a label from an oracle or to pseudo-label it instead. We show that BALC can outperform start-of-the-art acquisition functions such as BALD, and SoQal outperforms baseline methods even in the presence of a noisy oracle.

----

## [497] Curriculum Reinforcement Learning via Constrained Optimal Transport

**Authors**: *Pascal Klink, Haoyi Yang, Carlo D'Eramo, Jan Peters, Joni Pajarinen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/klink22a.html](https://proceedings.mlr.press/v162/klink22a.html)

**Abstract**:

Curriculum reinforcement learning (CRL) allows solving complex tasks by generating a tailored sequence of learning tasks, starting from easy ones and subsequently increasing their difficulty. Although the potential of curricula in RL has been clearly shown in a variety of works, it is less clear how to generate them for a given learning environment, resulting in a variety of methods aiming to automate this task. In this work, we focus on the idea of framing curricula as interpolations between task distributions, which has previously been shown to be a viable approach to CRL. Identifying key issues of existing methods, we frame the generation of a curriculum as a constrained optimal transport problem between task distributions. Benchmarks show that this way of curriculum generation can improve upon existing CRL methods, yielding high performance in a variety of tasks with different characteristics.

----

## [498] Exploiting Redundancy: Separable Group Convolutional Networks on Lie Groups

**Authors**: *David M. Knigge, David W. Romero, Erik J. Bekkers*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/knigge22a.html](https://proceedings.mlr.press/v162/knigge22a.html)

**Abstract**:

Group convolutional neural networks (G-CNNs) have been shown to increase parameter efficiency and model accuracy by incorporating geometric inductive biases. In this work, we investigate the properties of representations learned by regular G-CNNs, and show considerable parameter redundancy in group convolution kernels. This finding motivates further weight-tying by sharing convolution kernels over subgroups. To this end, we introduce convolution kernels that are separable over the subgroup and channel dimensions. In order to obtain equivariance to arbitrary affine Lie groups we provide a continuous parameterisation of separable convolution kernels. We evaluate our approach across several vision datasets, and show that our weight sharing leads to improved performance and computational efficiency. In many settings, separable G-CNNs outperform their non-separable counterpart, while only using a fraction of their training time. In addition, thanks to the increase in computational efficiency, we are able to implement G-CNNs equivariant to the $\mathrm{Sim(2)}$ group; the group of dilations, rotations and translations of the plane. $\mathrm{Sim(2)}$-equivariance further improves performance on all tasks considered, and achieves state-of-the-art performance on rotated MNIST.

----

## [499] Revisiting Contrastive Learning through the Lens of Neighborhood Component Analysis: an Integrated Framework

**Authors**: *Ching-Yun Ko, Jeet Mohapatra, Sijia Liu, Pin-Yu Chen, Luca Daniel, Lily Weng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ko22a.html](https://proceedings.mlr.press/v162/ko22a.html)

**Abstract**:

As a seminal tool in self-supervised representation learning, contrastive learning has gained unprecedented attention in recent years. In essence, contrastive learning aims to leverage pairs of positive and negative samples for representation learning, which relates to exploiting neighborhood information in a feature space. By investigating the connection between contrastive learning and neighborhood component analysis (NCA), we provide a novel stochastic nearest neighbor viewpoint of contrastive learning and subsequently propose a series of contrastive losses that outperform the existing ones. Under our proposed framework, we show a new methodology to design integrated contrastive losses that could simultaneously achieve good accuracy and robustness on downstream tasks. With the integrated framework, we achieve up to 6% improvement on the standard accuracy and 17% improvement on the robust accuracy.

----

## [500] Transfer Learning In Differential Privacy's Hybrid-Model

**Authors**: *Refael Kohen, Or Sheffet*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kohen22a.html](https://proceedings.mlr.press/v162/kohen22a.html)

**Abstract**:

The hybrid-model (Avent et al 2017) in Differential Privacy is a an augmentation of the local-model where in addition to $N$ local-agents we are assisted by one special agent who is in fact a curator holding the sensitive details of $n$ additional individuals. Here we study the problem of machine learning in the hybrid-model where the $n$ individuals in the curator’s dataset are drawn from a different distribution than the one of the general population (the local-agents). We give a general scheme – Subsample-Test-Reweigh – for this transfer learning problem, which reduces any curator-model learner to a learner in the hybrid-model using iterative subsampling and reweighing of the $n$ examples held by the curator based on a smooth variation (introduced by Bun et al 2020) of the Multiplicative-Weights algorithm. Our scheme has a sample complexity which relies on the $\chi^2$-divergence between the two distributions. We give worst-case analysis bounds on the sample complexity required for our private reduction. Aiming to reduce said sample complexity, we give two specific instances our sample complexity can be drastically reduced (one instance is analyzed mathematically, while the other - empirically) and pose several directions for follow-up work.

----

## [501] Markov Chain Monte Carlo for Continuous-Time Switching Dynamical Systems

**Authors**: *Lukas Köhs, Bastian Alt, Heinz Koeppl*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kohs22a.html](https://proceedings.mlr.press/v162/kohs22a.html)

**Abstract**:

Switching dynamical systems are an expressive model class for the analysis of time-series data. As in many fields within the natural and engineering sciences, the systems under study typically evolve continuously in time, it is natural to consider continuous-time model formulations consisting of switching stochastic differential equations governed by an underlying Markov jump process. Inference in these types of models is however notoriously difficult, and tractable computational schemes are rare. In this work, we propose a novel inference algorithm utilizing a Markov Chain Monte Carlo approach. The presented Gibbs sampler allows to efficiently obtain samples from the exact continuous-time posterior processes. Our framework naturally enables Bayesian parameter estimation, and we also include an estimate for the diffusion covariance, which is oftentimes assumed fixed in stochastic differential equations models. We evaluate our framework under the modeling assumption and compare it against an existing variational inference approach.

----

## [502] Partial disentanglement for domain adaptation

**Authors**: *Lingjing Kong, Shaoan Xie, Weiran Yao, Yujia Zheng, Guangyi Chen, Petar Stojanov, Victor Akinwande, Kun Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kong22a.html](https://proceedings.mlr.press/v162/kong22a.html)

**Abstract**:

Unsupervised domain adaptation is critical to many real-world applications where label information is unavailable in the target domain. In general, without further assumptions, the joint distribution of the features and the label is not identifiable in the target domain. To address this issue, we rely on a property of minimal changes of causal mechanisms across domains to minimize unnecessary influences of domain shift. To encode this property, we first formulate the data generating process using a latent variable model with two partitioned latent subspaces: invariant components whose distributions stay the same across domains, and sparse changing components that vary across domains. We further constrain the domain shift to have a restrictive influence on the changing components. Under mild conditions, we show that the latent variables are partially identifiable, from which it follows that the joint distribution of data and labels in the target domain is also identifiable. Given the theoretical insights, we propose a practical domain adaptation framework, called iMSDA. Extensive experimental results reveal that iMSDA outperforms state-of-the-art domain adaptation algorithms on benchmark datasets, demonstrating the effectiveness of our framework.

----

## [503] Simultaneously Learning Stochastic and Adversarial Bandits with General Graph Feedback

**Authors**: *Fang Kong, Yichi Zhou, Shuai Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kong22b.html](https://proceedings.mlr.press/v162/kong22b.html)

**Abstract**:

The problem of online learning with graph feedback has been extensively studied in the literature due to its generality and potential to model various learning tasks. Existing works mainly study the adversarial and stochastic feedback separately. If the prior knowledge of the feedback mechanism is unavailable or wrong, such specially designed algorithms could suffer great loss. To avoid this problem, \citet{erez2021towards} try to optimize for both environments. However, they assume the feedback graphs are undirected and each vertex has a self-loop, which compromises the generality of the framework and may not be satisfied in applications. With a general feedback graph, the observation of an arm may not be available when this arm is pulled, which makes the exploration more expensive and the algorithms more challenging to perform optimally in both environments. In this work, we overcome this difficulty by a new trade-off mechanism with a carefully-designed proportion for exploration and exploitation. We prove the proposed algorithm simultaneously achieves $\mathrm{poly} \log T$ regret in the stochastic setting and minimax-optimal regret of $\tilde{O}(T^{2/3})$ in the adversarial setting where $T$ is the horizon and $\tilde{O}$ hides parameters independent of $T$ as well as logarithmic terms. To our knowledge, this is the first best-of-both-worlds result for general feedback graphs.

----

## [504] Adaptive Data Analysis with Correlated Observations

**Authors**: *Aryeh Kontorovich, Menachem Sadigurschi, Uri Stemmer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kontorovich22a.html](https://proceedings.mlr.press/v162/kontorovich22a.html)

**Abstract**:

The vast majority of the work on adaptive data analysis focuses on the case where the samples in the dataset are independent. Several approaches and tools have been successfully applied in this context, such as differential privacy, max-information, compression arguments, and more. The situation is far less well-understood without the independence assumption. We embark on a systematic study of the possibilities of adaptive data analysis with correlated observations. First, we show that, in some cases, differential privacy guarantees generalization even when there are dependencies within the sample, which we quantify using a notion we call Gibbs-dependence. We complement this result with a tight negative example. % Second, we show that the connection between transcript-compression and adaptive data analysis can be extended to the non-iid setting.

----

## [505] Controlling Conditional Language Models without Catastrophic Forgetting

**Authors**: *Tomasz Korbak, Hady Elsahar, Germán Kruszewski, Marc Dymetman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/korbak22a.html](https://proceedings.mlr.press/v162/korbak22a.html)

**Abstract**:

Machine learning is shifting towards general-purpose pretrained generative models, trained in a self-supervised manner on large amounts of data, which can then be applied to solve a large number of tasks. However, due to their generic training methodology, these models often fail to meet some of the downstream requirements (e.g., hallucinations in abstractive summarization or style violations in code generation). This raises the important question of how to adapt pre-trained generative models to meet all requirements without destroying their general capabilities ("catastrophic forgetting"). Recent work has proposed to solve this problem by representing task-specific requirements through energy-based models (EBMs) and approximating these EBMs using distributional policy gradients (DPG). Despite its effectiveness, this approach is however limited to unconditional distributions. In this paper, we extend DPG to conditional tasks by proposing Conditional DPG (CDPG). We evaluate CDPG on four different control objectives across three tasks (translation, summarization and code generation) and two pretrained models (T5 and GPT-Neo). Our results show that fine-tuning using CDPG robustly moves these pretrained models closer towards meeting control objectives and — in contrast with baseline approaches — does not result in catastrophic forgetting.

----

## [506] Batch Greenkhorn Algorithm for Entropic-Regularized Multimarginal Optimal Transport: Linear Rate of Convergence and Iteration Complexity

**Authors**: *Vladimir R. Kostic, Saverio Salzo, Massimiliano Pontil*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kostic22a.html](https://proceedings.mlr.press/v162/kostic22a.html)

**Abstract**:

In this work we propose a batch multimarginal version of the Greenkhorn algorithm for the entropic-regularized optimal transport problem. This framework is general enough to cover, as particular cases, existing Sinkhorn and Greenkhorn algorithms for the bi-marginal setting, and greedy MultiSinkhorn for the general multimarginal case. We provide a comprehensive convergence analysis based on the properties of the iterative Bregman projections method with greedy control. Linear rate of convergence as well as explicit bounds on the iteration complexity are obtained. When specialized to the above mentioned algorithms, our results give new convergence rates or provide key improvements over the state-of-the-art rates. We present numerical experiments showing that the flexibility of the batch can be exploited to improve performance of Sinkhorn algorithm both in bi-marginal and multimarginal settings.

----

## [507] Certified Adversarial Robustness Under the Bounded Support Set

**Authors**: *Yiwen Kou, Qinyuan Zheng, Yisen Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kou22a.html](https://proceedings.mlr.press/v162/kou22a.html)

**Abstract**:

Deep neural networks (DNNs) have revealed severe vulnerability to adversarial perturbations, beside empirical adversarial training for robustness, the design of provably robust classifiers attracts more and more attention. Randomized smoothing methods provide the certified robustness with agnostic architecture, which is further extended to a provable robustness framework using f-divergence. While these methods cannot be applied to smoothing measures with bounded support set such as uniform probability measure due to the use of likelihood ratio in their certification methods. In this paper, we generalize the $f$-divergence-based framework to a Wasserstein-distance-based and total-variation-distance-based framework that is first able to analyze robustness properties of bounded support set smoothing measures both theoretically and experimentally. By applying our methodology to uniform probability measures with support set $l_p (p=1,2,\infty\text{ and general})$ ball, we prove negative certified robustness properties with respect to $l_q (q=1, 2, \infty)$ perturbations and present experimental results on CIFAR-10 dataset with ResNet to validate our theory. And it is also worth mentioning that our certification procedure only costs constant computation time.

----

## [508] Exact Learning of Preference Structure: Single-peaked Preferences and Beyond

**Authors**: *Sonja Kraiczy, Edith Elkind*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kraiczy22a.html](https://proceedings.mlr.press/v162/kraiczy22a.html)

**Abstract**:

We consider the setting where the members of a society (voters) have preferences over candidates, and the candidates can be ordered on an axis so that the voters’ preferences are single-peaked on this axis. We ask whether this axis can be identified by sampling the voters’ preferences. For several natural distributions, we obtain tight bounds on the number of samples required and show that, surprisingly, the bounds are independent of the number of candidates. We extend our results to the case where voters’ preferences are sampled from two different axes over the same candidate set (one of which may be known). We also consider two alternative models of learning: (1) sampling pairwise comparisons rather than entire votes, and (2) learning from equivalence queries.

----

## [509] Reconstructing Nonlinear Dynamical Systems from Multi-Modal Time Series

**Authors**: *Daniel Kramer, Philine Lou Bommer, Daniel Durstewitz, Carlo Tombolini, Georgia Koppe*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kramer22a.html](https://proceedings.mlr.press/v162/kramer22a.html)

**Abstract**:

Empirically observed time series in physics, biology, or medicine, are commonly generated by some underlying dynamical system (DS) which is the target of scientific interest. There is an increasing interest to harvest machine learning methods to reconstruct this latent DS in a data-driven, unsupervised way. In many areas of science it is common to sample time series observations from many data modalities simultaneously, e.g. electrophysiological and behavioral time series in a typical neuroscience experiment. However, current machine learning tools for reconstructing DSs usually focus on just one data modality. Here we propose a general framework for multi-modal data integration for the purpose of nonlinear DS reconstruction and the analysis of cross-modal relations. This framework is based on dynamically interpretable recurrent neural networks as general approximators of nonlinear DSs, coupled to sets of modality-specific decoder models from the class of generalized linear models. Both an expectation-maximization and a variational inference algorithm for model training are advanced and compared. We show on nonlinear DS benchmarks that our algorithms can efficiently compensate for too noisy or missing information in one data channel by exploiting other channels, and demonstrate on experimental neuroscience data how the algorithm learns to link different data domains to the underlying dynamics.

----

## [510] Probabilistic ODE Solutions in Millions of Dimensions

**Authors**: *Nicholas Krämer, Nathanael Bosch, Jonathan Schmidt, Philipp Hennig*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kramer22b.html](https://proceedings.mlr.press/v162/kramer22b.html)

**Abstract**:

Probabilistic solvers for ordinary differential equations (ODEs) have emerged as an efficient framework for uncertainty quantification and inference on dynamical systems. In this work, we explain the mathematical assumptions and detailed implementation schemes behind solving high-dimensional ODEs with a probabilistic numerical algorithm. This has not been possible before due to matrix-matrix operations in each solver step, but is crucial for scientifically relevant problems—most importantly, the solution of discretised partial differential equations. In a nutshell, efficient high-dimensional probabilistic ODE solutions build either on independence assumptions or on Kronecker structure in the prior model. We evaluate the resulting efficiency on a range of problems, including the probabilistic numerical simulation of a differential equation with millions of dimensions.

----

## [511] Active Nearest Neighbor Regression Through Delaunay Refinement

**Authors**: *Alexander Kravberg, Giovanni Luca Marchetti, Vladislav Polianskii, Anastasiia Varava, Florian T. Pokorny, Danica Kragic*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kravberg22a.html](https://proceedings.mlr.press/v162/kravberg22a.html)

**Abstract**:

We introduce an algorithm for active function approximation based on nearest neighbor regression. Our Active Nearest Neighbor Regressor (ANNR) relies on the Voronoi-Delaunay framework from computational geometry to subdivide the space into cells with constant estimated function value and select novel query points in a way that takes the geometry of the function graph into account. We consider the recent state-of-the-art active function approximator called DEFER, which is based on incremental rectangular partitioning of the space, as the main baseline. The ANNR addresses a number of limitations that arise from the space subdivision strategy used in DEFER. We provide a computationally efficient implementation of our method, as well as theoretical halting guarantees. Empirical results show that ANNR outperforms the baseline for both closed-form functions and real-world examples, such as gravitational wave parameter inference and exploration of the latent space of a generative model.

----

## [512] Functional Generalized Empirical Likelihood Estimation for Conditional Moment Restrictions

**Authors**: *Heiner Kremer, Jia-Jie Zhu, Krikamol Muandet, Bernhard Schölkopf*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kremer22a.html](https://proceedings.mlr.press/v162/kremer22a.html)

**Abstract**:

Important problems in causal inference, economics, and, more generally, robust machine learning can be expressed as conditional moment restrictions, but estimation becomes challenging as it requires solving a continuum of unconditional moment restrictions. Previous works addressed this problem by extending the generalized method of moments (GMM) to continuum moment restrictions. In contrast, generalized empirical likelihood (GEL) provides a more general framework and has been shown to enjoy favorable small-sample properties compared to GMM-based estimators. To benefit from recent developments in machine learning, we provide a functional reformulation of GEL in which arbitrary models can be leveraged. Motivated by a dual formulation of the resulting infinite dimensional optimization problem, we devise a practical method and explore its asymptotic properties. Finally, we provide kernel- and neural network-based implementations of the estimator, which achieve state-of-the-art empirical performance on two conditional moment restriction problems.

----

## [513] Calibrated and Sharp Uncertainties in Deep Learning via Density Estimation

**Authors**: *Volodymyr Kuleshov, Shachi Deshpande*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kuleshov22a.html](https://proceedings.mlr.press/v162/kuleshov22a.html)

**Abstract**:

Accurate probabilistic predictions can be characterized by two properties{—}calibration and sharpness. However, standard maximum likelihood training yields models that are poorly calibrated and thus inaccurate{—}a 90% confidence interval typically does not contain the true outcome 90% of the time. This paper argues that calibration is important in practice and is easy to maintain by performing low-dimensional density estimation. We introduce a simple training procedure based on recalibration that yields calibrated models without sacrificing overall performance; unlike previous approaches, ours ensures the most general property of distribution calibration and applies to any model, including neural networks. We formally prove the correctness of our procedure assuming that we can estimate densities in low dimensions and we establish uniform convergence bounds. Our results yield empirical performance improvements on linear and deep Bayesian models and suggest that calibration should be increasingly leveraged across machine learning.

----

## [514] ActiveHedge: Hedge meets Active Learning

**Authors**: *Bhuvesh Kumar, Jacob D. Abernethy, Venkatesh Saligrama*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kumar22a.html](https://proceedings.mlr.press/v162/kumar22a.html)

**Abstract**:

We consider the classical problem of multi-class prediction with expert advice, but with an active learning twist. In this new setting the learner will only query the labels of a small number of examples, but still aims to minimize regret to the best expert as usual; the learner is also allowed a very short "burn-in" phase where it can fast-forward and query certain highly-informative examples. We design an algorithm that utilizes Hedge (aka Exponential Weights) as a subroutine, and we show that under a very particular combinatorial constraint on the matrix of expert predictions we can obtain a very strong regret guarantee while querying very few labels. This constraint, which we refer to as $\zeta$-compactness, or just compactness, can be viewed as a non-stochastic variant of the disagreement coefficient, another popular parameter used to reason about the sample complexity of active learning in the IID setting. We also give a polynomial-time algorithm to calculate the $\zeta$-compactness of a matrix up to an approximation factor of 3.

----

## [515] Balancing Discriminability and Transferability for Source-Free Domain Adaptation

**Authors**: *Jogendra Nath Kundu, Akshay R. Kulkarni, Suvaansh Bhambri, Deepesh Mehta, Shreyas Anand Kulkarni, Varun Jampani, Venkatesh Babu Radhakrishnan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kundu22a.html](https://proceedings.mlr.press/v162/kundu22a.html)

**Abstract**:

Conventional domain adaptation (DA) techniques aim to improve domain transferability by learning domain-invariant representations; while concurrently preserving the task-discriminability knowledge gathered from the labeled source data. However, the requirement of simultaneous access to labeled source and unlabeled target renders them unsuitable for the challenging source-free DA setting. The trivial solution of realizing an effective original to generic domain mapping improves transferability but degrades task discriminability. Upon analyzing the hurdles from both theoretical and empirical standpoints, we derive novel insights to show that a mixup between original and corresponding translated generic samples enhances the discriminability-transferability trade-off while duly respecting the privacy-oriented source-free setting. A simple but effective realization of the proposed insights on top of the existing source-free DA approaches yields state-of-the-art performance with faster convergence. Beyond single-source, we also outperform multi-source prior-arts across both classification and semantic segmentation benchmarks.

----

## [516] Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters

**Authors**: *Vladislav Kurenkov, Sergey Kolesnikov*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kurenkov22a.html](https://proceedings.mlr.press/v162/kurenkov22a.html)

**Abstract**:

In this work, we argue for the importance of an online evaluation budget for a reliable comparison of deep offline RL algorithms. First, we delineate that the online evaluation budget is problem-dependent, where some problems allow for less but others for more. And second, we demonstrate that the preference between algorithms is budget-dependent across a diverse range of decision-making domains such as Robotics, Finance, and Energy Management. Following the points above, we suggest reporting the performance of deep offline RL algorithms under varying online evaluation budgets. To facilitate this, we propose to use a reporting tool from the NLP field, Expected Validation Performance. This technique makes it possible to reliably estimate expected maximum performance under different budgets while not requiring any additional computation beyond hyperparameter search. By employing this tool, we also show that Behavioral Cloning is often more favorable to offline RL algorithms when working within a limited budget.

----

## [517] Equivariant Priors for compressed sensing with unknown orientation

**Authors**: *Anna Kuzina, Kumar Pratik, Fabio Valerio Massoli, Arash Behboodi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kuzina22a.html](https://proceedings.mlr.press/v162/kuzina22a.html)

**Abstract**:

In compressed sensing, the goal is to reconstruct the signal from an underdetermined system of linear measurements. Thus, prior knowledge about the signal of interest and its structure is required. Additionally, in many scenarios, the signal has an unknown orientation prior to measurements. To address such recovery problems, we propose using equivariant generative models as a prior, which encapsulate orientation information in their latent space. Thereby, we show that signals with unknown orientations can be recovered with iterative gradient descent on the latent space of these models and provide additional theoretical recovery guarantees. We construct an equivariant variational autoencoder and use the decoder as generative prior for compressed sensing. We discuss additional potential gains of the proposed approach in terms of convergence and latency.

----

## [518] Coordinated Attacks against Contextual Bandits: Fundamental Limits and Defense Mechanisms

**Authors**: *Jeongyeol Kwon, Yonathan Efroni, Constantine Caramanis, Shie Mannor*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/kwon22a.html](https://proceedings.mlr.press/v162/kwon22a.html)

**Abstract**:

Motivated by online recommendation systems, we propose the problem of finding the optimal policy in multitask contextual bandits when a small fraction $\alpha < 1/2$ of tasks (users) are arbitrary and adversarial. The remaining fraction of good users share the same instance of contextual bandits with $S$ contexts and $A$ actions (items). Naturally, whether a user is good or adversarial is not known in advance. The goal is to robustly learn the policy that maximizes rewards for good users with as few user interactions as possible. Without adversarial users, established results in collaborative filtering show that $O(1/\epsilon^2)$ per-user interactions suffice to learn a good policy, precisely because information can be shared across users. This parallelization gain is fundamentally altered by the presence of adversarial users: unless there are super-polynomial number of users, we show a lower bound of $\tilde{\Omega}(\min(S,A) \cdot \alpha^2 / \epsilon^2)$ per-user interactions to learn an $\epsilon$-optimal policy for the good users. We then show we can achieve an $\tilde{O}(\min(S,A)\cdot \alpha/\epsilon^2)$ upper-bound, by employing efficient robust mean estimators for both uni-variate and high-dimensional random variables. We also show that this can be improved depending on the distributions of contexts.

----

## [519] Large Batch Experience Replay

**Authors**: *Thibault Lahire, Matthieu Geist, Emmanuel Rachelson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lahire22a.html](https://proceedings.mlr.press/v162/lahire22a.html)

**Abstract**:

Several algorithms have been proposed to sample non-uniformly the replay buffer of deep Reinforcement Learning (RL) agents to speed-up learning, but very few theoretical foundations of these sampling schemes have been provided. Among others, Prioritized Experience Replay appears as a hyperparameter sensitive heuristic, even though it can provide good performance. In this work, we cast the replay buffer sampling problem as an importance sampling one for estimating the gradient. This allows deriving the theoretically optimal sampling distribution, yielding the best theoretical convergence speed. Elaborating on the knowledge of the ideal sampling scheme, we exhibit new theoretical foundations of Prioritized Experience Replay. The optimal sampling distribution being intractable, we make several approximations providing good results in practice and introduce, among others, LaBER (Large Batch Experience Replay), an easy-to-code and efficient method for sampling the replay buffer. LaBER, which can be combined with Deep Q-Networks, distributional RL agents or actor-critic methods, yields improved performance over a diverse range of Atari games and PyBullet environments, compared to the base agent it is implemented on and to other prioritization schemes.

----

## [520] FedScale: Benchmarking Model and System Performance of Federated Learning at Scale

**Authors**: *Fan Lai, Yinwei Dai, Sanjay Sri Vallabh Singapuram, Jiachen Liu, Xiangfeng Zhu, Harsha V. Madhyastha, Mosharaf Chowdhury*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lai22a.html](https://proceedings.mlr.press/v162/lai22a.html)

**Abstract**:

We present FedScale, a federated learning (FL) benchmarking suite with realistic datasets and a scalable runtime to enable reproducible FL research. FedScale datasets encompass a wide range of critical FL tasks, ranging from image classification and object detection to language modeling and speech recognition. Each dataset comes with a unified evaluation protocol using real-world data splits and evaluation metrics. To reproduce realistic FL behavior, FedScale contains a scalable and extensible runtime. It provides high-level APIs to implement FL algorithms, deploy them at scale across diverse hardware and software backends, and evaluate them at scale, all with minimal developer efforts. We combine the two to perform systematic benchmarking experiments and highlight potential opportunities for heterogeneity-aware co-optimizations in FL. FedScale is open-source and actively maintained by contributors from different institutions at http://fedscale.ai. We welcome feedback and contributions from the community.

----

## [521] Smoothed Adaptive Weighting for Imbalanced Semi-Supervised Learning: Improve Reliability Against Unknown Distribution Data

**Authors**: *Zhengfeng Lai, Chao Wang, Henrry Gunawan, Sen-Ching S. Cheung, Chen-Nee Chuah*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lai22b.html](https://proceedings.mlr.press/v162/lai22b.html)

**Abstract**:

Despite recent promising results on semi-supervised learning (SSL), data imbalance, particularly in the unlabeled dataset, could significantly impact the training performance of a SSL algorithm if there is a mismatch between the expected and actual class distributions. The efforts on how to construct a robust SSL framework that can effectively learn from datasets with unknown distributions remain limited. We first investigate the feasibility of adding weights to the consistency loss and then we verify the necessity of smoothed weighting schemes. Based on this study, we propose a self-adaptive algorithm, named Smoothed Adaptive Weighting (SAW). SAW is designed to enhance the robustness of SSL by estimating the learning difficulty of each class and synthesizing the weights in the consistency loss based on such estimation. We show that SAW can complement recent consistency-based SSL algorithms and improve their reliability on various datasets including three standard datasets and one gigapixel medical imaging application without making any assumptions about the distribution of the unlabeled set.

----

## [522] Functional Output Regression with Infimal Convolution: Exploring the Huber and ε-insensitive Losses

**Authors**: *Alex Lambert, Dimitri Bouche, Zoltán Szabó, Florence d'Alché-Buc*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lambert22a.html](https://proceedings.mlr.press/v162/lambert22a.html)

**Abstract**:

The focus of the paper is functional output regression (FOR) with convoluted losses. While most existing work consider the square loss setting, we leverage extensions of the Huber and the $\epsilon$-insensitive loss (induced by infimal convolution) and propose a flexible framework capable of handling various forms of outliers and sparsity in the FOR family. We derive computationally tractable algorithms relying on duality to tackle the resulting tasks in the context of vector-valued reproducing kernel Hilbert spaces. The efficiency of the approach is demonstrated and contrasted with the classical squared loss setting on both synthetic and real-world benchmarks.

----

## [523] Tell me why! Explanations support learning relational and causal structure

**Authors**: *Andrew K. Lampinen, Nicholas A. Roy, Ishita Dasgupta, Stephanie C. Y. Chan, Allison C. Tam, James L. McClelland, Chen Yan, Adam Santoro, Neil C. Rabinowitz, Jane X. Wang, Felix Hill*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lampinen22a.html](https://proceedings.mlr.press/v162/lampinen22a.html)

**Abstract**:

Inferring the abstract relational and causal structure of the world is a major challenge for reinforcement-learning (RL) agents. For humans, language{—}particularly in the form of explanations{—}plays a considerable role in overcoming this challenge. Here, we show that language can play a similar role for deep RL agents in complex environments. While agents typically struggle to acquire relational and causal knowledge, augmenting their experience by training them to predict language descriptions and explanations can overcome these limitations. We show that language can help agents learn challenging relational tasks, and examine which aspects of language contribute to its benefits. We then show that explanations can help agents to infer not only relational but also causal structure. Language can shape the way that agents to generalize out-of-distribution from ambiguous, causally-confounded training, and explanations even allow agents to learn to perform experimental interventions to identify causal relationships. Our results suggest that language description and explanation may be powerful tools for improving agent learning and generalization.

----

## [524] Generative Cooperative Networks for Natural Language Generation

**Authors**: *Sylvain Lamprier, Thomas Scialom, Antoine Chaffin, Vincent Claveau, Ewa Kijak, Jacopo Staiano, Benjamin Piwowarski*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lamprier22a.html](https://proceedings.mlr.press/v162/lamprier22a.html)

**Abstract**:

Generative Adversarial Networks (GANs) have known a tremendous success for many continuous generation tasks, especially in the field of image generation. However, for discrete outputs such as language, optimizing GANs remains an open problem with many instabilities, as no gradient can be properly back-propagated from the discriminator output to the generator parameters. An alternative is to learn the generator network via reinforcement learning, using the discriminator signal as a reward, but such a technique suffers from moving rewards and vanishing gradient problems. Finally, it often falls short compared to direct maximum-likelihood approaches. In this paper, we introduce Generative Cooperative Networks, in which the discriminator architecture is cooperatively used along with the generation policy to output samples of realistic texts for the task at hand. We give theoretical guarantees of convergence for our approach, and study various efficient decoding schemes to empirically achieve state-of-the-art results in two main NLG tasks.

----

## [525] DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting

**Authors**: *Shiyong Lan, Yitong Ma, Weikang Huang, Wenwu Wang, Hongyu Yang, Pyang Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lan22a.html](https://proceedings.mlr.press/v162/lan22a.html)

**Abstract**:

As a typical problem in time series analysis, traffic flow prediction is one of the most important application fields of machine learning. However, achieving highly accurate traffic flow prediction is a challenging task, due to the presence of complex dynamic spatial-temporal dependencies within a road network. This paper proposes a novel Dynamic Spatial-Temporal Aware Graph Neural Network (DSTAGNN) to model the complex spatial-temporal interaction in road network. First, considering the fact that historical data carries intrinsic dynamic information about the spatial structure of road networks, we propose a new dynamic spatial-temporal aware graph based on a data-driven strategy to replace the pre-defined static graph usually used in traditional graph convolution. Second, we design a novel graph neural network architecture, which can not only represent dynamic spatial relevance among nodes with an improved multi-head attention mechanism, but also acquire the wide range of dynamic temporal dependency from multi-receptive field features via multi-scale gated convolution. Extensive experiments on real-world data sets demonstrate that our proposed method significantly outperforms the state-of-the-art methods.

----

## [526] Cooperative Online Learning in Stochastic and Adversarial MDPs

**Authors**: *Tal Lancewicki, Aviv Rosenberg, Yishay Mansour*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lancewicki22a.html](https://proceedings.mlr.press/v162/lancewicki22a.html)

**Abstract**:

We study cooperative online learning in stochastic and adversarial Markov decision process (MDP). That is, in each episode, $m$ agents interact with an MDP simultaneously and share information in order to minimize their individual regret. We consider environments with two types of randomness: fresh – where each agent’s trajectory is sampled i.i.d, and non-fresh – where the realization is shared by all agents (but each agent’s trajectory is also affected by its own actions). More precisely, with non-fresh randomness the realization of every cost and transition is fixed at the start of each episode, and agents that take the same action in the same state at the same time observe the same cost and next state. We thoroughly analyze all relevant settings, highlight the challenges and differences between the models, and prove nearly-matching regret lower and upper bounds. To our knowledge, we are the first to consider cooperative reinforcement learning (RL) with either non-fresh randomness or in adversarial MDPs.

----

## [527] PINs: Progressive Implicit Networks for Multi-Scale Neural Representations

**Authors**: *Zoe Landgraf, Alexander Sorkine-Hornung, Ricardo Silveira Cabral*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/landgraf22a.html](https://proceedings.mlr.press/v162/landgraf22a.html)

**Abstract**:

Multi-layer perceptrons (MLP) have proven to be effective scene encoders when combined with higher-dimensional projections of the input, commonly referred to as positional encoding. However, scenes with a wide frequency spectrum remain a challenge: choosing high frequencies for positional encoding introduces noise in low structure areas, while low frequencies results in poor fitting of detailed regions. To address this, we propose a progressive positional encoding, exposing a hierarchical MLP structure to incremental sets of frequency encodings. Our model accurately reconstructs scenes with wide frequency bands and learns a scene representation at progressive level of detail without explicit per-level supervision. The architecture is modular: each level encodes a continuous implicit representation that can be leveraged separately for its respective resolution, meaning a smaller network for coarser reconstructions. Experiments on several 2D and 3D datasets shows improvements in reconstruction accuracy, representational capacity and training speed compared to baselines.

----

## [528] Co-training Improves Prompt-based Learning for Large Language Models

**Authors**: *Hunter Lang, Monica N. Agrawal, Yoon Kim, David A. Sontag*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lang22a.html](https://proceedings.mlr.press/v162/lang22a.html)

**Abstract**:

We demonstrate that co-training (Blum & Mitchell, 1998) can improve the performance of prompt-based learning by using unlabeled data. While prompting has emerged as a promising paradigm for few-shot and zero-shot learning, it is often brittle and requires much larger models compared to the standard supervised setup. We find that co-training makes it possible to improve the original prompt model and at the same time learn a smaller, downstream task-specific model. In the case where we only have partial access to a prompt model (e.g., output probabilities from GPT-3 (Brown et al., 2020)) we learn a calibration model over the prompt outputs. When we have full access to the prompt model’s gradients but full finetuning remains prohibitively expensive (e.g., T0 (Sanh et al., 2021)), we learn a set of soft prompt continuous vectors to iteratively update the prompt model. We find that models trained in this manner can significantly improve performance on challenging datasets where there is currently a large gap between prompt-based learning and fully-supervised models.

----

## [529] Goal Misgeneralization in Deep Reinforcement Learning

**Authors**: *Lauro Langosco di Langosco, Jack Koch, Lee D. Sharkey, Jacob Pfau, David Krueger*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/langosco22a.html](https://proceedings.mlr.press/v162/langosco22a.html)

**Abstract**:

We study goal misgeneralization, a type of out-of-distribution robustness failure in reinforcement learning (RL). Goal misgeneralization occurs when an RL agent retains its capabilities out-of-distribution yet pursues the wrong goal. For instance, an agent might continue to competently avoid obstacles, but navigate to the wrong place. In contrast, previous works have typically focused on capability generalization failures, where an agent fails to do anything sensible at test time.We provide the first explicit empirical demonstrations of goal misgeneralization and present a partial characterization of its causes.

----

## [530] Marginal Tail-Adaptive Normalizing Flows

**Authors**: *Mike Laszkiewicz, Johannes Lederer, Asja Fischer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/laszkiewicz22a.html](https://proceedings.mlr.press/v162/laszkiewicz22a.html)

**Abstract**:

Learning the tail behavior of a distribution is a notoriously difficult problem. By definition, the number of samples from the tail is small, and deep generative models, such as normalizing flows, tend to concentrate on learning the body of the distribution. In this paper, we focus on improving the ability of normalizing flows to correctly capture the tail behavior and, thus, form more accurate models. We prove that the marginal tailedness of an autoregressive flow can be controlled via the tailedness of the marginals of its base distribution. This theoretical insight leads us to a novel type of flows based on flexible base distributions and data-driven linear layers. An empirical analysis shows that the proposed method improves on the accuracy{—}especially on the tails of the distribution{—}and is able to generate heavy-tailed data. We demonstrate its application on a weather and climate example, in which capturing the tail behavior is essential.

----

## [531] Bregman Proximal Langevin Monte Carlo via Bregman-Moreau Envelopes

**Authors**: *Tim Tsz-Kit Lau, Han Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lau22a.html](https://proceedings.mlr.press/v162/lau22a.html)

**Abstract**:

We propose efficient Langevin Monte Carlo algorithms for sampling distributions with nonsmooth convex composite potentials, which is the sum of a continuously differentiable function and a possibly nonsmooth function. We devise such algorithms leveraging recent advances in convex analysis and optimization methods involving Bregman divergences, namely the Bregman–Moreau envelopes and the Bregman proximity operators, and in the Langevin Monte Carlo algorithms reminiscent of mirror descent. The proposed algorithms extend existing Langevin Monte Carlo algorithms in two aspects—the ability to sample nonsmooth distributions with mirror descent-like algorithms, and the use of the more general Bregman–Moreau envelope in place of the Moreau envelope as a smooth approximation of the nonsmooth part of the potential. A particular case of the proposed scheme is reminiscent of the Bregman proximal gradient algorithm. The efficiency of the proposed methodology is illustrated with various sampling tasks at which existing Langevin Monte Carlo methods are known to perform poorly.

----

## [532] Scalable Deep Reinforcement Learning Algorithms for Mean Field Games

**Authors**: *Mathieu Laurière, Sarah Perrin, Sertan Girgin, Paul Muller, Ayush Jain, Theophile Cabannes, Georgios Piliouras, Julien Pérolat, Romuald Elie, Olivier Pietquin, Matthieu Geist*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lauriere22a.html](https://proceedings.mlr.press/v162/lauriere22a.html)

**Abstract**:

Mean Field Games (MFGs) have been introduced to efficiently approximate games with very large populations of strategic agents. Recently, the question of learning equilibria in MFGs has gained momentum, particularly using model-free reinforcement learning (RL) methods. One limiting factor to further scale up using RL is that existing algorithms to solve MFGs require the mixing of approximated quantities such as strategies or $q$-values. This is far from being trivial in the case of non-linear function approximation that enjoy good generalization properties, e.g. neural networks. We propose two methods to address this shortcoming. The first one learns a mixed strategy from distillation of historical data into a neural network and is applied to the Fictitious Play algorithm. The second one is an online mixing method based on regularization that does not require memorizing historical data or previous estimates. It is used to extend Online Mirror Descent. We demonstrate numerically that these methods efficiently enable the use of Deep RL algorithms to solve various MFGs. In addition, we show that these methods outperform SotA baselines from the literature.

----

## [533] Implicit Bias of Linear Equivariant Networks

**Authors**: *Hannah Lawrence, Bobak Toussi Kiani, Kristian G. Georgiev, Andrew K. Dienes*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lawrence22a.html](https://proceedings.mlr.press/v162/lawrence22a.html)

**Abstract**:

Group equivariant convolutional neural networks (G-CNNs) are generalizations of convolutional neural networks (CNNs) which excel in a wide range of technical applications by explicitly encoding symmetries, such as rotations and permutations, in their architectures. Although the success of G-CNNs is driven by their explicit symmetry bias, a recent line of work has proposed that the implicit bias of training algorithms on particular architectures is key to understanding generalization for overparameterized neural nets. In this context, we show that L-layer full-width linear G-CNNs trained via gradient descent for binary classification converge to solutions with low-rank Fourier matrix coefficients, regularized by the 2/L-Schatten matrix norm. Our work strictly generalizes previous analysis on the implicit bias of linear CNNs to linear G-CNNs over all finite groups, including the challenging setting of non-commutative groups (such as permutations), as well as band-limited G-CNNs over infinite groups. We validate our theorems via experiments on a variety of groups, and empirically explore more realistic nonlinear networks, which locally capture similar regularization patterns. Finally, we provide intuitive interpretations of our Fourier space implicit regularization results in real space via uncertainty principles.

----

## [534] Differentially Private Maximal Information Coefficients

**Authors**: *John Lazarsfeld, Aaron Johnson, Emmanuel Adéníran*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lazarsfeld22a.html](https://proceedings.mlr.press/v162/lazarsfeld22a.html)

**Abstract**:

The Maximal Information Coefficient (MIC) is a powerful statistic to identify dependencies between variables. However, it may be applied to sensitive data, and publishing it could leak private information. As a solution, we present algorithms to approximate MIC in a way that provides differential privacy. We show that the natural application of the classic Laplace mechanism yields insufficient accuracy. We therefore introduce the MICr statistic, which is a new MIC approximation that is more compatible with differential privacy. We prove MICr is a consistent estimator for MIC, and we provide two differentially private versions of it. We perform experiments on a variety of real and synthetic datasets. The results show that the private MICr statistics significantly outperform direct application of the Laplace mechanism. Moreover, experiments on real-world datasets show accuracy that is usable when the sample size is at least moderately large.

----

## [535] Entropic Gromov-Wasserstein between Gaussian Distributions

**Authors**: *Khang Le, Dung Q. Le, Huy Nguyen, Dat Do, Tung Pham, Nhat Ho*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/le22a.html](https://proceedings.mlr.press/v162/le22a.html)

**Abstract**:

We study the entropic Gromov-Wasserstein and its unbalanced version between (unbalanced) Gaussian distributions with different dimensions. When the metric is the inner product, which we refer to as inner product Gromov-Wasserstein (IGW), we demonstrate that the optimal transportation plans of entropic IGW and its unbalanced variant are (unbalanced) Gaussian distributions. Via an application of von Neumann’s trace inequality, we obtain closed-form expressions for the entropic IGW between these Gaussian distributions. Finally, we consider an entropic inner product Gromov-Wasserstein barycenter of multiple Gaussian distributions. We prove that the barycenter is a Gaussian distribution when the entropic regularization parameter is small. We further derive a closed-form expression for the covariance matrix of the barycenter.

----

## [536] Neurocoder: General-Purpose Computation Using Stored Neural Programs

**Authors**: *Hung Le, Svetha Venkatesh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/le22b.html](https://proceedings.mlr.press/v162/le22b.html)

**Abstract**:

Artificial Neural Networks are functionally equivalent to special-purpose computers. Their inter-neuronal connection weights represent the learnt Neural Program that instructs the networks on how to compute the data. However, without storing Neural Programs, they are restricted to only one, overwriting learnt programs when trained on new data. Here we design Neurocoder, a new class of general-purpose neural networks in which the neural network “codes” itself in a data-responsive way by composing relevant programs from a set of shareable, modular programs stored in external memory. This time, a Neural Program is efficiently treated as data in memory. Integrating Neurocoder into current neural architectures, we demonstrate new capacity to learn modular programs, reuse simple programs to build complex ones, handle pattern shifts and remember old programs as new ones are learnt, and show substantial performance improvement in solving object recognition, playing video games and continual learning tasks.

----

## [537] Convergence of Policy Gradient for Entropy Regularized MDPs with Neural Network Approximation in the Mean-Field Regime

**Authors**: *James-Michael Leahy, Bekzhan Kerimkulov, David Siska, Lukasz Szpruch*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/leahy22a.html](https://proceedings.mlr.press/v162/leahy22a.html)

**Abstract**:

We study the global convergence of policy gradient for infinite-horizon, continuous state and action space, and entropy-regularized Markov decision processes (MDPs). We consider a softmax policy with (one-hidden layer) neural network approximation in a mean-field regime. Additional entropic regularization in the associated mean-field probability measure is added, and the corresponding gradient flow is studied in the 2-Wasserstein metric. We show that the objective function is increasing along the gradient flow. Further, we prove that if the regularization in terms of the mean-field measure is sufficient, the gradient flow converges exponentially fast to the unique stationary solution, which is the unique maximizer of the regularized MDP objective. Lastly, we study the sensitivity of the value function along the gradient flow with respect to regularization parameters and the initial condition. Our results rely on the careful analysis of the non-linear Fokker–Planck–Kolmogorov equation and extend the pioneering work of \cite{mei2020global} and \cite{agarwal2020optimality}, which quantify the global convergence rate of policy gradient for entropy-regularized MDPs in the tabular setting.

----

## [538] A Random Matrix Analysis of Data Stream Clustering: Coping With Limited Memory Resources

**Authors**: *Hugo Lebeau, Romain Couillet, Florent Chatelain*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lebeau22a.html](https://proceedings.mlr.press/v162/lebeau22a.html)

**Abstract**:

This article introduces a random matrix framework for the analysis of clustering on high-dimensional data streams, a particularly relevant setting for a more sober processing of large amounts of data with limited memory and energy resources. Assuming data $\mathbf{x}_1, \mathbf{x}_2, \ldots$ arrives as a continuous flow and a small number $L$ of them can be kept in the learning pipeline, one has only access to the diagonal elements of the Gram kernel matrix: $\left[ \mathbf{K}_L \right]_{i, j} = \frac{1}{p} \mathbf{x}_i^\top \mathbf{x}_j \mathbf{1}_{\left\lvert i - j \right\rvert < L}$. Under a large-dimensional data regime, we derive the limiting spectral distribution of the banded kernel matrix $\mathbf{K}_L$ and study its isolated eigenvalues and eigenvectors, which behave in an unfamiliar way. We detail how these results can be used to perform efficient online kernel spectral clustering and provide theoretical performance guarantees. Our findings are empirically confirmed on image clustering tasks. Leveraging on optimality results of spectral methods for clustering, this work offers insights on efficient online clustering techniques for high-dimensional data.

----

## [539] Neural Tangent Kernel Analysis of Deep Narrow Neural Networks

**Authors**: *Jongmin Lee, Joo Young Choi, Ernest K. Ryu, Albert No*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22a.html](https://proceedings.mlr.press/v162/lee22a.html)

**Abstract**:

The tremendous recent progress in analyzing the training dynamics of overparameterized neural networks has primarily focused on wide networks and therefore does not sufficiently address the role of depth in deep learning. In this work, we present the first trainability guarantee of infinitely deep but narrow neural networks. We study the infinite-depth limit of a multilayer perceptron (MLP) with a specific initialization and establish a trainability guarantee using the NTK theory. We then extend the analysis to an infinitely deep convolutional neural network (CNN) and perform brief experiments.

----

## [540] Dataset Condensation with Contrastive Signals

**Authors**: *Saehyung Lee, Sanghyuk Chun, Sangwon Jung, Sangdoo Yun, Sungroh Yoon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22b.html](https://proceedings.mlr.press/v162/lee22b.html)

**Abstract**:

Recent studies have demonstrated that gradient matching-based dataset synthesis, or dataset condensation (DC), methods can achieve state-of-theart performance when applied to data-efficient learning tasks. However, in this study, we prove that the existing DC methods can perform worse than the random selection method when taskirrelevant information forms a significant part of the training dataset. We attribute this to the lack of participation of the contrastive signals between the classes resulting from the class-wise gradient matching strategy. To address this problem, we propose Dataset Condensation with Contrastive signals (DCC) by modifying the loss function to enable the DC methods to effectively capture the differences between classes. In addition, we analyze the new loss function in terms of training dynamics by tracking the kernel velocity. Furthermore, we introduce a bi-level warm-up strategy to stabilize the optimization. Our experimental results indicate that while the existing methods are ineffective for fine-grained image classification tasks, the proposed method can successfully generate informative synthetic datasets for the same tasks. Moreover, we demonstrate that the proposed method outperforms the baselines even on benchmark datasets such as SVHN, CIFAR-10, and CIFAR-100. Finally, we demonstrate the high applicability of the proposed method by applying it to continual learning tasks.

----

## [541] Confidence Score for Source-Free Unsupervised Domain Adaptation

**Authors**: *Jonghyun Lee, Dahuin Jung, Junho Yim, Sungroh Yoon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22c.html](https://proceedings.mlr.press/v162/lee22c.html)

**Abstract**:

Source-free unsupervised domain adaptation (SFUDA) aims to obtain high performance in the unlabeled target domain using the pre-trained source model, not the source data. Existing SFUDA methods assign the same importance to all target samples, which is vulnerable to incorrect pseudo-labels. To differentiate between sample importance, in this study, we propose a novel sample-wise confidence score, the Joint Model-Data Structure (JMDS) score for SFUDA. Unlike existing confidence scores that use only one of the source or target domain knowledge, the JMDS score uses both knowledge. We then propose a Confidence score Weighting Adaptation using the JMDS (CoWA-JMDS) framework for SFUDA. CoWA-JMDS consists of the JMDS scores as sample weights and weight Mixup that is our proposed variant of Mixup. Weight Mixup promotes the model make more use of the target domain knowledge. The experimental results show that the JMDS score outperforms the existing confidence scores. Moreover, CoWA-JMDS achieves state-of-the-art performance on various SFUDA scenarios: closed, open, and partial-set scenarios.

----

## [542] A Statistical Manifold Framework for Point Cloud Data

**Authors**: *Yonghyeon Lee, Seungyeon Kim, Jinwon Choi, Frank Chongwoo Park*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22d.html](https://proceedings.mlr.press/v162/lee22d.html)

**Abstract**:

Many problems in machine learning involve data sets in which each data point is a point cloud in $\mathbb{R}^D$. A growing number of applications require a means of measuring not only distances between point clouds, but also angles, volumes, derivatives, and other more advanced concepts. To formulate and quantify these concepts in a coordinate-invariant way, we develop a Riemannian geometric framework for point cloud data. By interpreting each point in a point cloud as a sample drawn from some given underlying probability density, the space of point cloud data can be given the structure of a statistical manifold – each point on this manifold represents a point cloud – with the Fisher information metric acting as a natural Riemannian metric. Two autoencoder applications of our framework are presented: (i) smoothly deforming one 3D object into another via interpolation between the two corresponding point clouds; (ii) learning an optimal set of latent space coordinates for point cloud data that best preserves angles and distances, and thus produces a more discriminative representation space. Experiments with large-scale standard benchmark point cloud data show greatly improved classification accuracy vis-á-vis existing methods. Code is available at https://github.com/seungyeon-k/SMF-public.

----

## [543] Low-Complexity Deep Convolutional Neural Networks on Fully Homomorphic Encryption Using Multiplexed Parallel Convolutions

**Authors**: *Eunsang Lee, Joon-Woo Lee, Junghyun Lee, Young-Sik Kim, Yongjune Kim, Jong-Seon No, Woosuk Choi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22e.html](https://proceedings.mlr.press/v162/lee22e.html)

**Abstract**:

Recently, the standard ResNet-20 network was successfully implemented on the fully homomorphic encryption scheme, residue number system variant Cheon-Kim-Kim-Song (RNS-CKKS) scheme using bootstrapping, but the implementation lacks practicality due to high latency and low security level. To improve the performance, we first minimize total bootstrapping runtime using multiplexed parallel convolution that collects sparse output data for multiple channels compactly. We also propose the imaginary-removing bootstrapping to prevent the deep neural networks from catastrophic divergence during approximate ReLU operations. In addition, we optimize level consumptions and use lighter and tighter parameters. Simulation results show that we have 4.67x lower inference latency and 134x less amortized runtime (runtime per image) for ResNet-20 compared to the state-of-the-art previous work, and we achieve standard 128-bit security. Furthermore, we successfully implement ResNet-110 with high accuracy on the RNS-CKKS scheme for the first time.

----

## [544] Statistical inference with implicit SGD: proximal Robbins-Monro vs Polyak-Ruppert

**Authors**: *Yoonhyung Lee, Sungdong Lee, Joong-Ho Won*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22f.html](https://proceedings.mlr.press/v162/lee22f.html)

**Abstract**:

The implicit stochastic gradient descent (ISGD), a proximal version of SGD, is gaining interest in the literature due to its stability over (explicit) SGD. In this paper, we conduct an in-depth analysis of the two modes of ISGD for smooth convex functions, namely proximal Robbins-Monro (proxRM) and proximal Poylak-Ruppert (proxPR) procedures, for their use in statistical inference on model parameters. Specifically, we derive non-asymptotic point estimation error bounds of both proxRM and proxPR iterates and their limiting distributions, and propose on-line estimators of their asymptotic covariance matrices that require only a single run of ISGD. The latter estimators are used to construct valid confidence intervals for the model parameters. Our analysis is free of the generalized linear model assumption that has limited the preceding analyses, and employs feasible procedures. Our on-line covariance matrix estimators appear to be the first of this kind in the ISGD literature.

----

## [545] Maslow's Hammer in Catastrophic Forgetting: Node Re-Use vs Node Activation

**Authors**: *Sebastian Lee, Stefano Sarao Mannelli, Claudia Clopath, Sebastian Goldt, Andrew M. Saxe*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22g.html](https://proceedings.mlr.press/v162/lee22g.html)

**Abstract**:

Continual learning—learning new tasks in sequence while maintaining performance on old tasks—remains particularly challenging for artificial neural networks. Surprisingly, the amount of forgetting does not increase with the dissimilarity between the learned tasks, but appears to be worst in an intermediate similarity regime. In this paper we theoretically analyse both a synthetic teacher-student framework and a real data setup to provide an explanation of this phenomenon that we name Maslow’s Hammer hypothesis. Our analysis reveals the presence of a trade-off between node activation and node re-use that results in worst forgetting in the intermediate regime. Using this understanding we reinterpret popular algorithmic interventions for catastrophic interference in terms of this trade-off, and identify the regimes in which they are most effective.

----

## [546] Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization

**Authors**: *Deokjae Lee, Seungyong Moon, Junhyeok Lee, Hyun Oh Song*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22h.html](https://proceedings.mlr.press/v162/lee22h.html)

**Abstract**:

We focus on the problem of adversarial attacks against models on discrete sequential data in the black-box setting where the attacker aims to craft adversarial examples with limited query access to the victim model. Existing black-box attacks, mostly based on greedy algorithms, find adversarial examples using pre-computed key positions to perturb, which severely limits the search space and might result in suboptimal solutions. To this end, we propose a query-efficient black-box attack using Bayesian optimization, which dynamically computes important positions using an automatic relevance determination (ARD) categorical kernel. We introduce block decomposition and history subsampling techniques to improve the scalability of Bayesian optimization when an input sequence becomes long. Moreover, we develop a post-optimization algorithm that finds adversarial examples with smaller perturbation size. Experiments on natural language and protein classification tasks demonstrate that our method consistently achieves higher attack success rate with significant reduction in query count and modification rate compared to the previous state-of-the-art methods.

----

## [547] Least Squares Estimation using Sketched Data with Heteroskedastic Errors

**Authors**: *Sokbae Lee, Serena Ng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22i.html](https://proceedings.mlr.press/v162/lee22i.html)

**Abstract**:

Researchers may perform regressions using a sketch of data of size m instead of the full sample of size n for a variety of reasons. This paper considers the case when the regression errors do not have constant variance and heteroskedasticity robust standard errors would normally be needed for test statistics to provide accurate inference. We show that estimates using data sketched by random projections will behave ’as if’ the errors were homoskedastic. Estimation by random sampling would not have this property. The result arises because the sketched estimates in the case of random projections can be expressed as degenerate U-statistics, and under certain conditions, these statistics are asymptotically normal with homoskedastic variance. We verify that the conditions hold not only in the case of least squares regression when the covariates are exogenous, but also in instrumental variables estimation when the covariates are endogenous. The result implies that inference can be simpler than the full sample case if the sketching scheme is appropriately chosen.

----

## [548] Why the Rich Get Richer? On the Balancedness of Random Partition Models

**Authors**: *Changwoo J. Lee, Huiyan Sang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22j.html](https://proceedings.mlr.press/v162/lee22j.html)

**Abstract**:

Random partition models are widely used in Bayesian methods for various clustering tasks, such as mixture models, topic models, and community detection problems. While the number of clusters induced by random partition models has been studied extensively, another important model property regarding the balancedness of partition has been largely neglected. We formulate a framework to define and theoretically study the balancedness of exchangeable random partition models, by analyzing how a model assigns probabilities to partitions with different levels of balancedness. We demonstrate that the "rich-get-richer" characteristic of many existing popular random partition models is an inevitable consequence of two common assumptions: product-form exchangeability and projectivity. We propose a principled way to compare the balancedness of random partition models, which gives a better understanding of what model works better and what doesn’t for different applications. We also introduce the "rich-get-poorer" random partition models and illustrate their application to entity resolution tasks.

----

## [549] Model Selection in Batch Policy Optimization

**Authors**: *Jonathan Lee, George Tucker, Ofir Nachum, Bo Dai*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lee22k.html](https://proceedings.mlr.press/v162/lee22k.html)

**Abstract**:

We study the problem of model selection in batch policy optimization: given a fixed, partial-feedback dataset and M model classes, learn a policy with performance that is competitive with the policy derived from the best model class. We formalize the problem in the contextual bandit setting with linear model classes by identifying three sources of error that any model selection algorithm should optimally trade-off in order to be competitive: (1) approximation error, (2) statistical complexity, and (3) coverage. The first two sources are common in model selection for supervised learning, where optimally trading off these two is well-studied. In contrast, the third source is unique to batch policy optimization and is due to dataset shift inherent to the setting. We first show that no batch policy optimization algorithm can achieve a guarantee addressing all three simultaneously, revealing a stark contrast between difficulties in batch policy optimization and the positive results available in supervised learning. Despite this negative result, we show that relaxing any one of the three error sources enables the design of algorithms achieving near-oracle inequalities for the remaining two. We conclude with experiments demonstrating the efficacy of these algorithms.

----

## [550] Supervised Learning with General Risk Functionals

**Authors**: *Liu Leqi, Audrey Huang, Zachary C. Lipton, Kamyar Azizzadenesheli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/leqi22a.html](https://proceedings.mlr.press/v162/leqi22a.html)

**Abstract**:

Standard uniform convergence results bound the generalization gap of the expected loss over a hypothesis class. The emergence of risk-sensitive learning requires generalization guarantees for functionals of the loss distribution beyond the expectation. While prior works specialize in uniform convergence of particular functionals, our work provides uniform convergence for a general class of Hölder risk functionals for which the closeness in the Cumulative Distribution Function (CDF) entails closeness in risk. We establish the first uniform convergence results for estimating the CDF of the loss distribution, which yield uniform convergence guarantees that hold simultaneously both over a class of Hölder risk functionals and over a hypothesis class. Thus licensed to perform empirical risk minimization, we develop practical gradient-based methods for minimizing distortion risks (widely studied subset of Hölder risks that subsumes the spectral risks, including the mean, conditional value at risk, cumulative prospect theory risks, and others) and provide convergence guarantees. In experiments, we demonstrate the efficacy of our learning procedure, both in settings where uniform convergence results hold and in high-dimensional settings with deep networks.

----

## [551] Generalized Strategic Classification and the Case of Aligned Incentives

**Authors**: *Sagi Levanon, Nir Rosenfeld*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/levanon22a.html](https://proceedings.mlr.press/v162/levanon22a.html)

**Abstract**:

Strategic classification studies learning in settings where self-interested users can strategically modify their features to obtain favorable predictive outcomes. A key working assumption, however, is that “favorable” always means “positive”; this may be appropriate in some applications (e.g., loan approval), but reduces to a fairly narrow view of what user interests can be. In this work we argue for a broader perspective on what accounts for strategic user behavior, and propose and study a flexible model of generalized strategic classification. Our generalized model subsumes most current models but includes other novel settings; among these, we identify and target one intriguing sub-class of problems in which the interests of users and the system are aligned. This setting reveals a surprising fact: that standard max-margin losses are ill-suited for strategic inputs. Returning to our fully generalized model, we propose a novel max-margin framework for strategic learning that is practical and effective, and which we analyze theoretically. We conclude with a set of experiments that empirically demonstrate the utility of our approach.

----

## [552] A Simple Unified Framework for High Dimensional Bandit Problems

**Authors**: *Wenjie Li, Adarsh Barik, Jean Honorio*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22a.html](https://proceedings.mlr.press/v162/li22a.html)

**Abstract**:

Stochastic high dimensional bandit problems with low dimensional structures are useful in different applications such as online advertising and drug discovery. In this work, we propose a simple unified algorithm for such problems and present a general analysis framework for the regret upper bound of our algorithm. We show that under some mild unified assumptions, our algorithm can be applied to different high-dimensional bandit problems. Our framework utilizes the low dimensional structure to guide the parameter estimation in the problem, therefore our algorithm achieves the comparable regret bounds in the LASSO bandit as a sanity check, as well as novel bounds that depend logarithmically on dimensions in the low-rank matrix bandit, the group sparse matrix bandit, and in a new problem: the multi-agent LASSO bandit.

----

## [553] Robust Training of Neural Networks Using Scale Invariant Architectures

**Authors**: *Zhiyuan Li, Srinadh Bhojanapalli, Manzil Zaheer, Sashank J. Reddi, Sanjiv Kumar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22b.html](https://proceedings.mlr.press/v162/li22b.html)

**Abstract**:

In contrast to SGD, adaptive gradient methods like Adam allow robust training of modern deep networks, especially large language models. However, the use of adaptivity not only comes at the cost of extra memory but also raises the fundamental question: can non-adaptive methods like SGD enjoy similar benefits? In this paper, we provide an affirmative answer to this question by proposing to achieve both robust and memory-efficient training via the following general recipe: (1) modify the architecture and make it scale invariant, (2) train with SGD and weight decay, and optionally (3) clip the global gradient norm proportional to weight norm multiplied by $\sqrt{\frac{2\lambda}{\eta}}$, where $\eta$ is learning rate and $\lambda$ is weight decay. We show that this general approach is robust to rescaling of parameter and loss by proving that its convergence only depends logarithmically on the scale of initialization and loss, whereas the standard SGD might not even converge for many initializations. Following our recipe, we design a scale invariant version of BERT, called SIBERT, which when trained simply by vanilla SGD achieves performance comparable to BERT trained by adaptive methods like Adam on downstream tasks.

----

## [554] Spatial-Channel Token Distillation for Vision MLPs

**Authors**: *Yanxi Li, Xinghao Chen, Minjing Dong, Yehui Tang, Yunhe Wang, Chang Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22c.html](https://proceedings.mlr.press/v162/li22c.html)

**Abstract**:

Recently, neural architectures with all Multi-layer Perceptrons (MLPs) have attracted great research interest from the computer vision community. However, the inefficient mixing of spatial-channel information causes MLP-like vision models to demand tremendous pre-training on large-scale datasets. This work solves the problem from a novel knowledge distillation perspective. We propose a novel Spatial-channel Token Distillation (STD) method, which improves the information mixing in the two dimensions by introducing distillation tokens to each of them. A mutual information regularization is further introduced to let distillation tokens focus on their specific dimensions and maximize the performance gain. Extensive experiments on ImageNet for several MLP-like architectures demonstrate that the proposed token distillation mechanism can efficiently improve the accuracy. For example, the proposed STD boosts the top-1 accuracy of Mixer-S16 on ImageNet from 73.8% to 75.7% without any costly pre-training on JFT-300M. When applied to stronger architectures, e.g. CycleMLP-B1 and CycleMLP-B2, STD can still harvest about 1.1% and 0.5% accuracy gains, respectively.

----

## [555] An Analytical Update Rule for General Policy Optimization

**Authors**: *Hepeng Li, Nicholas Clavette, Haibo He*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22d.html](https://proceedings.mlr.press/v162/li22d.html)

**Abstract**:

We present an analytical policy update rule that is independent of parametric function approximators. The policy update rule is suitable for optimizing general stochastic policies and has a monotonic improvement guarantee. It is derived from a closed-form solution to trust-region optimization using calculus of variation, following a new theoretical result that tightens existing bounds for policy improvement using trust-region methods. The update rule builds a connection between policy search methods and value function methods. Moreover, off-policy reinforcement learning algorithms can be derived from the update rule since it does not need to compute integration over on-policy states. In addition, the update rule extends immediately to cooperative multi-agent systems when policy updates are performed by one agent at a time.

----

## [556] On Convergence of Gradient Descent Ascent: A Tight Local Analysis

**Authors**: *Haochuan Li, Farzan Farnia, Subhro Das, Ali Jadbabaie*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22e.html](https://proceedings.mlr.press/v162/li22e.html)

**Abstract**:

Gradient Descent Ascent (GDA) methods are the mainstream algorithms for minimax optimization in generative adversarial networks (GANs). Convergence properties of GDA have drawn significant interest in the recent literature. Specifically, for $\min_{x} \max_{y} f(x;y)$ where $f$ is strongly-concave in $y$ and possibly nonconvex in $x$, (Lin et al., 2020) proved the convergence of GDA with a stepsize ratio $\eta_y/\eta_x=\Theta(\kappa^2)$ where $\eta_x$ and $\eta_y$ are the stepsizes for $x$ and $y$ and $\kappa$ is the condition number for $y$. While this stepsize ratio suggests a slow training of the min player, practical GAN algorithms typically adopt similar stepsizes for both variables, indicating a wide gap between theoretical and empirical results. In this paper, we aim to bridge this gap by analyzing the local convergence of general nonconvex-nonconcave minimax problems. We demonstrate that a stepsize ratio of $\Theta(\kappa)$ is necessary and sufficient for local convergence of GDA to a Stackelberg Equilibrium, where $\kappa$ is the local condition number for $y$. We prove a nearly tight convergence rate with a matching lower bound. We further extend the convergence guarantees to stochastic GDA and extra-gradient methods (EG). Finally, we conduct several numerical experiments to support our theoretical findings.

----

## [557] On the Finite-Time Performance of the Knowledge Gradient Algorithm

**Authors**: *Yanwen Li, Siyang Gao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22f.html](https://proceedings.mlr.press/v162/li22f.html)

**Abstract**:

The knowledge gradient (KG) algorithm is a popular and effective algorithm for the best arm identification (BAI) problem. Due to the complex calculation of KG, theoretical analysis of this algorithm is difficult, and existing results are mostly about the asymptotic performance of it, e.g., consistency, asymptotic sample allocation, etc. In this research, we present new theoretical results about the finite-time performance of the KG algorithm. Under independent and normally distributed rewards, we derive lower bounds and upper bounds for the probability of error and simple regret of the algorithm. With these bounds, existing asymptotic results become simple corollaries. We also show the performance of the algorithm for the multi-armed bandit (MAB) problem. These developments not only extend the existing analysis of the KG algorithm, but can also be used to analyze other improvement-based algorithms. Last, we use numerical experiments to further demonstrate the finite-time behavior of the KG algorithm.

----

## [558] Phasic Self-Imitative Reduction for Sparse-Reward Goal-Conditioned Reinforcement Learning

**Authors**: *Yunfei Li, Tian Gao, Jiaqi Yang, Huazhe Xu, Yi Wu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22g.html](https://proceedings.mlr.press/v162/li22g.html)

**Abstract**:

It has been a recent trend to leverage the power of supervised learning (SL) towards more effective reinforcement learning (RL) methods. We propose a novel phasic solution by alternating online RL and offline SL for tackling sparse-reward goal-conditioned problems. In the online phase, we perform RL training and collect rollout data while in the offline phase, we perform SL on those successful trajectories from the dataset. To further improve sample efficiency, we adopt additional techniques in the online phase including task reduction to generate more feasible trajectories and a value-difference-based intrinsic reward to alleviate the sparse-reward issue. We call this overall framework, PhAsic self-Imitative Reduction (PAIR). PAIR is compatible with various online and offline RL methods and substantially outperforms both non-phasic RL and phasic SL baselines on sparse-reward robotic control problems, including a particularly challenging stacking task. PAIR is the first RL method that learns to stack 6 cubes with only 0/1 success rewards from scratch.

----

## [559] G2CN: Graph Gaussian Convolution Networks with Concentrated Graph Filters

**Authors**: *Mingjie Li, Xiaojun Guo, Yifei Wang, Yisen Wang, Zhouchen Lin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22h.html](https://proceedings.mlr.press/v162/li22h.html)

**Abstract**:

Recently, linear GCNs have shown competitive performance against non-linear ones with less computation cost, and the key lies in their propagation layers. Spectral analysis has been widely adopted in designing and analyzing existing graph propagations. Nevertheless, we notice that existing spectral analysis fails to explain why existing graph propagations with the same global tendency, such as low-pass or high-pass, still yield very different results. Motivated by this situation, we develop a new framework for spectral analysis in this paper called concentration analysis. In particular, we propose three attributes: concentration centre, maximum response, and bandwidth for our analysis. Through a dissection of the limitations of existing graph propagations via the above analysis, we propose a new kind of propagation layer, Graph Gaussian Convolution Networks (G^2CN), in which the three properties are decoupled and the whole structure becomes more flexible and applicable to different kinds of graphs. Extensive experiments show that we can obtain state-of-the-art performance on heterophily and homophily datasets with our proposed G^2CN.

----

## [560] Decomposing Temporal High-Order Interactions via Latent ODEs

**Authors**: *Shibo Li, Robert M. Kirby, Shandian Zhe*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22i.html](https://proceedings.mlr.press/v162/li22i.html)

**Abstract**:

High-order interactions between multiple objects are common in real-world applications. Although tensor decomposition is a popular framework for high-order interaction analysis and prediction, most methods cannot well exploit the valuable timestamp information in data. The existent methods either discard the timestamps or convert them into discrete steps or use over-simplistic decomposition models. As a result, these methods might not be capable enough of capturing complex, fine-grained temporal dynamics or making accurate predictions for long-term interaction results. To overcome these limitations, we propose a novel Temporal High-order Interaction decompoSition model based on Ordinary Differential Equations (THIS-ODE). We model the time-varying interaction result with a latent ODE. To capture the complex temporal dynamics, we use a neural network (NN) to learn the time derivative of the ODE state. We use the representation of the interaction objects to model the initial value of the ODE and to constitute a part of the NN input to compute the state. In this way, the temporal relationships of the participant objects can be estimated and encoded into their representations. 	For tractable and scalable inference, we use forward sensitivity analysis to efficiently compute the gradient of ODE state, based on which we use integral transform to develop a stochastic mini-batch learning algorithm. We demonstrate the advantage of our approach in simulation and four real-world applications.

----

## [561] Neural Inverse Transform Sampler

**Authors**: *Henry Li, Yuval Kluger*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22j.html](https://proceedings.mlr.press/v162/li22j.html)

**Abstract**:

Any explicit functional representation $f$ of a density is hampered by two main obstacles when we wish to use it as a generative model: designing $f$ so that sampling is fast, and estimating $Z = \int f$ so that $Z^{-1}f$ integrates to 1. This becomes increasingly complicated as $f$ itself becomes complicated. In this paper, we show that when modeling one-dimensional conditional densities with a neural network, $Z$ can be exactly and efficiently computed by letting the network represent the cumulative distribution function of a target density, and applying a generalized fundamental theorem of calculus. We also derive a fast algorithm for sampling from the resulting representation by the inverse transform method. By extending these principles to higher dimensions, we introduce the \textbf{Neural Inverse Transform Sampler (NITS)}, a novel deep learning framework for modeling and sampling from general, multidimensional, compactly-supported probability densities. NITS is a highly expressive density estimator that boasts end-to-end differentiability, fast sampling, and exact and cheap likelihood evaluation. We demonstrate the applicability of NITS by applying it to realistic, high-dimensional density estimation tasks: likelihood-based generative modeling on the CIFAR-10 dataset, and density estimation on the UCI suite of benchmark datasets, where NITS produces compelling results rivaling or surpassing the state of the art.

----

## [562] PLATINUM: Semi-Supervised Model Agnostic Meta-Learning using Submodular Mutual Information

**Authors**: *Changbin Li, Suraj Kothawade, Feng Chen, Rishabh K. Iyer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22k.html](https://proceedings.mlr.press/v162/li22k.html)

**Abstract**:

Few-shot classification (FSC) requires training models using a few (typically one to five) data points per class. Meta-learning has proven to be able to learn a parametrized model for FSC by training on various other classification tasks. In this work, we propose PLATINUM (semi-suPervised modeL Agnostic meTa learnIng usiNg sUbmodular Mutual information ), a novel semi-supervised model agnostic meta learning framework that uses the submodular mutual in- formation (SMI) functions to boost the perfor- mance of FSC. PLATINUM leverages unlabeled data in the inner and outer loop using SMI func- tions during meta-training and obtains richer meta- learned parameterizations. We study the per- formance of PLATINUM in two scenarios - 1) where the unlabeled data points belong to the same set of classes as the labeled set of a cer- tain episode, and 2) where there exist out-of- distribution classes that do not belong to the la- beled set. We evaluate our method on various settings on the miniImageNet, tieredImageNet and CIFAR-FS datasets. Our experiments show that PLATINUM outperforms MAML and semi- supervised approaches like pseduo-labeling for semi-supervised FSC, especially for small ratio of labeled to unlabeled samples.

----

## [563] Deconfounded Value Decomposition for Multi-Agent Reinforcement Learning

**Authors**: *Jiahui Li, Kun Kuang, Baoxiang Wang, Furui Liu, Long Chen, Changjie Fan, Fei Wu, Jun Xiao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22l.html](https://proceedings.mlr.press/v162/li22l.html)

**Abstract**:

Value decomposition (VD) methods have been widely used in cooperative multi-agent reinforcement learning (MARL), where credit assignment plays an important role in guiding the agents’ decentralized execution. In this paper, we investigate VD from a novel perspective of causal inference. We first show that the environment in existing VD methods is an unobserved confounder as the common cause factor of the global state and the joint value function, which leads to the confounding bias on learning credit assignment. We then present our approach, deconfounded value decomposition (DVD), which cuts off the backdoor confounding path from the global state to the joint value function. The cut is implemented by introducing the trajectory graph, which depends only on the local trajectories, as a proxy confounder. DVD is general enough to be applied to various VD methods, and extensive experiments show that DVD can consistently achieve significant performance gains over different state-of-the-art VD methods on StarCraft II and MACO benchmarks.

----

## [564] C-MinHash: Improving Minwise Hashing with Circulant Permutation

**Authors**: *Xiaoyun Li, Ping Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22m.html](https://proceedings.mlr.press/v162/li22m.html)

**Abstract**:

Minwise hashing (MinHash) is an important and practical algorithm for generating random hashes to approximate the Jaccard (resemblance) similarity in massive binary (0/1) data. The basic theory of MinHash requires applying hundreds or even thousands of independent random permutations to each data vector in the dataset, in order to obtain reliable results for (e.g.,) building large-scale learning models or approximate near neighbor search. In this paper, we propose Circulant MinHash (C-MinHash) and provide the surprising theoretical results that using only two independent random permutations in a circulant manner leads to uniformly smaller Jaccard estimation variance than that of the classical MinHash with K independent permutations. Experiments are conducted to show the effectiveness of the proposed method. We also propose a more convenient C-MinHash variant which reduces two permutations to just one, with extensive numerical results to validate that it achieves essentially the same estimation accuracy as using two permutations.

----

## [565] BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

**Authors**: *Junnan Li, Dongxu Li, Caiming Xiong, Steven C. H. Hoi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22n.html](https://proceedings.mlr.press/v162/li22n.html)

**Abstract**:

Vision-Language Pre-training (VLP) has advanced the performance for many vision-language tasks. However, most existing pre-trained models only excel in either understanding-based tasks or generation-based tasks. Furthermore, performance improvement has been largely achieved by scaling up the dataset with noisy image-text pairs collected from the web, which is a suboptimal source of supervision. In this paper, we propose BLIP, a new VLP framework which transfers flexibly to both vision-language understanding and generation tasks. BLIP effectively utilizes the noisy web data by bootstrapping the captions, where a captioner generates synthetic captions and a filter removes the noisy ones. We achieve state-of-the-art results on a wide range of vision-language tasks, such as image-text retrieval (+2.7% in average recall@1), image captioning (+2.8% in CIDEr), and VQA (+1.6% in VQA score). BLIP also demonstrates strong generalization ability when directly transferred to video-language tasks in a zero-shot manner. Code and models are available at https://github.com/salesforce/BLIP.

----

## [566] Restarted Nonconvex Accelerated Gradient Descent: No More Polylogarithmic Factor in the O(ε-7/4) Complexity

**Authors**: *Huan Li, Zhouchen Lin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22o.html](https://proceedings.mlr.press/v162/li22o.html)

**Abstract**:

This paper studies the accelerated gradient descent for general nonconvex problems under the gradient Lipschitz and Hessian Lipschitz assumptions. We establish that a simple restarted accelerated gradient descent (AGD) finds an $\epsilon$-approximate first-order stationary point in $O(\epsilon^{-7/4})$ gradient computations with simple proofs. Our complexity does not hide any polylogarithmic factors, and thus it improves over the state-of-the-art one by the $O(\log\frac{1}{\epsilon})$ factor. Our simple algorithm only consists of Nesterov’s classical AGD and a restart mechanism, and it does not need the negative curvature exploitation or the optimization of regularized surrogate functions. Technically, our simple proof does not invoke the analysis for the strongly convex AGD, which is crucial to remove the $O(\log\frac{1}{\epsilon})$ factor.

----

## [567] Achieving Fairness at No Utility Cost via Data Reweighing with Influence

**Authors**: *Peizhao Li, Hongfu Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22p.html](https://proceedings.mlr.press/v162/li22p.html)

**Abstract**:

With the fast development of algorithmic governance, fairness has become a compulsory property for machine learning models to suppress unintentional discrimination. In this paper, we focus on the pre-processing aspect for achieving fairness, and propose a data reweighing approach that only adjusts the weight for samples in the training phase. Different from most previous reweighing methods which usually assign a uniform weight for each (sub)group, we granularly model the influence of each training sample with regard to fairness-related quantity and predictive utility, and compute individual weights based on influence under the constraints from both fairness and utility. Experimental results reveal that previous methods achieve fairness at a non-negligible cost of utility, while as a significant advantage, our approach can empirically release the tradeoff and obtain cost-free fairness for equal opportunity. We demonstrate the cost-free fairness through vanilla classifiers and standard training processes, compared to baseline methods on multiple real-world tabular datasets. Code available at https://github.com/brandeis-machine-learning/influence-fairness.

----

## [568] High Probability Guarantees for Nonconvex Stochastic Gradient Descent with Heavy Tails

**Authors**: *Shaojie Li, Yong Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22q.html](https://proceedings.mlr.press/v162/li22q.html)

**Abstract**:

Stochastic gradient descent (SGD) is the workhorse in modern machine learning and data-driven optimization. Despite its popularity, existing theoretical guarantees for SGD are mainly derived in expectation and for convex learning problems. High probability guarantees of nonconvex SGD are scarce, and typically rely on “light-tail” noise assumptions and study the optimization and generalization performance separately. In this paper, we develop high probability bounds for nonconvex SGD with a joint perspective of optimization and generalization performance. Instead of the light tail assumption, we consider the gradient noise following a heavy-tailed sub-Weibull distribution, a novel class generalizing the sub-Gaussian and sub-Exponential families to potentially heavier-tailed distributions. Under these complicated settings, we first present high probability bounds with best-known rates in general nonconvex learning, then move to nonconvex learning with a gradient dominance curvature condition, for which we improve the learning guarantees to fast rates. We further obtain sharper learning guarantees by considering a mild Bernstein-type noise condition. Our analysis also reveals the effect of trade-offs between the optimization and generalization performance under different conditions. In the last, we show that gradient clipping can be employed to remove the bounded gradient-type assumptions. Additionally, in this case, the stepsize of SGD is completely oblivious to the knowledge of smoothness.

----

## [569] MetAug: Contrastive Learning via Meta Feature Augmentation

**Authors**: *Jiangmeng Li, Wenwen Qiang, Changwen Zheng, Bing Su, Hui Xiong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22r.html](https://proceedings.mlr.press/v162/li22r.html)

**Abstract**:

What matters for contrastive learning? We argue that contrastive learning heavily relies on informative features, or “hard” (positive or negative) features. Early works include more informative features by applying complex data augmentations and large batch size or memory bank, and recent works design elaborate sampling approaches to explore informative features. The key challenge toward exploring such features is that the source multi-view data is generated by applying random data augmentations, making it infeasible to always add useful information in the augmented data. Consequently, the informativeness of features learned from such augmented data is limited. In response, we propose to directly augment the features in latent space, thereby learning discriminative representations without a large amount of input data. We perform a meta learning technique to build the augmentation generator that updates its network parameters by considering the performance of the encoder. However, insufficient input data may lead the encoder to learn collapsed features and therefore malfunction the augmentation generator. A new margin-injected regularization is further added in the objective function to avoid the encoder learning a degenerate mapping. To contrast all features in one gradient back-propagation step, we adopt the proposed optimization-driven unified contrastive loss instead of the conventional contrastive loss. Empirically, our method achieves state-of-the-art results on several benchmark datasets.

----

## [570] PMIC: Improving Multi-Agent Reinforcement Learning with Progressive Mutual Information Collaboration

**Authors**: *Pengyi Li, Hongyao Tang, Tianpei Yang, Xiaotian Hao, Tong Sang, Yan Zheng, Jianye Hao, Matthew E. Taylor, Wenyuan Tao, Zhen Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22s.html](https://proceedings.mlr.press/v162/li22s.html)

**Abstract**:

Learning to collaborate is critical in Multi-Agent Reinforcement Learning (MARL). Previous works promote collaboration by maximizing the correlation of agents’ behaviors, which is typically characterized by Mutual Information (MI) in different forms. However, we reveal sub-optimal collaborative behaviors also emerge with strong correlations, and simply maximizing the MI can, surprisingly, hinder the learning towards better collaboration. To address this issue, we propose a novel MARL framework, called Progressive Mutual Information Collaboration (PMIC), for more effective MI-driven collaboration. PMIC uses a new collaboration criterion measured by the MI between global states and joint actions. Based on this criterion, the key idea of PMIC is maximizing the MI associated with superior collaborative behaviors and minimizing the MI associated with inferior ones. The two MI objectives play complementary roles by facilitating better collaborations while avoiding falling into sub-optimal ones. Experiments on a wide range of MARL benchmarks show the superior performance of PMIC compared with other algorithms.

----

## [571] CerDEQ: Certifiable Deep Equilibrium Model

**Authors**: *Mingjie Li, Yisen Wang, Zhouchen Lin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22t.html](https://proceedings.mlr.press/v162/li22t.html)

**Abstract**:

Recently, certifiable robust training methods via bound propagation have been proposed for training neural networks with certifiable robustness guarantees. However, no neural architectures with regular convolution and linear layers perform better in the certifiable training than the plain CNNs, since the output bounds for the deep explicit models increase quickly as their depth increases. And such a phenomenon significantly hinders certifiable training. Meanwhile, the Deep Equilibrium Model (DEQ) is more representative and robust due to their equivalent infinite depth and controllable global Lipschitz. But no work has been proposed to explore whether DEQ can show advantages in certified training. In this work, we aim to tackle the problem of DEQ’s certified training. To obtain the output bound based on the bound propagation scheme in the implicit model, we first involve the adjoint DEQ for bound approximation. Furthermore, we also use the weight orthogonalization method and other tricks specified for DEQ to stabilize the certifiable training. With our approach, we can obtain the certifiable DEQ called CerDEQ. Our CerDEQ can achieve state-of-the-art performance compared with models using regular convolution and linear layers on $\ell_\infty$ tasks with $\epsilon=8/255$: $64.72%$ certified error for CIFAR-$10$ and $94.45%$ certified error for Tiny ImageNet.

----

## [572] Generalization Guarantee of Training Graph Convolutional Networks with Graph Topology Sampling

**Authors**: *Hongkang Li, Meng Wang, Sijia Liu, Pin-Yu Chen, Jinjun Xiong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22u.html](https://proceedings.mlr.press/v162/li22u.html)

**Abstract**:

Graph convolutional networks (GCNs) have recently achieved great empirical success in learning graph-structured data. To address its scalability issue due to the recursive embedding of neighboring features, graph topology sampling has been proposed to reduce the memory and computational cost of training GCNs, and it has achieved comparable test performance to those without topology sampling in many empirical studies. To the best of our knowledge, this paper provides the first theoretical justification of graph topology sampling in training (up to) three-layer GCNs for semi-supervised node classification. We formally characterize some sufficient conditions on graph topology sampling such that GCN training leads to diminishing generalization error. Moreover, our method tackles the non-convex interaction of weights across layers, which is under-explored in the existing theoretical analyses of GCNs. This paper characterizes the impact of graph structures and topology sampling on the generalization performance and sample complexity explicitly, and the theoretical findings are also justified through numerical experiments.

----

## [573] Let Invariant Rationale Discovery Inspire Graph Contrastive Learning

**Authors**: *Sihang Li, Xiang Wang, An Zhang, Yingxin Wu, Xiangnan He, Tat-Seng Chua*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22v.html](https://proceedings.mlr.press/v162/li22v.html)

**Abstract**:

Leading graph contrastive learning (GCL) methods perform graph augmentations in two fashions: (1) randomly corrupting the anchor graph, which could cause the loss of semantic information, or (2) using domain knowledge to maintain salient features, which undermines the generalization to other domains. Taking an invariance look at GCL, we argue that a high-performing augmentation should preserve the salient semantics of anchor graphs regarding instance-discrimination. To this end, we relate GCL with invariant rationale discovery, and propose a new framework, Rationale-aware Graph Contrastive Learning (RGCL). Specifically, without supervision signals, RGCL uses a rationale generator to reveal salient features about graph instance-discrimination as the rationale, and then creates rationale-aware views for contrastive learning. This rationale-aware pre-training scheme endows the backbone model with the powerful representation ability, further facilitating the fine-tuning on downstream tasks. On MNIST-Superpixel and MUTAG datasets, visual inspections on the discovered rationales showcase that the rationale generator successfully captures the salient features (\ie distinguishing semantic nodes in graphs). On biochemical molecule and social network benchmark datasets, the state-of-the-art performance of RGCL demonstrates the effectiveness of rationale-aware views for contrastive learning. Our codes are available at https://github.com/lsh0520/RGCL.

----

## [574] Difference Advantage Estimation for Multi-Agent Policy Gradients

**Authors**: *Yueheng Li, Guangming Xie, Zongqing Lu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22w.html](https://proceedings.mlr.press/v162/li22w.html)

**Abstract**:

Multi-agent policy gradient methods in centralized training with decentralized execution recently witnessed many progresses. During centralized training, multi-agent credit assignment is crucial, which can substantially promote learning performance. However, explicit multi-agent credit assignment in multi-agent policy gradient methods still receives less attention. In this paper, we investigate multi-agent credit assignment induced by reward shaping and provide a theoretical understanding in terms of its credit assignment and policy bias. Based on this, we propose an exponentially weighted advantage estimator, which is analogous to GAE, to enable multi-agent credit assignment while allowing the tradeoff with policy bias. Empirical results show that our approach can successfully perform effective multi-agent credit assignment, and thus substantially outperforms other advantage estimators.

----

## [575] Private Adaptive Optimization with Side information

**Authors**: *Tian Li, Manzil Zaheer, Sashank J. Reddi, Virginia Smith*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22x.html](https://proceedings.mlr.press/v162/li22x.html)

**Abstract**:

Adaptive optimization methods have become the default solvers for many machine learning tasks. Unfortunately, the benefits of adaptivity may degrade when training with differential privacy, as the noise added to ensure privacy reduces the effectiveness of the adaptive preconditioner. To this end, we propose AdaDPS, a general framework that uses non-sensitive side information to precondition the gradients, allowing the effective use of adaptive methods in private settings. We formally show AdaDPS reduces the amount of noise needed to achieve similar privacy guarantees, thereby improving optimization performance. Empirically, we leverage simple and readily available side information to explore the performance of AdaDPS in practice, comparing to strong baselines in both centralized and federated settings. Our results show that AdaDPS improves accuracy by 7.7% (absolute) on average—yielding state-of-the-art privacy-utility trade-offs on large-scale text and image benchmarks.

----

## [576] Permutation Search of Tensor Network Structures via Local Sampling

**Authors**: *Chao Li, Junhua Zeng, Zerui Tao, Qibin Zhao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22y.html](https://proceedings.mlr.press/v162/li22y.html)

**Abstract**:

Recent works put much effort into tensor network structure search (TN-SS), aiming to select suitable tensor network (TN) structures, involving the TN-ranks, formats, and so on, for the decomposition or learning tasks. In this paper, we consider a practical variant of TN-SS, dubbed TN permutation search (TN-PS), in which we search for good mappings from tensor modes onto TN vertices (core tensors) for compact TN representations. We conduct a theoretical investigation of TN-PS and propose a practically-efficient algorithm to resolve the problem. Theoretically, we prove the counting and metric properties of search spaces of TN-PS, analyzing for the first time the impact of TN structures on these unique properties. Numerically, we propose a novel meta-heuristic algorithm, in which the searching is done by randomly sampling in a neighborhood established in our theory, and then recurrently updating the neighborhood until convergence. Numerical results demonstrate that the new algorithm can reduce the required model size of TNs in extensive benchmarks, implying the improvement in the expressive power of TNs. Furthermore, the computational cost for the new algorithm is significantly less than that in (Li and Sun, 2020).

----

## [577] Hessian-Free High-Resolution Nesterov Acceleration For Sampling

**Authors**: *Ruilin Li, Hongyuan Zha, Molei Tao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22z.html](https://proceedings.mlr.press/v162/li22z.html)

**Abstract**:

Nesterov’s Accelerated Gradient (NAG) for optimization has better performance than its continuous time limit (noiseless kinetic Langevin) when a finite step-size is employed (Shi et al., 2021). This work explores the sampling counterpart of this phenonemon and proposes a diffusion process, whose discretizations can yield accelerated gradient-based MCMC methods. More precisely, we reformulate the optimizer of NAG for strongly convex functions (NAG-SC) as a Hessian-Free High-Resolution ODE, change its high-resolution coefficient to a hyperparameter, inject appropriate noise, and discretize the resulting diffusion process. The acceleration effect of the new hyperparameter is quantified and it is not an artificial one created by time-rescaling. Instead, acceleration beyond underdamped Langevin in $W_2$ distance is quantitatively established for log-strongly-concave-and-smooth targets, at both the continuous dynamics level and the discrete algorithm level. Empirical experiments in both log-strongly-concave and multi-modal cases also numerically demonstrate this acceleration.

----

## [578] Double Sampling Randomized Smoothing

**Authors**: *Linyi Li, Jiawei Zhang, Tao Xie, Bo Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22aa.html](https://proceedings.mlr.press/v162/li22aa.html)

**Abstract**:

Neural networks (NNs) are known to be vulnerable against adversarial perturbations, and thus there is a line of work aiming to provide robustness certification for NNs, such as randomized smoothing, which samples smoothing noises from a certain distribution to certify the robustness for a smoothed classifier. However, as previous work shows, the certified robust radius in randomized smoothing suffers from scaling to large datasets ("curse of dimensionality"). To overcome this hurdle, we propose a Double Sampling Randomized Smoothing (DSRS) framework, which exploits the sampled probability from an additional smoothing distribution to tighten the robustness certification of the previous smoothed classifier. Theoretically, under mild assumptions, we prove that DSRS can certify $\Theta(\sqrt d)$ robust radius under $\ell_2$ norm where $d$ is the input dimension, which implies that DSRS may be able to break the curse of dimensionality of randomized smoothing. We instantiate DSRS for a generalized family of Gaussian smoothing and propose an efficient and sound computing method based on customized dual optimization considering sampling error. Extensive experiments on MNIST, CIFAR-10, and ImageNet verify our theory and show that DSRS certifies larger robust radii than existing baselines consistently under different settings. Code is available at https://github.com/llylly/DSRS.

----

## [579] HousE: Knowledge Graph Embedding with Householder Parameterization

**Authors**: *Rui Li, Jianan Zhao, Chaozhuo Li, Di He, Yiqi Wang, Yuming Liu, Hao Sun, Senzhang Wang, Weiwei Deng, Yanming Shen, Xing Xie, Qi Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22ab.html](https://proceedings.mlr.press/v162/li22ab.html)

**Abstract**:

The effectiveness of knowledge graph embedding (KGE) largely depends on the ability to model intrinsic relation patterns and mapping properties. However, existing approaches can only capture some of them with insufficient modeling capacity. In this work, we propose a more powerful KGE framework named HousE, which involves a novel parameterization based on two kinds of Householder transformations: (1) Householder rotations to achieve superior capacity of modeling relation patterns; (2) Householder projections to handle sophisticated relation mapping properties. Theoretically, HousE is capable of modeling crucial relation patterns and mapping properties simultaneously. Besides, HousE is a generalization of existing rotation-based models while extending the rotations to high-dimensional spaces. Empirically, HousE achieves new state-of-the-art performance on five benchmark datasets. Our code is available at https://github.com/anrep/HousE.

----

## [580] Learning Multiscale Transformer Models for Sequence Generation

**Authors**: *Bei Li, Tong Zheng, Yi Jing, Chengbo Jiao, Tong Xiao, Jingbo Zhu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22ac.html](https://proceedings.mlr.press/v162/li22ac.html)

**Abstract**:

Multiscale feature hierarchies have been witnessed the success in the computer vision area. This further motivates researchers to design multiscale Transformer for natural language processing, mostly based on the self-attention mechanism. For example, restricting the receptive field across heads or extracting local fine-grained features via convolutions. However, most of existing works directly modeled local features but ignored the word-boundary information. This results in redundant and ambiguous attention distributions, which lacks of interpretability. In this work, we define those scales in different linguistic units, including sub-words, words and phrases. We built a multiscale Transformer model by establishing relationships among scales based on word-boundary information and phrase-level prior knowledge. The proposed \textbf{U}niversal \textbf{M}ulti\textbf{S}cale \textbf{T}ransformer, namely \textsc{Umst}, was evaluated on two sequence generation tasks. Notably, it yielded consistent performance gains over the strong baseline on several test sets without sacrificing the efficiency.

----

## [581] Finding Global Homophily in Graph Neural Networks When Meeting Heterophily

**Authors**: *Xiang Li, Renyu Zhu, Yao Cheng, Caihua Shan, Siqiang Luo, Dongsheng Li, Weining Qian*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/li22ad.html](https://proceedings.mlr.press/v162/li22ad.html)

**Abstract**:

We investigate graph neural networks on graphs with heterophily. Some existing methods amplify a node’s neighborhood with multi-hop neighbors to include more nodes with homophily. However, it is a significant challenge to set personalized neighborhood sizes for different nodes. Further, for other homophilous nodes excluded in the neighborhood, they are ignored for information aggregation. To address these problems, we propose two models GloGNN and GloGNN++, which generate a node’s embedding by aggregating information from global nodes in the graph. In each layer, both models learn a coefficient matrix to capture the correlations between nodes, based on which neighborhood aggregation is performed. The coefficient matrix allows signed values and is derived from an optimization problem that has a closed-form solution. We further accelerate neighborhood aggregation and derive a linear time complexity. We theoretically explain the models’ effectiveness by proving that both the coefficient matrix and the generated node embedding matrix have the desired grouping effect. We conduct extensive experiments to compare our models against 11 other competitors on 15 benchmark datasets in a wide range of domains, scales and graph heterophilies. Experimental results show that our methods achieve superior performance and are also very efficient.

----

## [582] Fat-Tailed Variational Inference with Anisotropic Tail Adaptive Flows

**Authors**: *Feynman T. Liang, Michael W. Mahoney, Liam Hodgkinson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liang22a.html](https://proceedings.mlr.press/v162/liang22a.html)

**Abstract**:

While fat-tailed densities commonly arise as posterior and marginal distributions in robust models and scale mixtures, they present a problematic scenario when Gaussian-based variational inference fails to accurately capture tail decay. We first improve previous theory on tails of Lipschitz flows by quantifying how they affect the rate of tail decay and expanding the theory to non-Lipschitz polynomial flows. Next, we develop an alternative theory for multivariate tail parameters which is sensitive to tail-anisotropy. In doing so, we unveil a fundamental problem which plagues many existing flow-based methods: they can only model tail-isotropic distributions (i.e., distributions having the same tail parameter in every direction). To mitigate this and enable modeling of tail-anisotropic targets, we propose anisotropic tail-adaptive flows (ATAF). Experimental results confirm ATAF on both synthetic and real-world targets is competitive with prior work while also exhibiting appropriate tail-anisotropy.

----

## [583] Exploring and Exploiting Hubness Priors for High-Quality GAN Latent Sampling

**Authors**: *Yuanbang Liang, Jing Wu, Yu-Kun Lai, Yipeng Qin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liang22b.html](https://proceedings.mlr.press/v162/liang22b.html)

**Abstract**:

Despite the extensive studies on Generative Adversarial Networks (GANs), how to reliably sample high-quality images from their latent spaces remains an under-explored topic. In this paper, we propose a novel GAN latent sampling method by exploring and exploiting the hubness priors of GAN latent distributions. Our key insight is that the high dimensionality of the GAN latent space will inevitably lead to the emergence of hub latents that usually have much larger sampling densities than other latents in the latent space. As a result, these hub latents are better trained and thus contribute more to the synthesis of high-quality images. Unlike the a posterior "cherry-picking", our method is highly efficient as it is an a priori method that identifies high-quality latents before the synthesis of images. Furthermore, we show that the well-known but purely empirical truncation trick is a naive approximation to the central clustering effect of hub latents, which not only uncovers the rationale of the truncation trick, but also indicates the superiority and fundamentality of our method. Extensive experimental results demonstrate the effectiveness of the proposed method. Our code is available at: https://github.com/Byronliang8/HubnessGANSampling.

----

## [584] Reducing Variance in Temporal-Difference Value Estimation via Ensemble of Deep Networks

**Authors**: *Litian Liang, Yaosheng Xu, Stephen McAleer, Dailin Hu, Alexander Ihler, Pieter Abbeel, Roy Fox*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liang22c.html](https://proceedings.mlr.press/v162/liang22c.html)

**Abstract**:

In temporal-difference reinforcement learning algorithms, variance in value estimation can cause instability and overestimation of the maximal target value. Many algorithms have been proposed to reduce overestimation, including several recent ensemble methods, however none have shown success in sample-efficient learning through addressing estimation variance as the root cause of overestimation. In this paper, we propose MeanQ, a simple ensemble method that estimates target values as ensemble means. Despite its simplicity, MeanQ shows remarkable sample efficiency in experiments on the Atari Learning Environment benchmark. Importantly, we find that an ensemble of size 5 sufficiently reduces estimation variance to obviate the lagging target network, eliminating it as a source of bias and further gaining sample efficiency. We justify intuitively and empirically the design choices in MeanQ, including the necessity of independent experience sampling. On a set of 26 benchmark Atari environments, MeanQ outperforms all tested baselines, including the best available baseline, SUNRISE, at 100K interaction steps in 16/26 environments, and by 68% on average. MeanQ also outperforms Rainbow DQN at 500K steps in 21/26 environments, and by 49% on average, and achieves average human-level performance using 200K ($\pm$100K) interaction steps. Our implementation is available at https://github.com/indylab/MeanQ.

----

## [585] TSPipe: Learn from Teacher Faster with Pipelines

**Authors**: *Hwijoon Lim, Yechan Kim, Sukmin Yun, Jinwoo Shin, Dongsu Han*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lim22a.html](https://proceedings.mlr.press/v162/lim22a.html)

**Abstract**:

The teacher-student (TS) framework, training a (student) network by utilizing an auxiliary superior (teacher) network, has been adopted as a popular training paradigm in many machine learning schemes, since the seminal work—Knowledge distillation (KD) for model compression and transfer learning. Many recent self-supervised learning (SSL) schemes also adopt the TS framework, where teacher networks are maintained as the moving average of student networks, called the momentum networks. This paper presents TSPipe, a pipelined approach to accelerate the training process of any TS frameworks including KD and SSL. Under the observation that the teacher network does not need a backward pass, our main idea is to schedule the computation of the teacher and student network separately, and fully utilize the GPU during training by interleaving the computations of the two networks and relaxing their dependencies. In case the teacher network requires a momentum update, we use delayed parameter updates only on the teacher network to attain high model accuracy. Compared to existing pipeline parallelism schemes, which sacrifice either training throughput or model accuracy, TSPipe provides better performance trade-offs, achieving up to 12.15x higher throughput.

----

## [586] Order Constraints in Optimal Transport

**Authors**: *Yu Chin, Fabian Lim, Laura Wynter, Shiau Hong Lim*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lim22b.html](https://proceedings.mlr.press/v162/lim22b.html)

**Abstract**:

Optimal transport is a framework for comparing measures whereby a cost is incurred for transporting one measure to another. Recent works have aimed to improve optimal transport plans through the introduction of various forms of structure. We introduce novel order constraints into the optimal transport formulation to allow for the incorporation of structure. We define an efficient method for obtaining explainable solutions to the new formulation that scales far better than standard approaches. The theoretical properties of the method are provided. We demonstrate experimentally that order constraints improve explainability using the e-SNLI (Stanford Natural Language Inference) dataset that includes human-annotated rationales as well as on several image color transfer examples.

----

## [587] Flow-Guided Sparse Transformer for Video Deblurring

**Authors**: *Jing Lin, Yuanhao Cai, Xiaowan Hu, Haoqian Wang, Youliang Yan, Xueyi Zou, Henghui Ding, Yulun Zhang, Radu Timofte, Luc Van Gool*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22a.html](https://proceedings.mlr.press/v162/lin22a.html)

**Abstract**:

Exploiting similar and sharper scene patches in spatio-temporal neighborhoods is critical for video deblurring. However, CNN-based methods show limitations in capturing long-range dependencies and modeling non-local self-similarity. In this paper, we propose a novel framework, Flow-Guided Sparse Transformer (FGST), for video deblurring. In FGST, we customize a self-attention module, Flow-Guided Sparse Window-based Multi-head Self-Attention (FGSW-MSA). For each $query$ element on the blurry reference frame, FGSW-MSA enjoys the guidance of the estimated optical flow to globally sample spatially sparse yet highly related $key$ elements corresponding to the same scene patch in neighboring frames. Besides, we present a Recurrent Embedding (RE) mechanism to transfer information from past frames and strengthen long-range temporal dependencies. Comprehensive experiments demonstrate that our proposed FGST outperforms state-of-the-art (SOTA) methods on both DVD and GOPRO datasets and yields visually pleasant results in real video deblurring. https://github.com/linjing7/VR-Baseline

----

## [588] Federated Learning with Positive and Unlabeled Data

**Authors**: *Xinyang Lin, Hanting Chen, Yixing Xu, Chao Xu, Xiaolin Gui, Yiping Deng, Yunhe Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22b.html](https://proceedings.mlr.press/v162/lin22b.html)

**Abstract**:

We study the problem of learning from positive and unlabeled (PU) data in the federated setting, where each client only labels a little part of their dataset due to the limitation of resources and time. Different from the settings in traditional PU learning where the negative class consists of a single class, the negative samples which cannot be identified by a client in the federated setting may come from multiple classes which are unknown to the client. Therefore, existing PU learning methods can be hardly applied in this situation. To address this problem, we propose a novel framework, namely Federated learning with Positive and Unlabeled data (FedPU), to minimize the expected risk of multiple negative classes by leveraging the labeled data in other clients. We theoretically analyze the generalization bound of the proposed FedPU. Empirical experiments show that the FedPU can achieve much better performance than conventional supervised and semi-supervised federated learning methods.

----

## [589] Decentralized Online Convex Optimization in Networked Systems

**Authors**: *Yiheng Lin, Judy Gan, Guannan Qu, Yash Kanoria, Adam Wierman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22c.html](https://proceedings.mlr.press/v162/lin22c.html)

**Abstract**:

We study the problem of networked online convex optimization, where each agent individually decides on an action at every time step and agents cooperatively seek to minimize the total global cost over a finite horizon. The global cost is made up of three types of local costs: convex node costs, temporal interaction costs, and spatial interaction costs. In deciding their individual action at each time, an agent has access to predictions of local cost functions for the next $k$ time steps in an $r$-hop neighborhood. Our work proposes a novel online algorithm, Localized Predictive Control (LPC), which generalizes predictive control to multi-agent systems. We show that LPC achieves a competitive ratio of $1 + \tilde{O}(\rho_T^k) + \tilde{O}(\rho_S^r)$ in an adversarial setting, where $\rho_T$ and $\rho_S$ are constants in $(0, 1)$ that increase with the relative strength of temporal and spatial interaction costs, respectively. This is the first competitive ratio bound on decentralized predictive control for networked online convex optimization. Further, we show that the dependence on $k$ and $r$ in our results is near optimal by lower bounding the competitive ratio of any decentralized online algorithm.

----

## [590] Unsupervised Flow-Aligned Sequence-to-Sequence Learning for Video Restoration

**Authors**: *Jing Lin, Xiaowan Hu, Yuanhao Cai, Haoqian Wang, Youliang Yan, Xueyi Zou, Yulun Zhang, Luc Van Gool*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22d.html](https://proceedings.mlr.press/v162/lin22d.html)

**Abstract**:

How to properly model the inter-frame relation within the video sequence is an important but unsolved challenge for video restoration (VR). In this work, we propose an unsupervised flow-aligned sequence-to-sequence model (S2SVR) to address this problem. On the one hand, the sequence-to-sequence model, which has proven capable of sequence modeling in the field of natural language processing, is explored for the first time in VR. Optimized serialization modeling shows potential in capturing long-range dependencies among frames. On the other hand, we equip the sequence-to-sequence model with an unsupervised optical flow estimator to maximize its potential. The flow estimator is trained with our proposed unsupervised distillation loss, which can alleviate the data discrepancy and inaccurate degraded optical flow issues of previous flow-based methods. With reliable optical flow, we can establish accurate correspondence among multiple frames, narrowing the domain difference between 1D language and 2D misaligned frames and improving the potential of the sequence-to-sequence model. S2SVR shows superior performance in multiple VR tasks, including video deblurring, video super-resolution, and compressed video quality enhancement. https://github.com/linjing7/VR-Baseline

----

## [591] Constrained Gradient Descent: A Powerful and Principled Evasion Attack Against Neural Networks

**Authors**: *Weiran Lin, Keane Lucas, Lujo Bauer, Michael K. Reiter, Mahmood Sharif*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22e.html](https://proceedings.mlr.press/v162/lin22e.html)

**Abstract**:

We propose new, more efficient targeted white-box attacks against deep neural networks. Our attacks better align with the attacker’s goal: (1) tricking a model to assign higher probability to the target class than to any other class, while (2) staying within an $\epsilon$-distance of the attacked input. First, we demonstrate a loss function that explicitly encodes (1) and show that Auto-PGD finds more attacks with it. Second, we propose a new attack method, Constrained Gradient Descent (CGD), using a refinement of our loss function that captures both (1) and (2). CGD seeks to satisfy both attacker objectives—misclassification and bounded $\ell_{p}$-norm—in a principled manner, as part of the optimization, instead of via ad hoc post-processing techniques (e.g., projection or clipping). We show that CGD is more successful on CIFAR10 (0.9–4.2%) and ImageNet (8.6–13.6%) than state-of-the-art attacks while consuming less time (11.4–18.8%). Statistical tests confirm that our attack outperforms others against leading defenses on different datasets and values of $\epsilon$.

----

## [592] Learning Augmented Binary Search Trees

**Authors**: *Honghao Lin, Tian Luo, David P. Woodruff*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22f.html](https://proceedings.mlr.press/v162/lin22f.html)

**Abstract**:

A treap is a classic randomized binary search tree data structure that is easy to implement and supports O(log n) expected time access. However, classic treaps do not take advantage of the input distribution or patterns in the input. Given recent advances in algorithms with predictions, we propose pairing treaps with machine advice to form a learning-augmented treap. We are the first to propose a learning-augmented data structure that supports binary search tree operations such as range-query and successor functionalities. With the assumption that we have access to advice from a frequency estimation oracle, we assign learned priorities to the nodes to better improve the treap’s structure. We theoretically analyze the learning-augmented treap’s performance under various input distributions and show that under those circumstances, our learning-augmented treap has stronger guarantees than classic treaps and other classic tree-based data structures. Further, we experimentally evaluate our learned treap on synthetic datasets and demonstrate a performance advantage over other search tree data structures. We also present experiments on real world datasets with known frequency estimation oracles and show improvements as well.

----

## [593] Online Nonsubmodular Minimization with Delayed Costs: From Full Information to Bandit Feedback

**Authors**: *Tianyi Lin, Aldo Pacchiano, Yaodong Yu, Michael I. Jordan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22g.html](https://proceedings.mlr.press/v162/lin22g.html)

**Abstract**:

Motivated by applications to online learning in sparse estimation and Bayesian optimization, we consider the problem of online unconstrained nonsubmodular minimization with delayed costs in both full information and bandit feedback settings. In contrast to previous works on online unconstrained submodular minimization, we focus on a class of nonsubmodular functions with special structure, and prove regret guarantees for several variants of the online and approximate online bandit gradient descent algorithms in static and delayed scenarios. We derive bounds for the agent’s regret in the full information and bandit feedback setting, even if the delay between choosing a decision and receiving the incurred cost is unbounded. Key to our approach is the notion of $(\alpha, \beta)$-regret and the extension of the generic convex relaxation model from \citet{El-2020-Optimal}, the analysis of which is of independent interest. We conduct and showcase several simulation studies to demonstrate the efficacy of our algorithms.

----

## [594] Measuring the Effect of Training Data on Deep Learning Predictions via Randomized Experiments

**Authors**: *Jinkun Lin, Anqi Zhang, Mathias Lécuyer, Jinyang Li, Aurojit Panda, Siddhartha Sen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lin22h.html](https://proceedings.mlr.press/v162/lin22h.html)

**Abstract**:

We develop a new, principled algorithm for estimating the contribution of training data points to the behavior of a deep learning model, such as a specific prediction it makes. Our algorithm estimates the AME, a quantity that measures the expected (average) marginal effect of adding a data point to a subset of the training data, sampled from a given distribution. When subsets are sampled from the uniform distribution, the AME reduces to the well-known Shapley value. Our approach is inspired by causal inference and randomized experiments: we sample different subsets of the training data to train multiple submodels, and evaluate each submodel’s behavior. We then use a LASSO regression to jointly estimate the AME of each data point, based on the subset compositions. Under sparsity assumptions ($k \ll N$ datapoints have large AME), our estimator requires only $O(k\log N)$ randomized submodel trainings, improving upon the best prior Shapley value estimators.

----

## [595] Interactively Learning Preference Constraints in Linear Bandits

**Authors**: *David Lindner, Sebastian Tschiatschek, Katja Hofmann, Andreas Krause*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lindner22a.html](https://proceedings.mlr.press/v162/lindner22a.html)

**Abstract**:

We study sequential decision-making with known rewards and unknown constraints, motivated by situations where the constraints represent expensive-to-evaluate human preferences, such as safe and comfortable driving behavior. We formalize the challenge of interactively learning about these constraints as a novel linear bandit problem which we call constrained linear best-arm identification. To solve this problem, we propose the Adaptive Constraint Learning (ACOL) algorithm. We provide an instance-dependent lower bound for constrained linear best-arm identification and show that ACOL’s sample complexity matches the lower bound in the worst-case. In the average case, ACOL’s sample complexity bound is still significantly tighter than bounds of simpler approaches. In synthetic experiments, ACOL performs on par with an oracle solution and outperforms a range of baselines. As an application, we consider learning constraints to represent human preferences in a driving simulation. ACOL is significantly more sample efficient than alternatives for this application. Further, we find that learning preferences as constraints is more robust to changes in the driving scenario than encoding the preferences directly in the reward function.

----

## [596] Delayed Reinforcement Learning by Imitation

**Authors**: *Pierre Liotet, Davide Maran, Lorenzo Bisi, Marcello Restelli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liotet22a.html](https://proceedings.mlr.press/v162/liotet22a.html)

**Abstract**:

When the agent’s observations or interactions are delayed, classic reinforcement learning tools usually fail. In this paper, we propose a simple yet new and efficient solution to this problem. We assume that, in the undelayed environment, an efficient policy is known or can be easily learnt, but the task may suffer from delays in practice and we thus want to take them into account. We present a novel algorithm, Delayed Imitation with Dataset Aggregation (DIDA), which builds upon imitation learning methods to learn how to act in a delayed environment from undelayed demonstrations. We provide a theoretical analysis of the approach that will guide the practical design of DIDA. These results are also of general interest in the delayed reinforcement learning literature by providing bounds on the performance between delayed and undelayed tasks, under smoothness conditions. We show empirically that DIDA obtains high performances with a remarkable sample efficiency on a variety of tasks, including robotic locomotion, classic control, and trading.

----

## [597] CITRIS: Causal Identifiability from Temporal Intervened Sequences

**Authors**: *Phillip Lippe, Sara Magliacane, Sindy Löwe, Yuki M. Asano, Taco Cohen, Stratis Gavves*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/lippe22a.html](https://proceedings.mlr.press/v162/lippe22a.html)

**Abstract**:

Understanding the latent causal factors of a dynamical system from visual observations is considered a crucial step towards agents reasoning in complex environments. In this paper, we propose CITRIS, a variational autoencoder framework that learns causal representations from temporal sequences of images in which underlying causal factors have possibly been intervened upon. In contrast to the recent literature, CITRIS exploits temporality and observing intervention targets to identify scalar and multidimensional causal factors, such as 3D rotation angles. Furthermore, by introducing a normalizing flow, CITRIS can be easily extended to leverage and disentangle representations obtained by already pretrained autoencoders. Extending previous results on scalar causal factors, we prove identifiability in a more general setting, in which only some components of a causal factor are affected by interventions. In experiments on 3D rendered image sequences, CITRIS outperforms previous methods on recovering the underlying causal variables. Moreover, using pretrained autoencoders, CITRIS can even generalize to unseen instantiations of causal factors, opening future research areas in sim-to-real generalization for causal representation learning.

----

## [598] StreamingQA: A Benchmark for Adaptation to New Knowledge over Time in Question Answering Models

**Authors**: *Adam Liska, Tomás Kociský, Elena Gribovskaya, Tayfun Terzi, Eren Sezener, Devang Agrawal, Cyprien de Masson d'Autume, Tim Scholtes, Manzil Zaheer, Susannah Young, Ellen Gilsenan-McMahon, Sophia Austin, Phil Blunsom, Angeliki Lazaridou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liska22a.html](https://proceedings.mlr.press/v162/liska22a.html)

**Abstract**:

Knowledge and language understanding of models evaluated through question answering (QA) has been usually studied on static snapshots of knowledge, like Wikipedia. However, our world is dynamic, evolves over time, and our models’ knowledge becomes outdated. To study how semi-parametric QA models and their underlying parametric language models (LMs) adapt to evolving knowledge, we construct a new large-scale dataset, StreamingQA, with human written and generated questions asked on a given date, to be answered from 14 years of time-stamped news articles. We evaluate our models quarterly as they read new articles not seen in pre-training. We show that parametric models can be updated without full retraining, while avoiding catastrophic forgetting. For semi-parametric models, adding new articles into the search space allows for rapid adaptation, however, models with an outdated underlying LM under-perform those with a retrained LM. For questions about higher-frequency named entities, parametric updates are particularly beneficial. In our dynamic world, the StreamingQA dataset enables a more realistic evaluation of QA models, and our experiments highlight several promising directions for future research.

----

## [599] Distributionally Robust Q-Learning

**Authors**: *Zijian Liu, Qinxun Bai, Jose H. Blanchet, Perry Dong, Wei Xu, Zhengqing Zhou, Zhengyuan Zhou*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/liu22a.html](https://proceedings.mlr.press/v162/liu22a.html)

**Abstract**:

Reinforcement learning (RL) has demonstrated remarkable achievements in simulated environments. However, carrying this success to real environments requires the important attribute of robustness, which the existing RL algorithms often lack as they assume that the future deployment environment is the same as the training environment (i.e. simulator) in which the policy is learned. This assumption often does not hold due to the discrepancy between the simulator and the real environment and, as a result, and hence renders the learned policy fragile when deployed. In this paper, we propose a novel distributionally robust $Q$-learning algorithm that learns the best policy in the worst distributional perturbation of the environment. Our algorithm first transforms the infinite-dimensional learning problem (since the environment MDP perturbation lies in an infinite-dimensional space) into a finite-dimensional dual problem and subsequently uses a multi-level Monte-Carlo scheme to approximate the dual value using samples from the simulator. Despite the complexity, we show that the resulting distributionally robust $Q$-learning algorithm asymptotically converges to optimal worst-case policy, thus making it robust to future environment changes. Simulation results further demonstrate its strong empirical robustness.

----



[Go to the previous page](ICML-2022-list02.md)

[Go to the next page](ICML-2022-list04.md)

[Go to the catalog section](README.md)