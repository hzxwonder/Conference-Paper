## [0] Bi-Level Offline Policy Optimization with Limited Exploration

**Authors**: *Wenzhuo Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac6de776b8de8c9aed1d356997eb54b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac6de776b8de8c9aed1d356997eb54b8-Abstract-Conference.html)

**Abstract**:

We study offline reinforcement learning (RL) which seeks to learn a good policy based on a fixed, pre-collected dataset. A fundamental challenge behind this task is the distributional shift due to the dataset lacking sufficient exploration, especially under function approximation. To tackle this issue, we propose a bi-level structured policy optimization algorithm that models a hierarchical interaction between the policy (upper-level) and the value function (lower-level). The lower level focuses on constructing a confidence set of value estimates that maintain sufficiently small weighted average Bellman errors, while controlling uncertainty arising from distribution mismatch. Subsequently, at the upper level, the policy aims to maximize a conservative value estimate from the confidence set formed at the lower level. This novel formulation preserves the maximum flexibility of the implicitly induced exploratory data distribution, enabling the power of model extrapolation. In practice, it can be solved through a computationally efficient, penalized adversarial estimation procedure. Our theoretical regret guarantees do not rely on any data-coverage and completeness-type assumptions, only requiring realizability. These guarantees also demonstrate that the learned policy represents the ``best effort'' among all policies, as no other policies can outperform it. We evaluate our model using a blend of synthetic, benchmark, and real-world datasets for offline RL, showing that it performs competitively with state-of-the-art methods.

----

## [0] Generating QM1B with PySCFIPU

**Authors**: *Alexander Mathiasen, Hatem Helal, Kerstin Klaser, Paul Balanca, Josef Dean, Carlo Luschi, Dominique Beaini, Andrew W. Fitzgibbon, Dominic Masters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac7f98dd0b342edaf3be79844a180a6b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac7f98dd0b342edaf3be79844a180a6b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The emergence of foundation models in Computer Vision and Natural Language Processing have resulted in immense progress on downstream tasks.  This progress was enabled by datasets with billions of training examples. Similar benefits are yet to be unlocked for quantum chemistry, where the potential of deep learning is constrained by comparatively small datasets with 100k to 20M training examples. These datasets are limited in size because the labels are computed using the accurate (but computationally demanding) predictions of Density Functional Theory (DFT). Notably, prior DFT datasets were created using CPU supercomputers without leveraging hardware acceleration.  In this paper, we take a first step towards utilising hardware accelerators by introducing the data generator PySCF$_{\text{IPU}}$ using Intelligence Processing Units (IPUs). This allows us to create the dataset QM1B with one billion training examples containing 9-11 heavy atoms. We demonstrate that a simple baseline neural network (SchNet 9M) improves its performance by simply increasing the amount of training data without additional inductive biases. To encourage future researchers to use QM1B responsibly, we highlight several limitations of QM1B and emphasise the low resolution of our DFT options, which also serves as motivation for even larger, more accurate datasets.

----

## [0] Unified Enhancement of Privacy Bounds for Mixture Mechanisms via f-Differential Privacy

**Authors**: *Chendi Wang, Buxin Su, Jiayuan Ye, Reza Shokri, Weijie J. Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acb3e20075b0a2dfa3565f06681578e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acb3e20075b0a2dfa3565f06681578e5-Abstract-Conference.html)

**Abstract**:

Differentially private (DP) machine learning algorithms incur many sources of randomness, such as random initialization, random batch subsampling, and shuffling. However, such randomness is difficult to take into account when proving differential privacy bounds because it induces mixture distributions for the algorithm's output that are difficult to analyze. This paper focuses on improving privacy bounds for shuffling models and one-iteration  differentially private gradient descent (DP-GD) with random initializations using $f$-DP. We derive a closed-form expression of the trade-off function for shuffling models that outperforms the most up-to-date results based on $(\epsilon,\delta)$-DP.Moreover, we investigate the effects of random initialization on the privacy of one-iteration DP-GD. Our numerical computations of the trade-off function indicate that random initialization can enhance the privacy of DP-GD.Our analysis of $f$-DP guarantees for these mixture mechanisms relies on an inequality for trade-off functions introduced in this paper. This inequality implies the joint convexity of $F$-divergences. Finally, we study an $f$-DP analog of the advanced joint convexity of the hockey-stick divergence related to $(\epsilon,\delta)$-DP  and apply it to analyze the privacy of mixture mechanisms.

----

## [0] On the Role of Entanglement and Statistics in Learning

**Authors**: *Srinivasan Arunachalam, Vojtech Havlícek, Louis Schatzki*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acb7ce5aab6e134300a2361dd90a501f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acb7ce5aab6e134300a2361dd90a501f-Abstract-Conference.html)

**Abstract**:

In this work we make progress in understanding the relationship between learning models when given access to entangled measurements, separable measurements and statistical measurements in the quantum statistical query ($\mathsf{QSQ}$) model. To this end, we show the following results.$\textbf{Entanglement versus separable measurements.}$ The goal here is to learn an unknown $f$ from the concept class $\mathcal{C} \subseteq \{f:\{0,1\}^n\rightarrow [k]\}$ given copies of  $\frac{1}{\sqrt{2^n}}\sum_x \ket{x,f(x)}$.  We show that, if $T$ copies suffice to learn $f$ using entangled measurements, then $O(nT^2)$ copies suffice to learn $f$ using just separable measurements. Additionally, we exhibit a concept class $\mathcal{C}$ for which, in order to learn some \emph{property} of $f$, the sample complexity of learning using entangled measurements is exponentially smaller than separable measurements.$\textbf{Entangled versus statistical measurements}$ The goal here is to learn a function $f \in \mathcal{C}$ given access to separable measurements and statistical measurements.   We exhibit a concept class $\mathcal{C}$ based on degree-$2$ functions that gives an exponential separation between $\mathsf{QSQ}$ learning and quantum learning with entangled measurements (even in the presence of noise). This proves the "quantum analogue" of the seminal result of (Blum, 2003) that separates classical $\mathsf{SQ}$ learning from classical $\mathsf{PAC}$ learning with classification~noise.$\textbf{$\mathsf{QSQ}$ lower bounds for learning states.}$ The main technical contribution is to introduce a quantum statistical query dimension ($\mathsf{QSDA}$), which we use to give lower bounds on the $\mathsf{QSQ}$ complexity of learning. Using this, we prove exponential $\mathsf{QSQ}$ lower bounds for testing purity of quantum states, learning CCHL states, coset states of Abelian groups, degree-$2$ functions, planted bi-clique states and learning output states of Clifford circuits of depth polylog($n$).$\textbf{Further applications.}$ Using our $\mathsf{QSQ}$ lower bounds give an $\textit{unconditional}$ separation between weak and strong error mitigation and prove lower bounds for learning distributions in the $\mathsf{QSQ}$ model. Prior works by (Quek et al., 2022), (Hinsche et al., 2022), and (Neitner et al., 23) proved the analogous results $\textit{assuming}$ diagonal measurements and our work removes this assumption.

----

## [0] Molecule Joint Auto-Encoding: Trajectory Pretraining with 2D and 3D Diffusion

**Authors**: *Weitao Du, Jiujiu Chen, Xuecang Zhang, Zhi-Ming Ma, Shengchao Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html)

**Abstract**:

Recently, artificial intelligence for drug discovery has raised increasing interest in both machine learning and chemistry domains. The fundamental building block for drug discovery is molecule geometry and thus, the molecule's geometrical representation is the main bottleneck to better utilize machine learning techniques for drug discovery. In this work, we propose a pretraining method for molecule joint auto-encoding (MoleculeJAE). MoleculeJAE can learn both the 2D bond (topology) and 3D conformation (geometry) information, and a diffusion process model is applied to mimic the augmented trajectories of such two modalities, based on which, MoleculeJAE will learn the inherent chemical structure in a self-supervised manner. Thus, the pretrained geometrical representation in MoleculeJAE is expected to benefit downstream geometry-related tasks. Empirically, MoleculeJAE proves its effectiveness by reaching state-of-the-art performance on 15 out of 20 tasks by comparing it with 12 competitive baselines.

----

## [0] Federated Learning with Manifold Regularization and Normalized Update Reaggregation

**Authors**: *Xuming An, Li Shen, Han Hu, Yong Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acf2b98eeb09b21968c2de6b1c6952e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acf2b98eeb09b21968c2de6b1c6952e9-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is an emerging collaborative machine learning framework where multiple clients train the global model without sharing their own datasets. In FL, the model inconsistency caused by the local data heterogeneity across clients results in the near-orthogonality of client updates, which leads to the global update norm reduction and slows down the convergence.  Most previous works focus on eliminating the difference of parameters (or gradients) between the local and global models, which may fail to reflect the model inconsistency due to the complex structure of the machine learning model and the Euclidean space's limitation in meaningful geometric representations.In this paper, we propose FedMRUR by adopting the manifold model fusion scheme and a new global optimizer to alleviate the negative impacts.Concretely, FedMRUR adopts a hyperbolic graph manifold regularizer enforcing the representations of the data in the local and global models are close to each other in a low-dimensional subspace. Because the machine learning model has the graph structure, the distance in hyperbolic space can reflect the model bias better than the Euclidean distance.In this way, FedMRUR exploits the manifold structures of the representations to significantly reduce the model inconsistency.FedMRUR also aggregates the client updates norms as the global update norm, which can appropriately enlarge each client's contribution to the global update, thereby mitigating the norm reduction introduced by the near-orthogonality of client updates.Furthermore, we theoretically prove that our algorithm can achieve a linear speedup property $\mathcal{O}(\frac{1}{\sqrt{SKT}})$ for non-convex setting under partial client participation, where $S$ is the participated clients number, $K$ is the local interval and $T$ is the total number of communication rounds.Experiments demonstrate that FedMRUR can achieve a new state-of-the-art (SOTA) accuracy with less communication.

----

## [0] Long-Term Fairness with Unknown Dynamics

**Authors**: *Tongxin Yin, Reilly Raab, Mingyan Liu, Yang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acf4a08f67724e9d2de34099f57a9c25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acf4a08f67724e9d2de34099f57a9c25-Abstract-Conference.html)

**Abstract**:

While machine learning can myopically reinforce social inequalities, it may also be used to dynamically seek equitable outcomes. In this paper, we formalize long-term fairness as an online reinforcement learning problem for a policy affecting human populations. This formulation accommodates dynamical control objectives, such as achieving equitable population states, that cannot be incorporated into static formulations of fairness. We demonstrate that algorithmic solutions to the proposed fairness problem can adapt to unknown dynamics and, by sacrificing short-term incentives, drive the policy-population system towards more desirable equilibria. For the proposed setting, we develop an algorithm that adapts recent work in online learning and prove that this algorithm achieves simultaneous probabilistic bounds on cumulative loss and cumulative violations of fairness. In the classification setting subject to group fairness, we compare our proposed algorithm to several baselines, including the repeated retraining of myopic or distributionally robust classifiers, and to a deep reinforcement learning algorithm that lacks fairness guarantees. Our experiments model human populations according to evolutionary game theory and integrate real-world datasets.

----

## [0] YouTubePD: A Multimodal Benchmark for Parkinson's Disease Analysis

**Authors**: *Andy Zhou, Samuel Li, Pranav Sriram, Xiang Li, Jiahua Dong, Ansh Sharma, Yuanyi Zhong, Shirui Luo, Volodymyr V. Kindratenko, George Heintz, Christopher Zallek, Yu-Xiong Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acffd5024f52c3a9ecc8ccb4b75b4e5c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/acffd5024f52c3a9ecc8ccb4b75b4e5c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The healthcare and AI communities have witnessed a growing interest in the development of AI-assisted systems for automated diagnosis of Parkinson's Disease (PD), one of the most prevalent neurodegenerative disorders. However, the progress in this area has been significantly impeded by the absence of a unified, publicly available benchmark, which prevents comprehensive evaluation of existing PD analysis methods and the development of advanced models. This work overcomes these challenges by introducing YouTubePD -- the first publicly available multimodal benchmark designed for PD analysis. We crowd-source existing videos featured with PD from YouTube, exploit multimodal information including in-the-wild videos, audio data, and facial landmarks across 200+ subject videos, and provide dense and diverse annotations from clinical expert. Based on our benchmark, we propose three challenging and complementary tasks encompassing both discriminative and generative tasks, along with a comprehensive set of corresponding baselines. Experimental evaluation showcases the potential of modern deep learning and computer vision techniques, in particular the generalizability of the models developed on YouTubePD to real-world clinical settings, while revealing their limitations. We hope our work paves the way for future research in this direction.

----

## [0] Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training

**Authors**: *Aleksandra Nowak, Bram Grooten, Decebal Constantin Mocanu, Jacek Tabor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad02c6f3824f871395112ae71a28eff7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad02c6f3824f871395112ae71a28eff7-Abstract-Conference.html)

**Abstract**:

Dynamic Sparse Training (DST) is a rapidly evolving area of research that seeks to optimize the sparse initialization of a neural network by adapting its topology during training.  It has been shown that under specific conditions, DST is able to outperform dense models. The key components of this framework are the pruning and growing criteria, which are repeatedly applied during the training process to adjust the networkâ€™s sparse connectivity. While the growing criterion's impact on DST performance is relatively well studied, the influence of the pruning criterion remains overlooked. To address this issue, we design and perform an extensive empirical analysis of various pruning criteria to better understand their impact on the dynamics of DST solutions. Surprisingly, we find that most of the studied methods yield similar results. The differences become more significant in the low-density regime, where the best performance is predominantly given by the simplest technique: magnitude-based pruning.

----

## [0] Tree-Based Diffusion Schrödinger Bridge with Applications to Wasserstein Barycenters

**Authors**: *Maxence Noble, Valentin De Bortoli, Arnaud Doucet, Alain Durmus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad08767706825033b99122332293033d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad08767706825033b99122332293033d-Abstract-Conference.html)

**Abstract**:

Multi-marginal Optimal Transport (mOT), a generalization of OT, aims at minimizing the integral of a cost function with respect to a distribution with some prescribed marginals. In this paper, we consider an entropic version of mOT  with a tree-structured quadratic cost, i.e., a function that can be written as  a sum of pairwise cost functions between the nodes of a tree. To address this  problem, we develop Tree-based Diffusion Schr\"odinger Bridge (TreeDSB), an  extension of the Diffusion Schr\"odinger Bridge (DSB) algorithm. TreeDSB  corresponds to a dynamic and continuous state-space counterpart of the  multimarginal Sinkhorn algorithm. A notable use case of our methodology is to  compute Wasserstein barycenters which can be recast as the solution of a mOT  problem on a star-shaped tree. We demonstrate that our methodology can be applied in high-dimensional settings such as image interpolation and  Bayesian fusion.

----

## [0] Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders

**Authors**: *Jan Dubinski, Stanislaw Pawlak, Franziska Boenisch, Tomasz Trzcinski, Adam Dziedzic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad1efab57a04d93f097e7fbb2d4fc054-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad1efab57a04d93f097e7fbb2d4fc054-Abstract-Conference.html)

**Abstract**:

Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task. B4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.

----

## [0] A General Framework for Equivariant Neural Networks on Reductive Lie Groups

**Authors**: *Ilyes Batatia, Mario Geiger, Jose M. Munoz, Tess E. Smidt, Lior Silberman, Christoph Ortner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad1f2197941348b1c4373fd6c19ee0b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad1f2197941348b1c4373fd6c19ee0b4-Abstract-Conference.html)

**Abstract**:

Reductive Lie Groups, such as the orthogonal groups, the Lorentz group, or the unitary groups, play essential roles across scientific fields as diverse as high energy physics, quantum mechanics, quantum chromodynamics, molecular dynamics, computer vision, and imaging. In this paper, we present a general Equivariant Neural Network architecture capable of respecting the symmetries of the finite-dimensional representations of any reductive Lie Group. Our approach generalizes the successful ACE and MACE architectures for atomistic point clouds to any data equivariant to a reductive Lie group action. We also introduce the lie-nn software library, which provides all the necessary tools to develop and implement such general G-equivariant neural networks. It implements routines for the reduction of generic tensor products of representations into irreducible representations, making it easy to apply our architecture to a wide range of problems and groups. The generality and performance of our approach are demonstrated by applying it to the tasks of top quark decay tagging (Lorentz group) and shape recognition (orthogonal group).

----

## [0] Context-PIPs: Persistent Independent Particles Demands Context Features

**Authors**: *Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yitong Dong, Yijin Li, Hongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad2fa437f7c23e4e9875599c6065d18a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad2fa437f7c23e4e9875599c6065d18a-Abstract-Conference.html)

**Abstract**:

We tackle the problem of Persistent Independent Particles (PIPs), also called Tracking Any Point (TAP), in videos, which specifically aims at estimating persistent long-term trajectories of query points in videos. Previous methods attempted to estimate these trajectories independently to incorporate longer image sequences, therefore, ignoring the potential benefits of incorporating spatial context features. We argue that independent video point tracking also demands spatial context features. To this end, we propose a novel framework Context-PIPs, which effectively improves point trajectory accuracy by aggregating spatial context features in videos. Context-PIPs contains two main modules: 1) a SOurse Feature Enhancement (SOFE) module, and 2) a TArget Feature Aggregation (TAFA) module. Context-PIPs significantly improves PIPs all-sided, reducing 11.4\% Average Trajectory Error of Occluded Points (ATE-Occ) on CroHD and increasing 11.8\% Average Percentage of Correct Keypoint (A-PCK) on TAP-Vid-Kinetics. Demos are available at \url{https://wkbian.github.io/Projects/Context-PIPs/}.

----

## [0] GloptiNets: Scalable Non-Convex Optimization with Certificates

**Authors**: *Gaspard Beugnot, Julien Mairal, Alessandro Rudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad350eaaa93c8c3ab762cdc119d12889-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad350eaaa93c8c3ab762cdc119d12889-Abstract-Conference.html)

**Abstract**:

We present a novel approach to non-convex optimization with certificates, which handles smooth functions on the hypercube or on the torus. Unlike traditional methods that rely on algebraic properties, our algorithm exploits the regularity of the target function intrinsic in the decay of its Fourier spectrum. By defining a tractable family of models, we allow {\em at the same time} to obtain precise certificates and to leverage the advanced and powerful computational techniques developed to optimize neural networks. In this way the scalability of our approach is naturally enhanced by parallel computing with GPUs. Our approach, when applied to the case of polynomials of moderate dimensions but with thousands of coefficients, outperforms the state-of-the-art optimization methods with certificates, as the ones based on Lasserre's hierarchy, addressing problems intractable for the competitors.

----

## [0] Ethical Considerations for Responsible Data Curation

**Authors**: *Jerone Theodore Alexander Andrews, Dora Zhao, William Thong, Apostolos Modas, Orestis Papakyriakopoulos, Alice Xiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad3ebc951f43d1e9ed20187a7b5bc4ee-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad3ebc951f43d1e9ed20187a7b5bc4ee-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human-centric computer vision (HCCV) data curation practices often neglect privacy and bias concerns, leading to dataset retractions and unfair models. HCCV datasets constructed through nonconsensual web scraping lack crucial metadata for comprehensive fairness and robustness evaluations. Current remedies are post hoc, lack persuasive justification for adoption, or fail to provide proper contextualization for appropriate application. Our research focuses on proactive, domain-specific recommendations, covering purpose, privacy and consent, and diversity, for curating HCCV evaluation datasets, addressing privacy and bias concerns. We adopt an ante hoc reflective perspective, drawing from current practices, guidelines, dataset withdrawals, and audits, to inform our considerations and recommendations.

----

## [0] Meta-Adapter: An Online Few-shot Learner for Vision-Language Model

**Authors**: *Cheng Cheng, Lin Song, Ruoyi Xue, Hang Wang, Hongbin Sun, Yixiao Ge, Ying Shan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad48f017e6c3d474caf511208e600459-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad48f017e6c3d474caf511208e600459-Abstract-Conference.html)

**Abstract**:

The contrastive vision-language pre-training, known as CLIP, demonstrates remarkable potential in perceiving open-world visual concepts, enabling effective zero-shot image recognition.  Nevertheless, few-shot learning methods based on CLIP typically require offline fine-tuning of the parameters on few-shot samples, resulting in longer inference time and the risk of overfitting in certain domains.  To tackle these challenges, we propose the Meta-Adapter, a lightweight residual-style adapter, to refine the CLIP features guided by the few-shot samples in an online manner.  With a few training samples, our method can enable effective few-shot learning capabilities and generalize to unseen data or tasks without additional fine-tuning, achieving competitive performance and high efficiency.  Without bells and whistles, our approach outperforms the state-of-the-art online few-shot learning method by an average of 3.6\% on eight image classification datasets with higher inference speed.  Furthermore, our model is simple and flexible, serving as a plug-and-play module directly applicable to downstream tasks.  Without further fine-tuning, Meta-Adapter obtains notable performance improvements in open-vocabulary object detection and segmentation tasks.

----

## [0] Taming Local Effects in Graph-based Spatiotemporal Forecasting

**Authors**: *Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html)

**Abstract**:

Spatiotemporal graph neural networks have shown to be effective in time series forecasting applications, achieving better performance than standard univariate predictors in several settings. These architectures take advantage of a graph structure and relational inductive biases to learn a single (global) inductive model to predict any number of the input time series, each associated with a graph node. Despite the gain achieved in computational and data efficiency w.r.t. fitting a set of local models, relying on a single global model can be a limitation whenever some of the time series are generated by a different spatiotemporal stochastic process. The main objective of this paper is to understand the interplay between globality and locality in graph-based spatiotemporal forecasting, while contextually proposing a methodological framework to rationalize the practice of including trainable node embeddings in such architectures. We ascribe to trainable node embeddings the role of amortizing the learning of specialized components. Moreover, embeddings allow for 1) effectively combining the advantages of shared message-passing layers with node-specific parameters and 2) efficiently transferring the learned model to new node sets. Supported by strong empirical evidence, we provide insights and guidelines for specializing graph-based models to the dynamics of each time series and show how this aspect plays a crucial role in obtaining accurate predictions.

----

## [0] Latent Space Translation via Semantic Alignment

**Authors**: *Valentino Maiorca, Luca Moschella, Antonio Norelli, Marco Fumero, Francesco Locatello, Emanuele Rodolà*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract-Conference.html)

**Abstract**:

While different neural models often exhibit latent spaces that are alike when exposed to semantically related data, this intrinsic similarity is not always immediately discernible. Towards a better understanding of this phenomenon, our work shows how representations learned from these neural modules can be translated between different pre-trained networks via simpler transformations than previously thought. An advantage of this approach is the ability to estimate these transformations using standard, well-understood algebraic procedures that have closed-form solutions. Our method directly estimates a transformation between two given latent spaces, thereby enabling effective stitching of encoders and decoders without additional training. We extensively validate the adaptability of this translation procedure in different experimental settings: across various trainings, domains, architectures (e.g., ResNet, CNN, ViT), and in multiple downstream tasks (classification, reconstruction). Notably, we show how it is possible to zero-shot stitch text encoders and vision decoders, or vice-versa, yielding surprisingly good classification performance in this multimodal setting.

----

## [0] A case for reframing automated medical image classification as segmentation

**Authors**: *Sarah M. Hooper, Mayee F. Chen, Khaled Saab, Kush Bhatia, Curtis P. Langlotz, Christopher Ré*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad6a3bd12095fdca71c306871bdec400-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad6a3bd12095fdca71c306871bdec400-Abstract-Conference.html)

**Abstract**:

Image classification and segmentation are common applications of deep learning to radiology. While many tasks can be framed using either classification or segmentation, classification has historically been cheaper to label and more widely used. However, recent work has drastically reduced the cost of training segmentation networks. In light of this recent work, we reexamine the choice of training classification vs. segmentation models. First, we use an information theoretic approach to analyze why segmentation vs. classification models may achieve different performance on the same dataset and overarching task. We then implement multiple methods for using segmentation models to classify medical images, which we call segmentation-for-classification, and compare these methods against traditional classification on three retrospective datasets. We use our analysis and experiments to summarize the benefits of switching from segmentation to classification, including: improved sample efficiency, enabling improved performance with fewer labeled images (up to an order of magnitude lower), on low-prevalence classes, and on certain rare subgroups (up to 161.1\% improved recall); improved robustness to spurious correlations (up to 44.8\% improved robust AUROC); and improved model interpretability, evaluation, and error analysis.

----

## [0] Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents

**Authors**: *Wonje Choi, Woo Kyung Kim, Seunghyun Kim, Honguk Woo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad72633e034990a97e878fc2fc100afb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad72633e034990a97e878fc2fc100afb-Abstract-Conference.html)

**Abstract**:

For embodied reinforcement learning (RL) agents interacting with the environment, it is desirable to have rapid policy adaptation to unseen visual observations, but achieving zero-shot adaptation capability is considered as a challenging problem in the RL context. To address the problem, we present a novel contrastive prompt ensemble (ConPE) framework which utilizes a pretrained vision-language model and a set of visual prompts, thus enables efficient policy learning and adaptation upon a wide range of environmental and physical changes encountered by embodied agents. Specifically, we devise a guided-attention-based ensemble approach with multiple visual prompts on the vision-language model to construct robust state representations. Each prompt is contrastively learned in terms of an individual domain factors that significantly affects the agent's egocentric perception and observation. For a given task, the attention-based ensemble and policy are jointly learned so that the resulting state representations not only generalize to various domains but are also optimized for learning the task. Through experiments, we show that ConPE outperforms other state-of-the-art algorithms for several embodied agent tasks including navigation in AI2THOR, manipulation in Metaworld, and autonomous driving in CARLA, while also improving the sample efficiency of policy learning and adaptation.

----

## [0] Improvements on Uncertainty Quantification for Node Classification via Distance Based Regularization

**Authors**: *Russell Hart, Linlin Yu, Yifei Lou, Feng Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad84864002a72c344c2227d7eb8842b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad84864002a72c344c2227d7eb8842b1-Abstract-Conference.html)

**Abstract**:

Deep neural networks have achieved significant success in the last decades, but they are not well-calibrated and often produce unreliable predictions. A large number of literature relies on uncertainty quantification to evaluate the reliability of a learning model, which is particularly important for applications of out-of-distribution (OOD) detection and misclassification detection. We are interested in uncertainty quantification for interdependent node-level classification. We start our analysis based on graph posterior networks (GPNs) that optimize the uncertainty cross-entropy (UCE)-based loss function. We describe the theoretical limitations of the widely-used UCE loss. To alleviate the identified drawbacks, we propose a distance-based regularization that encourages clustered OOD nodes to remain clustered in the latent space. We conduct extensive comparison experiments on eight standard datasets and demonstrate that the proposed regularization outperforms the state-of-the-art in both OOD detection and misclassification detection.

----

## [0] Efficient Learning of Linear Graph Neural Networks via Node Subsampling

**Authors**: *Seiyun Shin, Ilan Shomorony, Han Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ada418ae9b6677dcda32d9dca0f7441f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ada418ae9b6677dcda32d9dca0f7441f-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) are a powerful class of machine learning models with applications in recommender systems, drug discovery, social network analysis, and computer vision. One challenge with their implementation is that GNNs often take large-scale graphs as inputs, which imposes significant computational/storage costs in the training and testing phases. In particular, the message passing operations of a GNN require multiplication of the graph adjacency matrix $A \in \mathbb{R}^{n \times n}$ and the data matrix $X \in \mathbb{R}^{n \times d}$, and the $O(n^2 d)$ time complexity can be prohibitive for large $n$. Thus, a natural question is whether it is possible to perform the GNN operations in (quasi-)linear time by avoiding the full computation of $A X$. To study this question, we consider the setting of a regression task on a two-layer Linear Graph Convolutional Network (GCN). We develop an efficient training algorithm based on (1) performing node subsampling, (2) estimating the leverage scores of $A X$ based on the subsampled graph, and (3) performing leverage score sampling on $A X$. We show that our proposed scheme learns the regression model observing only $O(nd\epsilon^{-2}\log n)$ entries of $A$ in time $O(nd^2 \epsilon^{-2}\log n)$, with the guarantee that the learned weights deviate by at most $\epsilon$ under the $\ell_2$ norm from the model learned using the entire adjacency matrix $A$. We present empirical results for regression problems on real-world graphs and show that our algorithm significantly outperforms other baseline sampling strategies that exploit the same number of observations.

----

## [0] DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics

**Authors**: *Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ada8de994b46571bdcd7eeff2d3f9cff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ada8de994b46571bdcd7eeff2d3f9cff-Abstract-Conference.html)

**Abstract**:

Diffusion probabilistic models (DPMs) have exhibited excellent performance for high-fidelity image generation while suffering from inefficient sampling. Recent works accelerate the sampling procedure by proposing fast ODE solvers that leverage the specific ODE form of DPMs. However, they highly rely on specific parameterization during inference (such as noise/data prediction), which might not be the optimal choice. In this work, we propose a novel formulation towards the optimal parameterization during sampling that minimizes the first-order discretization error of the ODE solution. Based on such formulation, we propose \textit{DPM-Solver-v3}, a new fast ODE solver for DPMs by introducing several coefficients efficiently computed on the pretrained model, which we call \textit{empirical model statistics}. We further incorporate multistep methods and a predictor-corrector framework, and propose some techniques for improving sample quality at small numbers of function evaluations (NFE) or large guidance scales. Experiments show that DPM-Solver-v3 achieves consistently better or comparable performance in both unconditional and conditional sampling with both pixel-space and latent-space DPMs, especially in 5$\sim$10 NFEs. We achieve FIDs of 12.21 (5 NFE), 2.51 (10 NFE) on unconditional CIFAR10, and MSE of 0.55 (5 NFE, 7.5 guidance scale) on Stable Diffusion, bringing a speed-up of 15\%$\sim$30\% compared to previous state-of-the-art training-free methods. Code is available at \url{https://github.com/thu-ml/DPM-Solver-v3}.

----

## [0] Active Bipartite Ranking

**Authors**: *James Cheshire, Vincent Laurent, Stéphan Clémençon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/adb77ecc8ba1c2d3135c86a46b8f2496-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/adb77ecc8ba1c2d3135c86a46b8f2496-Abstract-Conference.html)

**Abstract**:

In this paper, we develop an active learning framework for the bipartite ranking problem.Motivated by numerous applications, ranging from supervised anomaly detection to credit-scoring through the design of medical diagnosis support systems, and usually formulated as the problem of optimizing (a scalar summary of) the ROC curve, bipartite ranking has been the subject of much attention in the passive context. Various dedicated algorithms have been recently proposed and studied by the machine-learning community. In contrast, active bipartite ranking rule is poorly documented in the literature. Due to its global nature, a strategy for labeling sequentially data points that are difficult to rank w.r.t. to the others is required. This learning task is much more complex than binary classification, for which many active algorithms have been designed. It is the goal of this article to provide a rigorous formulation of such a selective sampling approach. We propose a dedicated algorithm, referred to as active-rank, which aims to minimise the distance between the ROC curve of the ranking function built and the optimal one, w.r.t. the sup norm. We show that, for a fixed confidence level $\epsilon$ and probability $\delta$,  active-rank is PAC$(\epsilon,\delta)$. In addition, we provide a problem dependent upper bound on the expected sampling time of  active-rank and also demonstrate a problem dependent lower bound on the expected sampling time of any PAC$(\epsilon,\delta)$ algorithm. Beyond the theoretical analysis carried out, numerical results are presented, providing strong empirical evidence of the performance of the algorithm proposed, which compares favorably with more naive approaches.

----

## [0] Are Emergent Abilities of Large Language Models a Mirage?

**Authors**: *Rylan Schaeffer, Brando Miranda, Sanmi Koyejo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html)

**Abstract**:

Recent work claims that large language models display \textit{emergent abilities}, abilities not present in smaller-scale models that are present in larger-scale models.What makes emergent abilities intriguing is two-fold: their \textit{sharpness}, transitioning seemingly instantaneously from not present to present, and their \textit{unpredictability}, appearing at seemingly unforeseeable model scales.Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due the researcherâ€™s choice of metric rather than due to fundamental changes in model behavior with scale. Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance.We present our alternative explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metric choice using the InstructGPT/GPT-3 family on tasks with claimed emergent abilities, (2) make, test and confirm two predictions about metric choices in a meta-analysis of emergent abilities on BIG-Bench; and (3) show how to choose metrics to produce never-before-seen seemingly emergent abilities in multiple vision tasks across diverse deep networks.Via all three analyses, we provide evidence that alleged emergent abilities evaporate with different metrics or with better statistics, and may not be a fundamental property of scaling AI models.

----

## [0] Reward-agnostic Fine-tuning: Provable Statistical Benefits of Hybrid Reinforcement Learning

**Authors**: *Gen Li, Wenhao Zhan, Jason D. Lee, Yuejie Chi, Yuxin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ade04fd4f26263f86b47ffb535c4cafb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ade04fd4f26263f86b47ffb535c4cafb-Abstract-Conference.html)

**Abstract**:

This paper studies tabular reinforcement learning (RL) in the hybrid setting, which assumes access to both an offline dataset and online interactions with the unknown environment. A central question boils down to how to efficiently utilize online data to strengthen and complement the offline dataset and enable effective policy fine-tuning. Leveraging recent advances in reward-agnostic exploration and offline RL, we design a three-stage hybrid RL algorithm that beats the best of both worlds --- pure offline RL and pure online RL --- in terms of sample complexities. The proposed algorithm does not require any reward information during data collection. Our theory is developed based on a new notion called single-policy partial concentrability, which captures the trade-off between distribution mismatch and miscoverage and guides the interplay between offline and online data.

----

## [0] The Exact Sample Complexity Gain from Invariances for Kernel Regression

**Authors**: *Behrooz Tahmasebi, Stefanie Jegelka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/adf5a38a2e2e7606fbfc3eff72998afa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/adf5a38a2e2e7606fbfc3eff72998afa-Abstract-Conference.html)

**Abstract**:

In practice, encoding invariances into models improves sample complexity. In this work, we study this phenomenon from a theoretical perspective. In particular, we provide minimax optimal rates for kernel ridge regression on compact manifolds, with a target function that is invariant to a group action on the manifold. Our results hold for any smooth compact Lie group action, even groups of positive dimension. For a finite group, the gain effectively multiplies the number of samples by the group size. For groups of positive dimension, the gain is observed by a reduction in the manifold's dimension, in addition to a factor proportional to the volume of the quotient space. Our proof takes the viewpoint of differential geometry, in contrast to the more common strategy of using invariant polynomials. This new geometric viewpoint on learning with invariances may be of independent interest.

----

## [0] FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models

**Authors**: *Hao Zhang, Tianyuan Dai, Yanbo Xu, Yu-Wing Tai, Chi-Keung Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html)

**Abstract**:

The ability to create high-quality 3D faces from a single image has become increasingly important with wide applications in video conferencing, AR/VR, and advanced video editing in movie industries. In this paper, we propose Face Diffusion NeRF (FaceDNeRF), a new generative method to reconstruct high-quality Face NeRFs from single images, complete with semantic editing and relighting capabilities. FaceDNeRF utilizes high-resolution 3D GAN inversion and expertly trained 2D latent-diffusion model, allowing users to manipulate and construct Face NeRFs in zero-shot learning without the need for explicit 3D data. With carefully designed illumination and identity preserving loss, as well as multi-modal pre-training, FaceDNeRF offers users unparalleled control over the editing process enabling them to create and edit face NeRFs using just single-view images, text prompts, and explicit target lighting. The advanced features of FaceDNeRF have been designed to produce more impressive results than existing 2D editing approaches that rely on 2D segmentation maps for editable attributes. Experiments show that our FaceDNeRF achieves exceptionally realistic results and unprecedented flexibility in editing compared with state-of-the-art 3D face reconstruction and editing methods. Our code will be available at https://github.com/BillyXYB/FaceDNeRF.

----

## [0] Chanakya: Learning Runtime Decisions for Adaptive Real-Time Perception

**Authors**: *Anurag Ghosh, Vaibhav Balloli, Akshay Nambi, Aditya Singh, Tanuja Ganu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae2d574d2c309f3a45880e4460efd176-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae2d574d2c309f3a45880e4460efd176-Abstract-Conference.html)

**Abstract**:

Real-time perception requires planned resource utilization. Computational planning in real-time perception is governed by two considerations -- accuracy and latency. There exist run-time decisions (e.g. choice of input resolution) that induce tradeoffs affecting performance on a given hardware, arising from intrinsic (content, e.g. scene clutter) and extrinsic (system, e.g. resource contention) characteristics. Earlier runtime execution frameworks employed rule-based decision algorithms and operated with a fixed algorithm latency budget to balance these concerns, which is sub-optimal and inflexible. We propose Chanakya, a learned approximate execution framework that naturally derives from the streaming perception paradigm, to automatically learn decisions induced by these tradeoffs instead. Chanakya is trained via novel rewards balancing accuracy and latency implicitly, without approximating either objectives. Chanakya simultaneously considers intrinsic and extrinsic context, and predicts decisions in a flexible manner. Chanakya, designed with low overhead in mind, outperforms state-of-the-art static and dynamic execution policies on public datasets on both server GPUs and edge devices.

----

## [0] RoboCLIP: One Demonstration is Enough to Learn Robot Policies

**Authors**: *Sumedh Sontakke, Jesse Zhang, Sébastien M. R. Arnold, Karl Pertsch, Erdem Biyik, Dorsa Sadigh, Chelsea Finn, Laurent Itti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html)

**Abstract**:

Reward specification is a notoriously difficult problem in reinforcement learning, requiring extensive expert supervision to design robust reward functions. Imitation learning (IL) methods attempt to circumvent these problems by utilizing expert demonstrations instead of using an extrinsic reward function but typically require a large number of in-domain expert demonstrations. Inspired by advances in the field of Video-and-Language Models (VLMs), we present RoboCLIP, an online imitation learning method that uses a single demonstration (overcoming the large data requirement) in the form of a video demonstration or a textual description of the task to generate rewards without manual reward function design. Additionally, RoboCLIP can also utilize out-of-domain demonstrations, like videos of humans solving the task for reward generation, circumventing the need to have the same demonstration and deployment domains. RoboCLIP utilizes pretrained VLMs without any finetuning for reward generation. Reinforcement learning agents trained with RoboCLIP rewards demonstrate 2-3 times higher zero-shot performance than competing imitation learning methods on downstream robot manipulation tasks, doing so using only one video/text demonstration. Visit our website at https://sites.google.com/view/roboclip/home for experiment videos.

----

## [0] Contrast Everything: A Hierarchical Contrastive Framework for Medical Time-Series

**Authors**: *Yihe Wang, Yu Han, Haishuai Wang, Xiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae7d9c77b5ff9e3b7833a68523b880f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae7d9c77b5ff9e3b7833a68523b880f2-Abstract-Conference.html)

**Abstract**:

Contrastive representation learning is crucial in medical time series analysis as it alleviates dependency on labor-intensive, domain-specific, and scarce expert annotations. However, existing contrastive learning methods primarily focus on one single data level, which fails to fully exploit the intricate nature of medical time series. To address this issue, we present COMET, an innovative hierarchical framework that leverages data consistencies at all inherent levels in medical time series. Our meticulously designed model systematically captures data consistency from four potential levels: observation, sample, trial, and patient levels. By developing contrastive loss at multiple levels, we can learn effective representations that preserve comprehensive data consistency, maximizing information utilization in a self-supervised manner. We conduct experiments in the challenging patient-independent setting. We compare COMET against six baselines using three diverse datasets, which include ECG signals for myocardial infarction and EEG signals for Alzheimer’s and Parkinson’s diseases. The results demonstrate that COMET consistently outperforms all baselines, particularly in setup with 10% and 1% labeled data fractions across all datasets. These results underscore the significant impact of our framework in advancing contrastive representation learning techniques for medical time series. The source code is available at https://github.com/DL4mHealth/COMET.

----

## [0] Importance-aware Co-teaching for Offline Model-based Optimization

**Authors**: *Ye Yuan, Can Chen, Zixuan Liu, Willie Neiswanger, Xue (Steve) Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html)

**Abstract**:

Offline model-based optimization aims to find a design that maximizes a property of interest using only an offline dataset, with applications in robot, protein, and molecule design, among others. A prevalent approach is gradient ascent, where a proxy model is trained on the offline dataset and then used to optimize the design. This method suffers from an out-of-distribution issue, where the proxy is not accurate for unseen designs. To mitigate this issue, we explore using a pseudo-labeler to generate valuable data for fine-tuning the proxy. Specifically, we propose $\textit{\textbf{I}mportance-aware \textbf{C}o-\textbf{T}eaching for Offline Model-based Optimization}~(\textbf{ICT})$. This method maintains three symmetric proxies with their mean ensemble as the final proxy, and comprises two steps. The first step is $\textit{pseudo-label-driven co-teaching}$. In this step, one proxy is iteratively selected as the pseudo-labeler for designs near the current optimization point, generating pseudo-labeled data.  Subsequently, a co-teaching process identifies small-loss samples as valuable data and exchanges them between the other two proxies for fine-tuning, promoting knowledge transfer.  This procedure is repeated three times, with a different proxy chosen as the pseudo-labeler each time, ultimately enhancing the ensemble performance.To further improve accuracy of pseudo-labels, we perform a secondary step of $\textit{meta-learning-based sample reweighting}$,which assigns importance weights to samples in the pseudo-labeled dataset and updates them via meta-learning. ICT achieves state-of-the-art results across multiple design-bench tasks, achieving the best mean rank $3.1$ and median rank $2$ among $15$ methods.Our source code can be accessed here.

----

## [0] Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias

**Authors**: *Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander J. Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae9500c4f5607caf2eff033c67daa9d7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae9500c4f5607caf2eff033c67daa9d7-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large language models (LLMs) have been recently leveraged as training data generators for various natural language processing (NLP) tasks. While previous research has explored different approaches to training models using generated data, they generally rely on simple class-conditional prompts, which may limit the diversity of the generated data and inherit systematic biases of LLM. Thus, we investigate training data generation with diversely attributed prompts (e.g., specifying attributes like length and style), which have the potential to yield diverse and attributed generated data. Our investigation focuses on datasets with high cardinality and diverse domains, wherein we demonstrate that attributed prompts outperform simple class-conditional prompts in terms of the resulting model's performance. Additionally, we present a comprehensive empirical study on data generation encompassing vital aspects like bias, diversity, and efficiency, and highlight three key observations: firstly, synthetic datasets generated by simple prompts exhibit significant biases, such as regional bias; secondly, attribute diversity plays a pivotal role in enhancing model performance; lastly, attributed prompts achieve the performance of simple class-conditional prompts while utilizing only 5\% of the querying cost of ChatGPT associated with the latter. The data and code are available on {\url{https://github.com/yueyu1030/AttrPrompt}}.

----

## [0] GraphPatcher: Mitigating Degree Bias for Graph Neural Networks via Test-time Augmentation

**Authors**: *Mingxuan Ju, Tong Zhao, Wenhao Yu, Neil Shah, Yanfang Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae9bbdcea94d808882f3535e8ca00542-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae9bbdcea94d808882f3535e8ca00542-Abstract-Conference.html)

**Abstract**:

Recent studies have shown that graph neural networks (GNNs) exhibit strong biases towards the node degree: they usually perform satisfactorily on high-degree nodes with rich neighbor information but struggle with low-degree nodes. Existing works tackle this problem by deriving either designated GNN architectures or training strategies specifically for low-degree nodes. Though effective, these approaches unintentionally create an artificial out-of-distribution scenario, where models mainly or even only observe low-degree nodes during the training, leading to a downgraded performance for high-degree nodes that GNNs originally perform well at. In light of this, we propose a test-time augmentation framework, namely GraphPatcher, to enhance test-time generalization of any GNNs on low-degree nodes. Specifically, GraphPatcher iteratively generates virtual nodes to patch artificially created low-degree nodes via corruptions, aiming at progressively reconstructing target GNN's predictions over a sequence of increasingly corrupted nodes. Through this scheme, GraphPatcher not only learns how to enhance low-degree nodes (when the neighborhoods are heavily corrupted) but also preserves the original superior performance of GNNs on high-degree nodes (when lightly corrupted). Additionally, GraphPatcher is model-agnostic and can also mitigate the degree bias for either self-supervised or supervised GNNs. Comprehensive experiments are conducted over seven benchmark datasets and GraphPatcher consistently enhances common GNNs' overall performance by up to 3.6% and low-degree performance by up to 6.5%, significantly outperforming state-of-the-art baselines. The source code is publicly available at https://github.com/jumxglhf/GraphPatcher.

----

## [0] Optimal privacy guarantees for a relaxed threat model: Addressing sub-optimal adversaries in differentially private machine learning

**Authors**: *Georgios Kaissis, Alexander Ziller, Stefan Kolek, Anneliese Riess, Daniel Rueckert*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aea831d6c7af37fd4230937225be3414-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aea831d6c7af37fd4230937225be3414-Abstract-Conference.html)

**Abstract**:

Differentially private mechanisms restrict the membership inference capabilities of powerful (optimal) adversaries against machine learning models. Such adversaries are rarely encountered in practice. In this work, we examine a more realistic threat model relaxation, where (sub-optimal) adversaries lack access to the exact model training database, but may possess related or partial data. We then formally characterise and experimentally validate adversarial membership inference capabilities in this setting in terms of hypothesis testing errors. Our work helps users to interpret the privacy properties of sensitive data processing systems under realistic threat model relaxations and choose appropriate noise levels for their use-case.

----

## [0] Stochastic Approximation Algorithms for Systems of Interacting Particles

**Authors**: *Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aebbbfa9680eafefd43a0edc85c101f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aebbbfa9680eafefd43a0edc85c101f9-Abstract-Conference.html)

**Abstract**:

Interacting particle systems have proven highly successful in various machinelearning tasks, including approximate Bayesian inference and neural network optimization. However, the analysis of thesesystems often relies on the simplifying assumption of the \emph{mean-field} limit, where particlenumbers approach infinity and infinitesimal step sizes are used. In practice, discrete time steps,finite particle numbers, and complex integration schemes are employed, creating a theoretical gapbetween continuous-time and discrete-time processes. In this paper, we present a novel frameworkthat establishes a precise connection between these discrete-time schemes and their correspondingmean-field limits in terms of convergence properties and asymptotic behavior. By adopting a dynamical system perspective, our framework seamlessly integrates various numerical schemes that are typically analyzed independently. For example, our framework provides a unified treatment of optimizing an infinite-width two-layer neural network and sampling via Stein Variational Gradient descent, which were previously studied in isolation.

----

## [0] Provable Guarantees for Neural Networks via Gradient Feature Learning

**Authors**: *Zhenmei Shi, Junyi Wei, Yingyu Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aebec8058f23a445353c83ede0e1ec48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aebec8058f23a445353c83ede0e1ec48-Abstract-Conference.html)

**Abstract**:

Neural networks have achieved remarkable empirical performance, while the current theoretical analysis is not adequate for understanding their success, e.g., the Neural Tangent Kernel approach fails to capture their key feature learning ability, while recent analyses on feature learning are typically problem-specific. This work proposes a unified analysis framework for two-layer networks trained by gradient descent. The framework is centered around the principle of feature learning from gradients, and its effectiveness is demonstrated by applications in several prototypical problems, such as mixtures of Gaussians and parity functions.The framework also sheds light on interesting network learning phenomena such as feature learning beyond kernels and the lottery ticket hypothesis.

----

## [0] Binary Radiance Fields

**Authors**: *Seungjoo Shin, Jaesik Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aebf6284fe85a8f44b4785d41bc8249a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aebf6284fe85a8f44b4785d41bc8249a-Abstract-Conference.html)

**Abstract**:

In this paper, we propose \textit{binary radiance fields} (BiRF), a storage-efficient radiance field representation employing binary feature encoding in a format of either $+1$ or $-1$. This binarization strategy lets us represent the feature grid with highly compact feature encoding and a dramatic reduction in storage size. Furthermore, our 2D-3D hybrid feature grid design enhances the compactness of feature encoding as the 3D grid includes main components while 2D grids capture details. In our experiments, binary radiance field representation successfully outperforms the reconstruction performance of state-of-the-art (SOTA) storage-efficient radiance field models with lower storage allocation. In particular, our model achieves impressive results in static scene reconstruction, with a PSNR of 32.03 dB for Synthetic-NeRF scenes, 34.48 dB for Synthetic-NSVF scenes, 28.20 dB for Tanks and Temples scenes while only utilizing 0.5 MB of storage space, respectively. We hope the proposed binary radiance field representation will make radiance fields more accessible without a storage bottleneck.

----

## [0] A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning

**Authors**: *Alicia Curth, Alan Jeffares, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aec5e2847c5ae90f939ab786774856cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aec5e2847c5ae90f939ab786774856cc-Abstract-Conference.html)

**Abstract**:

Conventional statistical wisdom established a well-understood relationship between model complexity and prediction error, typically presented as a _U-shaped curve_ reflecting a transition between under- and overfitting regimes. However, motivated by the success of overparametrized neural networks, recent influential work has suggested this theory to be generally incomplete, introducing an additional regime that exhibits a second descent in test error as the parameter count $p$ grows past sample size $n$  -- a  phenomenon dubbed  _double descent_. While most attention has naturally been given to the deep-learning setting, double descent was shown to emerge more generally across non-neural models: known cases include _linear regression, trees, and boosting_. In this work, we take a closer look at the evidence surrounding these more classical statistical machine learning methods and challenge the claim that observed cases of  double descent truly extend the limits of a traditional U-shaped complexity-generalization curve therein. We show that once careful consideration is given to _what is being plotted_ on the x-axes of their double descent plots, it becomes apparent that there are implicitly multiple, distinct complexity axes along which the parameter count grows. We demonstrate that the second descent appears exactly (and _only_) when and where the transition between these underlying axes occurs, and that its location is thus _not_ inherently tied to the interpolation threshold $p=n$. We then gain further insight by adopting a classical nonparametric statistics perspective. We interpret the investigated methods as _smoothers_ and propose a generalized measure for the _effective_ number of parameters they use _on unseen examples_, using which we find that their apparent double descent curves do indeed fold back into more traditional convex shapes -- providing a resolution to the ostensible tension between double descent and traditional statistical intuition.

----

## [0] FABind: Fast and Accurate Protein-Ligand Binding

**Authors**: *Qizhi Pei, Kaiyuan Gao, Lijun Wu, Jinhua Zhu, Yingce Xia, Shufang Xie, Tao Qin, Kun He, Tie-Yan Liu, Rui Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aee1de5f335558b546b7e58c380be087-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aee1de5f335558b546b7e58c380be087-Abstract-Conference.html)

**Abstract**:

Modeling the interaction between proteins and ligands and accurately predicting their binding structures is a critical yet challenging task in drug discovery. Recent advancements in deep learning have shown promise in addressing this challenge, with sampling-based and regression-based methods emerging as two prominent approaches. However, these methods have notable limitations. Sampling-based methods often suffer from low efficiency due to the need for generating multiple candidate structures for selection. On the other hand, regression-based methods offer fast predictions but may experience decreased accuracy. Additionally, the variation in protein sizes often requires external modules for selecting suitable binding pockets, further impacting efficiency. In this work, we propose FABind, an end-to-end model that combines pocket prediction and docking to achieve accurate and fast protein-ligand binding.  FABind incorporates a unique ligand-informed pocket prediction module, which is also leveraged for docking pose estimation. The model further enhances the docking process by incrementally integrating the predicted pocket to optimize protein-ligand binding, reducing discrepancies between training and inference. Through extensive experiments on benchmark datasets, our proposed FABind demonstrates strong advantages in terms of effectiveness and efficiency compared to existing methods. Our code is available at https://github.com/QizhiPei/FABind.

----

## [0] Geometric Transformer with Interatomic Positional Encoding

**Authors**: *Yusong Wang, Shaoning Li, Tong Wang, Bin Shao, Nanning Zheng, Tie-Yan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aee2f03ecb2b2c1ea55a43946b651cfd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aee2f03ecb2b2c1ea55a43946b651cfd-Abstract-Conference.html)

**Abstract**:

The widespread adoption of Transformer architectures in various data modalities has opened new avenues for the applications in molecular modeling. Nevertheless, it remains elusive that whether the Transformer-based architecture can do molecular modeling as good as equivariant GNNs. In this paper, by designing Interatomic Positional Encoding (IPE) thatparameterizes atomic environments as Transformer's positional encodings,we propose Geoformer, a novel geometric Transformer to effectively model molecular structures for various molecular property prediction. We evaluate Geoformer on several benchmarks, including the QM9 dataset and the recently proposed Molecule3D dataset. Compared with both Transformers and equivariant GNN models, Geoformer outperforms the state-of-the-art (SoTA) algorithms on QM9, and achieves the best performance on Molecule3D for both random and scaffold splits.By introducing IPE, Geoformer paves the way for molecular geometric modeling based on Transformer architecture.Codes are available at https://github.com/microsoft/AI2BMD/tree/Geoformer.

----

## [0] A Diffusion-Model of Joint Interactive Navigation

**Authors**: *Matthew Niedoba, Jonathan Wilder Lavington, Yunpeng Liu, Vasileios Lioutas, Justice Sefas, Xiaoxuan Liang, Dylan Green, Setareh Dabiri, Berend Zwartsenberg, Adam Scibior, Frank Wood*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aeeddfbab4e99763ebac9221732c80dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aeeddfbab4e99763ebac9221732c80dd-Abstract-Conference.html)

**Abstract**:

Simulation of autonomous vehicle systems requires that simulated traffic participants exhibit diverse and realistic behaviors. The use of prerecorded real-world traffic scenarios in simulation ensures realism but the rarity of safety critical events makes large scale collection of driving scenarios expensive. In this paper, we present DJINN -- a diffusion based method of generating traffic scenarios. Our approach jointly diffuses the trajectories of all agents, conditioned on a flexible set of state observations from the past, present, or future. On popular trajectory forecasting datasets, we report state of the art performance on joint trajectory metrics. In addition, we demonstrate how DJINN flexibly enables direct test-time sampling from a variety of valuable conditional distributions including goal-based sampling, behavior-class sampling, and scenario editing.

----

## [0] Diversifying Spatial-Temporal Perception for Video Domain Generalization

**Authors**: *Kun-Yu Lin, Jia-Run Du, Yipeng Gao, Jiaming Zhou, Wei-Shi Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html)

**Abstract**:

Video domain generalization aims to learn generalizable video classification models for unseen target domains by training in a source domain.A critical challenge of video domain generalization is to defend against the heavy reliance on domain-specific cues extracted from the source domain when recognizing target videos. To this end, we propose to perceive diverse spatial-temporal cues in videos, aiming to discover potential domain-invariant cues in addition to domain-specific cues. We contribute a novel model named Spatial-Temporal Diversification Network (STDN), which improves the diversity from both space and time dimensions of video data. First, our STDN proposes to discover various types of spatial cues within individual frames by spatial grouping. Then, our STDN proposes to explicitly model spatial-temporal dependencies between video contents at multiple space-time scales by spatial-temporal relation modeling. Extensive experiments on three benchmarks of different types demonstrate the effectiveness and versatility of our approach.

----

## [0] Conformal Prediction for Time Series with Modern Hopfield Networks

**Authors**: *Andreas Auer, Martin Gauch, Daniel Klotz, Sepp Hochreiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html)

**Abstract**:

To quantify uncertainty, conformal prediction methods are gaining continuously more interest and have already been successfully applied to various domains. However, they are difficult to apply to time series as the autocorrelative structure of time series violates basic assumptions required by conformal prediction. We propose HopCPT, a novel conformal prediction approach for time series that not only copes with temporal structures but leverages them. We show that our approach is theoretically well justified for time series where temporal dependencies are present. In experiments, we demonstrate that our new approach outperforms state-of-the-art conformal prediction methods on multiple real-world time series datasets from four different domains.

----

## [0] Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks

**Authors**: *Haoyi Duan, Yan Xia, Mingze Zhou, Li Tang, Jieming Zhu, Zhou Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af01716e08073368a7c8a62be46dba17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af01716e08073368a7c8a62be46dba17-Abstract-Conference.html)

**Abstract**:

In recent years, the deployment of large-scale pre-trained models in audio-visual downstream tasks has yielded remarkable outcomes. However, these models, primarily trained on single-modality unconstrained datasets, still encounter challenges in feature extraction for multi-modal tasks, leading to suboptimal performance. This limitation arises due to the introduction of irrelevant modality-specific information during encoding, which adversely affects the performance of downstream tasks. To address this challenge, this paper proposes a novel Dual-Guided Spatial-Channel-Temporal (DG-SCT) attention mechanism. This mechanism leverages audio and visual modalities as soft prompts to dynamically adjust the parameters of pre-trained models based on the current multi-modal input features. Specifically, the DG-SCT module incorporates trainable cross-modal interaction layers into pre-trained audio-visual encoders, allowing adaptive extraction of crucial information from the current modality across spatial, channel, and temporal dimensions, while preserving the frozen parameters of large-scale pre-trained models. Experimental evaluations demonstrate that our proposed model achieves state-of-the-art results across multiple downstream tasks, including AVE, AVVP, AVS, and AVQA. Furthermore, our model exhibits promising performance in challenging few-shot and zero-shot scenarios. The source code and pre-trained models are available at https://github.com/haoyi-duan/DG-SCT.

----

## [0] Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks

**Authors**: *Alexander Schlögl, Nora Hofer, Rainer Böhme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html)

**Abstract**:

Hardware-specific optimizations in machine learning (ML) frameworks can cause numerical deviations of inference results. Quite surprisingly, despite using a fixed trained model and fixed input data, inference results are not consistent across platforms, and sometimes not even deterministic on the same platform. We study the causes of these numerical deviations for convolutional neural networks (CNN) on realistic end-to-end inference pipelines and in isolated experiments. Results from 75 distinct platforms suggest that the main causes of deviations on CPUs are differences in SIMD use, and the selection of convolution algorithms at runtime on GPUs. We link the causes and propagation effects to properties of the ML model and evaluate potential mitigations. We make our research code publicly available.

----

## [0] Learning via Wasserstein-Based High Probability Generalisation Bounds

**Authors**: *Paul Viallard, Maxime Haddouche, Umut Simsekli, Benjamin Guedj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html)

**Abstract**:

Minimising upper bounds on the population risk or the generalisation gap has been widely used in structural risk minimisation (SRM) -- this is in particular at the core of PAC-Bayesian learning. Despite its successes and unfailing surge of interest in recent years, a limitation of the PAC-Bayesian framework is that most bounds involve a Kullback-Leibler (KL) divergence term (or its variations), which might exhibit erratic behavior and fail to capture the underlying geometric structure of the learning problem -- hence restricting its use in practical applications.As a remedy, recent studies have attempted to replace the KL divergence in the PAC-Bayesian bounds with the Wasserstein distance. Even though these bounds alleviated the aforementioned issues to a certain extent, they either hold in expectation, are for bounded losses, or are nontrivial to minimize in an SRM framework.  In this work, we contribute to this line of research and prove novel Wasserstein distance-based PAC-Bayesian generalisation bounds for both batch learning with independent and identically distributed (i.i.d.) data, and online learning with potentially non-i.i.d. data. Contrary to previous art, our bounds are stronger in the sense that (i) they hold with high probability, (ii) they apply to unbounded (potentially heavy-tailed) losses, and (iii) they lead to optimizable training objectives that can be used in SRM. As a result we derive novel Wasserstein-based PAC-Bayesian learning algorithms and we illustrate their empirical advantage on a variety of experiments.

----

## [0] Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity

**Authors**: *Metod Jazbec, James Urquhart Allingham, Dan Zhang, Eric T. Nalisnick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af2d9fb5bcee19ef2dfa70d843520c97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af2d9fb5bcee19ef2dfa70d843520c97-Abstract-Conference.html)

**Abstract**:

Modern predictive models are often deployed to environments in which computational budgets are dynamic. Anytime algorithms  are well-suited to such environments as, at any point during computation, they can output a prediction whose quality is a function of computation time. Early-exit neural networks have garnered attention in the context of anytime computation due to their capability to provide intermediate predictions at various stages throughout the network. However, we demonstrate that current early-exit networks are not directly applicable to anytime settings, as the quality of predictions for individual data points is not guaranteed to improve with longer computation. To address this shortcoming, we propose an elegant post-hoc modification, based on the Product-of-Experts, that encourages an early-exit network to become gradually confident. This gives our deep models the property of conditional monotonicity in the prediction quality---an essential building block towards truly anytime predictive modeling using early-exit architectures. Our empirical results on standard image-classification tasks demonstrate that such behaviors can be achieved while preserving competitive accuracy on average.

----

## [0] A Scalable Neural Network for DSIC Affine Maximizer Auction Design

**Authors**: *Zhijian Duan, Haoran Sun, Yurong Chen, Xiaotie Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html)

**Abstract**:

Automated auction design aims to find empirically high-revenue mechanisms through machine learning. Existing works on multi item auction scenarios can be roughly divided into RegretNet-like and affine maximizer auctions (AMAs) approaches. However, the former cannot strictly ensure dominant strategy incentive compatibility (DSIC), while the latter faces scalability issue due to the large number of allocation candidates. To address these limitations, we propose AMenuNet, a scalable neural network that constructs the AMA parameters (even including the allocation menu) from bidder and item representations. AMenuNet is always DSIC and individually rational (IR) due to the properties of AMAs, and it enhances scalability by generating candidate allocations through a neural network. Additionally, AMenuNet is permutation equivariant, and its number of parameters is independent of auction scale. We conduct extensive experiments to demonstrate that AMenuNet outperforms strong baselines in both contextual and non-contextual multi-item auctions, scales well to larger auctions, generalizes well to different settings, and identifies useful deterministic allocations. Overall, our proposed approach offers an effective solution to automated DSIC auction design, with improved scalability and strong revenue performance in various settings.

----

## [0] Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias

**Authors**: *Zhongwei Wan, Che Liu, Mi Zhang, Jie Fu, Benyou Wang, Sibo Cheng, Lei Ma, César Quilodrán Casas, Rossella Arcucci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af38fb8e90d586f209235c94119ba193-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af38fb8e90d586f209235c94119ba193-Abstract-Conference.html)

**Abstract**:

The scarcity of data presents a critical obstacle to the efficacy of medical vision-language pre-training (VLP). A potential solution lies in the combination of datasets from various language communities.Nevertheless, the main challenge stems from the complexity of integrating diverse syntax and semantics, language-specific medical terminology, and culture-specific implicit knowledge. Therefore, one crucial aspect to consider is the presence of community bias caused by different languages.This paper presents a novel framework named Unifying Cross-Lingual Medical Vision-Language Pre-Training (\textbf{Med-UniC}), designed to integrate multi-modal medical data from the two most prevalent languages, English and Spanish. Specifically, we propose \textbf{C}ross-lingual \textbf{T}ext Alignment \textbf{R}egularization (\textbf{CTR}) to explicitly unify cross-lingual semantic representations of medical reports originating from diverse language communities. \textbf{CTR} is optimized through latent language disentanglement, rendering our optimization objective to not depend on negative samples, thereby significantly mitigating the bias from determining positive-negative sample pairs within analogous medical reports. Furthermore, it ensures that the cross-lingual representation is not biased toward any specific language community.\textbf{Med-UniC} reaches superior performance across 5 medical image tasks and 10 datasets encompassing over 30 diseases, offering a versatile framework for unifying multi-modal medical data within diverse linguistic communities.The experimental outcomes highlight the presence of community bias in cross-lingual VLP. Reducing this bias enhances the performance not only in vision-language tasks but also in uni-modal visual tasks.

----

## [0] CD-GraB: Coordinating Distributed Example Orders for Provably Accelerated Training

**Authors**: *A. Feder Cooper, Wentao Guo, Khiem Pham, Tiancheng Yuan, Charlie Ruan, Yucheng Lu, Christopher De Sa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af9ac087ed9123957bb3a45dca56b9d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af9ac087ed9123957bb3a45dca56b9d4-Abstract-Conference.html)

**Abstract**:

Recent research on online Gradient Balancing (GraB) has revealed that there exist permutation-based example orderings for SGD that are guaranteed to outperform random reshuffling (RR). Whereas RR arbitrarily permutes training examples, GraB leverages stale gradients from prior epochs to order examples -- achieving a provably faster convergence rate than RR. However, GraB is limited by design: while it demonstrates an impressive ability to scale-up training on centralized data, it does not naturally extend to modern distributed ML workloads. We therefore propose Coordinated Distributed GraB (CD-GraB), which uses insights from prior work on kernel thinning to translate the benefits of provably faster permutation-based example ordering to distributed settings. With negligible overhead, CD-GraB exhibits a linear speedup in convergence rate over centralized GraB and outperforms distributed RR on a variety of benchmark tasks.

----

## [0] Individualized Dosing Dynamics via Neural Eigen Decomposition

**Authors**: *Stav Belogolovsky, Ido Greenberg, Danny Eytan, Shie Mannor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afc9f18089928eca34c347fee4757f72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afc9f18089928eca34c347fee4757f72-Abstract-Conference.html)

**Abstract**:

Dosing models often use differential equations to model biological dynamics. Neural differential equations in particular can learn to predict the derivative of a process, which permits predictions at irregular points of time. However, this temporal flexibility often comes with a high sensitivity to noise, whereas medical problems often present high noise and limited data. Moreover, medical dosing models must generalize reliably over individual patients and changing treatment policies. To address these challenges, we introduce the Neural Eigen Stochastic Differential Equation algorithm (NESDE). NESDE provides individualized modeling (using a hypernetwork over patient-level parameters); generalization to new treatment policies (using decoupled control); tunable expressiveness according to the noise level (using piecewise linearity); and fast, continuous, closed-form prediction (using spectral representation). We demonstrate the robustness of NESDE in both synthetic and real medical problems, and use the learned dynamics to publish simulated medical gym environments.

----

## [0] Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems

**Authors**: *Benjamin Coleman, Wang-Cheng Kang, Matthew Fahrbach, Ruoxi Wang, Lichan Hong, Ed H. Chi, Derek Zhiyuan Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afcac2e300bc243d15c25cd4f4040f0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afcac2e300bc243d15c25cd4f4040f0d-Abstract-Conference.html)

**Abstract**:

Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a $d$-dimensional embedding, which introduces hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used for many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multiplexed representations give Pareto-optimal space-accuracy tradeoffs for three public benchmark datasets. Further, we propose a highly practical approach called Unified Embedding with three major benefits: simplified feature configuration, strong adaptation to dynamic data distributions, and compatibility with modern hardware. Unified embedding gives significant improvements in offline and online metrics compared to highly competitive baselines across five web-scale search, ads, and recommender systems, where it serves billions of users across the world in industry-leading products.

----

## [0] Counterfactual Generation with Identifiability Guarantees

**Authors**: *Hanqi Yan, Lingjing Kong, Lin Gui, Yuejie Chi, Eric P. Xing, Yulan He, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afda6bf3fb086eabbaf161ba1cec5a9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afda6bf3fb086eabbaf161ba1cec5a9a-Abstract-Conference.html)

**Abstract**:

Counterfactual generation lies at the core of various machine learning tasks, including image translation and controllable text generation. This generation process usually requires the identification of the disentangled latent representations, such as content and style, that underlie the observed data. However, it becomes more  challenging when faced with a scarcity of paired data and labelling information. Existing disentangled methods crucially rely on oversimplified assumptions, such as assuming independent content and style variables, to identify the latent variables, even though such assumptions may not hold for complex data distributions. For instance, food reviews tend to involve words like “tasty”, whereas movie reviews commonly contain words such as “thrilling” for the same positive sentiment. This problem is exacerbated when data are sampled from multiple domains since the dependence between content and style may vary significantly over domains. In this work, we tackle the domain-varying dependence between the content and the style variables inherent in the counterfactual generation task. We provide identification guarantees for such latent-variable models by leveraging the relative sparsity of the influences from different latent variables. Our theoretical insights enable the development of a doMain AdapTive counTerfactual gEneration model, called (MATTE). Our theoretically grounded framework achieves state-of-the-art performance in unsupervised style transfer tasks, where neither paired data nor style labels are utilized, across four large-scale datasets.

----

## [0] A Batch-to-Online Transformation under Random-Order Model

**Authors**: *Jing Dong, Yuichi Yoshida*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html)

**Abstract**:

We introduce a transformation framework that can be utilized to develop online algorithms with low $\epsilon$-approximate regret in the random-order model from offline approximation algorithms. We first give a general reduction theorem that transforms an offline approximation algorithm with low average sensitivity to an online algorithm with low $\epsilon$-approximate regret. We then demonstrate that offline approximation algorithms can be transformed into a low-sensitivity version using a coreset construction method. To showcase the versatility of our approach, we apply it to various problems, including online $(k,z)$-clustering, online matrix approximation, and online regression, and successfully achieve polylogarithmic $\epsilon$-approximate regret for each problem. Moreover, we show that in all three cases, our algorithm also enjoys low inconsistency, which may be desired in some online applications.

----

## [0] The CLIP Model is Secretly an Image-to-Prompt Converter

**Authors**: *Yuxuan Ding, Chunna Tian, Haoxuan Ding, Lingqiao Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b00ef390dcd5f147fd7c5c2bb35f09be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b00ef390dcd5f147fd7c5c2bb35f09be-Abstract-Conference.html)

**Abstract**:

The Stable Diffusion model is a prominent text-to-image generation model that relies on a text prompt as its input, which is encoded using the Contrastive Language-Image Pre-Training (CLIP). However, text prompts have limitations when it comes to incorporating implicit information from reference images. Existing methods have attempted to address this limitation by employing expensive training procedures involving millions of training samples for image-to-image generation. In contrast, this paper demonstrates that the CLIP model, as utilized in Stable Diffusion, inherently possesses the ability to instantaneously convert images into text prompts. Such an image-to-prompt conversion can be achieved by utilizing a linear projection matrix that is calculated in a closed form. Moreover, the paper showcases that this capability can be further enhanced by either utilizing a small amount of similar-domain training data (approximately 100 images) or incorporating several online training steps (around 30 iterations) on the reference images. By leveraging these approaches, the proposed method offers a simple and flexible solution to bridge the gap between images and text prompts. This methodology can be applied to various tasks such as image variation and image editing, facilitating more effective and seamless interaction between images and textual prompts.

----

## [0] Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective

**Authors**: *João B. S. Carvalho, Mengtao Zhang, Robin Geyer, Carlos Cotrini, Joachim M. Buhmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b010241b9f1cdfc7d4c392db899cef86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b010241b9f1cdfc7d4c392db899cef86-Abstract-Conference.html)

**Abstract**:

Anomaly detection (AD) is the machine learning task of identifying highly discrepant abnormal samples by solely relying on the consistency of the normal training samples. Under the constraints of a distribution shift, the assumption that training samples and test samples are drawn from the same distribution breaks down. In this work, by leveraging tools from causal inference we attempt to increase the resilience of anomaly detection models to different kinds of distribution shifts. We begin by elucidating a simple yet necessary statistical property that ensures invariant representations, which is critical for robust AD under both domain and covariate shifts. From this property, we derive a regularization term which, when minimized, leads to partial distribution invariance across environments. Through extensive experimental evaluation on both synthetic and real-world tasks, covering a range of six different AD methods, we demonstrated significant improvements in out-of-distribution performance. Under both covariate and domain shift, models regularized with our proposed term showed marked increased robustness. Code is available at: https://github.com/JoaoCarv/invariant-anomaly-detection

----

## [0] Stable Bias: Evaluating Societal Representations in Diffusion Models

**Authors**: *Sasha Luccioni, Christopher Akiki, Margaret Mitchell, Yacine Jernite*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

As machine learning-enabled Text-to-Image (TTI) systems are becoming increasingly prevalent and seeing growing adoption as commercial services, characterizing the social biases they exhibit is a necessary first step to lowering their risk of discriminatory outcomes. This evaluation, however, is made more difficult by the synthetic nature of these systems’ outputs: common definitions of diversity are grounded in social categories of people living in the world, whereas the artificial depictions of fictive humans created by these systems have no inherent gender or ethnicity. To address this need, we propose a new method for exploring the social biases in TTI systems. Our approach relies on characterizing the variation in generated images triggered by enumerating gender and ethnicity markers in the prompts, and comparing it to the variation engendered by spanning different professions. This allows us to (1) identify specific bias trends, (2) provide targeted scores to directly compare models in terms of diversity and representation, and (3) jointly model interdependent social variables to support a multidimensional analysis. We leverage this method to analyze images generated by 3 popular TTI systems (Dall·E 2 , Stable Diffusion v 1.4 and 2) and find that while all of their outputs show correlations with US labor demographics, they also consistently under-represent marginalized identities to different extents. We also release the datasets and low-code interactive bias exploration platforms developed forthis work, as well as the necessary tools to similarly evaluate additional TTI systems.

----

## [0] Locality Sensitive Hashing in Fourier Frequency Domain For Soft Set Containment Search

**Authors**: *Indradyumna Roy, Rishi Agarwal, Soumen Chakrabarti, Anirban Dasgupta, Abir De*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b016cbec36ff7118db303229c9048733-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b016cbec36ff7118db303229c9048733-Abstract-Conference.html)

**Abstract**:

In many search applications related to passage retrieval, text entailment, and subgraph search, the query and each 'document' is a set of elements, with a document being relevant if it contains the query. These elements are not represented by atomic IDs, but by  embedded representations, thereby extending set containment to soft set containment. Recent applications address soft set containment by encoding sets into fixed-size vectors and checking for elementwise vector dominance. This 0/1 property can be relaxed to an asymmetric hinge distance for scoring and ranking candidate documents. Here we focus on data-sensitive, trainable indices for fast retrieval of relevant documents. Existing LSH methods are designed for mostly symmetric or few  simple asymmetric distance functions, which are not suitable for hinge distance. Instead, we transform hinge distance into a proposed dominance similarity measure, to which we then apply a Fourier transform, thereby expressing dominance similarity as an expectation of inner products of functions in the frequency domain. Next, we approximate the expectation with an importance-sampled estimate. The overall consequence is that now we can use a traditional LSH, but in the frequency domain. To ensure that the LSH uses hash bits efficiently, we learn hash functions that are sensitive to both corpus and query distributions, mapped to the frequency domain. Our experiments show that the proposed asymmetric dominance similarity is critical to the targeted applications, and that our LSH, which we call FourierHashNet, provides a better query time vs. retrieval quality trade-off, compared to several baselines. Both the Fourier transform and the trainable hash codes contribute to performance gains.

----

## [0] L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference

**Authors**: *Julia Linhart, Alexandre Gramfort, Pedro Rodrigues*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0313c2f4501a81d0e0d4a1e8fbf4995-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0313c2f4501a81d0e0d4a1e8fbf4995-Abstract-Conference.html)

**Abstract**:

Many recent works in simulation-based inference  (SBI) rely on deep generative models to approximate complex, high-dimensional posterior distributions. However, evaluating whether or not these approximations can be trusted remains a challenge. Most approaches evaluate the posterior estimator only in expectation  over the observation space. This limits their interpretability and is not sufficient to identify for which observations the approximation can be trusted or should be improved. Building upon the well-known classifier two-sample test (C2ST), we introduce $\ell$-C2ST, a new method that allows for a local evaluation of the posterior estimator at any given observation. It offers theoretically grounded and easy to interpret -- e.g. graphical -- diagnostics, and unlike C2ST, does not require access to samples from the true posterior. In the case of normalizing flow-based posterior estimators, $\ell$-C2ST can be specialized to offer better statistical power, while being computationally more efficient. On standard SBI benchmarks, $\ell$-C2ST  provides comparable results to C2ST and outperforms alternative local approaches such as coverage tests based on highest predictive density (HPD). We further highlight the importance of local evaluation and the benefit of interpretability of $\ell$-C2ST on a challenging application from computational neuroscience.

----

## [0] Self-Supervised Reinforcement Learning that Transfers using Random Features

**Authors**: *Boyuan Chen, Chuning Zhu, Pulkit Agrawal, Kaiqing Zhang, Abhishek Gupta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b048dd19ba6d85b9066aa93b4de9ad4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b048dd19ba6d85b9066aa93b4de9ad4a-Abstract-Conference.html)

**Abstract**:

Model-free reinforcement learning algorithms have exhibited great potential in solving single-task sequential decision-making problems with high-dimensional observations and long horizons, but are known to be hard to generalize across tasks. Model-based RL, on the other hand, learns task-agnostic models of the world that naturally enables transfer across different reward functions, but struggles to scale to complex environments due to the compounding error. To get the best of both worlds, we propose a self-supervised reinforcement learning method that enables the transfer of behaviors across tasks with different rewards, while circumventing the challenges of model-based RL. In particular, we show self-supervised pre-training of model-free reinforcement learning with a number of random features as rewards allows implicit modeling of long-horizon environment dynamics. Then, planning techniques like model-predictive control using these implicit models enable fast adaptation to problems with new reward functions. Our method is self-supervised in that it can be trained on offline datasets without reward labels, but can then be quickly deployed on new tasks. We validate that our proposed method enables transfer across tasks on a variety of manipulation and locomotion domains in simulation, opening the door to generalist decision-making agents.

----

## [0] MGDD: A Meta Generator for Fast Dataset Distillation

**Authors**: *Songhua Liu, Xinchao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0506debbf49e31d25690fbd1e69cd2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0506debbf49e31d25690fbd1e69cd2f-Abstract-Conference.html)

**Abstract**:

Existing dataset distillation (DD) techniques typically rely on iterative strategies to synthesize condensed datasets, where datasets before and after distillation are forward and backward through neural networks a massive number of times. Despite the promising results achieved, the time efficiency of prior approaches is still far from satisfactory. Moreover, when different sizes of synthetic datasets are required, they have to repeat the iterative training procedures, which is highly cumbersome and lacks flexibility. In this paper, different from the time-consuming forward-backward passes, we introduce a generative fashion for dataset distillation with significantly improved efficiency. Specifically, synthetic samples are produced by a generator network conditioned on the initialization of DD, while synthetic labels are obtained by solving a least-squares problem in a feature space. Our theoretical analysis reveals that the errors of synthetic datasets solved in the original space and then processed by any conditional generators are upper-bounded. To find a satisfactory generator efficiently, we propose a meta-learning algorithm, where a meta generator is trained on a large dataset so that only a few steps are required to adapt to a target dataset. The meta generator is termed as MGDD in our approach. Once adapted, it can handle arbitrary sizes of synthetic datasets, even for those unseen during adaptation. Experiments demonstrate that the generator adapted with only a limited number of steps performs on par with those state-of-the-art DD methods and yields $22\times$ acceleration.

----

## [0] New Complexity-Theoretic Frontiers of Tractability for Neural Network Training

**Authors**: *Cornelius Brand, Robert Ganian, Mathis Rocton*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b07091c16719ad3990e3d1ccee6641f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b07091c16719ad3990e3d1ccee6641f1-Abstract-Conference.html)

**Abstract**:

In spite of the fundamental role of neural networks in contemporary machine learning research, our understanding of the computational complexity of optimally training neural networks remains limited even when dealing with the simplest kinds of activation functions. Indeed, while there has been a number of very recent results that establish ever-tighter lower bounds for the problem under linear and ReLU activation functions, little progress has been made towards the identification of novel polynomial-time tractable network architectures. In this article we obtain novel algorithmic upper bounds for training linear- and ReLU-activated neural networks to optimality which push the boundaries of tractability for these problems beyond the previous state of the art.

----

## [0] V-InFoR: A Robust Graph Neural Networks Explainer for Structurally Corrupted Graphs

**Authors**: *Senzhang Wang, Jun Yin, Chaozhuo Li, Xing Xie, Jianxin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b07d36fb3fae0630897700593c8cf49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b07d36fb3fae0630897700593c8cf49d-Abstract-Conference.html)

**Abstract**:

GNN explanation method aims to identify an explanatory subgraph which contains the most informative components of the full graph. However, a major limitation of existing GNN explainers is that they are not robust to the structurally corrupted graphs, e.g., graphs with noisy or adversarial edges. On the one hand, existing GNN explainers mostly explore explanations based on either the raw graph features or the learned latent representations, both of which can be easily corrupted. On the other hand, the corruptions in graphs are irregular in terms of the structural properties, e.g., the size or connectivity of graphs, which makes the rigorous constraints used by previous GNN explainers unfeasible. To address these issues, we propose a robust GNN explainer called V-InfoR. Specifically, a robust graph representation extractor, which takes insights of variational inference, is proposed to infer the latent distribution of graph representations. Instead of directly using the corrupted raw features or representations of each single graph, we sample the graph representations from the inferred distribution for the downstream explanation generator, which can effectively eliminate the minor corruption. We next formulate the explanation exploration as a graph information bottleneck (GIB) optimization problem. As a more general method that does not need any rigorous structural constraints, our GIB-based method can adaptively capture both the regularity and irregularity of the severely corrupted graphs for explanation. Extensive evaluations on both synthetic and real-world datasets indicate that V-InfoR significantly improves the GNN explanation performance for the structurally corrupted graphs. Code and dataset are available at https://anonymous.4open.science/r/V-InfoR-EF88

----

## [0] Beyond Average Return in Markov Decision Processes

**Authors**: *Alexandre Marthe, Aurélien Garivier, Claire Vernade*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0a34e3c64f7e842f20ec10479c32b35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0a34e3c64f7e842f20ec10479c32b35-Abstract-Conference.html)

**Abstract**:

What are the functionals of the reward that can be computed and optimized exactly in Markov Decision Processes?In the finite-horizon, undiscounted setting, Dynamic Programming (DP) can only handle these operations efficiently for certain classes of statistics. We summarize the characterization of these classes for policy evaluation, and give a new answer for the planning problem. Interestingly, we prove that only generalized means can be optimized exactly, even in the more general framework of Distributional Reinforcement Learning (DistRL).DistRL permits, however, to evaluate other functionals approximately. We provide error bounds on the resulting estimators, and discuss the potential of this approach as well as its limitations.These results contribute to advancing the theory of Markov Decision Processes by examining overall characteristics of the return, and particularly risk-conscious strategies.

----

## [0] Latent exploration for Reinforcement Learning

**Authors**: *Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Zixiang Huang, Alexander Mathis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0ca717599b7ba84d5e4f4c8b1ef6657-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0ca717599b7ba84d5e4f4c8b1ef6657-Abstract-Conference.html)

**Abstract**:

In Reinforcement Learning, agents learn policies by exploring and interacting with the environment. Due to the curse of dimensionality, learning policies that map high-dimensional sensory input to motor output is particularly challenging. During training, state of the art methods (SAC, PPO, etc.) explore the environment by perturbing the actuation with independent Gaussian noise. While this unstructured exploration has proven successful in numerous tasks, it can be suboptimal for overactuated systems. When multiple actuators, such as motors or muscles, drive behavior, uncorrelated perturbations risk diminishing each other's effect, or modifying the behavior in a task-irrelevant way. While solutions to introduce time correlation across action perturbations exist, introducing correlation across actuators has been largely ignored. Here, we propose LATent TIme-Correlated Exploration (Lattice), a method to inject temporally-correlated noise into the latent state of the policy network, which can be seamlessly integrated with on- and off-policy algorithms. We demonstrate that the noisy actions generated by perturbing the network's activations can be modeled as a multivariate Gaussian distribution with a full covariance matrix. In the PyBullet locomotion tasks, Lattice-SAC achieves state of the art results, and reaches 18\% higher reward than unstructured exploration in the Humanoid environment. In the musculoskeletal control environments of MyoSuite, Lattice-PPO achieves higher reward in most reaching and object manipulation tasks, while also finding more energy-efficient policies with reductions of 20-60\%. Overall, we demonstrate the effectiveness of structured action noise in time and actuator space for complex motor control tasks. The code is available at: https://github.com/amathislab/lattice.

----

## [0] Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

**Authors**: *Tyler Kastner, Murat A. Erdogdu, Amir-massoud Farahmand*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)

**Abstract**:

We consider the problem of learning models for risk-sensitive reinforcement learning. We theoretically demonstrate that proper value equivalence, a method of learning models which can be used to plan optimally in the risk-neutral setting, is not sufficient to plan optimally in the risk-sensitive setting. We leverage distributional reinforcement learning to introduce two new notions of model equivalence, one which is general and can be used to plan for any risk measure, but is intractable; and a practical variation which allows one to choose which risk measures they may plan optimally for. We demonstrate how our models can be used to augment any model-free risk-sensitive algorithm, and provide both tabular and large-scale experiments to demonstrate our methodâ€™s ability.

----

## [0] Group Robust Classification Without Any Group Information

**Authors**: *Christos Tsirigotis, João Monteiro, Pau Rodríguez, David Vázquez, Aaron C. Courville*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0d9ceb3d11d013e55da201d2a2c07b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0d9ceb3d11d013e55da201d2a2c07b2-Abstract-Conference.html)

**Abstract**:

Empirical risk minimization (ERM) is sensitive to spurious correlations present in training data, which poses a significant risk when deploying systems trained under this paradigm in high-stake applications. While the existing literature focuses on maximizing group-balanced or worst-group accuracy, estimating these quantities is hindered by costly bias annotations. This study contends that current bias-unsupervised approaches to group robustness continue to rely on group information to achieve optimal performance. Firstly, these methods implicitly assume that all group combinations are represented during training. To illustrate this, we introduce a systematic generalization task on the MPI3D dataset and discover that current algorithms fail to improve the ERM baseline when combinations of observed attribute values are missing. Secondly, bias labels are still crucial for effective model selection, restricting the practicality of these methods in real-world scenarios. To address these limitations, we propose a revised methodology for training and validating debiased models in an entirely bias-unsupervised manner. We achieve this by employing pretrained self-supervised models to reliably extract bias information, which enables the integration of a logit adjustment training loss with our validation criterion. Our empirical analysis on synthetic and real-world tasks provides evidence that our approach overcomes the identified challenges and consistently enhances robust accuracy, attaining performance which is competitive with or outperforms that of state-of-the-art methods, which, conversely, rely on bias labels for validation.

----

## [0] Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation: Minimax Optimal and Instance-Dependent Regret Bounds

**Authors**: *Jiayi Huang, Han Zhong, Liwei Wang, Lin Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b11393733b1ea5890100302ab8a0f74c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b11393733b1ea5890100302ab8a0f74c-Abstract-Conference.html)

**Abstract**:

While numerous works have focused on devising efficient algorithms for reinforcement learning (RL) with uniformly bounded rewards, it remains an open question whether sample or time-efficient algorithms for RL with large state-action space exist when the rewards are \emph{heavy-tailed}, i.e., with only finite $(1+\epsilon)$-th moments for some $\epsilon\in(0,1]$. In this work, we address the challenge of such rewards in RL with linear function approximation. We first design an algorithm, \textsc{Heavy-OFUL}, for heavy-tailed linear bandits, achieving an \emph{instance-dependent} $T$-round regret of $\tilde{O}\big(d T^{\frac{1-\epsilon}{2(1+\epsilon)}} \sqrt{\sum_{t=1}^T \nu_t^2} + d T^{\frac{1-\epsilon}{2(1+\epsilon)}}\big)$, the \emph{first} of this kind. Here, $d$ is the feature dimension, and $\nu_t^{1+\epsilon}$ is the $(1+\epsilon)$-th central moment of the reward at the $t$-th round. We further show the above bound is minimax optimal when applied to the worst-case instances in stochastic and deterministic linear bandits. We then extend this algorithm to the RL settings with linear function approximation. Our algorithm, termed as \textsc{Heavy-LSVI-UCB}, achieves the \emph{first} computationally efficient \emph{instance-dependent} $K$-episode regret of $\tilde{O}(d \sqrt{H \mathcal{U}^*} K^\frac{1}{1+\epsilon} + d \sqrt{H \mathcal{V}^* K})$. Here, $H$ is length of the episode, and $\mathcal{U}^*, \mathcal{V}^*$ are instance-dependent quantities scaling with the central moment of reward and value functions, respectively. We also provide a matching minimax lower bound $\Omega(d H K^{\frac{1}{1+\epsilon}} + d \sqrt{H^3 K})$ to demonstrate the optimality of our algorithm in the worst case. Our result is achieved via a novel robust self-normalized concentration inequality that may be of independent interest in handling heavy-tailed noise in general online regression problems.

----

## [0] Learning Dictionary for Visual Attention

**Authors**: *Yingjie Liu, Xuan Liu, Hui Yu, Xuan Tang, Xian Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b113e1441ad107b80c576b5028fd2c51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b113e1441ad107b80c576b5028fd2c51-Abstract-Conference.html)

**Abstract**:

Recently, the attention mechanism has shown outstanding competence in capturing global structure information and long-range relationships within data, thus enhancing the performance of deep vision models on various computer vision tasks. In this work, we propose a novel dictionary learning-based attention (\textit{Dic-Attn}) module, which models this issue as a decomposition and reconstruction problem with the sparsity prior, inspired by sparse coding in the human visual perception system. The proposed \textit{Dic-Attn} module decomposes the input into a dictionary and corresponding sparse representations, allowing for the disentanglement of underlying nonlinear structural information in visual data and the reconstruction of an attention embedding. By applying transformation operations in the spatial and channel domains, the module dynamically selects the dictionary's atoms and sparse representations. Finally, the updated dictionary and sparse representations capture the global contextual information and reconstruct the attention maps. The proposed \textit{Dic-Attn} module is designed with plug-and-play compatibility, allowing for integration into deep attention encoders. Our approach offers an intuitive and elegant means to exploit the discriminative information from data, promoting visual attention construction. Extensive experimental results on various computer vision tasks, e.g., image and point cloud classification, validate that our method achieves promising performance, and shows a strong competitive comparison with state-of-the-art attention methods.

----

## [0] A Bayesian Take on Gaussian Process Networks

**Authors**: *Enrico Giudice, Jack Kuipers, Giusi Moffa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b146e7c87685fa208bd95ce4b08e330c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b146e7c87685fa208bd95ce4b08e330c-Abstract-Conference.html)

**Abstract**:

Gaussian Process Networks (GPNs) are a class of directed graphical models which employ Gaussian processes as priors for the conditional expectation of each variable given its parents in the network. The model allows the description of continuous joint distributions in a compact but flexible manner with minimal parametric assumptions on the dependencies between variables. Bayesian structure learning of GPNs requires computing the posterior over graphs of the network and is computationally infeasible even in low dimensions. This work implements Monte Carlo and Markov Chain Monte Carlo methods to sample from the posterior distribution of network structures. As such, the approach follows the Bayesian paradigm, comparing models via their marginal likelihood and computing the posterior probability of the GPN features. Simulation studies show that our method outperforms state-of-the-art algorithms in recovering the graphical structure of the network and provides an accurate approximation of its posterior distribution.

----

## [0] Exploring Question Decomposition for Zero-Shot VQA

**Authors**: *Zaid Khan, Vijay Kumar B. G, Samuel Schulter, Manmohan Chandraker, Yun Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b14cf0a01f7a8b9cd3e365e40f910272-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b14cf0a01f7a8b9cd3e365e40f910272-Abstract-Conference.html)

**Abstract**:

Visual question answering (VQA) has traditionally been treated as a single-step task where each question receives the same amount of effort, unlike natural human question-answering strategies. We explore a question decomposition strategy for VQA to overcome this limitation. We probe the ability of recently developed large vision-language models to use human-written decompositions and produce their own decompositions of visual questions, finding they are capable of learning both tasks from demonstrations alone.However, we show that naive application of model-written decompositions can hurt performance.We introduce a model-driven selective decomposition approach for second-guessing predictions and correcting errors, and validate its effectiveness on eight VQA tasks across three domains, showing consistent improvements in accuracy, including improvements of >20% on medical VQA datasets and boosting the zero-shot performance of BLIP-2 above chance on a VQA reformulation of the challenging Winoground task. Project Site: https://zaidkhan.me/decomposition-0shot-vqa/

----

## [0] Sharp Recovery Thresholds of Tensor PCA Spectral Algorithms

**Authors**: *Michael Feldman, David Donoho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b14d76c7266be21b338527cd25deac45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b14d76c7266be21b338527cd25deac45-Abstract-Conference.html)

**Abstract**:

Many applications seek to recover low-rank approximations of noisy tensor data. We consider several practical and effective matricization strategies which construct specific matrices from such tensors and then apply spectral methods; the strategies include tensor unfolding, partial tracing, power iteration, and recursive unfolding. We settle the behaviors of unfolding and partial tracing, identifying sharp thresholds in signal-to-noise ratio above which the signal is partially recovered. In particular, we extend previous results to a much larger class of tensor shapes where axis lengths may be different. For power iteration and recursive unfolding, we prove that under conditions where previous algorithms partially recovery the signal, these methods achieve (asymptotically) exact recovery. Our analysis deploys random matrix theory to obtain sharp thresholds which elude perturbation and concentration bounds. Specifically, we rely upon recent disproportionate random matrix results, which describe sequences of matrices with diverging aspect ratio.

----

## [0] R-divergence for Estimating Model-oriented Distribution Discrepancy

**Authors**: *Zhilin Zhao, Longbing Cao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b157cfde6794e93b2353b9712bbd45a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b157cfde6794e93b2353b9712bbd45a5-Abstract-Conference.html)

**Abstract**:

Real-life data are often non-IID due to complex distributions and interactions, and the sensitivity to the distribution of samples can differ among learning models. Accordingly, a key question for any supervised or unsupervised model is whether the probability distributions of two given datasets can be considered identical. To address this question, we introduce R-divergence, designed to assess model-oriented distribution discrepancies. The core insight is that two distributions are likely identical if their optimal hypothesis yields the same expected risk for each distribution. To estimate the distribution discrepancy between two datasets, R-divergence learns a minimum hypothesis on the mixed data and then gauges the empirical risk difference between them. We evaluate the test power across various unsupervised and supervised tasks and find that R-divergence achieves state-of-the-art performance. To demonstrate the practicality of R-divergence, we employ R-divergence to train robust neural networks on samples with noisy labels.

----

## [0] On-the-Fly Adapting Code Summarization on Trainable Cost-Effective Language Models

**Authors**: *YuFan Cai, Yun Lin, Chenyan Liu, Jinglian Wu, Yifan Zhang, Yiming Liu, Yeyun Gong, Jin Song Dong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b16e6de5fbbdcb2df237aa66b302bc17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b16e6de5fbbdcb2df237aa66b302bc17-Abstract-Conference.html)

**Abstract**:

Deep learning models are emerging to summarize source code to comment,  facilitating tasks of code documentation and program comprehension.  Scaled-up large language models trained on large open corpus have achieved good performance in such tasks.  However, in practice, the subject code in one certain project can be specific,  which may not align with the overall training corpus.  Some code samples from other projects may be contradictory and introduce inconsistencies when the models try to fit all the samples.  In this work, we introduce a novel approach, Adacom, to improve the performance of comment generators by on-the-fly model adaptation.  This research is motivated by the observation that deep comment generators  often need to strike a balance as they need to fit all the training samples.  Specifically, for one certain target code $c$,  some training samples $S_p$ could have made more contributions while other samples $S_o$ could have counter effects.  However, the traditional fine-tuned models need to fit both $S_p$ and $S_o$ from a global perspective,   leading to compromised performance for one certain target code $c$.  In this context, we design Adacom to  (1) detect whether the model might have a compromised performance on a target code $c$ and  (2) retrieve a few helpful training samples $S_p$ that have contradictory samples in the training dataset and,  (3) adapt the model on the fly by re-training the $S_p$ to strengthen the helpful samples and unlearn the harmful samples.  Our extensive experiments on 7 comment generators and 4 public datasets show that  (1) can significantly boost the performance of comment generation (BLEU4 score by on average 14.9\%, METEOR by 12.2\%, and ROUGE-L by 7.4\%),  (2) the adaptation on one code sample is cost-effective and acceptable as an on-the-fly solution, and  (3) can adapt well on out-of-distribution code samples.

----

## [0] Exploring and Interacting with the Set of Good Sparse Generalized Additive Models

**Authors**: *Chudi Zhong, Zhi Chen, Jiachang Liu, Margo I. Seltzer, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1719f44953c2e0754a016ab267fe4e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1719f44953c2e0754a016ab267fe4e7-Abstract-Conference.html)

**Abstract**:

In real applications, interaction between machine learning models and domain experts is critical; however, the classical machine learning paradigm that usually produces only a single model does not facilitate such interaction. Approximating and exploring the Rashomon set, i.e., the set of all near-optimal models, addresses this practical challenge by providing the user with a searchable space containing a diverse set of models from which domain experts can choose. We present algorithms to efficiently and accurately approximate the Rashomon set of sparse, generalized additive models with ellipsoids for fixed support sets and use these ellipsoids to approximate Rashomon sets for many different support sets. The approximated Rashomon set serves as a cornerstone to solve practical challenges such as (1) studying the variable importance for the model class; (2) finding models under user-specified constraints (monotonicity, direct editing); and (3) investigating sudden changes in the shape functions. Experiments demonstrate the fidelity of the approximated Rashomon set and its effectiveness in solving practical challenges.

----

## [0] Convergence Analysis of Sequential Federated Learning on Heterogeneous Data

**Authors**: *Yipeng Li, Xinchen Lyu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b18e5d6a10ba57d5273871f38189f062-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b18e5d6a10ba57d5273871f38189f062-Abstract-Conference.html)

**Abstract**:

There are two categories of methods in Federated Learning (FL) for joint training across multiple clients: i) parallel FL (PFL), where clients train models in a parallel manner; and ii) sequential FL (SFL), where clients train models in a sequential manner. In contrast to that of PFL, the convergence theory of SFL on heterogeneous data is still lacking. In this paper, we establish the convergence guarantees of SFL for strongly/general/non-convex objectives on heterogeneous data. The convergence guarantees of SFL are better than that of PFL on heterogeneous data with both full and partial client participation. Experimental results validate the counterintuitive analysis result that SFL outperforms PFL on extremely heterogeneous data in cross-device settings.

----

## [0] Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning

**Authors**: *Wei Tang, Weijia Zhang, Min-Ling Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1917a4bcfab403c3cdd6c6bbaf9fda0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1917a4bcfab403c3cdd6c6bbaf9fda0-Abstract-Conference.html)

**Abstract**:

In many real-world tasks, the concerned objects can be represented as a multi-instance bag associated with a candidate label set, which consists of one ground-truth label and several false positive labels. Multi-instance partial-label learning (MIPL) is a learning paradigm to deal with such tasks and has achieved favorable performances. Existing MIPL approach follows the instance-space paradigm by assigning augmented candidate label sets of bags to each instance and aggregating bag-level labels from instance-level labels. However, this scheme may be suboptimal as global bag-level information is ignored and the predicted labels of bags are sensitive to predictions of negative instances. In this paper, we study an alternative scheme where a multi-instance bag is embedded into a single vector representation. Accordingly, an intuitive algorithm named DEMIPL, i.e., Disambiguated attention Embedding for Multi-Instance Partial-Label learning, is proposed. DEMIPL employs a disambiguation attention mechanism to aggregate a multi-instance bag into a single vector representation, followed by a momentum-based disambiguation strategy to identify the ground-truth label from the candidate label set. Furthermore, we introduce a real-world MIPL dataset for colorectal cancer classification. Experimental results on benchmark and real-world datasets validate the superiority of DEMIPL against the compared MIPL and partial-label learning approaches.

----

## [0] Efficient Batched Algorithm for Contextual Linear Bandits with Large Action Space via Soft Elimination

**Authors**: *Osama A. Hanna, Lin Yang, Christina Fragouli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1bdb0f22c9748203c62f29aa297ac57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1bdb0f22c9748203c62f29aa297ac57-Abstract-Conference.html)

**Abstract**:

In this paper, we provide the first efficient batched algorithm for contextual linear bandits with large action spaces. Unlike existing batched algorithms that rely on action elimination, which are not implementable for large action sets, our algorithm only uses a linear optimization oracle over the action set to design the policy. The proposed algorithm achieves a regret upper bound $\tilde{O}(\sqrt{T})$ with high probability,  and uses $O(\log\log T)$ batches, matching the lower bound on the number of batches (Gao et al., 2019). When specialized to linear bandits, our algorithm can achieve a high probability gap-dependent regret bound of $\tilde{O}(1/\Delta_{\min})$ with the optimal  $\log T$ number of batches, where $\Delta_{\min}$ is the minimum reward gap between a suboptimal arm and the optimal. Our result is achieved via a novel soft elimination approach, that entails $\text{``}$shaping$\text{"}$ the action sets at each batch so that we can efficiently identify (near) optimal actions.

----

## [0] Add and Thin: Diffusion for Temporal Point Processes

**Authors**: *David Lüdke, Marin Bilos, Oleksandr Shchur, Marten Lienen, Stephan Günnemann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract-Conference.html)

**Abstract**:

Autoregressive neural networks within the temporal point process (TPP) framework have become the standard for modeling continuous-time event data. Even though these models can expressively capture event sequences in a one-step-ahead fashion, they are inherently limited for long-term forecasting applications due to the accumulation of errors caused by their sequential nature. To overcome these limitations, we derive ADD-THIN, a principled probabilistic denoising diffusion model for TPPs that operates on entire event sequences. Unlike existing diffusion approaches, ADD-THIN naturally handles data with discrete and continuous components. In experiments on synthetic and real-world datasets, our model matches the state-of-the-art TPP models in density estimation and strongly outperforms them in forecasting.

----

## [0] Pitfall of Optimism: Distributional Reinforcement Learning by Randomizing Risk Criterion

**Authors**: *Taehyun Cho, Seungyub Han, Heesoo Lee, Kyungjae Lee, Jungwoo Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1eb88348ee19a33c81cf5bc3fb8e9d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1eb88348ee19a33c81cf5bc3fb8e9d2-Abstract-Conference.html)

**Abstract**:

Distributional reinforcement learning algorithms have attempted to utilize estimated uncertainty for exploration, such as optimism in the face of uncertainty. However, using the estimated variance for optimistic exploration may cause biased data collection and hinder convergence or performance. In this paper, we present a novel distributional reinforcement learning that selects actions by randomizing risk criterion without losing the risk-neutral objective. We provide a perturbed distributional Bellman optimality operator by distorting the risk measure. Also,we prove the convergence and optimality of the proposed method with the weaker contraction property. Our theoretical results support that the proposed method does not fall into biased exploration and is guaranteed to converge to an optimal return. Finally, we empirically show that our method outperforms other existing distribution-based algorithms in various environments including Atari 55 games.

----

## [0] Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization

**Authors**: *Jinbiao Chen, Jiahai Wang, Zizhen Zhang, Zhiguang Cao, Te Ye, Siyuan Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)

**Abstract**:

Recently, neural heuristics based on deep reinforcement learning have exhibited promise in solving multi-objective combinatorial optimization problems (MOCOPs). However, they are still struggling to achieve high learning efficiency and solution quality. To tackle this issue, we propose an efficient meta neural heuristic (EMNH), in which a meta-model is first trained and then fine-tuned with a few steps to solve corresponding single-objective subproblems. Specifically, for the training process, a (partial) architecture-shared multi-task model is leveraged to achieve parallel learning for the meta-model, so as to speed up the training; meanwhile, a scaled symmetric sampling method with respect to the weight vectors is designed to stabilize the training. For the fine-tuning process, an efficient hierarchical method is proposed to systematically tackle all the subproblems. Experimental results on the multi-objective traveling salesman problem (MOTSP), multi-objective capacitated vehicle routing problem (MOCVRP), and multi-objective knapsack problem (MOKP) show that, EMNH is able to outperform the state-of-the-art neural heuristics in terms of solution quality and learning efficiency, and yield competitive solutions to the strong traditional heuristics while consuming much shorter time.

----

## [0] QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution

**Authors**: *Haotong Qin, Yulun Zhang, Yifu Ding, Yifan Liu, Xianglong Liu, Martin Danelljan, Fisher Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2169d573d75ff90c7b12dc3a5fc2898-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2169d573d75ff90c7b12dc3a5fc2898-Abstract-Conference.html)

**Abstract**:

Low-bit quantization in image super-resolution (SR) has attracted copious attention in recent research due to its ability to reduce parameters and operations significantly. However, many quantized SR models suffer from accuracy degradation compared to their full-precision counterparts, especially at ultra-low bit widths (2-4 bits), limiting their practical applications. To address this issue, we propose a novel quantized image SR network, called QuantSR, which achieves accurate and efficient SR processing under low-bit quantization. To overcome the representation homogeneity caused by quantization in the network, we introduce the Redistribution-driven Learnable Quantizer (RLQ). This is accomplished through an inference-agnostic efficient redistribution design, which adds additional information in both forward and backward passes to improve the representation ability of quantized networks. Furthermore, to achieve flexible inference and break the upper limit of accuracy, we propose the Depth-dynamic Quantized Architecture (DQA). Our DQA allows for the trade-off between efficiency and accuracy during inference through weight sharing. Our comprehensive experiments show that QuantSR outperforms existing state-of-the-art quantized SR networks in terms of accuracy while also providing more competitive computational efficiency. In addition, we demonstrate the scheme's satisfactory architecture generality by providing QuantSR-C and QuantSR-T for both convolution and Transformer versions, respectively. Our code and models are released at https://github.com/htqin/QuantSR .

----

## [0] Streaming Factor Trajectory Learning for Temporal Tensor Decomposition

**Authors**: *Shikai Fang, Xin Yu, Shibo Li, Zheng Wang, Mike Kirby, Shandian Zhe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b231d91e700c465dfdd6116d091a4194-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b231d91e700c465dfdd6116d091a4194-Abstract-Conference.html)

**Abstract**:

Practical tensor data is often along with time information. Most existing temporal decomposition approaches estimate a set of fixed factors for the objects in each tensor mode, and hence cannot capture the temporal evolution of the objects' representation. More important, we lack an effective approach to capture such evolution from streaming data, which is common in real-world applications.  To address these issues, we propose Streaming Factor Trajectory Learning (SFTL) for temporal tensor decomposition. We use Gaussian processes (GPs) to model the trajectory of  factors so as to flexibly estimate their temporal evolution. To address the computational challenges in handling streaming data, we convert the GPs into a state-space prior by constructing an equivalent stochastic differential equation (SDE).  We develop an efficient online filtering algorithm to estimate a decoupled running posterior of the involved factor states upon receiving new data. The decoupled estimation enables us to conduct standard Rauch-Tung-Striebel smoothing to compute the full posterior of all the  trajectories in parallel, without the need for revisiting any previous data. We have shown the advantage of SFTL in both synthetic tasks and real-world applications.

----

## [0] DDF-HO: Hand-Held Object Reconstruction via Conditional Directed Distance Field

**Authors**: *Chenyangguang Zhang, Yan Di, Ruida Zhang, Guangyao Zhai, Fabian Manhardt, Federico Tombari, Xiangyang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2876deb92cbd098219a10da25671577-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2876deb92cbd098219a10da25671577-Abstract-Conference.html)

**Abstract**:

Reconstructing hand-held objects from a single RGB image is an important and challenging problem. Existing works utilizing Signed Distance Fields (SDF) reveal limitations in comprehensively capturing the complex hand-object interactions, since SDF is  only reliable within the proximity of the target, and hence, infeasible to simultaneously encode local hand and object cues. To address this issue, we propose DDF-HO, a novel approach leveraging Directed Distance Field (DDF) as the shape representation. Unlike SDF, DDF maps a ray in 3D space, consisting of an origin and a direction, to corresponding DDF values, including a binary visibility signal determining whether the ray intersects the objects and a distance value measuring the distance from origin to target in the given direction. We randomly sample multiple rays and collect local to global geometric features for them by introducing a novel 2D ray-based feature aggregation scheme and a 3D intersection-aware hand pose embedding, combining 2D-3D features to model hand-object interactions. Extensive experiments on synthetic and real-world datasets demonstrate that DDF-HO consistently outperforms all baseline methods by a large margin, especially under Chamfer Distance, with about 80% leap forward. Codes are available at https://github.com/ZhangCYG/DDFHO.

----

## [0] Effective Targeted Attacks for Adversarial Self-Supervised Learning

**Authors**: *Minseon Kim, Hyeonjeong Ha, Sooel Son, Sung Ju Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b28ae1166e1035c26b89d20f0286c9eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b28ae1166e1035c26b89d20f0286c9eb-Abstract-Conference.html)

**Abstract**:

Recently, unsupervised adversarial training (AT) has been highlighted as a means of achieving robustness in models without any label information. Previous studies in unsupervised AT have mostly focused on implementing self-supervised learning (SSL) frameworks, which maximize the instance-wise classification loss to generate adversarial examples. However, we observe that simply maximizing the self-supervised training loss with an untargeted adversarial attack often results in generating ineffective adversaries that may not help improve the robustness of the trained model, especially for non-contrastive SSL frameworks without negative examples. To tackle this problem, we propose a novel positive mining for targeted adversarial attack to generate effective adversaries for adversarial SSL frameworks. Specifically, we introduce an algorithm that selects the most confusing yet similar target example for a given instance based on entropy and similarity, and subsequently perturbs the given instance towards the selected target. Our method demonstrates significant enhancements in robustness when applied to non-contrastive SSL frameworks, and less but consistent robustness improvements with contrastive SSL frameworks, on the benchmark datasets.

----

## [0] Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory

**Authors**: *Sokhna Diarra Mbacke, Florence Clerc, Pascal Germain*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29500824d22ee9bbd25e4cd97c49b55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29500824d22ee9bbd25e4cd97c49b55-Abstract-Conference.html)

**Abstract**:

Since their inception, Variational Autoencoders (VAEs) have become central in machine learning. Despite their widespread use, numerous questions regarding their theoretical properties remain open. Using PAC-Bayesian theory, this work develops statistical guarantees for VAEs. First, we derive the first PAC-Bayesian bound for posterior distributions conditioned on individual samples from the data-generating distribution. Then, we utilize this result to develop generalization guarantees for the VAE's reconstruction loss, as well as upper bounds on the distance between the input and the regenerated distributions. More importantly, we provide upper bounds on the Wasserstein distance between the input distribution and the distribution defined by the VAE's generative model.

----

## [0] Multi-Head Adapter Routing for Cross-Task Generalization

**Authors**: *Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, Alessandro Sordoni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html)

**Abstract**:

Parameter-efficient fine-tuning (PEFT) for cross-task generalization consists in pre-training adapters on a multi-task training set before few-shot adaptation to test tasks. Polytropon [Ponti et al., 2023] ($\texttt{Poly}$) jointly learns an inventory of adapters and a *routing* function that selects a (variable-size) subset of adapters for each task during both pre-training and few-shot adaptation. In this paper, we investigate the role that adapter routing plays in its success and design new variants based on our findings.First, we build on the intuition that finer-grained routing provides more expressivity. Hence,we propose  $\texttt{MHR}$ (Multi-Head Routing) which combines *subsets* of adapter parameters and outperforms $\texttt{Poly}$ under a comparable parameter budget; by only fine-tuning the routing function and not the adapters ($\texttt{MHR}$-$z$) we achieve competitive performance with extreme parameter efficiency. Second, we find that  $\texttt{Poly}$/$\texttt{MHR}$ performance is a result of better multi-task optimization, rather than modular inductive biases that facilitate adapter recombination and local adaptation, as previously hypothesized. In fact, we find that $\texttt{MHR}$ exhibits high gradient alignment between training tasks. We find that routing is most beneficial during multi-task pre-training rather than during few-shot adaptation and propose $\texttt{MHR}$-$\mu$, which discards routing and fine-tunes the average of the pre-trained adapters on each downstream tasks. This establishes $\texttt{MHR}$-$\mu$ as an effective method for single-adapter fine-tuning. We also show that $\texttt{MHR}$-$\mu$ can be used as an effective zero-shot transfer method by training the average of the pre-trained adapters for a few additional steps on the multi-task training set: this yields gains up to 3\% on absolute accuracy w.r.t. the baselines. Code is available at .

----

## [0] GenS: Generalizable Neural Surface Reconstruction from Multi-View Images

**Authors**: *Rui Peng, Xiaodong Gu, Luyang Tang, Shihe Shen, Fanqi Yu, Ronggang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29ab822442a1616f9bd390fddf6e425-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29ab822442a1616f9bd390fddf6e425-Abstract-Conference.html)

**Abstract**:

Combining the signed distance function (SDF) and differentiable volume rendering has emerged as a powerful paradigm for surface reconstruction from multi-view images without 3D supervision. However, current methods are impeded by requiring long-time per-scene optimizations and cannot generalize to new scenes. In this paper, we present GenS, an end-to-end generalizable neural surface reconstruction model. Unlike coordinate-based methods that train a separate network for each scene, we construct a generalized multi-scale volume to directly encode all scenes. Compared with existing solutions, our representation is more powerful, which can recover high-frequency details while maintaining global smoothness. Meanwhile, we introduce a multi-scale feature-metric consistency to impose the multi-view consistency in a more discriminative multi-scale feature space, which is robust to the failures of the photometric consistency. And the learnable feature can be self-enhanced to continuously improve the matching accuracy and mitigate aggregation ambiguity. Furthermore, we design a view contrast loss to force the model to be robust to those regions covered by few viewpoints through distilling the geometric prior from dense input to sparse input. Extensive experiments on popular benchmarks show that our model can generalize well to new scenes and outperform existing state-of-the-art methods even those employing ground-truth depth supervision. Code will be available at https://github.com/prstrive/GenS.

----

## [0] Better with Less: A Data-Active Perspective on Pre-Training Graph Neural Networks

**Authors**: *Jiarong Xu, Renhong Huang, Xin Jiang, Yuxuan Cao, Carl Yang, Chunping Wang, Yang Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29adb4bf2364acec8fb402ef731bb3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29adb4bf2364acec8fb402ef731bb3b-Abstract-Conference.html)

**Abstract**:

Pre-training on graph neural networks (GNNs) aims to learn transferable knowledge for downstream tasks with unlabeled data, and it has recently become an active research area. The success of graph pre-training models is often attributed to the massive amount of input data. In this paper, however, we identify the curse of big data phenomenon in graph pre-training: more training data do not necessarily lead to better downstream performance. Motivated by this observation, we propose a better-with-less framework for graph pre-training: fewer, but carefully chosen data are fed into a GNN model to enhance pre-training. The proposed pre-training pipeline is called the data-active graph pre-training (APT) framework, and is composed of a graph selector and a pre-training model. The graph selector chooses the most representative and instructive data points based on the inherent properties of graphs as well as predictive uncertainty. The proposed predictive uncertainty, as feedback from the pre-training model, measures the confidence level of the model in the data. When fed with the chosen data, on the other hand, the pre-training model grasps an initial understanding of the new, unseen data, and at the same time attempts to remember the knowledge learned from previous data. Therefore, the integration and interaction between these two components form a unified framework (APT), in which graph pre-training is performed in a progressive and iterative way. Experiment results show that the proposed APT is able to obtain an efficient pre-training model with fewer training data and better downstream performance.

----

## [0] Brain-like Flexible Visual Inference by Harnessing Feedback Feedforward Alignment

**Authors**: *Tahereh Toosi, Elias B. Issa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29ec434e049fb96f3c4245a405ee976-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29ec434e049fb96f3c4245a405ee976-Abstract-Conference.html)

**Abstract**:

In natural vision, feedback connections support versatile visual inference capabilities such as making sense of the occluded or noisy bottom-up sensory information or mediating pure top-down processes such as imagination. However, the mechanisms by which the feedback pathway learns to give rise to these capabilities flexibly are not clear. We propose that top-down effects emerge through alignment between feedforward and feedback pathways, each optimizing its own objectives. To achieve this co-optimization, we introduce Feedback-Feedforward Alignment (FFA), a learning algorithm that leverages feedback and feedforward pathways as mutual credit assignment computational graphs, enabling alignment. In our study, we demonstrate the effectiveness of FFA in co-optimizing classification and reconstruction tasks on widely used MNIST and CIFAR10 datasets. Notably, the alignment mechanism in FFA endows feedback connections with emergent visual inference functions, including denoising, resolving occlusions, hallucination, and imagination. Moreover, FFA offers bio-plausibility compared to traditional backpropagation (BP) methods in implementation. By repurposing the computational graph of credit assignment into a goal-driven feedback pathway, FFA alleviates weight transport problems encountered in BP, enhancing the bio-plausibility of the learning algorithm. Our study presents FFA as a promising proof-of-concept for the mechanisms underlying how feedback connections in the visual cortex support flexible visual functions. This work also contributes to the broader field of visual inference underlying perceptual phenomena and has implications for developing more biologically inspired learning algorithms.

----

## [0] Latent Diffusion for Language Generation

**Authors**: *Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, Kilian Q. Weinberger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html)

**Abstract**:

Diffusion models have achieved great success in modeling continuous data modalities such as images, audio, and video, but have seen limited use in discrete domains such as language. Recent attempts to adapt diffusion to language have presented diffusion as an alternative to existing pretrained language models. We view diffusion and existing language models as complementary. We demonstrate that encoder-decoder language models can be utilized to efficiently learn high-quality language autoencoders. We then demonstrate that continuous diffusion models can be learned in the latent space of the language autoencoder, enabling us to sample continuous latent representations that can be decoded into natural language with the pretrained decoder. We validate the effectiveness of our approach for unconditional, class-conditional, and sequence-to-sequence language generation. We demonstrate across multiple diverse data sets that our latent language diffusion models are significantly more effective than previous diffusion language models. Our code is available at \url{https://github.com/justinlovelace/latent-diffusion-for-language}.

----

## [0] The emergence of clusters in self-attention dynamics

**Authors**: *Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html)

**Abstract**:

Viewing Transformers as interacting particle systems, we describe the geometry of learned representations when the weights are not time-dependent. We show that particles, representing tokens, tend to cluster toward particular limiting objects as time tends to infinity. Using techniques from dynamical systems and partial differential equations, we show that type of limiting object that emerges depends on the spectrum of the value matrix. Additionally, in the one-dimensional case we prove that the self-attention matrix converges to a low-rank Boolean matrix. The combination of these results mathematically confirms the empirical observation made by Vaswani et al. [ VSP`17 ] that leaders appear in a sequence of tokens when processed by Transformers.

----

## [0] Self-Consistent Velocity Matching of Probability Flows

**Authors**: *Lingxiao Li, Samuel Hurault, Justin M. Solomon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html)

**Abstract**:

We present a discretization-free scalable framework for solving a large class of mass-conserving partial differential equations (PDEs), including the time-dependent Fokker-Planck equation and the Wasserstein gradient flow. The main observation is that the time-varying velocity field of the PDE solution needs to be self-consistent: it must satisfy a fixed-point equation involving the probability flow characterized by the same velocity field. Instead of directly minimizing the residual of the fixed-point equation with neural parameterization, we use an iterative formulation with a biased gradient estimator that bypasses significant computational obstacles with strong empirical performance. Compared to existing approaches, our method does not suffer from temporal or spatial discretization, covers a wider range of PDEs, and scales to high dimensions. Experimentally, our method recovers analytical solutions accurately when they are available and achieves superior performance in high dimensions with less training time compared to alternatives.

----

## [0] Deep Momentum Multi-Marginal Schrödinger Bridge

**Authors**: *Tianrong Chen, Guan-Horng Liu, Molei Tao, Evangelos A. Theodorou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2c39fe6ce838440faf03a0f780e7a63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2c39fe6ce838440faf03a0f780e7a63-Abstract-Conference.html)

**Abstract**:

It is a crucial challenge to reconstruct population dynamics using unlabeled samples from distributions at coarse time intervals. Recent approaches such as flow-based models or Schrödinger Bridge (SB) models have demonstrated appealing performance, yet the inferred sample trajectories either fail to account for the underlying stochasticity or are unnecessarily rigid. In this article, we extend SB into phase space and propose $\underline{D}$eep $\underline{M}$omentum Multi-Marginal $\underline{S}$chrödinger $\underline{B}$ridge (DMSB), a novel computational framework that learns the smooth measure-valued spline for stochastic systems that satisfy position marginal constraints across time. By tailoring the celebrated Bregman Iteration and extending the Iteration Proportional Fitting to phase space, we manage to handle high-dimensional multi-marginal trajectory inference tasks efficiently. Our algorithm outperforms baselines significantly, as evidenced by experiments for synthetic datasets and a real-world single-cell RNA sequence dataset. Additionally, the proposed approach can reasonably reconstruct the evolution of velocity distribution, from position snapshots only, when there is a ground truth velocity that is nevertheless inaccessible.

----

## [0] Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation

**Authors**: *Weihang Dai, Yao Du, Hanru Bai, Kwang-Ting Cheng, Xiaomeng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2d4051f03a7038a2771dfbbe5c7b54e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2d4051f03a7038a2771dfbbe5c7b54e-Abstract-Conference.html)

**Abstract**:

Contrastive learning methods can be applied to deep regression by enforcing label distance relationships in feature space. However, these methods are limited to labeled data only unlike for classification, where unlabeled data can be used for contrastive pretraining. In this work, we extend contrastive regression methods to allow unlabeled data to be used in a semi-supervised setting, thereby reducing the reliance on manual annotations. We observe that the feature similarity matrix between unlabeled samples still reflect inter-sample relationships, and that an accurate ordinal relationship can be recovered through spectral seriation algorithms if the level of error is within certain bounds. By using the recovered ordinal relationship for contrastive learning on unlabeled samples, we can allow more data to be used for feature representation learning, thereby achieve more robust results. The ordinal rankings can also be used to supervise predictions on unlabeled samples, which can serve as an additional training signal. We provide theoretical guarantees and empirical support through experiments on different datasets, demonstrating that our method can surpass existing state-of-the-art semi-supervised deep regression methods. To the best of our knowledge, this work is the first to explore using unlabeled data to perform contrastive learning for regression.

----

## [0] Domain Re-Modulation for Few-Shot Generative Domain Adaptation

**Authors**: *Yi Wu, Ziqiang Li, Chaoyue Wang, Heliang Zheng, Shanshan Zhao, Bin Li, Dacheng Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html)

**Abstract**:

In this study, we delve into the task of few-shot Generative Domain Adaptation (GDA), which involves transferring a pre-trained generator from one domain to a new domain using only a few reference images. Inspired by the way human brains acquire knowledge in new domains, we present an innovative generator structure called $\textbf{Domain Re-Modulation (DoRM)}$. DoRM not only meets the criteria of $\textit{high quality}$, $\textit{large synthesis diversity}$, and $\textit{cross-domain consistency}$, which were achieved by previous research in GDA, but also incorporates $\textit{memory}$ and $\textit{domain association}$, akin to how human brains operate. Specifically, DoRM freezes the source generator and introduces new mapping and affine modules (M\&A modules) to capture the attributes of the target domain during GDA. This process resembles the formation of new synapses in human brains. Consequently, a linearly combinable domain shift occurs in the style space. By incorporating multiple new M\&A modules, the generator gains the capability to perform high-fidelity multi-domain and hybrid-domain generation. Moreover, to maintain cross-domain consistency more effectively, we introduce a similarity-based structure loss. This loss aligns the auto-correlation map of the target image with its corresponding auto-correlation map of the source image during training. Through extensive experiments, we demonstrate the superior performance of our DoRM and similarity-based structure loss in few-shot GDA, both quantitatively and qualitatively. Code will be available at https://github.com/wuyi2020/DoRM.

----

## [0] Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

**Authors**: *Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2e63e36c57e153b9015fece2352a9f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2e63e36c57e153b9015fece2352a9f9-Abstract-Conference.html)

**Abstract**:

Neural sequence models based on the transformer architecture have demonstrated remarkable \emph{in-context learning} (ICL) abilities, where they can perform new tasks when prompted with training and test examples, without any parameter update to the model. This work first provides a comprehensive statistical theory for transformers to perform ICL. Concretely, we show that transformers can implement a broad class of standard machine learning algorithms in context, such as least squares, ridge regression, Lasso, learning generalized linear models, and gradient descent on two-layer neural networks, with near-optimal predictive power on various in-context data distributions. Using an efficient implementation of in-context gradient descent as the underlying mechanism, our transformer constructions admit mild size bounds, and can be learned with polynomially many pretraining sequences.    Building on these ``base'' ICL algorithms, intriguingly, we show that transformers can implement more complex ICL procedures involving \emph{in-context algorithm selection}, akin to what a statistician can do in real life---A \emph{single} transformer can adaptively select different base ICL algorithms---or even perform qualitatively different tasks---on different input sequences, without any explicit prompting of the right algorithm or task. We both establish this in theory by explicit constructions, and also observe this phenomenon experimentally. In theory, we construct two general mechanisms for algorithm selection with concrete examples: pre-ICL testing, and post-ICL validation. As an example, we use the post-ICL validation mechanism to construct a transformer that can perform nearly Bayes-optimal ICL on a challenging task---noisy linear models with mixed noise levels. Experimentally, we demonstrate the strong in-context algorithm selection capabilities of standard transformer architectures.

----

## [0] Arbitrarily Scalable Environment Generators via Neural Cellular Automata

**Authors**: *Yulun Zhang, Matthew C. Fontaine, Varun Bhatt, Stefanos Nikolaidis, Jiaoyang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2fbf1c9bc92e7ef2f6cab2e8a3e09af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2fbf1c9bc92e7ef2f6cab2e8a3e09af-Abstract-Conference.html)

**Abstract**:

We study the problem of generating arbitrarily large environments to improve the throughput of multi-robot systems. Prior work proposes Quality Diversity (QD) algorithms as an effective method for optimizing the environments of automated warehouses. However, these approaches optimize only relatively small environments, falling short when it comes to replicating real-world warehouse sizes. The challenge arises from the exponential increase in the search space as the environment size increases. Additionally, the previous methods have only been tested with up to 350 robots in simulations, while practical warehouses could host thousands of robots. In this paper, instead of optimizing environments, we propose to optimize Neural Cellular Automata (NCA) environment generators via QD algorithms. We train a collection of NCA generators with QD algorithms in small environments and then generate arbitrarily large environments from the generators at test time. We show that NCA environment generators maintain consistent, regularized patterns regardless of environment size, significantly enhancing the scalability of multi-robot systems in two different domains with up to 2,350 robots. Additionally, we demonstrate that our method scales a single-agent reinforcement learning policy to arbitrarily large environments with similar patterns. We include the source code at https://github.com/lunjohnzhang/warehouseenvgenncapublic.

----

## [0] FAMO: Fast Adaptive Multitask Optimization

**Authors**: *Bo Liu, Yihao Feng, Peter Stone, Qiang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html)

**Abstract**:

One of the grand enduring goals of AI is to create generalist agents that can learn multiple different tasks from diverse data via multitask learning (MTL). However, in practice, applying gradient descent (GD) on the average loss across all tasks may yield poor multitask performance due to severe under-optimization of certain tasks. Previous approaches that manipulate task gradients for a more balanced loss decrease require storing and computing all task gradients ($\mathcal{O}(k)$ space and time where $k$ is the number of tasks), limiting their use in large-scale scenarios. In this work, we introduce Fast Adaptive Multitask Optimization (FAMO), a dynamic weighting method that decreases task losses in a balanced way using $\mathcal{O}(1)$ space and time. We conduct an extensive set of experiments covering multi-task supervised and reinforcement learning problems. Our results indicate that FAMO achieves comparable or superior performance to state-of-the-art gradient manipulation techniques while offering significant improvements in space and computational efficiency. Code is available at \url{https://github.com/Cranial-XIX/FAMO}.

----



[Go to the previous page](NIPS-2023-list2400.md)

[Go to the next page](NIPS-2023-list2402.md)

[Go to the catalog section](README.md)