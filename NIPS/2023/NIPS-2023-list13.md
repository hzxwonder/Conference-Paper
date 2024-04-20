## [2400] Bi-Level Offline Policy Optimization with Limited Exploration

        **Authors**: *Wenzhuo Zhou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac6de776b8de8c9aed1d356997eb54b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac6de776b8de8c9aed1d356997eb54b8-Abstract-Conference.html)

        **Abstract**:

        We study offline reinforcement learning (RL) which seeks to learn a good policy based on a fixed, pre-collected dataset. A fundamental challenge behind this task is the distributional shift due to the dataset lacking sufficient exploration, especially under function approximation. To tackle this issue, we propose a bi-level structured policy optimization algorithm that models a hierarchical interaction between the policy (upper-level) and the value function (lower-level). The lower level focuses on constructing a confidence set of value estimates that maintain sufficiently small weighted average Bellman errors, while controlling uncertainty arising from distribution mismatch. Subsequently, at the upper level, the policy aims to maximize a conservative value estimate from the confidence set formed at the lower level. This novel formulation preserves the maximum flexibility of the implicitly induced exploratory data distribution, enabling the power of model extrapolation. In practice, it can be solved through a computationally efficient, penalized adversarial estimation procedure. Our theoretical regret guarantees do not rely on any data-coverage and completeness-type assumptions, only requiring realizability. These guarantees also demonstrate that the learned policy represents the ``best effort'' among all policies, as no other policies can outperform it. We evaluate our model using a blend of synthetic, benchmark, and real-world datasets for offline RL, showing that it performs competitively with state-of-the-art methods.

        ----

        ## [2401] Generating QM1B with PySCFIPU

        **Authors**: *Alexander Mathiasen, Hatem Helal, Kerstin Klaser, Paul Balanca, Josef Dean, Carlo Luschi, Dominique Beaini, Andrew W. Fitzgibbon, Dominic Masters*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ac7f98dd0b342edaf3be79844a180a6b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ac7f98dd0b342edaf3be79844a180a6b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The emergence of foundation models in Computer Vision and Natural Language Processing have resulted in immense progress on downstream tasks.  This progress was enabled by datasets with billions of training examples. Similar benefits are yet to be unlocked for quantum chemistry, where the potential of deep learning is constrained by comparatively small datasets with 100k to 20M training examples. These datasets are limited in size because the labels are computed using the accurate (but computationally demanding) predictions of Density Functional Theory (DFT). Notably, prior DFT datasets were created using CPU supercomputers without leveraging hardware acceleration.  In this paper, we take a first step towards utilising hardware accelerators by introducing the data generator PySCF$_{\text{IPU}}$ using Intelligence Processing Units (IPUs). This allows us to create the dataset QM1B with one billion training examples containing 9-11 heavy atoms. We demonstrate that a simple baseline neural network (SchNet 9M) improves its performance by simply increasing the amount of training data without additional inductive biases. To encourage future researchers to use QM1B responsibly, we highlight several limitations of QM1B and emphasise the low resolution of our DFT options, which also serves as motivation for even larger, more accurate datasets.

        ----

        ## [2402] Unified Enhancement of Privacy Bounds for Mixture Mechanisms via f-Differential Privacy

        **Authors**: *Chendi Wang, Buxin Su, Jiayuan Ye, Reza Shokri, Weijie J. Su*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acb3e20075b0a2dfa3565f06681578e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acb3e20075b0a2dfa3565f06681578e5-Abstract-Conference.html)

        **Abstract**:

        Differentially private (DP) machine learning algorithms incur many sources of randomness, such as random initialization, random batch subsampling, and shuffling. However, such randomness is difficult to take into account when proving differential privacy bounds because it induces mixture distributions for the algorithm's output that are difficult to analyze. This paper focuses on improving privacy bounds for shuffling models and one-iteration  differentially private gradient descent (DP-GD) with random initializations using $f$-DP. We derive a closed-form expression of the trade-off function for shuffling models that outperforms the most up-to-date results based on $(\epsilon,\delta)$-DP.Moreover, we investigate the effects of random initialization on the privacy of one-iteration DP-GD. Our numerical computations of the trade-off function indicate that random initialization can enhance the privacy of DP-GD.Our analysis of $f$-DP guarantees for these mixture mechanisms relies on an inequality for trade-off functions introduced in this paper. This inequality implies the joint convexity of $F$-divergences. Finally, we study an $f$-DP analog of the advanced joint convexity of the hockey-stick divergence related to $(\epsilon,\delta)$-DP  and apply it to analyze the privacy of mixture mechanisms.

        ----

        ## [2403] On the Role of Entanglement and Statistics in Learning

        **Authors**: *Srinivasan Arunachalam, Vojtech Havlícek, Louis Schatzki*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acb7ce5aab6e134300a2361dd90a501f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acb7ce5aab6e134300a2361dd90a501f-Abstract-Conference.html)

        **Abstract**:

        In this work we make progress in understanding the relationship between learning models when given access to entangled measurements, separable measurements and statistical measurements in the quantum statistical query ($\mathsf{QSQ}$) model. To this end, we show the following results.$\textbf{Entanglement versus separable measurements.}$ The goal here is to learn an unknown $f$ from the concept class $\mathcal{C} \subseteq \{f:\{0,1\}^n\rightarrow [k]\}$ given copies of  $\frac{1}{\sqrt{2^n}}\sum_x \ket{x,f(x)}$.  We show that, if $T$ copies suffice to learn $f$ using entangled measurements, then $O(nT^2)$ copies suffice to learn $f$ using just separable measurements. Additionally, we exhibit a concept class $\mathcal{C}$ for which, in order to learn some \emph{property} of $f$, the sample complexity of learning using entangled measurements is exponentially smaller than separable measurements.$\textbf{Entangled versus statistical measurements}$ The goal here is to learn a function $f \in \mathcal{C}$ given access to separable measurements and statistical measurements.   We exhibit a concept class $\mathcal{C}$ based on degree-$2$ functions that gives an exponential separation between $\mathsf{QSQ}$ learning and quantum learning with entangled measurements (even in the presence of noise). This proves the "quantum analogue" of the seminal result of (Blum, 2003) that separates classical $\mathsf{SQ}$ learning from classical $\mathsf{PAC}$ learning with classification~noise.$\textbf{$\mathsf{QSQ}$ lower bounds for learning states.}$ The main technical contribution is to introduce a quantum statistical query dimension ($\mathsf{QSDA}$), which we use to give lower bounds on the $\mathsf{QSQ}$ complexity of learning. Using this, we prove exponential $\mathsf{QSQ}$ lower bounds for testing purity of quantum states, learning CCHL states, coset states of Abelian groups, degree-$2$ functions, planted bi-clique states and learning output states of Clifford circuits of depth polylog($n$).$\textbf{Further applications.}$ Using our $\mathsf{QSQ}$ lower bounds give an $\textit{unconditional}$ separation between weak and strong error mitigation and prove lower bounds for learning distributions in the $\mathsf{QSQ}$ model. Prior works by (Quek et al., 2022), (Hinsche et al., 2022), and (Neitner et al., 23) proved the analogous results $\textit{assuming}$ diagonal measurements and our work removes this assumption.

        ----

        ## [2404] Molecule Joint Auto-Encoding: Trajectory Pretraining with 2D and 3D Diffusion

        **Authors**: *Weitao Du, Jiujiu Chen, Xuecang Zhang, Zhi-Ming Ma, Shengchao Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html)

        **Abstract**:

        Recently, artificial intelligence for drug discovery has raised increasing interest in both machine learning and chemistry domains. The fundamental building block for drug discovery is molecule geometry and thus, the molecule's geometrical representation is the main bottleneck to better utilize machine learning techniques for drug discovery. In this work, we propose a pretraining method for molecule joint auto-encoding (MoleculeJAE). MoleculeJAE can learn both the 2D bond (topology) and 3D conformation (geometry) information, and a diffusion process model is applied to mimic the augmented trajectories of such two modalities, based on which, MoleculeJAE will learn the inherent chemical structure in a self-supervised manner. Thus, the pretrained geometrical representation in MoleculeJAE is expected to benefit downstream geometry-related tasks. Empirically, MoleculeJAE proves its effectiveness by reaching state-of-the-art performance on 15 out of 20 tasks by comparing it with 12 competitive baselines.

        ----

        ## [2405] Federated Learning with Manifold Regularization and Normalized Update Reaggregation

        **Authors**: *Xuming An, Li Shen, Han Hu, Yong Luo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acf2b98eeb09b21968c2de6b1c6952e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acf2b98eeb09b21968c2de6b1c6952e9-Abstract-Conference.html)

        **Abstract**:

        Federated Learning (FL) is an emerging collaborative machine learning framework where multiple clients train the global model without sharing their own datasets. In FL, the model inconsistency caused by the local data heterogeneity across clients results in the near-orthogonality of client updates, which leads to the global update norm reduction and slows down the convergence.  Most previous works focus on eliminating the difference of parameters (or gradients) between the local and global models, which may fail to reflect the model inconsistency due to the complex structure of the machine learning model and the Euclidean space's limitation in meaningful geometric representations.In this paper, we propose FedMRUR by adopting the manifold model fusion scheme and a new global optimizer to alleviate the negative impacts.Concretely, FedMRUR adopts a hyperbolic graph manifold regularizer enforcing the representations of the data in the local and global models are close to each other in a low-dimensional subspace. Because the machine learning model has the graph structure, the distance in hyperbolic space can reflect the model bias better than the Euclidean distance.In this way, FedMRUR exploits the manifold structures of the representations to significantly reduce the model inconsistency.FedMRUR also aggregates the client updates norms as the global update norm, which can appropriately enlarge each client's contribution to the global update, thereby mitigating the norm reduction introduced by the near-orthogonality of client updates.Furthermore, we theoretically prove that our algorithm can achieve a linear speedup property $\mathcal{O}(\frac{1}{\sqrt{SKT}})$ for non-convex setting under partial client participation, where $S$ is the participated clients number, $K$ is the local interval and $T$ is the total number of communication rounds.Experiments demonstrate that FedMRUR can achieve a new state-of-the-art (SOTA) accuracy with less communication.

        ----

        ## [2406] Long-Term Fairness with Unknown Dynamics

        **Authors**: *Tongxin Yin, Reilly Raab, Mingyan Liu, Yang Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acf4a08f67724e9d2de34099f57a9c25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/acf4a08f67724e9d2de34099f57a9c25-Abstract-Conference.html)

        **Abstract**:

        While machine learning can myopically reinforce social inequalities, it may also be used to dynamically seek equitable outcomes. In this paper, we formalize long-term fairness as an online reinforcement learning problem for a policy affecting human populations. This formulation accommodates dynamical control objectives, such as achieving equitable population states, that cannot be incorporated into static formulations of fairness. We demonstrate that algorithmic solutions to the proposed fairness problem can adapt to unknown dynamics and, by sacrificing short-term incentives, drive the policy-population system towards more desirable equilibria. For the proposed setting, we develop an algorithm that adapts recent work in online learning and prove that this algorithm achieves simultaneous probabilistic bounds on cumulative loss and cumulative violations of fairness. In the classification setting subject to group fairness, we compare our proposed algorithm to several baselines, including the repeated retraining of myopic or distributionally robust classifiers, and to a deep reinforcement learning algorithm that lacks fairness guarantees. Our experiments model human populations according to evolutionary game theory and integrate real-world datasets.

        ----

        ## [2407] YouTubePD: A Multimodal Benchmark for Parkinson's Disease Analysis

        **Authors**: *Andy Zhou, Samuel Li, Pranav Sriram, Xiang Li, Jiahua Dong, Ansh Sharma, Yuanyi Zhong, Shirui Luo, Volodymyr V. Kindratenko, George Heintz, Christopher Zallek, Yu-Xiong Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/acffd5024f52c3a9ecc8ccb4b75b4e5c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/acffd5024f52c3a9ecc8ccb4b75b4e5c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The healthcare and AI communities have witnessed a growing interest in the development of AI-assisted systems for automated diagnosis of Parkinson's Disease (PD), one of the most prevalent neurodegenerative disorders. However, the progress in this area has been significantly impeded by the absence of a unified, publicly available benchmark, which prevents comprehensive evaluation of existing PD analysis methods and the development of advanced models. This work overcomes these challenges by introducing YouTubePD -- the first publicly available multimodal benchmark designed for PD analysis. We crowd-source existing videos featured with PD from YouTube, exploit multimodal information including in-the-wild videos, audio data, and facial landmarks across 200+ subject videos, and provide dense and diverse annotations from clinical expert. Based on our benchmark, we propose three challenging and complementary tasks encompassing both discriminative and generative tasks, along with a comprehensive set of corresponding baselines. Experimental evaluation showcases the potential of modern deep learning and computer vision techniques, in particular the generalizability of the models developed on YouTubePD to real-world clinical settings, while revealing their limitations. We hope our work paves the way for future research in this direction.

        ----

        ## [2408] Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training

        **Authors**: *Aleksandra Nowak, Bram Grooten, Decebal Constantin Mocanu, Jacek Tabor*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad02c6f3824f871395112ae71a28eff7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad02c6f3824f871395112ae71a28eff7-Abstract-Conference.html)

        **Abstract**:

        Dynamic Sparse Training (DST) is a rapidly evolving area of research that seeks to optimize the sparse initialization of a neural network by adapting its topology during training.  It has been shown that under specific conditions, DST is able to outperform dense models. The key components of this framework are the pruning and growing criteria, which are repeatedly applied during the training process to adjust the networkâ€™s sparse connectivity. While the growing criterion's impact on DST performance is relatively well studied, the influence of the pruning criterion remains overlooked. To address this issue, we design and perform an extensive empirical analysis of various pruning criteria to better understand their impact on the dynamics of DST solutions. Surprisingly, we find that most of the studied methods yield similar results. The differences become more significant in the low-density regime, where the best performance is predominantly given by the simplest technique: magnitude-based pruning.

        ----

        ## [2409] Tree-Based Diffusion Schrödinger Bridge with Applications to Wasserstein Barycenters

        **Authors**: *Maxence Noble, Valentin De Bortoli, Arnaud Doucet, Alain Durmus*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad08767706825033b99122332293033d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad08767706825033b99122332293033d-Abstract-Conference.html)

        **Abstract**:

        Multi-marginal Optimal Transport (mOT), a generalization of OT, aims at minimizing the integral of a cost function with respect to a distribution with some prescribed marginals. In this paper, we consider an entropic version of mOT  with a tree-structured quadratic cost, i.e., a function that can be written as  a sum of pairwise cost functions between the nodes of a tree. To address this  problem, we develop Tree-based Diffusion Schr\"odinger Bridge (TreeDSB), an  extension of the Diffusion Schr\"odinger Bridge (DSB) algorithm. TreeDSB  corresponds to a dynamic and continuous state-space counterpart of the  multimarginal Sinkhorn algorithm. A notable use case of our methodology is to  compute Wasserstein barycenters which can be recast as the solution of a mOT  problem on a star-shaped tree. We demonstrate that our methodology can be applied in high-dimensional settings such as image interpolation and  Bayesian fusion.

        ----

        ## [2410] Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders

        **Authors**: *Jan Dubinski, Stanislaw Pawlak, Franziska Boenisch, Tomasz Trzcinski, Adam Dziedzic*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad1efab57a04d93f097e7fbb2d4fc054-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad1efab57a04d93f097e7fbb2d4fc054-Abstract-Conference.html)

        **Abstract**:

        Machine Learning as a Service (MLaaS) APIs provide ready-to-use and high-utility encoders that generate vector representations for given inputs. Since these encoders are very costly to train, they become lucrative targets for model stealing attacks during which an adversary leverages query access to the API to replicate the encoder locally at a fraction of the original training costs. We propose Bucks for Buckets (B4B), the first active defense that prevents stealing while the attack is happening without degrading representation quality for legitimate API users. Our defense relies on the observation that the representations returned to adversaries who try to steal the encoder's functionality cover a significantly larger fraction of the embedding space than representations of legitimate users who utilize the encoder to solve a particular downstream task. B4B leverages this to adaptively adjust the utility of the returned representations according to a user's coverage of the embedding space. To prevent adaptive adversaries from eluding our defense by simply creating multiple user accounts (sybils), B4B also individually transforms each user's representations. This prevents the adversary from directly aggregating representations over multiple accounts to create their stolen encoder copy. Our active defense opens a new path towards securely sharing and democratizing encoders over public APIs.

        ----

        ## [2411] A General Framework for Equivariant Neural Networks on Reductive Lie Groups

        **Authors**: *Ilyes Batatia, Mario Geiger, Jose M. Munoz, Tess E. Smidt, Lior Silberman, Christoph Ortner*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad1f2197941348b1c4373fd6c19ee0b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad1f2197941348b1c4373fd6c19ee0b4-Abstract-Conference.html)

        **Abstract**:

        Reductive Lie Groups, such as the orthogonal groups, the Lorentz group, or the unitary groups, play essential roles across scientific fields as diverse as high energy physics, quantum mechanics, quantum chromodynamics, molecular dynamics, computer vision, and imaging. In this paper, we present a general Equivariant Neural Network architecture capable of respecting the symmetries of the finite-dimensional representations of any reductive Lie Group. Our approach generalizes the successful ACE and MACE architectures for atomistic point clouds to any data equivariant to a reductive Lie group action. We also introduce the lie-nn software library, which provides all the necessary tools to develop and implement such general G-equivariant neural networks. It implements routines for the reduction of generic tensor products of representations into irreducible representations, making it easy to apply our architecture to a wide range of problems and groups. The generality and performance of our approach are demonstrated by applying it to the tasks of top quark decay tagging (Lorentz group) and shape recognition (orthogonal group).

        ----

        ## [2412] Context-PIPs: Persistent Independent Particles Demands Context Features

        **Authors**: *Weikang Bian, Zhaoyang Huang, Xiaoyu Shi, Yitong Dong, Yijin Li, Hongsheng Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad2fa437f7c23e4e9875599c6065d18a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad2fa437f7c23e4e9875599c6065d18a-Abstract-Conference.html)

        **Abstract**:

        We tackle the problem of Persistent Independent Particles (PIPs), also called Tracking Any Point (TAP), in videos, which specifically aims at estimating persistent long-term trajectories of query points in videos. Previous methods attempted to estimate these trajectories independently to incorporate longer image sequences, therefore, ignoring the potential benefits of incorporating spatial context features. We argue that independent video point tracking also demands spatial context features. To this end, we propose a novel framework Context-PIPs, which effectively improves point trajectory accuracy by aggregating spatial context features in videos. Context-PIPs contains two main modules: 1) a SOurse Feature Enhancement (SOFE) module, and 2) a TArget Feature Aggregation (TAFA) module. Context-PIPs significantly improves PIPs all-sided, reducing 11.4\% Average Trajectory Error of Occluded Points (ATE-Occ) on CroHD and increasing 11.8\% Average Percentage of Correct Keypoint (A-PCK) on TAP-Vid-Kinetics. Demos are available at \url{https://wkbian.github.io/Projects/Context-PIPs/}.

        ----

        ## [2413] GloptiNets: Scalable Non-Convex Optimization with Certificates

        **Authors**: *Gaspard Beugnot, Julien Mairal, Alessandro Rudi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad350eaaa93c8c3ab762cdc119d12889-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad350eaaa93c8c3ab762cdc119d12889-Abstract-Conference.html)

        **Abstract**:

        We present a novel approach to non-convex optimization with certificates, which handles smooth functions on the hypercube or on the torus. Unlike traditional methods that rely on algebraic properties, our algorithm exploits the regularity of the target function intrinsic in the decay of its Fourier spectrum. By defining a tractable family of models, we allow {\em at the same time} to obtain precise certificates and to leverage the advanced and powerful computational techniques developed to optimize neural networks. In this way the scalability of our approach is naturally enhanced by parallel computing with GPUs. Our approach, when applied to the case of polynomials of moderate dimensions but with thousands of coefficients, outperforms the state-of-the-art optimization methods with certificates, as the ones based on Lasserre's hierarchy, addressing problems intractable for the competitors.

        ----

        ## [2414] Ethical Considerations for Responsible Data Curation

        **Authors**: *Jerone Theodore Alexander Andrews, Dora Zhao, William Thong, Apostolos Modas, Orestis Papakyriakopoulos, Alice Xiang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad3ebc951f43d1e9ed20187a7b5bc4ee-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad3ebc951f43d1e9ed20187a7b5bc4ee-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Human-centric computer vision (HCCV) data curation practices often neglect privacy and bias concerns, leading to dataset retractions and unfair models. HCCV datasets constructed through nonconsensual web scraping lack crucial metadata for comprehensive fairness and robustness evaluations. Current remedies are post hoc, lack persuasive justification for adoption, or fail to provide proper contextualization for appropriate application. Our research focuses on proactive, domain-specific recommendations, covering purpose, privacy and consent, and diversity, for curating HCCV evaluation datasets, addressing privacy and bias concerns. We adopt an ante hoc reflective perspective, drawing from current practices, guidelines, dataset withdrawals, and audits, to inform our considerations and recommendations.

        ----

        ## [2415] Meta-Adapter: An Online Few-shot Learner for Vision-Language Model

        **Authors**: *Cheng Cheng, Lin Song, Ruoyi Xue, Hang Wang, Hongbin Sun, Yixiao Ge, Ying Shan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad48f017e6c3d474caf511208e600459-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad48f017e6c3d474caf511208e600459-Abstract-Conference.html)

        **Abstract**:

        The contrastive vision-language pre-training, known as CLIP, demonstrates remarkable potential in perceiving open-world visual concepts, enabling effective zero-shot image recognition.  Nevertheless, few-shot learning methods based on CLIP typically require offline fine-tuning of the parameters on few-shot samples, resulting in longer inference time and the risk of overfitting in certain domains.  To tackle these challenges, we propose the Meta-Adapter, a lightweight residual-style adapter, to refine the CLIP features guided by the few-shot samples in an online manner.  With a few training samples, our method can enable effective few-shot learning capabilities and generalize to unseen data or tasks without additional fine-tuning, achieving competitive performance and high efficiency.  Without bells and whistles, our approach outperforms the state-of-the-art online few-shot learning method by an average of 3.6\% on eight image classification datasets with higher inference speed.  Furthermore, our model is simple and flexible, serving as a plug-and-play module directly applicable to downstream tasks.  Without further fine-tuning, Meta-Adapter obtains notable performance improvements in open-vocabulary object detection and segmentation tasks.

        ----

        ## [2416] Taming Local Effects in Graph-based Spatiotemporal Forecasting

        **Authors**: *Andrea Cini, Ivan Marisca, Daniele Zambon, Cesare Alippi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad58c61c71efd5436134a3ecc87da6ea-Abstract-Conference.html)

        **Abstract**:

        Spatiotemporal graph neural networks have shown to be effective in time series forecasting applications, achieving better performance than standard univariate predictors in several settings. These architectures take advantage of a graph structure and relational inductive biases to learn a single (global) inductive model to predict any number of the input time series, each associated with a graph node. Despite the gain achieved in computational and data efficiency w.r.t. fitting a set of local models, relying on a single global model can be a limitation whenever some of the time series are generated by a different spatiotemporal stochastic process. The main objective of this paper is to understand the interplay between globality and locality in graph-based spatiotemporal forecasting, while contextually proposing a methodological framework to rationalize the practice of including trainable node embeddings in such architectures. We ascribe to trainable node embeddings the role of amortizing the learning of specialized components. Moreover, embeddings allow for 1) effectively combining the advantages of shared message-passing layers with node-specific parameters and 2) efficiently transferring the learned model to new node sets. Supported by strong empirical evidence, we provide insights and guidelines for specializing graph-based models to the dynamics of each time series and show how this aspect plays a crucial role in obtaining accurate predictions.

        ----

        ## [2417] Latent Space Translation via Semantic Alignment

        **Authors**: *Valentino Maiorca, Luca Moschella, Antonio Norelli, Marco Fumero, Francesco Locatello, Emanuele Rodolà*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract-Conference.html)

        **Abstract**:

        While different neural models often exhibit latent spaces that are alike when exposed to semantically related data, this intrinsic similarity is not always immediately discernible. Towards a better understanding of this phenomenon, our work shows how representations learned from these neural modules can be translated between different pre-trained networks via simpler transformations than previously thought. An advantage of this approach is the ability to estimate these transformations using standard, well-understood algebraic procedures that have closed-form solutions. Our method directly estimates a transformation between two given latent spaces, thereby enabling effective stitching of encoders and decoders without additional training. We extensively validate the adaptability of this translation procedure in different experimental settings: across various trainings, domains, architectures (e.g., ResNet, CNN, ViT), and in multiple downstream tasks (classification, reconstruction). Notably, we show how it is possible to zero-shot stitch text encoders and vision decoders, or vice-versa, yielding surprisingly good classification performance in this multimodal setting.

        ----

        ## [2418] A case for reframing automated medical image classification as segmentation

        **Authors**: *Sarah M. Hooper, Mayee F. Chen, Khaled Saab, Kush Bhatia, Curtis P. Langlotz, Christopher Ré*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad6a3bd12095fdca71c306871bdec400-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad6a3bd12095fdca71c306871bdec400-Abstract-Conference.html)

        **Abstract**:

        Image classification and segmentation are common applications of deep learning to radiology. While many tasks can be framed using either classification or segmentation, classification has historically been cheaper to label and more widely used. However, recent work has drastically reduced the cost of training segmentation networks. In light of this recent work, we reexamine the choice of training classification vs. segmentation models. First, we use an information theoretic approach to analyze why segmentation vs. classification models may achieve different performance on the same dataset and overarching task. We then implement multiple methods for using segmentation models to classify medical images, which we call segmentation-for-classification, and compare these methods against traditional classification on three retrospective datasets. We use our analysis and experiments to summarize the benefits of switching from segmentation to classification, including: improved sample efficiency, enabling improved performance with fewer labeled images (up to an order of magnitude lower), on low-prevalence classes, and on certain rare subgroups (up to 161.1\% improved recall); improved robustness to spurious correlations (up to 44.8\% improved robust AUROC); and improved model interpretability, evaluation, and error analysis.

        ----

        ## [2419] Efficient Policy Adaptation with Contrastive Prompt Ensemble for Embodied Agents

        **Authors**: *Wonje Choi, Woo Kyung Kim, Seunghyun Kim, Honguk Woo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad72633e034990a97e878fc2fc100afb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad72633e034990a97e878fc2fc100afb-Abstract-Conference.html)

        **Abstract**:

        For embodied reinforcement learning (RL) agents interacting with the environment, it is desirable to have rapid policy adaptation to unseen visual observations, but achieving zero-shot adaptation capability is considered as a challenging problem in the RL context. To address the problem, we present a novel contrastive prompt ensemble (ConPE) framework which utilizes a pretrained vision-language model and a set of visual prompts, thus enables efficient policy learning and adaptation upon a wide range of environmental and physical changes encountered by embodied agents. Specifically, we devise a guided-attention-based ensemble approach with multiple visual prompts on the vision-language model to construct robust state representations. Each prompt is contrastively learned in terms of an individual domain factors that significantly affects the agent's egocentric perception and observation. For a given task, the attention-based ensemble and policy are jointly learned so that the resulting state representations not only generalize to various domains but are also optimized for learning the task. Through experiments, we show that ConPE outperforms other state-of-the-art algorithms for several embodied agent tasks including navigation in AI2THOR, manipulation in Metaworld, and autonomous driving in CARLA, while also improving the sample efficiency of policy learning and adaptation.

        ----

        ## [2420] Improvements on Uncertainty Quantification for Node Classification via Distance Based Regularization

        **Authors**: *Russell Hart, Linlin Yu, Yifei Lou, Feng Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ad84864002a72c344c2227d7eb8842b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ad84864002a72c344c2227d7eb8842b1-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks have achieved significant success in the last decades, but they are not well-calibrated and often produce unreliable predictions. A large number of literature relies on uncertainty quantification to evaluate the reliability of a learning model, which is particularly important for applications of out-of-distribution (OOD) detection and misclassification detection. We are interested in uncertainty quantification for interdependent node-level classification. We start our analysis based on graph posterior networks (GPNs) that optimize the uncertainty cross-entropy (UCE)-based loss function. We describe the theoretical limitations of the widely-used UCE loss. To alleviate the identified drawbacks, we propose a distance-based regularization that encourages clustered OOD nodes to remain clustered in the latent space. We conduct extensive comparison experiments on eight standard datasets and demonstrate that the proposed regularization outperforms the state-of-the-art in both OOD detection and misclassification detection.

        ----

        ## [2421] Efficient Learning of Linear Graph Neural Networks via Node Subsampling

        **Authors**: *Seiyun Shin, Ilan Shomorony, Han Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ada418ae9b6677dcda32d9dca0f7441f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ada418ae9b6677dcda32d9dca0f7441f-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) are a powerful class of machine learning models with applications in recommender systems, drug discovery, social network analysis, and computer vision. One challenge with their implementation is that GNNs often take large-scale graphs as inputs, which imposes significant computational/storage costs in the training and testing phases. In particular, the message passing operations of a GNN require multiplication of the graph adjacency matrix $A \in \mathbb{R}^{n \times n}$ and the data matrix $X \in \mathbb{R}^{n \times d}$, and the $O(n^2 d)$ time complexity can be prohibitive for large $n$. Thus, a natural question is whether it is possible to perform the GNN operations in (quasi-)linear time by avoiding the full computation of $A X$. To study this question, we consider the setting of a regression task on a two-layer Linear Graph Convolutional Network (GCN). We develop an efficient training algorithm based on (1) performing node subsampling, (2) estimating the leverage scores of $A X$ based on the subsampled graph, and (3) performing leverage score sampling on $A X$. We show that our proposed scheme learns the regression model observing only $O(nd\epsilon^{-2}\log n)$ entries of $A$ in time $O(nd^2 \epsilon^{-2}\log n)$, with the guarantee that the learned weights deviate by at most $\epsilon$ under the $\ell_2$ norm from the model learned using the entire adjacency matrix $A$. We present empirical results for regression problems on real-world graphs and show that our algorithm significantly outperforms other baseline sampling strategies that exploit the same number of observations.

        ----

        ## [2422] DPM-Solver-v3: Improved Diffusion ODE Solver with Empirical Model Statistics

        **Authors**: *Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ada8de994b46571bdcd7eeff2d3f9cff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ada8de994b46571bdcd7eeff2d3f9cff-Abstract-Conference.html)

        **Abstract**:

        Diffusion probabilistic models (DPMs) have exhibited excellent performance for high-fidelity image generation while suffering from inefficient sampling. Recent works accelerate the sampling procedure by proposing fast ODE solvers that leverage the specific ODE form of DPMs. However, they highly rely on specific parameterization during inference (such as noise/data prediction), which might not be the optimal choice. In this work, we propose a novel formulation towards the optimal parameterization during sampling that minimizes the first-order discretization error of the ODE solution. Based on such formulation, we propose \textit{DPM-Solver-v3}, a new fast ODE solver for DPMs by introducing several coefficients efficiently computed on the pretrained model, which we call \textit{empirical model statistics}. We further incorporate multistep methods and a predictor-corrector framework, and propose some techniques for improving sample quality at small numbers of function evaluations (NFE) or large guidance scales. Experiments show that DPM-Solver-v3 achieves consistently better or comparable performance in both unconditional and conditional sampling with both pixel-space and latent-space DPMs, especially in 5$\sim$10 NFEs. We achieve FIDs of 12.21 (5 NFE), 2.51 (10 NFE) on unconditional CIFAR10, and MSE of 0.55 (5 NFE, 7.5 guidance scale) on Stable Diffusion, bringing a speed-up of 15\%$\sim$30\% compared to previous state-of-the-art training-free methods. Code is available at \url{https://github.com/thu-ml/DPM-Solver-v3}.

        ----

        ## [2423] Active Bipartite Ranking

        **Authors**: *James Cheshire, Vincent Laurent, Stéphan Clémençon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/adb77ecc8ba1c2d3135c86a46b8f2496-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/adb77ecc8ba1c2d3135c86a46b8f2496-Abstract-Conference.html)

        **Abstract**:

        In this paper, we develop an active learning framework for the bipartite ranking problem.Motivated by numerous applications, ranging from supervised anomaly detection to credit-scoring through the design of medical diagnosis support systems, and usually formulated as the problem of optimizing (a scalar summary of) the ROC curve, bipartite ranking has been the subject of much attention in the passive context. Various dedicated algorithms have been recently proposed and studied by the machine-learning community. In contrast, active bipartite ranking rule is poorly documented in the literature. Due to its global nature, a strategy for labeling sequentially data points that are difficult to rank w.r.t. to the others is required. This learning task is much more complex than binary classification, for which many active algorithms have been designed. It is the goal of this article to provide a rigorous formulation of such a selective sampling approach. We propose a dedicated algorithm, referred to as active-rank, which aims to minimise the distance between the ROC curve of the ranking function built and the optimal one, w.r.t. the sup norm. We show that, for a fixed confidence level $\epsilon$ and probability $\delta$,  active-rank is PAC$(\epsilon,\delta)$. In addition, we provide a problem dependent upper bound on the expected sampling time of  active-rank and also demonstrate a problem dependent lower bound on the expected sampling time of any PAC$(\epsilon,\delta)$ algorithm. Beyond the theoretical analysis carried out, numerical results are presented, providing strong empirical evidence of the performance of the algorithm proposed, which compares favorably with more naive approaches.

        ----

        ## [2424] Are Emergent Abilities of Large Language Models a Mirage?

        **Authors**: *Rylan Schaeffer, Brando Miranda, Sanmi Koyejo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/adc98a266f45005c403b8311ca7e8bd7-Abstract-Conference.html)

        **Abstract**:

        Recent work claims that large language models display \textit{emergent abilities}, abilities not present in smaller-scale models that are present in larger-scale models.What makes emergent abilities intriguing is two-fold: their \textit{sharpness}, transitioning seemingly instantaneously from not present to present, and their \textit{unpredictability}, appearing at seemingly unforeseeable model scales.Here, we present an alternative explanation for emergent abilities: that for a particular task and model family, when analyzing fixed model outputs, emergent abilities appear due the researcherâ€™s choice of metric rather than due to fundamental changes in model behavior with scale. Specifically, nonlinear or discontinuous metrics produce apparent emergent abilities, whereas linear or continuous metrics produce smooth, continuous, predictable changes in model performance.We present our alternative explanation in a simple mathematical model, then test it in three complementary ways: we (1) make, test and confirm three predictions on the effect of metric choice using the InstructGPT/GPT-3 family on tasks with claimed emergent abilities, (2) make, test and confirm two predictions about metric choices in a meta-analysis of emergent abilities on BIG-Bench; and (3) show how to choose metrics to produce never-before-seen seemingly emergent abilities in multiple vision tasks across diverse deep networks.Via all three analyses, we provide evidence that alleged emergent abilities evaporate with different metrics or with better statistics, and may not be a fundamental property of scaling AI models.

        ----

        ## [2425] Reward-agnostic Fine-tuning: Provable Statistical Benefits of Hybrid Reinforcement Learning

        **Authors**: *Gen Li, Wenhao Zhan, Jason D. Lee, Yuejie Chi, Yuxin Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ade04fd4f26263f86b47ffb535c4cafb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ade04fd4f26263f86b47ffb535c4cafb-Abstract-Conference.html)

        **Abstract**:

        This paper studies tabular reinforcement learning (RL) in the hybrid setting, which assumes access to both an offline dataset and online interactions with the unknown environment. A central question boils down to how to efficiently utilize online data to strengthen and complement the offline dataset and enable effective policy fine-tuning. Leveraging recent advances in reward-agnostic exploration and offline RL, we design a three-stage hybrid RL algorithm that beats the best of both worlds --- pure offline RL and pure online RL --- in terms of sample complexities. The proposed algorithm does not require any reward information during data collection. Our theory is developed based on a new notion called single-policy partial concentrability, which captures the trade-off between distribution mismatch and miscoverage and guides the interplay between offline and online data.

        ----

        ## [2426] The Exact Sample Complexity Gain from Invariances for Kernel Regression

        **Authors**: *Behrooz Tahmasebi, Stefanie Jegelka*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/adf5a38a2e2e7606fbfc3eff72998afa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/adf5a38a2e2e7606fbfc3eff72998afa-Abstract-Conference.html)

        **Abstract**:

        In practice, encoding invariances into models improves sample complexity. In this work, we study this phenomenon from a theoretical perspective. In particular, we provide minimax optimal rates for kernel ridge regression on compact manifolds, with a target function that is invariant to a group action on the manifold. Our results hold for any smooth compact Lie group action, even groups of positive dimension. For a finite group, the gain effectively multiplies the number of samples by the group size. For groups of positive dimension, the gain is observed by a reduction in the manifold's dimension, in addition to a factor proportional to the volume of the quotient space. Our proof takes the viewpoint of differential geometry, in contrast to the more common strategy of using invariant polynomials. This new geometric viewpoint on learning with invariances may be of independent interest.

        ----

        ## [2427] FaceDNeRF: Semantics-Driven Face Reconstruction, Prompt Editing and Relighting with Diffusion Models

        **Authors**: *Hao Zhang, Tianyuan Dai, Yanbo Xu, Yu-Wing Tai, Chi-Keung Tang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae0cba715b60c4052359b3d52a2cff7f-Abstract-Conference.html)

        **Abstract**:

        The ability to create high-quality 3D faces from a single image has become increasingly important with wide applications in video conferencing, AR/VR, and advanced video editing in movie industries. In this paper, we propose Face Diffusion NeRF (FaceDNeRF), a new generative method to reconstruct high-quality Face NeRFs from single images, complete with semantic editing and relighting capabilities. FaceDNeRF utilizes high-resolution 3D GAN inversion and expertly trained 2D latent-diffusion model, allowing users to manipulate and construct Face NeRFs in zero-shot learning without the need for explicit 3D data. With carefully designed illumination and identity preserving loss, as well as multi-modal pre-training, FaceDNeRF offers users unparalleled control over the editing process enabling them to create and edit face NeRFs using just single-view images, text prompts, and explicit target lighting. The advanced features of FaceDNeRF have been designed to produce more impressive results than existing 2D editing approaches that rely on 2D segmentation maps for editable attributes. Experiments show that our FaceDNeRF achieves exceptionally realistic results and unprecedented flexibility in editing compared with state-of-the-art 3D face reconstruction and editing methods. Our code will be available at https://github.com/BillyXYB/FaceDNeRF.

        ----

        ## [2428] Chanakya: Learning Runtime Decisions for Adaptive Real-Time Perception

        **Authors**: *Anurag Ghosh, Vaibhav Balloli, Akshay Nambi, Aditya Singh, Tanuja Ganu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae2d574d2c309f3a45880e4460efd176-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae2d574d2c309f3a45880e4460efd176-Abstract-Conference.html)

        **Abstract**:

        Real-time perception requires planned resource utilization. Computational planning in real-time perception is governed by two considerations -- accuracy and latency. There exist run-time decisions (e.g. choice of input resolution) that induce tradeoffs affecting performance on a given hardware, arising from intrinsic (content, e.g. scene clutter) and extrinsic (system, e.g. resource contention) characteristics. Earlier runtime execution frameworks employed rule-based decision algorithms and operated with a fixed algorithm latency budget to balance these concerns, which is sub-optimal and inflexible. We propose Chanakya, a learned approximate execution framework that naturally derives from the streaming perception paradigm, to automatically learn decisions induced by these tradeoffs instead. Chanakya is trained via novel rewards balancing accuracy and latency implicitly, without approximating either objectives. Chanakya simultaneously considers intrinsic and extrinsic context, and predicts decisions in a flexible manner. Chanakya, designed with low overhead in mind, outperforms state-of-the-art static and dynamic execution policies on public datasets on both server GPUs and edge devices.

        ----

        ## [2429] RoboCLIP: One Demonstration is Enough to Learn Robot Policies

        **Authors**: *Sumedh Sontakke, Jesse Zhang, Sébastien M. R. Arnold, Karl Pertsch, Erdem Biyik, Dorsa Sadigh, Chelsea Finn, Laurent Itti*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae54ce310476218f26dd48c1626d5187-Abstract-Conference.html)

        **Abstract**:

        Reward specification is a notoriously difficult problem in reinforcement learning, requiring extensive expert supervision to design robust reward functions. Imitation learning (IL) methods attempt to circumvent these problems by utilizing expert demonstrations instead of using an extrinsic reward function but typically require a large number of in-domain expert demonstrations. Inspired by advances in the field of Video-and-Language Models (VLMs), we present RoboCLIP, an online imitation learning method that uses a single demonstration (overcoming the large data requirement) in the form of a video demonstration or a textual description of the task to generate rewards without manual reward function design. Additionally, RoboCLIP can also utilize out-of-domain demonstrations, like videos of humans solving the task for reward generation, circumventing the need to have the same demonstration and deployment domains. RoboCLIP utilizes pretrained VLMs without any finetuning for reward generation. Reinforcement learning agents trained with RoboCLIP rewards demonstrate 2-3 times higher zero-shot performance than competing imitation learning methods on downstream robot manipulation tasks, doing so using only one video/text demonstration. Visit our website at https://sites.google.com/view/roboclip/home for experiment videos.

        ----

        ## [2430] Contrast Everything: A Hierarchical Contrastive Framework for Medical Time-Series

        **Authors**: *Yihe Wang, Yu Han, Haishuai Wang, Xiang Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae7d9c77b5ff9e3b7833a68523b880f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae7d9c77b5ff9e3b7833a68523b880f2-Abstract-Conference.html)

        **Abstract**:

        Contrastive representation learning is crucial in medical time series analysis as it alleviates dependency on labor-intensive, domain-specific, and scarce expert annotations. However, existing contrastive learning methods primarily focus on one single data level, which fails to fully exploit the intricate nature of medical time series. To address this issue, we present COMET, an innovative hierarchical framework that leverages data consistencies at all inherent levels in medical time series. Our meticulously designed model systematically captures data consistency from four potential levels: observation, sample, trial, and patient levels. By developing contrastive loss at multiple levels, we can learn effective representations that preserve comprehensive data consistency, maximizing information utilization in a self-supervised manner. We conduct experiments in the challenging patient-independent setting. We compare COMET against six baselines using three diverse datasets, which include ECG signals for myocardial infarction and EEG signals for Alzheimer’s and Parkinson’s diseases. The results demonstrate that COMET consistently outperforms all baselines, particularly in setup with 10% and 1% labeled data fractions across all datasets. These results underscore the significant impact of our framework in advancing contrastive representation learning techniques for medical time series. The source code is available at https://github.com/DL4mHealth/COMET.

        ----

        ## [2431] Importance-aware Co-teaching for Offline Model-based Optimization

        **Authors**: *Ye Yuan, Can Chen, Zixuan Liu, Willie Neiswanger, Xue (Steve) Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html)

        **Abstract**:

        Offline model-based optimization aims to find a design that maximizes a property of interest using only an offline dataset, with applications in robot, protein, and molecule design, among others. A prevalent approach is gradient ascent, where a proxy model is trained on the offline dataset and then used to optimize the design. This method suffers from an out-of-distribution issue, where the proxy is not accurate for unseen designs. To mitigate this issue, we explore using a pseudo-labeler to generate valuable data for fine-tuning the proxy. Specifically, we propose $\textit{\textbf{I}mportance-aware \textbf{C}o-\textbf{T}eaching for Offline Model-based Optimization}~(\textbf{ICT})$. This method maintains three symmetric proxies with their mean ensemble as the final proxy, and comprises two steps. The first step is $\textit{pseudo-label-driven co-teaching}$. In this step, one proxy is iteratively selected as the pseudo-labeler for designs near the current optimization point, generating pseudo-labeled data.  Subsequently, a co-teaching process identifies small-loss samples as valuable data and exchanges them between the other two proxies for fine-tuning, promoting knowledge transfer.  This procedure is repeated three times, with a different proxy chosen as the pseudo-labeler each time, ultimately enhancing the ensemble performance.To further improve accuracy of pseudo-labels, we perform a secondary step of $\textit{meta-learning-based sample reweighting}$,which assigns importance weights to samples in the pseudo-labeled dataset and updates them via meta-learning. ICT achieves state-of-the-art results across multiple design-bench tasks, achieving the best mean rank $3.1$ and median rank $2$ among $15$ methods.Our source code can be accessed here.

        ----

        ## [2432] Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias

        **Authors**: *Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander J. Ratner, Ranjay Krishna, Jiaming Shen, Chao Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae9500c4f5607caf2eff033c67daa9d7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae9500c4f5607caf2eff033c67daa9d7-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Large language models (LLMs) have been recently leveraged as training data generators for various natural language processing (NLP) tasks. While previous research has explored different approaches to training models using generated data, they generally rely on simple class-conditional prompts, which may limit the diversity of the generated data and inherit systematic biases of LLM. Thus, we investigate training data generation with diversely attributed prompts (e.g., specifying attributes like length and style), which have the potential to yield diverse and attributed generated data. Our investigation focuses on datasets with high cardinality and diverse domains, wherein we demonstrate that attributed prompts outperform simple class-conditional prompts in terms of the resulting model's performance. Additionally, we present a comprehensive empirical study on data generation encompassing vital aspects like bias, diversity, and efficiency, and highlight three key observations: firstly, synthetic datasets generated by simple prompts exhibit significant biases, such as regional bias; secondly, attribute diversity plays a pivotal role in enhancing model performance; lastly, attributed prompts achieve the performance of simple class-conditional prompts while utilizing only 5\% of the querying cost of ChatGPT associated with the latter. The data and code are available on {\url{https://github.com/yueyu1030/AttrPrompt}}.

        ----

        ## [2433] GraphPatcher: Mitigating Degree Bias for Graph Neural Networks via Test-time Augmentation

        **Authors**: *Mingxuan Ju, Tong Zhao, Wenhao Yu, Neil Shah, Yanfang Ye*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ae9bbdcea94d808882f3535e8ca00542-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ae9bbdcea94d808882f3535e8ca00542-Abstract-Conference.html)

        **Abstract**:

        Recent studies have shown that graph neural networks (GNNs) exhibit strong biases towards the node degree: they usually perform satisfactorily on high-degree nodes with rich neighbor information but struggle with low-degree nodes. Existing works tackle this problem by deriving either designated GNN architectures or training strategies specifically for low-degree nodes. Though effective, these approaches unintentionally create an artificial out-of-distribution scenario, where models mainly or even only observe low-degree nodes during the training, leading to a downgraded performance for high-degree nodes that GNNs originally perform well at. In light of this, we propose a test-time augmentation framework, namely GraphPatcher, to enhance test-time generalization of any GNNs on low-degree nodes. Specifically, GraphPatcher iteratively generates virtual nodes to patch artificially created low-degree nodes via corruptions, aiming at progressively reconstructing target GNN's predictions over a sequence of increasingly corrupted nodes. Through this scheme, GraphPatcher not only learns how to enhance low-degree nodes (when the neighborhoods are heavily corrupted) but also preserves the original superior performance of GNNs on high-degree nodes (when lightly corrupted). Additionally, GraphPatcher is model-agnostic and can also mitigate the degree bias for either self-supervised or supervised GNNs. Comprehensive experiments are conducted over seven benchmark datasets and GraphPatcher consistently enhances common GNNs' overall performance by up to 3.6% and low-degree performance by up to 6.5%, significantly outperforming state-of-the-art baselines. The source code is publicly available at https://github.com/jumxglhf/GraphPatcher.

        ----

        ## [2434] Optimal privacy guarantees for a relaxed threat model: Addressing sub-optimal adversaries in differentially private machine learning

        **Authors**: *Georgios Kaissis, Alexander Ziller, Stefan Kolek, Anneliese Riess, Daniel Rueckert*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aea831d6c7af37fd4230937225be3414-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aea831d6c7af37fd4230937225be3414-Abstract-Conference.html)

        **Abstract**:

        Differentially private mechanisms restrict the membership inference capabilities of powerful (optimal) adversaries against machine learning models. Such adversaries are rarely encountered in practice. In this work, we examine a more realistic threat model relaxation, where (sub-optimal) adversaries lack access to the exact model training database, but may possess related or partial data. We then formally characterise and experimentally validate adversarial membership inference capabilities in this setting in terms of hypothesis testing errors. Our work helps users to interpret the privacy properties of sensitive data processing systems under realistic threat model relaxations and choose appropriate noise levels for their use-case.

        ----

        ## [2435] Stochastic Approximation Algorithms for Systems of Interacting Particles

        **Authors**: *Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aebbbfa9680eafefd43a0edc85c101f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aebbbfa9680eafefd43a0edc85c101f9-Abstract-Conference.html)

        **Abstract**:

        Interacting particle systems have proven highly successful in various machinelearning tasks, including approximate Bayesian inference and neural network optimization. However, the analysis of thesesystems often relies on the simplifying assumption of the \emph{mean-field} limit, where particlenumbers approach infinity and infinitesimal step sizes are used. In practice, discrete time steps,finite particle numbers, and complex integration schemes are employed, creating a theoretical gapbetween continuous-time and discrete-time processes. In this paper, we present a novel frameworkthat establishes a precise connection between these discrete-time schemes and their correspondingmean-field limits in terms of convergence properties and asymptotic behavior. By adopting a dynamical system perspective, our framework seamlessly integrates various numerical schemes that are typically analyzed independently. For example, our framework provides a unified treatment of optimizing an infinite-width two-layer neural network and sampling via Stein Variational Gradient descent, which were previously studied in isolation.

        ----

        ## [2436] Provable Guarantees for Neural Networks via Gradient Feature Learning

        **Authors**: *Zhenmei Shi, Junyi Wei, Yingyu Liang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aebec8058f23a445353c83ede0e1ec48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aebec8058f23a445353c83ede0e1ec48-Abstract-Conference.html)

        **Abstract**:

        Neural networks have achieved remarkable empirical performance, while the current theoretical analysis is not adequate for understanding their success, e.g., the Neural Tangent Kernel approach fails to capture their key feature learning ability, while recent analyses on feature learning are typically problem-specific. This work proposes a unified analysis framework for two-layer networks trained by gradient descent. The framework is centered around the principle of feature learning from gradients, and its effectiveness is demonstrated by applications in several prototypical problems, such as mixtures of Gaussians and parity functions.The framework also sheds light on interesting network learning phenomena such as feature learning beyond kernels and the lottery ticket hypothesis.

        ----

        ## [2437] Binary Radiance Fields

        **Authors**: *Seungjoo Shin, Jaesik Park*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aebf6284fe85a8f44b4785d41bc8249a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aebf6284fe85a8f44b4785d41bc8249a-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose \textit{binary radiance fields} (BiRF), a storage-efficient radiance field representation employing binary feature encoding in a format of either $+1$ or $-1$. This binarization strategy lets us represent the feature grid with highly compact feature encoding and a dramatic reduction in storage size. Furthermore, our 2D-3D hybrid feature grid design enhances the compactness of feature encoding as the 3D grid includes main components while 2D grids capture details. In our experiments, binary radiance field representation successfully outperforms the reconstruction performance of state-of-the-art (SOTA) storage-efficient radiance field models with lower storage allocation. In particular, our model achieves impressive results in static scene reconstruction, with a PSNR of 32.03 dB for Synthetic-NeRF scenes, 34.48 dB for Synthetic-NSVF scenes, 28.20 dB for Tanks and Temples scenes while only utilizing 0.5 MB of storage space, respectively. We hope the proposed binary radiance field representation will make radiance fields more accessible without a storage bottleneck.

        ----

        ## [2438] A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning

        **Authors**: *Alicia Curth, Alan Jeffares, Mihaela van der Schaar*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aec5e2847c5ae90f939ab786774856cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aec5e2847c5ae90f939ab786774856cc-Abstract-Conference.html)

        **Abstract**:

        Conventional statistical wisdom established a well-understood relationship between model complexity and prediction error, typically presented as a _U-shaped curve_ reflecting a transition between under- and overfitting regimes. However, motivated by the success of overparametrized neural networks, recent influential work has suggested this theory to be generally incomplete, introducing an additional regime that exhibits a second descent in test error as the parameter count $p$ grows past sample size $n$  -- a  phenomenon dubbed  _double descent_. While most attention has naturally been given to the deep-learning setting, double descent was shown to emerge more generally across non-neural models: known cases include _linear regression, trees, and boosting_. In this work, we take a closer look at the evidence surrounding these more classical statistical machine learning methods and challenge the claim that observed cases of  double descent truly extend the limits of a traditional U-shaped complexity-generalization curve therein. We show that once careful consideration is given to _what is being plotted_ on the x-axes of their double descent plots, it becomes apparent that there are implicitly multiple, distinct complexity axes along which the parameter count grows. We demonstrate that the second descent appears exactly (and _only_) when and where the transition between these underlying axes occurs, and that its location is thus _not_ inherently tied to the interpolation threshold $p=n$. We then gain further insight by adopting a classical nonparametric statistics perspective. We interpret the investigated methods as _smoothers_ and propose a generalized measure for the _effective_ number of parameters they use _on unseen examples_, using which we find that their apparent double descent curves do indeed fold back into more traditional convex shapes -- providing a resolution to the ostensible tension between double descent and traditional statistical intuition.

        ----

        ## [2439] FABind: Fast and Accurate Protein-Ligand Binding

        **Authors**: *Qizhi Pei, Kaiyuan Gao, Lijun Wu, Jinhua Zhu, Yingce Xia, Shufang Xie, Tao Qin, Kun He, Tie-Yan Liu, Rui Yan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aee1de5f335558b546b7e58c380be087-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aee1de5f335558b546b7e58c380be087-Abstract-Conference.html)

        **Abstract**:

        Modeling the interaction between proteins and ligands and accurately predicting their binding structures is a critical yet challenging task in drug discovery. Recent advancements in deep learning have shown promise in addressing this challenge, with sampling-based and regression-based methods emerging as two prominent approaches. However, these methods have notable limitations. Sampling-based methods often suffer from low efficiency due to the need for generating multiple candidate structures for selection. On the other hand, regression-based methods offer fast predictions but may experience decreased accuracy. Additionally, the variation in protein sizes often requires external modules for selecting suitable binding pockets, further impacting efficiency. In this work, we propose FABind, an end-to-end model that combines pocket prediction and docking to achieve accurate and fast protein-ligand binding.  FABind incorporates a unique ligand-informed pocket prediction module, which is also leveraged for docking pose estimation. The model further enhances the docking process by incrementally integrating the predicted pocket to optimize protein-ligand binding, reducing discrepancies between training and inference. Through extensive experiments on benchmark datasets, our proposed FABind demonstrates strong advantages in terms of effectiveness and efficiency compared to existing methods. Our code is available at https://github.com/QizhiPei/FABind.

        ----

        ## [2440] Geometric Transformer with Interatomic Positional Encoding

        **Authors**: *Yusong Wang, Shaoning Li, Tong Wang, Bin Shao, Nanning Zheng, Tie-Yan Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aee2f03ecb2b2c1ea55a43946b651cfd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aee2f03ecb2b2c1ea55a43946b651cfd-Abstract-Conference.html)

        **Abstract**:

        The widespread adoption of Transformer architectures in various data modalities has opened new avenues for the applications in molecular modeling. Nevertheless, it remains elusive that whether the Transformer-based architecture can do molecular modeling as good as equivariant GNNs. In this paper, by designing Interatomic Positional Encoding (IPE) thatparameterizes atomic environments as Transformer's positional encodings,we propose Geoformer, a novel geometric Transformer to effectively model molecular structures for various molecular property prediction. We evaluate Geoformer on several benchmarks, including the QM9 dataset and the recently proposed Molecule3D dataset. Compared with both Transformers and equivariant GNN models, Geoformer outperforms the state-of-the-art (SoTA) algorithms on QM9, and achieves the best performance on Molecule3D for both random and scaffold splits.By introducing IPE, Geoformer paves the way for molecular geometric modeling based on Transformer architecture.Codes are available at https://github.com/microsoft/AI2BMD/tree/Geoformer.

        ----

        ## [2441] A Diffusion-Model of Joint Interactive Navigation

        **Authors**: *Matthew Niedoba, Jonathan Wilder Lavington, Yunpeng Liu, Vasileios Lioutas, Justice Sefas, Xiaoxuan Liang, Dylan Green, Setareh Dabiri, Berend Zwartsenberg, Adam Scibior, Frank Wood*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aeeddfbab4e99763ebac9221732c80dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aeeddfbab4e99763ebac9221732c80dd-Abstract-Conference.html)

        **Abstract**:

        Simulation of autonomous vehicle systems requires that simulated traffic participants exhibit diverse and realistic behaviors. The use of prerecorded real-world traffic scenarios in simulation ensures realism but the rarity of safety critical events makes large scale collection of driving scenarios expensive. In this paper, we present DJINN -- a diffusion based method of generating traffic scenarios. Our approach jointly diffuses the trajectories of all agents, conditioned on a flexible set of state observations from the past, present, or future. On popular trajectory forecasting datasets, we report state of the art performance on joint trajectory metrics. In addition, we demonstrate how DJINN flexibly enables direct test-time sampling from a variety of valuable conditional distributions including goal-based sampling, behavior-class sampling, and scenario editing.

        ----

        ## [2442] Diversifying Spatial-Temporal Perception for Video Domain Generalization

        **Authors**: *Kun-Yu Lin, Jia-Run Du, Yipeng Gao, Jiaming Zhou, Wei-Shi Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aef34c770664d06eabdfebc5d3d58a9c-Abstract-Conference.html)

        **Abstract**:

        Video domain generalization aims to learn generalizable video classification models for unseen target domains by training in a source domain.A critical challenge of video domain generalization is to defend against the heavy reliance on domain-specific cues extracted from the source domain when recognizing target videos. To this end, we propose to perceive diverse spatial-temporal cues in videos, aiming to discover potential domain-invariant cues in addition to domain-specific cues. We contribute a novel model named Spatial-Temporal Diversification Network (STDN), which improves the diversity from both space and time dimensions of video data. First, our STDN proposes to discover various types of spatial cues within individual frames by spatial grouping. Then, our STDN proposes to explicitly model spatial-temporal dependencies between video contents at multiple space-time scales by spatial-temporal relation modeling. Extensive experiments on three benchmarks of different types demonstrate the effectiveness and versatility of our approach.

        ----

        ## [2443] Conformal Prediction for Time Series with Modern Hopfield Networks

        **Authors**: *Andreas Auer, Martin Gauch, Daniel Klotz, Sepp Hochreiter*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/aef75887979ae1287b5deb54a1e3cbda-Abstract-Conference.html)

        **Abstract**:

        To quantify uncertainty, conformal prediction methods are gaining continuously more interest and have already been successfully applied to various domains. However, they are difficult to apply to time series as the autocorrelative structure of time series violates basic assumptions required by conformal prediction. We propose HopCPT, a novel conformal prediction approach for time series that not only copes with temporal structures but leverages them. We show that our approach is theoretically well justified for time series where temporal dependencies are present. In experiments, we demonstrate that our new approach outperforms state-of-the-art conformal prediction methods on multiple real-world time series datasets from four different domains.

        ----

        ## [2444] Cross-modal Prompts: Adapting Large Pre-trained Models for Audio-Visual Downstream Tasks

        **Authors**: *Haoyi Duan, Yan Xia, Mingze Zhou, Li Tang, Jieming Zhu, Zhou Zhao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af01716e08073368a7c8a62be46dba17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af01716e08073368a7c8a62be46dba17-Abstract-Conference.html)

        **Abstract**:

        In recent years, the deployment of large-scale pre-trained models in audio-visual downstream tasks has yielded remarkable outcomes. However, these models, primarily trained on single-modality unconstrained datasets, still encounter challenges in feature extraction for multi-modal tasks, leading to suboptimal performance. This limitation arises due to the introduction of irrelevant modality-specific information during encoding, which adversely affects the performance of downstream tasks. To address this challenge, this paper proposes a novel Dual-Guided Spatial-Channel-Temporal (DG-SCT) attention mechanism. This mechanism leverages audio and visual modalities as soft prompts to dynamically adjust the parameters of pre-trained models based on the current multi-modal input features. Specifically, the DG-SCT module incorporates trainable cross-modal interaction layers into pre-trained audio-visual encoders, allowing adaptive extraction of crucial information from the current modality across spatial, channel, and temporal dimensions, while preserving the frozen parameters of large-scale pre-trained models. Experimental evaluations demonstrate that our proposed model achieves state-of-the-art results across multiple downstream tasks, including AVE, AVVP, AVS, and AVQA. Furthermore, our model exhibits promising performance in challenging few-shot and zero-shot scenarios. The source code and pre-trained models are available at https://github.com/haoyi-duan/DG-SCT.

        ----

        ## [2445] Causes and Effects of Unanticipated Numerical Deviations in Neural Network Inference Frameworks

        **Authors**: *Alexander Schlögl, Nora Hofer, Rainer Böhme*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af076c3bdbf935b81d808e37c5ede463-Abstract-Conference.html)

        **Abstract**:

        Hardware-specific optimizations in machine learning (ML) frameworks can cause numerical deviations of inference results. Quite surprisingly, despite using a fixed trained model and fixed input data, inference results are not consistent across platforms, and sometimes not even deterministic on the same platform. We study the causes of these numerical deviations for convolutional neural networks (CNN) on realistic end-to-end inference pipelines and in isolated experiments. Results from 75 distinct platforms suggest that the main causes of deviations on CPUs are differences in SIMD use, and the selection of convolution algorithms at runtime on GPUs. We link the causes and propagation effects to properties of the ML model and evaluate potential mitigations. We make our research code publicly available.

        ----

        ## [2446] Learning via Wasserstein-Based High Probability Generalisation Bounds

        **Authors**: *Paul Viallard, Maxime Haddouche, Umut Simsekli, Benjamin Guedj*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af2bb2b2280d36f8842e440b4e275152-Abstract-Conference.html)

        **Abstract**:

        Minimising upper bounds on the population risk or the generalisation gap has been widely used in structural risk minimisation (SRM) -- this is in particular at the core of PAC-Bayesian learning. Despite its successes and unfailing surge of interest in recent years, a limitation of the PAC-Bayesian framework is that most bounds involve a Kullback-Leibler (KL) divergence term (or its variations), which might exhibit erratic behavior and fail to capture the underlying geometric structure of the learning problem -- hence restricting its use in practical applications.As a remedy, recent studies have attempted to replace the KL divergence in the PAC-Bayesian bounds with the Wasserstein distance. Even though these bounds alleviated the aforementioned issues to a certain extent, they either hold in expectation, are for bounded losses, or are nontrivial to minimize in an SRM framework.  In this work, we contribute to this line of research and prove novel Wasserstein distance-based PAC-Bayesian generalisation bounds for both batch learning with independent and identically distributed (i.i.d.) data, and online learning with potentially non-i.i.d. data. Contrary to previous art, our bounds are stronger in the sense that (i) they hold with high probability, (ii) they apply to unbounded (potentially heavy-tailed) losses, and (iii) they lead to optimizable training objectives that can be used in SRM. As a result we derive novel Wasserstein-based PAC-Bayesian learning algorithms and we illustrate their empirical advantage on a variety of experiments.

        ----

        ## [2447] Towards Anytime Classification in Early-Exit Architectures by Enforcing Conditional Monotonicity

        **Authors**: *Metod Jazbec, James Urquhart Allingham, Dan Zhang, Eric T. Nalisnick*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af2d9fb5bcee19ef2dfa70d843520c97-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af2d9fb5bcee19ef2dfa70d843520c97-Abstract-Conference.html)

        **Abstract**:

        Modern predictive models are often deployed to environments in which computational budgets are dynamic. Anytime algorithms  are well-suited to such environments as, at any point during computation, they can output a prediction whose quality is a function of computation time. Early-exit neural networks have garnered attention in the context of anytime computation due to their capability to provide intermediate predictions at various stages throughout the network. However, we demonstrate that current early-exit networks are not directly applicable to anytime settings, as the quality of predictions for individual data points is not guaranteed to improve with longer computation. To address this shortcoming, we propose an elegant post-hoc modification, based on the Product-of-Experts, that encourages an early-exit network to become gradually confident. This gives our deep models the property of conditional monotonicity in the prediction quality---an essential building block towards truly anytime predictive modeling using early-exit architectures. Our empirical results on standard image-classification tasks demonstrate that such behaviors can be achieved while preserving competitive accuracy on average.

        ----

        ## [2448] A Scalable Neural Network for DSIC Affine Maximizer Auction Design

        **Authors**: *Zhijian Duan, Haoran Sun, Yurong Chen, Xiaotie Deng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html)

        **Abstract**:

        Automated auction design aims to find empirically high-revenue mechanisms through machine learning. Existing works on multi item auction scenarios can be roughly divided into RegretNet-like and affine maximizer auctions (AMAs) approaches. However, the former cannot strictly ensure dominant strategy incentive compatibility (DSIC), while the latter faces scalability issue due to the large number of allocation candidates. To address these limitations, we propose AMenuNet, a scalable neural network that constructs the AMA parameters (even including the allocation menu) from bidder and item representations. AMenuNet is always DSIC and individually rational (IR) due to the properties of AMAs, and it enhances scalability by generating candidate allocations through a neural network. Additionally, AMenuNet is permutation equivariant, and its number of parameters is independent of auction scale. We conduct extensive experiments to demonstrate that AMenuNet outperforms strong baselines in both contextual and non-contextual multi-item auctions, scales well to larger auctions, generalizes well to different settings, and identifies useful deterministic allocations. Overall, our proposed approach offers an effective solution to automated DSIC auction design, with improved scalability and strong revenue performance in various settings.

        ----

        ## [2449] Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias

        **Authors**: *Zhongwei Wan, Che Liu, Mi Zhang, Jie Fu, Benyou Wang, Sibo Cheng, Lei Ma, César Quilodrán Casas, Rossella Arcucci*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af38fb8e90d586f209235c94119ba193-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af38fb8e90d586f209235c94119ba193-Abstract-Conference.html)

        **Abstract**:

        The scarcity of data presents a critical obstacle to the efficacy of medical vision-language pre-training (VLP). A potential solution lies in the combination of datasets from various language communities.Nevertheless, the main challenge stems from the complexity of integrating diverse syntax and semantics, language-specific medical terminology, and culture-specific implicit knowledge. Therefore, one crucial aspect to consider is the presence of community bias caused by different languages.This paper presents a novel framework named Unifying Cross-Lingual Medical Vision-Language Pre-Training (\textbf{Med-UniC}), designed to integrate multi-modal medical data from the two most prevalent languages, English and Spanish. Specifically, we propose \textbf{C}ross-lingual \textbf{T}ext Alignment \textbf{R}egularization (\textbf{CTR}) to explicitly unify cross-lingual semantic representations of medical reports originating from diverse language communities. \textbf{CTR} is optimized through latent language disentanglement, rendering our optimization objective to not depend on negative samples, thereby significantly mitigating the bias from determining positive-negative sample pairs within analogous medical reports. Furthermore, it ensures that the cross-lingual representation is not biased toward any specific language community.\textbf{Med-UniC} reaches superior performance across 5 medical image tasks and 10 datasets encompassing over 30 diseases, offering a versatile framework for unifying multi-modal medical data within diverse linguistic communities.The experimental outcomes highlight the presence of community bias in cross-lingual VLP. Reducing this bias enhances the performance not only in vision-language tasks but also in uni-modal visual tasks.

        ----

        ## [2450] CD-GraB: Coordinating Distributed Example Orders for Provably Accelerated Training

        **Authors**: *A. Feder Cooper, Wentao Guo, Khiem Pham, Tiancheng Yuan, Charlie Ruan, Yucheng Lu, Christopher De Sa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/af9ac087ed9123957bb3a45dca56b9d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/af9ac087ed9123957bb3a45dca56b9d4-Abstract-Conference.html)

        **Abstract**:

        Recent research on online Gradient Balancing (GraB) has revealed that there exist permutation-based example orderings for SGD that are guaranteed to outperform random reshuffling (RR). Whereas RR arbitrarily permutes training examples, GraB leverages stale gradients from prior epochs to order examples -- achieving a provably faster convergence rate than RR. However, GraB is limited by design: while it demonstrates an impressive ability to scale-up training on centralized data, it does not naturally extend to modern distributed ML workloads. We therefore propose Coordinated Distributed GraB (CD-GraB), which uses insights from prior work on kernel thinning to translate the benefits of provably faster permutation-based example ordering to distributed settings. With negligible overhead, CD-GraB exhibits a linear speedup in convergence rate over centralized GraB and outperforms distributed RR on a variety of benchmark tasks.

        ----

        ## [2451] Individualized Dosing Dynamics via Neural Eigen Decomposition

        **Authors**: *Stav Belogolovsky, Ido Greenberg, Danny Eytan, Shie Mannor*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afc9f18089928eca34c347fee4757f72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afc9f18089928eca34c347fee4757f72-Abstract-Conference.html)

        **Abstract**:

        Dosing models often use differential equations to model biological dynamics. Neural differential equations in particular can learn to predict the derivative of a process, which permits predictions at irregular points of time. However, this temporal flexibility often comes with a high sensitivity to noise, whereas medical problems often present high noise and limited data. Moreover, medical dosing models must generalize reliably over individual patients and changing treatment policies. To address these challenges, we introduce the Neural Eigen Stochastic Differential Equation algorithm (NESDE). NESDE provides individualized modeling (using a hypernetwork over patient-level parameters); generalization to new treatment policies (using decoupled control); tunable expressiveness according to the noise level (using piecewise linearity); and fast, continuous, closed-form prediction (using spectral representation). We demonstrate the robustness of NESDE in both synthetic and real medical problems, and use the learned dynamics to publish simulated medical gym environments.

        ----

        ## [2452] Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems

        **Authors**: *Benjamin Coleman, Wang-Cheng Kang, Matthew Fahrbach, Ruoxi Wang, Lichan Hong, Ed H. Chi, Derek Zhiyuan Cheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afcac2e300bc243d15c25cd4f4040f0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afcac2e300bc243d15c25cd4f4040f0d-Abstract-Conference.html)

        **Abstract**:

        Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a $d$-dimensional embedding, which introduces hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used for many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multiplexed representations give Pareto-optimal space-accuracy tradeoffs for three public benchmark datasets. Further, we propose a highly practical approach called Unified Embedding with three major benefits: simplified feature configuration, strong adaptation to dynamic data distributions, and compatibility with modern hardware. Unified embedding gives significant improvements in offline and online metrics compared to highly competitive baselines across five web-scale search, ads, and recommender systems, where it serves billions of users across the world in industry-leading products.

        ----

        ## [2453] Counterfactual Generation with Identifiability Guarantees

        **Authors**: *Hanqi Yan, Lingjing Kong, Lin Gui, Yuejie Chi, Eric P. Xing, Yulan He, Kun Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afda6bf3fb086eabbaf161ba1cec5a9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afda6bf3fb086eabbaf161ba1cec5a9a-Abstract-Conference.html)

        **Abstract**:

        Counterfactual generation lies at the core of various machine learning tasks, including image translation and controllable text generation. This generation process usually requires the identification of the disentangled latent representations, such as content and style, that underlie the observed data. However, it becomes more  challenging when faced with a scarcity of paired data and labelling information. Existing disentangled methods crucially rely on oversimplified assumptions, such as assuming independent content and style variables, to identify the latent variables, even though such assumptions may not hold for complex data distributions. For instance, food reviews tend to involve words like “tasty”, whereas movie reviews commonly contain words such as “thrilling” for the same positive sentiment. This problem is exacerbated when data are sampled from multiple domains since the dependence between content and style may vary significantly over domains. In this work, we tackle the domain-varying dependence between the content and the style variables inherent in the counterfactual generation task. We provide identification guarantees for such latent-variable models by leveraging the relative sparsity of the influences from different latent variables. Our theoretical insights enable the development of a doMain AdapTive counTerfactual gEneration model, called (MATTE). Our theoretically grounded framework achieves state-of-the-art performance in unsupervised style transfer tasks, where neither paired data nor style labels are utilized, across four large-scale datasets.

        ----

        ## [2454] A Batch-to-Online Transformation under Random-Order Model

        **Authors**: *Jing Dong, Yuichi Yoshida*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/afe99e55be23b3523818da1fefa33494-Abstract-Conference.html)

        **Abstract**:

        We introduce a transformation framework that can be utilized to develop online algorithms with low $\epsilon$-approximate regret in the random-order model from offline approximation algorithms. We first give a general reduction theorem that transforms an offline approximation algorithm with low average sensitivity to an online algorithm with low $\epsilon$-approximate regret. We then demonstrate that offline approximation algorithms can be transformed into a low-sensitivity version using a coreset construction method. To showcase the versatility of our approach, we apply it to various problems, including online $(k,z)$-clustering, online matrix approximation, and online regression, and successfully achieve polylogarithmic $\epsilon$-approximate regret for each problem. Moreover, we show that in all three cases, our algorithm also enjoys low inconsistency, which may be desired in some online applications.

        ----

        ## [2455] The CLIP Model is Secretly an Image-to-Prompt Converter

        **Authors**: *Yuxuan Ding, Chunna Tian, Haoxuan Ding, Lingqiao Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b00ef390dcd5f147fd7c5c2bb35f09be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b00ef390dcd5f147fd7c5c2bb35f09be-Abstract-Conference.html)

        **Abstract**:

        The Stable Diffusion model is a prominent text-to-image generation model that relies on a text prompt as its input, which is encoded using the Contrastive Language-Image Pre-Training (CLIP). However, text prompts have limitations when it comes to incorporating implicit information from reference images. Existing methods have attempted to address this limitation by employing expensive training procedures involving millions of training samples for image-to-image generation. In contrast, this paper demonstrates that the CLIP model, as utilized in Stable Diffusion, inherently possesses the ability to instantaneously convert images into text prompts. Such an image-to-prompt conversion can be achieved by utilizing a linear projection matrix that is calculated in a closed form. Moreover, the paper showcases that this capability can be further enhanced by either utilizing a small amount of similar-domain training data (approximately 100 images) or incorporating several online training steps (around 30 iterations) on the reference images. By leveraging these approaches, the proposed method offers a simple and flexible solution to bridge the gap between images and text prompts. This methodology can be applied to various tasks such as image variation and image editing, facilitating more effective and seamless interaction between images and textual prompts.

        ----

        ## [2456] Invariant Anomaly Detection under Distribution Shifts: A Causal Perspective

        **Authors**: *João B. S. Carvalho, Mengtao Zhang, Robin Geyer, Carlos Cotrini, Joachim M. Buhmann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b010241b9f1cdfc7d4c392db899cef86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b010241b9f1cdfc7d4c392db899cef86-Abstract-Conference.html)

        **Abstract**:

        Anomaly detection (AD) is the machine learning task of identifying highly discrepant abnormal samples by solely relying on the consistency of the normal training samples. Under the constraints of a distribution shift, the assumption that training samples and test samples are drawn from the same distribution breaks down. In this work, by leveraging tools from causal inference we attempt to increase the resilience of anomaly detection models to different kinds of distribution shifts. We begin by elucidating a simple yet necessary statistical property that ensures invariant representations, which is critical for robust AD under both domain and covariate shifts. From this property, we derive a regularization term which, when minimized, leads to partial distribution invariance across environments. Through extensive experimental evaluation on both synthetic and real-world tasks, covering a range of six different AD methods, we demonstrated significant improvements in out-of-distribution performance. Under both covariate and domain shift, models regularized with our proposed term showed marked increased robustness. Code is available at: https://github.com/JoaoCarv/invariant-anomaly-detection

        ----

        ## [2457] Stable Bias: Evaluating Societal Representations in Diffusion Models

        **Authors**: *Sasha Luccioni, Christopher Akiki, Margaret Mitchell, Yacine Jernite*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b01153e7112b347d8ed54f317840d8af-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        As machine learning-enabled Text-to-Image (TTI) systems are becoming increasingly prevalent and seeing growing adoption as commercial services, characterizing the social biases they exhibit is a necessary first step to lowering their risk of discriminatory outcomes. This evaluation, however, is made more difficult by the synthetic nature of these systems’ outputs: common definitions of diversity are grounded in social categories of people living in the world, whereas the artificial depictions of fictive humans created by these systems have no inherent gender or ethnicity. To address this need, we propose a new method for exploring the social biases in TTI systems. Our approach relies on characterizing the variation in generated images triggered by enumerating gender and ethnicity markers in the prompts, and comparing it to the variation engendered by spanning different professions. This allows us to (1) identify specific bias trends, (2) provide targeted scores to directly compare models in terms of diversity and representation, and (3) jointly model interdependent social variables to support a multidimensional analysis. We leverage this method to analyze images generated by 3 popular TTI systems (Dall·E 2 , Stable Diffusion v 1.4 and 2) and find that while all of their outputs show correlations with US labor demographics, they also consistently under-represent marginalized identities to different extents. We also release the datasets and low-code interactive bias exploration platforms developed forthis work, as well as the necessary tools to similarly evaluate additional TTI systems.

        ----

        ## [2458] Locality Sensitive Hashing in Fourier Frequency Domain For Soft Set Containment Search

        **Authors**: *Indradyumna Roy, Rishi Agarwal, Soumen Chakrabarti, Anirban Dasgupta, Abir De*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b016cbec36ff7118db303229c9048733-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b016cbec36ff7118db303229c9048733-Abstract-Conference.html)

        **Abstract**:

        In many search applications related to passage retrieval, text entailment, and subgraph search, the query and each 'document' is a set of elements, with a document being relevant if it contains the query. These elements are not represented by atomic IDs, but by  embedded representations, thereby extending set containment to soft set containment. Recent applications address soft set containment by encoding sets into fixed-size vectors and checking for elementwise vector dominance. This 0/1 property can be relaxed to an asymmetric hinge distance for scoring and ranking candidate documents. Here we focus on data-sensitive, trainable indices for fast retrieval of relevant documents. Existing LSH methods are designed for mostly symmetric or few  simple asymmetric distance functions, which are not suitable for hinge distance. Instead, we transform hinge distance into a proposed dominance similarity measure, to which we then apply a Fourier transform, thereby expressing dominance similarity as an expectation of inner products of functions in the frequency domain. Next, we approximate the expectation with an importance-sampled estimate. The overall consequence is that now we can use a traditional LSH, but in the frequency domain. To ensure that the LSH uses hash bits efficiently, we learn hash functions that are sensitive to both corpus and query distributions, mapped to the frequency domain. Our experiments show that the proposed asymmetric dominance similarity is critical to the targeted applications, and that our LSH, which we call FourierHashNet, provides a better query time vs. retrieval quality trade-off, compared to several baselines. Both the Fourier transform and the trainable hash codes contribute to performance gains.

        ----

        ## [2459] L-C2ST: Local Diagnostics for Posterior Approximations in Simulation-Based Inference

        **Authors**: *Julia Linhart, Alexandre Gramfort, Pedro Rodrigues*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0313c2f4501a81d0e0d4a1e8fbf4995-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0313c2f4501a81d0e0d4a1e8fbf4995-Abstract-Conference.html)

        **Abstract**:

        Many recent works in simulation-based inference  (SBI) rely on deep generative models to approximate complex, high-dimensional posterior distributions. However, evaluating whether or not these approximations can be trusted remains a challenge. Most approaches evaluate the posterior estimator only in expectation  over the observation space. This limits their interpretability and is not sufficient to identify for which observations the approximation can be trusted or should be improved. Building upon the well-known classifier two-sample test (C2ST), we introduce $\ell$-C2ST, a new method that allows for a local evaluation of the posterior estimator at any given observation. It offers theoretically grounded and easy to interpret -- e.g. graphical -- diagnostics, and unlike C2ST, does not require access to samples from the true posterior. In the case of normalizing flow-based posterior estimators, $\ell$-C2ST can be specialized to offer better statistical power, while being computationally more efficient. On standard SBI benchmarks, $\ell$-C2ST  provides comparable results to C2ST and outperforms alternative local approaches such as coverage tests based on highest predictive density (HPD). We further highlight the importance of local evaluation and the benefit of interpretability of $\ell$-C2ST on a challenging application from computational neuroscience.

        ----

        ## [2460] Self-Supervised Reinforcement Learning that Transfers using Random Features

        **Authors**: *Boyuan Chen, Chuning Zhu, Pulkit Agrawal, Kaiqing Zhang, Abhishek Gupta*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b048dd19ba6d85b9066aa93b4de9ad4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b048dd19ba6d85b9066aa93b4de9ad4a-Abstract-Conference.html)

        **Abstract**:

        Model-free reinforcement learning algorithms have exhibited great potential in solving single-task sequential decision-making problems with high-dimensional observations and long horizons, but are known to be hard to generalize across tasks. Model-based RL, on the other hand, learns task-agnostic models of the world that naturally enables transfer across different reward functions, but struggles to scale to complex environments due to the compounding error. To get the best of both worlds, we propose a self-supervised reinforcement learning method that enables the transfer of behaviors across tasks with different rewards, while circumventing the challenges of model-based RL. In particular, we show self-supervised pre-training of model-free reinforcement learning with a number of random features as rewards allows implicit modeling of long-horizon environment dynamics. Then, planning techniques like model-predictive control using these implicit models enable fast adaptation to problems with new reward functions. Our method is self-supervised in that it can be trained on offline datasets without reward labels, but can then be quickly deployed on new tasks. We validate that our proposed method enables transfer across tasks on a variety of manipulation and locomotion domains in simulation, opening the door to generalist decision-making agents.

        ----

        ## [2461] MGDD: A Meta Generator for Fast Dataset Distillation

        **Authors**: *Songhua Liu, Xinchao Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0506debbf49e31d25690fbd1e69cd2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0506debbf49e31d25690fbd1e69cd2f-Abstract-Conference.html)

        **Abstract**:

        Existing dataset distillation (DD) techniques typically rely on iterative strategies to synthesize condensed datasets, where datasets before and after distillation are forward and backward through neural networks a massive number of times. Despite the promising results achieved, the time efficiency of prior approaches is still far from satisfactory. Moreover, when different sizes of synthetic datasets are required, they have to repeat the iterative training procedures, which is highly cumbersome and lacks flexibility. In this paper, different from the time-consuming forward-backward passes, we introduce a generative fashion for dataset distillation with significantly improved efficiency. Specifically, synthetic samples are produced by a generator network conditioned on the initialization of DD, while synthetic labels are obtained by solving a least-squares problem in a feature space. Our theoretical analysis reveals that the errors of synthetic datasets solved in the original space and then processed by any conditional generators are upper-bounded. To find a satisfactory generator efficiently, we propose a meta-learning algorithm, where a meta generator is trained on a large dataset so that only a few steps are required to adapt to a target dataset. The meta generator is termed as MGDD in our approach. Once adapted, it can handle arbitrary sizes of synthetic datasets, even for those unseen during adaptation. Experiments demonstrate that the generator adapted with only a limited number of steps performs on par with those state-of-the-art DD methods and yields $22\times$ acceleration.

        ----

        ## [2462] New Complexity-Theoretic Frontiers of Tractability for Neural Network Training

        **Authors**: *Cornelius Brand, Robert Ganian, Mathis Rocton*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b07091c16719ad3990e3d1ccee6641f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b07091c16719ad3990e3d1ccee6641f1-Abstract-Conference.html)

        **Abstract**:

        In spite of the fundamental role of neural networks in contemporary machine learning research, our understanding of the computational complexity of optimally training neural networks remains limited even when dealing with the simplest kinds of activation functions. Indeed, while there has been a number of very recent results that establish ever-tighter lower bounds for the problem under linear and ReLU activation functions, little progress has been made towards the identification of novel polynomial-time tractable network architectures. In this article we obtain novel algorithmic upper bounds for training linear- and ReLU-activated neural networks to optimality which push the boundaries of tractability for these problems beyond the previous state of the art.

        ----

        ## [2463] V-InFoR: A Robust Graph Neural Networks Explainer for Structurally Corrupted Graphs

        **Authors**: *Senzhang Wang, Jun Yin, Chaozhuo Li, Xing Xie, Jianxin Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b07d36fb3fae0630897700593c8cf49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b07d36fb3fae0630897700593c8cf49d-Abstract-Conference.html)

        **Abstract**:

        GNN explanation method aims to identify an explanatory subgraph which contains the most informative components of the full graph. However, a major limitation of existing GNN explainers is that they are not robust to the structurally corrupted graphs, e.g., graphs with noisy or adversarial edges. On the one hand, existing GNN explainers mostly explore explanations based on either the raw graph features or the learned latent representations, both of which can be easily corrupted. On the other hand, the corruptions in graphs are irregular in terms of the structural properties, e.g., the size or connectivity of graphs, which makes the rigorous constraints used by previous GNN explainers unfeasible. To address these issues, we propose a robust GNN explainer called V-InfoR. Specifically, a robust graph representation extractor, which takes insights of variational inference, is proposed to infer the latent distribution of graph representations. Instead of directly using the corrupted raw features or representations of each single graph, we sample the graph representations from the inferred distribution for the downstream explanation generator, which can effectively eliminate the minor corruption. We next formulate the explanation exploration as a graph information bottleneck (GIB) optimization problem. As a more general method that does not need any rigorous structural constraints, our GIB-based method can adaptively capture both the regularity and irregularity of the severely corrupted graphs for explanation. Extensive evaluations on both synthetic and real-world datasets indicate that V-InfoR significantly improves the GNN explanation performance for the structurally corrupted graphs. Code and dataset are available at https://anonymous.4open.science/r/V-InfoR-EF88

        ----

        ## [2464] Beyond Average Return in Markov Decision Processes

        **Authors**: *Alexandre Marthe, Aurélien Garivier, Claire Vernade*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0a34e3c64f7e842f20ec10479c32b35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0a34e3c64f7e842f20ec10479c32b35-Abstract-Conference.html)

        **Abstract**:

        What are the functionals of the reward that can be computed and optimized exactly in Markov Decision Processes?In the finite-horizon, undiscounted setting, Dynamic Programming (DP) can only handle these operations efficiently for certain classes of statistics. We summarize the characterization of these classes for policy evaluation, and give a new answer for the planning problem. Interestingly, we prove that only generalized means can be optimized exactly, even in the more general framework of Distributional Reinforcement Learning (DistRL).DistRL permits, however, to evaluate other functionals approximately. We provide error bounds on the resulting estimators, and discuss the potential of this approach as well as its limitations.These results contribute to advancing the theory of Markov Decision Processes by examining overall characteristics of the return, and particularly risk-conscious strategies.

        ----

        ## [2465] Latent exploration for Reinforcement Learning

        **Authors**: *Alberto Silvio Chiappa, Alessandro Marin Vargas, Ann Zixiang Huang, Alexander Mathis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0ca717599b7ba84d5e4f4c8b1ef6657-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0ca717599b7ba84d5e4f4c8b1ef6657-Abstract-Conference.html)

        **Abstract**:

        In Reinforcement Learning, agents learn policies by exploring and interacting with the environment. Due to the curse of dimensionality, learning policies that map high-dimensional sensory input to motor output is particularly challenging. During training, state of the art methods (SAC, PPO, etc.) explore the environment by perturbing the actuation with independent Gaussian noise. While this unstructured exploration has proven successful in numerous tasks, it can be suboptimal for overactuated systems. When multiple actuators, such as motors or muscles, drive behavior, uncorrelated perturbations risk diminishing each other's effect, or modifying the behavior in a task-irrelevant way. While solutions to introduce time correlation across action perturbations exist, introducing correlation across actuators has been largely ignored. Here, we propose LATent TIme-Correlated Exploration (Lattice), a method to inject temporally-correlated noise into the latent state of the policy network, which can be seamlessly integrated with on- and off-policy algorithms. We demonstrate that the noisy actions generated by perturbing the network's activations can be modeled as a multivariate Gaussian distribution with a full covariance matrix. In the PyBullet locomotion tasks, Lattice-SAC achieves state of the art results, and reaches 18\% higher reward than unstructured exploration in the Humanoid environment. In the musculoskeletal control environments of MyoSuite, Lattice-PPO achieves higher reward in most reaching and object manipulation tasks, while also finding more energy-efficient policies with reductions of 20-60\%. Overall, we demonstrate the effectiveness of structured action noise in time and actuator space for complex motor control tasks. The code is available at: https://github.com/amathislab/lattice.

        ----

        ## [2466] Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

        **Authors**: *Tyler Kastner, Murat A. Erdogdu, Amir-massoud Farahmand*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of learning models for risk-sensitive reinforcement learning. We theoretically demonstrate that proper value equivalence, a method of learning models which can be used to plan optimally in the risk-neutral setting, is not sufficient to plan optimally in the risk-sensitive setting. We leverage distributional reinforcement learning to introduce two new notions of model equivalence, one which is general and can be used to plan for any risk measure, but is intractable; and a practical variation which allows one to choose which risk measures they may plan optimally for. We demonstrate how our models can be used to augment any model-free risk-sensitive algorithm, and provide both tabular and large-scale experiments to demonstrate our methodâ€™s ability.

        ----

        ## [2467] Group Robust Classification Without Any Group Information

        **Authors**: *Christos Tsirigotis, João Monteiro, Pau Rodríguez, David Vázquez, Aaron C. Courville*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b0d9ceb3d11d013e55da201d2a2c07b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b0d9ceb3d11d013e55da201d2a2c07b2-Abstract-Conference.html)

        **Abstract**:

        Empirical risk minimization (ERM) is sensitive to spurious correlations present in training data, which poses a significant risk when deploying systems trained under this paradigm in high-stake applications. While the existing literature focuses on maximizing group-balanced or worst-group accuracy, estimating these quantities is hindered by costly bias annotations. This study contends that current bias-unsupervised approaches to group robustness continue to rely on group information to achieve optimal performance. Firstly, these methods implicitly assume that all group combinations are represented during training. To illustrate this, we introduce a systematic generalization task on the MPI3D dataset and discover that current algorithms fail to improve the ERM baseline when combinations of observed attribute values are missing. Secondly, bias labels are still crucial for effective model selection, restricting the practicality of these methods in real-world scenarios. To address these limitations, we propose a revised methodology for training and validating debiased models in an entirely bias-unsupervised manner. We achieve this by employing pretrained self-supervised models to reliably extract bias information, which enables the integration of a logit adjustment training loss with our validation criterion. Our empirical analysis on synthetic and real-world tasks provides evidence that our approach overcomes the identified challenges and consistently enhances robust accuracy, attaining performance which is competitive with or outperforms that of state-of-the-art methods, which, conversely, rely on bias labels for validation.

        ----

        ## [2468] Tackling Heavy-Tailed Rewards in Reinforcement Learning with Function Approximation: Minimax Optimal and Instance-Dependent Regret Bounds

        **Authors**: *Jiayi Huang, Han Zhong, Liwei Wang, Lin Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b11393733b1ea5890100302ab8a0f74c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b11393733b1ea5890100302ab8a0f74c-Abstract-Conference.html)

        **Abstract**:

        While numerous works have focused on devising efficient algorithms for reinforcement learning (RL) with uniformly bounded rewards, it remains an open question whether sample or time-efficient algorithms for RL with large state-action space exist when the rewards are \emph{heavy-tailed}, i.e., with only finite $(1+\epsilon)$-th moments for some $\epsilon\in(0,1]$. In this work, we address the challenge of such rewards in RL with linear function approximation. We first design an algorithm, \textsc{Heavy-OFUL}, for heavy-tailed linear bandits, achieving an \emph{instance-dependent} $T$-round regret of $\tilde{O}\big(d T^{\frac{1-\epsilon}{2(1+\epsilon)}} \sqrt{\sum_{t=1}^T \nu_t^2} + d T^{\frac{1-\epsilon}{2(1+\epsilon)}}\big)$, the \emph{first} of this kind. Here, $d$ is the feature dimension, and $\nu_t^{1+\epsilon}$ is the $(1+\epsilon)$-th central moment of the reward at the $t$-th round. We further show the above bound is minimax optimal when applied to the worst-case instances in stochastic and deterministic linear bandits. We then extend this algorithm to the RL settings with linear function approximation. Our algorithm, termed as \textsc{Heavy-LSVI-UCB}, achieves the \emph{first} computationally efficient \emph{instance-dependent} $K$-episode regret of $\tilde{O}(d \sqrt{H \mathcal{U}^*} K^\frac{1}{1+\epsilon} + d \sqrt{H \mathcal{V}^* K})$. Here, $H$ is length of the episode, and $\mathcal{U}^*, \mathcal{V}^*$ are instance-dependent quantities scaling with the central moment of reward and value functions, respectively. We also provide a matching minimax lower bound $\Omega(d H K^{\frac{1}{1+\epsilon}} + d \sqrt{H^3 K})$ to demonstrate the optimality of our algorithm in the worst case. Our result is achieved via a novel robust self-normalized concentration inequality that may be of independent interest in handling heavy-tailed noise in general online regression problems.

        ----

        ## [2469] Learning Dictionary for Visual Attention

        **Authors**: *Yingjie Liu, Xuan Liu, Hui Yu, Xuan Tang, Xian Wei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b113e1441ad107b80c576b5028fd2c51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b113e1441ad107b80c576b5028fd2c51-Abstract-Conference.html)

        **Abstract**:

        Recently, the attention mechanism has shown outstanding competence in capturing global structure information and long-range relationships within data, thus enhancing the performance of deep vision models on various computer vision tasks. In this work, we propose a novel dictionary learning-based attention (\textit{Dic-Attn}) module, which models this issue as a decomposition and reconstruction problem with the sparsity prior, inspired by sparse coding in the human visual perception system. The proposed \textit{Dic-Attn} module decomposes the input into a dictionary and corresponding sparse representations, allowing for the disentanglement of underlying nonlinear structural information in visual data and the reconstruction of an attention embedding. By applying transformation operations in the spatial and channel domains, the module dynamically selects the dictionary's atoms and sparse representations. Finally, the updated dictionary and sparse representations capture the global contextual information and reconstruct the attention maps. The proposed \textit{Dic-Attn} module is designed with plug-and-play compatibility, allowing for integration into deep attention encoders. Our approach offers an intuitive and elegant means to exploit the discriminative information from data, promoting visual attention construction. Extensive experimental results on various computer vision tasks, e.g., image and point cloud classification, validate that our method achieves promising performance, and shows a strong competitive comparison with state-of-the-art attention methods.

        ----

        ## [2470] A Bayesian Take on Gaussian Process Networks

        **Authors**: *Enrico Giudice, Jack Kuipers, Giusi Moffa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b146e7c87685fa208bd95ce4b08e330c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b146e7c87685fa208bd95ce4b08e330c-Abstract-Conference.html)

        **Abstract**:

        Gaussian Process Networks (GPNs) are a class of directed graphical models which employ Gaussian processes as priors for the conditional expectation of each variable given its parents in the network. The model allows the description of continuous joint distributions in a compact but flexible manner with minimal parametric assumptions on the dependencies between variables. Bayesian structure learning of GPNs requires computing the posterior over graphs of the network and is computationally infeasible even in low dimensions. This work implements Monte Carlo and Markov Chain Monte Carlo methods to sample from the posterior distribution of network structures. As such, the approach follows the Bayesian paradigm, comparing models via their marginal likelihood and computing the posterior probability of the GPN features. Simulation studies show that our method outperforms state-of-the-art algorithms in recovering the graphical structure of the network and provides an accurate approximation of its posterior distribution.

        ----

        ## [2471] Exploring Question Decomposition for Zero-Shot VQA

        **Authors**: *Zaid Khan, Vijay Kumar B. G, Samuel Schulter, Manmohan Chandraker, Yun Fu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b14cf0a01f7a8b9cd3e365e40f910272-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b14cf0a01f7a8b9cd3e365e40f910272-Abstract-Conference.html)

        **Abstract**:

        Visual question answering (VQA) has traditionally been treated as a single-step task where each question receives the same amount of effort, unlike natural human question-answering strategies. We explore a question decomposition strategy for VQA to overcome this limitation. We probe the ability of recently developed large vision-language models to use human-written decompositions and produce their own decompositions of visual questions, finding they are capable of learning both tasks from demonstrations alone.However, we show that naive application of model-written decompositions can hurt performance.We introduce a model-driven selective decomposition approach for second-guessing predictions and correcting errors, and validate its effectiveness on eight VQA tasks across three domains, showing consistent improvements in accuracy, including improvements of >20% on medical VQA datasets and boosting the zero-shot performance of BLIP-2 above chance on a VQA reformulation of the challenging Winoground task. Project Site: https://zaidkhan.me/decomposition-0shot-vqa/

        ----

        ## [2472] Sharp Recovery Thresholds of Tensor PCA Spectral Algorithms

        **Authors**: *Michael Feldman, David Donoho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b14d76c7266be21b338527cd25deac45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b14d76c7266be21b338527cd25deac45-Abstract-Conference.html)

        **Abstract**:

        Many applications seek to recover low-rank approximations of noisy tensor data. We consider several practical and effective matricization strategies which construct specific matrices from such tensors and then apply spectral methods; the strategies include tensor unfolding, partial tracing, power iteration, and recursive unfolding. We settle the behaviors of unfolding and partial tracing, identifying sharp thresholds in signal-to-noise ratio above which the signal is partially recovered. In particular, we extend previous results to a much larger class of tensor shapes where axis lengths may be different. For power iteration and recursive unfolding, we prove that under conditions where previous algorithms partially recovery the signal, these methods achieve (asymptotically) exact recovery. Our analysis deploys random matrix theory to obtain sharp thresholds which elude perturbation and concentration bounds. Specifically, we rely upon recent disproportionate random matrix results, which describe sequences of matrices with diverging aspect ratio.

        ----

        ## [2473] R-divergence for Estimating Model-oriented Distribution Discrepancy

        **Authors**: *Zhilin Zhao, Longbing Cao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b157cfde6794e93b2353b9712bbd45a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b157cfde6794e93b2353b9712bbd45a5-Abstract-Conference.html)

        **Abstract**:

        Real-life data are often non-IID due to complex distributions and interactions, and the sensitivity to the distribution of samples can differ among learning models. Accordingly, a key question for any supervised or unsupervised model is whether the probability distributions of two given datasets can be considered identical. To address this question, we introduce R-divergence, designed to assess model-oriented distribution discrepancies. The core insight is that two distributions are likely identical if their optimal hypothesis yields the same expected risk for each distribution. To estimate the distribution discrepancy between two datasets, R-divergence learns a minimum hypothesis on the mixed data and then gauges the empirical risk difference between them. We evaluate the test power across various unsupervised and supervised tasks and find that R-divergence achieves state-of-the-art performance. To demonstrate the practicality of R-divergence, we employ R-divergence to train robust neural networks on samples with noisy labels.

        ----

        ## [2474] On-the-Fly Adapting Code Summarization on Trainable Cost-Effective Language Models

        **Authors**: *YuFan Cai, Yun Lin, Chenyan Liu, Jinglian Wu, Yifan Zhang, Yiming Liu, Yeyun Gong, Jin Song Dong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b16e6de5fbbdcb2df237aa66b302bc17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b16e6de5fbbdcb2df237aa66b302bc17-Abstract-Conference.html)

        **Abstract**:

        Deep learning models are emerging to summarize source code to comment,  facilitating tasks of code documentation and program comprehension.  Scaled-up large language models trained on large open corpus have achieved good performance in such tasks.  However, in practice, the subject code in one certain project can be specific,  which may not align with the overall training corpus.  Some code samples from other projects may be contradictory and introduce inconsistencies when the models try to fit all the samples.  In this work, we introduce a novel approach, Adacom, to improve the performance of comment generators by on-the-fly model adaptation.  This research is motivated by the observation that deep comment generators  often need to strike a balance as they need to fit all the training samples.  Specifically, for one certain target code $c$,  some training samples $S_p$ could have made more contributions while other samples $S_o$ could have counter effects.  However, the traditional fine-tuned models need to fit both $S_p$ and $S_o$ from a global perspective,   leading to compromised performance for one certain target code $c$.  In this context, we design Adacom to  (1) detect whether the model might have a compromised performance on a target code $c$ and  (2) retrieve a few helpful training samples $S_p$ that have contradictory samples in the training dataset and,  (3) adapt the model on the fly by re-training the $S_p$ to strengthen the helpful samples and unlearn the harmful samples.  Our extensive experiments on 7 comment generators and 4 public datasets show that  (1) can significantly boost the performance of comment generation (BLEU4 score by on average 14.9\%, METEOR by 12.2\%, and ROUGE-L by 7.4\%),  (2) the adaptation on one code sample is cost-effective and acceptable as an on-the-fly solution, and  (3) can adapt well on out-of-distribution code samples.

        ----

        ## [2475] Exploring and Interacting with the Set of Good Sparse Generalized Additive Models

        **Authors**: *Chudi Zhong, Zhi Chen, Jiachang Liu, Margo I. Seltzer, Cynthia Rudin*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1719f44953c2e0754a016ab267fe4e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1719f44953c2e0754a016ab267fe4e7-Abstract-Conference.html)

        **Abstract**:

        In real applications, interaction between machine learning models and domain experts is critical; however, the classical machine learning paradigm that usually produces only a single model does not facilitate such interaction. Approximating and exploring the Rashomon set, i.e., the set of all near-optimal models, addresses this practical challenge by providing the user with a searchable space containing a diverse set of models from which domain experts can choose. We present algorithms to efficiently and accurately approximate the Rashomon set of sparse, generalized additive models with ellipsoids for fixed support sets and use these ellipsoids to approximate Rashomon sets for many different support sets. The approximated Rashomon set serves as a cornerstone to solve practical challenges such as (1) studying the variable importance for the model class; (2) finding models under user-specified constraints (monotonicity, direct editing); and (3) investigating sudden changes in the shape functions. Experiments demonstrate the fidelity of the approximated Rashomon set and its effectiveness in solving practical challenges.

        ----

        ## [2476] Convergence Analysis of Sequential Federated Learning on Heterogeneous Data

        **Authors**: *Yipeng Li, Xinchen Lyu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b18e5d6a10ba57d5273871f38189f062-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b18e5d6a10ba57d5273871f38189f062-Abstract-Conference.html)

        **Abstract**:

        There are two categories of methods in Federated Learning (FL) for joint training across multiple clients: i) parallel FL (PFL), where clients train models in a parallel manner; and ii) sequential FL (SFL), where clients train models in a sequential manner. In contrast to that of PFL, the convergence theory of SFL on heterogeneous data is still lacking. In this paper, we establish the convergence guarantees of SFL for strongly/general/non-convex objectives on heterogeneous data. The convergence guarantees of SFL are better than that of PFL on heterogeneous data with both full and partial client participation. Experimental results validate the counterintuitive analysis result that SFL outperforms PFL on extremely heterogeneous data in cross-device settings.

        ----

        ## [2477] Disambiguated Attention Embedding for Multi-Instance Partial-Label Learning

        **Authors**: *Wei Tang, Weijia Zhang, Min-Ling Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1917a4bcfab403c3cdd6c6bbaf9fda0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1917a4bcfab403c3cdd6c6bbaf9fda0-Abstract-Conference.html)

        **Abstract**:

        In many real-world tasks, the concerned objects can be represented as a multi-instance bag associated with a candidate label set, which consists of one ground-truth label and several false positive labels. Multi-instance partial-label learning (MIPL) is a learning paradigm to deal with such tasks and has achieved favorable performances. Existing MIPL approach follows the instance-space paradigm by assigning augmented candidate label sets of bags to each instance and aggregating bag-level labels from instance-level labels. However, this scheme may be suboptimal as global bag-level information is ignored and the predicted labels of bags are sensitive to predictions of negative instances. In this paper, we study an alternative scheme where a multi-instance bag is embedded into a single vector representation. Accordingly, an intuitive algorithm named DEMIPL, i.e., Disambiguated attention Embedding for Multi-Instance Partial-Label learning, is proposed. DEMIPL employs a disambiguation attention mechanism to aggregate a multi-instance bag into a single vector representation, followed by a momentum-based disambiguation strategy to identify the ground-truth label from the candidate label set. Furthermore, we introduce a real-world MIPL dataset for colorectal cancer classification. Experimental results on benchmark and real-world datasets validate the superiority of DEMIPL against the compared MIPL and partial-label learning approaches.

        ----

        ## [2478] Efficient Batched Algorithm for Contextual Linear Bandits with Large Action Space via Soft Elimination

        **Authors**: *Osama A. Hanna, Lin Yang, Christina Fragouli*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1bdb0f22c9748203c62f29aa297ac57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1bdb0f22c9748203c62f29aa297ac57-Abstract-Conference.html)

        **Abstract**:

        In this paper, we provide the first efficient batched algorithm for contextual linear bandits with large action spaces. Unlike existing batched algorithms that rely on action elimination, which are not implementable for large action sets, our algorithm only uses a linear optimization oracle over the action set to design the policy. The proposed algorithm achieves a regret upper bound $\tilde{O}(\sqrt{T})$ with high probability,  and uses $O(\log\log T)$ batches, matching the lower bound on the number of batches (Gao et al., 2019). When specialized to linear bandits, our algorithm can achieve a high probability gap-dependent regret bound of $\tilde{O}(1/\Delta_{\min})$ with the optimal  $\log T$ number of batches, where $\Delta_{\min}$ is the minimum reward gap between a suboptimal arm and the optimal. Our result is achieved via a novel soft elimination approach, that entails $\text{``}$shaping$\text{"}$ the action sets at each batch so that we can efficiently identify (near) optimal actions.

        ----

        ## [2479] Add and Thin: Diffusion for Temporal Point Processes

        **Authors**: *David Lüdke, Marin Bilos, Oleksandr Shchur, Marten Lienen, Stephan Günnemann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract-Conference.html)

        **Abstract**:

        Autoregressive neural networks within the temporal point process (TPP) framework have become the standard for modeling continuous-time event data. Even though these models can expressively capture event sequences in a one-step-ahead fashion, they are inherently limited for long-term forecasting applications due to the accumulation of errors caused by their sequential nature. To overcome these limitations, we derive ADD-THIN, a principled probabilistic denoising diffusion model for TPPs that operates on entire event sequences. Unlike existing diffusion approaches, ADD-THIN naturally handles data with discrete and continuous components. In experiments on synthetic and real-world datasets, our model matches the state-of-the-art TPP models in density estimation and strongly outperforms them in forecasting.

        ----

        ## [2480] Pitfall of Optimism: Distributional Reinforcement Learning by Randomizing Risk Criterion

        **Authors**: *Taehyun Cho, Seungyub Han, Heesoo Lee, Kyungjae Lee, Jungwoo Lee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1eb88348ee19a33c81cf5bc3fb8e9d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1eb88348ee19a33c81cf5bc3fb8e9d2-Abstract-Conference.html)

        **Abstract**:

        Distributional reinforcement learning algorithms have attempted to utilize estimated uncertainty for exploration, such as optimism in the face of uncertainty. However, using the estimated variance for optimistic exploration may cause biased data collection and hinder convergence or performance. In this paper, we present a novel distributional reinforcement learning that selects actions by randomizing risk criterion without losing the risk-neutral objective. We provide a perturbed distributional Bellman optimality operator by distorting the risk measure. Also,we prove the convergence and optimality of the proposed method with the weaker contraction property. Our theoretical results support that the proposed method does not fall into biased exploration and is guaranteed to converge to an optimal return. Finally, we empirically show that our method outperforms other existing distribution-based algorithms in various environments including Atari 55 games.

        ----

        ## [2481] Efficient Meta Neural Heuristic for Multi-Objective Combinatorial Optimization

        **Authors**: *Jinbiao Chen, Jiahai Wang, Zizhen Zhang, Zhiguang Cao, Te Ye, Siyuan Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)

        **Abstract**:

        Recently, neural heuristics based on deep reinforcement learning have exhibited promise in solving multi-objective combinatorial optimization problems (MOCOPs). However, they are still struggling to achieve high learning efficiency and solution quality. To tackle this issue, we propose an efficient meta neural heuristic (EMNH), in which a meta-model is first trained and then fine-tuned with a few steps to solve corresponding single-objective subproblems. Specifically, for the training process, a (partial) architecture-shared multi-task model is leveraged to achieve parallel learning for the meta-model, so as to speed up the training; meanwhile, a scaled symmetric sampling method with respect to the weight vectors is designed to stabilize the training. For the fine-tuning process, an efficient hierarchical method is proposed to systematically tackle all the subproblems. Experimental results on the multi-objective traveling salesman problem (MOTSP), multi-objective capacitated vehicle routing problem (MOCVRP), and multi-objective knapsack problem (MOKP) show that, EMNH is able to outperform the state-of-the-art neural heuristics in terms of solution quality and learning efficiency, and yield competitive solutions to the strong traditional heuristics while consuming much shorter time.

        ----

        ## [2482] QuantSR: Accurate Low-bit Quantization for Efficient Image Super-Resolution

        **Authors**: *Haotong Qin, Yulun Zhang, Yifu Ding, Yifan Liu, Xianglong Liu, Martin Danelljan, Fisher Yu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2169d573d75ff90c7b12dc3a5fc2898-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2169d573d75ff90c7b12dc3a5fc2898-Abstract-Conference.html)

        **Abstract**:

        Low-bit quantization in image super-resolution (SR) has attracted copious attention in recent research due to its ability to reduce parameters and operations significantly. However, many quantized SR models suffer from accuracy degradation compared to their full-precision counterparts, especially at ultra-low bit widths (2-4 bits), limiting their practical applications. To address this issue, we propose a novel quantized image SR network, called QuantSR, which achieves accurate and efficient SR processing under low-bit quantization. To overcome the representation homogeneity caused by quantization in the network, we introduce the Redistribution-driven Learnable Quantizer (RLQ). This is accomplished through an inference-agnostic efficient redistribution design, which adds additional information in both forward and backward passes to improve the representation ability of quantized networks. Furthermore, to achieve flexible inference and break the upper limit of accuracy, we propose the Depth-dynamic Quantized Architecture (DQA). Our DQA allows for the trade-off between efficiency and accuracy during inference through weight sharing. Our comprehensive experiments show that QuantSR outperforms existing state-of-the-art quantized SR networks in terms of accuracy while also providing more competitive computational efficiency. In addition, we demonstrate the scheme's satisfactory architecture generality by providing QuantSR-C and QuantSR-T for both convolution and Transformer versions, respectively. Our code and models are released at https://github.com/htqin/QuantSR .

        ----

        ## [2483] Streaming Factor Trajectory Learning for Temporal Tensor Decomposition

        **Authors**: *Shikai Fang, Xin Yu, Shibo Li, Zheng Wang, Mike Kirby, Shandian Zhe*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b231d91e700c465dfdd6116d091a4194-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b231d91e700c465dfdd6116d091a4194-Abstract-Conference.html)

        **Abstract**:

        Practical tensor data is often along with time information. Most existing temporal decomposition approaches estimate a set of fixed factors for the objects in each tensor mode, and hence cannot capture the temporal evolution of the objects' representation. More important, we lack an effective approach to capture such evolution from streaming data, which is common in real-world applications.  To address these issues, we propose Streaming Factor Trajectory Learning (SFTL) for temporal tensor decomposition. We use Gaussian processes (GPs) to model the trajectory of  factors so as to flexibly estimate their temporal evolution. To address the computational challenges in handling streaming data, we convert the GPs into a state-space prior by constructing an equivalent stochastic differential equation (SDE).  We develop an efficient online filtering algorithm to estimate a decoupled running posterior of the involved factor states upon receiving new data. The decoupled estimation enables us to conduct standard Rauch-Tung-Striebel smoothing to compute the full posterior of all the  trajectories in parallel, without the need for revisiting any previous data. We have shown the advantage of SFTL in both synthetic tasks and real-world applications.

        ----

        ## [2484] DDF-HO: Hand-Held Object Reconstruction via Conditional Directed Distance Field

        **Authors**: *Chenyangguang Zhang, Yan Di, Ruida Zhang, Guangyao Zhai, Fabian Manhardt, Federico Tombari, Xiangyang Ji*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2876deb92cbd098219a10da25671577-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2876deb92cbd098219a10da25671577-Abstract-Conference.html)

        **Abstract**:

        Reconstructing hand-held objects from a single RGB image is an important and challenging problem. Existing works utilizing Signed Distance Fields (SDF) reveal limitations in comprehensively capturing the complex hand-object interactions, since SDF is  only reliable within the proximity of the target, and hence, infeasible to simultaneously encode local hand and object cues. To address this issue, we propose DDF-HO, a novel approach leveraging Directed Distance Field (DDF) as the shape representation. Unlike SDF, DDF maps a ray in 3D space, consisting of an origin and a direction, to corresponding DDF values, including a binary visibility signal determining whether the ray intersects the objects and a distance value measuring the distance from origin to target in the given direction. We randomly sample multiple rays and collect local to global geometric features for them by introducing a novel 2D ray-based feature aggregation scheme and a 3D intersection-aware hand pose embedding, combining 2D-3D features to model hand-object interactions. Extensive experiments on synthetic and real-world datasets demonstrate that DDF-HO consistently outperforms all baseline methods by a large margin, especially under Chamfer Distance, with about 80% leap forward. Codes are available at https://github.com/ZhangCYG/DDFHO.

        ----

        ## [2485] Effective Targeted Attacks for Adversarial Self-Supervised Learning

        **Authors**: *Minseon Kim, Hyeonjeong Ha, Sooel Son, Sung Ju Hwang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b28ae1166e1035c26b89d20f0286c9eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b28ae1166e1035c26b89d20f0286c9eb-Abstract-Conference.html)

        **Abstract**:

        Recently, unsupervised adversarial training (AT) has been highlighted as a means of achieving robustness in models without any label information. Previous studies in unsupervised AT have mostly focused on implementing self-supervised learning (SSL) frameworks, which maximize the instance-wise classification loss to generate adversarial examples. However, we observe that simply maximizing the self-supervised training loss with an untargeted adversarial attack often results in generating ineffective adversaries that may not help improve the robustness of the trained model, especially for non-contrastive SSL frameworks without negative examples. To tackle this problem, we propose a novel positive mining for targeted adversarial attack to generate effective adversaries for adversarial SSL frameworks. Specifically, we introduce an algorithm that selects the most confusing yet similar target example for a given instance based on entropy and similarity, and subsequently perturbs the given instance towards the selected target. Our method demonstrates significant enhancements in robustness when applied to non-contrastive SSL frameworks, and less but consistent robustness improvements with contrastive SSL frameworks, on the benchmark datasets.

        ----

        ## [2486] Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory

        **Authors**: *Sokhna Diarra Mbacke, Florence Clerc, Pascal Germain*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29500824d22ee9bbd25e4cd97c49b55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29500824d22ee9bbd25e4cd97c49b55-Abstract-Conference.html)

        **Abstract**:

        Since their inception, Variational Autoencoders (VAEs) have become central in machine learning. Despite their widespread use, numerous questions regarding their theoretical properties remain open. Using PAC-Bayesian theory, this work develops statistical guarantees for VAEs. First, we derive the first PAC-Bayesian bound for posterior distributions conditioned on individual samples from the data-generating distribution. Then, we utilize this result to develop generalization guarantees for the VAE's reconstruction loss, as well as upper bounds on the distance between the input and the regenerated distributions. More importantly, we provide upper bounds on the Wasserstein distance between the input distribution and the distribution defined by the VAE's generative model.

        ----

        ## [2487] Multi-Head Adapter Routing for Cross-Task Generalization

        **Authors**: *Lucas Page-Caccia, Edoardo Maria Ponti, Zhan Su, Matheus Pereira, Nicolas Le Roux, Alessandro Sordoni*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html)

        **Abstract**:

        Parameter-efficient fine-tuning (PEFT) for cross-task generalization consists in pre-training adapters on a multi-task training set before few-shot adaptation to test tasks. Polytropon [Ponti et al., 2023] ($\texttt{Poly}$) jointly learns an inventory of adapters and a *routing* function that selects a (variable-size) subset of adapters for each task during both pre-training and few-shot adaptation. In this paper, we investigate the role that adapter routing plays in its success and design new variants based on our findings.First, we build on the intuition that finer-grained routing provides more expressivity. Hence,we propose  $\texttt{MHR}$ (Multi-Head Routing) which combines *subsets* of adapter parameters and outperforms $\texttt{Poly}$ under a comparable parameter budget; by only fine-tuning the routing function and not the adapters ($\texttt{MHR}$-$z$) we achieve competitive performance with extreme parameter efficiency. Second, we find that  $\texttt{Poly}$/$\texttt{MHR}$ performance is a result of better multi-task optimization, rather than modular inductive biases that facilitate adapter recombination and local adaptation, as previously hypothesized. In fact, we find that $\texttt{MHR}$ exhibits high gradient alignment between training tasks. We find that routing is most beneficial during multi-task pre-training rather than during few-shot adaptation and propose $\texttt{MHR}$-$\mu$, which discards routing and fine-tunes the average of the pre-trained adapters on each downstream tasks. This establishes $\texttt{MHR}$-$\mu$ as an effective method for single-adapter fine-tuning. We also show that $\texttt{MHR}$-$\mu$ can be used as an effective zero-shot transfer method by training the average of the pre-trained adapters for a few additional steps on the multi-task training set: this yields gains up to 3\% on absolute accuracy w.r.t. the baselines. Code is available at .

        ----

        ## [2488] GenS: Generalizable Neural Surface Reconstruction from Multi-View Images

        **Authors**: *Rui Peng, Xiaodong Gu, Luyang Tang, Shihe Shen, Fanqi Yu, Ronggang Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29ab822442a1616f9bd390fddf6e425-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29ab822442a1616f9bd390fddf6e425-Abstract-Conference.html)

        **Abstract**:

        Combining the signed distance function (SDF) and differentiable volume rendering has emerged as a powerful paradigm for surface reconstruction from multi-view images without 3D supervision. However, current methods are impeded by requiring long-time per-scene optimizations and cannot generalize to new scenes. In this paper, we present GenS, an end-to-end generalizable neural surface reconstruction model. Unlike coordinate-based methods that train a separate network for each scene, we construct a generalized multi-scale volume to directly encode all scenes. Compared with existing solutions, our representation is more powerful, which can recover high-frequency details while maintaining global smoothness. Meanwhile, we introduce a multi-scale feature-metric consistency to impose the multi-view consistency in a more discriminative multi-scale feature space, which is robust to the failures of the photometric consistency. And the learnable feature can be self-enhanced to continuously improve the matching accuracy and mitigate aggregation ambiguity. Furthermore, we design a view contrast loss to force the model to be robust to those regions covered by few viewpoints through distilling the geometric prior from dense input to sparse input. Extensive experiments on popular benchmarks show that our model can generalize well to new scenes and outperform existing state-of-the-art methods even those employing ground-truth depth supervision. Code will be available at https://github.com/prstrive/GenS.

        ----

        ## [2489] Better with Less: A Data-Active Perspective on Pre-Training Graph Neural Networks

        **Authors**: *Jiarong Xu, Renhong Huang, Xin Jiang, Yuxuan Cao, Carl Yang, Chunping Wang, Yang Yang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29adb4bf2364acec8fb402ef731bb3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29adb4bf2364acec8fb402ef731bb3b-Abstract-Conference.html)

        **Abstract**:

        Pre-training on graph neural networks (GNNs) aims to learn transferable knowledge for downstream tasks with unlabeled data, and it has recently become an active research area. The success of graph pre-training models is often attributed to the massive amount of input data. In this paper, however, we identify the curse of big data phenomenon in graph pre-training: more training data do not necessarily lead to better downstream performance. Motivated by this observation, we propose a better-with-less framework for graph pre-training: fewer, but carefully chosen data are fed into a GNN model to enhance pre-training. The proposed pre-training pipeline is called the data-active graph pre-training (APT) framework, and is composed of a graph selector and a pre-training model. The graph selector chooses the most representative and instructive data points based on the inherent properties of graphs as well as predictive uncertainty. The proposed predictive uncertainty, as feedback from the pre-training model, measures the confidence level of the model in the data. When fed with the chosen data, on the other hand, the pre-training model grasps an initial understanding of the new, unseen data, and at the same time attempts to remember the knowledge learned from previous data. Therefore, the integration and interaction between these two components form a unified framework (APT), in which graph pre-training is performed in a progressive and iterative way. Experiment results show that the proposed APT is able to obtain an efficient pre-training model with fewer training data and better downstream performance.

        ----

        ## [2490] Brain-like Flexible Visual Inference by Harnessing Feedback Feedforward Alignment

        **Authors**: *Tahereh Toosi, Elias B. Issa*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b29ec434e049fb96f3c4245a405ee976-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b29ec434e049fb96f3c4245a405ee976-Abstract-Conference.html)

        **Abstract**:

        In natural vision, feedback connections support versatile visual inference capabilities such as making sense of the occluded or noisy bottom-up sensory information or mediating pure top-down processes such as imagination. However, the mechanisms by which the feedback pathway learns to give rise to these capabilities flexibly are not clear. We propose that top-down effects emerge through alignment between feedforward and feedback pathways, each optimizing its own objectives. To achieve this co-optimization, we introduce Feedback-Feedforward Alignment (FFA), a learning algorithm that leverages feedback and feedforward pathways as mutual credit assignment computational graphs, enabling alignment. In our study, we demonstrate the effectiveness of FFA in co-optimizing classification and reconstruction tasks on widely used MNIST and CIFAR10 datasets. Notably, the alignment mechanism in FFA endows feedback connections with emergent visual inference functions, including denoising, resolving occlusions, hallucination, and imagination. Moreover, FFA offers bio-plausibility compared to traditional backpropagation (BP) methods in implementation. By repurposing the computational graph of credit assignment into a goal-driven feedback pathway, FFA alleviates weight transport problems encountered in BP, enhancing the bio-plausibility of the learning algorithm. Our study presents FFA as a promising proof-of-concept for the mechanisms underlying how feedback connections in the visual cortex support flexible visual functions. This work also contributes to the broader field of visual inference underlying perceptual phenomena and has implications for developing more biologically inspired learning algorithms.

        ----

        ## [2491] Latent Diffusion for Language Generation

        **Authors**: *Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, Kilian Q. Weinberger*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2a2bd5d5051ff6af52e1ef60aefd255-Abstract-Conference.html)

        **Abstract**:

        Diffusion models have achieved great success in modeling continuous data modalities such as images, audio, and video, but have seen limited use in discrete domains such as language. Recent attempts to adapt diffusion to language have presented diffusion as an alternative to existing pretrained language models. We view diffusion and existing language models as complementary. We demonstrate that encoder-decoder language models can be utilized to efficiently learn high-quality language autoencoders. We then demonstrate that continuous diffusion models can be learned in the latent space of the language autoencoder, enabling us to sample continuous latent representations that can be decoded into natural language with the pretrained decoder. We validate the effectiveness of our approach for unconditional, class-conditional, and sequence-to-sequence language generation. We demonstrate across multiple diverse data sets that our latent language diffusion models are significantly more effective than previous diffusion language models. Our code is available at \url{https://github.com/justinlovelace/latent-diffusion-for-language}.

        ----

        ## [2492] The emergence of clusters in self-attention dynamics

        **Authors**: *Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, Philippe Rigollet*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2b3e1d9840eba17ad9bbf073e009afe-Abstract-Conference.html)

        **Abstract**:

        Viewing Transformers as interacting particle systems, we describe the geometry of learned representations when the weights are not time-dependent. We show that particles, representing tokens, tend to cluster toward particular limiting objects as time tends to infinity. Using techniques from dynamical systems and partial differential equations, we show that type of limiting object that emerges depends on the spectrum of the value matrix. Additionally, in the one-dimensional case we prove that the self-attention matrix converges to a low-rank Boolean matrix. The combination of these results mathematically confirms the empirical observation made by Vaswani et al. [ VSP`17 ] that leaders appear in a sequence of tokens when processed by Transformers.

        ----

        ## [2493] Self-Consistent Velocity Matching of Probability Flows

        **Authors**: *Lingxiao Li, Samuel Hurault, Justin M. Solomon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html)

        **Abstract**:

        We present a discretization-free scalable framework for solving a large class of mass-conserving partial differential equations (PDEs), including the time-dependent Fokker-Planck equation and the Wasserstein gradient flow. The main observation is that the time-varying velocity field of the PDE solution needs to be self-consistent: it must satisfy a fixed-point equation involving the probability flow characterized by the same velocity field. Instead of directly minimizing the residual of the fixed-point equation with neural parameterization, we use an iterative formulation with a biased gradient estimator that bypasses significant computational obstacles with strong empirical performance. Compared to existing approaches, our method does not suffer from temporal or spatial discretization, covers a wider range of PDEs, and scales to high dimensions. Experimentally, our method recovers analytical solutions accurately when they are available and achieves superior performance in high dimensions with less training time compared to alternatives.

        ----

        ## [2494] Deep Momentum Multi-Marginal Schrödinger Bridge

        **Authors**: *Tianrong Chen, Guan-Horng Liu, Molei Tao, Evangelos A. Theodorou*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2c39fe6ce838440faf03a0f780e7a63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2c39fe6ce838440faf03a0f780e7a63-Abstract-Conference.html)

        **Abstract**:

        It is a crucial challenge to reconstruct population dynamics using unlabeled samples from distributions at coarse time intervals. Recent approaches such as flow-based models or Schrödinger Bridge (SB) models have demonstrated appealing performance, yet the inferred sample trajectories either fail to account for the underlying stochasticity or are unnecessarily rigid. In this article, we extend SB into phase space and propose $\underline{D}$eep $\underline{M}$omentum Multi-Marginal $\underline{S}$chrödinger $\underline{B}$ridge (DMSB), a novel computational framework that learns the smooth measure-valued spline for stochastic systems that satisfy position marginal constraints across time. By tailoring the celebrated Bregman Iteration and extending the Iteration Proportional Fitting to phase space, we manage to handle high-dimensional multi-marginal trajectory inference tasks efficiently. Our algorithm outperforms baselines significantly, as evidenced by experiments for synthetic datasets and a real-world single-cell RNA sequence dataset. Additionally, the proposed approach can reasonably reconstruct the evolution of velocity distribution, from position snapshots only, when there is a ground truth velocity that is nevertheless inaccessible.

        ----

        ## [2495] Semi-Supervised Contrastive Learning for Deep Regression with Ordinal Rankings from Spectral Seriation

        **Authors**: *Weihang Dai, Yao Du, Hanru Bai, Kwang-Ting Cheng, Xiaomeng Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2d4051f03a7038a2771dfbbe5c7b54e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2d4051f03a7038a2771dfbbe5c7b54e-Abstract-Conference.html)

        **Abstract**:

        Contrastive learning methods can be applied to deep regression by enforcing label distance relationships in feature space. However, these methods are limited to labeled data only unlike for classification, where unlabeled data can be used for contrastive pretraining. In this work, we extend contrastive regression methods to allow unlabeled data to be used in a semi-supervised setting, thereby reducing the reliance on manual annotations. We observe that the feature similarity matrix between unlabeled samples still reflect inter-sample relationships, and that an accurate ordinal relationship can be recovered through spectral seriation algorithms if the level of error is within certain bounds. By using the recovered ordinal relationship for contrastive learning on unlabeled samples, we can allow more data to be used for feature representation learning, thereby achieve more robust results. The ordinal rankings can also be used to supervise predictions on unlabeled samples, which can serve as an additional training signal. We provide theoretical guarantees and empirical support through experiments on different datasets, demonstrating that our method can surpass existing state-of-the-art semi-supervised deep regression methods. To the best of our knowledge, this work is the first to explore using unlabeled data to perform contrastive learning for regression.

        ----

        ## [2496] Domain Re-Modulation for Few-Shot Generative Domain Adaptation

        **Authors**: *Yi Wu, Ziqiang Li, Chaoyue Wang, Heliang Zheng, Shanshan Zhao, Bin Li, Dacheng Tao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2e20d7402c9985eae4ba924c65370a8-Abstract-Conference.html)

        **Abstract**:

        In this study, we delve into the task of few-shot Generative Domain Adaptation (GDA), which involves transferring a pre-trained generator from one domain to a new domain using only a few reference images. Inspired by the way human brains acquire knowledge in new domains, we present an innovative generator structure called $\textbf{Domain Re-Modulation (DoRM)}$. DoRM not only meets the criteria of $\textit{high quality}$, $\textit{large synthesis diversity}$, and $\textit{cross-domain consistency}$, which were achieved by previous research in GDA, but also incorporates $\textit{memory}$ and $\textit{domain association}$, akin to how human brains operate. Specifically, DoRM freezes the source generator and introduces new mapping and affine modules (M\&A modules) to capture the attributes of the target domain during GDA. This process resembles the formation of new synapses in human brains. Consequently, a linearly combinable domain shift occurs in the style space. By incorporating multiple new M\&A modules, the generator gains the capability to perform high-fidelity multi-domain and hybrid-domain generation. Moreover, to maintain cross-domain consistency more effectively, we introduce a similarity-based structure loss. This loss aligns the auto-correlation map of the target image with its corresponding auto-correlation map of the source image during training. Through extensive experiments, we demonstrate the superior performance of our DoRM and similarity-based structure loss in few-shot GDA, both quantitatively and qualitatively. Code will be available at https://github.com/wuyi2020/DoRM.

        ----

        ## [2497] Transformers as Statisticians: Provable In-Context Learning with In-Context Algorithm Selection

        **Authors**: *Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2e63e36c57e153b9015fece2352a9f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2e63e36c57e153b9015fece2352a9f9-Abstract-Conference.html)

        **Abstract**:

        Neural sequence models based on the transformer architecture have demonstrated remarkable \emph{in-context learning} (ICL) abilities, where they can perform new tasks when prompted with training and test examples, without any parameter update to the model. This work first provides a comprehensive statistical theory for transformers to perform ICL. Concretely, we show that transformers can implement a broad class of standard machine learning algorithms in context, such as least squares, ridge regression, Lasso, learning generalized linear models, and gradient descent on two-layer neural networks, with near-optimal predictive power on various in-context data distributions. Using an efficient implementation of in-context gradient descent as the underlying mechanism, our transformer constructions admit mild size bounds, and can be learned with polynomially many pretraining sequences.    Building on these ``base'' ICL algorithms, intriguingly, we show that transformers can implement more complex ICL procedures involving \emph{in-context algorithm selection}, akin to what a statistician can do in real life---A \emph{single} transformer can adaptively select different base ICL algorithms---or even perform qualitatively different tasks---on different input sequences, without any explicit prompting of the right algorithm or task. We both establish this in theory by explicit constructions, and also observe this phenomenon experimentally. In theory, we construct two general mechanisms for algorithm selection with concrete examples: pre-ICL testing, and post-ICL validation. As an example, we use the post-ICL validation mechanism to construct a transformer that can perform nearly Bayes-optimal ICL on a challenging task---noisy linear models with mixed noise levels. Experimentally, we demonstrate the strong in-context algorithm selection capabilities of standard transformer architectures.

        ----

        ## [2498] Arbitrarily Scalable Environment Generators via Neural Cellular Automata

        **Authors**: *Yulun Zhang, Matthew C. Fontaine, Varun Bhatt, Stefanos Nikolaidis, Jiaoyang Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2fbf1c9bc92e7ef2f6cab2e8a3e09af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2fbf1c9bc92e7ef2f6cab2e8a3e09af-Abstract-Conference.html)

        **Abstract**:

        We study the problem of generating arbitrarily large environments to improve the throughput of multi-robot systems. Prior work proposes Quality Diversity (QD) algorithms as an effective method for optimizing the environments of automated warehouses. However, these approaches optimize only relatively small environments, falling short when it comes to replicating real-world warehouse sizes. The challenge arises from the exponential increase in the search space as the environment size increases. Additionally, the previous methods have only been tested with up to 350 robots in simulations, while practical warehouses could host thousands of robots. In this paper, instead of optimizing environments, we propose to optimize Neural Cellular Automata (NCA) environment generators via QD algorithms. We train a collection of NCA generators with QD algorithms in small environments and then generate arbitrarily large environments from the generators at test time. We show that NCA environment generators maintain consistent, regularized patterns regardless of environment size, significantly enhancing the scalability of multi-robot systems in two different domains with up to 2,350 robots. Additionally, we demonstrate that our method scales a single-agent reinforcement learning policy to arbitrarily large environments with similar patterns. We include the source code at https://github.com/lunjohnzhang/warehouseenvgenncapublic.

        ----

        ## [2499] FAMO: Fast Adaptive Multitask Optimization

        **Authors**: *Bo Liu, Yihao Feng, Peter Stone, Qiang Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html)

        **Abstract**:

        One of the grand enduring goals of AI is to create generalist agents that can learn multiple different tasks from diverse data via multitask learning (MTL). However, in practice, applying gradient descent (GD) on the average loss across all tasks may yield poor multitask performance due to severe under-optimization of certain tasks. Previous approaches that manipulate task gradients for a more balanced loss decrease require storing and computing all task gradients ($\mathcal{O}(k)$ space and time where $k$ is the number of tasks), limiting their use in large-scale scenarios. In this work, we introduce Fast Adaptive Multitask Optimization (FAMO), a dynamic weighting method that decreases task losses in a balanced way using $\mathcal{O}(1)$ space and time. We conduct an extensive set of experiments covering multi-task supervised and reinforcement learning problems. Our results indicate that FAMO achieves comparable or superior performance to state-of-the-art gradient manipulation techniques while offering significant improvements in space and computational efficiency. Code is available at \url{https://github.com/Cranial-XIX/FAMO}.

        ----

        ## [2500] A Theory of Multimodal Learning

        **Authors**: *Zhou Lu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b316495425d076b4abffc065a64c2cca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b316495425d076b4abffc065a64c2cca-Abstract-Conference.html)

        **Abstract**:

        Human perception of the empirical world involves recognizing the diverse appearances, or 'modalities', of underlying objects. Despite the longstanding consideration of this perspective in philosophy and cognitive science, the study of multimodality remains relatively under-explored within the field of machine learning. Nevertheless, current studies of multimodal machine learning are limited to empirical practices, lacking theoretical foundations beyond heuristic arguments. An intriguing finding from the practice of multimodal learning is that a model trained on multiple modalities can outperform a finely-tuned unimodal model, even on unimodal tasks. This paper provides a theoretical framework that explains this phenomenon, by studying generalization properties of multimodal learning algorithms. We demonstrate that multimodal learning allows for a superior generalization bound compared to unimodal learning, up to a factor of $O(\sqrt{n})$, where $n$ represents the sample size. Such advantage occurs when both connection and heterogeneity exist between the modalities.

        ----

        ## [2501] IDEA: An Invariant Perspective for Efficient Domain Adaptive Image Retrieval

        **Authors**: *Haixin Wang, Hao Wu, Jinan Sun, Shikun Zhang, Chong Chen, Xian-Sheng Hua, Xiao Luo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b33ad9d46ab2a23b6783d954121d26e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b33ad9d46ab2a23b6783d954121d26e3-Abstract-Conference.html)

        **Abstract**:

        In this paper, we investigate the problem of unsupervised domain adaptive hashing, which leverage knowledge from a label-rich source domain to expedite learning to hash on a label-scarce target domain. Although numerous existing approaches attempt to incorporate transfer learning techniques into deep hashing frameworks, they often neglect the essential invariance for adequate alignment between these two domains. Worse yet, these methods fail to distinguish between causal and non-causal effects embedded in images, rendering cross-domain retrieval ineffective. To address these challenges, we propose an Invariance-acquired Domain AdaptivE HAshing (IDEA) model. Our IDEA first decomposes each image into a causal feature representing label information, and a non-causal feature indicating domain information. Subsequently, we generate discriminative hash codes using causal features with consistency learning on both source and target domains. More importantly, we employ a generative model for synthetic samples to simulate the intervention of various non-causal effects, ultimately minimizing their impact on hash codes for domain invariance. Comprehensive experiments conducted on benchmark datasets validate the superior performance of our IDEA compared to a variety of competitive baselines.

        ----

        ## [2502] Learning Provably Robust Estimators for Inverse Problems via Jittering

        **Authors**: *Anselm Krainovic, Mahdi Soltanolkotabi, Reinhard Heckel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3411e30afa6caeefa4d6d39a5ea84cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3411e30afa6caeefa4d6d39a5ea84cd-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks provide excellent performance for inverse problems such as denoising. However, neural networks can be sensitive to adversarial or worst-case perturbations. This raises the question of whether such networks can be trained efficiently to be worst-case robust. In this paper, we investigate whether jittering, a simple regularization technique that adds isotropic Gaussian noise during training, is effective for learning worst-case robust estimators for inverse problems. While well studied for prediction in classification tasks, the effectiveness of jittering for inverse problems has not been systematically investigated. In this paper, we present a novel analytical characterization of the optimal $\ell_2$-worst-case robust estimator for linear denoising and show that jittering yields optimal robust denoisers. Furthermore, we examine jittering empirically via training deep neural networks (U-nets) for natural image denoising, deconvolution, and accelerated magnetic resonance imaging (MRI). The results show that jittering significantly enhances the worst-case robustness, but can be suboptimal for inverse problems beyond denoising. Moreover, our results imply that training on real data which often contains slight noise is somewhat robustness enhancing.

        ----

        ## [2503] On Occlusions in Video Action Detection: Benchmark Datasets And Training Recipes

        **Authors**: *Rajat Modi, Vibhav Vineet, Yogesh S. Rawat*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3640c2d3e58f716c67066046318db0f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3640c2d3e58f716c67066046318db0f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        This paper explores the impact of occlusions in video action detection. We facilitatethis study by introducing five new benchmark datasets namely O-UCF and O-JHMDB consisting of synthetically controlled static/dynamic occlusions, OVIS-UCF and OVIS-JHMDB consisting of occlusions with realistic motions and Real-OUCF for occlusions in realistic-world scenarios. We formally confirm an intuitiveexpectation: existing models suffer a lot as occlusion severity is increased andexhibit different behaviours when occluders are static vs when they are moving.We discover several intriguing phenomenon emerging in neural nets: 1) transformerscan naturally outperform CNN models which might have even used occlusion as aform of data augmentation during training 2) incorporating symbolic-componentslike capsules to such backbones allows them to bind to occluders never even seenduring training and 3) Islands of agreement (similar to the ones hypothesized inHinton et Alâ€™s GLOM) can emerge in realistic images/videos without instance-levelsupervision, distillation or contrastive-based objectives(eg. video-textual training).Such emergent properties allow us to derive simple yet effective training recipeswhich lead to robust occlusion models inductively satisfying the first two stages ofthe binding mechanism (grouping/segregation). Models leveraging these recipesoutperform existing video action-detectors under occlusion by 32.3% on O-UCF,32.7% on O-JHMDB & 2.6% on Real-OUCF in terms of the vMAP metric. The code for this work has been released at https: //github.com/rajatmodi62/OccludedActionBenchmark.

        ----

        ## [2504] Black-box Backdoor Defense via Zero-shot Image Purification

        **Authors**: *Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, Ninghao Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b36554b97da741b1c48c9de05c73993e-Abstract-Conference.html)

        **Abstract**:

        Backdoor attacks inject poisoned samples into the training data, resulting in the misclassification of the poisoned input during a model's deployment. Defending against such attacks is challenging, especially for real-world black-box models where only query access is permitted. In this paper, we propose a novel defense framework against backdoor attacks through Zero-shot Image Purification (ZIP). Our framework can be applied to poisoned models without requiring internal information about the model or any prior knowledge of the clean/poisoned samples. Our defense framework involves two steps. First, we apply a linear transformation (e.g., blurring) on the poisoned image to destroy the backdoor pattern. Then, we use a pre-trained diffusion model to recover the missing semantic information removed by the transformation. In particular, we design a new reverse process by using the transformed image to guide the generation of high-fidelity purified images, which works in zero-shot settings. We evaluate our ZIP framework on multiple datasets with different types of attacks. Experimental results demonstrate the superiority of our ZIP framework compared to state-of-the-art backdoor defense baselines. We believe that our results will provide valuable insights for future defense methods for black-box models. Our code is available at https://github.com/sycny/ZIP.

        ----

        ## [2505] Beyond NTK with Vanilla Gradient Descent: A Mean-Field Analysis of Neural Networks with Polynomial Width, Samples, and Time

        **Authors**: *Arvind Mahankali, Haochen Zhang, Kefan Dong, Margalit Glasgow, Tengyu Ma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3748cdac932d91f0a51a37db90dec50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3748cdac932d91f0a51a37db90dec50-Abstract-Conference.html)

        **Abstract**:

        Despite recent theoretical progress on the non-convex optimization of two-layer neural networks, it is still an open question whether gradient descent on neural networks without unnatural modifications can achieve better sample complexity than kernel methods. This paper provides a clean mean-field analysis of projected gradient flow on polynomial-width two-layer neural networks. Different from prior works, our analysis does not require unnatural modifications of the optimization algorithm. We prove that with sample size $n = O(d^{3.1})$ where $d$ is the dimension of the inputs, the network trained with projected gradient flow converges in polynomial time to a non-trivial error that is not achievable by kernel methods using $n \ll  d^4$ samples, hence demonstrating a clear separation between unmodified gradient descent and NTK. As a corollary, we show that projected gradient descent with a positive learning rate and a polynomial number of iterations converges to low error with the same sample complexity.

        ----

        ## [2506] Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding

        **Authors**: *Zhejun Zhang, Alexander Liniger, Christos Sakaridis, Fisher Yu, Luc Van Gool*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b37c2e26b75ee02fcabd65a2a0367136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b37c2e26b75ee02fcabd65a2a0367136-Abstract-Conference.html)

        **Abstract**:

        The real-world deployment of an autonomous driving system requires its components to run on-board and in real-time, including the motion prediction module that predicts the future trajectories of surrounding traffic participants. Existing agent-centric methods have demonstrated outstanding performance on public benchmarks. However, they suffer from high computational overhead and poor scalability as the number of agents to be predicted increases. To address this problem, we introduce the K-nearest neighbor attention with relative pose encoding (KNARPE), a novel attention mechanism allowing the pairwise-relative representation to be used by Transformers. Then, based on KNARPE we present the Heterogeneous Polyline Transformer with Relative pose encoding (HPTR), a hierarchical framework enabling asynchronous token update during the online inference. By sharing contexts among agents and reusing the unchanged contexts, our approach is as efficient as scene-centric methods, while performing on par with state-of-the-art agent-centric methods. Experiments on Waymo and Argoverse-2 datasets show that HPTR achieves superior performance among end-to-end methods that do not apply expensive post-processing or model ensembling. The code is available at https://github.com/zhejz/HPTR.

        ----

        ## [2507] Customizable Image Synthesis with Multiple Subjects

        **Authors**: *Zhiheng Liu, Yifei Zhang, Yujun Shen, Kecheng Zheng, Kai Zhu, Ruili Feng, Yu Liu, Deli Zhao, Jingren Zhou, Yang Cao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html)

        **Abstract**:

        Synthesizing images with user-specified subjects has received growing attention due to its practical applications. Despite the recent success in single subject customization, existing algorithms suffer from high training cost and low success rate along with increased number of subjects. Towards controllable image synthesis with multiple subjects as the constraints, this work studies how to efficiently represent a particular subject as well as how to appropriately compose different subjects. We find that the text embedding regarding the subject token already serves as a simple yet effective representation that supports arbitrary combinations without any model tuning. Through learning a residual on top of the base embedding, we manage to robustly shift the raw subject to the customized subject given various text conditions. We then propose to employ layout, a very abstract and easy-to-obtain prior, as the spatial guidance for subject arrangement. By rectifying the activations in the cross-attention map, the layout appoints and separates the location of different subjects in the image, significantly alleviating the interference across them. Using cross-attention map as the intermediary, we could strengthen the signal of target subjects and weaken the signal of irrelevant subjects within a certain region, significantly alleviating the interference across subjects. Both qualitative and quantitative experimental results demonstrate our superiority over state-of-the-art alternatives under a variety of settings for multi-subject customization.

        ----

        ## [2508] How do Minimum-Norm Shallow Denoisers Look in Function Space?

        **Authors**: *Chen Zeno, Greg Ongie, Yaniv Blumenfeld, Nir Weinberger, Daniel Soudry*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b39cef2ef90591cffdc9c674cd55bebe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b39cef2ef90591cffdc9c674cd55bebe-Abstract-Conference.html)

        **Abstract**:

        Neural network (NN) denoisers are an essential building block in many common tasks, ranging from image reconstruction to image generation. However, the success of these models is not well understood from a theoretical perspective. In this paper, we aim to characterize the functions realized by shallow ReLU NN denoisers --- in the common theoretical setting of interpolation (i.e., zero training loss) with a minimal representation cost (i.e., minimal $\ell^2$ norm weights). First, for univariate data, we derive a closed form for the NN denoiser function, find it is contractive toward the clean data points, and prove it generalizes better than the empirical MMSE estimator at a low noise level. Next, for multivariate data, we find the NN denoiser functions in a closed form under various geometric assumptions on the training data: data contained in a low-dimensional subspace, data contained in a union of one-sided rays, or several types of simplexes. These functions decompose into a sum of simple rank-one piecewise linear interpolations aligned with edges and/or faces connecting training samples. We empirically verify this alignment phenomenon on synthetic data and real images.

        ----

        ## [2509] Lookup Table meets Local Laplacian Filter: Pyramid Reconstruction Network for Tone Mapping

        **Authors**: *Feng Zhang, Ming Tian, Zhiqiang Li, Bin Xu, Qingbo Lu, Changxin Gao, Nong Sang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3a08d179347e33414badadf100e4e8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3a08d179347e33414badadf100e4e8d-Abstract-Conference.html)

        **Abstract**:

        Tone mapping aims to convert high dynamic range (HDR) images to low dynamic range (LDR) representations, a critical task in the camera imaging pipeline. In recent years, 3-Dimensional LookUp Table (3D LUT) based methods have gained attention due to their ability to strike a favorable balance between enhancement performance and computational efficiency. However, these methods often fail to deliver satisfactory results in local areas since the look-up table is a global operator for tone mapping, which works based on pixel values and fails to incorporate crucial local information. To this end, this paper aims to address this issue by exploring a novel strategy that integrates global and local operators by utilizing closed-form Laplacian pyramid decomposition and reconstruction. Specifically, we employ image-adaptive 3D LUTs to manipulate the tone in the low-frequency image by leveraging the specific characteristics of the frequency information. Furthermore, we utilize local Laplacian filters to refine the edge details in the high-frequency components in an adaptive manner. Local Laplacian filters are widely used to preserve edge details in photographs, but their conventional usage involves manual tuning and fixed implementation within camera imaging pipelines or photo editing tools. We propose to learn parameter value maps progressively for local Laplacian filters from annotated data using a lightweight network. Our model achieves simultaneous global tone manipulation and local edge detail preservation in an end-to-end manner. Extensive experimental results on two benchmark datasets demonstrate that the proposed method performs favorably against state-of-the-art methods.

        ----

        ## [2510] Masked Image Residual Learning for Scaling Deeper Vision Transformers

        **Authors**: *Guoxi Huang, Hongtao Fu, Adrian G. Bors*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3bac97f3227c52c0179a6d967480867-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3bac97f3227c52c0179a6d967480867-Abstract-Conference.html)

        **Abstract**:

        Deeper Vision Transformers (ViTs) are more challenging to train. We expose a degradation problem in deeper layers of ViT when using masked image modeling (MIM) for pre-training.To ease the training of deeper ViTs, we introduce a self-supervised learning framework called  $\textbf{M}$asked $\textbf{I}$mage $\textbf{R}$esidual $\textbf{L}$earning ($\textbf{MIRL}$), which significantly alleviates the degradation problem, making scaling ViT along depth a promising direction for performance upgrade. We reformulate the pre-training objective for deeper layers of ViT as learning to recover the residual of the masked image.We provide extensive empirical evidence showing that deeper ViTs can be effectively optimized using MIRL and easily gain accuracy from increased depth. With the same level of computational complexity as ViT-Base and ViT-Large, we instantiate $4.5{\times}$ and $2{\times}$ deeper ViTs, dubbed ViT-S-54 and ViT-B-48.The deeper ViT-S-54, costing $3{\times}$ less than ViT-Large, achieves performance on par with ViT-Large.ViT-B-48 achieves 86.2\% top-1 accuracy on ImageNet. On one hand, deeper ViTs pre-trained with MIRL exhibit excellent generalization capabilities on downstream tasks, such as object detection and semantic segmentation. On the other hand, MIRL demonstrates high pre-training efficiency. With less pre-training time, MIRL yields competitive performance compared to other approaches.

        ----

        ## [2511] Revisiting Area Convexity: Faster Box-Simplex Games and Spectrahedral Generalizations

        **Authors**: *Arun Jambulapati, Kevin Tian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3bec3f5ad96055b7f60c93edc3606c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3bec3f5ad96055b7f60c93edc3606c8-Abstract-Conference.html)

        **Abstract**:

        We investigate area convexity [Sherman17], a mysterious tool introduced to tackle optimization problems under the challenging $\ell_\infty$ geometry. We develop a deeper understanding of its relationship with conventional analyses of extragradient methods [Nemirovski04, Nesterov07]. We also give improved solvers for the subproblems required by variants of the [Sherman17] algorithm, designed through the lens of relative smoothness [BBT17, LFN18}.Leveraging these new tools, we give a state-of-the-art first-order algorithm for solving box-simplex games (a primal-dual formulation of $\ell_\infty$ regression) in a $d \times n$ matrix with bounded rows, using $O(\log d \cdot \epsilon^{-1})$ matrix-vector queries. As a consequence, we obtain improved complexities for approximate maximum flow, optimal transport, min-mean-cycle, and other basic combinatorial optimization problems. We also develop a near-linear time algorithm for a matrix generalization of box-simplex games, capturing a family of problems closely related to semidefinite programs recently used as subroutines in robust statistics and numerical linear algebra.

        ----

        ## [2512] Adversarial Learning for Feature Shift Detection and Correction

        **Authors**: *Míriam Barrabés, Daniel Mas Montserrat, Margarita Geleta, Xavier Giró-i-Nieto, Alexander G. Ioannidis*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3cd64ddad0a28da0f28a0e03a73ea7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3cd64ddad0a28da0f28a0e03a73ea7d-Abstract-Conference.html)

        **Abstract**:

        Data shift is a phenomenon present in many real-world applications, and while there are multiple methods attempting to detect shifts, the task of localizing and correcting the features originating such shifts has not been studied in depth. Feature shifts can occur in many datasets, including in multi-sensor data, where some sensors are malfunctioning, or in tabular and structured data, including biomedical, financial, and survey data, where faulty standardization and data processing pipelines can lead to erroneous features. In this work, we explore using the principles of adversarial learning, where the information from several discriminators trained to distinguish between two distributions is used to both detect the corrupted features and fix them in order to remove the distribution shift between datasets. We show that mainstream supervised classifiers, such as random forest or gradient boosting trees, combined with simple iterative heuristics, can localize and correct feature shifts, outperforming current statistical and neural network-based techniques. The code is available at https://github.com/AI-sandbox/DataFix.

        ----

        ## [2513] General Munchausen Reinforcement Learning with Tsallis Kullback-Leibler Divergence

        **Authors**: *Lingwei Zhu, Zheng Chen, Matthew Schlegel, Martha White*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3e866c228f8f4ea18021ae63aea5453-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3e866c228f8f4ea18021ae63aea5453-Abstract-Conference.html)

        **Abstract**:

        Many policy optimization approaches in reinforcement learning incorporate a Kullback-Leilbler (KL) divergence to the previous policy, to prevent the policy from changing too quickly. This idea was initially proposed in a seminal paper on Conservative Policy Iteration, with approximations given by algorithms like TRPO and Munchausen Value Iteration (MVI). We continue this line of work by investigating a generalized KL divergence---called the Tsallis KL divergence. Tsallis KL defined by the $q$-logarithm is a strict generalization, as $q = 1$ corresponds to the standard KL divergence; $q > 1$ provides a range of new options. We characterize the types of policies learned under the Tsallis KL, and motivate when $q >1$ could be beneficial.  To obtain a practical algorithm that incorporates Tsallis KL regularization, we extend MVI, which is one of the simplest approaches to incorporate KL regularization. We show that this generalized MVI($q$) obtains significant improvements over the standard MVI($q = 1$) across 35 Atari games.

        ----

        ## [2514] Residual Alignment: Uncovering the Mechanisms of Residual Networks

        **Authors**: *Jianing Li, Vardan Papyan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b3f48945f6fb402b4b5cdcf490e72847-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b3f48945f6fb402b4b5cdcf490e72847-Abstract-Conference.html)

        **Abstract**:

        The ResNet architecture has been widely adopted in deep learning due to its significant boost to performance through the use of simple skip connections, yet the underlying mechanisms leading to its success remain largely unknown. In this paper, we conduct a thorough empirical study of the ResNet architecture in classification tasks by linearizing its constituent residual blocks using Residual Jacobians and measuring their singular value decompositions. Our measurements ([code](https://colab.research.google.com/drive/1yKjEg2yF616tnZFAfuN0aQ-E9v3JmyjN?usp=sharing)) reveal a process called Residual Alignment (RA) characterized by four properties:- **(RA1):** intermediate representations of a given input are *equispaced* on a *line*, embedded in high dimensional space, as observed by Gai and Zhang [2021];- **(RA2):** top left and right singular vectors of Residual Jacobians align with each other and across different depths;- **(RA3):** Residual Jacobians are at most rank $C$ for fully-connected ResNets, where $C$ is the number of classes; and- **(RA4):** top singular values of Residual Jacobians scale inversely with depth.RA consistently occurs in models that generalize well, in both fully-connected and convolutional architectures, across various depths and widths, for varying numbers of classes, on all tested benchmark datasets, but ceases to occur once the skip connections are removed. It also provably occurs in a novel mathematical model we propose. This phenomenon reveals a strong alignment between residual branches of a ResNet (RA2+4), imparting a highly rigid geometric structure to the intermediate representations as they progress *linearly* through the network (RA1) up to the final layer, where they undergo Neural Collapse.

        ----

        ## [2515] Globally injective and bijective neural operators

        **Authors**: *Takashi Furuya, Michael Puthawala, Matti Lassas, Maarten V. de Hoop*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html)

        **Abstract**:

        Recently there has been great interest in operator learning, where networks learn operators between function spaces from an essentially infinite-dimensional perspective. In this work we present results for when the operators learned by these networks are injective and surjective. As a warmup, we combine prior work in both the finite-dimensional ReLU and operator learning setting by giving sharp conditions under which ReLU layers with linear neural operators are injective. We then consider the case when the activation function is pointwise bijective and obtain sufficient conditions for the layer to be injective. We remark that this question, while trivial in the finite-rank setting, is subtler in the infinite-rank setting and is proven using tools from Fredholm theory. Next, we prove that our supplied injective neural operators are universal approximators and that their implementation, with finite-rank neural networks, are still injective. This ensures that injectivity is not 'lost' in the transcription from analytical operators to their finite-rank implementation with networks. Finally, we conclude with an increase in abstraction and consider general conditions when subnetworks, which may have many layers, are injective and surjective and provide an exact inversion from a 'linearization.â€™ This section uses general arguments from Fredholm theory and Leray-Schauder degree theory for non-linear integral equations to analyze the mapping properties of neural operators in function spaces. These results apply to subnetworks formed from the layers considered in this work, under natural conditions. We believe that our work has applications in Bayesian uncertainty quantification where injectivity enables likelihood estimation and in inverse problems where surjectivity and injectivity corresponds to existence and uniqueness of the solutions, respectively.

        ----

        ## [2516] On the Convergence of CART under Sufficient Impurity Decrease Condition

        **Authors**: *Rahul Mazumder, Haoyue Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b418964bafb4fdd9aef9017301323a8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b418964bafb4fdd9aef9017301323a8a-Abstract-Conference.html)

        **Abstract**:

        The decision tree is a flexible machine-learning model that finds its success in numerous applications. It is usually fitted in a recursively greedy manner using CART. In this paper, we study the convergence rate of CART under a regression setting. First, we prove an upper bound on the prediction error of CART under a sufficient impurity decrease (SID) condition \cite{chi2020asymptotic} -- our result is an improvement over the known result by \cite{chi2020asymptotic} under a similar assumption. We show via examples that this error bound cannot be further improved by more than a constant or a log factor. Second, we introduce a few easy-to-check sufficient conditions of the SID condition. In particular, we show that the SID condition can be satisfied by an additive model when the component functions satisfy a ``locally reverse Poincare inequality". We discuss a few familiar function classes in non-parametric estimation to demonstrate the usefulness of this conception.

        ----

        ## [2517] PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis

        **Authors**: *Ali TehraniJamsaz, Quazi Ishtiaque Mahmud, Le Chen, Nesreen K. Ahmed, Ali Jannesari*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b41907dd4df5c60f86216b73fe0c7465-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b41907dd4df5c60f86216b73fe0c7465-Abstract-Conference.html)

        **Abstract**:

        The remarkable growth and significant success of machine learning have expanded its applications into programming languages and program analysis. However, a key challenge in adopting the latest machine learning methods is the representation of programming languages which has a direct impact on the ability of machine learning methods to reason about programs. The absence of numerical awareness, aggregate data structure information, and improper way of presenting variables in previous representation works have limited their performances.  To overcome the limitations and challenges of current program representations, we propose a novel graph-based program representation called PERFOGRAPH. PERFOGRAPH can capture numerical information and the aggregate data structure by introducing new nodes and edges. Furthermore, we propose an adapted embedding method to incorporate numerical awareness.These enhancements make PERFOGRAPH a highly flexible and scalable representation that can effectively capture programs' intricate dependencies and semantics. Consequently, it serves as a powerful tool for various applications such as program analysis, performance optimization, and parallelism discovery. Our experimental results demonstrate that PERFOGRAPH outperforms existing representations and sets new state-of-the-art results by reducing the error rate by 7.4% (AMD dataset) and 10% (NVIDIA dataset) in the well-known Device Mapping challenge. It also sets new state-of-the-art results in various performance optimization tasks like Parallelism Discovery and Numa and Prefetchers Configuration prediction.

        ----

        ## [2518] Penalising the biases in norm regularisation enforces sparsity

        **Authors**: *Etienne Boursier, Nicolas Flammarion*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b444ad72520a5f5c467343be88e352ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b444ad72520a5f5c467343be88e352ed-Abstract-Conference.html)

        **Abstract**:

        Controlling the parameters' norm often yields good generalisation when training neural networks. Beyond simple intuitions, the relation between regularising parameters' norm and obtained estimators remains theoretically misunderstood. For one hidden ReLU layer networks with unidimensional data, this work shows the parameters' norm required to represent a function is given by the total variation of its second derivative, weighted by a $\sqrt{1+x^2}$ factor. Notably, this weighting factor disappears when the norm of bias terms is not regularised. The presence of this additional weighting factor is of utmost significance as it is shown to enforce the uniqueness and sparsity (in the number of kinks) of the minimal norm interpolator. Conversely, omitting the bias' norm  allows for non-sparse solutions.Penalising the bias terms in the regularisation, either explicitly or implicitly, thus leads to sparse estimators.

        ----

        ## [2519] Distributional Learning of Variational AutoEncoder: Application to Synthetic Data Generation

        **Authors**: *SeungHwan An, Jong-June Jeon*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b456a00e145ad56f6f251f79f8c8a7de-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b456a00e145ad56f6f251f79f8c8a7de-Abstract-Conference.html)

        **Abstract**:

        The Gaussianity assumption has been consistently criticized as a main limitation of the Variational Autoencoder (VAE) despite its efficiency in computational modeling. In this paper, we propose a new approach that expands the model capacity (i.e., expressive power of distributional family) without sacrificing the computational advantages of the VAE framework. Our VAE model's decoder is composed of an infinite mixture of asymmetric Laplace distribution, which possesses general distribution fitting capabilities for continuous variables. Our model is represented by a special form of a nonparametric M-estimator for estimating general quantile functions, and we theoretically establish the relevance between the proposed model and quantile estimation. We apply the proposed model to synthetic data generation, and particularly, our model demonstrates superiority in easily adjusting the level of data privacy.

        ----

        ## [2520] Optimistic Meta-Gradients

        **Authors**: *Sebastian Flennerhag, Tom Zahavy, Brendan O'Donoghue, Hado Philip van Hasselt, András György, Satinder Singh*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b46bc1449205888e1883f692aff1a252-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b46bc1449205888e1883f692aff1a252-Abstract-Conference.html)

        **Abstract**:

        We study the connection between gradient-based meta-learning and convex optimisation. We observe that gradient descent with momentum is a special case of meta-gradients, and building on recent results in optimisation, we prove convergence rates for meta learning in the single task setting. While a meta-learned update rule can yield faster convergence up to constant factor, it is not sufficient for acceleration. Instead, some form of optimism is required. We show that optimism in meta-learning can be captured through the recently proposed Bootstrapped Meta-Gradient (Flennerhag et. al., 2022) method, providing deeper insight into its underlying mechanics.

        ----

        ## [2521] Norm-guided latent space exploration for text-to-image generation

        **Authors**: *Dvir Samuel, Rami Ben-Ari, Nir Darshan, Haggai Maron, Gal Chechik*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b49213694c3e752252d62ca360b72a36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b49213694c3e752252d62ca360b72a36-Abstract-Conference.html)

        **Abstract**:

        Text-to-image diffusion models show great potential in synthesizing a large variety of concepts in new compositions and scenarios. However, the latent space of initial seeds is still not well understood and its structure was shown to impact the generation of various concepts. Specifically, simple operations like interpolation and finding the centroid of a set of seeds perform poorly when using standard Euclidean or spherical metrics in the latent space. This paper makes the observation that, in current training procedures, diffusion models observed inputs with a narrow range of norm values. This has strong implications for methods that rely on seed manipulation for image generation, with applications to few-shot and long-tail learning tasks. To address this issue, we propose a novel method for interpolating between two seeds and demonstrate that it defines a new non-Euclidean metric that takes into account a norm-based prior on seeds. We describe a simple yet efficient algorithm for approximating this interpolation procedure and use it to further define centroids in the latent seed space. We show that our new interpolation and centroid techniques significantly enhance the generation of rare concept images. This further leads to state-of-the-art performance on few-shot and long-tail benchmarks, improving prior approaches in terms of generation speed, image quality, and semantic content.

        ----

        ## [2522] Scale Alone Does not Improve Mechanistic Interpretability in Vision Models

        **Authors**: *Roland S. Zimmermann, Thomas Klein, Wieland Brendel*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4aadf04d6fde46346db455402860708-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4aadf04d6fde46346db455402860708-Abstract-Conference.html)

        **Abstract**:

        In light of the recent widespread adoption of AI systems, understanding the internal information processing of neural networks has become increasingly critical. Most recently, machine vision has seen remarkable progress by scaling neural networks to unprecedented levels in dataset and model size. We here ask whether this extraordinary increase in scale also positively impacts the field of mechanistic interpretability. In other words, has our understanding of the inner workings of scaled neural networks improved as well? We use a psychophysical paradigm to quantify one form of mechanistic interpretability for a diverse suite of nine models and find no scaling effect for interpretability - neither for model nor dataset size. Specifically, none of the investigated state-of-the-art models are easier to interpret than the GoogLeNet model from almost a decade ago. Latest-generation vision models appear even less interpretable than older architectures, hinting at a regression rather than improvement, with modern models sacrificing interpretability for accuracy. These results highlight the need for models explicitly designed to be mechanistically interpretable and the need for more helpful interpretability methods to increase our understanding of networks at an atomic level. We release a dataset containing more than 130'000 human responses from our psychophysical evaluation of 767 units across nine models. This dataset facilitates research on automated instead of human-based interpretability evaluations, which can ultimately be leveraged to directly optimize the mechanistic interpretability of models.

        ----

        ## [2523] The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications

        **Authors**: *Mirac Suzgun, Luke Melas-Kyriazi, Suproteem K. Sarkar, Scott Duke Kominers, Stuart M. Shieber*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4b02a09f2e6ad29fdbeb1386d68f4c4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4b02a09f2e6ad29fdbeb1386d68f4c4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Innovation is a major driver of economic and social development, and information about many kinds of innovation is embedded in semi-structured data from patents and patent applications. Though the impact and novelty of innovations expressed in patent data are difficult to measure through traditional means, machine learning offers a promising set of techniques for evaluating novelty, summarizing contributions, and embedding semantics. In this paper, we introduce the Harvard USPTO Patent Dataset (HUPD), a large-scale, well-structured, and multi-purpose corpus of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions of patent applications, not the final versions of granted patents, allowing us to study patentability at the time of filing using NLP methods for the first time. It is also novel in its inclusion of rich structured data alongside the text of patent filings: By providing each applicationâ€™s metadata along with all of its text fields, HUPD enables researchers to perform new sets of NLP tasks that leverage variation in structured covariates. As a case study on the types of research HUPD makes possible, we introduce a new task to the NLP community -- patent acceptance prediction. We additionally show the structured metadata provided in HUPD allows us to conduct explicit studies of concept shifts for this task. We find that performance on patent acceptance prediction decays when models trained in one context are evaluated on different innovation categories and over time. Finally, we demonstrate how HUPD can be used for three additional tasks: Multi-class classification of patent subject areas, language modeling, and abstractive summarization. Put together, our publicly-available dataset aims to advance research extending language and classification models to diverse and dynamic real-world data distributions.

        ----

        ## [2524] MEMTO: Memory-guided Transformer for Multivariate Time Series Anomaly Detection

        **Authors**: *Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4c898eb1fb556b8d871fbe9ead92256-Abstract-Conference.html)

        **Abstract**:

        Detecting anomalies in real-world multivariate time series data is challenging due to complex temporal dependencies and inter-variable correlations. Recently, reconstruction-based deep models have been widely used to solve the problem. However, these methods still suffer from an over-generalization issue and fail to deliver consistently high performance. To address this issue, we propose the MEMTO, a memory-guided Transformer using a reconstruction-based approach. It is designed to incorporate a novel memory module that can learn the degree to which each memory item should be updated in response to the input data. To stabilize the training procedure, we use a two-phase training paradigm which involves using K-means clustering for initializing memory items. Additionally, we introduce a bi-dimensional deviation-based detection criterion that calculates anomaly scores considering both input space and latent space. We evaluate our proposed method on five real-world datasets from diverse domains, and it achieves an average anomaly detection F1-score of 95.74%, significantly outperforming the previous state-of-the-art methods. We also conduct extensive experiments to empirically validate the effectiveness of our proposed model's key components.

        ----

        ## [2525] Minimax Risks and Optimal Procedures for Estimation under Functional Local Differential Privacy

        **Authors**: *Bonwoo Lee, Jeongyoun Ahn, Cheolwoo Park*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4dde7f1bc45bf9c0fda8db8f272b758-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4dde7f1bc45bf9c0fda8db8f272b758-Abstract-Conference.html)

        **Abstract**:

        As concerns about data privacy continue to grow, differential privacy (DP) has emerged as a fundamental concept that aims to guarantee privacy by ensuring individuals' indistinguishability in data analysis. Local differential privacy (LDP) is a rigorous type of DP that requires individual data to be privatized before being sent to the collector, thus removing the need for a trusted third party to collect data. Among the numerous (L)DP-based approaches, functional DP has gained considerable attention in the DP community because it connects DP to statistical decision-making by formulating it as a hypothesis-testing problem and also exhibits Gaussian-related properties. However, the utility of privatized data is generally lower than that of non-private data, prompting research into optimal mechanisms that maximize the statistical utility for given privacy constraints. In this study, we investigate how functional LDP preserves the statistical utility by analyzing minimax risks of univariate mean estimation as well as nonparametric density estimation. We leverage the contraction property of functional LDP mechanisms and classical information-theoretical bounds to derive private minimax lower bounds. Our theoretical study reveals that it is possible to establish an interpretable, continuous balance between the statistical utility and privacy level, which has not been achieved under the $\epsilon$-LDP framework. Furthermore, we suggest minimax optimal mechanisms based on Gaussian LDP (a type of functional LDP) that achieve the minimax upper bounds and show via a numerical study that they are superior to the counterparts derived under $\epsilon$-LDP. The theoretical and empirical findings of this work suggest that Gaussian LDP should be considered a reliable standard for LDP.

        ----

        ## [2526] Switching Autoregressive Low-rank Tensor Models

        **Authors**: *Hyun Dong Lee, Andrew Warrington, Joshua I. Glaser, Scott W. Linderman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b4e3fea367538ea6b1b5ba6ebf5c39a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b4e3fea367538ea6b1b5ba6ebf5c39a8-Abstract-Conference.html)

        **Abstract**:

        An important problem in time-series analysis is modeling systems with time-varying dynamics.  Probabilistic models with joint continuous and discrete latent states offer interpretable, efficient, and experimentally useful descriptions of such data.  Commonly used models include autoregressive hidden Markov models (ARHMMs) and switching linear dynamical systems (SLDSs), each with its own advantages and disadvantages.  ARHMMs permit exact inference and easy parameter estimation, but are parameter intensive when modeling long dependencies, and hence are prone to overfitting.  In contrast, SLDSs can capture long-range dependencies in a parameter efficient way through Markovian latent dynamics, but present an intractable likelihood and a challenging parameter estimation task.  In this paper, we propose switching autoregressive low-rank tensor SALT models, which retain the advantages of both approaches while ameliorating the weaknesses.  SALT parameterizes the tensor of an ARHMM with a low-rank factorization to control the number of parameters and allow longer range dependencies without overfitting.  We prove theoretical and discuss practical connections between SALT, linear dynamical systems, and SLDSs.  We empirically demonstrate quantitative advantages of SALT models on a range of simulated and real prediction tasks, including behavioral and neural datasets.  Furthermore, the learned low-rank tensor provides novel insights into temporal dependencies within each discrete state.

        ----

        ## [2527] From Tempered to Benign Overfitting in ReLU Neural Networks

        **Authors**: *Guy Kornowski, Gilad Yehudai, Ohad Shamir*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b52e8c6c1a798fed53ac2e6a5e23ddc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b52e8c6c1a798fed53ac2e6a5e23ddc8-Abstract-Conference.html)

        **Abstract**:

        Overparameterized neural networks (NNs) are observed to generalize well even when trained to perfectly fit noisy data. This phenomenon motivated a large body of work on "benign overfitting", where interpolating predictors achieve near-optimal performance. Recently, it was conjectured and empirically observed that the behavior of NNs is often better described as "tempered overfitting", where the performance is non-optimal yet also non-trivial, and degrades as a function of the noise level. However, a theoretical justification of this claim for non-linear NNs has been lacking so far. In this work, we provide several results that aim at bridging these complementing views. We study a simple classification setting with 2-layer ReLU NNs, and prove that under various assumptions, the type of overfitting transitions from tempered in the extreme case of one-dimensional data, to benign in high dimensions. Thus, we show that the input dimension has a crucial role on the overfitting profile in this setting, which we also validate empirically for intermediate dimensions. Overall, our results shed light on the intricate connections between the dimension, sample size, architecture and training algorithm on the one hand, and the type of resulting overfitting on the other hand.

        ----

        ## [2528] Tree-Rings Watermarks: Invisible Fingerprints for Diffusion Images

        **Authors**: *Yuxin Wen, John Kirchenbauer, Jonas Geiping, Tom Goldstein*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b54d1757c190ba20dbc4f9e4a2f54149-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b54d1757c190ba20dbc4f9e4a2f54149-Abstract-Conference.html)

        **Abstract**:

        Watermarking the outputs of generative models is a crucial technique for tracing copyright and preventing potential harm from AI-generated content. In this paper, we introduce a novel technique called Tree-Ring Watermarking that robustly fingerprints diffusion model outputs.  Unlike existing methods that perform post-hoc modifications to images after sampling, Tree-Ring Watermarking subtly influences the entire sampling process, resulting in a model fingerprint that is invisible to humans. The watermark embeds a pattern into the initial noise vector used for sampling. These patterns are structured in Fourier space so that they are invariant to convolutions, crops, dilations, flips, and rotations.  After image generation, the watermark signal is detected by inverting the diffusion process to retrieve the noise vector, which is then checked for the embedded signal.  We demonstrate that this technique can be easily applied to arbitrary diffusion models, including text-conditioned Stable Diffusion, as a plug-in with negligible loss in FID. Our watermark is semantically hidden in the image space and is far more robust than watermarking alternatives that are currently deployed.

        ----

        ## [2529] MVDoppler: Unleashing the Power of Multi-View Doppler for MicroMotion-based Gait Classification

        **Authors**: *Soheil Hor, Shubo Yang, Jaeho Choi, Amin Arbabian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5727c1bab903e0ff21cec84a9a7f5a6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5727c1bab903e0ff21cec84a9a7f5a6-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Modern perception systems rely heavily on high-resolution cameras, LiDARs, and advanced deep neural networks, enabling exceptional performance across various applications. However, these optical systems predominantly depend on geometric features and shapes of objects, which can be challenging to capture in long-range perception applications. To overcome this limitation, alternative approaches such as Doppler-based perception using high-resolution radars have been proposed. Doppler-based systems are capable of measuring micro-motions of targets remotely and with very high precision. When compared to geometric features, the resolution of micro-motion features exhibits significantly greater resilience to the influence of distance. However, the true potential of Doppler-based perception has yet to be fully realized due to several factors. These include the unintuitive nature of Doppler signals, the limited availability of public Doppler datasets, and the current datasets' inability to capture the specific co-factors that are unique to Doppler-based perception, such as the effect of the radar's observation angle and the target's motion trajectory.This paper introduces a new large multi-view Doppler dataset together with baseline perception models for micro-motion-based gait analysis and classification. The dataset captures the impact of the subject's walking trajectory and radar's observation angle on the classification performance. Additionally, baseline multi-view data fusion techniques are provided to mitigate these effects. This work demonstrates that sub-second micro-motion snapshots can be sufficient for reliable detection of hand movement patterns and even changes in a pedestrian's walking behavior when distracted by their phone. Overall, this research not only showcases the potential of Doppler-based perception, but also offers valuable solutions to tackle its fundamental challenges.

        ----

        ## [2530] Understanding and Improving Ensemble Adversarial Defense

        **Authors**: *Yian Deng, Tingting Mu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b589d92785e39486e978fa273d0dc343-Abstract-Conference.html)

        **Abstract**:

        The strategy of ensemble has become popular in adversarial defense, which trains multiple base classifiers to defend against adversarial attacks in a cooperative manner. Despite the empirical success, theoretical explanations on why an ensemble of adversarially trained classifiers is more robust than single ones remain unclear. To fill in this gap, we develop a new error theory dedicated to understanding ensemble adversarial defense,  demonstrating a provable 0-1 loss reduction on challenging sample sets in adversarial defense scenarios. Guided by this theory, we propose an effective approach to improve ensemble adversarial defense, named interactive global adversarial training (iGAT). The proposal includes (1) a probabilistic distributing rule that selectively allocates to different base classifiers adversarial examples that are globally challenging to the ensemble, and (2) a regularization term to rescue the severest weaknesses of the base classifiers. Being tested over various existing ensemble adversarial defense techniques,  iGAT is capable of boosting their performance by up to 17\%  evaluated using  CIFAR10 and CIFAR100 datasets under both white-box and black-box attacks.

        ----

        ## [2531] Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions

        **Authors**: *Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5a801e6bc4f4ffa3e6786518a324488-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5a801e6bc4f4ffa3e6786518a324488-Abstract-Conference.html)

        **Abstract**:

        Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training  (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.

        ----

        ## [2532] A Massive Scale Semantic Similarity Dataset of Historical English

        **Authors**: *Emily Silcock, Abhishek Arora, Melissa Dell*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5ae304ecd18c5d4ac4a011ab086ba23-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5ae304ecd18c5d4ac4a011ab086ba23-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        A diversity of tasks use language models trained on semantic similarity data. While there are a variety of datasets that capture semantic similarity, they are either constructed from modern web data or are relatively small datasets created in the past decade by human annotators. This study utilizes a novel source, newly digitized articles from off-copyright, local U.S. newspapers, to assemble a massive-scale semantic similarity dataset spanning 70 years from 1920 to 1989 and containing nearly 400M positive semantic similarity pairs. Historically, around half of articles in U.S. local newspapers came from newswires like the Associated Press. While local papers reproduced articles from the newswire, they wrote their own headlines, which form abstractive summaries of the associated articles. We associate articles and their headlines by exploiting document layouts and language understanding. We then use deep neural methods to detect which articles are from the same underlying source, in the presence of substantial noise and abridgement. The headlines of reproduced articles form positive semantic similarity pairs. The resulting publicly available HEADLINES dataset is significantly larger than most existing semantic similarity datasets and covers a much longer span of time. It will facilitate the application of contrastively trained semantic similarity models to a variety of tasks, including the study of semantic change across space and time.

        ----

        ## [2533] Joint Prompt Optimization of Stacked LLMs using Variational Inference

        **Authors**: *Alessandro Sordoni, Eric Yuan, Marc-Alexandre Côté, Matheus Pereira, Adam Trischler, Ziang Xiao, Arian Hosseini, Friederike Niedtner, Nicolas Le Roux*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5afe13494c825089b1e3944fdaba212-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) can be seen as atomic units of computation mapping sequences to a distribution over sequences. Thus, they can be seen as stochastic language layers in a language network, where the learnable parameters are the natural language prompts at each layer. By stacking two such layers and feeding the output of one layer to the next, we obtain a Deep Language Network (DLN). We first show how to effectively perform prompt optimization for a 1-Layer language network (DLN-1). Then, we present an extension that applies to 2-layer DLNs (DLN-2), where two prompts must be learned. The key idea is to consider the output of the first layer as a latent variable, which requires inference, and prompts to be learned as the parameters of the generative distribution. We first test the effectiveness of DLN-1 in multiple reasoning and natural language understanding tasks. Then, we show that DLN-2 can reach higher performance than a single layer, showing promise that we might reach comparable performance to GPT-4, even when each LLM in the network is smaller and less powerful.

        ----

        ## [2534] On the Properties of Kullback-Leibler Divergence Between Multivariate Gaussian Distributions

        **Authors**: *Yufeng Zhang, Jialu Pan, Li Ken Li, Wanwei Liu, Zhenbang Chen, Xinwang Liu, Ji Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5b4d92374323c53c24bbbc8ee0e715c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5b4d92374323c53c24bbbc8ee0e715c-Abstract-Conference.html)

        **Abstract**:

        Kullback-Leibler (KL) divergence is one of the most important measures to calculate the difference between probability distributions. In this paper, we theoretically study several properties of KL divergence between multivariate Gaussian distributions. Firstly, for any two $n$-dimensional Gaussian distributions $\mathcal{N}_1$ and $\mathcal{N}_2$, we prove that when $KL(\mathcal{N}_2||\mathcal{N}_1)\leq \varepsilon\ (\varepsilon>0)$ the supremum of $KL(\mathcal{N}_1||\mathcal{N}_2)$ is $(1/2)\left((-W_{0}(-e^{-(1+2\varepsilon)}))^{-1}+\log(-W_{0}(-e^{-(1+2\varepsilon)})) -1 \right)$, where $W_0$ is the principal branch of Lambert $W$ function.For small $\varepsilon$, the supremum is $\varepsilon + 2\varepsilon^{1.5} + O(\varepsilon^2)$. This quantifies the approximate symmetry of small KL divergence between Gaussian distributions. We further derive the infimum of $KL(\mathcal{N}_1||\mathcal{N}_2)$ when $KL(\mathcal{N}_2||\mathcal{N}_1)\geq M\ (M>0)$. We give the conditions when the supremum and infimum can be attained. Secondly, for any three $n$-dimensional Gaussian distributions $\mathcal{N}_1$, $\mathcal{N}_2$, and $\mathcal{N}_3$, we theoretically show that an upper bound of $KL(\mathcal{N}_1||\mathcal{N}_3)$ is $3\varepsilon_1+3\varepsilon_2+2\sqrt{\varepsilon_1\varepsilon_2}+o(\varepsilon_1)+o(\varepsilon_2)$ when $KL(\mathcal{N}_1||\mathcal{N}_2)\leq \varepsilon_1$ and $KL(\mathcal{N}_2||\mathcal{N}_3)\leq \varepsilon_2$ ($\varepsilon_1,\varepsilon_2\ge 0$). This reveals that KL divergence between Gaussian distributions follows a relaxed triangle inequality. Note that, all these bounds in the theorems presented in this work are independent of the dimension $n$. Finally, we discuss several applications of our theories in deep learning, reinforcement learning, and sample complexity research.

        ----

        ## [2535] Implicit Bias of (Stochastic) Gradient Descent for Rank-1 Linear Neural Network

        **Authors**: *Bochen Lyu, Zhanxing Zhu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html)

        **Abstract**:

        Studying the implicit bias of gradient descent (GD) and stochastic gradient descent (SGD) is critical to unveil the underlying mechanism of deep learning. Unfortunately, even for standard linear networks in regression setting, a comprehensive characterization of the implicit bias is still an open problem. This paper proposes to investigate a new proxy model of standard linear network,  rank-1 linear network, where each weight matrix is parameterized as a rank-1 form. For over-parameterized regression problem, we  precisely analyze the implicit bias of GD and SGD---by identifying a “potential” function such that GD converges to its minimizer constrained by zero training error (i.e., interpolation solution), and further characterizing the role of the noise introduced by  SGD in perturbing the form of this potential. Our results explicitly connect the depth of the network and the initialization with the implicit bias of GD and SGD. Furthermore, we emphasize a new implicit bias of SGD jointly induced by stochasticity and over-parameterization, which can reduce the dependence of the SGD's solution on the initialization. Our findings regarding the implicit bias are different from that of a recently popular model, the diagonal linear network. We highlight that the induced bias of our rank-1 model is more consistent with standard linear network while the diagonal one is not. This suggests that the proposed rank-1 linear network might be a plausible proxy for standard linear net.

        ----

        ## [2536] AdaPlanner: Adaptive Planning from Feedback with Language Models

        **Authors**: *Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, Chao Zhang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) have recently demonstrated the potential in acting as autonomous agents for sequential decision-making tasks. However, most existing methods either take actions greedily without planning or rely on static plans that are not adaptable to environmental feedback. Consequently, the sequential decision-making performance of LLM agents degenerates with problem complexity and plan horizons increase. We propose a closed-loop approach, AdaPlanner, which allows the LLM agent to refine its self-generated plan adaptively in response to environmental feedback. In AdaPlanner, the LLM agent adaptively refines its plan from feedback with both in-plan and out-of-plan refinement strategies. To mitigate hallucination, we develop a code-style LLM prompt structure that facilitates plan generation across a variety of tasks, environments, and agent capabilities. Furthermore, we propose a skill discovery mechanism that leverages successful plans as few-shot exemplars, enabling the agent to plan and refine with fewer task demonstrations. Our experiments in the ALFWorld and MiniWoB++ environments demonstrate that AdaPlanner outperforms state-of-the-art baselines by 3.73% and 4.11% while utilizing 2x and 600x fewer samples, respectively. The implementation of AdaPlanner is available at https://github.com/haotiansun14/AdaPlanner.

        ----

        ## [2537] Fairness Aware Counterfactuals for Subgroups

        **Authors**: *Loukas Kavouras, Konstantinos Tsopelas, Giorgos Giannopoulos, Dimitris Sacharidis, Eleni Psaroudaki, Nikolaos Theologitis, Dimitrios Rontogiannis, Dimitris Fotakis, Ioannis Z. Emiris*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b60161e93f3e0e4207081a3b4ef5e8d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b60161e93f3e0e4207081a3b4ef5e8d8-Abstract-Conference.html)

        **Abstract**:

        In this work, we present Fairness Aware Counterfactuals for Subgroups (FACTS), a framework for auditing subgroup fairness through counterfactual explanations. We start with revisiting (and generalizing) existing notions and introducing new, more refined notions of subgroup fairness. We aim to (a) formulate different aspects of the difficulty of individuals in certain subgroups to achieve recourse, i.e. receive the desired outcome, either at the micro level, considering members of the subgroup individually, or at the macro level, considering the subgroup as a whole, and (b) introduce notions of subgroup fairness that are robust, if not totally oblivious, to the cost of achieving recourse. We accompany these notions with an efficient, model-agnostic, highly parameterizable, and explainable framework for evaluating subgroup fairness. We demonstrate the advantages, the wide applicability, and the efficiency of our approach through a thorough experimental evaluation on different benchmark datasets.

        ----

        ## [2538] ProteinShake: Building datasets and benchmarks for deep learning on protein structures

        **Authors**: *Tim Kucera, Carlos G. Oliver, Dexiong Chen, Karsten M. Borgwardt*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6167294ed3d6fc61e11e1592ce5cb77-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6167294ed3d6fc61e11e1592ce5cb77-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We present ProteinShake, a Python software package that simplifies datasetcreation and model evaluation for deep learning on protein structures. Users cancreate custom datasets or load an extensive set of pre-processed datasets fromthe Protein Data Bank (PDB) and AlphaFoldDB. Each dataset is associated withprediction tasks and evaluation functions covering a broad array of biologicalchallenges. A benchmark on these tasks shows that pre-training almost alwaysimproves performance, the optimal data modality (graphs, voxel grids, or pointclouds) is task-dependent, and models struggle to generalize to new structures.ProteinShake makes protein structure data easily accessible and comparisonamong models straightforward, providing challenging benchmark settings withreal-world implications.ProteinShake is available at: https://proteinshake.ai

        ----

        ## [2539] Lovász Principle for Unsupervised Graph Representation Learning

        **Authors**: *Ziheng Sun, Chris Ding, Jicong Fan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b61da4f02b271cb7b5e3d538e2b78fb9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b61da4f02b271cb7b5e3d538e2b78fb9-Abstract-Conference.html)

        **Abstract**:

        This paper focuses on graph-level representation learning that aims to represent graphs as vectors that can be directly utilized in downstream tasks such as graph classification. We propose a novel graph-level representation learning principle called Lovász principle, which is motivated by the Lovász number in graph theory. The Lovász number of a graph is a real number that is an upper bound for graph Shannon capacity and is strongly connected with various global characteristics of the graph. Specifically, we show that the handle vector for computing the Lovász number is potentially a suitable choice for graph representation, as it captures a graph's global properties, though a direct application of the handle vector is difficult and problematic. We propose to use neural networks to address the problems and hence provide the Lovász principle. Moreover, we propose an enhanced Lovász principle that is able to exploit the subgraph Lovász numbers directly and efficiently. The experiments demonstrate that our Lovász principles achieve competitive performance compared to the baselines in unsupervised and semi-supervised graph-level representation learning tasks. The code of our Lovász principles is publicly available on GitHub.

        ----

        ## [2540] ComSL: A Composite Speech-Language Model for End-to-End Speech-to-Text Translation

        **Authors**: *Chenyang Le, Yao Qian, Long Zhou, Shujie Liu, Yanmin Qian, Michael Zeng, Xuedong Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6262f7a34e5d641cdb3d33dc9ad1a5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6262f7a34e5d641cdb3d33dc9ad1a5a-Abstract-Conference.html)

        **Abstract**:

        Joint speech-language training is challenging due to the large demand for training data and GPU consumption, as well as the modality gap between speech and language. We present ComSL, a speech-language model built atop a composite architecture of public pre-trained speech-only and language-only models and optimized data-efficiently for spoken language tasks. Particularly, we propose to incorporate cross-modality learning into transfer learning and conduct them simultaneously for downstream tasks in a multi-task learning manner. Our approach has demonstrated effectiveness in end-to-end speech-to-text translation tasks, achieving a new state-of-the-art average BLEU score of 31.5 on the multilingual speech to English text translation task for 21 languages, as measured on the public CoVoST2 evaluation set.

        ----

        ## [2541] Reverse Engineering Self-Supervised Learning

        **Authors**: *Ido Ben-Shaul, Ravid Shwartz-Ziv, Tomer Galanti, Shai Dekel, Yann LeCun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b63ad8c24354b0e5bcb7aea16490beab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b63ad8c24354b0e5bcb7aea16490beab-Abstract-Conference.html)

        **Abstract**:

        Understanding the learned representation and underlying mechanisms of Self-Supervised Learning (SSL) often poses a challenge. In this paper, we ‘reverse engineer’ SSL, conducting an in-depth empirical analysis of its learned internal representations, encompassing diverse models, architectures, and hyperparameters. Our study reveals an intriguing process within the SSL training: an inherent facilitation of semantic label-based clustering, which is surprisingly driven by the regularization component of the SSL objective. This clustering not only enhances downstream classification, but also compresses the information. We further illustrate that the alignment of the SSL-trained representation is more pronounced with semantic classes rather than random functions. Remarkably, the learned representations align with semantic classes across various hierarchical levels, with this alignment intensifying when going deeper into the network. This ‘reverse engineering’ approach provides valuable insights into the inner mechanism of SSL and their influences on the performance across different class sets.

        ----

        ## [2542] DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning

        **Authors**: *Alexander H. Liu, Heng-Jui Chang, Michael Auli, Wei-Ning Hsu, Jim Glass*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6404bf461c3c3186bdf5f55756af908-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6404bf461c3c3186bdf5f55756af908-Abstract-Conference.html)

        **Abstract**:

        In this paper, we introduce self-distillation and online clustering for self-supervised speech representation learning (DinoSR) which combines masked language modeling, self-distillation, and online clustering. We show that these concepts complement each other and result in a strong representation learning model for speech. DinoSR first extracts contextualized embeddings from the input audio with a teacher network, then runs an online clustering system on the embeddings to yield a machine-discovered phone inventory, and finally uses the discretized tokens to guide a student network. We show that DinoSR surpasses previous state-of-the-art performance in several downstream tasks, and provide a detailed analysis of the model and the learned discrete units.

        ----

        ## [2543] 4M: Massively Multimodal Masked Modeling

        **Authors**: *David Mizrahi, Roman Bachmann, Oguzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, Amir Zamir*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6446566965fa38e183650728ab70318-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6446566965fa38e183650728ab70318-Abstract-Conference.html)

        **Abstract**:

        Current machine learning models for vision are often highly specialized and limited to a single modality and task. In contrast, recent large language models exhibit a wide range of capabilities, hinting at a possibility for similarly versatile models in computer vision.In this paper, we take a step in this direction and propose a multimodal training scheme called 4M. It consists of training a single unified Transformer encoder-decoder using a masked modeling objective across a wide range of input/output modalities  â€“ including text, images, geometric, and semantic modalities, as well as neural network feature maps. 4M achieves scalability by unifying the representation space of all modalities through mapping them into discrete tokens and performing multimodal masked modeling on a small randomized subset of tokens.4M leads to models that exhibit several key capabilities: (1) they can perform a diverse set of vision tasks out of the box, (2) they excel when fine-tuned for unseen downstream tasks or new input modalities, and (3) they can function as a generative model that can be conditioned on arbitrary modalities, enabling a wide variety of expressive multimodal editing capabilities with remarkable flexibility.Through experimental analyses, we demonstrate the potential of 4M for training versatile and scalable foundation models for vision tasks, setting the stage for further exploration in multimodal learning for vision and other domains.

        ----

        ## [2544] Non-Rigid Shape Registration via Deep Functional Maps Prior

        **Authors**: *Puhua Jiang, Mingze Sun, Ruqi Huang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose a learning-based framework for non-rigid shape registra- tion without correspondence supervision. Traditional shape registration techniques typically rely on correspondences induced by extrinsic proximity, therefore can fail in the presence of large intrinsic deformations. Spectral mapping methods overcome this challenge by embedding shapes into, geometric or learned, high- dimensional spaces, where shapes are easier to align. However, due to the dependency on abstract, non-linear embedding schemes, the latter can be vulnerable with respect to perturbed or alien input. In light of this, our framework takes the best of both worlds. Namely, we deform source mesh towards the target point cloud, guided by correspondences induced by high-dimensional embeddings learned from deep functional maps (DFM). In particular, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Moreover, in order to alleviate the requirement of extrinsically aligned input, we train an orientation regressor on a set of aligned synthetic shapes independent of the training shapes for DFM. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic defor- mations, in which case neither traditional registration methods nor intrinsic methods work. The code is available at https://github.com/rqhuang88/DFR.

        ----

        ## [2545] Game Solving with Online Fine-Tuning

        **Authors**: *Ti-Rong Wu, Hung Guei, Ting-Han Wei, Chung-Chin Shih, Jui-Te Chin, I-Chen Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b663eb1512ce6c268e3e56f34c6d2959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b663eb1512ce6c268e3e56f34c6d2959-Abstract-Conference.html)

        **Abstract**:

        Game solving is a similar, yet more difficult task than mastering a game. Solving a game typically means to find the game-theoretic value (outcome given optimal play), and optionally a full strategy to follow in order to achieve that outcome. The AlphaZero algorithm has demonstrated super-human level play, and its powerful policy and value predictions have also served as heuristics in game solving. However, to solve a game and obtain a full strategy, a winning response must be found for all possible moves by the losing player. This includes very poor lines of play from the losing side, for which the AlphaZero self-play process will not encounter. AlphaZero-based heuristics can be highly inaccurate when evaluating these out-of-distribution positions, which occur throughout the entire search. To address this issue, this paper investigates applying online fine-tuning while searching and proposes two methods to learn tailor-designed heuristics for game solving. Our experiments show that using online fine-tuning can solve a series of challenging 7x7 Killall-Go problems, using only 23.54\% of computation time compared to the baseline without online fine-tuning. Results suggest that the savings scale with problem size. Our method can further be extended to any tree search algorithm for problem solving. Our code is available at https://rlg.iis.sinica.edu.tw/papers/neurips2023-online-fine-tuning-solver.

        ----

        ## [2546] Beyond probability partitions: Calibrating neural networks with semantic aware grouping

        **Authors**: *Jia-Qi Yang, De-Chuan Zhan, Le Gan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b693a240cf1009bff9fa4422141c9392-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b693a240cf1009bff9fa4422141c9392-Abstract-Conference.html)

        **Abstract**:

        Research has shown that deep networks tend to be overly optimistic about their predictions, leading to an underestimation of prediction errors. Due to the limited nature of data, existing studies have proposed various methods based on model prediction probabilities to bin the data and evaluate calibration error. We propose a more generalized definition of calibration error called Partitioned Calibration Error (PCE), revealing that the key difference among these calibration error metrics lies in how the data space is partitioned. We put forth an intuitive proposition that an accurate model should be calibrated across any partition, suggesting that the input space partitioning can extend beyond just the partitioning of prediction probabilities, and include partitions directly related to the input. Through semantic-related partitioning functions, we demonstrate that the relationship between model accuracy and calibration lies in the granularity of the partitioning function. This highlights the importance of partitioning criteria for training a calibrated and accurate model. To validate the aforementioned analysis, we propose a method that involves jointly learning a semantic aware grouping function based on deep model features and logits to partition the data space into subsets. Subsequently, a separate calibration function is learned for each subset. Experimental results demonstrate that our approach achieves significant performance improvements across multiple datasets and network architectures, thus highlighting the importance of the partitioning function for calibration.

        ----

        ## [2547] Identifiable Contrastive Learning with Automatic Feature Importance Discovery

        **Authors**: *Qi Zhang, Yifei Wang, Yisen Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html)

        **Abstract**:

        Existing contrastive learning methods rely on pairwise sample contrast $z_x^\top z_{x'}$ to learn data representations, but the learned features often lack clear interpretability from a human perspective. Theoretically, it lacks feature identifiability and different initialization may lead to totally different features. In this paper, we study a new method named tri-factor contrastive learning (triCL) that involves a 3-factor contrast in the form of $z_x^\top S z_{x'}$, where $S=\text{diag}(s_1,\dots,s_k)$ is a learnable diagonal matrix that automatically captures the importance of each feature. We show that by this simple extension, triCL can not only obtain identifiable features that eliminate randomness but also obtain more interpretable features that are ordered according to the importance matrix $S$. We show that features with high importance have nice interpretability by capturing common classwise features, and obtain superior performance when evaluated for image retrieval using a few features. The proposed triCL objective is general and can be applied to different contrastive learning methods like SimCLR and CLIP. We believe that it is a better alternative to existing 2-factor contrastive learning by improving its identifiability and interpretability with minimal overhead. Code is available at https://github.com/PKU-ML/Tri-factor-Contrastive-Learning.

        ----

        ## [2548] Revisiting Out-of-distribution Robustness in NLP: Benchmarks, Analysis, and LLMs Evaluations

        **Authors**: *Lifan Yuan, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Fangyuan Zou, Xingyi Cheng, Heng Ji, Zhiyuan Liu, Maosong Sun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6b5f50a2001ad1cbccca96e693c4ab4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6b5f50a2001ad1cbccca96e693c4ab4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        This paper reexamines the research on out-of-distribution (OOD) robustness in the field of NLP. We find that the distribution shift settings in previous studies commonly lack adequate challenges, hindering the accurate evaluation of OOD robustness. To address these issues, we propose a benchmark construction protocol that ensures clear differentiation and challenging distribution shifts. Then we introduceBOSS, a Benchmark suite for Out-of-distribution robustneSS evaluation covering 5 tasks and 20 datasets. Based on BOSS, we conduct a series of experiments on pretrained language models for analysis and evaluation of OOD robustness. First, for vanilla fine-tuning, we examine the relationship between in-distribution (ID) and OOD performance. We identify three typical types that unveil the inner learningmechanism, which could potentially facilitate the forecasting of OOD robustness, correlating with the advancements on ID datasets. Then, we evaluate 5 classic methods on BOSS and find that, despite exhibiting some effectiveness in specific cases, they do not offer significant improvement compared to vanilla fine-tuning. Further, we evaluate 5 LLMs with various adaptation paradigms and find that when sufficient ID data is available, fine-tuning domain-specific models outperform LLMs on ID examples significantly. However, in the case of OOD instances, prioritizing LLMs with in-context learning yields better results. We identify that both fine-tuned small models and LLMs face challenges in effectively addressing downstream tasks. The code is public at https://github.com/lifan-yuan/OOD_NLP.

        ----

        ## [2549] Towards Consistent Video Editing with Text-to-Image Diffusion Models

        **Authors**: *Zicheng Zhang, Bonan Li, Xuecheng Nie, Congying Han, Tiande Guo, Luoqi Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6c05f8254a00709e16fb0fdaae56cd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6c05f8254a00709e16fb0fdaae56cd8-Abstract-Conference.html)

        **Abstract**:

        Existing works have advanced Text-to-Image (TTI) diffusion models for video editing in a one-shot learning manner. Despite their low requirements of data and computation, these methods might produce results of unsatisfied consistency with text prompt as well as temporal sequence, limiting their applications in the real world. In this paper, we propose to address the above issues with a novel EI$^2$ model towards Enhancing vIdeo Editing consIstency of TTI-based frameworks. Specifically, we analyze and find that the inconsistent problem is caused by newly added modules into TTI models for learning temporal information. These modules lead to covariate shift in the feature space, which harms the editing capability. Thus, we design EI$^2$ to tackle the above drawbacks with two classical modules: Shift-restricted Temporal Attention Module (STAM) and Fine-coarse Frame Attention Module (FFAM). First, through theoretical analysis, we demonstrate that covariate shift is highly related to Layer Normalization, thus STAM employs a Instance Centering layer replacing it to preserve the distribution of temporal features.  In addition, STAM employs an attention layer with normalized mapping to transform temporal features while constraining the variance shift.  As the second part, we incorporate STAM with a novel FFAM, which efficiently leverages fine-coarse spatial information of overall frames to further enhance temporal consistency. Extensive experiments demonstrate the superiority of the proposed EI$^2$ model.

        ----

        ## [2550] Federated Spectral Clustering via Secure Similarity Reconstruction

        **Authors**: *Dong Qiao, Chris Ding, Jicong Fan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6cd2650926d332c86a84c48529cc421-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6cd2650926d332c86a84c48529cc421-Abstract-Conference.html)

        **Abstract**:

        Federated learning has a significant advantage in protecting information privacy. Many scholars proposed various secure learning methods within the framework of federated learning but the study on secure federated unsupervised learning especially clustering is limited. We in this work propose a secure kernelized factorization method for federated spectral clustering on distributed dataset. The method is non-trivial because the kernel or similarity matrix for spectral clustering is computed by data pairs, which violates the principle of privacy protection. Our method implicitly constructs an approximation for the kernel matrix on distributed data such that we can perform spectral clustering under the constraint of privacy protection. We provide a convergence guarantee of the optimization algorithm, reconstruction error bounds of the Gaussian kernel matrix, and the sufficient condition of correct clustering of our method. We also present some results of differential privacy. Numerical results on synthetic and real datasets demonstrate that the proposed method is efficient and accurate in comparison to the baselines.

        ----

        ## [2551] CS-Isolate: Extracting Hard Confident Examples by Content and Style Isolation

        **Authors**: *Yexiong Lin, Yu Yao, Xiaolong Shi, Mingming Gong, Xu Shen, Dong Xu, Tongliang Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6d67c380f8bde2adc4247d0036c0c73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6d67c380f8bde2adc4247d0036c0c73-Abstract-Conference.html)

        **Abstract**:

        Label noise widely exists in large-scale image datasets. To mitigate the side effects of label noise, state-of-the-art methods focus on selecting confident examples by leveraging semi-supervised learning. Existing research shows that the ability to extract hard confident examples, which are close to the decision boundary, significantly influences the generalization ability of the learned classifier.In this paper, we find that a key reason for some hard examples being close to the decision boundary is due to the entanglement of style factors with content factors. The hard examples become more discriminative when we focus solely on content factors, such as semantic information, while ignoring style factors. Nonetheless, given only noisy data, content factors are not directly observed and have to be inferred.To tackle the problem of inferring content factors for classification when learning with noisy labels, our objective is to ensure that the content factors of all examples in the same underlying clean class remain unchanged as their style information changes.To achieve this, we utilize different data augmentation techniques to alter the styles while regularizing content factors based on some confident examples. By training existing methods with our inferred content factors, CS-Isolate proves their effectiveness in learning hard examples on benchmark datasets. The implementation is available at https://github.com/tmllab/2023NeurIPSCS-isolate.

        ----

        ## [2552] Conditional independence testing under misspecified inductive biases

        **Authors**: *Felipe Maia Polo, Yuekai Sun, Moulinath Banerjee*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6f2b16abf590e80c9df30bb5f8e2b7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6f2b16abf590e80c9df30bb5f8e2b7d-Abstract-Conference.html)

        **Abstract**:

        Conditional independence (CI) testing is a fundamental and challenging task in modern statistics and machine learning. Many modern methods for CI testing rely on powerful supervised learning methods to learn regression functions or Bayes predictors as an intermediate step; we refer to this class of tests as regression-based tests. Although these methods are guaranteed to control Type-I error when the supervised learning methods accurately estimate the regression functions or Bayes predictors of interest, their behavior is less understood when they fail due to misspecified inductive biases; in other words, when the employed models are not flexible enough or when the training algorithm does not induce the desired predictors. Then, we study the performance of regression-based CI tests under misspecified inductive biases. Namely, we propose new approximations or upper bounds for the testing errors of three regression-based tests that depend on misspecification errors. Moreover, we introduce the Rao-Blackwellized Predictor Test (RBPT), a regression-based CI test robust against misspecified inductive biases. Finally, we conduct experiments with artificial and real data, showcasing the usefulness of our theory and methods.

        ----

        ## [2553] Blurred-Dilated Method for Adversarial Attacks

        **Authors**: *Yang Deng, Weibin Wu, Jianping Zhang, Zibin Zheng*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b6fa3ed9624c184bd73e435123bd576a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b6fa3ed9624c184bd73e435123bd576a-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) are vulnerable to adversarial attacks, which lead to incorrect predictions. In black-box settings, transfer attacks can be conveniently used to generate adversarial examples. However, such examples tend to overfit the specific architecture and feature representations of the source model, resulting in poor attack performance against other target models. To overcome this drawback, we propose a novel model modification-based transfer attack: Blurred-Dilated method (BD) in this paper. In summary, BD works by reducing downsampling while introducing BlurPool and dilated convolutions in the source model. Then BD employs the modified source model to generate adversarial samples. We think that BD can more comprehensively preserve the feature information than the original source model. It thus enables more thorough destruction of the image features, which can improve the transferability of the generated adversarial samples. Extensive experiments on the ImageNet dataset show that adversarial examples generated by BD achieve significantly higher transferability than the state-of-the-art baselines. Besides, BD can be conveniently combined with existing black-box attack techniques to further improve their performance.

        ----

        ## [2554] Towards Distribution-Agnostic Generalized Category Discovery

        **Authors**: *Jianhong Bai, Zuozhu Liu, Hualiang Wang, Ruizhe Chen, Lianrui Mu, Xiaomeng Li, Joey Tianyi Zhou, Yang Feng, Jian Wu, Haoji Hu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html)

        **Abstract**:

        Data imbalance and open-ended distribution are two intrinsic characteristics of the real visual world. Though encouraging progress has been made in tackling each challenge separately, few works dedicated to combining them towards real-world scenarios. While several previous works have focused on classifying close-set samples and detecting open-set samples during testing, it's still essential to be able to classify unknown subjects as human beings. In this paper, we formally define a more realistic task as distribution-agnostic generalized category discovery (DA-GCD): generating fine-grained predictions for both close- and open-set classes in a long-tailed open-world setting. To tackle the challenging problem, we propose a Self-Balanced Co-Advice contrastive framework (BaCon), which consists of a contrastive-learning branch and a pseudo-labeling branch, working collaboratively to provide interactive supervision to resolve the DA-GCD task. In particular, the contrastive-learning branch provides reliable distribution estimation to regularize the predictions of the pseudo-labeling branch, which in turn guides contrastive learning through self-balanced knowledge transfer and a proposed novel contrastive loss. We compare BaCon with state-of-the-art methods from two closely related fields: imbalanced semi-supervised learning and generalized category discovery. The effectiveness of BaCon is demonstrated with superior performance over all baselines and comprehensive analysis across various datasets. Our code is publicly available.

        ----

        ## [2555] Stable Diffusion is Unstable

        **Authors**: *Chengbin Du, Yanxi Li, Zhongwei Qiu, Chang Xu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b733cdd80ed2ae7e3156d8c33108c5d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b733cdd80ed2ae7e3156d8c33108c5d5-Abstract-Conference.html)

        **Abstract**:

        Recently, text-to-image models have been thriving. Despite their powerful generative capacity, our research has uncovered a lack of robustness in this generation process. Specifically, the introduction of small perturbations to the text prompts can result in the blending of primary subjects with other categories or their complete disappearance in the generated images. In this paper, we propose Auto-attack on Text-to-image Models (ATM), a gradient-based approach, to effectively and efficiently generate such perturbations. By learning a Gumbel Softmax distribution, we can make the discrete process of word replacement or extension continuous, thus ensuring the differentiability of the perturbation generation. Once the distribution is learned, ATM can sample multiple attack samples simultaneously. These attack samples can prevent the generative model from generating the desired subjects without tampering with the category keywords in the prompt. ATM has achieved a 91.1\% success rate in short-text attacks and an 81.2\% success rate in long-text attacks. Further empirical analysis revealed three attack patterns based on: 1) variability in generation speed, 2) similarity of coarse-grained characteristics, and 3) polysemy of words. The code is available at https://github.com/duchengbin8/StableDiffusionis_Unstable

        ----

        ## [2556] A Competitive Algorithm for Agnostic Active Learning

        **Authors**: *Yihan Zhou, Eric Price*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7385cb3fa76a0aeedb23d4163640db0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7385cb3fa76a0aeedb23d4163640db0-Abstract-Conference.html)

        **Abstract**:

        For some hypothesis classes and input distributions, \emph{active}  agnostic learning needs exponentially fewer samples than passive  learning; for other classes and distributions, it offers little to  no improvement.  The most popular algorithms for agnostic active  learning express their performance in terms of a parameter called  the disagreement coefficient, but it is known that these algorithms  are inefficient on some inputs.  We take a different approach to agnostic active learning, getting an  algorithm that is \emph{competitive} with the optimal algorithm for  any binary hypothesis class $H$ and distribution $\mathcal{D}_X$ over $X$.  In particular, if any algorithm can use $m^*$ queries to get  $O(\eta)$ error, then our algorithm uses $O(m^* \log H)$ queries to  get $O(\eta)$ error.  Our algorithm lies in the vein of the  splitting-based approach of Dasgupta [2004], which gets a similar  result for the realizable ($\eta = 0$) setting.  We also show that it is NP-hard to do better than our algorithm's  $O(\log H)$ overhead in general.

        ----

        ## [2557] Efficient Hyper-parameter Optimization with Cubic Regularization

        **Authors**: *Zhenqian Shen, Hansi Yang, Yong Li, James T. Kwok, Quanming Yao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7500454af92cf3934eb1cc2d59abbdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7500454af92cf3934eb1cc2d59abbdf-Abstract-Conference.html)

        **Abstract**:

        As hyper-parameters are ubiquitous and can significantly affect the model performance, hyper-parameter optimization is extremely important in machine learning. In this paper, we consider a sub-class of hyper-parameter optimization problems, where the hyper-gradients are not available. Such problems frequently appear when the performance metric is non-differentiable or the hyper-parameter is not continuous. However, existing algorithms, like Bayesian optimization and reinforcement learning, often get trapped in local optimals with poor performance. To address the above limitations, we propose to use cubic regularization to accelerate convergence and avoid saddle points. First, we adopt stochastic relaxation, which allows obtaining gradient and Hessian information without hyper-gradients. Then, we exploit the rich curvature information by cubic regularization. Theoretically, we prove that the proposed method can converge to approximate second-order stationary points, and the convergence is also guaranteed when the lower-level problem is inexactly solved. Experiments on synthetic and real-world data demonstrate the effectiveness of our proposed method.

        ----

        ## [2558] Joint Attribute and Model Generalization Learning for Privacy-Preserving Action Recognition

        **Authors**: *Duo Peng, Li Xu, Qiuhong Ke, Ping Hu, Jun Liu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b762632135b16f1225672f9fe2a9740b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b762632135b16f1225672f9fe2a9740b-Abstract-Conference.html)

        **Abstract**:

        Privacy-Preserving Action Recognition (PPAR) aims to transform raw videos into anonymous ones to prevent privacy leakage while maintaining action clues, which is an increasingly important problem in intelligent vision applications. Despite recent efforts in this task, it is still challenging to deal with novel privacy attributes and novel privacy attack models that are unavailable during the training phase. In this paper, from the perspective of meta-learning (learning to learn), we propose a novel Meta Privacy-Preserving Action Recognition (MPPAR) framework to improve both generalization abilities above (i.e., generalize to novel privacy attributes and novel privacy attack models) in a unified manner. Concretely, we simulate train/test task shifts by constructing disjoint support/query sets w.r.t. privacy attributes or attack models. Then, a virtual training and testing scheme is applied based on support/query sets to provide feedback to optimize the model's learning toward better generalization. Extensive experiments demonstrate the effectiveness and generalization of the proposed framework compared to state-of-the-arts.

        ----

        ## [2559] 3D-Aware Visual Question Answering about Parts, Poses and Occlusions

        **Authors**: *Xingrui Wang, Wufei Ma, Zhuowan Li, Adam Kortylewski, Alan L. Yuille*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b783c44ba9adbc30344473dc633b4869-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b783c44ba9adbc30344473dc633b4869-Abstract-Conference.html)

        **Abstract**:

        Despite rapid progress in Visual question answering (\textit{VQA}), existing datasets and models mainly focus on testing reasoning in 2D.  However, it is important that VQA models also understand the 3D structure of visual scenes, for example to support tasks like navigation or manipulation.  This includes an understanding of the 3D object pose, their parts and occlusions.   In this work, we introduce the task of 3D-aware VQA, which focuses on challenging questions that require a compositional reasoning over the 3D structure of visual scenes.   We address 3D-aware VQA from both the dataset and the model perspective.   First, we introduce Super-CLEVR-3D, a compositional reasoning dataset that contains questions about object parts, their 3D poses, and occlusions.   Second, we propose PO3D-VQA, a 3D-aware VQA model that marries two powerful ideas: probabilistic neural symbolic program execution for reasoning and deep neural networks with 3D generative representations of objects for robust visual recognition.  Our experimental results show our model PO3D-VQA outperforms existing methods significantly, but we still observe a significant performance gap compared to 2D VQA benchmarks, indicating that 3D-aware VQA remains an important open research area.

        ----

        ## [2560] MixFormerV2: Efficient Fully Transformer Tracking

        **Authors**: *Yutao Cui, Tianhui Song, Gangshan Wu, Limin Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7870bd43b2d133a1ed95582ae5d82a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7870bd43b2d133a1ed95582ae5d82a4-Abstract-Conference.html)

        **Abstract**:

        Transformer-based trackers have achieved strong accuracy on the standard benchmarks. However, their efficiency remains an obstacle to practical deployment on both GPU and CPU platforms. In this paper, to overcome this issue, we propose a fully transformer tracking framework, coined as \emph{MixFormerV2}, without any dense convolutional operation and complex score prediction module. Our key design is to introduce four special prediction tokens and concatenate them with the tokens from target template and search areas. Then, we apply the unified transformer backbone on these mixed token sequence. These prediction tokens are able to capture the complex correlation between target template and search area via mixed attentions. Based on them, we can easily predict the tracking box and estimate its confidence score through simple MLP heads. To further improve the efficiency of MixFormerV2, we present a new distillation-based model reduction paradigm, including dense-to-sparse distillation and deep-to-shallow distillation. The former one aims to transfer knowledge from the dense-head based MixViT to our fully transformer tracker, while the latter one is used to prune some layers of the backbone. We instantiate two types of MixForemrV2, where the MixFormerV2-B achieves an AUC of 70.6\% on LaSOT and AUC of 56.7\% on TNL2k with a high GPU speed of 165 FPS, and the MixFormerV2-S surpasses FEAR-L by 2.7\% AUC on LaSOT with a real-time CPU speed.

        ----

        ## [2561] Generalized Information-theoretic Multi-view Clustering

        **Authors**: *Weitian Huang, Sirui Yang, Hongmin Cai*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7aa34d2d24f9bab3056993b7bfa0f1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7aa34d2d24f9bab3056993b7bfa0f1b-Abstract-Conference.html)

        **Abstract**:

        In an era of more diverse data modalities, multi-view clustering has become a fundamental tool for comprehensive data analysis and exploration. However, existing multi-view unsupervised learning methods often rely on strict assumptions on semantic consistency among samples. In this paper, we reformulate the multi-view clustering problem from an information-theoretic perspective and propose a general theoretical model. In particular, we define three desiderata under multi-view unsupervised learning in terms of mutual information, namely, comprehensiveness, concentration, and cross-diversity. The multi-view variational lower bound is then obtained by approximating the samples' high-dimensional mutual information. The Kullbackâ€“Leibler divergence is utilized to deduce sample assignments. Ultimately the information-based multi-view clustering model leverages deep neural networks and Stochastic Gradient Variational Bayes to achieve representation learning and clustering simultaneously. Extensive experiments on both synthetic and real datasets with wide types demonstrate that the proposed method exhibits a more stable and superior clustering performance than state-of-the-art algorithms.

        ----

        ## [2562] Counterfactual Evaluation of Peer-Review Assignment Policies

        **Authors**: *Martin Saveski, Steven Jecmen, Nihar B. Shah, Johan Ugander*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7d795e655c1463d7299688d489e8ef4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7d795e655c1463d7299688d489e8ef4-Abstract-Conference.html)

        **Abstract**:

        Peer review assignment algorithms aim to match research papers to suitable expert reviewers, working to maximize the quality of the resulting reviews. A key challenge in designing effective assignment policies is evaluating how changes to the assignment algorithm map to changes in review quality. In this work, we leverage recently proposed policies that introduce randomness in peer-review assignment—in order to mitigate fraud—as a valuable opportunity to evaluate counterfactual assignment policies. Specifically, we exploit how such randomized assignments provide a positive probability of observing the reviews of many assignment policies of interest. To address challenges in applying standard off-policy evaluation methods, such as violations of positivity, we introduce novel methods for partial identification based on monotonicity and Lipschitz smoothness assumptions for the mapping between reviewer-paper covariates and outcomes. We apply our methods to peer-review data from two computer science venues: the TPDP'21 workshop (95 papers and 35 reviewers) and the AAAI'22 conference (8,450 papers and 3,145 reviewers). We consider estimates of (i) the effect on review quality when changing weights in the assignment algorithm, e.g., weighting reviewers' bids vs. textual similarity (between the review's past papers and the submission), and (ii) the "cost of randomization", capturing the difference in expected quality between the perturbed and unperturbed optimal match. We find that placing higher weight on text similarity results in higher review quality and that introducing randomization in the reviewer-paper assignment only marginally reduces the review quality. Our methods for partial identification may be of independent interest, while our off-policy approach can likely find use in evaluating a broad class of algorithmic matching systems.

        ----

        ## [2563] Temporal Causal Mediation through a Point Process: Direct and Indirect Effects of Healthcare Interventions

        **Authors**: *Çaglar Hizli, St John, Anne Juuti, Tuure Saarinen, Kirsi Pietiläinen, Pekka Marttinen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7d9b1d4a9464d5d1ece82198e351349-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7d9b1d4a9464d5d1ece82198e351349-Abstract-Conference.html)

        **Abstract**:

        Deciding on an appropriate intervention requires a causal model of a treatment, the outcome, and potential mediators. Causal mediation analysis lets us distinguish between direct and indirect effects of the intervention, but has mostly been studied in a static setting. In healthcare, data come in the form of complex, irregularly sampled time-series, with dynamic interdependencies between a treatment, outcomes, and mediators across time. Existing approaches to dynamic causal mediation analysis are limited to regular measurement intervals, simple parametric models, and disregard long-range mediator--outcome interactions. To address these limitations, we propose a non-parametric mediator--outcome model where the mediator is assumed to be a temporal point process that interacts with the outcome process. With this model, we estimate the direct and indirect effects of an external intervention on the outcome, showing how each of these affects the whole future trajectory. We demonstrate on semi-synthetic data that our method can accurately estimate direct and indirect effects. On real-world healthcare data, our model infers clinically  meaningful direct and indirect effect trajectories for blood glucose after a surgery.

        ----

        ## [2564] Randomized and Deterministic Maximin-share Approximations for Fractionally Subadditive Valuations

        **Authors**: *Hannaneh Akrami, Kurt Mehlhorn, Masoud Seddighin, Golnoosh Shahkarami*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b7ed46bd87cd51d4c031b96d9b1a8eb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b7ed46bd87cd51d4c031b96d9b1a8eb6-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of guaranteeing maximin-share ($\MMS$) when allocating a set of indivisible items to a set of agents with fractionally subadditive ($\XOS$) valuations.  For $\XOS$ valuations, it has been previously shown that for some instances no allocation can guarantee a fraction better than $1/2$ of maximin-share to all the agents. Also, a deterministic allocation exists that guarantees $0.219225$ of the maximin-share of each agent. Our results involve both deterministic and randomized allocations. On the deterministic side, we improve the best approximation guarantee for fractionally subadditive valuations to $3/13 = 0.230769$. We develop new ideas on allocating large items in our allocation algorithm which might be of independent interest. Furthermore, we investigate randomized algorithms and the Best-of-both-worlds fairness guarantees. We propose a randomized allocation that is $1/4$-$\MMS$ ex-ante and $1/8$-$\MMS$ ex-post for $\XOS$ valuations. Moreover, we prove an upper bound of $3/4$ on the ex-ante guarantee for this class of valuations.

        ----

        ## [2565] Causal normalizing flows: from theory to practice

        **Authors**: *Adrián Javaloy, Pablo Sánchez-Martín, Isabel Valera*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8402301e7f06bdc97a31bfaa653dc32-Abstract-Conference.html)

        **Abstract**:

        In this work, we deepen on the use of normalizing flows for causal reasoning. Specifically, we first leverage recent results on non-linear ICA to show that causal models are identifiable from observational data given a causal ordering, and thus can be recovered using autoregressive normalizing flows (NFs). Second, we analyze different design and learning choices for causal normalizing flows to capture the underlying causal data-generating process. Third, we describe how to implement the do-operator in causal NFs, and thus, how to answer interventional and counterfactual questions. Finally, in our experiments, we validate our design and training choices through a comprehensive ablation study; compare causal NFs to other approaches for approximating causal models; and empirically demonstrate that causal NFs can be used to address real-world problems—where the presence of mixed discrete-continuous data and partial knowledge on the causal graph is the norm. The code for this work can be found at https://github.com/psanch21/causal-flows.

        ----

        ## [2566] Maximum Average Randomly Sampled: A Scale Free and Non-parametric Algorithm for Stochastic Bandits

        **Authors**: *Masoud Moravej Khorasani, Erik Weyer*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b84adff45775e92a45f0cd87c37f5ce9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b84adff45775e92a45f0cd87c37f5ce9-Abstract-Conference.html)

        **Abstract**:

        Upper Confidence Bound (UCB) methods are one of the most effective methods in dealing with the exploration-exploitation trade-off in online decision-making problems. The confidence bounds utilized in UCB methods tend to be constructed based on concentration equalities which are usually dependent on a parameter of scale (e.g. a bound on the payoffs, a variance, or a subgaussian parameter) that must be known in advance. The necessity of knowing a scale parameter a priori and the fact that the confidence bounds only use the tail information can deteriorate the performance of the UCB methods.Here we propose a data-dependent UCB algorithm called MARS (Maximum Average Randomly Sampled) in a non-parametric setup for multi-armed bandits with symmetric rewards. The algorithm does not depend on any scaling, and the data-dependent upper confidence bound is constructed based on the maximum average of randomly sampled rewards inspired by the work of Hartigan in the 1960s and 70s. A regret bound for the multi-armed bandit problem is derived under the same assumptions as for the $\psi$-UCB method without incorporating any correction factors. The method is illustrated and compared with baseline algorithms in numerical experiments.

        ----

        ## [2567] Cappy: Outperforming and Boosting Large Multi-Task LMs with a Small Scorer

        **Authors**: *Bowen Tan, Yun Zhu, Lijuan Liu, Eric P. Xing, Zhiting Hu, Jindong Chen*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b860c0c546f4a3a786f9c9468228c99f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b860c0c546f4a3a786f9c9468228c99f-Abstract-Conference.html)

        **Abstract**:

        Large language models (LLMs) such as T0, FLAN, and OPT-IML excel in multi-tasking under a unified instruction-following paradigm, where they also exhibit remarkable generalization abilities to unseen tasks. Despite their impressive performance, these LLMs, with sizes ranging from several billion to hundreds of billions of parameters, demand substantial computational resources, making their training and inference expensive and inefficient. Furthermore, adapting these models to downstream applications, particularly complex tasks, is often unfeasible due to the extensive hardware requirements for finetuning, even when utilizing parameter-efficient approaches such as prompt tuning. Additionally, the most powerful multi-task LLMs, such as OPT-IML-175B and FLAN-PaLM-540B, are not publicly accessible, severely limiting their customization potential. To address these challenges, we introduce a pretrained small scorer, \textit{Cappy}, designed to enhance the performance and efficiency of multi-task LLMs. With merely 360 million parameters, Cappy functions either independently on classification tasks or serve as an auxiliary component for LLMs, boosting their performance. Moreover, Cappy enables efficiently integrating downstream supervision without requiring LLM finetuning nor the access to their parameters. Our experiments demonstrate that, when working independently on 11 language understanding tasks from PromptSource, Cappy outperforms LLMs that are several orders of magnitude larger. Besides, on 45 complex tasks from BIG-Bench, Cappy boosts the performance of the advanced multi-task LLM, FLAN-T5, by a large margin. Furthermore, Cappy is flexible to cooperate with other LLM adaptations, including finetuning and in-context learning, offering additional performance enhancement.

        ----

        ## [2568] Enhancing Adaptive History Reserving by Spiking Convolutional Block Attention Module in Recurrent Neural Networks

        **Authors**: *Qi Xu, Yuyuan Gao, Jiangrong Shen, Yaxin Li, Xuming Ran, Huajin Tang, Gang Pan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8734840bf65c8facd619f5105c6acd0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8734840bf65c8facd619f5105c6acd0-Abstract-Conference.html)

        **Abstract**:

        Spiking neural networks (SNNs) serve as one type of efficient model to process spatio-temporal patterns in time series, such as the Address-Event Representation data collected from Dynamic Vision Sensor (DVS). Although convolutional SNNs have achieved remarkable performance on these AER datasets, benefiting from the predominant spatial feature extraction ability of convolutional structure, they ignore temporal features related to sequential time points. In this paper, we develop a recurrent spiking neural network (RSNN) model embedded with an advanced spiking convolutional block attention module (SCBAM) component to combine both spatial and temporal features of spatio-temporal patterns. It invokes the history information in spatial and temporal channels adaptively through SCBAM, which brings the advantages of efficient memory calling and history redundancy elimination. The performance of our model was evaluated in DVS128-Gesture dataset and other time-series datasets. The experimental results show that the proposed SRNN-SCBAM model makes better use of the history information in spatial and temporal dimensions with less memory space, and achieves higher accuracy compared to other models.

        ----

        ## [2569] Reducing Shape-Radiance Ambiguity in Radiance Fields with a Closed-Form Color Estimation Method

        **Authors**: *Qihang Fang, Yafei Song, Keqiang Li, Liefeng Bo*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b87738474533cab76c7bee4e08443aca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b87738474533cab76c7bee4e08443aca-Abstract-Conference.html)

        **Abstract**:

        A neural radiance field (NeRF) enables the synthesis of cutting-edge realistic novel view images of a 3D scene. It includes density and color fields to model the shape and radiance of a scene, respectively. Supervised by the photometric loss in an end-to-end training manner, NeRF inherently suffers from the shape-radiance ambiguity problem, i.e., it can perfectly fit training views but does not guarantee decoupling the two fields correctly. To deal with this issue, existing works have incorporated prior knowledge to provide an independent supervision signal for the density field, including total variation loss, sparsity loss, distortion loss, etc. These losses are based on general assumptions about the density field, e.g., it should be smooth, sparse, or compact, which are not adaptive to a specific scene. In this paper, we propose a more adaptive method to reduce the shape-radiance ambiguity. The key is a rendering method that is only based on the density field. Specifically, we first estimate the color field based on the density field and posed images in a closed form. Then NeRF's rendering process can proceed. We address the problems in estimating the color field, including occlusion and non-uniformly distributed views. Afterwards, it is applied to regularize NeRF's density field. As our regularization is guided by photometric loss, it is more adaptive compared to existing ones. Experimental results show that our method improves the density field of NeRF both qualitatively and quantitatively. Our code is available at https://github.com/qihangGH/Closed-form-color-field.

        ----

        ## [2570] Text-to-Image Diffusion Models are Zero Shot Classifiers

        **Authors**: *Kevin Clark, Priyank Jaini*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b87bdcf963cad3d0b265fcb78ae7d11e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b87bdcf963cad3d0b265fcb78ae7d11e-Abstract-Conference.html)

        **Abstract**:

        The excellent generative capabilities of text-to-image diffusion models suggest they learn informative representations of image-text data.However, what knowledge their representations capture is not fully understood, and they have not been thoroughly explored on downstream tasks.We investigate diffusion models by proposing a method for evaluating them as zero-shot classifiers.The key idea is using a diffusion model's ability to denoise a noised image given a text description of a label as a proxy for that label's likelihood.We apply our method to Stable Diffusion and Imagen, using it to probe fine-grained aspects of the models' knowledge and comparing them with CLIP's zero-shot abilities. They perform competitively with CLIP on a wide range of zero-shot image classification datasets. Additionally, they achieve state-of-the-art results on shape/texture bias tests and can successfully perform attribute binding while CLIP cannot.Although generative pre-training is prevalent in NLP, visual foundation models often use other methods such as contrastive learning. Based on our findings, we argue that generative pre-training should be explored as a compelling alternative for vision and vision-language problems.

        ----

        ## [2571] MADG: Margin-based Adversarial Learning for Domain Generalization

        **Authors**: *Aveen Dayal, Vimal K. B., Linga Reddy Cenkeramaddi, C. Krishna Mohan, Abhinav Kumar, Vineeth N. Balasubramanian*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b87d9d19ecb5927f7e18c537908610ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b87d9d19ecb5927f7e18c537908610ef-Abstract-Conference.html)

        **Abstract**:

        Domain Generalization (DG) techniques have emerged as a popular approach to address the challenges of domain shift in Deep Learning (DL), with the goal of generalizing well to the target domain unseen during the training. In recent years, numerous methods have been proposed to address the DG setting, among which one popular approach is the adversarial learning-based methodology. The main idea behind adversarial DG methods is to learn domain-invariant features by minimizing a discrepancy metric. However, most adversarial DG methods use 0-1 loss based $\mathcal{H}\Delta\mathcal{H}$ divergence metric. In contrast, the margin loss-based discrepancy metric has the following advantages: more informative, tighter, practical, and efficiently optimizable. To mitigate this gap, this work proposes a novel adversarial learning DG algorithm, $\textbf{MADG}$, motivated by a margin loss-based discrepancy metric. The proposed $\textbf{MADG}$ model learns domain-invariant features across all source domains and uses adversarial training to generalize well to the unseen target domain. We also provide a theoretical analysis of the proposed $\textbf{MADG}$ model based on the unseen target error bound. Specifically, we construct the link between the source and unseen domains in the real-valued hypothesis space and derive the generalization bound using margin loss and Rademacher complexity. We extensively experiment with the $\textbf{MADG}$ model on popular real-world DG datasets, VLCS, PACS, OfficeHome, DomainNet, and TerraIncognita. We evaluate the proposed algorithm on DomainBed's benchmark and observe consistent performance across all the datasets.

        ----

        ## [2572] Bridging RL Theory and Practice with the Effective Horizon

        **Authors**: *Cassidy Laidlaw, Stuart J. Russell, Anca D. Dragan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8be628bf719550b560de8bec9456e0b-Abstract-Conference.html)

        **Abstract**:

        Deep reinforcement learning (RL) works impressively in some environments and fails catastrophically in others. Ideally, RL theory should be able to provide an understanding of why this is, i.e. bounds predictive of practical performance. Unfortunately, current theory does not quite have this ability. We compare standard deep RL algorithms to prior sample complexity bounds by introducing a new dataset, BRIDGE. It consists of 155 MDPs from common deep RL benchmarks, along with their corresponding tabular representations, which enables us to exactly compute instance-dependent bounds. We find that prior bounds do not correlate well with when deep RL succeeds vs. fails, but discover a surprising property that does. When actions with the highest Q-values under the random policy also have the highest Q-values under the optimal policy—i.e., when it is optimal to act greedily with respect to the random's policy Q function—deep RL tends to succeed; when they don't, deep RL tends to fail. We generalize this property into a new complexity measure of an MDP that we call the effective horizon, which roughly corresponds to how many steps of lookahead search would be needed in that MDP in order to identify the next optimal action, when leaf nodes are evaluated with random rollouts. Using BRIDGE, we show that the effective horizon-based bounds are more closely reflective of the empirical performance of PPO and DQN than prior sample complexity bounds across four metrics. We also show that, unlike existing bounds, the effective horizon can predict the effects of using reward shaping or a pre-trained exploration policy. Our code and data are available at https://github.com/cassidylaidlaw/effective-horizon.

        ----

        ## [2573] Fine-Grained Human Feedback Gives Better Rewards for Language Model Training

        **Authors**: *Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A. Smith, Mari Ostendorf, Hannaneh Hajishirzi*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8c90b65739ae8417e61eadb521f63d5-Abstract-Conference.html)

        **Abstract**:

        Language models (LMs) often exhibit undesirable text generation behaviors, including generating false, toxic, or irrelevant outputs. Reinforcement learning from human feedback (RLHF)---where human preference judgments on LM outputs are transformed into a learning signal---has recently shown promise in addressing these issues. However, such holistic feedback conveys limited information on long text outputs; it does not indicate which aspects of the outputs influenced user preference; e.g., which parts contain what type(s) of errors. In this paper, we use fine-grained human feedback (e.g., which sentence is false, which sub-sentence is irrelevant) as an explicit training signal. We introduce Fine-Grained RLHF, a framework that enables training and learning from reward functions that are fine-grained in two respects: (1) density, providing a reward after every segment (e.g., a sentence) is generated; and (2) incorporating multiple reward models associated with different feedback types (e.g., factual incorrectness, irrelevance, and information incompleteness). We conduct experiments on detoxification and long-form question answering to illustrate how learning with this reward function leads to improved performance, supported by both automatic and human evaluation. Additionally, we show that LM behaviors can be customized using different combinations of fine-grained reward models. We release all data, collected human feedback, and codes at https://FineGrainedRLHF.github.io.

        ----

        ## [2574] Towards Optimal Effective Resistance Estimation

        **Authors**: *Rajat Vadiraj Dwaraknath, Ishani Karmarkar, Aaron Sidford*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8e2046160a568145af6d42eeef199f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8e2046160a568145af6d42eeef199f4-Abstract-Conference.html)

        **Abstract**:

        We provide new algorithms and conditional hardness for the problem of estimating effective resistances in $n$-node $m$-edge undirected, expander graphs. We provide an $\widetilde{O}(m\epsilon^{-1})$-time algorithm that produces with high probability, an $\widetilde{O}(n\epsilon^{-1})$-bit sketch from which the effective resistance between any pair of nodes can be estimated, to $(1 \pm \epsilon)$-multiplicative accuracy, in $\widetilde{O}(1)$-time. Consequently, we obtain an $\widetilde{O}(m\epsilon^{-1})$-time algorithm for estimating the effective resistance of all edges in such graphs, improving (for sparse graphs) on the previous fastest runtimes of $\widetilde{O}(m\epsilon^{-3/2})$ [Chu et. al. 2018] and $\widetilde{O}(n^2\epsilon^{-1})$ [Jambulapati, Sidford, 2018] for general graphs and $\widetilde{O}(m + n\epsilon^{-2})$ for expanders [Li, Sachdeva 2022]. We complement this result by showing a conditional lower bound that a broad set of algorithms for computing such estimates of the effective resistances between all pairs of nodes require $\widetilde{\Omega}(n^2 \epsilon^{-1/2})$-time, improving upon the previous best such lower bound of $\widetilde{\Omega}(n^2 \epsilon^{-1/13})$ [Musco et. al. 2017]. Further, we leverage the tools underlying these results to obtain improved algorithms and conditional hardness for more general problems of sketching the pseudoinverse of positive semidefinite matrices and estimating functions of their eigenvalues.

        ----

        ## [2575] TradeMaster: A Holistic Quantitative Trading Platform Empowered by Reinforcement Learning

        **Authors**: *Shuo Sun, Molei Qin, Wentao Zhang, Haochong Xia, Chuqiao Zong, Jie Ying, Yonggang Xie, Lingxuan Zhao, Xinrun Wang, Bo An*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b8f6f7f2ba4137124ac976286eacb611-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The financial markets, which involve over \$90 trillion market capitals, attract the attention of innumerable profit-seeking investors globally. Recent explosion of reinforcement learning in financial trading (RLFT) research has shown stellar performance on many quantitative trading tasks. However, it is still challenging to deploy reinforcement learning (RL) methods into real-world financial markets due to the highly composite nature of this domain, which entails design choices and interactions between components that collect financial data, conduct feature engineering, build market environments, make investment decisions, evaluate model behaviors and offers user interfaces. Despite the availability of abundant financial data and advanced RL techniques, a remarkable gap still exists between the potential and realized utilization of RL in financial trading. In particular, orchestrating an RLFT project lifecycle poses challenges in engineering (i.e. hard to build), benchmarking (i.e. hard to compare) and usability (i.e. hard to optimize, maintain and use). To overcome these challenges, we introduce TradeMaster, a holistic open-source RLFT platform that serves as a i) software toolkit, ii) empirical benchmark, and iii) user interface. Our ultimate goal is to provide infrastructures for transparent and reproducible RLFT research and facilitate their real-world deployment with industry impact. TradeMaster will be updated continuously and welcomes contributions from both RL and finance communities.

        ----

        ## [2576] Towards Optimal Caching and Model Selection for Large Model Inference

        **Authors**: *Banghua Zhu, Ying Sheng, Lianmin Zheng, Clark W. Barrett, Michael I. Jordan, Jiantao Jiao*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b914a8fcea5c176cf1ed75c762ce27fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b914a8fcea5c176cf1ed75c762ce27fd-Abstract-Conference.html)

        **Abstract**:

        Large Language Models (LLMs) and other large foundation models have achieved impressive results, but their size exacerbates existing resource consumption and latency challenges. In particular, the large-scale deployment of these models is hindered by the significant resource requirements during inference. In this paper, we study  two  approaches for mitigating these challenges: employing a cache to store previous queries and learning a model selector to choose from an ensemble of models for query processing.Theoretically, we provide an optimal algorithm for jointly optimizing both approaches  to reduce the inference cost in both offline and online tabular settings. By combining a caching algorithm, namely Greedy Dual Size with Frequency (GDSF) or Least Expected Cost (LEC), with a model selector, we achieve optimal rates in both offline and online settings. Empirically, simulations show that our caching and model selection algorithm greatly improves over the baselines, with up to $50\times$ improvement over the baseline when the ratio between the maximum cost and minimum cost is $100$.  Experiments on real datasets show a $4.3\times$ improvement in FLOPs over the baseline when the ratio for FLOPs is $10$, and a $1.8\times$ improvement in latency when the ratio for average latency is $1.85$.

        ----

        ## [2577] Deep Non-line-of-sight Imaging from Under-scanning Measurements

        **Authors**: *Yue Li, Yueyi Zhang, Juntian Ye, Feihu Xu, Zhiwei Xiong*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b91cc0a242e6518ee731f74e82b2eebd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b91cc0a242e6518ee731f74e82b2eebd-Abstract-Conference.html)

        **Abstract**:

        Active confocal non-line-of-sight (NLOS) imaging has successfully enabled seeing around corners relying on high-quality transient measurements. However, acquiring spatial-dense transient measurement is time-consuming, raising the question of how to reconstruct satisfactory results from under-scanning measurements (USM). The existing solutions, involving the traditional algorithms, however, are hindered by unsatisfactory results or long computing times. To this end, we propose the first deep-learning-based approach to NLOS imaging from USM. Our proposed end-to-end network is composed of two main components: the transient recovery network (TRN) and the volume reconstruction network (VRN). Specifically, TRN takes the under-scanning measurements as input, utilizes a multiple kernel feature extraction module and a multiple feature fusion module, and outputs sufficient-scanning measurements at the high-spatial resolution. Afterwards, VRN incorporates the linear physics prior of the light-path transport model and reconstructs the hidden volume representation. Besides, we introduce regularized constraints that enhance the perception of more local details while suppressing smoothing effects. The proposed method achieves superior performance on both synthetic data and public real-world data, as demonstrated by extensive experimental results with different under-scanning grids. Moreover, the proposed method delivers impressive robustness at an extremely low scanning grid (i.e., 8$\times$8) and offers high-speed inference (i.e., 50 times faster than the existing iterative solution).

        ----

        ## [2578] Learning Adversarial Low-rank Markov Decision Processes with Unknown Transition and Full-information Feedback

        **Authors**: *Canzhe Zhao, Ruofeng Yang, Baoxiang Wang, Xuezhou Zhang, Shuai Li*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b93fda2862db7a7ac4a5c412adfb1ac2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b93fda2862db7a7ac4a5c412adfb1ac2-Abstract-Conference.html)

        **Abstract**:

        In this work, we study the low-rank MDPs with adversarially changed losses in the full-information feedback setting. In particular, the unknown transition probability kernel admits a low-rank matrix decomposition \citep{REPUCB22}, and the loss functions may change adversarially but are revealed to the learner at the end of each episode. We propose a policy optimization-based algorithm POLO, and we prove that it attains the $\widetilde{O}(K^{\frac{5}{6}}A^{\frac{1}{2}}d\ln(1+M)/(1-\gamma)^2)$ regret guarantee, where $d$ is rank of the transition kernel (and hence the dimension of the unknown representations), $A$ is the cardinality of the action space, $M$ is the cardinality of the model class that contains all the plausible representations, and $\gamma$ is the discounted factor. Notably, our algorithm is oracle-efficient and has a regret guarantee with no dependence on the size of potentially arbitrarily large state space. Furthermore, we also prove an $\Omega(\frac{\gamma^2}{1-\gamma} \sqrt{d A K})$ regret lower bound for this problem, showing that low-rank MDPs are statistically more difficult to learn than linear MDPs in the regret minimization setting. To the best of our knowledge, we present the first algorithm that interleaves representation learning, exploration, and exploitation to achieve the sublinear regret guarantee for RL with nonlinear function approximation and adversarial losses.

        ----

        ## [2579] UE4-NeRF: Neural Radiance Field for Real-Time Rendering of Large-Scale Scene

        **Authors**: *Jiaming Gu, Minchao Jiang, Hongsheng Li, Xiaoyuan Lu, Guangming Zhu, Syed Afaq Ali Shah, Liang Zhang, Mohammed Bennamoun*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b94d8b035e2183e47afef9e2f299ba47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b94d8b035e2183e47afef9e2f299ba47-Abstract-Conference.html)

        **Abstract**:

        Neural Radiance Fields (NeRF) is a novel implicit 3D reconstruction method that shows immense potential and has been gaining increasing attention. It enables the reconstruction of 3D scenes solely from a set of photographs. However, its real-time rendering capability, especially for interactive real-time rendering of large-scale scenes, still has significant limitations. To address these challenges, in this paper, we propose a novel neural rendering system called UE4-NeRF, specifically designed for real-time rendering of large-scale scenes. We partitioned each large scene into different sub-NeRFs. In order to represent the partitioned independent scene, we initialize polygonal meshes by constructing multiple regular octahedra within the scene and  the vertices of the polygonal faces are continuously optimized during the training process. Drawing inspiration from Level of Detail (LOD) techniques, we trained meshes of varying levels of detail for different observation levels. Our approach combines with the rasterization pipeline in Unreal Engine 4 (UE4), achieving real-time rendering of large-scale scenes at 4K resolution with a frame rate of up to 43 FPS. Rendering within UE4 also facilitates  scene editing in subsequent stages. Furthermore, through experiments, we have demonstrated that our method achieves rendering quality comparable to state-of-the-art approaches. Project page: https://jamchaos.github.io/UE4-NeRF/.

        ----

        ## [2580] H2RBox-v2: Incorporating Symmetry for Boosting Horizontal Box Supervised Oriented Object Detection

        **Authors**: *Yi Yu, Xue Yang, Qingyun Li, Yue Zhou, Feipeng Da, Junchi Yan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html)

        **Abstract**:

        With the rapidly increasing demand for oriented object detection, e.g. in autonomous driving and remote sensing, the recently proposed paradigm involving weakly-supervised detector H2RBox for learning rotated box (RBox) from the more readily-available horizontal box (HBox) has shown promise. This paper presents H2RBox-v2, to further bridge the gap between HBox-supervised and RBox-supervised oriented object detection. Specifically, we propose to leverage the reflection symmetry via flip and rotate consistencies, using a weakly-supervised network branch similar to H2RBox, together with a novel self-supervised branch that learns orientations from the symmetry inherent in visual objects. The detector is further stabilized and enhanced by practical techniques to cope with peripheral issues e.g. angular periodicity. To our best knowledge, H2RBox-v2 is the first symmetry-aware self-supervised paradigm for oriented object detection. In particular, our method shows less susceptibility to low-quality annotation and insufficient training data compared to H2RBox. Specifically, H2RBox-v2 achieves very close performance to a rotation annotation trained counterpart -- Rotated FCOS: 1) DOTA-v1.0/1.5/2.0: 72.31%/64.76%/50.33% vs. 72.44%/64.53%/51.77%; 2) HRSC: 89.66% vs. 88.99%; 3) FAIR1M: 42.27% vs. 41.25%.

        ----

        ## [2581] The Waymo Open Sim Agents Challenge

        **Authors**: *Nico Montali, John Lambert, Paul Mougin, Alex Kuefler, Nicholas Rhinehart, Michelle Li, Cole Gulino, Tristan Emrich, Zoey Yang, Shimon Whiteson, Brandyn White, Dragomir Anguelov*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Simulation with realistic, interactive agents represents a key task for autonomous vehicle software development. In this work, we introduce the Waymo Open Sim Agents Challenge (WOSAC). WOSAC is the first public challenge to tackle this task and propose corresponding metrics. The goal of the challenge is to stimulate the design of realistic simulators that can be used to evaluate and train a behavior model for autonomous driving. We outline our evaluation methodology, present results for a number of different baseline simulation agent methods, and analyze several submissions to the 2023 competition which ran from March 16, 2023 to May 23, 2023. The WOSAC evaluation server remains open for submissions and we discuss open problems for the task.

        ----

        ## [2582] Online RL in Linearly qπ-Realizable MDPs Is as Easy as in Linear MDPs If You Learn What to Ignore

        **Authors**: *Gellért Weisz, András György, Csaba Szepesvári*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b973a107336177a274069cefb011244c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b973a107336177a274069cefb011244c-Abstract-Conference.html)

        **Abstract**:

        We consider online reinforcement learning (RL) in episodic Markov decision processes (MDPs) under the linear $q^\pi$-realizability assumption, where it is assumed that the action-values of all policies can be  expressed as linear functions of state-action features. This class is known to be more general than  linear MDPs, where the transition kernel and the reward function are assumed to be linear functions of the feature vectors. As our first contribution, we show that the difference between the two classes is the presence of states in linearly $q^\pi$-realizable MDPs where for any policy, all the actions have  approximately equal values, and skipping over these states by following an arbitrarily fixed policy in those states transforms the problem to a linear MDP. Based on this observation, we derive a novel (computationally inefficient) learning algorithm for linearly $q^\pi$-realizable MDPs that simultaneously learns what states should be skipped over and runs another learning algorithm on the linear MDP hidden in the problem. The method returns an $\epsilon$-optimal policy after $\text{polylog}(H, d)/\epsilon^2$ interactions with the MDP, where $H$ is the time horizon and $d$ is the dimension of the feature vectors, giving the first polynomial-sample-complexity online RL algorithm for this setting. The results are proved for the misspecified case, where the sample complexity is shown to degrade gracefully with the misspecification error.

        ----

        ## [2583] On Transfer of Adversarial Robustness from Pretraining to Downstream Tasks

        **Authors**: *Laura Fee Nern, Harsh Raj, Maurice André Georgi, Yash Sharma*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9801626a6ffaf6664af1e983dbd0094-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9801626a6ffaf6664af1e983dbd0094-Abstract-Conference.html)

        **Abstract**:

        As large-scale training regimes have gained popularity, the use of pretrained models for downstream tasks has become common practice in machine learning. While pretraining has been shown to enhance the performance of models in practice, the transfer of robustness properties from pretraining to downstream tasks remains poorly understood. In this study, we demonstrate that the robustness of a linear predictor on downstream tasks can be constrained by the robustness of its underlying representation, regardless of the protocol used for pretraining. We prove (i) a bound on the loss that holds independent of any downstream task, as well as (ii) a criterion for robust classification in particular. We validate our theoretical results in practical applications, show how our results can be used for calibrating expectations of downstream robustness, and when our results are useful for optimal transfer learning. Taken together, our results offer an initial step towards characterizing the requirements of the representation function for reliable post-adaptation performance.

        ----

        ## [2584] Entropy-dissipation Informed Neural Network for McKean-Vlasov Type PDEs

        **Authors**: *Zebang Shen, Zhenfu Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9a17133e3943509243b5e197c1c23b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9a17133e3943509243b5e197c1c23b2-Abstract-Conference.html)

        **Abstract**:

        The McKean-Vlasov equation (MVE) describes the collective behavior of particles subject to drift, diffusion, and mean-field interaction. In physical systems, the interaction term can be singular, i.e. it diverges when two particles collide. Notable examples of such interactions include the Coulomb interaction, fundamental in plasma physics, and the Biot-Savart interaction, present in the vorticity formulation of the 2D Navier-Stokes equation (NSE) in fluid dynamics. Solving MVEs that involve singular interaction kernels presents a significant challenge, especially when aiming to provide rigorous theoretical guarantees. In this work, we propose a novel approach based on the concept of entropy dissipation in the underlying system. We derive a potential function that effectively controls the KL divergence between a hypothesis solution and the ground truth. Building upon this theoretical foundation, we introduce the Entropy-dissipation Informed Neural Network (EINN) framework for solving MVEs. In EINN, we utilize neural networks (NN) to approximate the underlying velocity field and minimize the proposed potential function. By leveraging the expressive power of NNs, our approach offers a promising avenue for tackling the complexities associated with singular interactions. To assess the empirical performance of our method, we compare EINN with SOTA NN-based MVE solvers. The results demonstrate the effectiveness of our approach in solving MVEs across various example problems.

        ----

        ## [2585] Revisiting Visual Model Robustness: A Frequency Long-Tailed Distribution View

        **Authors**: *Zhiyu Lin, Yifei Gao, Yunfan Yang, Jitao Sang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9a4d7b88a41652c63962ebcc21701b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9a4d7b88a41652c63962ebcc21701b7-Abstract-Conference.html)

        **Abstract**:

        A widely discussed hypothesis regarding the cause of visual models' lack of robustness is that they can exploit human-imperceptible high-frequency components (HFC) in images, which in turn leads to model vulnerabilities, such as the adversarial examples. However, (1) inconsistent findings regarding the validation of this hypothesis reflect in a limited understanding of HFC, and (2) solutions inspired by the hypothesis tend to involve a robustness-accuracy trade-off and leaning towards suppressing the model's learning on HFC. In this paper, inspired by the long-tailed characteristic observed in frequency spectrum, we first formally define the HFC from long-tailed perspective and then revisit the relationship between HFC and model robustness. In the frequency long-tailed scenario, experimental results on common datasets and various network structures consistently indicate that models in standard training exhibit high sensitivity to HFC. We investigate the reason of the sensitivity, which reflects in model's under-fitting behavior on HFC. Furthermore, the cause of the model's under-fitting behavior is attributed to the limited information content in HFC. Based on these findings, we propose a Balance Spectrum Sampling (BaSS) strategy, which effectively counteracts the long-tailed effect and enhances the model's learning on HFC. Extensive experimental results demonstrate that our method achieves a substantially better robustness-accuracy trade-off when combined with existing defense methods, while also indicating the potential of encouraging HFC learning in improving model performance.

        ----

        ## [2586] How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization

        **Authors**: *Hai Zhang, Hang Yu, Junqiao Zhao, Di Zhang, Xiao Zhang, Hongtu Zhou, Chang Huang, Chen Ye*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9b4f084b2e6709a2bfad0f601271aec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9b4f084b2e6709a2bfad0f601271aec-Abstract-Conference.html)

        **Abstract**:

        Designing and deriving effective model-based reinforcement learning (MBRL) algorithms with a performance improvement guarantee is challenging, mainly attributed to the high coupling between model learning and policy optimization. Many prior methods that rely on return discrepancy to guide model learning ignore the impacts of model shift, which can lead to performance deterioration due to excessive model updates. Other methods use performance difference bound to explicitly consider model shift. However, these methods rely on a fixed threshold to constrain model shift, resulting in a heavy dependence on the threshold and a lack of adaptability during the training process. In this paper, we theoretically derive an optimization objective that can unify model shift and model bias and then formulate a fine-tuning process. This process adaptively adjusts the model updates to get a performance improvement guarantee while avoiding model overfitting. Based on these, we develop a straightforward algorithm USB-PO (Unified model Shift and model Bias Policy Optimization). Empirical results show that USB-PO achieves state-of-the-art performance on several challenging benchmark tasks.

        ----

        ## [2587] Dynamic Pricing and Learning with Bayesian Persuasion

        **Authors**: *Shipra Agrawal, Yiding Feng, Wei Tang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9c2e8a0bbed5fcfaf62856a3a719ada-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9c2e8a0bbed5fcfaf62856a3a719ada-Abstract-Conference.html)

        **Abstract**:

        We consider a novel dynamic pricing and learning setting where in addition to setting prices of products in sequential rounds, the seller also ex-ante commits to ‘advertising schemes’. That is, in the beginning of each round the seller can decide what kind of signal they will provide to the buyer about the product’s quality upon realization. Using the popular Bayesian persuasion framework to model the effect of these signals on the buyers’ valuation and purchase responses, we formulate the problem of finding an optimal design of the advertising scheme along with a pricing scheme that maximizes the seller’s expected revenue. Without any apriori knowledge of the buyers’ demand function, our goal is to design an online algorithm that can use past purchase responses to adaptively learn the optimal pricing and advertising strategy. We study the regret of the algorithm when compared to the optimal clairvoyant price and advertisingscheme. Our main result is a computationally efficient online algorithm that achieves an $O(T^{2/3}(m \log T )^{1/3})$ regret bound when the valuation function is linear in the product quality. Here $m$ is the cardinality of the discrete product quality domain and $T$ is the time horizon. This result requires some natural monotonicity and Lipschitz assumptions on the valuation function, but no Lipschitz or smoothness assumption on the buyers’ demand function. For constant $m$, our result matches the regret lower bound for dynamic pricing within logarithmic factors, which is a special case of our problem. We also obtain several improved results for the widely considered special case of additive valuations, including an $\tilde{O}(T^{2/3})$ regret bound independent of $m$ when $m\le T^{1/3}$.

        ----

        ## [2588] RH-BrainFS: Regional Heterogeneous Multimodal Brain Networks Fusion Strategy

        **Authors**: *Hongting Ye, Yalu Zheng, Yueying Li, Ke Zhang, Youyong Kong, Yonggui Yuan*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9c353d02e565f0f7cba94c4f3584eaa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9c353d02e565f0f7cba94c4f3584eaa-Abstract-Conference.html)

        **Abstract**:

        Multimodal fusion has become an important research technique in neuroscience that completes downstream tasks by extracting complementary information from multiple modalities. Existing multimodal research on brain networks mainly focuses on two modalities, structural connectivity (SC) and functional connectivity (FC). Recently, extensive literature has shown that the relationship between SC and FC is complex and not a simple one-to-one mapping. The coupling of structure and function at the regional level is heterogeneous. However, all previous studies have neglected the modal regional heterogeneity between SC and FC and fused their representations via "simple patterns", which are inefficient ways of multimodal fusion and affect the overall performance of the model. In this paper, to alleviate the issue of regional heterogeneity of multimodal brain networks, we propose a novel Regional Heterogeneous multimodal Brain networks Fusion Strategy (RH-BrainFS). Briefly, we introduce a brain subgraph networks module to extract regional characteristics of brain networks, and further use a new transformer-based fusion bottleneck module to alleviate the issue of regional heterogeneity between SC and FC. To the best of our knowledge, this is the first paper to explicitly state the issue of structural-functional modal regional heterogeneity and to propose asolution. Extensive experiments demonstrate that the proposed method outperforms several state-of-the-art methods in a variety of neuroscience tasks.

        ----

        ## [2589] To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis

        **Authors**: *Fuzhao Xue, Yao Fu, Wangchunshu Zhou, Zangwei Zheng, Yang You*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html)

        **Abstract**:

        Recent research has highlighted the importance of dataset size in scaling language models. However, large language models (LLMs) are notoriously token-hungry during pre-training, and high-quality text data on the web is likely to be approaching its scaling limit for LLMs. To further enhance LLMs, a straightforward approach is to repeat the pre-training data for additional epochs. In this study, we empirically investigate three key aspects under this approach. First, we explore the consequences of repeating pre-training data, revealing that the model is susceptible to overfitting, leading to multi-epoch degradation. Second, we examine the key factors contributing to multi-epoch degradation, finding that significant factors include dataset size, model parameters, and training objectives, while less influential factors consist of dataset quality and model FLOPs. Finally, we explore whether widely used regularization can alleviate multi-epoch degradation. Most regularization techniques do not yield significant improvements, except for dropout, which demonstrates remarkable effectiveness but requires careful tuning when scaling up the model size. Additionally, we discover that leveraging mixture-of-experts (MoE) enables cost-effective and efficient hyper-parameter tuning for computationally intensive dense LLMs with comparable trainable parameters, potentially impacting efficient LLM development on a broader scale.

        ----

        ## [2590] A*Net: A Scalable Path-based Reasoning Approach for Knowledge Graphs

        **Authors**: *Zhaocheng Zhu, Xinyu Yuan, Michael Galkin, Louis-Pascal A. C. Xhonneux, Ming Zhang, Maxime Gazeau, Jian Tang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html)

        **Abstract**:

        Reasoning on large-scale knowledge graphs has been long dominated by embedding methods. While path-based methods possess the inductive capacity that embeddings lack, their scalability is limited by the exponential number of paths. Here we present A*Net, a scalable path-based method for knowledge graph reasoning. Inspired by the A* algorithm for shortest path problems, our A*Net learns a priority function to select important nodes and edges at each iteration, to reduce time and memory footprint for both training and inference. The ratio of selected nodes and edges can be specified to trade off between performance and efficiency. Experiments on both transductive and inductive knowledge graph reasoning benchmarks show that A*Net achieves competitive performance with existing state-of-the-art path-based methods, while merely visiting 10% nodes and 10% edges at each iteration. On a million-scale dataset ogbl-wikikg2, A*Net not only achieves a new state-of-the-art result, but also converges faster than embedding methods. A*Net is the first path-based method for knowledge graph reasoning at such scale.

        ----

        ## [2591] HASSOD: Hierarchical Adaptive Self-Supervised Object Detection

        **Authors**: *Shengcao Cao, Dhiraj Joshi, Liangyan Gui, Yu-Xiong Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9ecf4d84999a61783c360c3782e801e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9ecf4d84999a61783c360c3782e801e-Abstract-Conference.html)

        **Abstract**:

        The human visual perception system demonstrates exceptional capabilities in learning without explicit supervision and understanding the part-to-whole composition of objects. Drawing inspiration from these two abilities, we propose Hierarchical Adaptive Self-Supervised Object Detection (HASSOD), a novel approach that learns to detect objects and understand their compositions without human supervision. HASSOD employs a hierarchical adaptive clustering strategy to group regions into object masks based on self-supervised visual representations, adaptively determining the number of objects per image. Furthermore, HASSOD identifies the hierarchical levels of objects in terms of composition, by analyzing coverage relations between masks and constructing tree structures. This additional self-supervised learning task leads to improved detection performance and enhanced interpretability. Lastly, we abandon the inefficient multi-round self-training process utilized in prior methods and instead adapt the Mean Teacher framework from semi-supervised learning, which leads to a smoother and more efficient training process. Through extensive experiments on prevalent image datasets, we demonstrate the superiority of HASSOD over existing methods, thereby advancing the state of the art in self-supervised object detection. Notably, we improve Mask AR from 20.2 to 22.5 on LVIS, and from 17.0 to 26.0 on SA-1B. Project page: https://HASSOD-NeurIPS23.github.io.

        ----

        ## [2592] Addressing the speed-accuracy simulation trade-off for adaptive spiking neurons

        **Authors**: *Luke Taylor, Andrew King, Nicol S. Harper*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9f253c2758a323f9d2095f91de9a974-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9f253c2758a323f9d2095f91de9a974-Abstract-Conference.html)

        **Abstract**:

        The adaptive leaky integrate-and-fire (ALIF) model is fundamental within computational neuroscience and has been instrumental in studying our brains $\textit{in silico}$. Due to the sequential nature of simulating these neural models, a commonly faced issue is the speed-accuracy trade-off: either accurately simulate a neuron using a small discretisation time-step (DT), which is slow, or more quickly simulate a neuron using a larger DT and incur a loss in simulation accuracy. Here we provide a solution to this dilemma, by algorithmically reinterpreting the ALIF model, reducing the sequential simulation complexity and permitting a more efficient parallelisation on GPUs. We computationally validate our implementation to obtain over a $50\times$ training speedup using small DTs on synthetic benchmarks. We also obtained a comparable performance to the standard ALIF implementation on different supervised classification tasks - yet in a fraction of the training time. Lastly, we showcase how our model makes it possible to quickly and accurately fit real electrophysiological recordings of cortical neurons, where very fine sub-millisecond DTs are crucial for capturing exact spike timing.

        ----

        ## [2593] Re-Think and Re-Design Graph Neural Networks in Spaces of Continuous Graph Diffusion Functionals

        **Authors**: *Tingting Dan, Jiaqi Ding, Ziquan Wei, Shahar Z. Kovalsky, Minjeong Kim, Won Hwa Kim, Guorong Wu*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/b9fd027eb16434174b8bb3d3b18110af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/b9fd027eb16434174b8bb3d3b18110af-Abstract-Conference.html)

        **Abstract**:

        Graphs are ubiquitous in various domains, such as social networks and biological systems. Despite the great successes of graph neural networks (GNNs) in modeling and analyzing complex graph data, the inductive bias of locality assumption, which involves exchanging information only within neighboring connected nodes, restricts GNNs in capturing long-range dependencies and global patterns in graphs. Inspired by the classic Brachistochrone problem, we seek how to devise a new inductive bias for cutting-edge graph application and present a general framework through the lens of variational analysis. The backbone of our framework is a two-way mapping between the discrete GNN model and continuous diffusion functional, which allows us to design application-specific objective function in the continuous domain and engineer discrete deep model with mathematical guarantees. First, we address over-smoothing in current GNNs. Specifically, our inference reveals that the existing layer-by-layer models of graph embedding learning are equivalent to a ${\ell _2}$-norm integral functional of graph gradients, which is the underlying cause of the over-smoothing problem. Similar to edge-preserving filters in image denoising, we introduce the total variation (TV) to promote alignment of the graph diffusion pattern with the global information present in community topologies. On top of this, we devise a new selective mechanism for inductive bias that can be easily integrated into existing GNNs and effectively address the trade-off between model depth and over-smoothing. Second, we devise a novel generative adversarial network (GAN) to predict the spreading flows in the graph through a neural transport equation. To avoid the potential issue of vanishing flows, we tailor the objective function to minimize the transportation within each community while maximizing the inter-community flows. Our new GNN models achieve state-of-the-art (SOTA) performance on graph learning benchmarks such as Cora, Citeseer, and Pubmed.

        ----

        ## [2594] Adapting Fairness Interventions to Missing Values

        **Authors**: *Raymond Feng, Flávio Calmon, Hao Wang*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba0ad9d1e0c737800b2340b9cd68c208-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba0ad9d1e0c737800b2340b9cd68c208-Abstract-Conference.html)

        **Abstract**:

        Missing values in real-world data pose a significant and unique challenge to algorithmic fairness. Different demographic groups may be unequally affected by missing data, and the standard procedure for handling missing values where first data is imputed, then the imputed data is used for classification—a procedure referred to as "impute-then-classify"—can exacerbate discrimination. In this paper, we analyze how missing values affect algorithmic fairness. We first prove that training a classifier from imputed data can significantly worsen the achievable values of group fairness and average accuracy. This is because imputing data results in the loss of the missing pattern of the data, which often conveys information about the predictive label. We present scalable and adaptive algorithms for fair classification with missing values. These algorithms can be combined with any preexisting fairness-intervention algorithm to handle all possible missing patterns while preserving information encoded within the missing patterns. Numerical experiments with state-of-the-art fairness interventions demonstrate that our adaptive algorithms consistently achieve higher fairness and accuracy than impute-then-classify across different datasets.

        ----

        ## [2595] Probabilistic Weight Fixing: Large-scale training of neural network weight uncertainties for quantisation

        **Authors**: *Christopher Subia-Waud, Srinandan Dasmahapatra*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba178fab60f9306a0b2d7ec8973715a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba178fab60f9306a0b2d7ec8973715a6-Abstract-Conference.html)

        **Abstract**:

        Weight-sharing quantization has emerged as a technique to reduce energy expenditure during inference in large neural networks by constraining their weights to a limited set of values. However, existing methods often assume weights are treated solely based on value, neglecting the unique role of weight position. This paper proposes a probabilistic framework based on Bayesian neural networks (BNNs) and a variational relaxation to identify which weights can be moved to which cluster center and to what degree based on their individual position-specific learned uncertainty distributions. We introduce a new initialization setting and a regularization term, enabling the training of BNNs with complex dataset-model combinations. Leveraging the flexibility of weight values from probability distributions, we enhance noise resilience and compressibility. Our iterative clustering procedure demonstrates superior compressibility and higher accuracy compared to state-of-the-art methods on both ResNet models and the more complex transformer-based architectures. In particular, our method outperforms the state-of-the-art quantization method top-1 accuracy by 1.6\% on ImageNet using DeiT-Tiny, with its 5 million+ weights now represented by only 296 unique values.  Code available at https://github.com/subiawaud/PWFN.

        ----

        ## [2596] PID-Inspired Inductive Biases for Deep Reinforcement Learning in Partially Observable Control Tasks

        **Authors**: *Ian Char, Jeff Schneider*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html)

        **Abstract**:

        Deep reinforcement learning (RL) has shown immense potential for learning to control systems through data alone. However, one challenge deep RL faces is that the full state of the system is often not observable. When this is the case, the policy needs to leverage the history of observations to infer the current state. At the same time, differences between the training and testing environments makes it critical for the policy not to overfit to the sequence of observations it sees at training time. As such, there is an important balancing act between having the history encoder be flexible enough to extract relevant information, yet be robust to changes in the environment. To strike this balance, we look to the PID controller for inspiration. We assert the PID controller's success shows that only summing and differencing are needed to accumulate information over time for many control tasks. Following this principle, we propose two architectures for encoding history: one that directly uses PID features and another that extends these core ideas and can be used in arbitrary control tasks. When compared with prior approaches, our encoders produce policies that are often more robust and achieve better performance on a variety of tracking tasks. Going beyond tracking tasks, our policies achieve 1.7x better performance on average over previous state-of-the-art methods on a suite of locomotion control tasks.

        ----

        ## [2597] SustainGym: Reinforcement Learning Environments for Sustainable Energy Systems

        **Authors**: *Christopher Yeh, Victor Li, Rajeev Datta, Julio Arroyo, Nicolas Christianson, Chi Zhang, Yize Chen, Mohammad Mehdi Hosseini, Azarang Golmohammadi, Yuanyuan Shi, Yisong Yue, Adam Wierman*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba74855789913e5ed36f87288af79e5b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba74855789913e5ed36f87288af79e5b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The lack of standardized benchmarks for reinforcement learning (RL) in sustainability applications has made it difficult to both track progress on specific domains and identify bottlenecks for researchers to focus their efforts. In this paper, we present SustainGym, a suite of five environments designed to test the performance of RL algorithms on realistic sustainable energy system tasks, ranging from electric vehicle charging to carbon-aware data center job scheduling. The environments test RL algorithms under realistic distribution shifts as well as in multi-agent settings. We show that standard off-the-shelf RL algorithms leave significant room for improving performance and highlight the challenges ahead for introducing RL to real-world sustainability tasks.

        ----

        ## [2598] Policy Gradient for Rectangular Robust Markov Decision Processes

        **Authors**: *Navdeep Kumar, Esther Derman, Matthieu Geist, Kfir Y. Levy, Shie Mannor*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba8aee784ffe0813890288b334444eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba8aee784ffe0813890288b334444eda-Abstract-Conference.html)

        **Abstract**:

        Policy gradient methods have become a standard for training reinforcement learning agents in a scalable and efficient manner. However, they do not account for transition uncertainty, whereas learning robust policies can be computationally expensive. In this paper, we introduce robust policy gradient (RPG), a policy-based method that efficiently solves rectangular robust Markov decision processes (MDPs). We provide a closed-form expression for the worst occupation measure. Incidentally, we find that the worst kernel is a rank-one perturbation of the nominal. Combining the worst occupation measure with a robust Q-value estimation yields an explicit form of the robust gradient. Our resulting RPG can be estimated from data with the same time complexity as its non-robust equivalent. Hence, it relieves the computational burden of convex optimization problems required for training robust policies by current policy gradient approaches.

        ----

        ## [2599] MultiFusion: Fusing Pre-Trained Models for Multi-Lingual, Multi-Modal Image Generation

        **Authors**: *Marco Bellagente, Manuel Brack, Hannah Teufel, Felix Friedrich, Björn Deiseroth, Constantin Eichenberg, Andrew Dai, Robert Baldock, Souradeep Nanda, Koen Oostermeijer, Andrés Felipe Cruz-Salinas, Patrick Schramowski, Kristian Kersting, Samuel Weinbach*

        **Conference**: *nips 2023*

        **URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ba8d1b46292c5e82cbfb3b3dc3b968af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ba8d1b46292c5e82cbfb3b3dc3b968af-Abstract-Conference.html)

        **Abstract**:

        The recent popularity of text-to-image diffusion models (DM) can largely be attributed to the intuitive interface they provide to users. The intended generation can be expressed in natural language, with the model producing faithful interpretations of text prompts. However, expressing complex or nuanced ideas in text alone can be difficult. To ease image generation, we propose MultiFusion that allows one to express complex and nuanced concepts with arbitrarily interleaved inputs of multiple modalities and languages. MultiFusion leverages pre-trained models and aligns them for integration into a cohesive system, thereby avoiding the need for extensive training from scratch. Our experimental results demonstrate the efficient transfer of capabilities from individual modules to the downstream model. Specifically, the fusion of all independent components allows the image generation module to utilize multilingual, interleaved multimodal inputs despite being trained solely on monomodal data in a single language.

        ----

        

[Go to the previous page](NIPS-2023-list12.md)

[Go to the next page](NIPS-2023-list14.md)

[Go to the catalog section](README.md)