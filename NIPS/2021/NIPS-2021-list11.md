## [2000] Fast Bayesian Inference for Gaussian Cox Processes via Path Integral Formulation

        **Authors**: *Hideaki Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dba31bb5c75992690f20c2d3b370ec7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dba31bb5c75992690f20c2d3b370ec7c-Abstract.html)

        **Abstract**:

        Gaussian Cox processes are widely-used point process models that use a Gaussian process to describe the Bayesian a priori uncertainty present in latent intensity functions. In this paper, we propose a novel Bayesian inference scheme for Gaussian Cox processes by exploiting a conceptually-intuitive {Â¥it path integral} formulation. The proposed scheme does not rely on domain discretization, scales linearly with the number of observed events, has a lower complexity than the state-of-the-art variational Bayesian schemes with respect to the number of inducing points, and is applicable to a wide range of Gaussian Cox processes with various types of link functions. Our scheme is especially beneficial under the multi-dimensional input setting, where the number of inducing points tends to be large. We evaluate our scheme on synthetic and real-world data, and show that it achieves comparable predictive accuracy while being tens of times faster than reference methods.

        ----

        ## [2001] Lattice partition recovery with dyadic CART

        **Authors**: *Oscar Hernan Madrid Padilla, Yi Yu, Alessandro Rinaldo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dba4c1a117472f6aca95211285d0587e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dba4c1a117472f6aca95211285d0587e-Abstract.html)

        **Abstract**:

        We study piece-wise constant signals corrupted by additive Gaussian noise over a $d$-dimensional lattice. Data of this form naturally arise in a host of applications, and the tasks of signal detection or testing, de-noising and  estimation have been studied extensively in the statistical and signal processing literature. In this paper we consider instead the problem of partition recovery, i.e.~of estimating the partition of the lattice induced by the constancy regions of the unknown signal, using the computationally-efficient dyadic classification and regression tree (DCART)  methodology proposed by \citep{donoho1997cart}.   We prove that, under appropriate regularity conditions on the shape of the partition elements, a DCART-based procedure consistently estimates the underlying partition at a rate of order $\sigma^2 k^* \log (N)/\kappa^2$, where $k^*$ is the minimal number of rectangular sub-graphs obtained using recursive dyadic partitions supporting the signal partition, $\sigma^2$ is the noise variance, $\kappa$ is the minimal magnitude of the signal difference among contiguous elements of the partition and $N$ is the size of the lattice. Furthermore,  under stronger assumptions, our method attains a sharper estimation error of order $\sigma^2\log(N)/\kappa^2$, independent of $k^*$, which we show to be minimax rate optimal.  Our theoretical guarantees further extend to the partition estimator based on the optimal regression tree estimator (ORT) of \cite{chatterjee2019adaptive} and to the one obtained through an NP-hard exhaustive search method.  We corroborate our theoretical findings and the effectiveness of  DCART for partition recovery in simulations.

        ----

        ## [2002] Robust Deep Reinforcement Learning through Adversarial Loss

        **Authors**: *Tuomas P. Oikarinen, Wang Zhang, Alexandre Megretski, Luca Daniel, Tsui-Wei Weng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dbb422937d7ff56e049d61da730b3e11-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dbb422937d7ff56e049d61da730b3e11-Abstract.html)

        **Abstract**:

        Recent studies have shown that deep reinforcement learning agents are vulnerable to small adversarial perturbations on the agent's inputs, which raises concerns about deploying such agents in the real world. To address this issue, we propose RADIAL-RL, a principled framework to train reinforcement learning agents with improved robustness against $l_p$-norm bounded adversarial attacks. Our framework is compatible with popular deep reinforcement learning algorithms and we demonstrate its performance with deep Q-learning, A3C and PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to show the effectiveness of our robust training algorithm. Our RADIAL-RL agents consistently outperform prior methods when tested against attacks of varying strength and are more computationally efficient to train. In addition, we propose a new evaluation method called Greedy-Worst-Case Reward (GWC) to measure attack agnostic robustness of deep RL agents. We show that GWC can be evaluated efficiently and is a good estimate of the reward under the worst possible sequence of adversarial attacks. All code used for our experiments is available at https://github.com/tuomaso/radial_rl_v2.

        ----

        ## [2003] Provable Model-based Nonlinear Bandit and Reinforcement Learning: Shelve Optimism, Embrace Virtual Curvature

        **Authors**: *Kefan Dong, Jiaqi Yang, Tengyu Ma*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dc5d637ed5e62c36ecb73b654b05ba2a-Abstract.html)

        **Abstract**:

        This paper studies model-based bandit and reinforcement learning (RL) with nonlinear function approximations. We propose to study convergence to approximate local maxima because we show that global convergence is statistically intractable even for one-layer neural net bandit with a deterministic reward. For both nonlinear bandit and RL, the paper presents a model-based algorithm, Virtual Ascent with Online Model Learner (ViOlin), which provably converges to a local maximum with sample complexity that only depends on the sequential Rademacher complexity of the model class. Our results imply novel global or local regret bounds on several concrete settings such as linear bandit with finite or sparse model class, and two-layer neural net bandit. A key algorithmic insight is that optimism may lead to over-exploration even for two-layer neural net model class. On the other hand, for convergence to local maxima, it suffices to maximize the virtual return if the model can also reasonably predict the gradient and Hessian of the real return.

        ----

        ## [2004] You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection

        **Authors**: *Yuxin Fang, Bencheng Liao, Xinggang Wang, Jiemin Fang, Jiyang Qi, Rui Wu, Jianwei Niu, Wenyu Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dc912a253d1e9ba40e2c597ed2376640-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dc912a253d1e9ba40e2c597ed2376640-Abstract.html)

        **Abstract**:

        Can Transformer perform $2\mathrm{D}$ object- and region-level recognition from a pure sequence-to-sequence perspective with minimal knowledge about the $2\mathrm{D}$ spatial structure? To answer this question, we present You Only Look at One Sequence (YOLOS), a series of object detection models based on the vanilla Vision Transformer with the fewest possible modifications, region priors, as well as inductive biases of the target task. We find that YOLOS pre-trained on the mid-sized ImageNet-$1k$ dataset only can already achieve quite competitive performance on the challenging COCO object detection benchmark, e.g., YOLOS-Base directly adopted from BERT-Base architecture can obtain $42.0$ box AP on COCO val. We also discuss the impacts as well as limitations of current pre-train schemes and model scaling strategies for Transformer in vision through YOLOS. Code and pre-trained models are available at https://github.com/hustvl/YOLOS.

        ----

        ## [2005] Learning to delegate for large-scale vehicle routing

        **Authors**: *Sirui Li, Zhongxia Yan, Cathy Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dc9fa5f217a1e57b8a6adeb065560b38-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dc9fa5f217a1e57b8a6adeb065560b38-Abstract.html)

        **Abstract**:

        Vehicle routing problems (VRPs) form a class of combinatorial problems with wide practical applications. While previous heuristic or learning-based works achieve decent solutions on small problem instances, their performance deteriorates in large problems. This article presents a novel learning-augmented local search framework to solve large-scale VRP. The method iteratively improves the solution by identifying appropriate subproblems and $delegating$ their improvement to a black box subsolver. At each step, we leverage spatial locality to consider only a linear number of subproblems, rather than exponential. We frame subproblem selection as regression and train a Transformer on a generated training set of problem instances. Our method accelerates state-of-the-art VRP solvers by 10x to 100x while achieving competitive solution qualities for VRPs with sizes ranging from 500 to 3000. Learned subproblem selection offers a 1.5x to 2x speedup over heuristic or random selection. Our results generalize to a variety of VRP distributions, variants, and solvers.

        ----

        ## [2006] Effective Meta-Regularization by Kernelized Proximal Regularization

        **Authors**: *Weisen Jiang, James T. Kwok, Yu Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dcc5c249e15c211f21e1da0f3ba66169-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dcc5c249e15c211f21e1da0f3ba66169-Abstract.html)

        **Abstract**:

        We study the problem of meta-learning, which has proved to be advantageous to accelerate learning new tasks with a few samples. The recent approaches based on deep kernels achieve the state-of-the-art performance. However, the regularizers in their base learners are not learnable. In this paper, we propose an algorithm called MetaProx to learn a proximal regularizer for the base learner. We theoretically establish the convergence of MetaProx. Experimental results confirm the advantage of the proposed algorithm.

        ----

        ## [2007] Towards Context-Agnostic Learning Using Synthetic Data

        **Authors**: *Charles Jin, Martin C. Rinard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dccb1c3a558c50d389c24d69a9856730-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dccb1c3a558c50d389c24d69a9856730-Abstract.html)

        **Abstract**:

        We propose a novel setting for learning, where the input domain is the image of a map defined on the product of two sets, one of which completely determines the labels. We derive a new risk bound for this setting that decomposes into a bias and an error term, and exhibits a surprisingly weak dependence on the true labels. Inspired by these results, we present an algorithm aimed at minimizing the bias term by exploiting the ability to sample from each set independently. We apply our setting to visual classification tasks, where our approach enables us to train classifiers on datasets that consist entirely of a single synthetic example of each class. On several standard benchmarks for real-world image classification, we achieve robust performance in the context-agnostic setting, with good generalization to real world domains, whereas training directly on real world data without our techniques yields classifiers that are brittle to perturbations of the background.

        ----

        ## [2008] Minimax Optimal Quantile and Semi-Adversarial Regret via Root-Logarithmic Regularizers

        **Authors**: *Jeffrey Negrea, Blair L. Bilodeau, Nicolò Campolongo, Francesco Orabona, Daniel M. Roy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dcd2f3f312b6705fb06f4f9f1b55b55c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dcd2f3f312b6705fb06f4f9f1b55b55c-Abstract.html)

        **Abstract**:

        Quantile (and, more generally, KL) regret bounds, such as those achieved by NormalHedge (Chaudhuri, Freund, and Hsu 2009) and its variants, relax the goal of competing against the best individual expert to only competing against a majority of experts on adversarial data. More recently, the semi-adversarial paradigm (Bilodeau, Negrea, and Roy 2020) provides an alternative relaxation of adversarial online learning by considering data that may be neither fully adversarial nor stochastic (I.I.D.).  We achieve the minimax optimal regret in both paradigms using FTRL with separate, novel, root-logarithmic regularizers, both of which can be interpreted as yielding variants of NormalHedge. We extend existing KL regret upper bounds, which hold uniformly over target distributions, to possibly uncountable expert classes with arbitrary priors; provide the first full-information lower bounds for quantile regret on finite expert classes (which are tight); and provide an adaptively minimax optimal algorithm for the semi-adversarial paradigm that adapts to the true, unknown constraint faster, leading to uniformly improved regret bounds over existing methods.

        ----

        ## [2009] Gradient-Free Adversarial Training Against Image Corruption for Learning-based Steering

        **Authors**: *Yu Shen, Laura Y. Zheng, Manli Shu, Weizi Li, Tom Goldstein, Ming C. Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dce8af15f064d1accb98887a21029b08-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dce8af15f064d1accb98887a21029b08-Abstract.html)

        **Abstract**:

        We introduce a simple yet effective framework for improving the robustness of learning algorithms against image corruptions for autonomous driving. These corruptions can occur due to both internal (e.g., sensor noises and hardware abnormalities) and external factors (e.g., lighting, weather, visibility, and other environmental effects). Using sensitivity analysis with FID-based parameterization, we propose a novel algorithm exploiting basis perturbations to improve the overall performance of autonomous steering and other image processing tasks, such as classification and detection, for self-driving cars. Our model not only improves the performance on the original dataset, but also achieves significant performance improvement on datasets with multiple and unseen perturbations, up to 87% and 77%, respectively. A comparison between our approach and other SOTA techniques confirms the effectiveness of our technique in improving the robustness of neural network training for learning-based steering and other image processing tasks.

        ----

        ## [2010] Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation

        **Authors**: *Liyuan Xu, Heishiro Kanagawa, Arthur Gretton*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dcf3219715a7c9cd9286f19db46f2384-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dcf3219715a7c9cd9286f19db46f2384-Abstract.html)

        **Abstract**:

        Proxy causal learning (PCL) is a method for estimating the causal effect of treatments on outcomes in the presence of unobserved confounding, using proxies (structured side information) for the confounder. This is achieved via two-stage regression: in the first stage, we model relations among the treatment and proxies; in the second stage, we use this model to learn the effect of treatment on the outcome, given the context provided by the proxies. PCL  guarantees recovery of the true causal effect, subject to identifiability conditions. We propose a novel method for PCL, the deep feature proxy variable method (DFPV), to address the case where the proxies, treatments, and outcomes are high-dimensional and have nonlinear complex relationships, as represented by deep neural network features. We show that DFPV outperforms recent state-of-the-art PCL methods on challenging synthetic benchmarks, including settings involving high dimensional image data. Furthermore, we show that PCL can be applied to off-policy evaluation for the confounded bandit problem, in which DFPV also exhibits competitive performance.

        ----

        ## [2011] Certifying Robustness to Programmable Data Bias in Decision Trees

        **Authors**: *Anna P. Meyer, Aws Albarghouthi, Loris D'Antoni*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dcf531edc9b229acfe0f4b87e1e278dd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dcf531edc9b229acfe0f4b87e1e278dd-Abstract.html)

        **Abstract**:

        Datasets can be biased due to societal inequities, human biases, under-representation of minorities, etc. Our goal is to certify that models produced by a learning algorithm are pointwise-robust to dataset biases. This is a challenging problem: it entails learning models for a large, or even infinite, number of datasets, ensuring that they all produce the same prediction. We focus on decision-tree learning due to the interpretable nature of the models. Our approach allows programmatically specifying \emph{bias models} across a variety of dimensions (e.g., label-flipping or missing data), composing types of bias, and targeting bias towards a specific group. To certify robustness, we use a novel symbolic technique to evaluate a decision-tree learner on a large, or infinite, number of datasets, certifying that each and every dataset produces the same prediction for a specific test point. We evaluate our approach on datasets that are commonly used in the fairness literature, and demonstrate our approach's viability on a range of bias models.

        ----

        ## [2012] TöRF: Time-of-Flight Radiance Fields for Dynamic Scene View Synthesis

        **Authors**: *Benjamin Attal, Eliot Laidlaw, Aaron Gokaslan, Changil Kim, Christian Richardt, James Tompkin, Matthew O'Toole*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dd03de08bfdff4d8ab01117276564cc7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dd03de08bfdff4d8ab01117276564cc7-Abstract.html)

        **Abstract**:

        Neural networks can represent and accurately reconstruct radiance fields for static 3D scenes (e.g., NeRF). Several works extend these to dynamic scenes captured with monocular video, with promising performance. However, the monocular setting is known to be an under-constrained problem, and so methods rely on data-driven priors for reconstructing dynamic content. We replace these priors with measurements from a time-of-flight (ToF) camera, and introduce a neural representation based on an image formation model for continuous-wave ToF cameras. Instead of working with processed depth maps, we model the raw ToF sensor measurements to improve reconstruction quality and avoid issues with low reflectance regions, multi-path interference, and a sensor's limited unambiguous depth range. We show that this approach improves robustness of dynamic scene reconstruction to erroneous calibration and large motions, and discuss the benefits and limitations of integrating RGB+ToF sensors now available on modern smartphones.

        ----

        ## [2013] Sequence-to-Sequence Learning with Latent Neural Grammars

        **Authors**: *Yoon Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dd17e652cd2a08fdb8bf7f68e2ad3814-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dd17e652cd2a08fdb8bf7f68e2ad3814-Abstract.html)

        **Abstract**:

        Sequence-to-sequence learning with neural networks has become the de facto standard for sequence modeling. This approach typically models the local distribution over the next element with a powerful neural network that can condition on arbitrary context. While flexible and performant, these models often require large datasets for training and can fail spectacularly on benchmarks designed to test for compositional generalization. This work explores an alternative, hierarchical approach to sequence-to-sequence learning with synchronous grammars, where each node in the target tree is transduced by a subset of nodes in the source tree. The source and target trees are treated as fully latent and marginalized out during training. We develop a neural parameterization of the grammar which enables parameter sharing over combinatorial structures without the need for manual feature engineering. We apply this latent neural grammar to various domains---a diagnostic language navigation task designed to test for compositional generalization (SCAN), style transfer, and small-scale machine translation---and find that it performs respectably compared to standard baselines.

        ----

        ## [2014] Exploration-Exploitation in Multi-Agent Competition: Convergence with Bounded Rationality

        **Authors**: *Stefanos Leonardos, Georgios Piliouras, Kelly Spendlove*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dd1970fb03877a235d530476eb727dab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dd1970fb03877a235d530476eb727dab-Abstract.html)

        **Abstract**:

        The interplay between exploration and exploitation in competitive multi-agent learning is still far from being well understood. Motivated by this, we study smooth Q-learning, a prototypical learning model that explicitly captures the balance between game rewards and exploration costs. We show that Q-learning always converges to the unique quantal-response equilibrium (QRE), the standard solution concept for games under bounded rationality, in weighted zero-sum polymatrix games with heterogeneous learning agents using positive exploration rates. Complementing recent results about convergence in weighted potential games [16,34], we show that fast convergence of Q-learning in competitive settings obtains regardless of the number of agents and without any need for parameter fine-tuning. As showcased by our experiments in network zero-sum games, these theoretical results provide the necessary guarantees for an algorithmic approach to the currently open problem of equilibrium selection in competitive multi-agent settings.

        ----

        ## [2015] Low-Rank Extragradient Method for Nonsmooth and Low-Rank Matrix Optimization Problems

        **Authors**: *Atara Kaplan, Dan Garber*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dd32544610bf007f0def4abc9b7ff9ef-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dd32544610bf007f0def4abc9b7ff9ef-Abstract.html)

        **Abstract**:

        Low-rank and nonsmooth matrix optimization problems capture many fundamental tasks in statistics and machine learning. While significant progress has been made in recent years in developing efficient methods for \textit{smooth} low-rank optimization problems that avoid maintaining high-rank matrices and computing expensive high-rank SVDs, advances for nonsmooth problems have been slow paced. In this paper we consider standard convex relaxations for such problems. Mainly, we prove that under a natural \textit{generalized strict complementarity} condition and under the relatively mild assumption that the nonsmooth objective can be written as a maximum of smooth functions, the \textit{extragradient method}, when initialized with a "warm-start'' point, converges to an optimal solution with rate $O(1/t)$ while requiring only two \textit{low-rank} SVDs per iteration. We give a precise trade-off between the rank of the SVDs required and the radius of the ball in which we need to initialize the method. We support our theoretical results with empirical experiments on several nonsmooth low-rank matrix recovery tasks, demonstrating that using simple initializations, the extragradient method produces exactly the same iterates when full-rank SVDs are replaced with SVDs of rank that matches the rank of the (low-rank) ground-truth matrix to be recovered.

        ----

        ## [2016] Which Mutual-Information Representation Learning Objectives are Sufficient for Control?

        **Authors**: *Kate Rakelly, Abhishek Gupta, Carlos Florensa, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dd45045f8c68db9f54e70c67048d32e8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dd45045f8c68db9f54e70c67048d32e8-Abstract.html)

        **Abstract**:

        Mutual information (MI) maximization provides an appealing formalism for learning representations of data. In the context of reinforcement learning (RL), such representations can accelerate learning by discarding irrelevant and redundant information, while retaining the information necessary for control. Much prior work on these methods has addressed the practical difficulties of estimating MI from samples of high-dimensional observations, while comparatively less is understood about which MI objectives yield representations that are sufficient for RL from a theoretical perspective. In this paper, we formalize the sufficiency of a state representation for learning and representing the optimal policy, and study several popular MI based objectives through this lens. Surprisingly, we find that two of these objectives can yield insufficient representations given mild and common assumptions on the structure of the MDP. We corroborate our theoretical results with empirical experiments on a simulated game environment with visual observations.

        ----

        ## [2017] A Geometric Perspective towards Neural Calibration via Sensitivity Decomposition

        **Authors**: *Junjiao Tian, Dylan Yung, Yen-Chang Hsu, Zsolt Kira*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dda99de58ff020cfb57fec1404c97003-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dda99de58ff020cfb57fec1404c97003-Abstract.html)

        **Abstract**:

        It is well known that vision classification models suffer from poor calibration in the face of data distribution shifts. In this paper, we take a geometric approach to this problem. We propose Geometric Sensitivity Decomposition (GSD) which decomposes the norm of a sample feature embedding and the angular similarity to a target classifier into an instance-dependent and an instance-independent com-ponent. The instance-dependent component captures the sensitive information about changes in the input while the instance-independent component represents the insensitive information serving solely to minimize the loss on the training dataset. Inspired by the decomposition, we analytically derive a simple extension to current softmax-linear models, which learns to disentangle the two components during training. On several common vision models, the disentangled model out-performs other calibration methods on standard calibration metrics in the face of out-of-distribution (OOD) data and corruption with significantly less complexity. Specifically, we surpass the current state of the art by 30.8% relative improvement on corrupted CIFAR100 in Expected Calibration Error.

        ----

        ## [2018] Towards a Unified Information-Theoretic Framework for Generalization

        **Authors**: *Mahdi Haghifam, Gintare Karolina Dziugaite, Shay Moran, Daniel M. Roy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ddbc86dc4b2fbfd8a62e12096227e068-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ddbc86dc4b2fbfd8a62e12096227e068-Abstract.html)

        **Abstract**:

        In this work, we investigate the expressiveness of the "conditional mutual information" (CMI)  framework of Steinke and Zakynthinou (2020)  and the prospect of using it to provide a unified framework for proving generalization bounds in the realizable setting.  We first demonstrate that one can use this framework to express non-trivial (but sub-optimal) bounds for any learning algorithm that outputs hypotheses from a class of bounded VC dimension.  We then explore two directions of strengthening this bound: (i) Can the CMI framework express optimal bounds for VC classes? (ii) Can the CMI framework be used to analyze algorithms whose output hypothesis space is unrestricted (i.e. has an unbounded VC dimension)?    With respect to Item (i) we prove that the CMI framework yields the optimal bound on the expected risk  of Support Vector Machines (SVMs) for learning halfspaces. This result is an application of our general result showing that stable compression schemes Bousquet al. (2020) of size $k$ have uniformly bounded CMI of order $O(k)$. We further show that an inherent limitation of proper learning of VC classes contradicts the existence of a proper learner with constant CMI, and it implies a negative resolution to an open problem of Steinke and Zakynthinou (2020).  We further study the CMI of empirical risk minimizers (ERMs) of class $H$ and show that it is possible to output all  consistent classifiers (version space) with bounded CMI if and only if $H$ has a bounded star number (Hanneke and Yang (2015)). With respect to Item (ii) we prove a general reduction showing that "leave-one-out" analysis is expressible via the CMI framework. As a corollary we investigate the CMI of the one-inclusion-graph algorithm proposed by Haussler et al. (1994). More generally, we  show that the CMI framework is universal in the sense that for every consistent algorithm and data distribution, the expected risk vanishes as the number of  samples diverges if and only if its evaluated CMI has sublinear growth with the number of samples.

        ----

        ## [2019] Bayesian decision-making under misspecified priors with applications to meta-learning

        **Authors**: *Max Simchowitz, Christopher Tosh, Akshay Krishnamurthy, Daniel J. Hsu, Thodoris Lykouris, Miroslav Dudík, Robert E. Schapire*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ddcbe25988981920c872c1787382f04d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ddcbe25988981920c872c1787382f04d-Abstract.html)

        **Abstract**:

        Thompson sampling and other Bayesian sequential decision-making algorithms are among the most popular approaches to tackle explore/exploit trade-offs in (contextual) bandits.  The choice of prior in these algorithms offers flexibility to encode domain knowledge but can also lead to poor performance when misspecified.  In this paper, we demonstrate that performance degrades gracefully with misspecification.  We prove that the expected reward accrued by Thompson sampling (TS) with a misspecified prior differs by at most $\tilde{O}(H^2 \epsilon)$ from TS with a well-specified prior, where $\epsilon$ is the total-variation distance between priors and $H$ is the learning horizon.  Our bound does not require the prior to have any parametric form.  For priors with bounded support, our bound is independent of the cardinality or structure of the action space, and we show that it is tight up to universal constants in the worst case.Building on our sensitivity analysis, we establish generic PAC guarantees for algorithms in the recently studied Bayesian meta-learning setting and derive corollaries for various families of priors.  Our results generalize along two axes: (1) they apply to a broader family of Bayesian decision-making algorithms, including a Monte-Carlo implementation of the knowledge gradient algorithm (KG), and (2) they apply to Bayesian POMDPs, the most general Bayesian decision-making setting, encompassing contextual bandits as a special case. Through numerical simulations, we illustrate how prior misspecification and the deployment of one-step look-ahead (as in KG) can impact the convergence of meta-learning in multi-armed and contextual bandits with structured and correlated priors.

        ----

        ## [2020] Neural Trees for Learning on Graphs

        **Authors**: *Rajat Talak, Siyi Hu, Lisa Peng, Luca Carlone*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ddf88ea64eaed0f3de5531ac964a0a1a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ddf88ea64eaed0f3de5531ac964a0a1a-Abstract.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have emerged as a flexible and powerful approach for learning over graphs. Despite this success, existing GNNs are constrained by their local message-passing architecture and are provably limited in their expressive power. In this work, we propose a new GNN architecture â€“ the Neural Tree. The neural tree architecture does not perform message passing on the input graph, but on a tree-structured graph, called the H-tree, that is constructed from the input graph. Nodes in the H-tree correspond to subgraphs in the input graph, and they are reorganized in a hierarchical manner such that the parent of a node in the H-tree always corresponds to a larger subgraph in the input graph. We show that the neural tree architecture can approximate any smooth probability distribution function over an undirected graph. We also prove that the number of parameters needed to achieve an $\epsilon$-approximation of the distribution function is exponential in the treewidth of the input graph, but linear in its size. We prove that any continuous G-invariant/equivariant function can be approximated by a nonlinear combination of such probability distribution functions over G. We apply the neural tree to semi-supervised node classification in 3D scene graphs, and show that these theoretical properties translate into significant gains in prediction accuracy, over the more traditional GNN architectures. We also show the applicability of the neural tree architecture to citation networks with large treewidth, by using a graph sub-sampling technique.

        ----

        ## [2021] Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization

        **Authors**: *Pranav Subramani, Nicholas Vadivelu, Gautam Kamath*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ddf9029977a61241841edeae15e9b53f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ddf9029977a61241841edeae15e9b53f-Abstract.html)

        **Abstract**:

        A common pain point in differentially private machine learning is the significant runtime overhead incurred when executing Differentially Private Stochastic Gradient Descent (DPSGD), which may be as large as two orders of magnitude. We thoroughly demonstrate that by exploiting powerful language primitives, including vectorization, just-in-time compilation, and static graph optimization, one can dramatically reduce these overheads, in many cases nearly matching the best non-private running times. These gains are realized in two frameworks: one is JAX, which provides rich support for these primitives through the XLA compiler. We also rebuild core parts of TensorFlow Privacy, integrating more effective vectorization as well as XLA compilation, granting significant memory and runtime improvements over previous release versions. Our proposed approaches allow us to achieve up to 50x speedups compared to the best alternatives. Our code is available at https://github.com/TheSalon/fast-dpsgd.

        ----

        ## [2022] The effectiveness of feature attribution methods and its correlation with automatic evaluation scores

        **Authors**: *Giang Nguyen, Daeyoung Kim, Anh Nguyen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/de043a5e421240eb846da8effe472ff1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/de043a5e421240eb846da8effe472ff1-Abstract.html)

        **Abstract**:

        Explaining the decisions of an Artificial Intelligence (AI) model is increasingly critical in many real-world, high-stake applications.Hundreds of papers have either proposed new feature attribution methods, discussed or harnessed these tools in their work.However, despite humans being the target end-users, most attribution methods were only evaluated on proxy automatic-evaluation metrics (Zhang et al. 2018; Zhou et al. 2016; Petsiuk et al. 2018). In this paper, we conduct the first user study to measure attribution map effectiveness in assisting humans in ImageNet classification and Stanford Dogs fine-grained classification, and when an image is natural or adversarial (i.e., contains adversarial perturbations). Overall, feature attribution is surprisingly not more effective than showing humans nearest training-set examples. On a harder task of fine-grained dog categorization, presenting attribution maps to humans does not help, but instead hurts the performance of human-AI teams compared to AI alone. Importantly, we found automatic attribution-map evaluation measures to correlate poorly with the actual human-AI team performance. Our findings encourage the community to rigorously test their methods on the downstream human-in-the-loop applications and to rethink the existing evaluation metrics.

        ----

        ## [2023] Coordinated Proximal Policy Optimization

        **Authors**: *Zifan Wu, Chao Yu, Deheng Ye, Junge Zhang, Haiyin Piao, Hankz Hankui Zhuo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/de73998802680548b916f1947ffbad76-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/de73998802680548b916f1947ffbad76-Abstract.html)

        **Abstract**:

        We present Coordinated Proximal Policy Optimization (CoPPO), an algorithm that extends the original Proximal Policy Optimization (PPO) to the multi-agent setting. The key idea lies in the coordinated adaptation of step size during the policy update process among multiple agents. We prove the monotonicity of policy improvement when optimizing a theoretically-grounded joint objective, and derive a simplified optimization objective based on a set of approximations. We then interpret that such an objective in CoPPO can achieve dynamic credit assignment among agents, thereby alleviating the high variance issue during the concurrent update of agent policies. Finally, we demonstrate that CoPPO outperforms several strong baselines and is competitive with the latest multi-agent PPO method (i.e. MAPPO) under typical multi-agent settings, including cooperative matrix games and the StarCraft II micromanagement tasks.

        ----

        ## [2024] Unbiased Classification through Bias-Contrastive and Bias-Balanced Learning

        **Authors**: *Youngkyu Hong, Eunho Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/de8aa43e5d5fa8536cf23e54244476fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/de8aa43e5d5fa8536cf23e54244476fa-Abstract.html)

        **Abstract**:

        Datasets for training machine learning models tend to be biased unless the data is collected with complete care. In such a biased dataset, models are susceptible to making predictions based on the biased features of the data. The biased model fails to generalize to the case where correlations between biases and targets are shifted. To mitigate this, we propose Bias-Contrastive (BiasCon) loss based on the contrastive learning framework, which effectively leverages the knowledge of bias labels. We further suggest Bias-Balanced (BiasBal) regression which trains the classification model toward the data distribution with balanced target-bias correlation. Furthermore, we propose Soft Bias-Contrastive (SoftCon) loss which handles the dataset without bias labels by softening the pair assignment of the BiasCon loss based on the distance in the feature space of the bias-capturing model. Our experiments show that our proposed methods significantly improve previous debiasing methods in various realistic datasets.

        ----

        ## [2025] Learning from Inside: Self-driven Siamese Sampling and Reasoning for Video Question Answering

        **Authors**: *Weijiang Yu, Haoteng Zheng, Mengfei Li, Lei Ji, Lijun Wu, Nong Xiao, Nan Duan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dea184826614d3f4c608731389ed0c74-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dea184826614d3f4c608731389ed0c74-Abstract.html)

        **Abstract**:

        Recent advances in the video question answering (i.e., VideoQA) task have achieved strong success by following the paradigm of fine-tuning each clip-text pair independently on the pretrained transformer-based model via supervised learning. Intuitively, multiple samples (i.e., clips) should be interdependent to capture similar visual and key semantic information in the same video. To consider the interdependent knowledge between contextual clips into the network inference, we propose a Siamese Sampling and Reasoning (SiaSamRea) approach, which consists of a siamese sampling mechanism to generate sparse and similar clips (i.e., siamese clips) from the same video, and a novel reasoning strategy for integrating the interdependent knowledge between contextual clips into the network. The reasoning strategy contains two modules: (1) siamese knowledge generation to learn the inter-relationship among clips; (2) siamese knowledge reasoning to produce the refined soft label by propagating the weights of inter-relationship to the predicted candidates of all clips. Finally, our SiaSamRea can endow the current multimodal reasoning paradigm with the ability of learning from inside via the guidance of soft labels. Extensive experiments demonstrate our SiaSamRea achieves state-of-the-art performance on five VideoQA benchmarks, e.g., a significant +2.1% gain on MSRVTT-QA, +2.9% on MSVD-QA, +1.0% on ActivityNet-QA, +1.8% on How2QA and +4.3% (action) on TGIF-QA.

        ----

        ## [2026] Identification and Estimation of Joint Probabilities of Potential Outcomes in Observational Studies with Covariate Information

        **Authors**: *Ryusei Shingaki, Manabu Kuroki*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dea9ddb25cbf2352cf4dec30222a02a5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dea9ddb25cbf2352cf4dec30222a02a5-Abstract.html)

        **Abstract**:

        The joint probabilities of potential outcomes are fundamental components of causal inference in the sense that (i) if they are identifiable, then the causal risk is also identifiable, but not vise versa (Pearl, 2009; Tian and Pearl, 2000) and (ii) they enable us to evaluate the probabilistic aspects of necessity'',sufficiency'', and ``necessity and sufficiency'', which are important concepts of successful explanation (Watson, et al., 2020). However, because they are not identifiable without any assumptions, various assumptions have been utilized to evaluate the joint probabilities of potential outcomes, e.g., the assumption of monotonicity (Pearl, 2009; Tian and Pearl, 2000), the independence between potential outcomes (Robins and Richardson, 2011),  the condition of gain equality (Li and Pearl, 2019), and the specific functional relationships between cause and effect (Pearl, 2009). Unlike existing identification conditions, in order to evaluate the joint probabilities of potential outcomes without such assumptions, this paper proposes two types of novel identification conditions using covariate information.  In addition, when the joint probabilities of potential outcomes are identifiable through the proposed conditions, the estimation problem of the joint probabilities of potential outcomes reduces to that of singular models and thus they can not be evaluated by standard statistical estimation methods. To solve the problem, this paper proposes a new statistical estimation method based on the augmented Lagrangian method and shows the asymptotic normality of the proposed estimators. Given space constraints, the proofs, the details on the statistical estimation method, some numerical experiments, and the case study are provided in the supplementary material.

        ----

        ## [2027] Online false discovery rate control for anomaly detection in time series

        **Authors**: *Quentin Rebjock, Baris Kurt, Tim Januschowski, Laurent Callot*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/def130d0b67eb38b7a8f4e7121ed432c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/def130d0b67eb38b7a8f4e7121ed432c-Abstract.html)

        **Abstract**:

        This article proposes novel rules for false discovery rate control (FDRC) geared towards online anomaly detection in time series. Online FDRC rules allow to control the properties of a sequence of statistical tests. In the context of anomaly detection, the null hypothesis is that an observation is normal and the alternative is that it is anomalous. FDRC rules allow users to target a lower bound on precision in unsupervised settings. The methods proposed in this article overcome short-comings of previous FDRC rules in the context of anomaly detection, in particular ensuring that power remains high even when the alternative is exceedingly rare (typical in anomaly detection) and the test statistics are serially dependent (typical in time series). We show the soundness of these rules in both theory and experiments.

        ----

        ## [2028] Pragmatic Image Compression for Human-in-the-Loop Decision-Making

        **Authors**: *Siddharth Reddy, Anca D. Dragan, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df0aab058ce179e4f7ab135ed4e641a9-Abstract.html)

        **Abstract**:

        Standard lossy image compression algorithms aim to preserve an image's appearance, while minimizing the number of bits needed to transmit it. However, the amount of information actually needed by the user for downstream tasks -- e.g., deciding which product to click on in a shopping website -- is likely much lower. To achieve this lower bitrate, we would ideally only transmit the visual features that drive user behavior, while discarding details irrelevant to the user's decisions. We approach this problem by training a compression model through human-in-the-loop learning as the user performs tasks with the compressed images. The key insight is to train the model to produce a compressed image that induces the user to take the same action that they would have taken had they seen the original image. To approximate the loss function for this model, we train a discriminator that tries to distinguish whether a user's action was taken in response to the compressed image or the original. We evaluate our method through experiments with human participants on four tasks: reading handwritten digits, verifying photos of faces, browsing an online shopping catalogue, and playing a car racing video game. The results show that our method learns to match the user's actions with and without compression at lower bitrates than baseline methods, and adapts the compression model to the user's behavior: it preserves the digit number and randomizes handwriting style in the digit reading task, preserves hats and eyeglasses while randomizing faces in the photo verification task, preserves the perceived price of an item while randomizing its color and background in the online shopping task, and preserves upcoming bends in the road in the car racing game.

        ----

        ## [2029] Generalized Linear Bandits with Local Differential Privacy

        **Authors**: *Yuxuan Han, Zhipeng Liang, Yang Wang, Jiheng Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df0e09d6f25a15a815563df9827f48fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df0e09d6f25a15a815563df9827f48fa-Abstract.html)

        **Abstract**:

        Contextual bandit algorithms are useful in personalized online decision-making. However, many applications such as personalized medicine and online advertising require the utilization of individual-specific information for effective learning, while user's data should remain private from the server due to privacy concerns. This motivates the introduction of local differential privacy (LDP), a stringent notion in privacy, to contextual bandits. In this paper, we design LDP algorithms for stochastic generalized linear bandits to achieve the same regret bound as in non-privacy settings. Our main idea is to develop a stochastic gradient-based estimator and update mechanism to ensure LDP. We then exploit the flexibility of stochastic gradient descent (SGD), whose theoretical guarantee for bandit problems is rarely explored, in dealing with generalized linear bandits. We also develop an estimator and update mechanism based on Ordinary Least Square (OLS) for linear bandits. Finally, we conduct experiments with both simulation and real-world datasets to demonstrate the consistently superb performance of our algorithms under LDP constraints with reasonably small parameters $(\varepsilon, \delta)$ to ensure strong privacy protection.

        ----

        ## [2030] On the Algorithmic Stability of Adversarial Training

        **Authors**: *Yue Xing, Qifan Song, Guang Cheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df1f1d20ee86704251795841e6a9405a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df1f1d20ee86704251795841e6a9405a-Abstract.html)

        **Abstract**:

        The adversarial training is a popular tool to remedy the vulnerability of deep learning models against adversarial attacks, and there is rich theoretical literature on the training loss of adversarial training algorithms. In contrast, this paper studies the algorithmic stability of a generic adversarial training algorithm, which can further help to establish an upper bound for generalization error. By figuring out the stability upper bound and lower bound, we argue that the non-differentiability issue of adversarial training causes worse algorithmic stability than their natural counterparts. To tackle this problem, we consider a noise injection method. While the non-differentiability problem seriously affects the stability of adversarial training, injecting noise enables the training trajectory to avoid the occurrence of non-differentiability with dominating probability, hence enhancing the stability performance of adversarial training. Our analysis also studies the relation between the algorithm stability and numerical approximation error of adversarial attacks.

        ----

        ## [2031] Width-based Lookaheads with Learnt Base Policies and Heuristics Over the Atari-2600 Benchmark

        **Authors**: *Stefan O'Toole, Nir Lipovetzky, Miquel Ramírez, Adrian R. Pearce*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df42e2244c97a0d80d565ae8176d3351-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df42e2244c97a0d80d565ae8176d3351-Abstract.html)

        **Abstract**:

        We propose new width-based planning and learning algorithms inspired from a careful analysis of the design decisions made by previous width-based planners. The algorithms are applied over the Atari-2600 games and our best performing algorithm, Novelty guided Critical Path Learning (N-CPL), outperforms the previously introduced width-based planning and learning algorithms $\pi$-IW(1), $\pi$-IW(1)+ and $\pi$-HIW(n, 1). Furthermore, we present a taxonomy of the Atari-2600 games according to some of their defining characteristics. This analysis of the games provides further insight into the behaviour and performance of the algorithms introduced. Namely, for games with large branching factors, and games with sparse meaningful rewards, N-CPL outperforms $\pi$-IW, $\pi$-IW(1)+ and $\pi$-HIW(n, 1).

        ----

        ## [2032] Characterizing possible failure modes in physics-informed neural networks

        **Authors**: *Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html)

        **Abstract**:

        Recent work in scientific machine learning has developed so-called physics-informed neural network (PINN) models. The typical approach is to incorporate physical domain knowledge as soft constraints on an empirical loss function and use existing machine learning methodologies to train the model. We demonstrate that, while existing PINN methodologies can learn good models for relatively trivial problems, they can easily fail to learn relevant physical phenomena for even slightly more complex problems. In particular, we analyze several distinct situations of widespread physical interest, including learning differential equations with convection, reaction, and diffusion operators. We provide evidence that the soft regularization in PINNs, which involves PDE-based differential operators, can introduce a number of subtle problems, including making the problem more ill-conditioned. Importantly, we show that these possible failure modes are not due to the lack of expressivity in the NN architecture, but that the PINN's setup makes the loss landscape very hard to optimize. We then describe two promising solutions to address these failure modes. The first approach is to use curriculum regularization, where the PINN's loss term starts from a simple PDE regularization, and becomes progressively more complex as the NN gets trained. The second approach is to pose the problem as a sequence-to-sequence learning task, rather than learning to predict the entire space-time at once. Extensive testing shows that we can achieve up to 1-2 orders of magnitude lower error with these methods as compared to regular PINN training.

        ----

        ## [2033] Artistic Style Transfer with Internal-external Learning and Contrastive Learning

        **Authors**: *Haibo Chen, Lei Zhao, Zhizhong Wang, Huiming Zhang, Zhiwen Zuo, Ailin Li, Wei Xing, Dongming Lu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df5354693177e83e8ba089e94b7b6b55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df5354693177e83e8ba089e94b7b6b55-Abstract.html)

        **Abstract**:

        Although existing artistic style transfer methods have achieved significant improvement with deep neural networks, they still suffer from artifacts such as disharmonious colors and repetitive patterns. Motivated by this, we propose an internal-external style transfer method with two contrastive losses. Specifically, we utilize internal statistics of a single style image to determine the colors and texture patterns of the stylized image, and in the meantime, we leverage the external information of the large-scale style dataset to learn the human-aware style information, which makes the color distributions and texture patterns in the stylized image more reasonable and harmonious. In addition, we argue that existing style transfer methods only consider the content-to-stylization and style-to-stylization relations, neglecting the stylization-to-stylization relations. To address this issue, we introduce two contrastive losses, which pull the multiple stylization embeddings closer to each other when they share the same content or style, but push far away otherwise. We conduct extensive experiments, showing that our proposed method can not only produce visually more harmonious and satisfying artistic images, but also promote the stability and consistency of rendered video clips.

        ----

        ## [2034] Fast Abductive Learning by Similarity-based Consistency Optimization

        **Authors**: *Yu-Xuan Huang, Wang-Zhou Dai, Le-Wen Cai, Stephen H. Muggleton, Yuan Jiang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df7e148cabfd9b608090fa5ee3348bfe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df7e148cabfd9b608090fa5ee3348bfe-Abstract.html)

        **Abstract**:

        To utilize the raw inputs and symbolic knowledge simultaneously, some recent neuro-symbolic learning methods use abduction, i.e., abductive reasoning, to integrate sub-symbolic perception and logical inference. While the perception model, e.g., a neural network, outputs some facts that are inconsistent with the symbolic background knowledge base, abduction can help revise the incorrect perceived facts by minimizing the inconsistency between them and the background knowledge. However, to enable effective abduction, previous approaches need an initialized perception model that discriminates the input raw instances. This limits the application of these methods, as the discrimination ability is usually acquired from a thorough pre-training when the raw inputs are difficult to classify. In this paper, we propose a novel abduction strategy, which leverages the similarity between samples, rather than the output information by the perceptual neural network, to guide the search in abduction. Based on this principle, we further present ABductive Learning with Similarity (ABLSim) and apply it to some difficult neuro-symbolic learning tasks. Experiments show that the efficiency of ABLSim is significantly higher than the state-of-the-art neuro-symbolic methods, allowing it to achieve better performance with less labeled data and weaker domain knowledge.

        ----

        ## [2035] To Beam Or Not To Beam: That is a Question of Cooperation for Language GANs

        **Authors**: *Thomas Scialom, Paul-Alexis Dray, Jacopo Staiano, Sylvain Lamprier, Benjamin Piwowarski*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/df9028fcb6b065e000ffe8a4f03eeb38-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/df9028fcb6b065e000ffe8a4f03eeb38-Abstract.html)

        **Abstract**:

        Due to the discrete nature of words, language GANs require to be optimized from rewards provided by discriminator networks, via reinforcement learning methods. This is a much harder setting than for continuous tasks, which enjoy gradient flows from discriminators to generators, usually leading to dramatic learning instabilities.   However, we claim that this can be solved by making discriminator and generator networks cooperate to produce output sequences during training. These cooperative outputs, inherently built to obtain higher discrimination scores, not only provide denser rewards for training but also form a more compact artificial set for discriminator training, hence improving its accuracy and stability.In this paper, we show that our SelfGAN framework, built on this cooperative principle, outperforms Teacher Forcing and obtains state-of-the-art results on two challenging tasks, Summarization and Question Generation.

        ----

        ## [2036] Shapley Residuals: Quantifying the limits of the Shapley value for explanations

        **Authors**: *Indra Kumar, Carlos Scheidegger, Suresh Venkatasubramanian, Sorelle A. Friedler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dfc6aa246e88ab3e32caeaaecf433550-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dfc6aa246e88ab3e32caeaaecf433550-Abstract.html)

        **Abstract**:

        Popular feature importance techniques compute additive approximations to nonlinear models by first defining a cooperative game describing the value of different subsets of the model's features, then calculating the resulting game's Shapley values to attribute credit additively between the features. However, the specific modeling settings in which the Shapley values are a poor approximation for the true game have not been well-described. In this paper we utilize an interpretation of Shapley values as the result of an orthogonal projection between vector spaces to calculate a residual representing the kernel component of that projection. We provide an algorithm for computing these residuals, characterize different modeling settings based on the value of the residuals, and demonstrate that they capture information about model predictions that Shapley values cannot. Shapley residuals can thus act as a warning to practitioners against overestimating the degree to which Shapley-value-based explanations give them insight into a model.

        ----

        ## [2037] The Elastic Lottery Ticket Hypothesis

        **Authors**: *Xiaohan Chen, Yu Cheng, Shuohang Wang, Zhe Gan, Jingjing Liu, Zhangyang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dfccdb8b1cc7e4dab6d33db0fef12b88-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dfccdb8b1cc7e4dab6d33db0fef12b88-Abstract.html)

        **Abstract**:

        Lottery Ticket Hypothesis (LTH) raises keen attention to identifying sparse trainable subnetworks, or winning tickets, which can be trained in isolation to achieve similar or even better performance compared to the full models. Despite many efforts being made, the most effective method to identify such winning tickets is still Iterative Magnitude-based Pruning (IMP), which is computationally expensive and has to be run thoroughly for every different network. A natural question that comes in is: can we “transform” the winning ticket found in one network to another with a different architecture, yielding a winning ticket for the latter at the beginning, without re-doing the expensive IMP? Answering this question is not only practically relevant for efficient “once-for-all” winning ticket ﬁnding, but also theoretically appealing for uncovering inherently scalable sparse patterns in networks. We conduct extensive experiments on CIFAR-10 and ImageNet, and propose a variety of strategies to tweak the winning tickets found from different networks of the same model family (e.g., ResNets). Based on these results, we articulate the Elastic Lottery Ticket Hypothesis (E-LTH): by mindfully replicating (or dropping) and re-ordering layers for one network, its corresponding winning ticket could be stretched (or squeezed) into a subnetwork for another deeper (or shallower) network from the same family, whose performance is nearly the same competitive as the latter’s winning ticket directly found by IMP. We have also extensively compared E-LTH with pruning-at-initialization and dynamic sparse training methods, as well as discussed the generalizability of E-LTH to different model families, layer types, and across datasets. Code is available at https://github.com/VITA-Group/ElasticLTH.

        ----

        ## [2038] Joint Inference for Neural Network Depth and Dropout Regularization

        **Authors**: *Kishan K. C., Rui Li, MohammadMahdi Gilany*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dfce06801e1a85d6d06f1fdd4475dacd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dfce06801e1a85d6d06f1fdd4475dacd-Abstract.html)

        **Abstract**:

        Dropout regularization methods prune a neural network's pre-determined backbone structure to avoid overfitting. However, a deep model still tends to be poorly calibrated with high confidence on incorrect predictions. We propose a unified Bayesian model selection method to jointly infer the most plausible network depth warranted by data, and perform dropout regularization simultaneously. In particular, to infer network depth we define a beta process over the number of hidden layers which allows it to go to infinity. Layer-wise activation probabilities induced by the beta process modulate neuron activation via binary vectors of a conjugate Bernoulli process. Experiments across domains show that by adapting network depth and dropout regularization to data, our method achieves superior performance comparing to state-of-the-art methods with well-calibrated uncertainty estimates. In continual learning, our method enables neural networks to dynamically evolve their depths to accommodate incrementally available data beyond their initial structures, and alleviate catastrophic forgetting.

        ----

        ## [2039] Tractable Density Estimation on Learned Manifolds with Conformal Embedding Flows

        **Authors**: *Brendan Leigh Ross, Jesse C. Cresswell*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dfd786998e082758be12670d856df755-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dfd786998e082758be12670d856df755-Abstract.html)

        **Abstract**:

        Normalizing flows are generative models that provide tractable density estimation via an invertible transformation from a simple base distribution to a complex target distribution. However, this technique cannot directly model data supported on an unknown low-dimensional manifold, a common occurrence in real-world domains such as image data. Recent attempts to remedy this limitation have introduced geometric complications that defeat a central benefit of normalizing flows: exact density estimation. We recover this benefit with Conformal Embedding Flows, a framework for designing flows that learn manifolds with tractable densities. We argue that composing a standard flow with a trainable conformal embedding is the most natural way to model manifold-supported data. To this end, we present a series of conformal building blocks and apply them in experiments with synthetic and real-world data to demonstrate that flows can model manifold-supported distributions without sacrificing tractable likelihoods.

        ----

        ## [2040] The Limits of Optimal Pricing in the Dark

        **Authors**: *Quinlan Dawkins, Minbiao Han, Haifeng Xu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e0126439e08ddfbdf4faa952dc910590-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e0126439e08ddfbdf4faa952dc910590-Abstract.html)

        **Abstract**:

        A ubiquitous learning problem in today’s digital market is, during repeated interactions between a seller and a buyer, how a seller can gradually learn optimal pricing decisions based on the buyer’s past purchase responses. A fundamental challenge of learning in such a strategic setup is that the buyer will naturally have incentives to manipulate his responses in order to induce more favorable learning outcomes for him. To understand the limits of the seller’s learning when facing such a strategic and possibly manipulative buyer, we study a natural yet powerful buyer manipulation strategy. That is, before the pricing game starts, the buyer simply commits to “imitate” a different value function by pretending to always react optimally according to this imitative value function. We fully characterize the optimal imitative value function that the buyer should imitate as well as the resultant seller revenue and buyer surplus under this optimal buyer manipulation. Our characterizations reveal many useful insights about what happens at equilibrium. For example, a seller with concave production cost will obtain essentially 0 revenue at equilibrium whereas the revenue for a seller with convex production cost is the Bregman divergence of her cost function between no production and certain production. Finally, and importantly, we show that a more powerful class of pricing schemes does not necessarily increase, in fact, may be harmful to, the seller’s revenue. Our results not only lead to an effective prescriptive way for buyers to manipulate learning algorithms but also shed lights on the limits of what a seller can really achieve when pricing in the dark.

        ----

        ## [2041] No RL, No Simulation: Learning to Navigate without Navigating

        **Authors**: *Meera Hahn, Devendra Singh Chaplot, Shubham Tulsiani, Mustafa Mukadam, James M. Rehg, Abhinav Gupta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e02a35b1563d0db53486ec068ebab80f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e02a35b1563d0db53486ec068ebab80f-Abstract.html)

        **Abstract**:

        Most prior methods for learning navigation policies require access to simulation environments, as they need online policy interaction and rely on ground-truth maps for rewards. However, building simulators is expensive (requires manual effort for each and every scene) and creates challenges in transferring learned policies to robotic platforms in the real-world, due to the sim-to-real domain gap. In this paper, we pose a simple question: Do we really need active interaction, ground-truth maps or even reinforcement-learning (RL) in order to solve the image-goal navigation task? We propose a self-supervised approach to learn to navigate from only passive videos of roaming. Our approach, No RL, No Simulator (NRNS), is simple and scalable, yet highly effective. NRNS outperforms RL-based formulations by a significant margin. We present NRNS as a strong baseline for any future image-based navigation tasks that use RL or Simulation.

        ----

        ## [2042] Analogous to Evolutionary Algorithm: Designing a Unified Sequence Model

        **Authors**: *Jiangning Zhang, Chao Xu, Jian Li, Wenzhou Chen, Yabiao Wang, Ying Tai, Shuo Chen, Chengjie Wang, Feiyue Huang, Yong Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e02e27e04fdff967ba7d76fb24b8069d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e02e27e04fdff967ba7d76fb24b8069d-Abstract.html)

        **Abstract**:

        Inspired by biological evolution, we explain the rationality of Vision Transformer by analogy with the proven practical Evolutionary Algorithm (EA) and derive that both of them have consistent mathematical representation. Analogous to the dynamic local population in EA, we improve the existing transformer structure and propose a more efficient EAT model, and design task-related heads to deal with different tasks more flexibly. Moreover, we introduce the spatial-filling curve into the current vision transformer to sequence image data into a uniform sequential format. Thus we can design a unified EAT framework to address multi-modal tasks, separating the network architecture from the data format adaptation. Our approach achieves state-of-the-art results on the ImageNet classification task compared with recent vision transformer works while having smaller parameters and greater throughput. We further conduct multi-modal tasks to demonstrate the superiority of the unified EAT, \eg, Text-Based Image Retrieval, and our approach improves the rank-1 by +3.7 points over the baseline on the CSS dataset.

        ----

        ## [2043] Improving Compositionality of Neural Networks by Decoding Representations to Inputs

        **Authors**: *Mike Wu, Noah D. Goodman, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e0308d73972d8dd5e2dd27853106386e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e0308d73972d8dd5e2dd27853106386e-Abstract.html)

        **Abstract**:

        In traditional software programs, it is easy to trace program logic from variables back to input, apply assertion statements to block erroneous behavior, and compose programs together. Although deep learning programs have demonstrated strong performance on novel applications, they sacrifice many of the functionalities of traditional software programs. With this as motivation, we take a modest first step towards improving deep learning programs by jointly training a generative model to constrain neural network activations to "decode" back to inputs. We call this design a Decodable Neural Network, or DecNN. Doing so enables a form of compositionality in neural networks, where one can recursively compose DecNN with itself to create an ensemble-like model with uncertainty. In our experiments, we demonstrate applications of this uncertainty to out-of-distribution detection, adversarial example detection, and calibration --- while matching standard neural networks in accuracy. We further explore this compositionality by combining DecNN with pretrained models, where we show promising results that neural networks can be regularized from using protected features.

        ----

        ## [2044] The Hardness Analysis of Thompson Sampling for Combinatorial Semi-bandits with Greedy Oracle

        **Authors**: *Fang Kong, Yueran Yang, Wei Chen, Shuai Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e0688d13958a19e087e123148555e4b4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e0688d13958a19e087e123148555e4b4-Abstract.html)

        **Abstract**:

        Thompson sampling (TS) has attracted a lot of interest in the bandit area. It was introduced in the 1930s but has not been theoretically proven until recent years. All of its analysis in the combinatorial multi-armed bandit (CMAB) setting requires an exact oracle to provide optimal solutions with any input. However, such an oracle is usually not feasible since many combinatorial optimization problems are NP-hard and only approximation oracles are available. An example \cite{WangC18} has shown the failure of TS to learn with an approximation oracle. However, this oracle is uncommon and is designed only for a specific problem instance. It is still an open question whether the convergence analysis of TS can be extended beyond the exact oracle in CMAB. In this paper, we study this question under the greedy oracle, which is a common (approximation) oracle with theoretical guarantees to solve many (offline) combinatorial optimization problems. We provide a problem-dependent regret lower bound of order $\Omega(\log T/\Delta^2)$ to quantify the hardness of TS to solve CMAB problems with greedy oracle, where $T$ is the time horizon and $\Delta$ is some reward gap. We also provide an almost matching regret upper bound. These are the first theoretical results for TS to solve CMAB with a common approximation oracle and break the misconception that TS cannot work with approximation oracles.

        ----

        ## [2045] Universal Semi-Supervised Learning

        **Authors**: *Zhuo Huang, Chao Xue, Bo Han, Jian Yang, Chen Gong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e06f967fb0d355592be4e7674fa31d26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e06f967fb0d355592be4e7674fa31d26-Abstract.html)

        **Abstract**:

        Universal Semi-Supervised Learning (UniSSL) aims to solve the open-set problem where both the class distribution (i.e., class set) and feature distribution (i.e., feature domain) are different between labeled dataset and unlabeled dataset. Such a problem seriously hinders the realistic landing of classical SSL. Different from the existing SSL methods targeting at the open-set problem that only study one certain scenario of class distribution mismatch and ignore the feature distribution mismatch, we consider a more general case where a mismatch exists in both class and feature distribution. In this case, we propose a ''Class-shAring data detection and Feature Adaptation'' (CAFA) framework which requires no prior knowledge of the class relationship between the labeled dataset and unlabeled dataset. Particularly, CAFA utilizes a novel scoring strategy to detect the data in the shared class set. Then, it conducts domain adaptation to fully exploit the value of the detected class-sharing data for better semi-supervised consistency training. Exhaustive experiments on several benchmark datasets show the effectiveness of our method in tackling open-set problems.

        ----

        ## [2046] Improving Deep Learning Interpretability by Saliency Guided Training

        **Authors**: *Aya Abdelsalam Ismail, Héctor Corrada Bravo, Soheil Feizi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e0cd3f16f9e883ca91c2a4c24f47b3d9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e0cd3f16f9e883ca91c2a4c24f47b3d9-Abstract.html)

        **Abstract**:

        Saliency methods have been widely used to highlight important input features in model predictions. Most existing methods use backpropagation on a modified gradient function to generate saliency maps. Thus, noisy gradients can result in unfaithful feature attributions. In this paper, we tackle this issue and introduce a {\it saliency guided training} procedure for neural networks to reduce noisy gradients used in predictions while retaining the predictive performance of the model. Our saliency guided training procedure iteratively masks features with small and potentially noisy gradients while maximizing the similarity of model outputs for both masked and unmasked inputs. We apply the saliency guided training procedure to various synthetic and real data sets from computer vision, natural language processing, and time series across diverse neural architectures, including Recurrent Neural Networks, Convolutional Networks, and Transformers. Through qualitative and quantitative evaluations, we show that saliency guided training procedure significantly improves model interpretability across various domains while preserving its predictive performance.

        ----

        ## [2047] SurvITE: Learning Heterogeneous Treatment Effects from Time-to-Event Data

        **Authors**: *Alicia Curth, Changhee Lee, Mihaela van der Schaar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e0eacd983971634327ae1819ea8b6214-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e0eacd983971634327ae1819ea8b6214-Abstract.html)

        **Abstract**:

        We study the problem of inferring heterogeneous treatment effects from time-to-event data. While both the related problems of (i) estimating treatment effects for binary or continuous outcomes and (ii) predicting survival outcomes have been well studied in the recent machine learning literature, their combination -- albeit of high practical relevance -- has received considerably less attention. With the ultimate goal of reliably estimating the effects of treatments on instantaneous risk and survival probabilities, we focus on the problem of learning (discrete-time) treatment-specific conditional hazard functions. We find that unique challenges arise in this context due to a variety of covariate shift issues that go beyond a mere combination of well-studied confounding and censoring biases. We theoretically analyse their effects by adapting recent generalization bounds from domain adaptation and treatment effect estimation to our setting and discuss implications for model design. We use the resulting insights to propose a novel deep learning method for treatment-specific hazard estimation based on balancing representations. We investigate performance across a range of experimental settings and empirically confirm that our method outperforms baselines by addressing covariate shifts from various sources.

        ----

        ## [2048] Optimal Rates for Nonparametric Density Estimation under Communication Constraints

        **Authors**: *Jayadev Acharya, Clément L. Canonne, Aditya Vikram Singh, Himanshu Tyagi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e1021d43911ca2c1845910d84f40aeae-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e1021d43911ca2c1845910d84f40aeae-Abstract.html)

        **Abstract**:

        We consider density estimation for Besov spaces when the estimator is restricted to use only a limited number of bits about each sample. We provide a noninteractive adaptive estimator which exploits the sparsity of wavelet bases, along with a simulate-and-infer technique from parametric estimation under communication constraints. We show that our estimator is nearly rate-optimal by deriving minmax lower bounds that hold even when interactive protocols are allowed. Interestingly, while our wavelet-based estimator is almost rate-optimal for Sobolev spaces as well, it is unclear whether the standard Fourier basis, which arise naturally for those spaces, can be used to achieve the same performance.

        ----

        ## [2049] Rank Overspecified Robust Matrix Recovery: Subgradient Method and Exact Recovery

        **Authors**: *Lijun Ding, Liwei Jiang, Yudong Chen, Qing Qu, Zhihui Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e13748298cfb23c19fdfd134a2221e7b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e13748298cfb23c19fdfd134a2221e7b-Abstract.html)

        **Abstract**:

        We study the robust recovery of a low-rank matrix from sparsely and grossly corrupted Gaussian measurements, with no prior knowledge on the intrinsic rank. We consider the robust matrix factorization approach. We employ a robust $\ell_1$ loss function and deal with the challenge of the unknown rank by using an overspecified factored representation of the matrix variable. We then solve the associated nonconvex nonsmooth problem using a subgradient method with diminishing stepsizes. We show that under a regularity condition on the sensing matrices and corruption, which we call restricted direction preserving property (RDPP), even with rank overspecified, the subgradient method converges to the exact low-rank solution at a sublinear rate. Moreover, our result is more general in the sense that it automatically speeds up to a linear rate once the factor rank matches the unknown rank. On the other hand, we show that the RDPP condition holds under generic settings, such as Gaussian measurements under independent or adversarial sparse corruptions, where the result could be of independent interest. Both the exact recovery and the convergence rate of the proposed subgradient method are numerically verified in the overspecified regime. Moreover, our experiment further shows that our particular design of diminishing stepsize effectively prevents overfitting for robust recovery under overparameterized models, such as robust matrix sensing and learning robust deep image prior. This regularization effect is worth further investigation.

        ----

        ## [2050] Improving Computational Efficiency in Visual Reinforcement Learning via Stored Embeddings

        **Authors**: *Lili Chen, Kimin Lee, Aravind Srinivas, Pieter Abbeel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e140dbab44e01e699491a59c9978b924-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e140dbab44e01e699491a59c9978b924-Abstract.html)

        **Abstract**:

        Recent advances in off-policy deep reinforcement learning (RL) have led to impressive success in complex tasks from visual observations. Experience replay improves sample-efficiency by reusing experiences from the past, and convolutional neural networks (CNNs) process high-dimensional inputs effectively. However, such techniques demand high memory and computational bandwidth. In this paper, we present Stored Embeddings for Efficient Reinforcement Learning (SEER), a simple modification of existing off-policy RL methods, to address these computational and memory requirements. To reduce the computational overhead of gradient updates in CNNs, we freeze the lower layers of CNN encoders early in training due to early convergence of their parameters. Additionally, we reduce memory requirements by storing the low-dimensional latent vectors for experience replay instead of high-dimensional images, enabling an adaptive increase in the replay buffer capacity, a useful technique in constrained-memory settings. In our experiments, we show that SEER does not degrade the performance of RL agents while significantly saving computation and memory across a diverse set of DeepMind Control environments and Atari games.

        ----

        ## [2051] Learning Generalized Gumbel-max Causal Mechanisms

        **Authors**: *Guy Lorberbom, Daniel D. Johnson, Chris J. Maddison, Daniel Tarlow, Tamir Hazan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e143c01e314f7b950daca31188cb5d0f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e143c01e314f7b950daca31188cb5d0f-Abstract.html)

        **Abstract**:

        To perform counterfactual reasoning in Structural Causal Models (SCMs), one needs to know the causal mechanisms, which provide factorizations of conditional distributions into noise sources and deterministic functions mapping realizations of noise to samples. Unfortunately, the causal mechanism is not uniquely identified by data that can be gathered by observing and interacting with the world, so there remains the question of how to choose causal mechanisms. In recent work, Oberst & Sontag (2019) propose Gumbel-max SCMs, which use Gumbel-max reparameterizations as the causal mechanism due to an appealing  counterfactual stability property. However, the justification requires appealing to intuition. In this work, we instead argue for choosing a causal mechanism that is best under a quantitative criteria such as minimizing variance when estimating counterfactual treatment effects. We propose a parameterized family of causal mechanisms that generalize Gumbel-max. We show that they can be trained to minimize counterfactual effect variance and other losses on a distribution of queries of interest, yielding lower variance estimates of counterfactual treatment effect than fixed alternatives, also generalizing to queries not seen at training time.

        ----

        ## [2052] Bandit Learning with Delayed Impact of Actions

        **Authors**: *Wei Tang, Chien-Ju Ho, Yang Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e17184bcb70dcf3942c54e0b537ffc6d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e17184bcb70dcf3942c54e0b537ffc6d-Abstract.html)

        **Abstract**:

        We consider a stochastic multi-armed bandit (MAB) problem with delayed impact of actions. In our setting, actions taken in the pastimpact the arm rewards in the subsequent future. This delayed impact of actions is prevalent in the real world. For example, the capability to pay back a loan for people in a certain social group might depend on historically how frequently that group has been approved loan applications. If banks keep rejecting loan applications to people in a disadvantaged group, it could create a feedback loop and further damage the chance of getting loans for people in that group. In this paper, we formulate this delayed and long-term impact of actions within the context of multi-armed bandits. We generalize the bandit setting to encode the dependency of this ``bias" due to the action history during learning. The goal is to maximize the collected utilities over time while taking into account the dynamics created by the delayed impacts of historical actions. We propose an algorithm that achieves a regret of $\tilde{O}(KT^{2/3})$ and show a matching regret lower bound of $\Omega(KT^{2/3})$, where $K$ is the number of arms and $T$ is the learning horizon. Our results complement the bandit literature by adding techniques to deal with actions with long-term impacts and have implications in designing fair algorithms.

        ----

        ## [2053] A Stochastic Newton Algorithm for Distributed Convex Optimization

        **Authors**: *Brian Bullins, Kumar Kshitij Patel, Ohad Shamir, Nathan Srebro, Blake E. Woodworth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e17a5a399de92e1d01a56c50afb2a68e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e17a5a399de92e1d01a56c50afb2a68e-Abstract.html)

        **Abstract**:

        We propose and analyze a stochastic Newton algorithm for homogeneous distributed stochastic convex optimization, where each machine can calculate stochastic gradients of the same population objective, as well as stochastic Hessian-vector products (products of an independent unbiased estimator of the Hessian of the population objective with arbitrary vectors), with many such stochastic computations performed between rounds of communication.  We show that our method can reduce the number, and frequency, of required communication rounds, compared to existing methods without hurting performance, by proving convergence guarantees for quasi-self-concordant objectives (e.g., logistic regression), alongside empirical evidence.

        ----

        ## [2054] Are Transformers more robust than CNNs?

        **Authors**: *Yutong Bai, Jieru Mei, Alan L. Yuille, Cihang Xie*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html)

        **Abstract**:

        Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks,  recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations. With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at: https://github.com/ytongbai/ViTs-vs-CNNs.

        ----

        ## [2055] Towards Sharper Generalization Bounds for Structured Prediction

        **Authors**: *Shaojie Li, Yong Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e1b90346c92331860b1391257a106bb1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e1b90346c92331860b1391257a106bb1-Abstract.html)

        **Abstract**:

        In this paper, we investigate the generalization performance of structured prediction learning and obtain state-of-the-art generalization bounds. Our analysis is based on factor graph decomposition of structured prediction algorithms, and we present novel margin guarantees from three different perspectives: Lipschitz continuity, smoothness, and space capacity condition. In the Lipschitz continuity scenario, we improve the square-root dependency on the label set cardinality of existing bounds to a logarithmic dependence. In the smoothness scenario, we provide generalization bounds that are not only a logarithmic dependency on the label set cardinality but a faster convergence rate of order $\mathcal{O}(\frac{1}{n})$ on the sample size $n$. In the space capacity scenario, we obtain bounds that do not depend on the label set cardinality and have faster convergence rates than $\mathcal{O}(\frac{1}{\sqrt{n}})$. In each scenario, applications are provided to suggest that these conditions are easy to be satisfied.

        ----

        ## [2056] Automated Discovery of Adaptive Attacks on Adversarial Defenses

        **Authors**: *Chengyuan Yao, Pavol Bielik, Petar Tsankov, Martin T. Vechev*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e1c13a13fc6b87616b787b986f98a111-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e1c13a13fc6b87616b787b986f98a111-Abstract.html)

        **Abstract**:

        Reliable evaluation of adversarial defenses is a challenging task, currently limited to an expert who manually crafts attacks that exploit the defenseâ€™s inner workings, or to approaches based on ensemble of fixed attacks, none of which may be effective for the specific defense at hand. Our key observation is that adaptive attacks are composed from a set of reusable building blocks that can be formalized in a search space and used to automatically discover attacks for unknown defenses. We evaluated our approach on 24 adversarial defenses and show that it outperforms AutoAttack, the current state-of-the-art tool for reliable evaluation of adversarial defenses: our tool discovered significantly stronger attacks by producing 3.0%-50.8% additional adversarial examples for 10 models, while obtaining attacks with slightly stronger or similar strength for the remaining models.

        ----

        ## [2057] PolarStream: Streaming Object Detection and Segmentation with Polar Pillars

        **Authors**: *Qi Chen, Sourabh Vora, Oscar Beijbom*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html)

        **Abstract**:

        Recent works recognized lidars as an inherently streaming data source and showed that the end-to-end latency of lidar perception models can be reduced significantly by operating on wedge-shaped point cloud sectors rather then the full point cloud. However, due to use of cartesian coordinate systems these methods  represent the sectors as rectangular regions, wasting memory and compute. In this work we propose using a polar coordinate system and make two key improvements on this design. First, we increase the spatial context by using multi-scale padding from neighboring sectors: preceding sector from the current scan and/or the following sector from the past scan. Second, we improve the core polar convolutional architecture by introducing feature undistortion and range stratified convolutions. Experimental results on the nuScenes dataset show significant improvements over other streaming based methods. We also achieve comparable results to existing non-streaming methods but with lower latencies.

        ----

        ## [2058] Representation Costs of Linear Neural Networks: Analysis and Design

        **Authors**: *Zhen Dai, Mina Karzand, Nathan Srebro*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e22cb9d6bbb4c290a94e4fff4d68a831-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e22cb9d6bbb4c290a94e4fff4d68a831-Abstract.html)

        **Abstract**:

        For different parameterizations (mappings from parameters to predictors), we study the regularization cost in predictor space induced by $l_2$ regularization on the parameters (weights).  We focus on linear neural networks as parameterizations of linear predictors.  We identify the representation cost of certain sparse linear ConvNets and residual networks.  In order to get a better understanding of how the architecture and parameterization affect the representation cost, we also study the reverse problem, identifying which regularizers on linear predictors (e.g., $l_p$ norms, group norms, the $k$-support-norm, elastic net) can be the representation cost induced by simple $l_2$ regularization, and designing the parameterizations that do so.

        ----

        ## [2059] Teaching via Best-Case Counterexamples in the Learning-with-Equivalence-Queries Paradigm

        **Authors**: *Akash Kumar, Yuxin Chen, Adish Singla*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e22dd5dabde45eda5a1a67772c8e25dd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e22dd5dabde45eda5a1a67772c8e25dd-Abstract.html)

        **Abstract**:

        We study the sample complexity of teaching, termed as "teaching dimension" (TD) in the literature, for the learning-with-equivalence-queries (LwEQ) paradigm. More concretely, we consider a learner who asks equivalence queries (i.e., "is the queried hypothesis the target hypothesis?"), and a teacher responds either "yes" or "no" along with a counterexample to the queried hypothesis. This learning paradigm has been extensively studied when the learner receives worst-case or random counterexamples; in this paper, we consider the optimal teacher who picks best-case counterexamples to teach the target hypothesis within a hypothesis class. For this optimal teacher, we introduce LwEQ-TD, a notion of TD capturing the teaching complexity (i.e., the number of queries made) in this paradigm. We show that a significant reduction in queries can be achieved with best-case counterexamples, in contrast to worst-case or random counterexamples, for different hypothesis classes. Furthermore, we establish new connections of LwEQ-TD to the well-studied notions of TD in the learning-from-samples paradigm.

        ----

        ## [2060] Distilling Meta Knowledge on Heterogeneous Graph for Illicit Drug Trafficker Detection on Social Media

        **Authors**: *Yiyue Qian, Yiming Zhang, Yanfang Ye, Chuxu Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e234e195f3789f05483378c397db1cb5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e234e195f3789f05483378c397db1cb5-Abstract.html)

        **Abstract**:

        Driven by the considerable profits, the crime of drug trafficking (a.k.a. illicit drug trading) has co-evolved with modern technologies, e.g., social media such as Instagram has become a popular platform for marketing and selling illicit drugs. The activities of online drug trafficking are nimble and resilient, which call for novel techniques to effectively detect, disrupt, and dismantle illicit drug trades. In this paper, we propose a holistic framework named MetaHG to automatically detect illicit drug traffickers on social media (i.e., Instagram), by tackling the following two new challenges: (1) different from existing works which merely focus on analyzing post content, MetaHG is capable of jointly modeling multi-modal content and relational structured information on social media for illicit drug trafficker detection; (2) in addition, through the proposed meta-learning technique, MetaHG addresses the issue of requiring sufficient data for model training. More specifically, in our proposed MetaHG, we first build a heterogeneous graph (HG) to comprehensively characterize the complex ecosystem of drug trafficking on social media.  Then, we employ a relation-based graph convolutional neural network to learn node (i.e., user) representations over the built HG, in which we introduce graph structure refinement to compensate the sparse connection among entities in the HG for more robust node representation learning. Afterwards, we propose a meta-learning algorithm for model optimization. A self-supervised module and a knowledge distillation module are further designed to exploit unlabeled data for improving the model. Extensive experiments based on the real-world data collected from Instagram demonstrate that the proposed MetaHG outperforms state-of-the-art methods.

        ----

        ## [2061] Curriculum Disentangled Recommendation with Noisy Multi-feedback

        **Authors**: *Hong Chen, Yudong Chen, Xin Wang, Ruobing Xie, Rui Wang, Feng Xia, Wenwu Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e242660df1b69b74dcc7fde711f924ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e242660df1b69b74dcc7fde711f924ff-Abstract.html)

        **Abstract**:

        Learning disentangled representations for user intentions from multi-feedback (i.e., positive and negative feedback) can enhance the accuracy and explainability of recommendation algorithms. However, learning such disentangled representations from multi-feedback data is challenging because  i) multi-feedback is complex: there exist complex relations among different types of feedback (e.g., click, unclick, and dislike, etc) as well as various user intentions, and ii) multi-feedback is noisy: there exists noisy (useless) information both in features and labels, which may deteriorate the recommendation performance.  Existing works on disentangled representation learning only focus on positive feedback, failing to handle the complex relations and noise hidden in multi-feedback data. To solve this problem, in this work we propose a Curriculum Disentangled Recommendation (CDR) model that is capable of efficiently learning disentangled representations from complex and noisy multi-feedback for better recommendation. Concretely, we design a co-filtering dynamic routing mechanism that simultaneously captures the complex relations among different behavioral feedback and user intentions as well as denoise the representations in the feature level. We then present an adjustable self-evaluating curriculum that is able to evaluate sample difficulties for better model training and conduct denoising in the label level via disregarding useless information. Our extensive experiments on several real-world datasets demonstrate that the proposed CDR model can significantly outperform several state-of-the-art methods in terms of recommendation accuracy.

        ----

        ## [2062] Interpretable agent communication from scratch (with a generic visual processor emerging on the side)

        **Authors**: *Roberto Dessì, Eugene Kharitonov, Marco Baroni*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e250c59336b505ed411d455abaa30b4d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e250c59336b505ed411d455abaa30b4d-Abstract.html)

        **Abstract**:

        As deep networks begin to be deployed as autonomous agents, the issue of how they can communicate with each other becomes important. Here, we train two deep nets from scratch to perform realistic referent identification through unsupervised emergent communication. We show that the largely interpretable emergent protocol allows the nets to successfully communicate even about object types they did not see at training time. The visual representations induced as a by-product of our training regime, moreover, show comparable quality, when re-used as generic visual features, to a recent self-supervised learning model. Our results provide concrete evidence of the viability of (interpretable) emergent deep net communication in a more realistic scenario than previously considered, as well as establishing an intriguing link between this field and self-supervised visual learning.

        ----

        ## [2063] MAU: A Motion-Aware Unit for Video Prediction and Beyond

        **Authors**: *Zheng Chang, Xinfeng Zhang, Shanshe Wang, Siwei Ma, Yan Ye, Xiang Xinguang, Wen Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e25cfa90f04351958216f97e3efdabe9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e25cfa90f04351958216f97e3efdabe9-Abstract.html)

        **Abstract**:

        Accurately predicting inter-frame motion information plays a key role in video prediction tasks. In this paper, we propose a Motion-Aware Unit (MAU) to capture reliable inter-frame motion information by broadening the temporal receptive field of the predictive units. The MAU consists of two modules, the attention module and the fusion module. The attention module aims to learn an attention map based on the correlations between the current spatial state and the historical spatial states. Based on the learned attention map, the historical temporal states are aggregated to an augmented motion information (AMI). In this way, the predictive unit can perceive more temporal dynamics from a wider receptive field. Then, the fusion module is utilized to further aggregate the augmented motion information (AMI) and current appearance information (current spatial state) to the final predicted frame. The computation load of MAU is relatively low and the proposed unit can be easily applied to other predictive models. Moreover, an information recalling scheme is employed into the encoders and decoders to help preserve the visual details of the predictions. We evaluate the MAU on both video prediction and early action recognition tasks. Experimental results show that the MAU outperforms the state-of-the-art methods on both tasks.

        ----

        ## [2064] Successor Feature Landmarks for Long-Horizon Goal-Conditioned Reinforcement Learning

        **Authors**: *Christopher Hoang, Sungryull Sohn, Jongwook Choi, Wilka Carvalho, Honglak Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e27c71957d1e6c223e0d48a165da2ee1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e27c71957d1e6c223e0d48a165da2ee1-Abstract.html)

        **Abstract**:

        Operating in the real-world often requires agents to learn about a complex environment and apply this understanding to achieve a breadth of goals. This problem, known as goal-conditioned reinforcement learning (GCRL), becomes especially challenging for long-horizon goals. Current methods have tackled this problem by augmenting goal-conditioned policies with graph-based planning algorithms. However, they struggle to scale to large, high-dimensional state spaces and assume access to exploration mechanisms for efficiently collecting training data. In this work, we introduce Successor Feature Landmarks (SFL), a framework for exploring large, high-dimensional environments so as to obtain a policy that is proficient for any goal. SFL leverages the ability of successor features (SF) to capture transition dynamics, using it to drive exploration by estimating state-novelty and to enable high-level planning by abstracting the state-space as a non-parametric landmark-based graph. We further exploit SF to directly compute a goal-conditioned policy for inter-landmark traversal, which we use to execute plans to "frontier" landmarks at the edge of the explored state space. We show in our experiments on MiniGrid and ViZDoom that SFL enables efficient exploration of large, high-dimensional state spaces and outperforms state-of-the-art baselines on long-horizon GCRL tasks.

        ----

        ## [2065] Streaming Belief Propagation for Community Detection

        **Authors**: *Yuchen Wu, Jakab Tardos, MohammadHossein Bateni, André Linhares, Filipe Miguel Gonçalves de Almeida, Andrea Montanari, Ashkan Norouzi-Fard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e2a2dcc36a08a345332c751b2f2e476c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e2a2dcc36a08a345332c751b2f2e476c-Abstract.html)

        **Abstract**:

        The community detection problem requires to cluster the nodes of a network into a small number of well-connected ‘communities’. There has been substantial recent progress in characterizing the fundamental statistical limits of community detection under simple stochastic block models.  However, in real-world applications, the network structure is typically dynamic, with nodes that join over time. In this setting, we would like a detection algorithm to perform only a limited number of updates at each node arrival. While standard voting approaches satisfy this constraint, it is unclear whether they exploit the network information optimally. We introduce a simple model for networks growing over time which we refer to as streaming stochastic block model (StSBM). Within this model, we prove that voting algorithms have fundamental limitations.  We also develop a streaming belief-propagation (STREAMBP)  approach, for which we prove optimality in certain regimes. We validate our theoretical findings on synthetic and real data

        ----

        ## [2066] The staircase property: How hierarchical structure can guide deep learning

        **Authors**: *Emmanuel Abbe, Enric Boix-Adserà, Matthew S. Brennan, Guy Bresler, Dheeraj Nagaraj*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e2db7186375992e729165726762cb4c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e2db7186375992e729165726762cb4c1-Abstract.html)

        **Abstract**:

        This paper identifies a structural property of data distributions that enables deep neural networks to learn hierarchically. We define the ``staircase'' property for functions over the Boolean hypercube, which posits that high-order Fourier coefficients are reachable from lower-order Fourier coefficients along increasing chains. We prove that functions satisfying this  property can be learned in polynomial time using layerwise  stochastic coordinate descent on regular neural networks -- a class of network architectures and initializations that have homogeneity properties. Our analysis shows that for such staircase functions and neural networks, the gradient-based algorithm learns high-level features by greedily combining lower-level features along the depth of the network. We further back our theoretical results with experiments showing that staircase functions are learnable by more standard ResNet architectures with stochastic gradient descent. Both the theoretical and experimental results support the fact that the staircase property has a role to play in understanding the capabilities of gradient-based learning on regular networks, in contrast to general polynomial-size networks that can emulate any Statistical Query or PAC algorithm, as recently shown.

        ----

        ## [2067] MagNet: A Neural Network for Directed Graphs

        **Authors**: *Xitong Zhang, Yixuan He, Nathan Brugnone, Michael Perlmutter, Matthew J. Hirn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e32084632d369461572832e6582aac36-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e32084632d369461572832e6582aac36-Abstract.html)

        **Abstract**:

        The prevalence of graph-based data has spurred the rapid development of graph neural networks (GNNs) and related machine learning algorithms. Yet, despite the many datasets naturally modeled as directed graphs, including citation, website, and traffic networks, the vast majority of this research focuses on undirected graphs. In this paper, we propose MagNet, a GNN for directed graphs based on a complex Hermitian matrix known as the magnetic Laplacian. This matrix encodes undirected geometric structure in the magnitude of its entries and directional information in their phase. A charge parameter attunes spectral information to variation among directed cycles. We apply our network to a variety of directed graph node classification and link prediction tasks showing that MagNet performs well on all tasks and that its performance exceeds all other methods on a  majority of such tasks. The underlying principles of MagNet are such that it can be adapted to other GNN architectures.

        ----

        ## [2068] Hardware-adaptive Efficient Latency Prediction for NAS via Meta-Learning

        **Authors**: *Hayeon Lee, Sewoong Lee, Song Chong, Sung Ju Hwang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e3251075554389fe91d17a794861d47b-Abstract.html)

        **Abstract**:

        For deployment, neural architecture search should be hardware-aware, in order to satisfy the device-specific constraints (e.g., memory usage, latency and energy consumption) and enhance the model efficiency. Existing methods on hardware-aware NAS collect a large number of samples (e.g., accuracy and latency) from a target device, either builds a lookup table or a latency estimator. However, such approach is impractical in real-world scenarios as there exist numerous devices with different hardware specifications, and collecting samples from such a large number of devices will require prohibitive computational and monetary cost. To overcome such limitations, we propose Hardware-adaptive Efficient Latency Predictor (HELP), which formulates the device-specific latency estimation problem as a meta-learning problem, such that we can estimate the latency of a model's performance for a given task on an unseen device with a few samples. To this end, we introduce novel hardware embeddings to embed any devices considering them as black-box functions that output latencies, and meta-learn the hardware-adaptive latency predictor in a device-dependent manner, using the hardware embeddings. We validate the proposed HELP for its latency estimation performance on unseen platforms, on which it achieves high estimation performance with as few as 10 measurement samples, outperforming all relevant baselines. We also validate end-to-end NAS frameworks using HELP against ones without it, and show that it largely reduces the total time cost of the base NAS method, in latency-constrained settings.

        ----

        ## [2069] Topological Relational Learning on Graphs

        **Authors**: *Yuzhou Chen, Baris Coskunuzer, Yulia R. Gel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e334fd9dac68f13fa1a57796148cf812-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e334fd9dac68f13fa1a57796148cf812-Abstract.html)

        **Abstract**:

        Graph neural networks (GNNs) have emerged as a powerful tool for graph classification and representation learning. However, GNNs tend to suffer from over-smoothing problems and are vulnerable to graph perturbations. To address these challenges, we propose a novel topological neural framework of topological relational inference (TRI) which allows for integrating higher-order graph information to GNNs and for systematically learning a local graph structure. The key idea is to rewire the original graph by using the persistent homology of the small neighborhoods of the nodes and then to incorporate the extracted topological summaries as the side information into the local algorithm. As a result, the new framework enables us to harness both the conventional information on the graph structure and information on higher order topological properties of the graph. We derive theoretical properties on stability of the new local topological representation of the graph and discuss its implications on the graph algebraic connectivity. The experimental results on node classification tasks demonstrate that the new TRI-GNN outperforms all 14 state-of-the-art baselines on 6 out 7 graphs and exhibit higher robustness to perturbations, yielding up to 10\% better performance under noisy scenarios.

        ----

        ## [2070] Learning Theory Can (Sometimes) Explain Generalisation in Graph Neural Networks

        **Authors**: *Pascal Mattia Esser, Leena C. Vankadara, Debarghya Ghoshdastidar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e34376937c784505d9b4fcd980c2f1ce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e34376937c784505d9b4fcd980c2f1ce-Abstract.html)

        **Abstract**:

        In recent years, several results in the supervised learning setting suggested that classical statistical learning-theoretic measures, such as VC dimension, do not adequately explain the performance of deep learning models which prompted a slew of work in the infinite-width and iteration regimes. However, there is little theoretical explanation for the success of neural networks beyond the supervised setting. In this paper we argue that, under some distributional assumptions, classical learning-theoretic measures can sufficiently explain generalization for graph neural networks in the transductive setting. In particular, we provide a rigorous analysis of the performance of neural networks in the context of transductive inference, specifically by analysing the generalisation properties of graph convolutional networks for the problem of node classification. While VC-dimension does result in trivial generalisation error bounds in this setting as well, we show that transductive Rademacher complexity can explain the generalisation properties of graph convolutional networks for stochastic block models. We further use the generalisation error bounds based on transductive Rademacher complexity to demonstrate the role of graph convolutions and network architectures in achieving smaller generalisation error and provide insights into when the graph structure can help in learning. The findings of this paper could re-new the interest in studying generalisation in neural networks in terms of learning-theoretic measures, albeit in specific problems.

        ----

        ## [2071] Federated Linear Contextual Bandits

        **Authors**: *Ruiquan Huang, Weiqiang Wu, Jing Yang, Cong Shen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e347c51419ffb23ca3fd5050202f9c3d-Abstract.html)

        **Abstract**:

        This paper presents a novel federated linear contextual bandits model, where individual clients face different $K$-armed stochastic bandits coupled through common global parameters. By leveraging the geometric structure of the linear rewards, a collaborative algorithm called Fed-PE is proposed to cope with the heterogeneity across clients without exchanging local feature vectors or raw data. Fed-PE relies on a novel multi-client G-optimal design, and achieves near-optimal regrets for both disjoint and shared parameter cases with logarithmic communication costs. In addition, a new concept called collinearly-dependent policies is introduced, based on which a tight minimax regret lower bound for the disjoint parameter case is derived. Experiments demonstrate the effectiveness of the proposed algorithms on both synthetic and real-world datasets.

        ----

        ## [2072] Least Square Calibration for Peer Reviews

        **Authors**: *Sijun Tan, Jibang Wu, Xiaohui Bei, Haifeng Xu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e354fd90b2d5c777bfec87a352a18976-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e354fd90b2d5c777bfec87a352a18976-Abstract.html)

        **Abstract**:

        Peer review systems such as conference paper review often suffer from the issue of miscalibration. Previous works on peer review calibration usually only use the ordinal information or assume simplistic reviewer scoring functions such as linear functions. In practice, applications like academic conferences often rely on manual methods, such as open discussions, to mitigate miscalibration. It remains an important question to develop algorithms that can   handle different types of miscalibrations based on available prior knowledge. In this paper, we propose a flexible framework, namely \emph{least square calibration} (LSC), for selecting top candidates from peer ratings. Our framework provably performs perfect calibration from noiseless linear scoring functions under mild assumptions, yet also provides competitive calibration results when the scoring function is from broader classes beyond linear functions and with arbitrary noise. On our synthetic dataset, we empirically demonstrate that our algorithm consistently outperforms the baseline which select top papers based on the highest average ratings.

        ----

        ## [2073] Scaling Up Exact Neural Network Compression by ReLU Stability

        **Authors**: *Thiago Serra, Xin Yu, Abhinav Kumar, Srikumar Ramalingam*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e35d7a5768c4b85b4780384d55dc3620-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e35d7a5768c4b85b4780384d55dc3620-Abstract.html)

        **Abstract**:

        We can compress a rectifier network while exactly preserving its underlying functionality with respect to a given input domain if some of its neurons are stable. However, current approaches to determine the stability of neurons with Rectified Linear Unit (ReLU) activations require solving or finding a good approximation to multiple discrete optimization problems. In this work, we introduce an algorithm based on solving a single optimization problem to identify all stable neurons. Our approach is on median 183 times faster than the state-of-art method on CIFAR-10, which allows us to explore exact compression on deeper (5 x 100) and wider (2 x 800) networks within minutes. For classifiers trained under an amount of L1 regularization that does not worsen accuracy, we can remove up to 56% of the connections on the CIFAR-10 dataset. The code is available at the following link, https://github.com/yuxwind/ExactCompression .

        ----

        ## [2074] Passive attention in artificial neural networks predicts human visual selectivity

        **Authors**: *Thomas A. Langlois, H. Charles Zhao, Erin Grant, Ishita Dasgupta, Thomas L. Griffiths, Nori Jacoby*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e360367584297ee8d2d5afa709cd440e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e360367584297ee8d2d5afa709cd440e-Abstract.html)

        **Abstract**:

        Developments in machine learning interpretability techniques over the past decade have provided new tools to observe the image regions that are most informative for classification and localization in artificial neural networks (ANNs). Are the same regions similarly informative to human observers? Using data from 79 new experiments and 7,810 participants, we show that passive attention techniques reveal a significant overlap with human visual selectivity estimates derived from 6 distinct behavioral tasks including visual discrimination, spatial localization, recognizability, free-viewing, cued-object search, and saliency search fixations. We find that input visualizations derived from relatively simple ANN architectures probed using guided backpropagation methods are the best predictors of a shared component in the joint variability of the human measures. We validate these correlational results with causal manipulations using recognition experiments. We show that images masked with ANN attention maps were easier for humans to classify than control masks in a speeded recognition experiment. Similarly, we find that recognition performance in the same ANN models was likewise influenced by masking input images using human visual selectivity maps. This work contributes a new approach to evaluating the biological and psychological validity of leading ANNs as models of human vision: by examining their similarities and differences in terms of their visual selectivity to the information contained in images.

        ----

        ## [2075] GRIN: Generative Relation and Intention Network for Multi-agent Trajectory Prediction

        **Authors**: *Longyuan Li, Jian Yao, Li K. Wenliang, Tong He, Tianjun Xiao, Junchi Yan, David Wipf, Zheng Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e3670ce0c315396e4836d7024abcf3dd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e3670ce0c315396e4836d7024abcf3dd-Abstract.html)

        **Abstract**:

        Learning the distribution of future trajectories conditioned on the past is a crucial problem for understanding multi-agent systems. This is challenging because humans make decisions based on complex social relations and personal intents, resulting in highly complex uncertainties over trajectories. To address this problem, we propose a conditional deep generative model that combines advances in graph neural networks. The prior and recognition model encodes two types of latent codes for each agent: an inter-agent latent code to represent social relations and an intra-agent latent code to represent agent intentions. The decoder is carefully devised to leverage the codes in a disentangled way to predict multi-modal future trajectory distribution. Specifically, a graph attention network built upon inter-agent latent code is used to learn continuous pair-wise relations, and an agent's motion is controlled by its latent intents and its observations of all other agents. Through experiments on both synthetic and real-world datasets, we show that our model outperforms previous work in multiple performance metrics. We also show that our model generates realistic multi-modal trajectories.

        ----

        ## [2076] Instance-Dependent Partial Label Learning

        **Authors**: *Ning Xu, Congyu Qiao, Xin Geng, Min-Ling Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e38e37a99f7de1f45d169efcdb288dd1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e38e37a99f7de1f45d169efcdb288dd1-Abstract.html)

        **Abstract**:

        Partial label learning (PLL) is a typical weakly supervised learning problem, where each training example is associated with a set of candidate labels among which only one is true. Most existing PLL approaches assume that the incorrect labels in each training example are randomly picked as the candidate labels. However, this assumption is not realistic since the candidate labels are always instance-dependent. In this paper, we consider instance-dependent PLL and assume that each example is associated with a latent label distribution constituted by the real number of each label, representing the degree to each label describing the feature. The incorrect label with a high degree is more likely to be annotated as the candidate label. Therefore, the latent label distribution is the essential labeling information in partially labeled examples and worth being leveraged for predictive model training. Motivated by this consideration, we propose a novel PLL method that recovers the label distribution as a label enhancement (LE) process and trains the predictive model iteratively in every epoch. Specifically, we assume the true posterior density of the latent label distribution takes on the variational approximate Dirichlet density parameterized by an inference model. Then the evidence lower bound is deduced for optimizing the inference model and the label distributions generated from the variational posterior are utilized for training the predictive model. Experiments on benchmark and real-world datasets validate the effectiveness of the proposed method. Source code is available at https://github.com/palm-ml/valen.

        ----

        ## [2077] Deep Learning with Label Differential Privacy

        **Authors**: *Badih Ghazi, Noah Golowich, Ravi Kumar, Pasin Manurangsi, Chiyuan Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e3a54649aeec04cf1c13907bc6c5c8aa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e3a54649aeec04cf1c13907bc6c5c8aa-Abstract.html)

        **Abstract**:

        The Randomized Response (RR) algorithm is a classical technique to improve robustness in survey aggregation, and has been widely adopted in applications with differential privacy guarantees. We propose a novel algorithm, Randomized Response with Prior (RRWithPrior), which can provide more accurate results while maintaining the same level of privacy guaranteed by RR. We then apply RRWithPrior to learn neural networks with label differential privacy (LabelDP), and show that when only the label needs to be protected, the model performance can be significantly improved over the previous state-of-the-art private baselines. Moreover, we study different ways to obtain priors, which when used with RRWithPrior can additionally improve the model performance, further reducing the accuracy gap between private and non-private models. We complement the empirical results with theoretical analysis showing that LabelDP is provably easier than protecting both the inputs and labels.

        ----

        ## [2078] Semialgebraic Representation of Monotone Deep Equilibrium Models and Applications to Certification

        **Authors**: *Tong Chen, Jean B. Lasserre, Victor Magron, Edouard Pauwels*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html)

        **Abstract**:

        Deep equilibrium models are based on implicitly defined functional relations and have shown competitive performance compared with the traditional deep networks. Monotone operator equilibrium networks (monDEQ) retain interesting performance with additional theoretical guaranties. Existing certification tools for classical deep networks cannot directly be applied to monDEQs for which much fewer tools exist. We introduce a semialgebraic representation for ReLU based monDEQs which allow to approximate the corresponding input output relation by semidefinite programs (SDP). We present several applications to network certification and obtain SDP models for the following problems : robustness certification, Lipschitz constant estimation, ellipsoidal uncertainty propagation. We use these models to certify robustness of monDEQs with respect to a general $L_p$ norm. Experimental results show that the proposed models outperform existing approaches for monDEQ certification. Furthermore, our investigations suggest that monDEQs are much more robust to $L_2$ perturbations than $L_{\infty}$ perturbations.

        ----

        ## [2079] The Role of Global Labels in Few-Shot Classification and How to Infer Them

        **Authors**: *Ruohan Wang, Massimiliano Pontil, Carlo Ciliberto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e3b6fb0fd4df098162eede3313c54a8d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e3b6fb0fd4df098162eede3313c54a8d-Abstract.html)

        **Abstract**:

        Few-shot learning is a central problem in meta-learning, where learners must quickly adapt to new tasks given limited training data. Recently, feature pre-training has become a ubiquitous component in state-of-the-art meta-learning methods and is shown to provide significant performance improvement. However, there is limited theoretical understanding of the connection between pre-training and meta-learning. Further, pre-training requires global labels shared across tasks, which may be unavailable in practice. In this paper, we show why exploiting pre-training is theoretically advantageous for meta-learning, and in particular the critical role of global labels. This motivates us to propose Meta Label Learning (MeLa), a novel meta-learning framework that automatically infers global labels to obtains robust few-shot models. Empirically, we demonstrate that MeLa is competitive with existing methods and provide extensive ablation experiments to highlight its key properties.

        ----

        ## [2080] NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction

        **Authors**: *Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, Wenping Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e41e164f7485ec4a28741a2d0ea41c74-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e41e164f7485ec4a28741a2d0ea41c74-Abstract.html)

        **Abstract**:

        We present a novel neural surface reconstruction method, called NeuS, for reconstructing objects and scenes with high fidelity from 2D image inputs. Existing neural surface reconstruction approaches, such as DVR [Niemeyer et al., 2020] and IDR [Yariv et al., 2020], require foreground mask as supervision, easily get trapped in local minima, and therefore struggle with the reconstruction of objects with severe self-occlusion or thin structures. Meanwhile, recent neural methods for novel view synthesis, such as NeRF [Mildenhall et al., 2020] and its variants, use volume rendering to produce a neural scene representation with robustness of optimization, even for highly complex objects. However, extracting high-quality surfaces from this learned implicit representation is difficult because there are not sufficient surface constraints in the representation. In NeuS, we propose to represent a surface as the zero-level set of a signed distance function (SDF) and develop a new volume rendering method to train a neural SDF representation. We observe that the conventional volume rendering method causes inherent geometric errors (i.e. bias) for surface reconstruction, and therefore propose a new formulation that is free of bias in the first order of approximation, thus leading to more accurate surface reconstruction even without the mask supervision. Experiments on the DTU dataset and the BlendedMVS dataset show that NeuS outperforms the state-of-the-arts in high-quality surface reconstruction, especially for objects and scenes with complex structures and self-occlusion.

        ----

        ## [2081] Improved Guarantees for Offline Stochastic Matching via new Ordered Contention Resolution Schemes

        **Authors**: *Brian Brubach, Nathaniel Grammel, Will Ma, Aravind Srinivasan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e43739bba7cdb577e9e3e4e42447f5a5-Abstract.html)

        **Abstract**:

        Matching is one of the most fundamental and broadly applicable problems across many domains. In these diverse real-world applications, there is often a degree of uncertainty in the input which has led to the study of stochastic matching models. Here, each edge in the graph has a known, independent probability of existing derived from some prediction. Algorithms must probe edges to determine existence and match them irrevocably if they exist. Further, each vertex may have a patience constraint denoting how many of its neighboring edges can be probed. We present new ordered contention resolution schemes yielding improved approximation guarantees for some of the foundational problems studied in this area. For stochastic matching with patience constraints in general graphs, we provide a $0.382$-approximate algorithm, significantly improving over the previous best $0.31$-approximation of Baveja et al. (2018). When the vertices do not have patience constraints, we describe a $0.432$-approximate random order probing algorithm with several corollaries such as an improved guarantee for the Prophet Secretary problem under Edge Arrivals. Finally, for the special case of bipartite graphs with unit patience constraints on one of the partitions, we show a $0.632$-approximate algorithm that improves on the recent $1/3$-guarantee of Hikima et al. (2021).

        ----

        ## [2082] UFC-BERT: Unifying Multi-Modal Controls for Conditional Image Synthesis

        **Authors**: *Zhu Zhang, Jianxin Ma, Chang Zhou, Rui Men, Zhikang Li, Ming Ding, Jie Tang, Jingren Zhou, Hongxia Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e46bc064f8e92ac2c404b9871b2a4ef2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e46bc064f8e92ac2c404b9871b2a4ef2-Abstract.html)

        **Abstract**:

        Conditional image synthesis aims to create an image according to some multi-modal guidance in the forms of textual descriptions, reference images, and image blocks to preserve, as well as their combinations. In this paper, instead of investigating these control signals separately, we propose a new two-stage architecture, UFC-BERT, to unify any number of multi-modal controls. In UFC-BERT, both the diverse control signals and the synthesized image are uniformly represented as a sequence of discrete tokens to be processed by Transformer. Different from existing two-stage autoregressive approaches such as DALL-E and VQGAN, UFC-BERT adopts non-autoregressive generation (NAR) at the second stage to enhance the holistic consistency of the synthesized image, to support preserving specified image blocks, and to improve the synthesis speed. Further, we design a progressive algorithm that iteratively improves the non-autoregressively generated image, with the help of two estimators developed for evaluating the compliance with the controls and evaluating the fidelity of the synthesized image, respectively. Extensive experiments on a newly collected large-scale clothing dataset M2C-Fashion and a facial dataset Multi-Modal CelebA-HQ verify that UFC-BERT can synthesize high-fidelity images that comply with flexible multi-modal controls.

        ----

        ## [2083] Is Bang-Bang Control All You Need? Solving Continuous Control with Bernoulli Policies

        **Authors**: *Tim Seyde, Igor Gilitschenski, Wilko Schwarting, Bartolomeo Stellato, Martin A. Riedmiller, Markus Wulfmeier, Daniela Rus*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e46be61f0050f9cc3a98d5d2192cb0eb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e46be61f0050f9cc3a98d5d2192cb0eb-Abstract.html)

        **Abstract**:

        Reinforcement learning (RL) for continuous control typically employs distributions whose support covers the entire action space. In this work, we investigate the colloquially known phenomenon that trained agents often prefer actions at the boundaries of that space. We draw theoretical connections to the emergence of bang-bang behavior in optimal control, and provide extensive empirical evaluation across a variety of recent RL algorithms. We replace the normal Gaussian by a Bernoulli distribution that solely considers the extremes along each action dimension - a bang-bang controller. Surprisingly, this achieves state-of-the-art performance on several continuous control benchmarks - in contrast to robotic hardware, where energy and maintenance cost affect controller choices. Since exploration, learning, and the final solution are entangled in RL, we provide additional imitation learning experiments to reduce the impact of exploration on our analysis. Finally, we show that our observations generalize to environments that aim to model real-world challenges and evaluate factors to mitigate the emergence of bang-bang solutions. Our findings emphasise challenges for benchmarking continuous control algorithms, particularly in light of potential real-world applications.

        ----

        ## [2084] Improving Generalization in Meta-RL with Imaginary Tasks from Latent Dynamics Mixture

        **Authors**: *Suyoung Lee, Sae-Young Chung*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e48e13207341b6bffb7fb1622282247b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e48e13207341b6bffb7fb1622282247b-Abstract.html)

        **Abstract**:

        The generalization ability of most meta-reinforcement learning (meta-RL) methods is largely limited to test tasks that are sampled from the same distribution used to sample training tasks. To overcome the limitation, we propose Latent Dynamics Mixture (LDM) that trains a reinforcement learning agent with imaginary tasks generated from mixtures of learned latent dynamics. By training a policy on mixture tasks along with original training tasks, LDM allows the agent to prepare for unseen test tasks during training and prevents the agent from overfitting the training tasks. LDM significantly outperforms standard meta-RL methods in test returns on the gridworld navigation and MuJoCo tasks where we strictly separate the training task distribution and the test task distribution.

        ----

        ## [2085] Localization with Sampling-Argmax

        **Authors**: *Jiefeng Li, Tong Chen, Ruiqi Shi, Yujing Lou, Yong-Lu Li, Cewu Lu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e4a6222cdb5b34375400904f03d8e6a5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e4a6222cdb5b34375400904f03d8e6a5-Abstract.html)

        **Abstract**:

        Soft-argmax operation is commonly adopted in detection-based methods to localize the target position in a differentiable manner. However, training the neural network with soft-argmax makes the shape of the probability map unconstrained. Consequently, the model lacks pixel-wise supervision through the map during training, leading to performance degradation. In this work, we propose sampling-argmax, a differentiable training method that imposes implicit constraints to the shape of the probability map by minimizing the expectation of the localization error. To approximate the expectation, we introduce a continuous formulation of the output distribution and develop a differentiable sampling process. The expectation can be approximated by calculating the average error of all samples drawn from the output distribution. We show that sampling-argmax can seamlessly replace the conventional soft-argmax operation on various localization tasks. Comprehensive experiments demonstrate the effectiveness and flexibility of the proposed method. Code is available at https://github.com/Jeff-sjtu/sampling-argmax

        ----

        ## [2086] Improved Regularization and Robustness for Fine-tuning in Neural Networks

        **Authors**: *Dongyue Li, Hongyang R. Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e4a93f0332b2519177ed55741ea4e5e7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e4a93f0332b2519177ed55741ea4e5e7-Abstract.html)

        **Abstract**:

        A widely used algorithm for transfer learning is fine-tuning, where a pre-trained model is fine-tuned on a target task with a small amount of labeled data. When the capacity of the pre-trained model is much larger than the size of the target data set, fine-tuning is prone to overfitting and "memorizing" the training labels. Hence, an important question is to regularize fine-tuning and ensure its robustness to noise. To address this question, we begin by analyzing the generalization properties of fine-tuning. We present a PAC-Bayes generalization bound that depends on the distance traveled in each layer during fine-tuning and the noise stability of the fine-tuned model. We empirically measure these quantities. Based on the analysis, we propose regularized self-labeling---the interpolation between regularization and self-labeling methods, including (i) layer-wise regularization to constrain the distance traveled in each layer; (ii) self label-correction and label-reweighting to correct mislabeled data points (that the model is confident) and reweight less confident data points. We validate our approach on an extensive collection of image and text data sets using multiple pre-trained model architectures. Our approach improves baseline methods by 1.76% (on average) for seven image classification tasks and 0.75% for a few-shot classification task. When the target data set includes noisy labels, our approach outperforms baseline methods by 3.56% on average in two noisy settings.

        ----

        ## [2087] BARTScore: Evaluating Generated Text as Text Generation

        **Authors**: *Weizhe Yuan, Graham Neubig, Pengfei Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e4d2b6e6fdeca3e60e0f1a62fee3d9dd-Abstract.html)

        **Abstract**:

        A wide variety of NLP applications, such as machine translation, summarization, and dialog, involve text generation. One major challenge for these applications is how to evaluate whether such generated texts are actually fluent, accurate, or effective. In this work, we conceptualize the evaluation of generated text as a text generation problem, modeled using pre-trained sequence-to-sequence models. The general idea is that models trained to convert the generated text to/from a reference output or the source text will achieve higher scores when the generated text is better. We operationalize this idea using BART, an encoder-decoder based pre-trained model, and propose a metric BARTScore with a number of variants that can be flexibly applied in an unsupervised fashion to evaluation of text from different perspectives (e.g. informativeness, fluency, or factuality). BARTScore is conceptually simple and empirically effective. It can outperform existing top-scoring metrics in 16 of 22 test settings, covering evaluation of 16 datasets (e.g., machine translation, text summarization) and 7 different perspectives (e.g., informativeness, factuality). Code to calculate BARTScore is available at https://github.com/neulab/BARTScore, and we have released an interactive leaderboard for meta-evaluation at http://explainaboard.nlpedia.ai/leaderboard/task-meval/ on the ExplainaBoard platform, which allows us to interactively understand the strengths, weaknesses, and complementarity of each metric.

        ----

        ## [2088] An analysis of Ermakov-Zolotukhin quadrature using kernels

        **Authors**: *Ayoub Belhadji*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e531e258fe3098c3bdd707c30a687d73-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e531e258fe3098c3bdd707c30a687d73-Abstract.html)

        **Abstract**:

        We study a quadrature, proposed by Ermakov and Zolotukhin in the sixties, through the lens of kernel methods. The nodes of this quadrature rule follow the distribution of a determinantal point process, while the weights are defined through a linear system, similarly to the optimal kernel quadrature. In this work, we show how these two classes of quadrature are related, and we prove a tractable formula of the expected value of the squared worst-case integration error on the unit ball of an RKHS of the former quadrature. In particular, this formula involves the eigenvalues of the corresponding kernel and leads to improving on the existing theoretical guarantees of the optimal kernel quadrature with determinantal point processes.

        ----

        ## [2089] Towards Understanding Why Lookahead Generalizes Better Than SGD and Beyond

        **Authors**: *Pan Zhou, Hanshu Yan, Xiaotong Yuan, Jiashi Feng, Shuicheng Yan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)

        **Abstract**:

        To train networks, lookahead algorithm~\cite{zhang2019lookahead} updates its fast weights $k$ times via  an  inner-loop   optimizer before updating its slow weights once by using the latest  fast weights. Any optimizer, e.g. SGD,  can serve as the inner-loop optimizer, and the derived lookahead  generally enjoys remarkable test performance improvement over the vanilla optimizer.  But theoretical understandings on the test performance improvement   of lookahead remain absent yet. To solve this issue, we theoretically justify  the advantages of lookahead  in terms of the excess risk error which measures the test performance. Specifically, we prove that lookahead using  SGD as its inner-loop optimizer can better balance the optimization error and generalization error to achieve smaller excess risk error than vanilla SGD  on (strongly) convex problems and nonconvex problems with Polyak-{\L}ojasiewicz condition which has been observed/proved in neural networks.   Moreover,  we show the stagewise optimization strategy~\cite{barshan2015stage} which decays learning rate several times during training can also benefit lookahead in  improving  its optimization and generalization errors on strongly convex problems. Finally, we propose a  stagewise locally-regularized lookahead (SLRLA) algorithm which sums up the vanilla objective and a local regularizer to minimize at each stage and  provably enjoys optimization and generalization improvement  over the conventional (stagewise) lookahead.    Experimental results on   CIFAR10/100 and ImageNet  testify its  advantages. Codes is available at  \url{https://github.com/sail-sg/SLRLA-optimizer}.

        ----

        ## [2090] Online Market Equilibrium with Application to Fair Division

        **Authors**: *Yuan Gao, Alex Peysakhovich, Christian Kroer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e562cd9c0768d5464b64cf61da7fc6bb-Abstract.html)

        **Abstract**:

        Computing market equilibria is a problem of both theoretical and applied interest. Much research to date focuses on the case of static Fisher markets with full information on buyers' utility functions and item supplies. Motivated by real-world markets, we consider an online setting: individuals have linear, additive utility functions; items arrive sequentially and must be allocated and priced irrevocably. We define the notion of an online market equilibrium in such a market as time-indexed allocations and prices which guarantee buyer optimality and market clearance in hindsight. We propose a simple, scalable and interpretable allocation and pricing dynamics termed as PACE. When items are drawn i.i.d. from an unknown distribution (with a possibly continuous support), we show that PACE leads to an online market equilibrium asymptotically. In particular, PACE ensures that buyers' time-averaged utilities converge to the equilibrium utilities w.r.t. a static market with item supplies being the unknown distribution and that buyers' time-averaged expenditures converge to their per-period budget. Hence, many desirable properties of market equilibrium-based fair division such as envy-freeness, Pareto optimality, and the proportional-share guarantee are also attained asymptotically in the online setting. Next, we extend the dynamics to handle quasilinear buyer utilities, which gives the first online algorithm for computing first-price pacing equilibria. Finally, numerical experiments on real and synthetic datasets show that the dynamics converges quickly under various metrics.

        ----

        ## [2091] Dynamic Resolution Network

        **Authors**: *Mingjian Zhu, Kai Han, Enhua Wu, Qiulin Zhang, Ying Nie, Zhenzhong Lan, Yunhe Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e56954b4f6347e897f954495eab16a88-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e56954b4f6347e897f954495eab16a88-Abstract.html)

        **Abstract**:

        Deep convolutional neural networks (CNNs) are often of sophisticated design with numerous learnable parameters for the accuracy reason. To alleviate the expensive costs of deploying them on mobile devices, recent works have made huge efforts for excavating redundancy in pre-defined architectures. Nevertheless, the redundancy on the input resolution of modern CNNs has not been fully investigated, i.e., the resolution of input image is fixed. In this paper, we observe that the smallest resolution for accurately predicting the given image is different using the same neural network. To this end, we propose a novel dynamic-resolution network (DRNet) in which the input resolution is determined dynamically based on each input sample. Wherein, a resolution predictor with negligible computational costs is explored and optimized jointly with the desired network. Specifically, the predictor learns the smallest resolution that can retain and even exceed the original recognition accuracy for each image. During the inference, each input image will be resized to its predicted resolution for minimizing the overall computation burden. We then conduct extensive experiments on several benchmark networks and datasets. The results show that our DRNet can be embedded in any off-the-shelf network architecture to obtain a considerable reduction in computational complexity. For instance, DR-ResNet-50 achieves similar performance with an about 34% computation reduction, while gaining 1.4% accuracy increase with 10% computation reduction compared to the original ResNet-50 on ImageNet. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/DRNet.

        ----

        ## [2092] Gauge Equivariant Transformer

        **Authors**: *Lingshen He, Yiming Dong, Yisen Wang, Dacheng Tao, Zhouchen Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html)

        **Abstract**:

        Attention mechanism has shown great performance and efficiency in a lot of deep learning models, in which relative position encoding plays a crucial role. However, when introducing attention to manifolds, there is no canonical local coordinate system to parameterize neighborhoods. To address this issue, we propose an equivariant transformer to make our model agnostic to the orientation of local coordinate systems (\textit{i.e.}, gauge equivariant), which employs multi-head self-attention to jointly incorporate both position-based and content-based information. To enhance expressive ability, we adopt regular field of cyclic groups as feature fields in intermediate layers, and propose a novel method to parallel transport the feature vectors in these fields. In addition, we project the position vector of each point onto its local coordinate system to disentangle the orientation of the coordinate system in ambient space (\textit{i.e.}, global coordinate system), achieving rotation invariance. To the best of our knowledge, we are the first to introduce gauge equivariance to self-attention, thus name our model Gauge Equivariant Transformer (GET), which can be efficiently implemented on triangle meshes. Extensive experiments show that GET achieves state-of-the-art performance on two common recognition tasks.

        ----

        ## [2093] Unsupervised Object-Based Transition Models For 3D Partially Observable Environments

        **Authors**: *Antonia Creswell, Rishabh Kabra, Christopher P. Burgess, Murray Shanahan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e5841df2166dd424a57127423d276bbe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e5841df2166dd424a57127423d276bbe-Abstract.html)

        **Abstract**:

        We present a slot-wise, object-based transition model that decomposes a scene into objects, aligns them (with respect to a slot-wise object memory) to maintain a consistent order across time, and predicts how those objects evolve over successive frames. The model is trained end-to-end without supervision using transition losses at the level of the object-structured representation rather than pixels. Thanks to the introduction of our novel alignment module, the model deals properly with two issues that are not handled satisfactorily by other transition models, namely object persistence and object identity. We show that the combination of an object-level loss and correct object alignment over time enables the model to outperform a state-of-the-art baseline, and allows it to deal well with object occlusion and re-appearance in partially observable environments.

        ----

        ## [2094] Robust Contrastive Learning Using Negative Samples with Diminished Semantics

        **Authors**: *Songwei Ge, Shlok Mishra, Chun-Liang Li, Haohan Wang, David Jacobs*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e5afb0f2dbc6d39b312d7406054cb4c6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e5afb0f2dbc6d39b312d7406054cb4c6-Abstract.html)

        **Abstract**:

        Unsupervised learning has recently made exceptional progress because of the development of more effective contrastive learning methods. However, CNNs are prone to depend on low-level features that humans deem non-semantic. This dependency has been conjectured to induce a lack of robustness to image perturbations or domain shift. In this paper, we show that by generating carefully designed negative samples, contrastive learning can learn more robust representations with less dependence on such features. Contrastive learning utilizes positive pairs which preserve semantic information while perturbing superficial features in the training images. Similarly, we propose to generate negative samples in a reversed way, where only the superfluous instead of the semantic features are preserved. We develop two methods, texture-based and patch-based augmentations, to generate negative samples.  These samples achieve better generalization, especially under out-of-domain settings. We also analyze our method and the generated texture-based samples, showing that texture features are indispensable in classifying particular ImageNet classes and especially finer classes. We also show that the model bias between texture and shape features favors them differently under different test settings.

        ----

        ## [2095] General Low-rank Matrix Optimization: Geometric Analysis and Sharper Bounds

        **Authors**: *Haixiang Zhang, Yingjie Bi, Javad Lavaei*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e60e81c4cbe5171cd654662d9887aec2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e60e81c4cbe5171cd654662d9887aec2-Abstract.html)

        **Abstract**:

        This paper considers the global geometry of general low-rank minimization problems via the Burer-Monterio factorization approach. For the rank-$1$ case, we prove that there is no spurious second-order critical point for both symmetric and asymmetric problems if the rank-$2$ RIP constant $\delta$ is less than $1/2$. Combining with a counterexample with $\delta=1/2$, we show that the derived bound is the sharpest possible. For the arbitrary rank-$r$ case, the same property is established when the rank-$2r$ RIP constant $\delta$ is at most $1/3$. We design a counterexample to show that the non-existence of spurious second-order critical points may not hold if $\delta$ is at least $1/2$. In addition, for any problem with $\delta$ between $1/3$ and $1/2$, we prove that all second-order critical points have a positive correlation to the ground truth. Finally, the strict saddle property, which can lead to the polynomial-time global convergence of various algorithms, is established for both the symmetric and asymmetric problems when the rank-$2r$ RIP constant $\delta$ is less than $1/3$. The results of this paper significantly extend several existing bounds in the literature.

        ----

        ## [2096] Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation

        **Authors**: *Emmanuel Bengio, Moksh Jain, Maksym Korablyov, Doina Precup, Yoshua Bengio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e614f646836aaed9f89ce58e837e2310-Abstract.html)

        **Abstract**:

        This paper is about the problem of learning a stochastic policy for generating an object (like a molecular graph) from a sequence of actions, such that the probability of generating an object is proportional to a given positive reward for that object. Whereas standard return maximization tends to converge to a single return-maximizing sequence, there are cases where we would like to sample a diverse set of high-return solutions. These arise, for example, in black-box function optimization when few rounds are possible, each with large batches of queries, where the batches should be diverse, e.g., in the design of new molecules. One can also see this as a problem of approximately converting an energy function to a generative distribution. While MCMC methods can achieve that, they are expensive and generally only perform local exploration. Instead, training a generative policy amortizes the cost of search during training and yields to fast generation.  Using insights from Temporal Difference learning, we propose GFlowNet, based on a view of the generative process as a flow network, making it possible to handle the tricky case where different trajectories can yield the same final state, e.g., there are many ways to sequentially add atoms to generate some molecular graph. We cast the set of trajectories as a flow and convert the flow consistency equations into a learning objective, akin to the casting of the Bellman equations into Temporal Difference methods. We prove that any global minimum of the proposed objectives yields a policy which samples from the desired distribution, and demonstrate the improved performance and diversity of GFlowNet on a simple domain where there are many modes to the reward function, and on a molecule synthesis task.

        ----

        ## [2097] Policy Finetuning: Bridging Sample-Efficient Offline and Online Reinforcement Learning

        **Authors**: *Tengyang Xie, Nan Jiang, Huan Wang, Caiming Xiong, Yu Bai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e61eaa38aed621dd776d0e67cfeee366-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e61eaa38aed621dd776d0e67cfeee366-Abstract.html)

        **Abstract**:

        Recent theoretical work studies sample-efficient reinforcement learning (RL) extensively in two settings: learning interactively in the environment (online RL), or learning from an offline dataset (offline RL). However, existing algorithms and theories for learning near-optimal policies in these two settings are rather different and disconnected. Towards bridging this gap, this paper initiates the theoretical study of *policy finetuning*, that is, online RL where the learner has additional access to a "reference policy" $\mu$ close to the optimal policy $\pi_\star$ in a certain sense. We consider the policy finetuning problem in episodic Markov Decision Processes (MDPs) with $S$ states, $A$ actions, and horizon length $H$. We first design a sharp *offline reduction* algorithm---which simply executes $\mu$ and runs offline policy optimization on the collected dataset---that finds an $\varepsilon$ near-optimal policy within $\widetilde{O}(H^3SC^\star/\varepsilon^2)$ episodes, where $C^\star$ is the single-policy concentrability coefficient between $\mu$ and $\pi_\star$. This offline result is the first that matches the sample complexity lower bound in this setting, and resolves a recent open question in offline RL. We then establish an $\Omega(H^3S\min\{C^\star, A\}/\varepsilon^2)$ sample complexity lower bound for *any* policy finetuning algorithm, including those that can adaptively explore the environment. This implies that---perhaps surprisingly---the optimal policy finetuning algorithm is either offline reduction or a purely online RL algorithm that does not use $\mu$. Finally, we design a new hybrid offline/online algorithm for policy finetuning that achieves better sample complexity than both vanilla offline reduction and purely online RL algorithms, in a relaxed setting where $\mu$ only satisfies concentrability partially up to a certain time step. Overall, our results offer a quantitative understanding on the benefit of a good reference policy, and make a step towards bridging offline and online RL.

        ----

        ## [2098] Reducing Information Bottleneck for Weakly Supervised Semantic Segmentation

        **Authors**: *Jungbeom Lee, Jooyoung Choi, Jisoo Mok, Sungroh Yoon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e6384711491713d29bc63fc5eeb5ba4f-Abstract.html)

        **Abstract**:

        Weakly supervised semantic segmentation produces pixel-level localization from class labels; however, a classifier trained on such labels is likely to focus on a small discriminative region of the target object. We interpret this phenomenon using the information bottleneck principle: the final layer of a deep neural network, activated by the sigmoid or softmax activation functions, causes an information bottleneck, and as a result, only a subset of the task-relevant information is passed on to the output. We first support this argument through a simulated toy experiment and then propose a method to reduce the information bottleneck by removing the last activation function. In addition, we introduce a new pooling method that further encourages the transmission of information from non-discriminative regions to the classification. Our experimental evaluations demonstrate that this simple modification significantly improves the quality of localization maps on both the PASCAL VOC 2012 and MS COCO 2014 datasets, exhibiting a new state-of-the-art performance for weakly supervised semantic segmentation.

        ----

        ## [2099] SGD: The Role of Implicit Regularization, Batch-size and Multiple-epochs

        **Authors**: *Ayush Sekhari, Karthik Sridharan, Satyen Kale*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e64c9ec33f19c7de745bd6b6d1a7a86e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e64c9ec33f19c7de745bd6b6d1a7a86e-Abstract.html)

        **Abstract**:

        Multi-epoch, small-batch, Stochastic Gradient Descent (SGD) has been the method of choice for learning with large over-parameterized models. A popular theory for explaining why SGD works well in practice is that the algorithm has an implicit regularization that biases its output towards a good solution. Perhaps the theoretically most well understood learning setting for SGD is that of Stochastic Convex Optimization (SCO), where it is well known that SGD learns at a rate of $O(1/\sqrt{n})$, where $n$ is the number of samples. In this paper, we consider the problem of SCO and explore the role of implicit regularization, batch size and multiple epochs for SGD. Our main contributions are threefold: * We show that for any regularizer, there is an SCO problem for which Regularized Empirical Risk Minimzation fails to learn. This automatically rules out any implicit regularization based explanation for the success of SGD.* We provide a separation between SGD and learning via Gradient Descent on empirical loss (GD) in terms of sample complexity. We show that there is an SCO problem such that GD with any step size and number of iterations can only learn at a suboptimal rate: at least $\widetilde{\Omega}(1/n^{5/12})$.*  We present a multi-epoch variant of SGD commonly used in practice. We prove that this algorithm is at least as good as single pass SGD in the worst case. However, for certain SCO problems, taking multiple passes over the dataset can significantly outperform single pass SGD. We extend our results to the general learning setting by showing a problem which is learnable for any data distribution, and for this problem, SGD is strictly better than RERM for any regularization function. We conclude by discussing the implications of our results for deep learning, and show a separation between SGD and ERM for two layer diagonal neural networks.

        ----

        ## [2100] AC-GC: Lossy Activation Compression with Guaranteed Convergence

        **Authors**: *R. David Evans, Tor M. Aamodt*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e655c7716a4b3ea67f48c6322fc42ed6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e655c7716a4b3ea67f48c6322fc42ed6-Abstract.html)

        **Abstract**:

        Parallel hardware devices (e.g., graphics processor units) have limited high-bandwidth memory capacity.This negatively impacts the training of deep neural networks (DNNs) by increasing runtime and/or decreasing accuracy when reducing model and/or batch size to fit this capacity. Lossy compression is a promising approach to tackling memory capacity constraints, but prior approaches rely on hyperparameter search to achieve a suitable trade-off between convergence and compression, negating runtime benefits. In this paper we build upon recent developments on Stochastic Gradient Descent convergence to prove an upper bound on the expected loss increase when training with compressed activation storage. We then express activation compression error in terms of this bound, allowing the compression rate to adapt to training conditions automatically. The advantage of our approach, called AC-GC, over existing lossy compression frameworks is that, given a preset allowable increase in loss, significant compression without significant increase in error can be achieved with a single training run. When combined with error-bounded methods, AC-GC achieves 15.1x compression with an average accuracy change of 0.1% on text and image datasets. AC-GC functions on any model composed of the layers analyzed and, by avoiding compression rate search, reduces overall training time by 4.6x over SuccessiveHalving.

        ----

        ## [2101] Label Noise SGD Provably Prefers Flat Global Minimizers

        **Authors**: *Alex Damian, Tengyu Ma, Jason D. Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e6af401c28c1790eaef7d55c92ab6ab6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e6af401c28c1790eaef7d55c92ab6ab6-Abstract.html)

        **Abstract**:

        In overparametrized models, the noise in stochastic gradient descent (SGD) implicitly regularizes the optimization trajectory and determines which local minimum SGD converges to. Motivated by empirical studies that demonstrate that training with noisy labels improves generalization, we study the implicit regularization effect of SGD with label noise. We show that SGD with label noise converges to a stationary point of a regularized loss $L(\theta) +\lambda R(\theta)$, where $L(\theta)$ is the training loss, $\lambda$ is an effective regularization parameter depending on the step size, strength of the label noise, and the batch size, and $R(\theta)$ is an explicit regularizer that penalizes sharp minimizers. Our analysis uncovers an additional regularization effect of large learning rates beyond the linear scaling rule that penalizes large eigenvalues of the Hessian more than small ones. We also prove extensions to classification with general loss functions, significantly strengthening the prior work of Blanc et al. to global convergence and large learning rates and of HaoChen et al. to general models.

        ----

        ## [2102] Can we have it all? On the Trade-off between Spatial and Adversarial Robustness of Neural Networks

        **Authors**: *Sandesh Kamath, Amit Deshpande, Subrahmanyam Kambhampati Venkata, Vineeth N. Balasubramanian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e6ff107459d435e38b54ad4c06202c33-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e6ff107459d435e38b54ad4c06202c33-Abstract.html)

        **Abstract**:

        (Non-)robustness of neural networks to small, adversarial pixel-wise perturbations, and as more recently shown, to even random spatial transformations (e.g., translations, rotations) entreats both theoretical and empirical understanding. Spatial robustness to random translations and rotations is commonly attained via equivariant models (e.g., StdCNNs, GCNNs) and training augmentation, whereas adversarial robustness is typically achieved by adversarial training. In this paper, we prove a quantitative trade-off between spatial and adversarial robustness in a simple statistical setting. We complement this empirically by showing that: (a) as the spatial robustness of equivariant models improves by training augmentation with progressively larger transformations, their adversarial robustness worsens progressively, and (b) as the state-of-the-art robust models are adversarially trained with progressively larger pixel-wise perturbations, their spatial robustness drops progressively. Towards achieving Pareto-optimality in this trade-off, we propose a method based on curriculum learning that trains gradually on more difficult perturbations (both spatial and adversarial) toÂ improve spatial and adversarial robustness simultaneously.

        ----

        ## [2103] Universal Off-Policy Evaluation

        **Authors**: *Yash Chandak, Scott Niekum, Bruno C. da Silva, Erik G. Learned-Miller, Emma Brunskill, Philip S. Thomas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e71e5cd119bbc5797164fb0cd7fd94a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e71e5cd119bbc5797164fb0cd7fd94a4-Abstract.html)

        **Abstract**:

        When faced with sequential decision-making problems, it is often useful to be able to predict what would happen if decisions were made using a new policy. Those predictions must often be based on data collected under some previously used decision-making rule.  Many previous methods enable such off-policy (or counterfactual) estimation of the expected value of a performance measure called the return.  In this paper, we take the first steps towards a 'universal off-policy estimator' (UnO)---one that provides off-policy estimates and high-confidence bounds for any parameter of the return distribution. We use UnO for estimating and simultaneously bounding the mean, variance, quantiles/median, inter-quantile range, CVaR, and the entire cumulative distribution of returns. Finally, we also discuss UnO's applicability in various settings, including fully observable, partially observable (i.e., with unobserved confounders), Markovian, non-Markovian, stationary, smoothly non-stationary, and discrete distribution shifts.

        ----

        ## [2104] A Non-commutative Extension of Lee-Seung's Algorithm for Positive Semidefinite Factorizations

        **Authors**: *Yong Sheng Soh, Antonios Varvitsiotis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e727fa59ddefcefb5d39501167623132-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e727fa59ddefcefb5d39501167623132-Abstract.html)

        **Abstract**:

        Given a data matrix $X\in \mathbb{R}_+^{m\times n}$ with non-negative entries, a Positive Semidefinite (PSD) factorization of $X$ is a collection of $r \times r$-dimensional PSD matrices $\{A_i\}$ and $\{B_j\}$ satisfying the condition $X_{ij}= \mathrm{tr}(A_i B_j)$ for all $\ i\in [m],\ j\in [n]$.  PSD factorizations are fundamentally linked to understanding the expressiveness of semidefinite programs as well as the power and limitations of quantum resources in information theory.  The PSD factorization task generalizes the Non-negative Matrix Factorization (NMF) problem in which we seek a collection of $r$-dimensional non-negative vectors $\{a_i\}$ and $\{b_j\}$ satisfying $X_{ij}= a_i^T b_j$,  for all $i\in [m],\ j\in [n]$ -- one can recover the latter problem by choosing matrices in the PSD factorization to be diagonal.  The most widely used algorithm for computing NMFs of a matrix is the Multiplicative Update algorithm developed by Lee and Seung, in which non-negativity of the updates is preserved by scaling with positive diagonal matrices.  In this paper, we describe a non-commutative extension of Lee-Seung's algorithm, which we call the Matrix Multiplicative Update (MMU) algorithm, for computing PSD factorizations.  The MMU algorithm ensures that updates remain PSD by congruence scaling with the matrix geometric mean of appropriate PSD matrices, and it retains the simplicity of implementation that the multiplicative update algorithm for NMF enjoys.  Building on the Majorization-Minimization framework, we show that under our update scheme the squared loss objective is non-increasing and fixed points correspond to critical points.  The analysis relies on a Lieb's Concavity Theorem.  Beyond PSD factorizations, we show that the MMU algorithm can be also used as a primitive to calculate block-diagonal PSD factorizations and tensor PSD factorizations.  We demonstrate the utility of our method with experiments on real and synthetic data.

        ----

        ## [2105] Efficiently Identifying Task Groupings for Multi-Task Learning

        **Authors**: *Chris Fifty, Ehsan Amid, Zhe Zhao, Tianhe Yu, Rohan Anil, Chelsea Finn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e77910ebb93b511588557806310f78f1-Abstract.html)

        **Abstract**:

        Multi-task learning can leverage information learned by one task to benefit the training of other tasks. Despite this capacity, naively training all tasks together in one model often degrades performance, and exhaustively searching through combinations of task groupings can be prohibitively expensive. As a result, efficiently identifying the tasks that would benefit from training together remains a challenging design question without a clear solution. In this paper, we suggest an approach to select which tasks should train together in multi-task learning models. Our method determines task groupings in a single run by training all tasks together and quantifying the effect to which one task's gradient would affect another task's loss. On the large-scale Taskonomy computer vision dataset, we find this method can decrease test loss by 10.0% compared to simply training all tasks together while operating 11.6 times faster than a state-of-the-art task grouping method.

        ----

        ## [2106] Instance-Conditioned GAN

        **Authors**: *Arantxa Casanova, Marlène Careil, Jakob Verbeek, Michal Drozdzal, Adriana Romero-Soriano*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e7ac288b0f2d41445904d071ba37aaff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e7ac288b0f2d41445904d071ba37aaff-Abstract.html)

        **Abstract**:

        Generative Adversarial Networks (GANs) can generate near photo realistic images in narrow domains such as human faces. Yet, modeling complex distributions of datasets such as ImageNet and COCO-Stuff remains challenging in unconditional settings. In this paper, we take inspiration from kernel density estimation techniques and introduce a non-parametric approach to modeling distributions of complex datasets. We partition the data manifold into a mixture of overlapping neighborhoods described by a datapoint and its nearest neighbors, and introduce a model, called instance-conditioned GAN (IC-GAN), which learns the distribution around each datapoint. Experimental results on ImageNet and COCO-Stuff show that IC-GAN significantly improves over unconditional models and unsupervised data partitioning baselines. Moreover, we show that IC-GAN can effortlessly transfer to datasets not seen during training by simply changing the conditioning instances, and still generate realistic images. Finally, we extend IC-GAN to the class-conditional case and show semantically controllable generation and competitive quantitative results on ImageNet; while improving over BigGAN on ImageNet-LT. Code and trained models to reproduce the reported results are available at https://github.com/facebookresearch/ic_gan.

        ----

        ## [2107] DeepSITH: Efficient Learning via Decomposition of What and When Across Time Scales

        **Authors**: *Brandon G. Jacques, Zoran Tiganj, Marc W. Howard, Per B. Sederberg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e7dfca01f394755c11f853602cb2608a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e7dfca01f394755c11f853602cb2608a-Abstract.html)

        **Abstract**:

        Extracting temporal relationships over a range of scales is a hallmark ofhuman perception and cognition---and thus it is a critical feature of machinelearning applied to real-world problems.  Neural networks are either plaguedby the exploding/vanishing gradient problem in recurrent neural networks(RNNs) or must adjust their parameters to learn the relevant time scales(e.g., in LSTMs). This paper introduces DeepSITH, a deep network comprisingbiologically-inspired Scale-Invariant Temporal History (SITH) modules inseries with dense connections between layers. Each SITH module is simply aset of time cells coding what happened when with a geometrically-spaced set oftime lags.  The dense connections between layers change the definition of whatfrom one layer to the next.  The geometric series of time lags implies thatthe network codes time on a logarithmic scale, enabling DeepSITH network tolearn problems requiring memory over a wide range of time scales. We compareDeepSITH to LSTMs and other recent RNNs on several time series prediction anddecoding tasks. DeepSITH achieves results comparable to state-of-the-artperformance on these problems and continues to perform well even as the delaysare increased.

        ----

        ## [2108] A Gaussian Process-Bayesian Bernoulli Mixture Model for Multi-Label Active Learning

        **Authors**: *Weishi Shi, Dayou Yu, Qi Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e7e69cdf28f8ce6b69b4e1853ee21bab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e7e69cdf28f8ce6b69b4e1853ee21bab-Abstract.html)

        **Abstract**:

        Multi-label classification (MLC) allows complex dependencies among labels, making it more suitable to model many real-world problems. However, data annotation for training MLC models becomes much more labor-intensive due to the correlated (hence non-exclusive) labels and a potential large and sparse label space.  We propose to conduct multi-label active learning (ML-AL) through a novel integrated Gaussian Process-Bayesian Bernoulli Mixture model (GP-B$^2$M) to accurately quantify a data sample's overall contribution to a correlated label space and choose the most informative samples for cost-effective annotation. In particular, the B$^2$M encodes label correlations using a Bayesian Bernoulli mixture of label clusters, where each mixture component corresponds to a global pattern of label correlations. To tackle highly sparse labels under AL, the B$^2$M is further integrated with a predictive GP to connect data features as an effective inductive bias and achieve a feature-component-label mapping. The GP predicts coefficients of mixture components that help to recover the final set of labels of a data sample. A novel auxiliary variable based variational inference algorithm is developed to tackle the non-conjugacy introduced along with the mapping process for efficient end-to-end posterior inference.  The model also outputs a predictive  distribution that provides both the label prediction and their correlations in the form of a label covariance matrix. A principled sampling function is designed accordingly to naturally capture both the feature uncertainty (through GP) and label covariance (through B$^2$M) for effective data sampling. Experiments on real-world multi-label datasets demonstrate the state-of-the-art AL performance of the proposed GP-B$^2$M model.

        ----

        ## [2109] Differentially Private Empirical Risk Minimization under the Fairness Lens

        **Authors**: *Cuong Tran, My H. Dinh, Ferdinando Fioretto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e7e8f8e5982b3298c8addedf6811d500-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e7e8f8e5982b3298c8addedf6811d500-Abstract.html)

        **Abstract**:

        Differential Privacy (DP) is an important privacy-enhancing technology for private machine learning systems. It allows to measure and bound the risk associated with an individual participation in a computation. However, it was recently observed that DP learning systems may exacerbate bias and unfairness for different groups of individuals. This paper builds on these important observations and sheds light on the causes of the disparate impacts arising in the problem of differentially private empirical risk minimization. It focuses on the accuracy disparity arising among groups of individuals in two well-studied DP learning methods: output perturbation and differentially private stochastic gradient descent. The paper analyzes which data and model properties are responsible for the disproportionate impacts, why these aspects are affecting different groups disproportionately, and proposes guidelines to mitigate these effects. The proposed approach is evaluated on several datasets and settings.

        ----

        ## [2110] A Unified View of cGANs with and without Classifiers

        **Authors**: *Si-An Chen, Chun-Liang Li, Hsuan-Tien Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8219d4c93f6c55c6b10fe6bfe997c6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8219d4c93f6c55c6b10fe6bfe997c6c-Abstract.html)

        **Abstract**:

        Conditional Generative Adversarial Networks (cGANs) are implicit generative models which allow to sample from class-conditional distributions. Existing cGANs are based on a wide range of different discriminator designs and training objectives. One popular design in earlier works is to include a classifier during training with the assumption that good classifiers can help eliminate samples generated with wrong classes. Nevertheless, including classifiers in cGANs often comes with a side effect of only generating easy-to-classify samples. Recently, some representative cGANs avoid the shortcoming and reach state-of-the-art performance without having classifiers. Somehow it remains unanswered whether the classifiers can be resurrected to design better cGANs. In this work, we demonstrate that classifiers can be properly leveraged to improve cGANs. We start by using the decomposition of the joint probability distribution to connect the goals of cGANs and classification as a unified framework. The framework, along with a classic energy model to parameterize distributions, justifies the use of classifiers for cGANs in a principled manner. It explains several popular cGAN variants, such as ACGAN, ProjGAN, and ContraGAN, as special cases with different levels of approximations, which provides a unified view and brings new insights to understanding cGANs. Experimental results demonstrate that the design inspired by the proposed framework outperforms state-of-the-art cGANs on multiple benchmark datasets, especially on the most challenging ImageNet. The code is available at https://github.com/sian-chen/PyTorch-ECGAN.

        ----

        ## [2111] Online and Offline Reinforcement Learning by Planning with a Learned Model

        **Authors**: *Julian Schrittwieser, Thomas Hubert, Amol Mandhane, Mohammadamin Barekatain, Ioannis Antonoglou, David Silver*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8258e5140317ff36c7f8225a3bf9590-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8258e5140317ff36c7f8225a3bf9590-Abstract.html)

        **Abstract**:

        Learning efficiently from small amounts of data has long been the focus of model-based reinforcement learning, both for the online case when interacting with the environment, and the offline case when learning from a fixed dataset. However, to date no single unified algorithm could demonstrate state-of-the-art results for both settings.In this work, we describe the Reanalyse algorithm, which uses model-based policy and value improvement operators to compute improved training targets for existing data points, allowing for efficient learning at data budgets varying by several orders of magnitude. We further show that Reanalyse can also be used to learn completely without environment interactions, as in the case of Offline Reinforcement Learning (Offline RL). Combining Reanalyse with the MuZero algorithm, we introduce MuZero Unplugged, a single unified algorithm for any data budget, including Offline RL. In contrast to previous work, our algorithm requires no special adaptations for the off-policy or Offline RL settings. MuZero Unplugged sets new state-of-the-art results for Atari in the standard 200 million frame online setting as well as in the RL Unplugged Offline RL benchmark.

        ----

        ## [2112] Stochastic Multi-Armed Bandits with Control Variates

        **Authors**: *Arun Verma, Manjesh Kumar Hanawal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8542a04d734d0cae36d648b3f519e5c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8542a04d734d0cae36d648b3f519e5c-Abstract.html)

        **Abstract**:

        This paper studies a new variant of the stochastic multi-armed bandits problem where auxiliary information about the arm rewards is available in the form of control variates. In many applications like queuing and wireless networks, the arm rewards are functions of some exogenous variables. The mean values of these variables are known a priori from historical data and can be used as control variates.  Leveraging the theory of control variates, we obtain mean estimates with smaller variance and tighter confidence bounds. We develop an upper confidence bound based algorithm named UCB-CV and characterize the regret bounds in terms of the correlation between rewards and control variates when they follow a multivariate normal distribution. We also extend UCB-CV to other distributions using resampling methods like Jackknifing and Splitting. Experiments on synthetic problem instances validate performance guarantees of the proposed algorithms.

        ----

        ## [2113] Near-Optimal No-Regret Learning in General Games

        **Authors**: *Constantinos Daskalakis, Maxwell Fishelson, Noah Golowich*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e85cc63b4f0f312f11e073fc68ccffd5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e85cc63b4f0f312f11e073fc68ccffd5-Abstract.html)

        **Abstract**:

        We show that Optimistic Hedge -- a common variant of multiplicative-weights-updates with recency bias -- attains ${\rm poly}(\log T)$ regret in multi-player general-sum games. In particular, when every player of the game uses Optimistic Hedge to iteratively update her action in response to the history of play so far, then after $T$ rounds of interaction, each player experiences total regret that is ${\rm poly}(\log T)$. Our bound improves, exponentially, the $O(T^{1/2})$ regret attainable  by standard no-regret learners in games, the $O(T^{1/4})$ regret attainable by no-regret learners with recency bias (Syrgkanis et al., NeurIPS 2015), and the $O(T^{1/6})$ bound that was recently shown for Optimistic Hedge in the special case of two-player games (Chen & Peng, NeurIPS 2020). A direct corollary of our bound is that Optimistic Hedge converges to coarse correlated equilibrium in general games at a rate of $\tilde{O}(1/T)$.

        ----

        ## [2114] Improving Self-supervised Learning with Automated Unsupervised Outlier Arbitration

        **Authors**: *Yu Wang, Jingyang Lin, Jingjing Zou, Yingwei Pan, Ting Yao, Tao Mei*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8855b3528cb03d1def9803220bd3cb9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8855b3528cb03d1def9803220bd3cb9-Abstract.html)

        **Abstract**:

        Our work reveals a structured shortcoming of the existing mainstream self-supervised learning methods. Whereas self-supervised learning frameworks usually take the prevailing perfect instance level invariance hypothesis for granted, we carefully investigate the pitfalls behind. Particularly, we argue that the existing augmentation pipeline for generating multiple positive views naturally introduces out-of-distribution (OOD) samples that undermine the learning of the downstream tasks. Generating diverse positive augmentations on the input does not always pay off in benefiting downstream tasks. To overcome this inherent deficiency, we introduce a lightweight latent variable model UOTA, targeting the view sampling issue for self-supervised learning. UOTA adaptively searches for the most important sampling region to produce views, and provides viable choice for outlier-robust self-supervised learning approaches. Our method directly generalizes to many mainstream self-supervised learning approaches, regardless of the loss's nature contrastive or not. We empirically show UOTA's advantage over the state-of-the-art self-supervised paradigms with evident margin, which well justifies the existence of the OOD sample issue embedded in the existing approaches. Especially, we theoretically prove that the merits of the proposal boil down to guaranteed estimator variance and bias reduction. Code is available: https://github.com/ssl-codelab/uota.

        ----

        ## [2115] Improving Anytime Prediction with Parallel Cascaded Networks and a Temporal-Difference Loss

        **Authors**: *Michael L. Iuzzolino, Michael C. Mozer, Samy Bengio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e894d787e2fd6c133af47140aa156f00-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e894d787e2fd6c133af47140aa156f00-Abstract.html)

        **Abstract**:

        Although deep feedforward neural networks share some characteristics with the primate visual system, a key distinction is their dynamics.  Deep nets typically operate in serial stages wherein each layer completes its computation before processing begins in subsequent layers.   In contrast, biological systems have cascaded dynamics: information propagates from neurons at all layers in parallel but transmission occurs gradually over time, leading to speed-accuracy trade offs even in feedforward architectures. We explore the consequences of biologically inspired parallel hardware by constructing cascaded ResNets in which each residual block has propagation delays but all blocks update in parallel in a stateful manner. Because information transmitted through skip connections avoids delays, the functional depth of the architecture increases over time, yielding  anytime predictions that improve with internal-processing time. We introduce a temporal-difference training loss that achieves a strictly superior speed-accuracy profile over standard losses and enables the cascaded architecture to outperform state-of-the-art anytime-prediction methods. The cascaded architecture has intriguing properties, including: it classifies typical instances more rapidly than atypical instances; it is more robust to both persistent and transient noise than is a conventional ResNet; and its time-varying output trace provides a signal that can be exploited to improve information processing and inference.

        ----

        ## [2116] Identifiable Generative models for Missing Not at Random Data Imputation

        **Authors**: *Chao Ma, Cheng Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8a642ed6a9ad20fb159472950db3d65-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8a642ed6a9ad20fb159472950db3d65-Abstract.html)

        **Abstract**:

        Real-world datasets often have missing values associated with complex generative processes, where the cause of the missingness may not be fully observed. This is known as missing not at random (MNAR) data. However, many imputation methods do not take into account the missingness mechanism, resulting in biased imputation values when MNAR data is present. Although there are a few methods that have considered the MNAR scenario, their model's identifiability under MNAR is generally not guaranteed. That is, model parameters can not be uniquely determined even with infinite data samples, hence the imputation results given by such models can still be biased. This issue is especially overlooked by many modern deep generative models. In this work, we fill in this gap by systematically analyzing the identifiability of generative models under MNAR. Furthermore, we propose a practical deep generative model which can provide identifiability guarantees under mild assumptions, for a wide range of MNAR mechanisms. Our method demonstrates a clear advantage for tasks on both synthetic data and multiple real-world scenarios with MNAR data.

        ----

        ## [2117] DNN-based Topology Optimisation: Spatial Invariance and Neural Tangent Kernel

        **Authors**: *Benjamin Dupuis, Arthur Jacot*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8aac01231200e7ef318b9db75c72695-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8aac01231200e7ef318b9db75c72695-Abstract.html)

        **Abstract**:

        We study the Solid Isotropic Material Penalization (SIMP) method with a density field generated by a fully-connected neural network, taking the coordinates as inputs. In the large width limit, we show that the use of DNNs leads to a filtering effect similar to traditional filtering techniques for SIMP, with a filter described by the Neural Tangent Kernel (NTK). This filter is however not invariant under translation, leading to visual artifacts and non-optimal shapes. We propose two embeddings of the input coordinates, which lead  to (approximate) spatial invariance of the NTK and of the filter. We empirically confirm our theoretical observations and study how the filter size is affected by the architecture of the network. Our solution can easily be applied to any other coordinates-based generation method.

        ----

        ## [2118] Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval

        **Authors**: *Omar Khattab, Christopher Potts, Matei A. Zaharia*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e8b1cbd05f6e6a358a81dee52493dd06-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e8b1cbd05f6e6a358a81dee52493dd06-Abstract.html)

        **Abstract**:

        Multi-hop reasoning (i.e., reasoning across two or more documents) is a key ingredient for NLP models that leverage large corpora to exhibit broad knowledge. To retrieve evidence passages, multi-hop models must contend with a fast-growing search space across the hops, represent complex queries that combine multiple information needs, and resolve ambiguity about the best order in which to hop between training passages. We tackle these problems via Baleen, a system that improves the accuracy of multi-hop retrieval while learning robustly from weak training signals in the many-hop setting. To tame the search space, we propose condensed retrieval, a pipeline that summarizes the retrieved passages after each hop into a single compact context. To model complex queries, we introduce a focused late interaction retriever that allows different parts of the same query representation to match disparate relevant passages. Lastly, to infer the hopping dependencies among unordered training passages, we devise latent hop ordering, a weak-supervision strategy in which the trained retriever itself selects the sequence of hops. We evaluate Baleen on retrieval for two-hop question answering and many-hop claim verification, establishing state-of-the-art performance.

        ----

        ## [2119] Local Hyper-Flow Diffusion

        **Authors**: *Kimon Fountoulakis, Pan Li, Shenghao Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e924517087669cf201ea91bd737a4ff4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e924517087669cf201ea91bd737a4ff4-Abstract.html)

        **Abstract**:

        Recently, hypergraphs have attracted a lot of attention due to their ability to capture complex relations among entities. The insurgence of hypergraphs has resulted in data of increasing size and complexity that exhibit interesting small-scale and local structure, e.g., small-scale communities and localized node-ranking around a given set of seed nodes. Popular and principled ways to capture the local structure are the local hypergraph clustering problem and the related seed set expansion problem. In this work, we propose the first local diffusion method that achieves edge-size-independent Cheeger-type guarantee for the problem of local hypergraph clustering while applying to a rich class of higher-order relations that covers a number of previously studied special cases. Our method is based on a primal-dual optimization formulation where the primal problem has a natural network flow interpretation, and the dual problem has a cut-based interpretation using the $\ell_2$-norm penalty on associated cut-costs. We demonstrate the new technique is significantly better than state-of-the-art methods on both synthetic and real-world data.

        ----

        ## [2120] Permuton-induced Chinese Restaurant Process

        **Authors**: *Masahiro Nakano, Yasuhiro Fujiwara, Akisato Kimura, Takeshi Yamada, Naonori Ueda*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e9287a53b94620249766921107fe70a3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e9287a53b94620249766921107fe70a3-Abstract.html)

        **Abstract**:

        This paper proposes the permuton-induced Chinese restaurant process (PCRP), a stochastic process on rectangular partitioning of a matrix. This distribution is suitable for use as a prior distribution in Bayesian nonparametric relational model to find hidden clusters in matrices and network data. Our main contribution is to introduce the notion of permutons into the well-known Chinese restaurant process (CRP) for sequence partitioning: a permuton is a probability measure on $[0,1]\times [0,1]$ and can be regarded as a geometric interpretation of the scaling limit of permutations. Specifically, we extend the model that the table order of CRPs has a random geometric arrangement on $[0,1]\times [0,1]$ drawn from the permuton. By analogy with the relationship between the stick-breaking process (SBP) and CRP for the infinite mixture model of a sequence, this model can be regarded as a multi-dimensional extension of CRP paired with the block-breaking process (BBP), which has been recently proposed as a multi-dimensional extension of SBP. While BBP always has an infinite number of redundant intermediate variables, PCRP can be composed of varying size intermediate variables in a data-driven manner depending on the size and quality of the observation data. Experiments show that PCRP can improve the prediction performance in relational data analysis by reducing the local optima and slow mixing problems compared with the conventional BNP models because the local transitions of PCRP in Markov chain Monte Carlo inference are more flexible than the previous models.

        ----

        ## [2121] Faster Algorithms and Constant Lower Bounds for the Worst-Case Expected Error

        **Authors**: *Jonah Brown-Cohen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e92b755eb0d8800175a02a35c2bf44fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e92b755eb0d8800175a02a35c2bf44fe-Abstract.html)

        **Abstract**:

        The study of statistical estimation without distributional assumptions on data values, but with knowledge of data collection methods was recently introduced by Chen, Valiant and Valiant (NeurIPS 2020). In this framework, the goal is to design estimators that minimize the worst-case expected error. Here the expectation is over a known, randomized data collection process from some population, and the data values corresponding to each element of the population are assumed to be worst-case. Chen, Valiant and Valiant show that, when data values are $\ell_{\infty}$-normalized, there is a polynomial time algorithm to compute an estimator for the mean with worst-case expected error that is within a factor $\frac{\pi}{2}$ of the optimum within the natural class of semilinear estimators. However, this algorithm is based on optimizing a somewhat complex concave objective function over a constrained set of positive semidefinite matrices, and thus does not come with explicit runtime guarantees beyond being polynomial time in the input.In this paper we design provably efficient algorithms for approximating the optimal semilinear estimator based on online convex optimization. In the setting where data values are $\ell_{\infty}$-normalized, our algorithm achieves a $\frac{\pi}{2}$-approximation by iteratively solving a sequence of standard SDPs. When data values are $\ell_2$-normalized, our algorithm iteratively computes the top eigenvector of a sequence of matrices, and does not lose any multiplicative approximation factor. Further, using experiments in settings where sample membership is correlated with data values (e.g. "importance sampling" and "snowball sampling"), we show that our $\ell_2$-normalized algorithm gives a similar advantage over standard estimators as the original $\ell_{\infty}$-normalized algorithm of Chen, Valiant and Valiant, but with much lower computational complexity. We complement these positive results by stating a simple combinatorial condition which, if satisfied by a data collection process, implies that any (not necessarily semilinear) estimator for the mean has constant worst-case expected error.

        ----

        ## [2122] On Learning Domain-Invariant Representations for Transfer Learning with Multiple Sources

        **Authors**: *Trung Phung, Trung Le, Long Vuong, Toan Tran, Anh Tran, Hung Bui, Dinh Q. Phung*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e9507053dd36cb9217816ffb566c8720-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e9507053dd36cb9217816ffb566c8720-Abstract.html)

        **Abstract**:

        Domain adaptation (DA) benefits from the rigorous theoretical works that study its insightful characteristics and various aspects, e.g., learning domain-invariant representations and its trade-off. However, it seems not the case for the multiple source DA and domain generalization (DG) settings which are remarkably more complicated and sophisticated due to the involvement of multiple source domains and potential unavailability of target domain during training. In this paper, we develop novel upper-bounds for the target general loss which appeal us to define two kinds of domain-invariant representations. We further study the pros and cons as well as the trade-offs of enforcing learning each domain-invariant representation. Finally, we conduct experiments to inspect the trade-off of these representations for offering practical hints regarding how to use them in practice and explore other interesting properties of our developed theory.

        ----

        ## [2123] You Never Cluster Alone

        **Authors**: *Yuming Shen, Ziyi Shen, Menghan Wang, Jie Qin, Philip H. S. Torr, Ling Shao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e96ed478dab8595a7dbda4cbcbee168f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e96ed478dab8595a7dbda4cbcbee168f-Abstract.html)

        **Abstract**:

        Recent advances in self-supervised learning with instance-level contrastive objectives facilitate unsupervised clustering. However, a standalone datum is not perceiving the context of the holistic cluster, and may undergo sub-optimal assignment. In this paper, we extend the mainstream contrastive learning paradigm to a cluster-level scheme, where all the data subjected to the same cluster contribute to a unified representation that encodes the context of each data group. Contrastive learning with this representation then rewards the assignment of each datum. To implement this vision, we propose twin-contrast clustering (TCC). We define a set of categorical variables as clustering assignment confidence, which links the instance-level learning track with the cluster-level one. On one hand, with the corresponding assignment variables being the weight, a weighted aggregation along the data points implements the set representation of a cluster. We further propose heuristic cluster augmentation equivalents to enable cluster-level contrastive learning. On the other hand, we derive the evidence lower-bound of the instance-level contrastive objective with the assignments. By reparametrizing the assignment variables, TCC is trained end-to-end, requiring no alternating steps. Extensive experiments show that TCC outperforms the state-of-the-art on benchmarked datasets.

        ----

        ## [2124] Dynamic COVID risk assessment accounting for community virus exposure from a spatial-temporal transmission model

        **Authors**: *Yuan Chen, Wenbo Fei, Qinxia Wang, Donglin Zeng, Yuanjia Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e97a4f04ef1b914f6a1698caa364f693-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e97a4f04ef1b914f6a1698caa364f693-Abstract.html)

        **Abstract**:

        COVID-19 pandemic has caused unprecedented negative impacts on our society, including further exposing inequity and disparity in public health. To study the impact of socioeconomic factors on COVID transmission, we first propose a spatial-temporal model to examine the socioeconomic heterogeneity and spatial correlation of COVID-19 transmission at the community level. Second, to assess the individual risk of severe COVID-19 outcomes after a positive diagnosis, we propose a dynamic, varying-coefficient model that integrates individual-level risk factors from electronic health records (EHRs) with community-level risk factors. The underlying neighborhood prevalence of infections (both symptomatic and pre-symptomatic) predicted from the previous spatial-temporal model is included in the individual risk assessment so as to better capture the background risk of virus exposure for each individual. We design a weighting scheme to mitigate multiple selection biases inherited in EHRs of COVID patients. We analyze COVID transmission data in New York City (NYC, the epicenter of the first surge in the United States) and EHRs from NYC hospitals, where time-varying effects of community risk factors and significant interactions between individual- and community-level risk factors are detected. By examining the socioeconomic disparity of infection risks and interaction among the risk factors, our methods can assist public health decision-making and facilitate better clinical management of COVID patients.

        ----

        ## [2125] Dueling Bandits with Adversarial Sleeping

        **Authors**: *Aadirupa Saha, Pierre Gaillard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e97ee2054defb209c35fe4dc94599061-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e97ee2054defb209c35fe4dc94599061-Abstract.html)

        **Abstract**:

        We introduce the problem of sleeping dueling bandits with stochastic preferences and adversarial availabilities (DB-SPAA). In almost all dueling bandit applications, the decision space often changes over time; eg, retail store management, online shopping, restaurant recommendation, search engine optimization, etc. Surprisingly, this `sleeping aspect' of dueling bandits has never been studied in the literature. Like dueling bandits, the goal is to compete with the best arm by sequentially querying the preference feedback of item pairs. The non-triviality however results due to the non-stationary item spaces that allow any arbitrary subsets items to go unavailable every round. The goal is to find an optimal `no-regret policy that can identify the best available item at each round, as opposed to the standard `fixed best-arm regret objective' of dueling bandits. We first derive an instance-specific lower bound for DB-SPAA $\Omega( \sum_{i =1}^{K-1}\sum_{j=i+1}^K \frac{\log T}{\Delta(i,j)})$, where $K$ is the number of items and $\Delta(i,j)$ is the gap between items $i$ and $j$. This indicates that the sleeping problem with preference feedback is inherently more difficult than that for classical multi-armed bandits (MAB).  We then propose two algorithms, with near optimal regret guarantees. Our results are corroborated empirically.

        ----

        ## [2126] Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game

        **Authors**: *Alexander G. Reisach, Christof Seiler, Sebastian Weichwald*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e987eff4a7c7b7e580d659feb6f60c1a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e987eff4a7c7b7e580d659feb6f60c1a-Abstract.html)

        **Abstract**:

        Simulated DAG models may exhibit properties that, perhaps inadvertently, render their structure identifiable and unexpectedly affect structure learning algorithms. Here, we show that marginal variance tends to increase along the causal order for generically sampled additive noise models. We introduce varsortability as a measure of the agreement between the order of increasing marginal variance and the causal order. For commonly sampled graphs and model parameters, we show that the remarkable performance of some continuous structure learning algorithms can be explained by high varsortability and matched by a simple baseline method. Yet, this performance may not transfer to real-world data where varsortability may be moderate or dependent on the choice of measurement scales. On standardized data, the same algorithms fail to identify the ground-truth DAG or its Markov equivalence class. While standardization removes the pattern in marginal variance, we show that data generating processes that incur high varsortability also leave a distinct covariance pattern that may be exploited even after standardization. Our findings challenge the significance of generic benchmarks with independently drawn parameters. The code is available at https://github.com/Scriddie/Varsortability.

        ----

        ## [2127] Automated Dynamic Mechanism Design

        **Authors**: *Hanrui Zhang, Vincent Conitzer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html)

        **Abstract**:

        We study Bayesian automated mechanism design in unstructured dynamic environments, where a principal repeatedly interacts with an agent, and takes actions based on the strategic agent's report of the current state of the world.  Both the principal and the agent can have arbitrary and potentially different valuations for the actions taken, possibly also depending on the actual state of the world.  Moreover, at any time, the state of the world may evolve arbitrarily depending on the action taken by the principal.  The goal is to compute an optimal mechanism which maximizes the principal's utility in the face of the self-interested strategic agent.We give an efficient algorithm for computing optimal mechanisms, with or without payments, under different individual-rationality constraints, when the time horizon is constant.  Our algorithm is based on a sophisticated linear program formulation, which can be customized in various ways to accommodate richer constraints.  For environments with large time horizons, we show that the principal's optimal utility is hard to approximate within a certain constant factor, complementing our algorithmic result.  These results paint a relatively complete picture for automated dynamic mechanism design in unstructured environments.  We further consider a special case of the problem where the agent is myopic, and give a refined efficient algorithm whose time complexity scales linearly in the time horizon.    In the full version of the paper, we show that memoryless mechanisms, which are without loss of generality optimal in Markov decision processes without strategic behavior, do not provide a good solution for our problem, in terms of both optimality and computational tractability.  Moreover, we present experimental results where our algorithms are applied to synthetic dynamic environments with different characteristics, which not only serve as a proof of concept for our algorithms, but also exhibit intriguing phenomena in dynamic mechanism design.

        ----

        ## [2128] A generative nonparametric Bayesian model for whole genomes

        **Authors**: *Alan Nawzad Amin, Eli N. Weinstein, Debora S. Marks*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e9dcb63ca828d0e00cd05b445099ed2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e9dcb63ca828d0e00cd05b445099ed2e-Abstract.html)

        **Abstract**:

        Generative probabilistic modeling of biological sequences has widespread existing and potential use across biology and biomedicine, particularly given advances in high-throughput sequencing, synthesis and editing. However, we still lack methods with nucleotide resolution that are tractable at the scale of whole genomes and that can achieve high predictive accuracy in theory and practice. In this article we propose a new generative sequence model, the Bayesian embedded autoregressive (BEAR) model, which uses a parametric autoregressive model to specify a conjugate prior over a nonparametric Bayesian Markov model. We explore, theoretically and empirically, applications of BEAR models to a variety of statistical problems including density estimation, robust parameter estimation, goodness-of-fit tests, and two-sample tests. We prove rigorous asymptotic consistency results including nonparametric posterior concentration rates. We scale inference in BEAR models to datasets containing tens of billions of nucleotides. On genomic, transcriptomic, and metagenomic sequence data we show that BEAR models provide large increases in predictive performance as compared to parametric autoregressive models, among other results. BEAR models offer a flexible and scalable framework, with theoretical guarantees, for building and critiquing generative models at the whole genome scale.

        ----

        ## [2129] Robust Predictable Control

        **Authors**: *Ben Eysenbach, Ruslan Salakhutdinov, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/e9f85782949743dcc42079e629332b5f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/e9f85782949743dcc42079e629332b5f-Abstract.html)

        **Abstract**:

        Many of the challenges facing today's reinforcement learning (RL) algorithms, such as robustness, generalization, transfer, and computational efficiency are closely related to compression.  Prior work has convincingly argued why minimizing information is useful in the supervised learning setting, but standard RL algorithms lack an explicit mechanism for compression. The RL setting is unique because (1) its sequential nature allows an agent to use past information to avoid looking at future observations and (2) the agent can optimize its behavior to prefer states where decision making requires few bits. We take advantage of these properties to propose a method (RPC) for learning simple policies. This method brings together ideas from information bottlenecks, model-based RL, and bits-back coding into a simple and theoretically-justified algorithm. Our method jointly optimizes a latent-space model and policy to be self-consistent, such that the policy avoids states where the model is inaccurate. We demonstrate that our method achieves much tighter compression than prior methods, achieving up to 5$\times$ higher reward than a standard information bottleneck when constrained to use just 0.3 bits per observation. We also demonstrate that our method learns policies that are more robust and generalize better to new tasks.

        ----

        ## [2130] Unsupervised Speech Recognition

        **Authors**: *Alexei Baevski, Wei-Ning Hsu, Alexis Conneau, Michael Auli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ea159dc9788ffac311592613b7f71fbb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ea159dc9788ffac311592613b7f71fbb-Abstract.html)

        **Abstract**:

        Despite rapid progress in the recent past, current speech recognition systems still require labeled training data which limits this technology to a small fraction of the languages spoken around the globe. This paper describes wav2vec-U, short for wav2vec Unsupervised, a method to train speech recognition models without any labeled data. We leverage self-supervised speech representations to segment unlabeled audio and learn a mapping from these representations to phonemes via adversarial training. The right representations are key to the success of our method. Compared to the best previous unsupervised work, wav2vec-U reduces the phone error rate on the TIMIT benchmark from 26.1 to 11.3. On the larger English Librispeech benchmark, wav2vec-U achieves a word error rate of 5.9 on test-other, rivaling some of the best published systems trained on 960 hours of labeled data from only two years ago. We also experiment on nine other languages, including low-resource languages such as Kyrgyz, Swahili and Tatar.

        ----

        ## [2131] Robustness between the worst and average case

        **Authors**: *Leslie Rice, Anna Bair, Huan Zhang, J. Zico Kolter*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ea4c796cccfc3899b5f9ae2874237c20-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ea4c796cccfc3899b5f9ae2874237c20-Abstract.html)

        **Abstract**:

        Several recent works in machine learning have focused on evaluating the test-time robustness of a classifier: how well the classifier performs not just on the target domain it was trained upon, but upon perturbed examples.  In these settings, the focus has largely been on two extremes of robustness: the robustness to perturbations drawn at random from within some distribution (i.e., robustness to random perturbations), and the robustness to the worst case perturbation in some set (i.e., adversarial robustness).  In this paper, we argue that a sliding scale between these two extremes provides a valuable additional metric by which to gauge robustness. Specifically, we illustrate that each of these two extremes is naturally characterized by a (functional) q-norm over perturbation space, with q=1 corresponding to robustness to random perturbations and q=\infty corresponding to adversarial perturbations.  We then present the main technical contribution of our paper: a method for efficiently estimating the value of these norms by interpreting them as the partition function of a particular distribution, then using path sampling with MCMC methods to estimate this partition function (either traditional Metropolis-Hastings for non-differentiable perturbations, or Hamiltonian Monte Carlo for differentiable perturbations).  We show that our approach provides substantially better estimates than simple random sampling of the actual “intermediate-q” robustness of both standard, data-augmented, and adversarially-trained classifiers, illustrating a clear tradeoff between classifiers that optimize different metrics. Code for reproducing experiments can be found at https://github.com/locuslab/intermediate_robustness.

        ----

        ## [2132] Online Learning and Control of Complex Dynamical Systems from Sensory Input

        **Authors**: *Oumayma Bounou, Jean Ponce, Justin Carpentier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ea6979872125d5acbac6068f186a0359-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ea6979872125d5acbac6068f186a0359-Abstract.html)

        **Abstract**:

        Identifying an effective model of a dynamical system from sensory data and using it for future state prediction and control is challenging. Recent data-driven algorithms based on Koopman theory are a promising approach to this problem, but they typically never update the model once it has been identified from a relatively small set of observation, thus making long-term prediction and control difficult for realistic systems, in robotics or fluid mechanics for example. This paper introduces a novel method for learning an embedding of the state space with linear dynamics from sensory data. Unlike previous approaches, the dynamics model can be updated online and thus easily applied to systems with non-linear dynamics in the original configuration space.  The proposed approach is evaluated empirically on several classical dynamical systems and sensory modalities, with good performance on long-term prediction and control.

        ----

        ## [2133] Self-Supervised Bug Detection and Repair

        **Authors**: *Miltiadis Allamanis, Henry Jackson-Flux, Marc Brockschmidt*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ea96efc03b9a050d895110db8c4af057-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ea96efc03b9a050d895110db8c4af057-Abstract.html)

        **Abstract**:

        Machine learning-based program analyses have recently shown the promise of integrating formal and probabilistic reasoning towards aiding software development. However, in the absence of large annotated corpora, training these analyses is challenging. Towards addressing this, we present BugLab, an approach for self-supervised learning of bug detection and repair.  BugLab co-trains two models: (1) a detector model that learns to detect and repair bugs in code, (2) a selector model that learns to create buggy code for the detector to use as training data. A Python implementation of BugLab improves by 30% upon baseline methods on a test dataset of 2374 real-life bugs and finds 19 previously unknown bugs in open-source software.

        ----

        ## [2134] Faster Neural Network Training with Approximate Tensor Operations

        **Authors**: *Menachem Adelman, Kfir Y. Levy, Ido Hakimi, Mark Silberstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eaa1da31f7991743d18dadcf5fd1336f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eaa1da31f7991743d18dadcf5fd1336f-Abstract.html)

        **Abstract**:

        We propose a novel technique for faster deep neural network training which systematically applies sample-based approximation to the constituent tensor operations, i.e., matrix multiplications and convolutions. We introduce new sampling techniques, study their theoretical properties, and prove that they provide the same convergence guarantees when applied to SGD training. We apply approximate tensor operations to single and multi-node training of MLP and CNN networks on MNIST, CIFAR-10 and ImageNet datasets. We demonstrate up to 66% reduction in the amount of computations and communication, and up to 1.37x faster training time while maintaining negligible or no impact on the final test accuracy.

        ----

        ## [2135] Learning Interpretable Decision Rule Sets: A Submodular Optimization Approach

        **Authors**: *Fan Yang, Kai He, Linxiao Yang, Hongxia Du, Jingbang Yang, Bo Yang, Liang Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eaa32c96f620053cf442ad32258076b9-Abstract.html)

        **Abstract**:

        Rule sets are highly interpretable logical models in which the predicates for decision are expressed in disjunctive normal form (DNF, OR-of-ANDs), or, equivalently, the overall model comprises an unordered collection of if-then decision rules. In this paper, we consider a submodular optimization based approach for learning rule sets. The learning problem is framed as a subset selection task in which a subset of all possible rules needs to be selected to form an accurate and interpretable rule set. We employ an objective function that exhibits submodularity and thus is amenable to submodular optimization techniques. To overcome the difficulty arose from dealing with the exponential-sized ground set of rules, the subproblem of searching a rule is casted as another subset selection task that asks for a subset of features. We show it is possible to write the induced objective function for the subproblem as a difference of two submodular (DS) functions to make it approximately solvable by DS optimization algorithms. Overall, the proposed approach is simple, scalable, and likely to be benefited from further research on submodular optimization. Experiments on real datasets demonstrate the effectiveness of our method.

        ----

        ## [2136] Spatial-Temporal Super-Resolution of Satellite Imagery via Conditional Pixel Synthesis

        **Authors**: *Yutong He, Dingjie Wang, Nicholas Lai, William Zhang, Chenlin Meng, Marshall Burke, David B. Lobell, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ead81fe8cfe9fda9e4c2093e17e4d024-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ead81fe8cfe9fda9e4c2093e17e4d024-Abstract.html)

        **Abstract**:

        High-resolution satellite imagery has proven useful for a broad range of tasks, including measurement of global human population, local economic livelihoods, and biodiversity, among many others. Unfortunately, high-resolution imagery is both infrequently collected and expensive to purchase, making it hard to efficiently and effectively scale these downstream tasks over both time and space. We propose a new conditional pixel synthesis model that uses abundant, low-cost, low-resolution imagery to generate accurate high-resolution imagery at locations and times in which it is unavailable. We show that our model attains photo-realistic sample quality and outperforms competing baselines on a key downstream task – object counting – particularly in geographic locations where conditions on the ground are changing rapidly.

        ----

        ## [2137] On Memorization in Probabilistic Deep Generative Models

        **Authors**: *Gerrit J. J. van den Burg, Christopher K. I. Williams*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eae15aabaa768ae4a5993a8a4f4fa6e4-Abstract.html)

        **Abstract**:

        Recent advances in deep generative models have led to impressive results in a variety of application domains. Motivated by the possibility that deep learning models might memorize part of the input data, there have been increased efforts to understand how memorization arises. In this work, we extend a recently proposed measure of memorization for supervised learning (Feldman, 2019) to the unsupervised density estimation problem and adapt it to be more computationally efficient. Next, we present a study that demonstrates how memorization can occur in probabilistic deep generative models such as variational autoencoders. This reveals that the form of memorization to which these models are susceptible differs fundamentally from mode collapse and overfitting. Furthermore, we show that the proposed memorization score measures a phenomenon that is not captured by commonly-used nearest neighbor tests. Finally, we discuss several strategies that can be used to limit memorization in practice. Our work thus provides a framework for understanding problematic memorization in probabilistic generative models.

        ----

        ## [2138] You Are the Best Reviewer of Your Own Papers: An Owner-Assisted Scoring Mechanism

        **Authors**: *Weijie J. Su*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eaf76caaba574ebf8e825f321c14ba29-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eaf76caaba574ebf8e825f321c14ba29-Abstract.html)

        **Abstract**:

        I consider the setting where reviewers offer very noisy scores for a number of items for the selection of high-quality ones (e.g., peer review of large conference proceedings) whereas the owner of these items knows the true underlying scores but prefers not to provide this information. To address this withholding of information, in this paper, I introduce the Isotonic Mechanism, a simple and efficient approach to improving on the imprecise raw scores by leveraging certain information that the owner is incentivized to provide. This mechanism takes as input the ranking of the items from best to worst provided by the owner, in addition to the raw scores provided by the reviewers. It reports adjusted scores for the items by solving a convex optimization problem. Under certain conditions, I show that the owner's optimal strategy is to honestly report the true ranking of the items to her best knowledge in order to maximize the expected utility. Moreover, I prove that the adjusted scores provided by this owner-assisted mechanism are indeed significantly moreaccurate than the raw scores provided by the reviewers. This paper concludes with several extensions of the Isotonic Mechanism and some refinements of the mechanism for practical considerations.

        ----

        ## [2139] Garment4D: Garment Reconstruction from Point Cloud Sequences

        **Authors**: *Fangzhou Hong, Liang Pan, Zhongang Cai, Ziwei Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eb160de1de89d9058fcb0b968dbbbd68-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eb160de1de89d9058fcb0b968dbbbd68-Abstract.html)

        **Abstract**:

        Learning to reconstruct 3D garments is important for dressing 3D human bodies of different shapes in different poses. Previous works typically rely on 2D images as input, which however suffer from the scale and pose ambiguities. To circumvent the problems caused by 2D images, we propose a principled framework, Garment4D, that uses 3D point cloud sequences of dressed humans for garment reconstruction. Garment4D has three dedicated steps: sequential garments registration, canonical garment estimation, and posed garment reconstruction. The main challenges are two-fold: 1) effective 3D feature learning for fine details, and 2) capture of garment dynamics caused by the interaction between garments and the human body, especially for loose garments like skirts. To unravel these problems, we introduce a novel Proposal-Guided Hierarchical Feature Network and Iterative Graph Convolution Network, which integrate both high-level semantic features and low-level geometric features for fine details reconstruction. Furthermore, we propose a Temporal Transformer for smooth garment motions capture. Unlike non-parametric methods, the reconstructed garment meshes by our method are separable from the human body and have strong interpretability, which is desirable for downstream tasks. As the first attempt at this task, high-quality reconstruction results are qualitatively and quantitatively illustrated through extensive experiments. Codes are available at https://github.com/hongfz16/Garment4D.

        ----

        ## [2140] Fast Policy Extragradient Methods for Competitive Games with Entropy Regularization

        **Authors**: *Shicong Cen, Yuting Wei, Yuejie Chi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eb1848290d5a7de9c9ccabc67fefa211-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eb1848290d5a7de9c9ccabc67fefa211-Abstract.html)

        **Abstract**:

        This paper investigates the problem of computing the equilibrium of competitive games, which is often modeled as a constrained saddle-point optimization problem with probability simplex constraints. Despite recent efforts in understanding the last-iterate convergence of extragradient methods in the unconstrained setting, the theoretical underpinnings of these methods in the constrained settings, especially those using multiplicative updates, remain highly inadequate, even when the objective function is bilinear. Motivated by the algorithmic role of entropy regularization in single-agent reinforcement learning and game theory, we develop provably efficient extragradient methods to find the quantal response equilibrium (QRE)---which are solutions to zero-sum two-player matrix games with entropy regularization---at a linear rate. The proposed algorithms can be implemented in a decentralized manner, where each player executes symmetric and multiplicative updates iteratively using its own payoff without observing the opponent's actions directly.  In addition, by controlling the knob of entropy regularization, the proposed algorithms can locate an approximate Nash equilibrium of the unregularized matrix game at a sublinear rate without assuming the Nash equilibrium to be unique. Our methods also lead to efficient policy extragradient algorithms for solving entropy-regularized zero-sum Markov games at a linear rate. All of our convergence rates are nearly dimension-free, which are independent of the size of the state and action spaces up to logarithm factors, highlighting the positive role of entropy regularization for accelerating convergence.

        ----

        ## [2141] Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training data

        **Authors**: *Qi Zhu, Natalia Ponomareva, Jiawei Han, Bryan Perozzi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html)

        **Abstract**:

        There has been a recent surge of interest in designing Graph Neural Networks (GNNs) for semi-supervised learning tasks. Unfortunately this work has assumed that the nodes labeled for use in training were selected uniformly at random (i.e. are an IID sample). However in many real world scenarios gathering labels for graph nodes is both expensive and inherently biased -- so this assumption can not be met. GNNs can suffer poor generalization when this occurs, by overfitting to superfluous regularities present in the training data. In this work we present a method, Shift-Robust GNN (SR-GNN), designed to account for distributional differences between biased training data and the graph's true inference distribution. SR-GNN adapts GNN models for the presence of distributional shifts between the nodes which have had labels provided for training and the rest of the dataset. We illustrate the effectiveness of SR-GNN in a variety of experiments with biased training datasets on common GNN benchmark datasets for semi-supervised learning, where we see that SR-GNN outperforms other GNN baselines by accuracy, eliminating at least (~40%) of the negative effects introduced by biased training data. On the largest dataset we consider, ogb-arxiv, we observe an 2% absolute improvement over the baseline and reduce 30% of the negative effects.

        ----

        ## [2142] RIM: Reliable Influence-based Active Learning on Graphs

        **Authors**: *Wentao Zhang, Yexin Wang, Zhenbang You, Meng Cao, Ping Huang, Jiulong Shan, Zhi Yang, Bin Cui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eb86d510361fc23b59f18c1bc9802cc6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eb86d510361fc23b59f18c1bc9802cc6-Abstract.html)

        **Abstract**:

        Message passing is the core of most graph models such as Graph Convolutional Network (GCN) and Label Propagation (LP), which usually require a large number of clean labeled data to smooth out the neighborhood over the graph. However, the labeling process can be tedious, costly, and error-prone in practice. In this paper, we propose to unify active learning (AL) and message passing towards minimizing labeling costs, e.g., making use of few and unreliable labels that can be obtained cheaply. We make two contributions towards that end. First, we open up a perspective by drawing a connection between AL enforcing message passing and social influence maximization, ensuring that the selected samples effectively improve the model performance. Second, we propose an extension to the influence model that incorporates an explicit quality factor to model label noise. In this way, we derive a fundamentally new AL selection criterion for GCN and LP--reliable influence maximization (RIM)--by considering quantity and quality of influence simultaneously. Empirical studies on public datasets show that RIM significantly outperforms current AL methods in terms of accuracy and efficiency.

        ----

        ## [2143] Dynamical Wasserstein Barycenters for Time-series Modeling

        **Authors**: *Kevin C. Cheng, Shuchin Aeron, Michael C. Hughes, Eric L. Miller*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ebb71045453f38676c40deb9864f811d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ebb71045453f38676c40deb9864f811d-Abstract.html)

        **Abstract**:

        Many time series can be modeled as a sequence of segments representing high-level discrete states, such as running and walking in a human activity application. Flexible models should describe the system state and observations in stationary ``pure-state'' periods as well as transition periods between adjacent segments, such as a gradual slowdown between running and walking. However, most prior work assumes instantaneous transitions between pure discrete states. We propose a dynamical Wasserstein barycentric (DWB) model that estimates the system state over time as well as the data-generating distributions of pure states in an unsupervised manner. Our model assumes each pure state generates data from a multivariate normal distribution, and characterizes transitions between states via displacement-interpolation specified by the Wasserstein barycenter. The system state is represented by a barycentric weight vector which evolves over time via a random walk on the simplex. Parameter learning leverages the natural Riemannian geometry of Gaussian distributions under the Wasserstein distance, which leads to improved convergence speeds. Experiments on several human activity datasets show that our proposed DWB model accurately learns the generating distribution of pure states while improving state estimation for transition periods compared to the commonly used linear interpolation mixture models.

        ----

        ## [2144] RelaySum for Decentralized Deep Learning on Heterogeneous Data

        **Authors**: *Thijs Vogels, Lie He, Anastasia Koloskova, Sai Praneeth Karimireddy, Tao Lin, Sebastian U. Stich, Martin Jaggi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ebbdfea212e3a756a1fded7b35578525-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ebbdfea212e3a756a1fded7b35578525-Abstract.html)

        **Abstract**:

        In decentralized machine learning, workers compute model updates on their local data.Because the workers only communicate with few neighbors without central coordination, these updates propagate progressively over the network.This paradigm enables distributed training on networks without all-to-all connectivity, helping to protect data privacy as well as to reduce the communication cost of distributed training in data centers.A key challenge, primarily in decentralized deep learning, remains the handling of differences between the workers' local data distributions.To tackle this challenge, we introduce the RelaySum mechanism for information propagation in decentralized learning.RelaySum uses spanning trees to distribute information exactly uniformly across all workers with finite delays depending on the distance between nodes.In contrast, the typical gossip averaging mechanism only distributes data uniformly asymptotically while using the same communication volume per step as RelaySum.We prove that RelaySGD, based on this mechanism, is independent of data heterogeneity and scales to many workers, enabling highly accurate decentralized deep learning on heterogeneous data.

        ----

        ## [2145] Transformers Generalize DeepSets and Can be Extended to Graphs & Hypergraphs

        **Authors**: *Jinwoo Kim, Saeyoon Oh, Seunghoon Hong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec0f40c389aeef789ce03eb814facc6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec0f40c389aeef789ce03eb814facc6c-Abstract.html)

        **Abstract**:

        We present a generalization of Transformers to any-order permutation invariant data (sets, graphs, and hypergraphs). We begin by observing that Transformers generalize DeepSets, or first-order (set-input) permutation invariant MLPs. Then, based on recently characterized higher-order invariant MLPs, we extend the concept of self-attention to higher orders and propose higher-order Transformers for order-$k$ data ($k=2$ for graphs and $k>2$ for hypergraphs). Unfortunately, higher-order Transformers turn out to have prohibitive complexity $\mathcal{O}(n^{2k})$ to the number of input nodes $n$. To address this problem, we present sparse higher-order Transformers that have quadratic complexity to the number of input hyperedges, and further adopt the kernel attention approach to reduce the complexity to linear. In particular, we show that the sparse second-order Transformers with kernel attention are theoretically more expressive than message passing operations while having an asymptotically identical complexity. Our models achieve significant performance improvement over invariant MLPs and message-passing graph neural networks in large-scale graph regression and set-to-(hyper)graph prediction tasks. Our implementation is available at https://github.com/jw9730/hot.

        ----

        ## [2146] No Regrets for Learning the Prior in Bandits

        **Authors**: *Soumya Basu, Branislav Kveton, Manzil Zaheer, Csaba Szepesvári*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec1f764517b7ffb52057af6df18142b7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec1f764517b7ffb52057af6df18142b7-Abstract.html)

        **Abstract**:

        We propose AdaTS, a Thompson sampling algorithm that adapts sequentially to bandit tasks that it interacts with. The key idea in AdaTS is to adapt to an unknown task prior distribution by maintaining a distribution over its parameters. When solving a bandit task, that uncertainty is marginalized out and properly accounted for. AdaTS is a fully-Bayesian algorithm that can be implemented efficiently in several classes of bandit problems. We derive upper bounds on its Bayes regret that quantify the loss due to not knowing the task prior, and show that it is small. Our theory is supported by experiments, where AdaTS outperforms prior algorithms and works well even in challenging real-world problems.

        ----

        ## [2147] Encoding Robustness to Image Style via Adversarial Feature Perturbations

        **Authors**: *Manli Shu, Zuxuan Wu, Micah Goldblum, Tom Goldstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec20019911a77ad39d023710be68aaa1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec20019911a77ad39d023710be68aaa1-Abstract.html)

        **Abstract**:

        Adversarial training is the industry standard for producing models that are robust to small adversarial perturbations.  However, machine learning practitioners need models that are robust to other kinds of changes that occur naturally, such as changes in the style or illumination of input images. Such changes in input distribution have been effectively modeled as shifts in the mean and variance of deep image features. We adapt adversarial training by directly perturbing feature statistics, rather than image pixels, to produce models that are robust to various unseen distributional shifts. We explore the relationship between these perturbations and distributional shifts by visualizing adversarial features. Our proposed method, Adversarial Batch Normalization (AdvBN), is a single network layer that generates worst-case feature perturbations during training. By fine-tuning neural networks on adversarial feature distributions, we observe improved robustness of networks to various unseen distributional shifts, including style variations and image corruptions. In addition, we show that our proposed adversarial feature perturbation can be complementary to existing image space data augmentation methods, leading to improved performance. The source code and pre-trained models are released at \url{https://github.com/azshue/AdvBN}.

        ----

        ## [2148] Continuized Accelerations of Deterministic and Stochastic Gradient Descents, and of Gossip Algorithms

        **Authors**: *Mathieu Even, Raphaël Berthier, Francis R. Bach, Nicolas Flammarion, Hadrien Hendrikx, Pierre Gaillard, Laurent Massoulié, Adrien B. Taylor*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec26fc2eb2b75aece19c70392dc744c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec26fc2eb2b75aece19c70392dc744c2-Abstract.html)

        **Abstract**:

        We introduce the ``continuized'' Nesterov acceleration, a close variant of Nesterov acceleration whose variables are indexed by a continuous time parameter. The two variables continuously mix following a linear ordinary differential equation and take gradient steps at random times. This continuized variant benefits from the best of the continuous and the discrete frameworks: as a continuous process, one can use differential calculus to analyze convergence and obtain analytical expressions for the parameters; but a discretization of the continuized process can be computed exactly with convergence rates similar to those of Nesterov original acceleration. We show that the discretization has the same structure as Nesterov acceleration, but with random parameters. We provide continuized Nesterov acceleration under deterministic as well as stochastic gradients, with either additive or multiplicative noise.  Finally, using our continuized framework and expressing the gossip averaging problem as the stochastic minimization of a certain energy function, we provide the first rigorous acceleration of asynchronous gossip algorithms.

        ----

        ## [2149] Natural continual learning: success is a journey, not (just) a destination

        **Authors**: *Ta-Chu Kao, Kristopher T. Jensen, Gido van de Ven, Alberto Bernacchia, Guillaume Hennequin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec5aa0b7846082a2415f0902f0da88f2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec5aa0b7846082a2415f0902f0da88f2-Abstract.html)

        **Abstract**:

        Biological agents are known to learn many different tasks over the course of their lives, and to be able to revisit previous tasks and behaviors with little to no loss in performance. In contrast, artificial agents are prone to ‘catastrophic forgetting’ whereby performance on previous tasks deteriorates rapidly as new ones are acquired. This shortcoming has recently been addressed using methods that encourage parameters to stay close to those used for previous tasks. This can be done by (i) using specific parameter regularizers that map out suitable destinations in parameter space, or (ii) guiding the optimization journey by projecting gradients into subspaces that do not interfere with previous tasks. However, these methods often exhibit subpar performance in both feedforward and recurrent neural networks, with recurrent networks being of interest to the study of neural dynamics supporting biological continual learning. In this work, we propose Natural Continual Learning (NCL), a new method that unifies weight regularization and projected gradient descent. NCL uses Bayesian weight regularization to encourage good performance on all tasks at convergence and combines this with gradient projection using the prior precision, which prevents catastrophic forgetting during optimization. Our method outperforms both standard weight regularization techniques and projection based approaches when applied to continual learning problems in feedforward and recurrent networks. Finally, the trained networks evolve task-specific dynamics that are strongly preserved as new tasks are learned, similar to experimental findings in biological circuits.

        ----

        ## [2150] Individual Privacy Accounting via a Rényi Filter

        **Authors**: *Vitaly Feldman, Tijana Zrnic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec7f346604f518906d35ef0492709f78-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec7f346604f518906d35ef0492709f78-Abstract.html)

        **Abstract**:

        We consider a sequential setting in which a single dataset of individuals is used to perform adaptively-chosen analyses, while ensuring that the differential privacy loss of each participant does not exceed a pre-specified privacy budget. The standard approach to this problem relies on bounding a worst-case estimate of the privacy loss over all individuals and all possible values of their data, for every single analysis. Yet, in many scenarios this approach is overly conservative, especially for "typical" data points which incur little privacy loss by participation in most of the analyses. In this work, we give a method for tighter privacy loss accounting based on the value of a personalized privacy loss estimate for each individual in each analysis. To implement the accounting method we design a filter for Rényi differential privacy. A filter is a tool that ensures that the privacy parameter of a composed sequence of algorithms with adaptively-chosen privacy parameters does not exceed a pre-specified budget. Our filter is simpler and tighter than the known filter for $(\epsilon,\delta)$-differential privacy by Rogers et al. (2016). We apply our results to the analysis of noisy gradient descent and show that personalized accounting can be practical, easy to implement, and can only make the privacy-utility tradeoff tighter.

        ----

        ## [2151] Post-Training Quantization for Vision Transformer

        **Authors**: *Zhenhua Liu, Yunhe Wang, Kai Han, Wei Zhang, Siwei Ma, Wen Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec8956637a99787bd197eacd77acce5e-Abstract.html)

        **Abstract**:

        Recently, transformer has achieved remarkable performance on a variety of computer vision applications. Compared with mainstream convolutional neural networks, vision transformers are often of sophisticated architectures for extracting powerful feature representations, which are more difficult to be developed on mobile devices. In this paper, we present an effective post-training quantization algorithm for reducing the memory storage and computational costs of vision transformers. Basically, the quantization task can be regarded as finding the optimal low-bit quantization intervals for weights and inputs, respectively. To preserve the functionality of the attention mechanism, we introduce a ranking loss into the conventional quantization objective that aims to keep the relative order of the self-attention results after quantization. Moreover, we thoroughly analyze the relationship between quantization loss of different layers and the feature diversity, and explore a mixed-precision quantization scheme by exploiting the nuclear norm of each attention map and output feature. The effectiveness of the proposed method is verified on several benchmark models and datasets, which outperforms the state-of-the-art post-training quantization algorithms. For instance, we can obtain an 81.29% top-1 accuracy using DeiT-B model on ImageNet dataset with about 8-bit quantization. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/VT-PTQ.

        ----

        ## [2152] Unsupervised Part Discovery from Contrastive Reconstruction

        **Authors**: *Subhabrata Choudhury, Iro Laina, Christian Rupprecht, Andrea Vedaldi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ec8ce6abb3e952a85b8551ba726a1227-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ec8ce6abb3e952a85b8551ba726a1227-Abstract.html)

        **Abstract**:

        The goal of self-supervised visual representation learning is to learn strong, transferable image representations, with the majority of research focusing on object or scene level. On the other hand, representation learning at part level has received significantly less attention. In this paper, we propose an unsupervised approach to object part discovery and segmentation and make three contributions. First, we construct a proxy task through a set of objectives that encourages the model to learn a meaningful decomposition of the image into its parts. Secondly, prior work argues for reconstructing or clustering pre-computed features as a proxy to parts; we show empirically that this alone is unlikely to find meaningful parts; mainly because of their low resolution and the tendency of classification networks to spatially smear out information. We suggest that image reconstruction at the level of pixels can alleviate this problem, acting as a complementary cue. Lastly, we show that the standard evaluation based on keypoint regression does not correlate well with segmentation quality and thus introduce different metrics, NMI and ARI, that better characterize the decomposition of objects into parts. Our method yields semantic parts which are consistent across fine-grained but visually distinct categories, outperforming the state of the art on three benchmark datasets. Code is available at the project page: https://www.robots.ox.ac.uk/~vgg/research/unsup-parts/.

        ----

        ## [2153] ASSANet: An Anisotropic Separable Set Abstraction for Efficient Point Cloud Representation Learning

        **Authors**: *Guocheng Qian, Hasan Hammoud, Guohao Li, Ali K. Thabet, Bernard Ghanem*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ecf5631507a8aedcae34cef231aa7348-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ecf5631507a8aedcae34cef231aa7348-Abstract.html)

        **Abstract**:

        Access to 3D point cloud representations has been widely facilitated by LiDAR sensors embedded in various mobile devices. This has led to an emerging need for fast and accurate point cloud processing techniques. In this paper, we revisit and dive deeper into PointNet++, one of the most influential yet under-explored networks, and develop faster and more accurate variants of the model. We first present a novel Separable Set Abstraction (SA) module that disentangles the vanilla SA module used in PointNet++ into two separate learning stages: (1) learning channel correlation and (2) learning spatial correlation. The Separable SA module is significantly faster than the vanilla version, yet it achieves comparable performance.  We then introduce a new Anisotropic Reduction function into our Separable SA module and propose an Anisotropic Separable SA (ASSA) module that substantially increases the network's accuracy. We later replace the vanilla SA modules in PointNet++ with the proposed ASSA modules, and denote the modified network as ASSANet. Extensive experiments on point cloud classification, semantic segmentation, and part segmentation show that ASSANet outperforms PointNet++ and other methods, achieving much higher accuracy and faster speeds. In particular, ASSANet outperforms PointNet++ by $7.4$ mIoU on S3DIS Area 5, while maintaining $1.6 \times $ faster inference speed on a single NVIDIA 2080Ti GPU. Our scaled ASSANet variant achieves $66.8$ mIoU and outperforms KPConv, while being more than $54 \times$ faster.

        ----

        ## [2154] An Empirical Investigation of Domain Generalization with Empirical Risk Minimizers

        **Authors**: *Ramakrishna Vedantam, David Lopez-Paz, David J. Schwab*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ecf9902e0f61677c8de25ae60b654669-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ecf9902e0f61677c8de25ae60b654669-Abstract.html)

        **Abstract**:

        Recent work demonstrates that deep neural networks trained using Empirical Risk Minimization (ERM) can generalize under distribution shift, outperforming specialized training algorithms for domain generalization. The goal of this paper is to further understand this phenomenon. In particular, we study the extent to which the seminal domain adaptation theory of Ben-David et al. (2007) explains the performance of ERMs. Perhaps surprisingly, we find that this theory does not provide a tight explanation of the out-of-domain generalization observed across a large number of ERM models trained on three popular domain generalization datasets. This motivates us to investigate other possible measures—that, however, lack theory—which could explain generalization in this setting. Our investigation reveals that measures relating to the Fisher information, predictive entropy, and maximum mean discrepancy are good predictors of the out-of-distribution generalization of ERM models. We hope that our work helps galvanize the community towards building a better understanding of when deep networks trained with ERM generalize out-of-distribution.

        ----

        ## [2155] Fair Sequential Selection Using Supervised Learning Models

        **Authors**: *Mohammad Mahdi Khalili, Xueru Zhang, Mahed Abroshan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ed277964a8959e72a0d987e598dfbe72-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ed277964a8959e72a0d987e598dfbe72-Abstract.html)

        **Abstract**:

        We consider a selection problem where sequentially arrived applicants apply for a limited number of positions/jobs. At each time step, a decision maker accepts or rejects the given applicant using a pre-trained supervised learning model until all the vacant positions are filled. In this paper, we discuss whether the fairness notions (e.g., equal opportunity, statistical parity, etc.) that are commonly used in classification problems are suitable for the sequential selection problems. In particular, we show that even with a pre-trained model that satisfies the common fairness notions, the selection outcomes may still be biased against certain demographic groups. This observation implies that the fairness notions used in classification problems are not suitable for a selection problem where the applicants compete for a limited number of positions.  We introduce a new fairness notion, ``Equal Selection (ES),'' suitable for sequential selection problems and propose a post-processing approach to satisfy the ES fairness notion. We also consider a setting where the applicants have privacy concerns, and the decision maker only has access to the noisy version of sensitive attributes. In this setting, we can show that the \textit{perfect} ES fairness can still be attained under certain conditions.

        ----

        ## [2156] Towards Sample-efficient Overparameterized Meta-learning

        **Authors**: *Yue Sun, Adhyyan Narang, Halil Ibrahim Gulluk, Samet Oymak, Maryam Fazel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ed46558a56a4a26b96a68738a0d28273-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ed46558a56a4a26b96a68738a0d28273-Abstract.html)

        **Abstract**:

        An overarching goal in machine learning is to build a generalizable model with few samples. To this end, overparameterization has been the subject of immense interest to explain the generalization ability of deep nets even when the size of the dataset is smaller than that of the model. While the prior literature focuses on the classical supervised setting, this paper aims to demystify overparameterization for meta-learning. Here we have a sequence of linear-regression tasks and we ask: (1) Given earlier tasks, what is the optimal linear representation of features for a new downstream task? and (2) How many samples do we need to build this representation? This work shows that surprisingly, overparameterization arises as a natural answer to these fundamental meta-learning questions. Specifically, for (1), we first show that learning the optimal representation coincides with the problem of designing a task-aware regularization to promote inductive bias. We leverage this inductive bias to explain how the downstream task actually benefits from overparameterization, in contrast to prior works on few-shot learning. For (2), we develop a theory to explain how feature covariance can implicitly help reduce the sample complexity well below the degrees of freedom and lead to small estimation error. We then integrate these findings to obtain an overall performance guarantee for our meta-learning algorithm. Numerical experiments on real and synthetic data verify our insights on overparameterized meta-learning.

        ----

        ## [2157] ScaleCert: Scalable Certified Defense against Adversarial Patches with Sparse Superficial Layers

        **Authors**: *Husheng Han, Kaidi Xu, Xing Hu, Xiaobing Chen, Ling Liang, Zidong Du, Qi Guo, Yanzhi Wang, Yunji Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ed519c02f134f2cdd836cba387b6a3c8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ed519c02f134f2cdd836cba387b6a3c8-Abstract.html)

        **Abstract**:

        Adversarial patch attacks that craft the pixels in a confined region of the input images show their powerful attack effectiveness in physical environments even with noises or deformations. Existing certified defenses towards adversarial patch attacks work well on small images like MNIST and CIFAR-10 datasets, but achieve very poor certified accuracy on higher-resolution images like ImageNet. It is urgent to design both robust and effective defenses against such a practical and harmful attack in industry-level larger images. In this work, we propose the certified defense methodology that achieves high provable robustness for high-resolution images and largely improves the practicality for real adoption of the certified defense. The basic insight of our work is that the adversarial patch intends to leverage localized superficial important neurons (SIN) to manipulate the prediction results. Hence, we leverage the SIN-based DNN compression techniques to significantly improve the certified accuracy, by reducing the adversarial region searching overhead and filtering the prediction noises. Our experimental results show that the certified accuracy is increased from 36.3%  (the state-of-the-art certified detection)  to 60.4%on the ImageNet dataset, largely pushing the certified defenses for practical use.

        ----

        ## [2158] Towards mental time travel: a hierarchical memory for reinforcement learning agents

        **Authors**: *Andrew K. Lampinen, Stephanie C. Y. Chan, Andrea Banino, Felix Hill*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ed519dacc89b2bead3f453b0b05a4a8b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ed519dacc89b2bead3f453b0b05a4a8b-Abstract.html)

        **Abstract**:

        Reinforcement learning agents often forget details of the past, especially after delays or distractor tasks. Agents with common memory architectures struggle to recall and integrate across multiple timesteps of a past event, or even to recall the details of a single timestep that is followed by distractor tasks. To address these limitations, we propose a Hierarchical Chunk Attention Memory (HCAM), that helps agents to remember the past in detail. HCAM stores memories by dividing the past into chunks, and recalls by first performing high-level attention over coarse summaries of the chunks, and then performing detailed attention within only the most relevant chunks. An agent with HCAM can therefore "mentally time-travel"--remember past events in detail without attending to all intervening events. We show that agents with HCAM substantially outperform agents with other memory architectures at tasks requiring long-term recall, retention, or reasoning over memory. These include recalling where an object is hidden in a 3D environment, rapidly learning to navigate efficiently in a new neighborhood, and rapidly learning and retaining new words. Agents with HCAM can extrapolate to task sequences much longer than they were trained on, and can even generalize zero-shot from a meta-learning setting to maintaining knowledge across episodes. HCAM improve agent sample efficiency, generalization, and generality (by solving tasks that previously required specialized architectures). Our work is a step towards agents that can learn, interact, and adapt in complex and temporally-extended environments.

        ----

        ## [2159] Beyond Tikhonov: faster learning with self-concordant losses, via iterative regularization

        **Authors**: *Gaspard Beugnot, Julien Mairal, Alessandro Rudi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eda80a3d5b344bc40f3bc04f65b7a357-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eda80a3d5b344bc40f3bc04f65b7a357-Abstract.html)

        **Abstract**:

        The theory of spectral filtering is a remarkable tool to understand the statistical properties of learning with kernels. For least squares, it allows to derive various regularization schemes that yield faster convergence rates of the excess risk than with Tikhonov regularization. This is typically achieved by leveraging classical assumptions called source and capacity conditions, which characterize the difficulty of the learning task. In order to understand estimators derived from other loss functions, Marteau-Ferey et al. have extended the theory of Tikhonov regularization to generalized self concordant loss functions (GSC), which contain, e.g., the logistic loss. In this paper, we go a step further and show that fast and optimal rates can be achieved for GSC by using the iterated Tikhonov regularization scheme, which is intrinsically related to the proximal point method in optimization, and overcomes the limitation of the classical Tikhonov regularization.

        ----

        ## [2160] Variational Bayesian Reinforcement Learning with Regret Bounds

        **Authors**: *Brendan O'Donoghue*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/edb446b67d69adbfe9a21068982000c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/edb446b67d69adbfe9a21068982000c2-Abstract.html)

        **Abstract**:

        We consider the exploration-exploitation trade-off in reinforcement learning and show that an agent endowed with an exponential epistemic-risk-seeking utility function explores efficiently, as measured by regret. The state-action values induced by the exponential utility satisfy a Bellman recursion, so we can use dynamic programming to compute them.  We call the resulting algorithm K-learning (for knowledge) and the risk-seeking utility ensures that the associated state-action values (K-values) are optimistic for the expected optimal Q-values under the posterior.  The exponential utility function induces a Boltzmann exploration policy for which the 'temperature' parameter is equal to the risk-seeking parameter and is carefully controlled to yield a Bayes regret bound of $\tilde O(L^{3/2} \sqrt{S A T})$, where $L$ is the time horizon, $S$ is the number of states, $A$ is the number of actions, and $T$ is the total number of elapsed timesteps. We conclude with a numerical example demonstrating that K-learning is competitive with other state-of-the-art algorithms in practice.

        ----

        ## [2161] Logarithmic Regret from Sublinear Hints

        **Authors**: *Aditya Bhaskara, Ashok Cutkosky, Ravi Kumar, Manish Purohit*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/edb947f2bbceb132245fdde9c59d3f59-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/edb947f2bbceb132245fdde9c59d3f59-Abstract.html)

        **Abstract**:

        We consider the online linear optimization problem, where at every step the algorithm plays a point $x_t$ in the unit ball, and suffers loss $\langle c_t, x_t \rangle$ for some cost vector $c_t$ that is then revealed to the algorithm. Recent work showed that if an algorithm  receives a _hint_ $h_t$ that has non-trivial correlation with $c_t$ before it plays $x_t$, then it can achieve a regret guarantee of $O(\log T)$, improving on the bound of $\Theta(\sqrt{T})$ in the standard setting. In this work, we study the question of whether an algorithm really requires a hint at _every_ time step. Somewhat surprisingly, we show that an algorithm can obtain $O(\log T)$ regret with just $O(\sqrt{T})$ hints under a natural query model; in contrast, we also show that $o(\sqrt{T})$ hints cannot guarantee better than $\Omega(\sqrt{T})$ regret. We give two applications of our result, to the well-studied setting of {\em optimistic} regret bounds, and to the problem of online learning with abstention.

        ----

        ## [2162] Independent mechanism analysis, a new concept?

        **Authors**: *Luigi Gresele, Julius von Kügelgen, Vincent Stimper, Bernhard Schölkopf, Michel Besserve*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/edc27f139c3b4e4bb29d1cdbc45663f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/edc27f139c3b4e4bb29d1cdbc45663f9-Abstract.html)

        **Abstract**:

        Independent component analysis provides a principled framework for unsupervised representation learning, with solid theory on the identifiability of the latent code that generated the data, given only observations of mixtures thereof. Unfortunately, when the mixing is nonlinear, the model is provably nonidentifiable, since statistical independence alone does not sufficiently constrain the problem. Identifiability can be recovered in settings where additional, typically observed variables are included in the generative process. We investigate an alternative path and consider instead including assumptions reflecting the principle of independent causal mechanisms exploited in the field of causality. Specifically, our approach is motivated by thinking of each source as independently influencing the mixing process. This gives rise to a framework which we term independent mechanism analysis. We provide theoretical and empirical evidence that our approach circumvents a number of nonidentifiability issues arising in nonlinear blind source separation.

        ----

        ## [2163] Momentum Centering and Asynchronous Update for Adaptive Gradient Methods

        **Authors**: *Juntang Zhuang, Yifan Ding, Tommy Tang, Nicha C. Dvornek, Sekhar Tatikonda, James S. Duncan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eddea82ad2755b24c4e168c5fc2ebd40-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eddea82ad2755b24c4e168c5fc2ebd40-Abstract.html)

        **Abstract**:

        We propose ACProp (Asynchronous-centering-Prop), an adaptive optimizer which combines centering of second momentum and asynchronous update (e.g. for $t$-th update, denominator uses information up to step $t-1$, while numerator uses gradient at $t$-th step).  ACProp has both strong theoretical properties and empirical performance. With the example by Reddi et al. (2018), we show that asynchronous optimizers (e.g. AdaShift, ACProp) have weaker convergence condition than synchronous optimizers (e.g. Adam, RMSProp, AdaBelief); within asynchronous optimizers, we show that centering of second momentum further weakens the convergence condition. We demonstrate that ACProp has a convergence rate of $O(\frac{1}{\sqrt{T}})$ for the stochastic non-convex case, which matches the oracle rate and outperforms the $O(\frac{logT}{\sqrt{T}})$ rate of RMSProp and Adam. We validate ACProp in extensive empirical studies: ACProp outperforms both SGD and other adaptive optimizers in image classification with CNN, and outperforms well-tuned adaptive optimizers in the training of various GAN models, reinforcement learning and transformers. To sum up, ACProp has good theoretical properties including weak convergence condition and optimal convergence rate, and strong empirical performance including good generalization like SGD and training stability like Adam. We provide the implementation at \url{ https://github.com/juntang-zhuang/ACProp-Optimizer}.

        ----

        ## [2164] Robustness via Uncertainty-aware Cycle Consistency

        **Authors**: *Uddeshya Upadhyay, Yanbei Chen, Zeynep Akata*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ede7e2b6d13a41ddf9f4bdef84fdc737-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ede7e2b6d13a41ddf9f4bdef84fdc737-Abstract.html)

        **Abstract**:

        Unpaired image-to-image translation refers to learning inter-image-domain mapping without corresponding image pairs. Existing methods learn deterministic mappings without explicitly modelling the robustness to outliers or predictive uncertainty, leading to performance degradation when encountering unseen perturbations at test time. To address this, we propose a novel probabilistic method based on Uncertainty-aware Generalized Adaptive Cycle Consistency (UGAC), which models the per-pixel residual by generalized Gaussian distribution, capable of modelling heavy-tailed distributions. We compare our model with a wide variety of state-of-the-art methods on various challenging tasks including unpaired image translation of natural images spanning autonomous driving, maps, facades, and also in the medical imaging domain consisting of MRI. Experimental results demonstrate that our method exhibits stronger robustness towards unseen perturbations in test data. Code is released here: https://github.com/ExplainableML/UncertaintyAwareCycleConsistency.

        ----

        ## [2165] CBP: backpropagation with constraint on weight precision using a pseudo-Lagrange multiplier method

        **Authors**: *Guhyun Kim, Doo Seok Jeong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/edea298442a67de045e88dfb6e5ea4a2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/edea298442a67de045e88dfb6e5ea4a2-Abstract.html)

        **Abstract**:

        Backward propagation of errors (backpropagation) is a method to minimize objective functions (e.g., loss functions) of deep neural networks by identifying optimal sets of weights and biases. Imposing constraints on weight precision is often required to alleviate prohibitive workloads on hardware. Despite the remarkable success of backpropagation, the algorithm itself is not capable of considering such constraints unless additional algorithms are applied simultaneously. To address this issue, we propose the constrained backpropagation (CBP) algorithm based on the pseudo-Lagrange multiplier method to obtain the optimal set of weights that satisfy a given set of constraints. The defining characteristic of the proposed CBP algorithm is the utilization of a Lagrangian function (loss function plus constraint function) as its objective function. We considered various types of constraints â€” binary, ternary, one-bit shift, and two-bit shift weight constraints. As a post-training method, CBP applied to AlexNet, ResNet-18, ResNet-50, and GoogLeNet on ImageNet, which were pre-trained using the conventional backpropagation. For most cases, the proposed algorithm outperforms the state-of-the-art methods on ImageNet, e.g., 66.6\%, 74.4\%, and 64.0\% top-1 accuracy for ResNet-18, ResNet-50, and GoogLeNet with binary weights, respectively. This highlights CBP as a learning algorithm to address diverse constraints with the minimal performance loss by employing appropriate constraint functions. The code for CBP is publicly available at \url{https://github.com/dooseokjeong/CBP}.

        ----

        ## [2166] On the Sample Complexity of Privately Learning Axis-Aligned Rectangles

        **Authors**: *Menachem Sadigurschi, Uri Stemmer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ee0e95249268b86ff2053bef214bfeda-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ee0e95249268b86ff2053bef214bfeda-Abstract.html)

        **Abstract**:

        We revisit the fundamental problem of learning Axis-Aligned-Rectangles over a finite grid $X^d\subseteq\mathbb{R}^d$ with differential privacy. Existing results show that the sample complexity of this problem is at most $\min\left\{ d{\cdot}\log|X| \;,\; d^{1.5}{\cdot}\left(\log^*|X| \right)^{1.5}\right\}$. That is, existing constructions either require sample complexity that grows linearly with $\log|X|$, or else it grows super linearly with the dimension $d$.  We present a novel algorithm that reduces the sample complexity to only $\tilde{O}\left\{d{\cdot}\left(\log^*|X|\right)^{1.5}\right\}$,  attaining a dimensionality optimal dependency without requiring the sample complexity to grow with $\log|X|$. The technique used in order to attain this improvement involves the deletion of "exposed" data-points on the go, in a fashion designed to avoid the cost of the adaptive composition theorems.The core of this technique may be of individual interest, introducing a new method for constructing statistically-efficient private algorithms.

        ----

        ## [2167] Implicit Sparse Regularization: The Impact of Depth and Early Stopping

        **Authors**: *Jiangyuan Li, Thanh Van Nguyen, Chinmay Hegde, Ka Wai Wong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ee188463935a061dee6df8bf449cb882-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ee188463935a061dee6df8bf449cb882-Abstract.html)

        **Abstract**:

        In this paper, we study the implicit bias of gradient descent for sparse regression. We extend results on regression with quadratic parametrization, which amounts to depth-2 diagonal linear networks, to more general depth-$N$ networks, under more realistic settings of noise and correlated designs. We show that early stopping is crucial for gradient descent to converge to a sparse model, a phenomenon that we call \emph{implicit sparse regularization}. This result is in sharp contrast to known results for noiseless and uncorrelated-design cases.   We characterize the impact of depth and early stopping and show that for a general depth parameter $N$, gradient descent with early stopping achieves minimax optimal sparse recovery with sufficiently small initialization $w_0$ and step size $\eta$. In particular, we show that increasing depth enlarges the scale of working initialization and the early-stopping window so that this implicit sparse regularization effect is more likely to take place.

        ----

        ## [2168] Efficient Generalization with Distributionally Robust Learning

        **Authors**: *Soumyadip Ghosh, Mark S. Squillante, Ebisa D. Wollega*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ee1abc6b5f7c6acb34ad076b05d40815-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ee1abc6b5f7c6acb34ad076b05d40815-Abstract.html)

        **Abstract**:

        Distributionally robust learning (DRL) is increasingly seen as a viable method to train machine learning models for improved model generalization. These min-max formulations, however, are more difﬁcult to solve. We provide a new stochastic gradient descent algorithm to efﬁciently solve this DRL formulation. Our approach applies gradient descent to the outer minimization formulation and estimates the gradient of the inner maximization based on a sample average approximation. The latter uses a subset of the data sampled without replacement in each iteration, progressively increasing the subset size to ensure convergence. We rigorously establish convergence to a near-optimal solution under standard regularity assumptions and, for strongly convex losses, match the best known $O(\epsilon{ −1})$ rate of convergence up to a known threshold. Empirical results demonstrate the signiﬁcant beneﬁts of our approach over previous work in improving learning for model generalization.

        ----

        ## [2169] No-regret Online Learning over Riemannian Manifolds

        **Authors**: *Xi Wang, Zhipeng Tu, Yiguang Hong, Yingyi Wu, Guodong Shi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ee389847678a3a9d1ce9e4ca69200d06-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ee389847678a3a9d1ce9e4ca69200d06-Abstract.html)

        **Abstract**:

        We consider online optimization over Riemannian manifolds, where a learner attempts to minimize a sequence of time-varying loss functions defined on Riemannian manifolds. Though many Euclidean online convex optimization algorithms have been proven useful in a wide range of areas, less attention has been paid to their Riemannian counterparts. In this paper, we study Riemannian online gradient descent (R-OGD) on Hadamard manifolds for both geodesically convex and strongly geodesically convex loss functions, and Riemannian bandit algorithm (R-BAN) on Hadamard homogeneous manifolds for geodesically convex functions. We establish upper bounds on the regrets of the problem with respect to time horizon, manifold curvature, and manifold dimension. We also find a universal lower bound for the achievable regret by constructing an online convex optimization problem on Hadamard manifolds. All the obtained regret bounds match the corresponding results are provided in Euclidean spaces. Finally, some numerical experiments validate our theoretical results.

        ----

        ## [2170] Landmark-Guided Subgoal Generation in Hierarchical Reinforcement Learning

        **Authors**: *Junsu Kim, Younggyo Seo, Jinwoo Shin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html)

        **Abstract**:

        Goal-conditioned hierarchical reinforcement learning (HRL) has shown promising results for solving complex and long-horizon RL tasks. However, the action space of high-level policy in the goal-conditioned HRL is often large, so it results in poor exploration, leading to inefficiency in training. In this paper, we present HIerarchical reinforcement learning Guided by Landmarks (HIGL), a novel framework for training a high-level policy with a reduced action space guided by landmarks, i.e., promising states to explore. The key component of HIGL is twofold: (a) sampling landmarks that are informative for exploration and (b) encouraging the high level policy to generate a subgoal towards a selected landmark. For (a), we consider two criteria: coverage of the entire visited state space (i.e., dispersion of states) and novelty of states (i.e., prediction error of a state). For (b), we select a landmark as the very first landmark in the shortest path in a graph whose nodes are landmarks. Our experiments demonstrate that our framework outperforms prior-arts across a variety of control tasks, thanks to efficient exploration guided by landmarks.

        ----

        ## [2171] Minimax Regret for Stochastic Shortest Path

        **Authors**: *Alon Cohen, Yonathan Efroni, Yishay Mansour, Aviv Rosenberg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eeb69a3cb92300456b6a5f4162093851-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eeb69a3cb92300456b6a5f4162093851-Abstract.html)

        **Abstract**:

        We study the Stochastic Shortest Path (SSP) problem in which an agent has to reach a goal state in minimum total expected cost. In the learning formulation of the problem, the agent has no prior knowledge about the costs and dynamics of the model. She repeatedly interacts with the model for $K$ episodes, and has to minimize her regret. In this work we show that the minimax regret for this setting is $\widetilde O(\sqrt{ (B_\star^2 + B_\star) |S| |A| K})$ where $B_\star$ is a bound on the expected cost of the optimal policy from any state, $S$ is the state space, and $A$ is the action space. This matches the $\Omega (\sqrt{ B_\star^2 |S| |A| K})$ lower bound of Rosenberg et al. [2020] for $B_\star \ge 1$, and improves their regret bound by a factor of $\sqrt{|S|}$. For $B_\star < 1$ we prove a matching lower bound of $\Omega (\sqrt{ B_\star |S| |A| K})$. Our algorithm is based on a novel reduction from SSP to finite-horizon MDPs.  To that end, we provide an algorithm for the finite-horizon setting whose leading term in the regret depends polynomially on the expected cost of the optimal policy and only logarithmically on the horizon.

        ----

        ## [2172] Parametrized Quantum Policies for Reinforcement Learning

        **Authors**: *Sofiène Jerbi, Casper Gyurik, Simon C. Marshall, Hans J. Briegel, Vedran Dunjko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eec96a7f788e88184c0e713456026f3f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eec96a7f788e88184c0e713456026f3f-Abstract.html)

        **Abstract**:

        With the advent of real-world quantum computing, the idea that parametrized quantum computations can be used as hypothesis families in a quantum-classical machine learning system is gaining increasing traction. Such hybrid systems have already shown the potential to tackle real-world tasks in supervised and generative learning, and recent works have established their provable advantages in special artificial tasks. Yet, in the case of reinforcement learning, which is arguably most challenging and where learning boosts would be extremely valuable, no proposal has been successful in solving even standard benchmarking tasks, nor in showing a theoretical learning advantage over classical algorithms. In this work, we achieve both. We propose a hybrid quantum-classical reinforcement learning model using very few qubits, which we show can be effectively trained to solve several standard benchmarking environments. Moreover, we demonstrate, and formally prove, the ability of parametrized quantum circuits to solve certain learning tasks that are intractable to classical models, including current state-of-art deep neural networks, under the widely-believed classical hardness of the discrete logarithm problem.

        ----

        ## [2173] On Pathologies in KL-Regularized Reinforcement Learning from Expert Demonstrations

        **Authors**: *Tim G. J. Rudner, Cong Lu, Michael A. Osborne, Yarin Gal, Yee Whye Teh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eecca5b6365d9607ee5a9d336962c534-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eecca5b6365d9607ee5a9d336962c534-Abstract.html)

        **Abstract**:

        KL-regularized reinforcement learning from expert demonstrations has proved successful in improving the sample efficiency of deep reinforcement learning algorithms, allowing them to be applied to challenging physical real-world tasks. However, we show that KL-regularized reinforcement learning with behavioral reference policies derived from expert demonstrations can suffer from pathological training dynamics that can lead to slow, unstable, and suboptimal online learning. We show empirically that the pathology occurs for commonly chosen behavioral policy classes and demonstrate its impact on sample efficiency and online policy performance. Finally, we show that the pathology can be remedied by non-parametric behavioral reference policies and that this allows KL-regularized reinforcement learning to significantly outperform state-of-the-art approaches on a variety of challenging locomotion and dexterous hand manipulation tasks.

        ----

        ## [2174] Conditional Generation Using Polynomial Expansions

        **Authors**: *Grigorios Chrysos, Markos Georgopoulos, Yannis Panagakis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef0d3930a7b6c95bd2b32ed45989c61f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef0d3930a7b6c95bd2b32ed45989c61f-Abstract.html)

        **Abstract**:

        Generative modeling has evolved to a notable field of machine learning. Deep polynomial neural networks (PNNs) have demonstrated impressive results in unsupervised image generation, where the task is to map an input vector (i.e., noise) to a synthesized image. However, the success of PNNs has not been replicated in conditional generation tasks, such as super-resolution. Existing PNNs focus on single-variable polynomial expansions which do not fare well to two-variable inputs, i.e., the noise variable and the conditional variable. In this work, we introduce a general framework, called CoPE, that enables a polynomial expansion of two input variables and captures their auto- and cross-correlations. We exhibit how CoPE can be trivially augmented to accept an arbitrary number of input variables. CoPE is evaluated in five tasks (class-conditional generation, inverse problems, edges-to-image translation, image-to-image translation, attribute-guided generation) involving eight datasets. The thorough evaluation suggests that CoPE can be useful for tackling diverse conditional generation tasks. The source code of CoPE is available at https://github.com/grigorisg9gr/polynomialnetsforconditionalgeneration.

        ----

        ## [2175] Efficient constrained sampling via the mirror-Langevin algorithm

        **Authors**: *Kwangjun Ahn, Sinho Chewi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef1e491a766ce3127556063d49bc2f98-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef1e491a766ce3127556063d49bc2f98-Abstract.html)

        **Abstract**:

        We propose a new discretization of the mirror-Langevin diffusion and give a crisp proof of its convergence. Our analysis uses relative convexity/smoothness and self-concordance, ideas which originated in convex optimization, together with a new result in optimal transport that generalizes the displacement convexity of the entropy. Unlike prior works, our result both (1) requires much weaker assumptions on the mirror map and the target distribution, and (2) has vanishing bias as the step size tends to zero. In particular, for the task of sampling from a log-concave distribution supported on a compact set, our theoretical results are significantly better than the existing guarantees.

        ----

        ## [2176] Adaptive Online Packing-guided Search for POMDPs

        **Authors**: *Chenyang Wu, Guoyu Yang, Zongzhang Zhang, Yang Yu, Dong Li, Wulong Liu, Jianye Hao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef41d488755367316f04fc0e0e9dc9fc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef41d488755367316f04fc0e0e9dc9fc-Abstract.html)

        **Abstract**:

        The partially observable Markov decision process (POMDP) provides a general framework for modeling an agent's decision process with state uncertainty, and online planning plays a pivotal role in solving it. A belief is a distribution of states representing state uncertainty. Methods for large-scale POMDP problems rely on the same idea of sampling both states and observations. That is, instead of exact belief updating, a collection of sampled states is used to approximate the belief; instead of considering all possible observations, only a set of sampled observations are considered. Inspired by this, we take one step further and propose an online planning algorithm, Adaptive Online Packing-guided Search (AdaOPS), to better approximate beliefs with adaptive particle filter technique and balance estimation bias and variance by fusing similar observation branches. Theoretically, our algorithm is guaranteed to find an $\epsilon$-optimal policy with a high probability given enough planning time under some mild assumptions. We evaluate our algorithm on several tricky POMDP domains, and it outperforms the state-of-the-art in all of them.

        ----

        ## [2177] Turing Completeness of Bounded-Precision Recurrent Neural Networks

        **Authors**: *Stephen Chung, Hava T. Siegelmann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef452c63f81d0105dd4486f775adec81-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef452c63f81d0105dd4486f775adec81-Abstract.html)

        **Abstract**:

        Previous works have proved that recurrent neural networks (RNNs) are Turing-complete. However, in the proofs, the RNNs allow for neurons with unbounded precision, which is neither practical in implementation nor biologically plausible. To remove this assumption, we propose a dynamically growing memory module made of neurons of fixed precision. The memory module dynamically recruits new neurons when more memories are needed, and releases them when memories become irrelevant. We prove that a 54-neuron bounded-precision RNN with growing memory modules can simulate a Universal Turing Machine, with time complexity linear in the simulated machine's time and independent of the memory size. The result is extendable to various other stack-augmented RNNs. Furthermore, we analyze the Turing completeness of both unbounded-precision and bounded-precision RNNs, revisiting and extending the theoretical foundations of RNNs.

        ----

        ## [2178] End-to-end Multi-modal Video Temporal Grounding

        **Authors**: *Yi-Wen Chen, Yi-Hsuan Tsai, Ming-Hsuan Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef50c335cca9f340bde656363ebd02fd-Abstract.html)

        **Abstract**:

        We address the problem of text-guided video temporal grounding, which aims to identify the time interval of a certain event based on a natural language description. Different from most existing methods that only consider RGB images as visual features, we propose a multi-modal framework to extract complementary information from videos. Specifically, we adopt RGB images for appearance, optical flow for motion, and depth maps for image structure. While RGB images provide abundant visual cues of certain events, the performance may be affected by background clutters. Therefore, we use optical flow to focus on large motion and depth maps to infer the scene configuration when the action is related to objects recognizable with their shapes. To integrate the three modalities more effectively and enable inter-modal learning, we design a dynamic fusion scheme with transformers to model the interactions between modalities. Furthermore, we apply intra-modal self-supervised learning to enhance feature representations across videos for each modality, which also facilitates multi-modal learning. We conduct extensive experiments on the Charades-STA and ActivityNet Captions datasets, and show that the proposed method performs favorably against state-of-the-art approaches.

        ----

        ## [2179] How Powerful are Performance Predictors in Neural Architecture Search?

        **Authors**: *Colin White, Arber Zela, Robin Ru, Yang Liu, Frank Hutter*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef575e8837d065a1683c022d2077d342-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef575e8837d065a1683c022d2077d342-Abstract.html)

        **Abstract**:

        Early methods in the rapidly developing field of neural architecture search (NAS) required fully training thousands of neural networks. To reduce this extreme computational cost, dozens of techniques have since been proposed to predict the final performance of neural architectures. Despite the success of such performance prediction methods, it is not well-understood how different families of techniques compare to one another, due to the lack of an agreed-upon evaluation metric and optimization for different constraints on the initialization time and query time. In this work, we give the first large-scale study of performance predictors by analyzing 31 techniques ranging from learning curve extrapolation, to weight-sharing, to supervised learning, to zero-cost proxies. We test a number of correlation- and rank-based performance measures in a variety of settings, as well as the ability of each technique to speed up predictor-based NAS frameworks. Our results act as recommendations for the best predictors to use in different settings, and we show that certain families of predictors can be combined to achieve even better predictive power, opening up promising research directions. We release our code, featuring a library of 31 performance predictors.

        ----

        ## [2180] Stylized Dialogue Generation with Multi-Pass Dual Learning

        **Authors**: *Jinpeng Li, Yingce Xia, Rui Yan, Hongda Sun, Dongyan Zhao, Tie-Yan Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef67f7c2d86352c2c42e19d20f881f53-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef67f7c2d86352c2c42e19d20f881f53-Abstract.html)

        **Abstract**:

        Stylized dialogue generation, which aims to generate a given-style response for an input context, plays a vital role in intelligent dialogue systems. Considering there is no parallel data between the contexts and the responses of target style S1, existing works mainly use back translation to generate stylized synthetic data for training, where the data about context, target style S1 and an intermediate style S0 is used. However, the interaction among these texts is not fully exploited, and the pseudo contexts are not adequately modeled. To overcome the above difficulties, we propose multi-pass dual learning (MPDL), which leverages the duality among the context, response of style S1 and response of style S_0. MPDL builds mappings among the above three domains, where the context should be reconstructed by the MPDL framework, and the reconstruction error is used as the training signal. To evaluate the quality of synthetic data, we also introduce discriminators that effectively measure how a pseudo sequence matches the specific domain, and the evaluation result is used as the weight for that data. Evaluation results indicate that our method obtains significant improvement over previous baselines.

        ----

        ## [2181] Entropy-based adaptive Hamiltonian Monte Carlo

        **Authors**: *Marcel Hirt, Michalis K. Titsias, Petros Dellaportas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef82567365bc6a0efb05d21837257424-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef82567365bc6a0efb05d21837257424-Abstract.html)

        **Abstract**:

        Hamiltonian Monte Carlo (HMC) is a popular Markov Chain Monte Carlo (MCMC) algorithm to sample from an unnormalized probability distribution. A leapfrog integrator is commonly used to implement HMC in practice, but its performance can be sensitive to the choice of mass matrix used therein. We develop a gradient-based algorithm that allows for the adaptation of the mass matrix by encouraging the leapfrog integrator to have high acceptance rates while also exploring all dimensions jointly. In contrast to previous work that adapt the hyperparameters of HMC using some form of expected squared jumping distance, the adaptation strategy suggested here aims to increase sampling efficiency by maximizing an approximation of the proposal entropy. We illustrate that using multiple gradients in the HMC proposal can be beneficial compared to a single gradient-step in Metropolis-adjusted Langevin proposals. Empirical evidence suggests that the adaptation method can outperform different versions of HMC schemes by adjusting the mass matrix to the geometry of the target distribution and by providing some control on the integration time.

        ----

        ## [2182] Continual World: A Robotic Benchmark For Continual Reinforcement Learning

        **Authors**: *Maciej Wolczyk, Michal Zajac, Razvan Pascanu, Lukasz Kucinski, Piotr Milos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef8446f35513a8d6aa2308357a268a7e-Abstract.html)

        **Abstract**:

        Continual learning (CL) --- the ability to continuously learn, building on previously acquired knowledge --- is a natural requirement for long-lived autonomous reinforcement learning (RL) agents. While building such agents, one needs to balance opposing desiderata, such as constraints on capacity and compute, the ability to not catastrophically forget, and to exhibit positive transfer on new tasks. Understanding the right trade-off is conceptually and computationally challenging, which we argue has led the community to overly focus on catastrophic forgetting.  In response to these issues, we advocate for the need to prioritize forward transfer and propose Continual World, a benchmark consisting of realistic and meaningfully diverse robotic tasks built on top of Meta-World  as a testbed. Following an in-depth empirical evaluation of existing CL methods, we pinpoint their limitations and highlight unique algorithmic challenges in the RL setting. Our benchmark aims to provide a meaningful and computationally inexpensive challenge for the community and thus help better understand the performance of existing and future solutions. Information about the benchmark, including the open-source code, is available at https://sites.google.com/view/continualworld.

        ----

        ## [2183] Towards Best-of-All-Worlds Online Learning with Feedback Graphs

        **Authors**: *Liad Erez, Tomer Koren*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ef8f94395be9fd78b7d0aeecf7864a03-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ef8f94395be9fd78b7d0aeecf7864a03-Abstract.html)

        **Abstract**:

        We study the online learning with feedback graphs framework introduced by Mannor and Shamir (2011), in which the feedback received by the online learner is specified by a graph $G$ over the available actions.  We develop an algorithm that simultaneously achieves regret bounds of the form: $O(\sqrt{\theta(G) T})$ with adversarial losses; $O(\theta(G)\mathrm{polylog}{T})$ with stochastic losses; and $O(\theta(G)\mathrm{polylog}{T} + \sqrt{\theta(G) C})$ with stochastic losses subject to $C$ adversarial corruptions.  Here, $\theta(G)$ is the $clique~covering~number$ of the graph $G$.  Our algorithm is an instantiation of Follow-the-Regularized-Leader with a novel regularization that can be seen as a product of a Tsallis entropy component (inspired by Zimmert and Seldin (2019)) and a Shannon entropy component (analyzed in the corrupted stochastic case by Amir et al. (2020)), thus subtly interpolating between the two forms of entropies.  One of our key technical contributions is in establishing the convexity of this regularizer and controlling its inverse Hessian, despite its complex product structure.

        ----

        ## [2184] ViTAE: Vision Transformer Advanced by Exploring Intrinsic Inductive Bias

        **Authors**: *Yufei Xu, Qiming Zhang, Jing Zhang, Dacheng Tao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/efb76cff97aaf057654ef2f38cd77d73-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/efb76cff97aaf057654ef2f38cd77d73-Abstract.html)

        **Abstract**:

        Transformers have shown great potential in various computer vision tasks owing to their strong capability in modeling long-range dependency using the self-attention mechanism. Nevertheless, vision transformers treat an image as 1D sequence of visual tokens, lacking an intrinsic inductive bias (IB) in modeling local visual structures and dealing with scale variance. Alternatively, they require large-scale training data and longer training schedules to learn the IB implicitly. In this paper, we propose a new Vision Transformer Advanced by Exploring intrinsic IB from convolutions, i.e., ViTAE. Technically, ViTAE has several spatial pyramid reduction modules to downsample and embed the input image into tokens with rich multi-scale context by using multiple convolutions with different dilation rates. In this way, it acquires an intrinsic scale invariance IB and is able to learn robust feature representation for objects at various scales. Moreover, in each transformer layer, ViTAE has a convolution block in parallel to the multi-head self-attention module, whose features are fused and fed into the feed-forward network. Consequently, it has the intrinsic locality IB and is able to learn local features and global dependencies collaboratively. Experiments on ImageNet as well as downstream tasks prove the superiority of ViTAE over the baseline transformer and concurrent works. Source code and pretrained models will be available at https://github.com/Annbless/ViTAE.

        ----

        ## [2185] Open Rule Induction

        **Authors**: *Wanyun Cui, Xingran Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/efe34c4e2190e97d1adc625902822b13-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/efe34c4e2190e97d1adc625902822b13-Abstract.html)

        **Abstract**:

        Rules have a number of desirable properties. It is easy to understand,  infer new knowledge, and communicate with other inference systems. One weakness of the previous rule induction systems is that they only find rules within a knowledge base (KB) and therefore cannot generalize to more open and complex real-world rules. Recently, the language model (LM)-based rule generation are proposed to enhance the expressive power of the rules.In this paper, we revisit the differences between KB-based rule induction and LM-based rule generation. We argue that, while KB-based methods inducted rules by discovering data commonalitiess, the current LM-based methods are learning rules from rules''. This limits these methods to only producecanned'' rules whose patterns are constrained by the annotated rules, while discarding the rich expressive power of LMs for free text.Therefore, in this paper, we propose the open rule induction problem, which aims to induce open rules utilizing the knowledge in LMs. Besides, we propose the Orion (\underline{o}pen \underline{r}ule \underline{i}nducti\underline{on}) system to automatically mine open rules from LMs without supervision of annotated rules. We conducted extensive experiments to verify the quality and quantity of the inducted open rules. Surprisingly, when applying the open rules in downstream tasks (i.e. relation extraction), these automatically inducted rules even outperformed the manually annotated rules.

        ----

        ## [2186] Post-Contextual-Bandit Inference

        **Authors**: *Aurélien Bibaut, Maria Dimakopoulou, Nathan Kallus, Antoine Chambaz, Mark J. van der Laan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/eff3058117fd4cf4d4c3af12e273a40f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/eff3058117fd4cf4d4c3af12e273a40f-Abstract.html)

        **Abstract**:

        Contextual bandit algorithms are increasingly replacing non-adaptive A/B tests in e-commerce, healthcare, and policymaking because they can both improve outcomes for study participants and increase the chance of identifying good or even best policies. To support credible inference on novel interventions at the end of the study, nonetheless, we still want to construct valid confidence intervals on average treatment effects, subgroup effects, or value of new policies. The adaptive nature of the data collected by contextual bandit algorithms, however, makes this difficult: standard estimators are no longer asymptotically normally distributed and classic confidence intervals fail to provide correct coverage. While this has been addressed in non-contextual settings by using stabilized estimators, variance stabilized estimators in the contextual setting pose unique challenges that we tackle for the first time in this paper. We propose the Contextual Adaptive Doubly Robust (CADR) estimator, a novel estimator for policy value that is asymptotically normal under contextual adaptive data collection. The main technical challenge in constructing CADR is designing adaptive and consistent conditional standard deviation estimators for stabilization. Extensive numerical experiments using 57 OpenML datasets demonstrate that confidence intervals based on CADR uniquely provide correct coverage.

        ----

        ## [2187] Revisiting Discriminator in GAN Compression: A Generator-discriminator Cooperative Compression Scheme

        **Authors**: *Shaojie Li, Jie Wu, Xuefeng Xiao, Fei Chao, Xudong Mao, Rongrong Ji*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/effc299a1addb07e7089f9b269c31f2f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/effc299a1addb07e7089f9b269c31f2f-Abstract.html)

        **Abstract**:

        Recently, a series of algorithms have been explored for GAN compression, which aims to reduce tremendous computational overhead and memory usages when deploying GANs on resource-constrained edge devices. However, most of the existing GAN compression work only focuses on how to compress the generator, while fails to take the discriminator into account. In this work, we revisit the role of discriminator in GAN compression and design a novel generator-discriminator cooperative compression scheme for GAN compression, termed GCC. Within GCC, a selective activation discriminator automatically selects and activates convolutional channels according to a local capacity constraint and a global coordination constraint, which help maintain the Nash equilibrium with the lightweight generator during the adversarial training and avoid mode collapse. The original generator and discriminator are also optimized from scratch, to play as a teacher model to progressively refine the pruned generator and the selective activation discriminator. A novel online collaborative distillation scheme is designed to take full advantage of the intermediate feature of the teacher generator and discriminator to further boost the performance of the lightweight generator. Extensive experiments on various GAN-based generation tasks demonstrate the effectiveness and generalization of GCC. Among them, GCC contributes to reducing 80% computational costs while maintains comparable performance in image translation tasks.

        ----

        ## [2188] Asymptotically Exact Error Characterization of Offline Policy Evaluation with Misspecified Linear Models

        **Authors**: *Kohei Miyaguchi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0282b5ff85e7c9c66200d780bd7e72e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0282b5ff85e7c9c66200d780bd7e72e-Abstract.html)

        **Abstract**:

        We consider the problem of offline policy evaluation~(OPE) with Markov decision processes~(MDPs), where the goal is to estimate the utility of given decision-making policies based on static datasets. Recently, theoretical understanding of OPE has been rapidly advanced under (approximate) realizability assumptions, i.e., where the environments of interest are well approximated with the given hypothetical models. On the other hand, the OPE under unrealizability has not been well understood as much as in the realizable setting despite its importance in real-world applications.To address this issue, we study the behavior of a simple existing OPE method called the linear direct method~(DM) under the unrealizability. Consequently, we obtain an asymptotically exact characterization of the OPE error in a doubly robust form. Leveraging this result, we also establish the nonparametric consistency of the tile-coding estimators under quite mild assumptions.

        ----

        ## [2189] Topographic VAEs learn Equivariant Capsules

        **Authors**: *T. Anderson Keller, Max Welling*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f03704cb51f02f80b09bffba15751691-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f03704cb51f02f80b09bffba15751691-Abstract.html)

        **Abstract**:

        In this work we seek to bridge the concepts of topographic organization and equivariance in neural networks. To accomplish this, we introduce the Topographic VAE: a novel method for efficiently training deep generative models with topographically organized latent variables. We show that such a model indeed learns to organize its activations according to salient characteristics such as digit class, width, and style on MNIST. Furthermore, through topographic organization over time (i.e. temporal coherence), we demonstrate how predefined latent space transformation operators can be encouraged for observed transformed input sequences -- a primitive form of unsupervised learned equivariance. We demonstrate that this model successfully learns sets of approximately equivariant features (i.e. "capsules") directly from sequences and achieves higher likelihood on correspondingly transforming test sequences. Equivariance is verified quantitatively by measuring the approximate commutativity of the inference network and the sequence transformations. Finally, we demonstrate approximate equivariance to complex transformations, expanding upon the capabilities of existing group equivariant neural networks.

        ----

        ## [2190] MobILE: Model-Based Imitation Learning From Observation Alone

        **Authors**: *Rahul Kidambi, Jonathan D. Chang, Wen Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f06048518ff8de2035363e00710c6a1d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f06048518ff8de2035363e00710c6a1d-Abstract.html)

        **Abstract**:

        This paper studies Imitation Learning from Observations alone (ILFO) where the learner is presented with expert demonstrations that consist only of states visited by an expert (without access to actions taken by the expert). We present a provably efficient model-based framework MobILE to solve the ILFO problem. MobILE involves carefully trading off exploration against imitation - this is achieved by integrating the idea of optimism in the face of uncertainty into the distribution matching imitation learning (IL) framework. We provide a unified analysis for MobILE, and demonstrate that MobILE enjoys strong performance guarantees for classes of MDP dynamics that satisfy certain well studied notions of complexity. We also show that the ILFO problem is strictly harder than the standard IL problem by reducing ILFO to a multi-armed bandit problem indicating that exploration is necessary for solving ILFO efficiently. We complement  these theoretical results with experimental simulations on benchmark OpenAI Gym tasks that indicate the efficacy of MobILE. Code for implementing the MobILE framework is available at https://github.com/rahulkidambi/MobILE-NeurIPS2021.

        ----

        ## [2191] Few-Round Learning for Federated Learning

        **Authors**: *Younghyun Park, Dong-Jun Han, Do-Yeon Kim, Jun Seo, Jaekyun Moon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f065d878ccfb4cc4f4265a4ff8bafa9a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f065d878ccfb4cc4f4265a4ff8bafa9a-Abstract.html)

        **Abstract**:

        In federated learning (FL), a number of distributed clients targeting the same task collaborate to train a single global model without sharing their data. The learning process typically starts from a randomly initialized or some pretrained model. In this paper, we aim at designing an initial model based on which an arbitrary group of clients can obtain a global model for its own purpose, within only a few rounds of FL. The key challenge here is that the downstream tasks for which the pretrained model will be used are generally unknown when the initial model is prepared. Our idea is to take a meta-learning approach to construct the initial model so that any group with a possibly unseen task can obtain a high-accuracy global model within only R rounds of FL. Our meta-learning itself could be done via federated learning among willing participants and is based on an episodic arrangement to mimic the R rounds of FL followed by inference in each episode. Extensive experimental results show that our method generalizes well for arbitrary groups of clients and provides large performance improvements given the same overall communication/computation resources, compared to other baselines relying on known pretraining methods.

        ----

        ## [2192] On Path Integration of Grid Cells: Group Representation and Isotropic Scaling

        **Authors**: *Ruiqi Gao, Jianwen Xie, Xue-Xin Wei, Song-Chun Zhu, Ying Nian Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f076073b2082f8741a9cd07b789c77a0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f076073b2082f8741a9cd07b789c77a0-Abstract.html)

        **Abstract**:

        Understanding how grid cells perform path integration calculations remains a fundamental problem. In this paper, we conduct theoretical analysis of a general representation model of path integration by grid cells, where the 2D self-position is encoded as a higher dimensional vector, and the 2D self-motion is represented by a general transformation of the vector. We identify two conditions on the transformation. One is a group representation condition that is necessary for path integration. The other is an isotropic scaling condition that ensures locally conformal embedding, so that the error in the vector representation translates conformally to the error in the 2D self-position. Then we investigate the simplest transformation, i.e., the linear transformation, uncover its explicit algebraic and geometric structure as matrix Lie group of rotation, and explore the connection between the isotropic scaling condition and a special class of hexagon grid patterns. Finally, with our optimization-based approach, we manage to learn hexagon grid patterns that share similar properties of the grid cells in the rodent brain. The learned model is capable of accurate long distance path integration. Code is available at https://github.com/ruiqigao/grid-cell-path.

        ----

        ## [2193] Online Convex Optimization with Continuous Switching Constraint

        **Authors**: *Guanghui Wang, Yuanyu Wan, Tianbao Yang, Lijun Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f08b7ac8aa30a2a9ab34394e200e1a71-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f08b7ac8aa30a2a9ab34394e200e1a71-Abstract.html)

        **Abstract**:

        In many sequential decision making applications, the change of decision would bring an additional cost, such as the wear-and-tear cost associated with changing server status. To control the switching cost, we introduce the problem of online convex optimization with continuous switching constraint, where the goal is to achieve a small regret given a budget on the \emph{overall} switching cost. We first investigate the hardness of the problem, and provide a lower bound of order $\Omega(\sqrt{T})$ when the switching cost budget $S=\Omega(\sqrt{T})$, and $\Omega(\min\{\frac{T}{S},T\})$ when $S=O(\sqrt{T})$, where $T$ is the time horizon. The essential idea is to carefully design an adaptive adversary, who can adjust the loss function according to the cumulative switching cost of the player incurred so far based on the orthogonal technique. We then develop a simple gradient-based algorithm which enjoys the minimax optimal regret bound. Finally, we show that, for strongly convex functions, the regret bound can be improved to $O(\log T)$ for $S=\Omega(\log T)$, and $O(\min\{T/\exp(S)+S,T\})$ for $S=O(\log T)$.

        ----

        ## [2194] Why Do Better Loss Functions Lead to Less Transferable Features?

        **Authors**: *Simon Kornblith, Ting Chen, Honglak Lee, Mohammad Norouzi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0bf4a2da952528910047c31b6c2e951-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0bf4a2da952528910047c31b6c2e951-Abstract.html)

        **Abstract**:

        Previous work has proposed many new loss functions and regularizers that improve test accuracy on image classification tasks. However, it is not clear whether these loss functions learn better representations for downstream tasks. This paper studies how the choice of training objective affects the transferability of the hidden representations of convolutional neural networks trained on ImageNet. We show that many objectives lead to statistically significant improvements in ImageNet accuracy over vanilla softmax cross-entropy, but the resulting fixed feature extractors transfer substantially worse to downstream tasks, and the choice of loss has little effect when networks are fully fine-tuned on the new tasks. Using centered kernel alignment to measure similarity between hidden representations of networks, we find that differences among loss functions are apparent only in the last few layers of the network. We delve deeper into representations of the penultimate layer, finding that different objectives and hyperparameter combinations lead to dramatically different levels of class separation. Representations with higher class separation obtain higher accuracy on the original task, but their features are less useful for downstream tasks. Our results suggest there exists a trade-off between learning invariant features for the original task and features relevant for transfer tasks.

        ----

        ## [2195] Breaking the centralized barrier for cross-device federated learning

        **Authors**: *Sai Praneeth Karimireddy, Martin Jaggi, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, Ananda Theertha Suresh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0e6be4ce76ccfa73c5a540d992d0756-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0e6be4ce76ccfa73c5a540d992d0756-Abstract.html)

        **Abstract**:

        Federated learning (FL) is a challenging setting for optimization due to the heterogeneity of the data across different clients which gives rise to the client drift phenomenon. In fact, obtaining an algorithm for FL which is uniformly better than simple centralized training has been a major open problem thus far. In this work, we propose a general algorithmic framework, Mime, which i) mitigates client drift and ii) adapts arbitrary centralized optimization algorithms such as momentum and Adam to the cross-device federated learning setting. Mime uses a combination of control-variates and server-level statistics (e.g. momentum) at every client-update step to ensure that each local update mimics that of the centralized method run on iid data. We prove a reduction result showing that Mime can translate the convergence of a generic algorithm in the centralized setting into convergence in the federated setting. Further, we show that when combined with momentum based variance reduction, Mime is provably faster than any centralized method--the first such result. We also perform a thorough experimental exploration of Mime's performance on real world datasets.

        ----

        ## [2196] Adversarially robust learning for security-constrained optimal power flow

        **Authors**: *Priya L. Donti, Aayushya Agarwal, Neeraj Vijay Bedmutha, Larry T. Pileggi, J. Zico Kolter*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0f07e680de407b0f12abf15bd520097-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0f07e680de407b0f12abf15bd520097-Abstract.html)

        **Abstract**:

        In recent years, the ML community has seen surges of interest in both adversarially robust learning and implicit layers, but connections between these two areas have seldom been explored. In this work, we combine innovations from these areas to tackle the problem of N-k security-constrained optimal power flow (SCOPF). N-k SCOPF is a core problem for the operation of electrical grids, and aims to schedule power generation in a manner that is robust to potentially $k$ simultaneous equipment outages. Inspired by methods in adversarially robust training, we frame N-k SCOPF as a minimax optimization problem -- viewing power generation settings as adjustable parameters and equipment outages as (adversarial) attacks -- and solve this problem via gradient-based techniques. The loss function of this minimax problem involves resolving implicit equations representing grid physics and operational decisions, which we differentiate through via the implicit function theorem. We demonstrate the efficacy of our framework in solving N-3 SCOPF, which has traditionally been considered as prohibitively expensive to solve given that the problem size depends combinatorially on the number of potential outages.

        ----

        ## [2197] Learning a Single Neuron with Bias Using Gradient Descent

        **Authors**: *Gal Vardi, Gilad Yehudai, Ohad Shamir*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0f6cc51dacebe556699ccb45e2d43a8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0f6cc51dacebe556699ccb45e2d43a8-Abstract.html)

        **Abstract**:

        We theoretically study the fundamental problem of learning a single neuron with a bias term ($\mathbf{x}\mapsto \sigma(\langle\mathbf{w},\mathbf{x}\rangle + b)$) in the realizable setting with the ReLU activation, using gradient descent. Perhaps surprisingly, we show that this is a significantly different and more challenging problem than the bias-less case (which was the focus of previous works on single neurons), both in terms of the optimization geometry as well as the ability of gradient methods to succeed in some scenarios. We provide a detailed study of this problem, characterizing the critical points of the objective, demonstrating failure cases, and providing positive convergence guarantees under different sets of assumptions. To prove our results, we develop some tools which may be of independent interest, and improve previous results on learning single neurons.

        ----

        ## [2198] Making a (Counterfactual) Difference One Rationale at a Time

        **Authors**: *Mitchell Plyler, Michael Green, Min Chi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0f800c92d191d736c4411f3b3f8ef4a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0f800c92d191d736c4411f3b3f8ef4a-Abstract.html)

        **Abstract**:

        Rationales, snippets of extracted text that explain an inference, have emerged as a popular framework for interpretable natural language processing (NLP). Rationale models typically consist of two cooperating modules: a selector and a classifier with the goal of maximizing the mutual information (MMI) between the "selected" text and the document label. Despite their promises, MMI-based methods often pick up on spurious text patterns and result in models with nonsensical behaviors. In this work, we investigate whether counterfactual data augmentation (CDA), without human assistance, can improve the performance of the selector by lowering the mutual information between spurious signals and the document label. Our counterfactuals are produced in an unsupervised fashion using class-dependent generative models. From an information theoretic lens, we derive properties of the unaugmented dataset for which our CDA approach would succeed. The effectiveness of CDA is empirically evaluated by comparing against several baselines including an improved MMI-based rationale schema on two multi-aspect datasets. Our results show that CDA produces rationales that better capture the signal of interest.

        ----

        ## [2199] 3D Siamese Voxel-to-BEV Tracker for Sparse Point Clouds

        **Authors**: *Le Hui, Lingpeng Wang, Mingmei Cheng, Jin Xie, Jian Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/f0fcf351df4eb6786e9bb6fc4e2dee02-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f0fcf351df4eb6786e9bb6fc4e2dee02-Abstract.html)

        **Abstract**:

        3D object tracking in point clouds is still a challenging problem due to the sparsity of LiDAR points in dynamic environments. In this work, we propose a Siamese voxel-to-BEV tracker, which can significantly improve the tracking performance in sparse 3D point clouds. Specifically, it consists of a Siamese shape-aware feature learning network and a voxel-to-BEV target localization network. The Siamese shape-aware feature learning network can capture 3D shape information of the object to learn the discriminative features of the object so that the potential target from the background in sparse point clouds can be identified. To this end, we first perform template feature embedding to embed the template's feature into the potential target and then generate a dense 3D shape to characterize the shape information of the potential target. For localizing the tracked target, the voxel-to-BEV target localization network regresses the target's 2D center and the z-axis center from the dense bird's eye view (BEV) feature map in an anchor-free manner. Concretely, we compress the voxelized point cloud along z-axis through max pooling to obtain a dense BEV feature map, where the regression of the 2D center and the z-axis center can be performed more effectively. Extensive evaluation on the KITTI tracking dataset shows that our method significantly outperforms the current state-of-the-art methods by a large margin. Code is available at https://github.com/fpthink/V2B.

        ----

        

[Go to the previous page](NIPS-2021-list10.md)

[Go to the next page](NIPS-2021-list12.md)

[Go to the catalog section](README.md)