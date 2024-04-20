## [400] Generalizable Episodic Memory for Deep Reinforcement Learning

**Authors**: *Hao Hu, Jianing Ye, Guangxiang Zhu, Zhizhou Ren, Chongjie Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hu21d.html](http://proceedings.mlr.press/v139/hu21d.html)

**Abstract**:

Episodic memory-based methods can rapidly latch onto past successful strategies by a non-parametric memory and improve sample efficiency of traditional reinforcement learning. However, little effort is put into the continuous domain, where a state is never visited twice, and previous episodic methods fail to efficiently aggregate experience across trajectories. To address this problem, we propose Generalizable Episodic Memory (GEM), which effectively organizes the state-action values of episodic memory in a generalizable manner and supports implicit planning on memorized trajectories. GEM utilizes a double estimator to reduce the overestimation bias induced by value propagation in the planning process. Empirical evaluation shows that our method significantly outperforms existing trajectory-based methods on various MuJoCo continuous control tasks. To further show the general applicability, we evaluate our method on Atari games with discrete action space, which also shows a significant improvement over baseline algorithms.

----

## [401] A Scalable Deterministic Global Optimization Algorithm for Clustering Problems

**Authors**: *Kaixun Hua, Mingfei Shi, Yankai Cao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hua21a.html](http://proceedings.mlr.press/v139/hua21a.html)

**Abstract**:

The minimum sum-of-squares clustering (MSSC) task, which can be treated as a Mixed Integer Second Order Cone Programming (MISOCP) problem, is rarely investigated in the literature through deterministic optimization to find its global optimal value. In this paper, we modelled the MSSC task as a two-stage optimization problem and proposed a tailed reduced-space branch and bound (BB) algorithm. We designed several approaches to construct lower and upper bounds at each node in the BB scheme, including a scenario grouping based Lagrangian decomposition approach. One key advantage of this reduced-space algorithm is that it only needs to perform branching on the centers of clusters to guarantee convergence, and the size of centers is independent of the number of data samples. Moreover, the lower bounds can be computed by solving small-scale sample subproblems, and upper bounds can be obtained trivially. These two properties enable our algorithm easy to be paralleled and can be scalable to the dataset with up to 200,000 samples for finding a global $\epsilon$-optimal solution of the MSSC task. We performed numerical experiments on both synthetic and real-world datasets and compared our proposed algorithms with the off-the-shelf global optimal solvers and classical local optimal algorithms. The results reveal a strong performance and scalability of our algorithm.

----

## [402] On Recovering from Modeling Errors Using Testing Bayesian Networks

**Authors**: *Haiying Huang, Adnan Darwiche*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huang21a.html](http://proceedings.mlr.press/v139/huang21a.html)

**Abstract**:

We consider the problem of supervised learning with Bayesian Networks when the used dependency structure is incomplete due to missing edges or missing variable states. These modeling errors induce independence constraints on the learned model that may not hold in the true, data-generating distribution. We provide a unified treatment of these modeling errors as instances of state-space abstractions. We then identify a class of Bayesian Networks and queries which allow one to fully recover from such modeling errors if one can choose Conditional Probability Tables (CPTs) dynamically based on evidence. We show theoretically that the recently proposed Testing Bayesian Networks (TBNs), which can be trained by compiling them into Testing Arithmetic Circuits (TACs), provide a promising construct for emulating this CPT selection mechanism. Finally, we present empirical results that illustrate the promise of TBNs as a tool for recovering from certain modeling errors in the context of supervised learning.

----

## [403] A Novel Sequential Coreset Method for Gradient Descent Algorithms

**Authors**: *Jiawei Huang, Ruomin Huang, Wenjie Liu, Nikolaos M. Freris, Hu Ding*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huang21b.html](http://proceedings.mlr.press/v139/huang21b.html)

**Abstract**:

A wide range of optimization problems arising in machine learning can be solved by gradient descent algorithms, and a central question in this area is how to efficiently compress a large-scale dataset so as to reduce the computational complexity. Coreset is a popular data compression technique that has been extensively studied before. However, most of existing coreset methods are problem-dependent and cannot be used as a general tool for a broader range of applications. A key obstacle is that they often rely on the pseudo-dimension and total sensitivity bound that can be very high or hard to obtain. In this paper, based on the “locality” property of gradient descent algorithms, we propose a new framework, termed “sequential coreset”, which effectively avoids these obstacles. Moreover, our method is particularly suitable for sparse optimization whence the coreset size can be further reduced to be only poly-logarithmically dependent on the dimension. In practice, the experimental results suggest that our method can save a large amount of running time compared with the baseline algorithms.

----

## [404] FL-NTK: A Neural Tangent Kernel-based Framework for Federated Learning Analysis

**Authors**: *Baihe Huang, Xiaoxiao Li, Zhao Song, Xin Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huang21c.html](http://proceedings.mlr.press/v139/huang21c.html)

**Abstract**:

Federated Learning (FL) is an emerging learning scheme that allows different distributed clients to train deep neural networks together without data sharing. Neural networks have become popular due to their unprecedented success. To the best of our knowledge, the theoretical guarantees of FL concerning neural networks with explicit forms and multi-step updates are unexplored. Nevertheless, training analysis of neural networks in FL is non-trivial for two reasons: first, the objective loss function we are optimizing is non-smooth and non-convex, and second, we are even not updating in the gradient direction. Existing convergence results for gradient descent-based methods heavily rely on the fact that the gradient direction is used for updating. The current paper presents a new class of convergence analysis for FL, Federated Neural Tangent Kernel (FL-NTK), which corresponds to overparamterized ReLU neural networks trained by gradient descent in FL and is inspired by the analysis in Neural Tangent Kernel (NTK). Theoretically, FL-NTK converges to a global-optimal solution at a linear rate with properly tuned learning parameters. Furthermore, with proper distributional assumptions, FL-NTK can also achieve good generalization. The proposed theoretical analysis scheme can be generalized to more complex neural networks.

----

## [405] STRODE: Stochastic Boundary Ordinary Differential Equation

**Authors**: *Hengguan Huang, Hongfu Liu, Hao Wang, Chang Xiao, Ye Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huang21d.html](http://proceedings.mlr.press/v139/huang21d.html)

**Abstract**:

Perception of time from sequentially acquired sensory inputs is rooted in everyday behaviors of individual organisms. Yet, most algorithms for time-series modeling fail to learn dynamics of random event timings directly from visual or audio inputs, requiring timing annotations during training that are usually unavailable for real-world applications. For instance, neuroscience perspectives on postdiction imply that there exist variable temporal ranges within which the incoming sensory inputs can affect the earlier perception, but such temporal ranges are mostly unannotated for real applications such as automatic speech recognition (ASR). In this paper, we present a probabilistic ordinary differential equation (ODE), called STochastic boundaRy ODE (STRODE), that learns both the timings and the dynamics of time series data without requiring any timing annotations during training. STRODE allows the usage of differential equations to sample from the posterior point processes, efficiently and analytically. We further provide theoretical guarantees on the learning of STRODE. Our empirical results show that our approach successfully infers event timings of time series data. Our method achieves competitive or superior performances compared to existing state-of-the-art methods for both synthetic and real-world datasets.

----

## [406] A Riemannian Block Coordinate Descent Method for Computing the Projection Robust Wasserstein Distance

**Authors**: *Minhui Huang, Shiqian Ma, Lifeng Lai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huang21e.html](http://proceedings.mlr.press/v139/huang21e.html)

**Abstract**:

The Wasserstein distance has become increasingly important in machine learning and deep learning. Despite its popularity, the Wasserstein distance is hard to approximate because of the curse of dimensionality. A recently proposed approach to alleviate the curse of dimensionality is to project the sampled data from the high dimensional probability distribution onto a lower-dimensional subspace, and then compute the Wasserstein distance between the projected data. However, this approach requires to solve a max-min problem over the Stiefel manifold, which is very challenging in practice. In this paper, we propose a Riemannian block coordinate descent (RBCD) method to solve this problem, which is based on a novel reformulation of the regularized max-min problem over the Stiefel manifold. We show that the complexity of arithmetic operations for RBCD to obtain an $\epsilon$-stationary point is $O(\epsilon^{-3})$, which is significantly better than the complexity of existing methods. Numerical results on both synthetic and real datasets demonstrate that our method is more efficient than existing methods, especially when the number of sampled data is very large.

----

## [407] Projection Robust Wasserstein Barycenters

**Authors**: *Minhui Huang, Shiqian Ma, Lifeng Lai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huang21f.html](http://proceedings.mlr.press/v139/huang21f.html)

**Abstract**:

Collecting and aggregating information from several probability measures or histograms is a fundamental task in machine learning. One of the popular solution methods for this task is to compute the barycenter of the probability measures under the Wasserstein metric. However, approximating the Wasserstein barycenter is numerically challenging because of the curse of dimensionality. This paper proposes the projection robust Wasserstein barycenter (PRWB) that has the potential to mitigate the curse of dimensionality, and a relaxed PRWB (RPRWB) model that is computationally more tractable. By combining the iterative Bregman projection algorithm and Riemannian optimization, we propose two algorithms for computing the RPRWB, which is a max-min problem over the Stiefel manifold. The complexity of arithmetic operations of the proposed algorithms for obtaining an $\epsilon$-stationary solution is analyzed. We incorporate the RPRWB into a discrete distribution clustering algorithm, and the numerical results on real text datasets confirm that our RPRWB model helps improve the clustering performance significantly.

----

## [408] Accurate Post Training Quantization With Small Calibration Sets

**Authors**: *Itay Hubara, Yury Nahshan, Yair Hanani, Ron Banner, Daniel Soudry*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hubara21a.html](http://proceedings.mlr.press/v139/hubara21a.html)

**Abstract**:

Lately, post-training quantization methods have gained considerable attention, as they are simple to use, and require only a small unlabeled calibration set. This small dataset cannot be used to fine-tune the model without significant over-fitting. Instead, these methods only use the calibration set to set the activations’ dynamic ranges. However, such methods always resulted in significant accuracy degradation, when used below 8-bits (except on small datasets). Here we aim to break the 8-bit barrier. To this end, we minimize the quantization errors of each layer or block separately by optimizing its parameters over the calibration set. We empirically demonstrate that this approach is: (1) much less susceptible to over-fitting than the standard fine-tuning approaches, and can be used even on a very small calibration set; and (2) more powerful than previous methods, which only set the activations’ dynamic ranges. We suggest two flavors for our method, parallel and sequential aim for a fixed and flexible bit-width allocation. For the latter, we demonstrate how to optimally allocate the bit-widths for each layer, while constraining accuracy degradation or model compression by proposing a novel integer programming formulation. Finally, we suggest model global statistics tuning, to correct biases introduced during quantization. Together, these methods yield state-of-the-art results for both vision and text models. For instance, on ResNet50, we obtain less than 1% accuracy degradation — with 4-bit weights and activations in all layers, but first and last. The suggested methods are two orders of magnitude faster than the traditional Quantize Aware Training approach used for lower than 8-bit quantization. We open-sourced our code \textit{https://github.com/papers-submission/CalibTIP}.

----

## [409] Learning and Planning in Complex Action Spaces

**Authors**: *Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, David Silver*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hubert21a.html](http://proceedings.mlr.press/v139/hubert21a.html)

**Abstract**:

Many important real-world problems have action spaces that are high-dimensional, continuous or both, making full enumeration of all possible actions infeasible. Instead, only small subsets of actions can be sampled for the purpose of policy evaluation and improvement. In this paper, we propose a general framework to reason in a principled way about policy evaluation and improvement over such sampled action subsets. This sample-based policy iteration framework can in principle be applied to any reinforcement learning algorithm based upon policy iteration. Concretely, we propose Sampled MuZero, an extension of the MuZero algorithm that is able to learn in domains with arbitrarily complex action spaces by planning over sampled actions. We demonstrate this approach on the classical board game of Go and on two continuous control benchmark domains: DeepMind Control Suite and Real-World RL Suite.

----

## [410] Generative Adversarial Transformers

**Authors**: *Drew A. Hudson, Larry Zitnick*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hudson21a.html](http://proceedings.mlr.press/v139/hudson21a.html)

**Abstract**:

We introduce the GANsformer, a novel and efficient type of transformer, and explore it for the task of visual generative modeling. The network employs a bipartite structure that enables long-range interactions across the image, while maintaining computation of linear efficiency, that can readily scale to high-resolution synthesis. It iteratively propagates information from a set of latent variables to the evolving visual features and vice versa, to support the refinement of each in light of the other and encourage the emergence of compositional representations of objects and scenes. In contrast to the classic transformer architecture, it utilizes multiplicative integration that allows flexible region-based modulation, and can thus be seen as a generalization of the successful StyleGAN network. We demonstrate the model’s strength and robustness through a careful evaluation over a range of datasets, from simulated multi-object environments to rich real-world indoor and outdoor scenes, showing it achieves state-of-the-art results in terms of image quality and diversity, while enjoying fast learning and better data-efficiency. Further qualitative and quantitative experiments offer us an insight into the model’s inner workings, revealing improved interpretability and stronger disentanglement, and illustrating the benefits and efficacy of our approach. An implementation of the model is available at https://github.com/dorarad/gansformer.

----

## [411] Neural Pharmacodynamic State Space Modeling

**Authors**: *Zeshan M. Hussain, Rahul G. Krishnan, David A. Sontag*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hussain21a.html](http://proceedings.mlr.press/v139/hussain21a.html)

**Abstract**:

Modeling the time-series of high-dimensional, longitudinal data is important for predicting patient disease progression. However, existing neural network based approaches that learn representations of patient state, while very flexible, are susceptible to overfitting. We propose a deep generative model that makes use of a novel attention-based neural architecture inspired by the physics of how treatments affect disease state. The result is a scalable and accurate model of high-dimensional patient biomarkers as they vary over time. Our proposed model yields significant improvements in generalization and, on real-world clinical data, provides interpretable insights into the dynamics of cancer progression.

----

## [412] Hyperparameter Selection for Imitation Learning

**Authors**: *Léonard Hussenot, Marcin Andrychowicz, Damien Vincent, Robert Dadashi, Anton Raichuk, Sabela Ramos, Nikola Momchev, Sertan Girgin, Raphaël Marinier, Lukasz Stafiniak, Manu Orsini, Olivier Bachem, Matthieu Geist, Olivier Pietquin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hussenot21a.html](http://proceedings.mlr.press/v139/hussenot21a.html)

**Abstract**:

We address the issue of tuning hyperparameters (HPs) for imitation learning algorithms in the context of continuous-control, when the underlying reward function of the demonstrating expert cannot be observed at any time. The vast literature in imitation learning mostly considers this reward function to be available for HP selection, but this is not a realistic setting. Indeed, would this reward function be available, it could then directly be used for policy training and imitation would not be necessary. To tackle this mostly ignored problem, we propose a number of possible proxies to the external reward. We evaluate them in an extensive empirical study (more than 10’000 agents across 9 environments) and make practical recommendations for selecting HPs. Our results show that while imitation learning algorithms are sensitive to HP choices, it is often possible to select good enough HPs through a proxy to the reward function.

----

## [413] Pareto GAN: Extending the Representational Power of GANs to Heavy-Tailed Distributions

**Authors**: *Todd Huster, Jeremy E. J. Cohen, Zinan Lin, Kevin Chan, Charles A. Kamhoua, Nandi O. Leslie, Cho-Yu Jason Chiang, Vyas Sekar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/huster21a.html](http://proceedings.mlr.press/v139/huster21a.html)

**Abstract**:

Generative adversarial networks (GANs) are often billed as "universal distribution learners", but precisely what distributions they can represent and learn is still an open question. Heavy-tailed distributions are prevalent in many different domains such as financial risk-assessment, physics, and epidemiology. We observe that existing GAN architectures do a poor job of matching the asymptotic behavior of heavy-tailed distributions, a problem that we show stems from their construction. Additionally, common loss functions produce unstable or near-zero gradients when faced with the infinite moments and large distances between outlier points characteristic of heavy-tailed distributions. We address these problems with the Pareto GAN. A Pareto GAN leverages extreme value theory and the functional properties of neural networks to learn a distribution that matches the asymptotic behavior of the marginal distributions of the features. We identify issues with standard loss functions and propose the use of alternative metric spaces that enable stable and efficient learning. Finally, we evaluate our proposed approach on a variety of heavy-tailed datasets.

----

## [414] LieTransformer: Equivariant Self-Attention for Lie Groups

**Authors**: *Michael J. Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, Hyunjik Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/hutchinson21a.html](http://proceedings.mlr.press/v139/hutchinson21a.html)

**Abstract**:

Group equivariant neural networks are used as building blocks of group invariant neural networks, which have been shown to improve generalisation performance and data efficiency through principled parameter sharing. Such works have mostly focused on group equivariant convolutions, building on the result that group equivariant linear maps are necessarily convolutions. In this work, we extend the scope of the literature to self-attention, that is emerging as a prominent building block of deep learning models. We propose the LieTransformer, an architecture composed of LieSelfAttention layers that are equivariant to arbitrary Lie groups and their discrete subgroups. We demonstrate the generality of our approach by showing experimental results that are competitive to baseline methods on a wide range of tasks: shape counting on point clouds, molecular property regression and modelling particle trajectories under Hamiltonian dynamics.

----

## [415] Crowdsourcing via Annotator Co-occurrence Imputation and Provable Symmetric Nonnegative Matrix Factorization

**Authors**: *Shahana Ibrahim, Xiao Fu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ibrahim21a.html](http://proceedings.mlr.press/v139/ibrahim21a.html)

**Abstract**:

Unsupervised learning of the Dawid-Skene (D&S) model from noisy, incomplete and crowdsourced annotations has been a long-standing challenge, and is a critical step towards reliably labeling massive data. A recent work takes a coupled nonnegative matrix factorization (CNMF) perspective, and shows appealing features: It ensures the identifiability of the D&S model and enjoys low sample complexity, as only the estimates of the co-occurrences of annotator labels are involved. However, the identifiability holds only when certain somewhat restrictive conditions are met in the context of crowdsourcing. Optimizing the CNMF criterion is also costly—and convergence assurances are elusive. This work recasts the pairwise co-occurrence based D&S model learning problem as a symmetric NMF (SymNMF) problem—which offers enhanced identifiability relative to CNMF. In practice, the SymNMF model is often (largely) incomplete, due to the lack of co-labeled items by some annotators. Two lightweight algorithms are proposed for co-occurrence imputation. Then, a low-complexity shifted rectified linear unit (ReLU)-empowered SymNMF algorithm is proposed to identify the D&S model. Various performance characterizations (e.g., missing co-occurrence recoverability, stability, and convergence) and evaluations are also presented.

----

## [416] Selecting Data Augmentation for Simulating Interventions

**Authors**: *Maximilian Ilse, Jakub M. Tomczak, Patrick Forré*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ilse21a.html](http://proceedings.mlr.press/v139/ilse21a.html)

**Abstract**:

Machine learning models trained with purely observational data and the principle of empirical risk minimization (Vapnik 1992) can fail to generalize to unseen domains. In this paper, we focus on the case where the problem arises through spurious correlation between the observed domains and the actual task labels. We find that many domain generalization methods do not explicitly take this spurious correlation into account. Instead, especially in more application-oriented research areas like medical imaging or robotics, data augmentation techniques that are based on heuristics are used to learn domain invariant features. To bridge the gap between theory and practice, we develop a causal perspective on the problem of domain generalization. We argue that causal concepts can be used to explain the success of data augmentation by describing how they can weaken the spurious correlation between the observed domains and the task labels. We demonstrate that data augmentation can serve as a tool for simulating interventional data. We use these theoretical insights to derive a simple algorithm that is able to select data augmentation techniques that will lead to better domain generalization.

----

## [417] Scalable Marginal Likelihood Estimation for Model Selection in Deep Learning

**Authors**: *Alexander Immer, Matthias Bauer, Vincent Fortuin, Gunnar Rätsch, Mohammad Emtiyaz Khan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/immer21a.html](http://proceedings.mlr.press/v139/immer21a.html)

**Abstract**:

Marginal-likelihood based model-selection, even though promising, is rarely used in deep learning due to estimation difficulties. Instead, most approaches rely on validation data, which may not be readily available. In this work, we present a scalable marginal-likelihood estimation method to select both hyperparameters and network architectures, based on the training data alone. Some hyperparameters can be estimated online during training, simplifying the procedure. Our marginal-likelihood estimate is based on Laplace’s method and Gauss-Newton approximations to the Hessian, and it outperforms cross-validation and manual tuning on standard regression and image classification datasets, especially in terms of calibration and out-of-distribution detection. Our work shows that marginal likelihoods can improve generalization and be useful when validation data is unavailable (e.g., in nonstationary settings).

----

## [418] Active Learning for Distributionally Robust Level-Set Estimation

**Authors**: *Yu Inatsu, Shogo Iwazaki, Ichiro Takeuchi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/inatsu21a.html](http://proceedings.mlr.press/v139/inatsu21a.html)

**Abstract**:

Many cases exist in which a black-box function $f$ with high evaluation cost depends on two types of variables $\bm x$ and $\bm w$, where $\bm x$ is a controllable \emph{design} variable and $\bm w$ are uncontrollable \emph{environmental} variables that have random variation following a certain distribution $P$. In such cases, an important task is to find the range of design variables $\bm x$ such that the function $f(\bm x, \bm w)$ has the desired properties by incorporating the random variation of the environmental variables $\bm w$. A natural measure of robustness is the probability that $f(\bm x, \bm w)$ exceeds a given threshold $h$, which is known as the \emph{probability threshold robustness} (PTR) measure in the literature on robust optimization. However, this robustness measure cannot be correctly evaluated when the distribution $P$ is unknown. In this study, we addressed this problem by considering the \textit{distributionally robust PTR} (DRPTR) measure, which considers the worst-case PTR within given candidate distributions. Specifically, we studied the problem of efficiently identifying a reliable set $H$, which is defined as a region in which the DRPTR measure exceeds a certain desired probability $\alpha$, which can be interpreted as a level set estimation (LSE) problem for DRPTR. We propose a theoretically grounded and computationally efficient active learning method for this problem. We show that the proposed method has theoretical guarantees on convergence and accuracy, and confirmed through numerical experiments that the proposed method outperforms existing methods.

----

## [419] Learning Randomly Perturbed Structured Predictors for Direct Loss Minimization

**Authors**: *Hedda Cohen Indelman, Tamir Hazan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/indelman21a.html](http://proceedings.mlr.press/v139/indelman21a.html)

**Abstract**:

Direct loss minimization is a popular approach for learning predictors over structured label spaces. This approach is computationally appealing as it replaces integration with optimization and allows to propagate gradients in a deep net using loss-perturbed prediction. Recently, this technique was extended to generative models, by introducing a randomized predictor that samples a structure from a randomly perturbed score function. In this work, we interpolate between these techniques by learning the variance of randomized structured predictors as well as their mean, in order to balance between the learned score function and the randomized noise. We demonstrate empirically the effectiveness of learning this balance in structured discrete spaces.

----

## [420] Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning

**Authors**: *Shariq Iqbal, Christian A. Schröder de Witt, Bei Peng, Wendelin Boehmer, Shimon Whiteson, Fei Sha*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/iqbal21a.html](http://proceedings.mlr.press/v139/iqbal21a.html)

**Abstract**:

Multi-agent settings in the real world often involve tasks with varying types and quantities of agents and non-agent entities; however, common patterns of behavior often emerge among these agents/entities. Our method aims to leverage these commonalities by asking the question: “What is the expected utility of each agent when only considering a randomly selected sub-group of its observed entities?” By posing this counterfactual question, we can recognize state-action trajectories within sub-groups of entities that we may have encountered in another task and use what we learned in that task to inform our prediction in the current one. We then reconstruct a prediction of the full returns as a combination of factors considering these disjoint groups of entities and train this “randomly factorized" value function as an auxiliary objective for value-based multi-agent reinforcement learning. By doing so, our model can recognize and leverage similarities across tasks to improve learning efficiency in a multi-task setting. Our approach, Randomized Entity-wise Factorization for Imagined Learning (REFIL), outperforms all strong baselines by a significant margin in challenging multi-task StarCraft micromanagement settings.

----

## [421] Randomized Exploration in Reinforcement Learning with General Value Function Approximation

**Authors**: *Haque Ishfaq, Qiwen Cui, Viet Nguyen, Alex Ayoub, Zhuoran Yang, Zhaoran Wang, Doina Precup, Lin Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ishfaq21a.html](http://proceedings.mlr.press/v139/ishfaq21a.html)

**Abstract**:

We propose a model-free reinforcement learning algorithm inspired by the popular randomized least squares value iteration (RLSVI) algorithm as well as the optimism principle. Unlike existing upper-confidence-bound (UCB) based approaches, which are often computationally intractable, our algorithm drives exploration by simply perturbing the training data with judiciously chosen i.i.d. scalar noises. To attain optimistic value function estimation without resorting to a UCB-style bonus, we introduce an optimistic reward sampling procedure. When the value functions can be represented by a function class $\mathcal{F}$, our algorithm achieves a worst-case regret bound of $\tilde{O}(\mathrm{poly}(d_EH)\sqrt{T})$ where $T$ is the time elapsed, $H$ is the planning horizon and $d_E$ is the \emph{eluder dimension} of $\mathcal{F}$. In the linear setting, our algorithm reduces to LSVI-PHE, a variant of RLSVI, that enjoys an $\tilde{\mathcal{O}}(\sqrt{d^3H^3T})$ regret. We complement the theory with an empirical evaluation across known difficult exploration tasks.

----

## [422] Distributed Second Order Methods with Fast Rates and Compressed Communication

**Authors**: *Rustem Islamov, Xun Qian, Peter Richtárik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/islamov21a.html](http://proceedings.mlr.press/v139/islamov21a.html)

**Abstract**:

We develop several new communication-efficient second-order methods for distributed optimization. Our first method, NEWTON-STAR, is a variant of Newton’s method from which it inherits its fast local quadratic rate. However, unlike Newton’s method, NEWTON-STAR enjoys the same per iteration communication cost as gradient descent. While this method is impractical as it relies on the use of certain unknown parameters characterizing the Hessian of the objective function at the optimum, it serves as the starting point which enables us to design practical variants thereof with strong theoretical guarantees. In particular, we design a stochastic sparsification strategy for learning the unknown parameters in an iterative fashion in a communication efficient manner. Applying this strategy to NEWTON-STAR leads to our next method, NEWTON-LEARN, for which we prove local linear and superlinear rates independent of the condition number. When applicable, this method can have dramatically superior convergence behavior when compared to state-of-the-art methods. Finally, we develop a globalization strategy using cubic regularization which leads to our next method, CUBIC-NEWTON-LEARN, for which we prove global sublinear and linear convergence rates, and a fast superlinear rate. Our results are supported with experimental results on real datasets, and show several orders of magnitude improvement on baseline and state-of-the-art methods in terms of communication complexity.

----

## [423] What Are Bayesian Neural Network Posteriors Really Like?

**Authors**: *Pavel Izmailov, Sharad Vikram, Matthew D. Hoffman, Andrew Gordon Wilson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/izmailov21a.html](http://proceedings.mlr.press/v139/izmailov21a.html)

**Abstract**:

The posterior over Bayesian neural network (BNN) parameters is extremely high-dimensional and non-convex. For computational reasons, researchers approximate this posterior using inexpensive mini-batch methods such as mean-field variational inference or stochastic-gradient Markov chain Monte Carlo (SGMCMC). To investigate foundational questions in Bayesian deep learning, we instead use full batch Hamiltonian Monte Carlo (HMC) on modern architectures. We show that (1) BNNs can achieve significant performance gains over standard training and deep ensembles; (2) a single long HMC chain can provide a comparable representation of the posterior to multiple shorter chains; (3) in contrast to recent studies, we find posterior tempering is not needed for near-optimal performance, with little evidence for a “cold posterior” effect, which we show is largely an artifact of data augmentation; (4) BMA performance is robust to the choice of prior scale, and relatively similar for diagonal Gaussian, mixture of Gaussian, and logistic priors; (5) Bayesian neural networks show surprisingly poor generalization under domain shift; (6) while cheaper alternatives such as deep ensembles and SGMCMC can provide good generalization, their predictive distributions are distinct from HMC. Notably, deep ensemble predictive distributions are similarly close to HMC as standard SGLD, and closer than standard variational inference.

----

## [424] How to Learn when Data Reacts to Your Model: Performative Gradient Descent

**Authors**: *Zachary Izzo, Lexing Ying, James Zou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/izzo21a.html](http://proceedings.mlr.press/v139/izzo21a.html)

**Abstract**:

Performative distribution shift captures the setting where the choice of which ML model is deployed changes the data distribution. For example, a bank which uses the number of open credit lines to determine a customer’s risk of default on a loan may induce customers to open more credit lines in order to improve their chances of being approved. Because of the interactions between the model and data distribution, finding the optimal model parameters is challenging. Works in this area have focused on finding stable points, which can be far from optimal. Here we introduce \emph{performative gradient descent} (PerfGD), an algorithm for computing performatively optimal points. Under regularity assumptions on the performative loss, PerfGD is the first algorithm which provably converges to an optimal point. PerfGD explicitly captures how changes in the model affects the data distribution and is simple to use. We support our findings with theory and experiments.

----

## [425] Perceiver: General Perception with Iterative Attention

**Authors**: *Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, João Carreira*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jaegle21a.html](http://proceedings.mlr.press/v139/jaegle21a.html)

**Abstract**:

Biological systems understand the world by simultaneously processing high-dimensional inputs from modalities as diverse as vision, audition, touch, proprioception, etc. The perception models used in deep learning on the other hand are designed for individual modalities, often relying on domain-specific assumptions such as the local grid structures exploited by virtually all existing vision models. These priors introduce helpful inductive biases, but also lock models to individual modalities. In this paper we introduce the Perceiver {–} a model that builds upon Transformers and hence makes few architectural assumptions about the relationship between its inputs, but that also scales to hundreds of thousands of inputs, like ConvNets. The model leverages an asymmetric attention mechanism to iteratively distill inputs into a tight latent bottleneck, allowing it to scale to handle very large inputs. We show that this architecture is competitive with or outperforms strong, specialized models on classification tasks across various modalities: images, point clouds, audio, video and video+audio. The Perceiver obtains performance comparable to ResNet-50 and ViT on ImageNet without 2D convolutions by directly attending to 50,000 pixels. It is also competitive in all modalities in AudioSet.

----

## [426] Imitation by Predicting Observations

**Authors**: *Andrew Jaegle, Yury Sulsky, Arun Ahuja, Jake Bruce, Rob Fergus, Greg Wayne*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jaegle21b.html](http://proceedings.mlr.press/v139/jaegle21b.html)

**Abstract**:

Imitation learning enables agents to reuse and adapt the hard-won expertise of others, offering a solution to several key challenges in learning behavior. Although it is easy to observe behavior in the real-world, the underlying actions may not be accessible. We present a new method for imitation solely from observations that achieves comparable performance to experts on challenging continuous control tasks while also exhibiting robustness in the presence of observations unrelated to the task. Our method, which we call FORM (for "Future Observation Reward Model") is derived from an inverse RL objective and imitates using a model of expert behavior learned by generative modelling of the expert’s observations, without needing ground truth actions. We show that FORM performs comparably to a strong baseline IRL method (GAIL) on the DeepMind Control Suite benchmark, while outperforming GAIL in the presence of task-irrelevant features.

----

## [427] Local Correlation Clustering with Asymmetric Classification Errors

**Authors**: *Jafar Jafarov, Sanchit Kalhan, Konstantin Makarychev, Yury Makarychev*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jafarov21a.html](http://proceedings.mlr.press/v139/jafarov21a.html)

**Abstract**:

In the Correlation Clustering problem, we are given a complete weighted graph $G$ with its edges labeled as “similar" and “dissimilar" by a noisy binary classifier. For a clustering $\mathcal{C}$ of graph $G$, a similar edge is in disagreement with $\mathcal{C}$, if its endpoints belong to distinct clusters; and a dissimilar edge is in disagreement with $\mathcal{C}$ if its endpoints belong to the same cluster. The disagreements vector, $\disagree$, is a vector indexed by the vertices of $G$ such that the $v$-th coordinate $\disagree_v$ equals the weight of all disagreeing edges incident on $v$. The goal is to produce a clustering that minimizes the $\ell_p$ norm of the disagreements vector for $p\geq 1$. We study the $\ell_p$ objective in Correlation Clustering under the following assumption: Every similar edge has weight in $[\alpha\mathbf{w},\mathbf{w}]$ and every dissimilar edge has weight at least $\alpha\mathbf{w}$ (where $\alpha \leq 1$ and $\mathbf{w}>0$ is a scaling parameter). We give an $O\left((\nicefrac{1}{\alpha})^{\nicefrac{1}{2}-\nicefrac{1}{2p}}\cdot \log\nicefrac{1}{\alpha}\right)$ approximation algorithm for this problem. Furthermore, we show an almost matching convex programming integrality gap.

----

## [428] Alternative Microfoundations for Strategic Classification

**Authors**: *Meena Jagadeesan, Celestine Mendler-Dünner, Moritz Hardt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jagadeesan21a.html](http://proceedings.mlr.press/v139/jagadeesan21a.html)

**Abstract**:

When reasoning about strategic behavior in a machine learning context it is tempting to combine standard microfoundations of rational agents with the statistical decision theory underlying classification. In this work, we argue that a direct combination of these ingredients leads to brittle solution concepts of limited descriptive and prescriptive value. First, we show that rational agents with perfect information produce discontinuities in the aggregate response to a decision rule that we often do not observe empirically. Second, when any positive fraction of agents is not perfectly strategic, desirable stable points—where the classifier is optimal for the data it entails—no longer exist. Third, optimal decision rules under standard microfoundations maximize a measure of negative externality known as social burden within a broad class of assumptions about agent behavior. Recognizing these limitations we explore alternatives to standard microfoundations for binary classification. We describe desiderata that help navigate the space of possible assumptions about agent responses, and we then propose the noisy response model. Inspired by smoothed analysis and empirical observations, noisy response incorporates imperfection in the agent responses, which we show mitigates the limitations of standard microfoundations. Our model retains analytical tractability, leads to more robust insights about stable points, and imposes a lower social burden at optimality.

----

## [429] Robust Density Estimation from Batches: The Best Things in Life are (Nearly) Free

**Authors**: *Ayush Jain, Alon Orlitsky*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jain21a.html](http://proceedings.mlr.press/v139/jain21a.html)

**Abstract**:

In many applications data are collected in batches, some potentially biased, corrupt, or even adversarial. Learning algorithms for this setting have therefore garnered considerable recent attention. In particular, a sequence of works has shown that all approximately piecewise polynomial distributions—and in particular all Gaussian, Gaussian-mixture, log-concave, low-modal, and monotone-hazard distributions—can be learned robustly in polynomial time. However, these results left open the question, stated explicitly in \cite{chen2020learning}, about the best possible sample complexity of such algorithms. We answer this question, showing that, perhaps surprisingly, up to logarithmic factors, the optimal sample complexity is the same as for genuine, non-adversarial, data! To establish the result, we reduce robust learning of approximately piecewise polynomial distributions to robust learning of the probability of all subsets of size at most $k$ of a larger discrete domain, and learn these probabilities in optimal sample complexity linear in $k$ regardless of the domain size. In simulations, the algorithm runs very quickly and estimates distributions to essentially the accuracy achieved when all adversarial batches are removed. The results also imply the first polynomial-time sample-optimal algorithm for robust interval-based classification based on batched data.

----

## [430] Instance-Optimal Compressed Sensing via Posterior Sampling

**Authors**: *Ajil Jalal, Sushrut Karmalkar, Alex Dimakis, Eric Price*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jalal21a.html](http://proceedings.mlr.press/v139/jalal21a.html)

**Abstract**:

We characterize the measurement complexity of compressed sensing of signals drawn from a known prior distribution, even when the support of the prior is the entire space (rather than, say, sparse vectors). We show for Gaussian measurements and \emph{any} prior distribution on the signal, that the posterior sampling estimator achieves near-optimal recovery guarantees. Moreover, this result is robust to model mismatch, as long as the distribution estimate (e.g., from an invertible generative model) is close to the true distribution in Wasserstein distance. We implement the posterior sampling estimator for deep generative priors using Langevin dynamics, and empirically find that it produces accurate estimates with more diversity than MAP.

----

## [431] Fairness for Image Generation with Uncertain Sensitive Attributes

**Authors**: *Ajil Jalal, Sushrut Karmalkar, Jessica Hoffmann, Alex Dimakis, Eric Price*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jalal21b.html](http://proceedings.mlr.press/v139/jalal21b.html)

**Abstract**:

This work tackles the issue of fairness in the context of generative procedures, such as image super-resolution, which entail different definitions from the standard classification setting. Moreover, while traditional group fairness definitions are typically defined with respect to specified protected groups – camouflaging the fact that these groupings are artificial and carry historical and political motivations – we emphasize that there are no ground truth identities. For instance, should South and East Asians be viewed as a single group or separate groups? Should we consider one race as a whole or further split by gender? Choosing which groups are valid and who belongs in them is an impossible dilemma and being “fair” with respect to Asians may require being “unfair” with respect to South Asians. This motivates the introduction of definitions that allow algorithms to be \emph{oblivious} to the relevant groupings. We define several intuitive notions of group fairness and study their incompatibilities and trade-offs. We show that the natural extension of demographic parity is strongly dependent on the grouping, and \emph{impossible} to achieve obliviously. On the other hand, the conceptually new definition we introduce, Conditional Proportional Representation, can be achieved obliviously through Posterior Sampling. Our experiments validate our theoretical results and achieve fair image reconstruction using state-of-the-art generative models.

----

## [432] Feature Clustering for Support Identification in Extreme Regions

**Authors**: *Hamid Jalalzai, Rémi Leluc*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jalalzai21a.html](http://proceedings.mlr.press/v139/jalalzai21a.html)

**Abstract**:

Understanding the complex structure of multivariate extremes is a major challenge in various fields from portfolio monitoring and environmental risk management to insurance. In the framework of multivariate Extreme Value Theory, a common characterization of extremes’ dependence structure is the angular measure. It is a suitable measure to work in extreme regions as it provides meaningful insights concerning the subregions where extremes tend to concentrate their mass. The present paper develops a novel optimization-based approach to assess the dependence structure of extremes. This support identification scheme rewrites as estimating clusters of features which best capture the support of extremes. The dimension reduction technique we provide is applied to statistical learning tasks such as feature clustering and anomaly detection. Numerical experiments provide strong empirical evidence of the relevance of our approach.

----

## [433] Improved Regret Bounds of Bilinear Bandits using Action Space Analysis

**Authors**: *Kyoungseok Jang, Kwang-Sung Jun, Se-Young Yun, Wanmo Kang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jang21a.html](http://proceedings.mlr.press/v139/jang21a.html)

**Abstract**:

We consider the bilinear bandit problem where the learner chooses a pair of arms, each from two different action spaces of dimension $d_1$ and $d_2$, respectively. The learner then receives a reward whose expectation is a bilinear function of the two chosen arms with an unknown matrix parameter $\Theta^*\in\mathbb{R}^{d_1 \times d_2}$ with rank $r$. Despite abundant applications such as drug discovery, the optimal regret rate is unknown for this problem, though it was conjectured to be $\tilde O(\sqrt{d_1d_2(d_1+d_2)r T})$ by Jun et al. (2019) where $\tilde O$ ignores polylogarithmic factors in $T$. In this paper, we make progress towards closing the gap between the upper and lower bound on the optimal regret. First, we reject the conjecture above by proposing algorithms that achieve the regret $\tilde O(\sqrt{d_1 d_2 (d_1+d_2) T})$ using the fact that the action space dimension $O(d_1+d_2)$ is significantly lower than the matrix parameter dimension $O(d_1 d_2)$. Second, we additionally devise an algorithm with better empirical performance than previous algorithms.

----

## [434] Inverse Decision Modeling: Learning Interpretable Representations of Behavior

**Authors**: *Daniel Jarrett, Alihan Hüyük, Mihaela van der Schaar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jarrett21a.html](http://proceedings.mlr.press/v139/jarrett21a.html)

**Abstract**:

Decision analysis deals with modeling and enhancing decision processes. A principal challenge in improving behavior is in obtaining a transparent *description* of existing behavior in the first place. In this paper, we develop an expressive, unifying perspective on *inverse decision modeling*: a framework for learning parameterized representations of sequential decision behavior. First, we formalize the *forward* problem (as a normative standard), subsuming common classes of control behavior. Second, we use this to formalize the *inverse* problem (as a descriptive model), generalizing existing work on imitation/reward learning—while opening up a much broader class of research problems in behavior representation. Finally, we instantiate this approach with an example (*inverse bounded rational control*), illustrating how this structure enables learning (interpretable) representations of (bounded) rationality—while naturally capturing intuitive notions of suboptimal actions, biased beliefs, and imperfect knowledge of environments.

----

## [435] Catastrophic Fisher Explosion: Early Phase Fisher Matrix Impacts Generalization

**Authors**: *Stanislaw Jastrzebski, Devansh Arpit, Oliver Åstrand, Giancarlo Kerg, Huan Wang, Caiming Xiong, Richard Socher, Kyunghyun Cho, Krzysztof J. Geras*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jastrzebski21a.html](http://proceedings.mlr.press/v139/jastrzebski21a.html)

**Abstract**:

The early phase of training a deep neural network has a dramatic effect on the local curvature of the loss function. For instance, using a small learning rate does not guarantee stable optimization because the optimization trajectory has a tendency to steer towards regions of the loss surface with increasing local curvature. We ask whether this tendency is connected to the widely observed phenomenon that the choice of the learning rate strongly influences generalization. We first show that stochastic gradient descent (SGD) implicitly penalizes the trace of the Fisher Information Matrix (FIM), a measure of the local curvature, from the start of training. We argue it is an implicit regularizer in SGD by showing that explicitly penalizing the trace of the FIM can significantly improve generalization. We highlight that poor final generalization coincides with the trace of the FIM attaining a large value early in training, to which we refer as catastrophic Fisher explosion. Finally, to gain insight into the regularization effect of penalizing the trace of the FIM, we show that it limits memorization by reducing the learning speed of examples with noisy labels more than that of the examples with clean labels.

----

## [436] Policy Gradient Bayesian Robust Optimization for Imitation Learning

**Authors**: *Zaynah Javed, Daniel S. Brown, Satvik Sharma, Jerry Zhu, Ashwin Balakrishna, Marek Petrik, Anca D. Dragan, Ken Goldberg*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/javed21a.html](http://proceedings.mlr.press/v139/javed21a.html)

**Abstract**:

The difficulty in specifying rewards for many real-world problems has led to an increased focus on learning rewards from human feedback, such as demonstrations. However, there are often many different reward functions that explain the human feedback, leaving agents with uncertainty over what the true reward function is. While most policy optimization approaches handle this uncertainty by optimizing for expected performance, many applications demand risk-averse behavior. We derive a novel policy gradient-style robust optimization approach, PG-BROIL, that optimizes a soft-robust objective that balances expected performance and risk. To the best of our knowledge, PG-BROIL is the first policy optimization algorithm robust to a distribution of reward hypotheses which can scale to continuous MDPs. Results suggest that PG-BROIL can produce a family of behaviors ranging from risk-neutral to risk-averse and outperforms state-of-the-art imitation learning algorithms when learning from ambiguous demonstrations by hedging against uncertainty, rather than seeking to uniquely identify the demonstrator’s reward function.

----

## [437] In-Database Regression in Input Sparsity Time

**Authors**: *Rajesh Jayaram, Alireza Samadian, David P. Woodruff, Peng Ye*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jayaram21a.html](http://proceedings.mlr.press/v139/jayaram21a.html)

**Abstract**:

Sketching is a powerful dimensionality reduction technique for accelerating algorithms for data analysis. A crucial step in sketching methods is to compute a subspace embedding (SE) for a large matrix $A \in \mathbb{R}^{N \times d}$. SE’s are the primary tool for obtaining extremely efficient solutions for many linear-algebraic tasks, such as least squares regression and low rank approximation. Computing an SE often requires an explicit representation of $A$ and running time proportional to the size of $A$. However, if $A= T_1 \Join T_2 \Join …\Join T_m$ is the result of a database join query on several smaller tables $T_i \in \mathbb{R}^{n_i \times d_i}$, then this running time can be prohibitive, as $A$ itself can have as many as $O(n_1 n_2 \cdots n_m)$ rows. In this work, we design subspace embeddings for database joins which can be computed significantly faster than computing the join. For the case of a two table join $A = T_1 \Join T_2$ we give input-sparsity algorithms for computing subspace embeddings, with running time bounded by the number of non-zero entries in $T_1,T_2$. This results in input-sparsity time algorithms for high accuracy regression, significantly improving upon the running time of prior FAQ-based methods for regression. We extend our results to arbitrary joins for the ridge regression problem, also considerably improving the running time of prior methods. Empirically, we apply our method to real datasets and show that it is significantly faster than existing algorithms.

----

## [438] Parallel and Flexible Sampling from Autoregressive Models via Langevin Dynamics

**Authors**: *Vivek Jayaram, John Thickstun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jayaram21b.html](http://proceedings.mlr.press/v139/jayaram21b.html)

**Abstract**:

This paper introduces an alternative approach to sampling from autoregressive models. Autoregressive models are typically sampled sequentially, according to the transition dynamics defined by the model. Instead, we propose a sampling procedure that initializes a sequence with white noise and follows a Markov chain defined by Langevin dynamics on the global log-likelihood of the sequence. This approach parallelizes the sampling process and generalizes to conditional sampling. Using an autoregressive model as a Bayesian prior, we can steer the output of a generative model using a conditional likelihood or constraints. We apply these techniques to autoregressive models in the visual and audio domains, with competitive results for audio source separation, super-resolution, and inpainting.

----

## [439] Objective Bound Conditional Gaussian Process for Bayesian Optimization

**Authors**: *Taewon Jeong, Heeyoung Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jeong21a.html](http://proceedings.mlr.press/v139/jeong21a.html)

**Abstract**:

A Gaussian process is a standard surrogate model for an unknown objective function in Bayesian optimization. In this paper, we propose a new surrogate model, called the objective bound conditional Gaussian process (OBCGP), to condition a Gaussian process on a bound on the optimal function value. The bound is obtained and updated as the best observed value during the sequential optimization procedure. Unlike the standard Gaussian process, the OBCGP explicitly incorporates the existence of a point that improves the best known bound. We treat the location of such a point as a model parameter and estimate it jointly with other parameters by maximizing the likelihood using variational inference. Within the standard Bayesian optimization framework, the OBCGP can be combined with various acquisition functions to select the next query point. In particular, we derive cumulative regret bounds for the OBCGP combined with the upper confidence bound acquisition algorithm. Furthermore, the OBCGP can inherently incorporate a new type of prior knowledge, i.e., the bounds on the optimum, if it is available. The incorporation of this type of prior knowledge into a surrogate model has not been studied previously. We demonstrate the effectiveness of the OBCGP through its application to Bayesian optimization tasks, such as the sequential design of experiments and hyperparameter optimization in neural networks.

----

## [440] Quantifying Ignorance in Individual-Level Causal-Effect Estimates under Hidden Confounding

**Authors**: *Andrew Jesson, Sören Mindermann, Yarin Gal, Uri Shalit*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jesson21a.html](http://proceedings.mlr.press/v139/jesson21a.html)

**Abstract**:

We study the problem of learning conditional average treatment effects (CATE) from high-dimensional, observational data with unobserved confounders. Unobserved confounders introduce ignorance—a level of unidentifiability—about an individual’s response to treatment by inducing bias in CATE estimates. We present a new parametric interval estimator suited for high-dimensional data, that estimates a range of possible CATE values when given a predefined bound on the level of hidden confounding. Further, previous interval estimators do not account for ignorance about the CATE associated with samples that may be underrepresented in the original study, or samples that violate the overlap assumption. Our interval estimator also incorporates model uncertainty so that practitioners can be made aware of such out-of-distribution data. We prove that our estimator converges to tight bounds on CATE when there may be unobserved confounding and assess it using semi-synthetic, high-dimensional datasets.

----

## [441] DeepReDuce: ReLU Reduction for Fast Private Inference

**Authors**: *Nandan Kumar Jha, Zahra Ghodsi, Siddharth Garg, Brandon Reagen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jha21a.html](http://proceedings.mlr.press/v139/jha21a.html)

**Abstract**:

The recent rise of privacy concerns has led researchers to devise methods for private neural inference—where inferences are made directly on encrypted data, never seeing inputs. The primary challenge facing private inference is that computing on encrypted data levies an impractically-high latency penalty, stemming mostly from non-linear operators like ReLU. Enabling practical and private inference requires new optimization methods that minimize network ReLU counts while preserving accuracy. This paper proposes DeepReDuce: a set of optimizations for the judicious removal of ReLUs to reduce private inference latency. The key insight is that not all ReLUs contribute equally to accuracy. We leverage this insight to drop, or remove, ReLUs from classic networks to significantly reduce inference latency and maintain high accuracy. Given a network architecture, DeepReDuce outputs a Pareto frontier of networks that tradeoff the number of ReLUs and accuracy. Compared to the state-of-the-art for private inference DeepReDuce improves accuracy and reduces ReLU count by up to 3.5% (iso-ReLU count) and 3.5x (iso-accuracy), respectively.

----

## [442] Factor-analytic inverse regression for high-dimension, small-sample dimensionality reduction

**Authors**: *Aditi Jha, Michael J. Morais, Jonathan W. Pillow*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jha21b.html](http://proceedings.mlr.press/v139/jha21b.html)

**Abstract**:

Sufficient dimension reduction (SDR) methods are a family of supervised methods for dimensionality reduction that seek to reduce dimensionality while preserving information about a target variable of interest. However, existing SDR methods typically require more observations than the number of dimensions ($N > p$). To overcome this limitation, we propose Class-conditional Factor Analytic Dimensions (CFAD), a model-based dimensionality reduction method for high-dimensional, small-sample data. We show that CFAD substantially outperforms existing SDR methods in the small-sample regime, and can be extended to incorporate prior information such as smoothness in the projection axes. We demonstrate the effectiveness of CFAD with an application to functional magnetic resonance imaging (fMRI) measurements during visual object recognition and working memory tasks, where it outperforms existing SDR and a variety of other dimensionality-reduction methods.

----

## [443] Fast margin maximization via dual acceleration

**Authors**: *Ziwei Ji, Nathan Srebro, Matus Telgarsky*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ji21a.html](http://proceedings.mlr.press/v139/ji21a.html)

**Abstract**:

We present and analyze a momentum-based gradient method for training linear classifiers with an exponentially-tailed loss (e.g., the exponential or logistic loss), which maximizes the classification margin on separable data at a rate of O(1/t^2). This contrasts with a rate of O(1/log(t)) for standard gradient descent, and O(1/t) for normalized gradient descent. The momentum-based method is derived via the convex dual of the maximum-margin problem, and specifically by applying Nesterov acceleration to this dual, which manages to result in a simple and intuitive method in the primal. This dual view can also be used to derive a stochastic variant, which performs adaptive non-uniform sampling via the dual variables.

----

## [444] Marginalized Stochastic Natural Gradients for Black-Box Variational Inference

**Authors**: *Geng Ji, Debora Sujono, Erik B. Sudderth*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ji21b.html](http://proceedings.mlr.press/v139/ji21b.html)

**Abstract**:

Black-box variational inference algorithms use stochastic sampling to analyze diverse statistical models, like those expressed in probabilistic programming languages, without model-specific derivations. While the popular score-function estimator computes unbiased gradient estimates, its variance is often unacceptably large, especially in models with discrete latent variables. We propose a stochastic natural gradient estimator that is as broadly applicable and unbiased, but improves efficiency by exploiting the curvature of the variational bound, and provably reduces variance by marginalizing discrete latent variables. Our marginalized stochastic natural gradients have intriguing connections to classic coordinate ascent variational inference, but allow parallel updates of variational parameters, and provide superior convergence guarantees relative to naive Monte Carlo approximations. We integrate our method with the probabilistic programming language Pyro and evaluate real-world models of documents, images, networks, and crowd-sourcing. Compared to score-function estimators, we require far fewer Monte Carlo samples and consistently convergence orders of magnitude faster.

----

## [445] Bilevel Optimization: Convergence Analysis and Enhanced Design

**Authors**: *Kaiyi Ji, Junjie Yang, Yingbin Liang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ji21c.html](http://proceedings.mlr.press/v139/ji21c.html)

**Abstract**:

Bilevel optimization has arisen as a powerful tool for many machine learning problems such as meta-learning, hyperparameter optimization, and reinforcement learning. In this paper, we investigate the nonconvex-strongly-convex bilevel optimization problem. For deterministic bilevel optimization, we provide a comprehensive convergence rate analysis for two popular algorithms respectively based on approximate implicit differentiation (AID) and iterative differentiation (ITD). For the AID-based method, we orderwisely improve the previous convergence rate analysis due to a more practical parameter selection as well as a warm start strategy, and for the ITD-based method we establish the first theoretical convergence rate. Our analysis also provides a quantitative comparison between ITD and AID based approaches. For stochastic bilevel optimization, we propose a novel algorithm named stocBiO, which features a sample-efficient hypergradient estimator using efficient Jacobian- and Hessian-vector product computations. We provide the convergence rate guarantee for stocBiO, and show that stocBiO outperforms the best known computational complexities orderwisely with respect to the condition number $\kappa$ and the target accuracy $\epsilon$. We further validate our theoretical results and demonstrate the efficiency of bilevel optimization algorithms by the experiments on meta-learning and hyperparameter optimization.

----

## [446] Efficient Statistical Tests: A Neural Tangent Kernel Approach

**Authors**: *Sheng Jia, Ehsan Nezhadarya, Yuhuai Wu, Jimmy Ba*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jia21a.html](http://proceedings.mlr.press/v139/jia21a.html)

**Abstract**:

For machine learning models to make reliable predictions in deployment, one needs to ensure the previously unknown test samples need to be sufficiently similar to the training data. The commonly used shift-invariant kernels do not have the compositionality and fail to capture invariances in high-dimensional data in computer vision. We propose a shift-invariant convolutional neural tangent kernel (SCNTK) based outlier detector and two-sample tests with maximum mean discrepancy (MMD) that is O(n) in the number of samples due to using the random feature approximation. On MNIST and CIFAR10 with various types of dataset shifts, we empirically show that statistical tests with such compositional kernels, inherited from infinitely wide neural networks, achieve higher detection accuracy than existing non-parametric methods. Our method also provides a competitive alternative to adapted kernel methods that require a training phase.

----

## [447] Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision

**Authors**: *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yun-Hsuan Sung, Zhen Li, Tom Duerig*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jia21b.html](http://proceedings.mlr.press/v139/jia21b.html)

**Abstract**:

Pre-trained representations are becoming crucial for many NLP and perception tasks. While representation learning in NLP has transitioned to training on raw text without human annotations, visual and vision-language representations still rely heavily on curated training datasets that are expensive or require expert knowledge. For vision applications, representations are mostly learned using datasets with explicit class labels such as ImageNet or OpenImages. For vision-language, popular datasets like Conceptual Captions, MSCOCO, or CLIP all involve a non-trivial data collection (and cleaning) process. This costly curation process limits the size of datasets and hence hinders the scaling of trained models. In this paper, we leverage a noisy dataset of over one billion image alt-text pairs, obtained without expensive filtering or post-processing steps in the Conceptual Captions dataset. A simple dual-encoder architecture learns to align visual and language representations of the image and text pairs using a contrastive loss. We show that the scale of our corpus can make up for its noise and leads to state-of-the-art representations even with such a simple learning scheme. Our visual representation achieves strong performance when transferred to classification tasks such as ImageNet and VTAB. The aligned visual and language representations enables zero-shot image classification and also set new state-of-the-art results on Flickr30K and MSCOCO image-text retrieval benchmarks, even when compared with more sophisticated cross-attention models. The representations also enable cross-modality search with complex text and text + image queries.

----

## [448] Multi-Dimensional Classification via Sparse Label Encoding

**Authors**: *Bin-Bin Jia, Min-Ling Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jia21c.html](http://proceedings.mlr.press/v139/jia21c.html)

**Abstract**:

In multi-dimensional classification (MDC), there are multiple class variables in the output space with each of them corresponding to one heterogeneous class space. Due to the heterogeneity of class spaces, it is quite challenging to consider the dependencies among class variables when learning from MDC examples. In this paper, we propose a novel MDC approach named SLEM which learns the predictive model in an encoded label space instead of the original heterogeneous one. Specifically, SLEM works in an encoding-training-decoding framework. In the encoding phase, each class vector is mapped into a real-valued one via three cascaded operations including pairwise grouping, one-hot conversion and sparse linear encoding. In the training phase, a multi-output regression model is learned within the encoded label space. In the decoding phase, the predicted class vector is obtained by adapting orthogonal matching pursuit over outputs of the learned multi-output regression model. Experimental results clearly validate the superiority of SLEM against state-of-the-art MDC approaches.

----

## [449] Self-Damaging Contrastive Learning

**Authors**: *Ziyu Jiang, Tianlong Chen, Bobak J. Mortazavi, Zhangyang Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21a.html](http://proceedings.mlr.press/v139/jiang21a.html)

**Abstract**:

The recent breakthrough achieved by contrastive learning accelerates the pace for deploying unsupervised training on real-world data applications. However, unlabeled data in reality is commonly imbalanced and shows a long-tail distribution, and it is unclear how robustly the latest contrastive learning methods could perform in the practical scenario. This paper proposes to explicitly tackle this challenge, via a principled framework called Self-Damaging Contrastive Learning (SDCLR), to automatically balance the representation learning without knowing the classes. Our main inspiration is drawn from the recent finding that deep models have difficult-to-memorize samples, and those may be exposed through network pruning. It is further natural to hypothesize that long-tail samples are also tougher for the model to learn well due to insufficient examples. Hence, the key innovation in SDCLR is to create a dynamic self-competitor model to contrast with the target model, which is a pruned version of the latter. During training, contrasting the two models will lead to adaptive online mining of the most easily forgotten samples for the current target model, and implicitly emphasize them more in the contrastive loss. Extensive experiments across multiple datasets and imbalance settings show that SDCLR significantly improves not only overall accuracies but also balancedness, in terms of linear evaluation on the full-shot and few-shot settings. Our code is available at https://github.com/VITA-Group/SDCLR.

----

## [450] Prioritized Level Replay

**Authors**: *Minqi Jiang, Edward Grefenstette, Tim Rocktäschel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21b.html](http://proceedings.mlr.press/v139/jiang21b.html)

**Abstract**:

Environments with procedurally generated content serve as important benchmarks for testing systematic generalization in deep reinforcement learning. In this setting, each level is an algorithmically created environment instance with a unique configuration of its factors of variation. Training on a prespecified subset of levels allows for testing generalization to unseen levels. What can be learned from a level depends on the current policy, yet prior work defaults to uniform sampling of training levels independently of the policy. We introduce Prioritized Level Replay (PLR), a general framework for selectively sampling the next training level by prioritizing those with higher estimated learning potential when revisited in the future. We show TD-errors effectively estimate a level’s future learning potential and, when used to guide the sampling procedure, induce an emergent curriculum of increasingly difficult levels. By adapting the sampling of training levels, PLR significantly improves sample-efficiency and generalization on Procgen Benchmark—matching the previous state-of-the-art in test return—and readily combines with other methods. Combined with the previous leading method, PLR raises the state-of-the-art to over 76% improvement in test return relative to standard RL baselines.

----

## [451] Monotonic Robust Policy Optimization with Model Discrepancy

**Authors**: *Yuankun Jiang, Chenglin Li, Wenrui Dai, Junni Zou, Hongkai Xiong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21c.html](http://proceedings.mlr.press/v139/jiang21c.html)

**Abstract**:

State-of-the-art deep reinforcement learning (DRL) algorithms tend to overfit due to the model discrepancy between source and target environments. Though applying domain randomization during training can improve the average performance by randomly generating a sufficient diversity of environments in simulator, the worst-case environment is still neglected without any performance guarantee. Since the average and worst-case performance are both important for generalization in RL, in this paper, we propose a policy optimization approach for concurrently improving the policy’s performance in the average and worst-case environment. We theoretically derive a lower bound for the worst-case performance of a given policy by relating it to the expected performance. Guided by this lower bound, we formulate an optimization problem to jointly optimize the policy and sampling distribution, and prove that by iteratively solving it the worst-case performance is monotonically improved. We then develop a practical algorithm, named monotonic robust policy optimization (MRPO). Experimental evaluations in several robot control tasks demonstrate that MRPO can generally improve both the average and worst-case performance in the source environments for training, and facilitate in all cases the learned policy with a better generalization capability in some unseen testing environments.

----

## [452] Approximation Theory of Convolutional Architectures for Time Series Modelling

**Authors**: *Haotian Jiang, Zhong Li, Qianxiao Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21d.html](http://proceedings.mlr.press/v139/jiang21d.html)

**Abstract**:

We study the approximation properties of convolutional architectures applied to time series modelling, which can be formulated mathematically as a functional approximation problem. In the recurrent setting, recent results reveal an intricate connection between approximation efficiency and memory structures in the data generation process. In this paper, we derive parallel results for convolutional architectures, with WaveNet being a prime example. Our results reveal that in this new setting, approximation efficiency is not only characterised by memory, but also additional fine structures in the target relationship. This leads to a novel definition of spectrum-based regularity that measures the complexity of temporal relationships under the convolutional approximation scheme. These analyses provide a foundation to understand the differences between architectural choices for time series modelling and can give theoretically grounded guidance for practical applications.

----

## [453] Streaming and Distributed Algorithms for Robust Column Subset Selection

**Authors**: *Shuli Jiang, Dennis Li, Irene Mengze Li, Arvind V. Mahankali, David P. Woodruff*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21e.html](http://proceedings.mlr.press/v139/jiang21e.html)

**Abstract**:

We give the first single-pass streaming algorithm for Column Subset Selection with respect to the entrywise $\ell_p$-norm with $1 \leq p < 2$. We study the $\ell_p$ norm loss since it is often considered more robust to noise than the standard Frobenius norm. Given an input matrix $A \in \mathbb{R}^{d \times n}$ ($n \gg d$), our algorithm achieves a multiplicative $k^{\frac{1}{p} - \frac{1}{2}}\poly(\log nd)$-approximation to the error with respect to the \textit{best possible column subset} of size $k$. Furthermore, the space complexity of the streaming algorithm is optimal up to a logarithmic factor. Our streaming algorithm also extends naturally to a 1-round distributed protocol with nearly optimal communication cost. A key ingredient in our algorithms is a reduction to column subset selection in the $\ell_{p,2}$-norm, which corresponds to the $p$-norm of the vector of Euclidean norms of each of the columns of $A$. This enables us to leverage strong coreset constructions for the Euclidean norm, which previously had not been applied in this context. We also give the first provable guarantees for greedy column subset selection in the $\ell_{1, 2}$ norm, which can be used as an alternative, practical subroutine in our algorithms. Finally, we show that our algorithms give significant practical advantages on real-world data analysis tasks.

----

## [454] Single Pass Entrywise-Transformed Low Rank Approximation

**Authors**: *Yifei Jiang, Yi Li, Yiming Sun, Jiaxin Wang, David P. Woodruff*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21f.html](http://proceedings.mlr.press/v139/jiang21f.html)

**Abstract**:

In applications such as natural language processing or computer vision, one is given a large $n \times n$ matrix $A = (a_{i,j})$ and would like to compute a matrix decomposition, e.g., a low rank approximation, of a function $f(A) = (f(a_{i,j}))$ applied entrywise to $A$. A very important special case is the likelihood function $f\left( A \right ) = \log{\left( \left| a_{ij}\right| +1\right)}$. A natural way to do this would be to simply apply $f$ to each entry of $A$, and then compute the matrix decomposition, but this requires storing all of $A$ as well as multiple passes over its entries. Recent work of Liang et al. shows how to find a rank-$k$ factorization to $f(A)$ using only $n \cdot \poly(\eps^{-1}k\log n)$ words of memory, with overall error $10\|f(A)-[f(A)]_k\|_F^2 + \poly(\epsilon/k) \|f(A)\|_{1,2}^2$, where $[f(A)]_k$ is the best rank-$k$ approximation to $f(A)$ and $\|f(A)\|_{1,2}^2$ is the square of the sum of Euclidean lengths of rows of $f(A)$. Their algorithm uses $3$ passes over the entries of $A$. The authors pose the open question of obtaining an algorithm with $n \cdot \poly(\eps^{-1}k\log n)$ words of memory using only a single pass over the entries of $A$. In this paper we resolve this open question, obtaining the first single-pass algorithm for this problem and for the same class of functions $f$ studied by Liang et al. Moreover, our error is $\|f(A)-[f(A)]_k\|_F^2 + \poly(\epsilon/k) \|f(A)\|_F^2$, where $\|f(A)\|_F^2$ is the sum of squares of Euclidean lengths of rows of $f(A)$. Thus our error is significantly smaller, as it removes the factor of $10$ and also $\|f(A)\|_F^2 \leq \|f(A)\|_{1,2}^2$.

----

## [455] The Emergence of Individuality

**Authors**: *Jiechuan Jiang, Zongqing Lu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21g.html](http://proceedings.mlr.press/v139/jiang21g.html)

**Abstract**:

Individuality is essential in human society. It induces the division of labor and thus improves the efficiency and productivity. Similarly, it should also be a key to multi-agent cooperation. Inspired by that individuality is of being an individual separate from others, we propose a simple yet efficient method for the emergence of individuality (EOI) in multi-agent reinforcement learning (MARL). EOI learns a probabilistic classifier that predicts a probability distribution over agents given their observation and gives each agent an intrinsic reward of being correctly predicted by the classifier. The intrinsic reward encourages the agents to visit their own familiar observations, and learning the classifier by such observations makes the intrinsic reward signals stronger and in turn makes the agents more identifiable. To further enhance the intrinsic reward and promote the emergence of individuality, two regularizers are proposed to increase the discriminability of the classifier. We implement EOI on top of popular MARL algorithms. Empirically, we show that EOI outperforms existing methods in a variety of multi-agent cooperative scenarios.

----

## [456] Online Selection Problems against Constrained Adversary

**Authors**: *Zhihao Jiang, Pinyan Lu, Zhihao Gavin Tang, Yuhao Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21h.html](http://proceedings.mlr.press/v139/jiang21h.html)

**Abstract**:

Inspired by a recent line of work in online algorithms with predictions, we study the constrained adversary model that utilizes predictions from a different perspective. Prior works mostly focused on designing simultaneously robust and consistent algorithms, without making assumptions on the quality of the predictions. In contrary, our model assumes the adversarial instance is consistent with the predictions and aim to design algorithms that have best worst-case performance against all such instances. We revisit classical online selection problems under the constrained adversary model. For the single item selection problem, we design an optimal algorithm in the adversarial arrival model and an improved algorithm in the random arrival model (a.k.a., the secretary problem). For the online edge-weighted bipartite matching problem, we extend the classical Water-filling and Ranking algorithms and achieve improved competitive ratios.

----

## [457] Active Covering

**Authors**: *Heinrich Jiang, Afshin Rostamizadeh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21i.html](http://proceedings.mlr.press/v139/jiang21i.html)

**Abstract**:

We analyze the problem of active covering, where the learner is given an unlabeled dataset and can sequentially label query examples. The objective is to label query all of the positive examples in the fewest number of total label queries. We show under standard non-parametric assumptions that a classical support estimator can be repurposed as an offline algorithm attaining an excess query cost of $\widetilde{\Theta}(n^{D/(D+1)})$ compared to the optimal learner, where $n$ is the number of datapoints and $D$ is the dimension. We then provide a simple active learning method that attains an improved excess query cost of $\widetilde{O}(n^{(D-1)/D})$. Furthermore, the proposed algorithms only require access to the positive labeled examples, which in certain settings provides additional computational and privacy benefits. Finally, we show that the active learning method consistently outperforms offline methods as well as a variety of baselines on a wide range of benchmark image-based datasets.

----

## [458] Emphatic Algorithms for Deep Reinforcement Learning

**Authors**: *Ray Jiang, Tom Zahavy, Zhongwen Xu, Adam White, Matteo Hessel, Charles Blundell, Hado van Hasselt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21j.html](http://proceedings.mlr.press/v139/jiang21j.html)

**Abstract**:

Off-policy learning allows us to learn about possible policies of behavior from experience generated by a different behavior policy. Temporal difference (TD) learning algorithms can become unstable when combined with function approximation and off-policy sampling—this is known as the “deadly triad”. Emphatic temporal difference (ETD($\lambda$)) algorithm ensures convergence in the linear case by appropriately weighting the TD($\lambda$) updates. In this paper, we extend the use of emphatic methods to deep reinforcement learning agents. We show that naively adapting ETD($\lambda$) to popular deep reinforcement learning algorithms, which use forward view multi-step returns, results in poor performance. We then derive new emphatic algorithms for use in the context of such algorithms, and we demonstrate that they provide noticeable benefits in small problems designed to highlight the instability of TD methods. Finally, we observed improved performance when applying these algorithms at scale on classic Atari games from the Arcade Learning Environment.

----

## [459] Characterizing Structural Regularities of Labeled Data in Overparameterized Models

**Authors**: *Ziheng Jiang, Chiyuan Zhang, Kunal Talwar, Michael C. Mozer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jiang21k.html](http://proceedings.mlr.press/v139/jiang21k.html)

**Abstract**:

Humans are accustomed to environments that contain both regularities and exceptions. For example, at most gas stations, one pays prior to pumping, but the occasional rural station does not accept payment in advance. Likewise, deep neural networks can generalize across instances that share common patterns or structures, yet have the capacity to memorize rare or irregular forms. We analyze how individual instances are treated by a model via a consistency score. The score characterizes the expected accuracy for a held-out instance given training sets of varying size sampled from the data distribution. We obtain empirical estimates of this score for individual instances in multiple data sets, and we show that the score identifies out-of-distribution and mislabeled examples at one end of the continuum and strongly regular examples at the other end. We identify computationally inexpensive proxies to the consistency score using statistics collected during training. We apply the score toward understanding the dynamics of representation learning and to filter outliers during training.

----

## [460] Optimal Streaming Algorithms for Multi-Armed Bandits

**Authors**: *Tianyuan Jin, Keke Huang, Jing Tang, Xiaokui Xiao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jin21a.html](http://proceedings.mlr.press/v139/jin21a.html)

**Abstract**:

This paper studies two variants of the best arm identification (BAI) problem under the streaming model, where we have a stream of n arms with reward distributions supported on [0,1] with unknown means. The arms in the stream are arriving one by one, and the algorithm cannot access an arm unless it is stored in a limited size memory. We first study the streaming \epslion-topk-arms identification problem, which asks for k arms whose reward means are lower than that of the k-th best arm by at most \epsilon with probability at least 1-\delta. For general \epsilon \in (0,1), the existing solution for this problem assumes k = 1 and achieves the optimal sample complexity O(\frac{n}{\epsilon^2} \log \frac{1}{\delta}) using O(\log^*(n)) memory and a single pass of the stream. We propose an algorithm that works for any k and achieves the optimal sample complexity O(\frac{n}{\epsilon^2} \log\frac{k}{\delta}) using a single-arm memory and a single pass of the stream. Second, we study the streaming BAI problem, where the objective is to identify the arm with the maximum reward mean with at least 1-\delta probability, using a single-arm memory and as few passes of the input stream as possible. We present a single-arm-memory algorithm that achieves a near instance-dependent optimal sample complexity within O(\log \Delta_2^{-1}) passes, where \Delta_2 is the gap between the mean of the best arm and that of the second best arm.

----

## [461] Towards Tight Bounds on the Sample Complexity of Average-reward MDPs

**Authors**: *Yujia Jin, Aaron Sidford*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jin21b.html](http://proceedings.mlr.press/v139/jin21b.html)

**Abstract**:

We prove new upper and lower bounds for sample complexity of finding an $\epsilon$-optimal policy of an infinite-horizon average-reward Markov decision process (MDP) given access to a generative model. When the mixing time of the probability transition matrix of all policies is at most $t_\mathrm{mix}$, we provide an algorithm that solves the problem using $\widetilde{O}(t_\mathrm{mix} \epsilon^{-3})$ (oblivious) samples per state-action pair. Further, we provide a lower bound showing that a linear dependence on $t_\mathrm{mix}$ is necessary in the worst case for any algorithm which computes oblivious samples. We obtain our results by establishing connections between infinite-horizon average-reward MDPs and discounted MDPs of possible further utility.

----

## [462] Almost Optimal Anytime Algorithm for Batched Multi-Armed Bandits

**Authors**: *Tianyuan Jin, Jing Tang, Pan Xu, Keke Huang, Xiaokui Xiao, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jin21c.html](http://proceedings.mlr.press/v139/jin21c.html)

**Abstract**:

In batched multi-armed bandit problems, the learner can adaptively pull arms and adjust strategy in batches. In many real applications, not only the regret but also the batch complexity need to be optimized. Existing batched bandit algorithms usually assume that the time horizon T is known in advance. However, many applications involve an unpredictable stopping time. In this paper, we study the anytime batched multi-armed bandit problem. We propose an anytime algorithm that achieves the asymptotically optimal regret for exponential families of reward distributions with $O(\log \log T \ilog^{\alpha} (T))$ \footnote{Notation \ilog^{\alpha} (T) is the result of iteratively applying the logarithm function on T for \alpha times, e.g., \ilog^{3} (T)=\log\log\log T.} batches, where $\alpha\in O_{T}(1)$. Moreover, we prove that for any constant c>0, no algorithm can achieve the asymptotically optimal regret within c\log\log T batches.

----

## [463] MOTS: Minimax Optimal Thompson Sampling

**Authors**: *Tianyuan Jin, Pan Xu, Jieming Shi, Xiaokui Xiao, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jin21d.html](http://proceedings.mlr.press/v139/jin21d.html)

**Abstract**:

Thompson sampling is one of the most widely used algorithms in many online decision problems due to its simplicity for implementation and superior empirical performance over other state-of-the-art methods. Despite its popularity and empirical success, it has remained an open problem whether Thompson sampling can achieve the minimax optimal regret O(\sqrt{TK}) for K-armed bandit problems, where T is the total time horizon. In this paper we fill this long open gap by proposing a new Thompson sampling algorithm called MOTS that adaptively truncates the sampling result of the chosen arm at each time step. We prove that this simple variant of Thompson sampling achieves the minimax optimal regret bound O(\sqrt{TK}) for finite time horizon T and also the asymptotic optimal regret bound when $T$ grows to infinity as well. This is the first time that the minimax optimality of multi-armed bandit problems has been attained by Thompson sampling type of algorithms.

----

## [464] Is Pessimism Provably Efficient for Offline RL?

**Authors**: *Ying Jin, Zhuoran Yang, Zhaoran Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jin21e.html](http://proceedings.mlr.press/v139/jin21e.html)

**Abstract**:

We study offline reinforcement learning (RL), which aims to learn an optimal policy based on a dataset collected a priori. Due to the lack of further interactions with the environment, offline RL suffers from the insufficient coverage of the dataset, which eludes most existing theoretical analysis. In this paper, we propose a pessimistic variant of the value iteration algorithm (PEVI), which incorporates an uncertainty quantifier as the penalty function. Such a penalty function simply flips the sign of the bonus function for promoting exploration in online RL, which makes it easily implementable and compatible with general function approximators. Without assuming the sufficient coverage of the dataset, we establish a data-dependent upper bound on the suboptimality of PEVI for general Markov decision processes (MDPs). When specialized to linear MDPs, it matches the information-theoretic lower bound up to multiplicative factors of the dimension and horizon. In other words, pessimism is not only provably efficient but also minimax optimal. In particular, given the dataset, the learned policy serves as the “best effort” among all policies, as no other policies can do better. Our theoretical analysis identifies the critical role of pessimism in eliminating a notion of spurious correlation, which emerges from the “irrelevant” trajectories that are less covered by the dataset and not informative for the optimal policy.

----

## [465] Adversarial Option-Aware Hierarchical Imitation Learning

**Authors**: *Mingxuan Jing, Wenbing Huang, Fuchun Sun, Xiaojian Ma, Tao Kong, Chuang Gan, Lei Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jing21a.html](http://proceedings.mlr.press/v139/jing21a.html)

**Abstract**:

It has been a challenge to learning skills for an agent from long-horizon unannotated demonstrations. Existing approaches like Hierarchical Imitation Learning(HIL) are prone to compounding errors or suboptimal solutions. In this paper, we propose Option-GAIL, a novel method to learn skills at long horizon. The key idea of Option-GAIL is modeling the task hierarchy by options and train the policy via generative adversarial optimization. In particular, we propose an Expectation-Maximization(EM)-style algorithm: an E-step that samples the options of expert conditioned on the current learned policy, and an M-step that updates the low- and high-level policies of agent simultaneously to minimize the newly proposed option-occupancy measurement between the expert and the agent. We theoretically prove the convergence of the proposed algorithm. Experiments show that Option-GAIL outperforms other counterparts consistently across a variety of tasks.

----

## [466] Discrete-Valued Latent Preference Matrix Estimation with Graph Side Information

**Authors**: *Changhun Jo, Kangwook Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jo21a.html](http://proceedings.mlr.press/v139/jo21a.html)

**Abstract**:

Incorporating graph side information into recommender systems has been widely used to better predict ratings, but relatively few works have focused on theoretical guarantees. Ahn et al. (2018) firstly characterized the optimal sample complexity in the presence of graph side information, but the results are limited due to strict, unrealistic assumptions made on the unknown latent preference matrix and the structure of user clusters. In this work, we propose a new model in which 1) the unknown latent preference matrix can have any discrete values, and 2) users can be clustered into multiple clusters, thereby relaxing the assumptions made in prior work. Under this new model, we fully characterize the optimal sample complexity and develop a computationally-efficient algorithm that matches the optimal sample complexity. Our algorithm is robust to model errors and outperforms the existing algorithms in terms of prediction performance on both synthetic and real data.

----

## [467] Provable Lipschitz Certification for Generative Models

**Authors**: *Matt Jordan, Alex Dimakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jordan21a.html](http://proceedings.mlr.press/v139/jordan21a.html)

**Abstract**:

We present a scalable technique for upper bounding the Lipschitz constant of generative models. We relate this quantity to the maximal norm over the set of attainable vector-Jacobian products of a given generative model. We approximate this set by layerwise convex approximations using zonotopes. Our approach generalizes and improves upon prior work using zonotope transformers and we extend to Lipschitz estimation of neural networks with large output dimension. This provides efficient and tight bounds on small networks and can scale to generative models on VAE and DCGAN architectures.

----

## [468] Isometric Gaussian Process Latent Variable Model for Dissimilarity Data

**Authors**: *Martin Jørgensen, Søren Hauberg*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jorgensen21a.html](http://proceedings.mlr.press/v139/jorgensen21a.html)

**Abstract**:

We present a probabilistic model where the latent variable respects both the distances and the topology of the modeled data. The model leverages the Riemannian geometry of the generated manifold to endow the latent space with a well-defined stochastic distance measure, which is modeled locally as Nakagami distributions. These stochastic distances are sought to be as similar as possible to observed distances along a neighborhood graph through a censoring process. The model is inferred by variational inference based on observations of pairwise distances. We demonstrate how the new model can encode invariances in the learned manifolds.

----

## [469] On the Generalization Power of Overfitted Two-Layer Neural Tangent Kernel Models

**Authors**: *Peizhong Ju, Xiaojun Lin, Ness B. Shroff*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ju21a.html](http://proceedings.mlr.press/v139/ju21a.html)

**Abstract**:

In this paper, we study the generalization performance of min $\ell_2$-norm overfitting solutions for the neural tangent kernel (NTK) model of a two-layer neural network with ReLU activation that has no bias term. We show that, depending on the ground-truth function, the test error of overfitted NTK models exhibits characteristics that are different from the "double-descent" of other overparameterized linear models with simple Fourier or Gaussian features. Specifically, for a class of learnable functions, we provide a new upper bound of the generalization error that approaches a small limiting value, even when the number of neurons $p$ approaches infinity. This limiting value further decreases with the number of training samples $n$. For functions outside of this class, we provide a lower bound on the generalization error that does not diminish to zero even when $n$ and $p$ are both large.

----

## [470] Improved Confidence Bounds for the Linear Logistic Model and Applications to Bandits

**Authors**: *Kwang-Sung Jun, Lalit Jain, Houssam Nassif, Blake Mason*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jun21a.html](http://proceedings.mlr.press/v139/jun21a.html)

**Abstract**:

We propose improved fixed-design confidence bounds for the linear logistic model. Our bounds significantly improve upon the state-of-the-art bound by Li et al. (2017) via recent developments of the self-concordant analysis of the logistic loss (Faury et al., 2020). Specifically, our confidence bound avoids a direct dependence on $1/\kappa$, where $\kappa$ is the minimal variance over all arms’ reward distributions. In general, $1/\kappa$ scales exponentially with the norm of the unknown linear parameter $\theta^*$. Instead of relying on this worst case quantity, our confidence bound for the reward of any given arm depends directly on the variance of that arm’s reward distribution. We present two applications of our novel bounds to pure exploration and regret minimization logistic bandits improving upon state-of-the-art performance guarantees. For pure exploration we also provide a lower bound highlighting a dependence on $1/\kappa$ for a family of instances.

----

## [471] Detection of Signal in the Spiked Rectangular Models

**Authors**: *Ji Hyung Jung, Hye Won Chung, Ji Oon Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jung21a.html](http://proceedings.mlr.press/v139/jung21a.html)

**Abstract**:

We consider the problem of detecting signals in the rank-one signal-plus-noise data matrix models that generalize the spiked Wishart matrices. We show that the principal component analysis can be improved by pre-transforming the matrix entries if the noise is non-Gaussian. As an intermediate step, we prove a sharp phase transition of the largest eigenvalues of spiked rectangular matrices, which extends the Baik–Ben Arous–Péché (BBP) transition. We also propose a hypothesis test to detect the presence of signal with low computational complexity, based on the linear spectral statistics, which minimizes the sum of the Type-I and Type-II errors when the noise is Gaussian.

----

## [472] Estimating Identifiable Causal Effects on Markov Equivalence Class through Double Machine Learning

**Authors**: *Yonghan Jung, Jin Tian, Elias Bareinboim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/jung21b.html](http://proceedings.mlr.press/v139/jung21b.html)

**Abstract**:

General methods have been developed for estimating causal effects from observational data under causal assumptions encoded in the form of a causal graph. Most of this literature assumes that the underlying causal graph is completely specified. However, only observational data is available in most practical settings, which means that one can learn at most a Markov equivalence class (MEC) of the underlying causal graph. In this paper, we study the problem of causal estimation from a MEC represented by a partial ancestral graph (PAG), which is learnable from observational data. We develop a general estimator for any identifiable causal effects in a PAG. The result fills a gap for an end-to-end solution to causal inference from observational data to effects estimation. Specifically, we develop a complete identification algorithm that derives an influence function for any identifiable causal effects from PAGs. We then construct a double/debiased machine learning (DML) estimator that is robust to model misspecification and biases in nuisance function estimation, permitting the use of modern machine learning techniques. Simulation results corroborate with the theory.

----

## [473] A Nullspace Property for Subspace-Preserving Recovery

**Authors**: *Mustafa Devrim Kaba, Chong You, Daniel P. Robinson, Enrique Mallada, René Vidal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kaba21a.html](http://proceedings.mlr.press/v139/kaba21a.html)

**Abstract**:

Much of the theory for classical sparse recovery is based on conditions on the dictionary that are both necessary and sufficient (e.g., nullspace property) or only sufficient (e.g., incoherence and restricted isometry). In contrast, much of the theory for subspace-preserving recovery, the theoretical underpinnings for sparse subspace classification and clustering methods, is based on conditions on the subspaces and the data that are only sufficient (e.g., subspace incoherence and data inner-radius). This paper derives a necessary and sufficient condition for subspace-preserving recovery that is inspired by the classical nullspace property.Based on this novel condition, called here the subspace nullspace property, we derive equivalent characterizations that either admit a clear geometric interpretation that relates data distribution and subspace separation to the recovery success, or can be verified using a finite set of extreme points of a properly defined set. We further exploit these characterizations to derive new sufficient conditions, based on inner-radius and outer-radius measures and dual bounds, that generalize existing conditions and preserve the geometric interpretations. These results fill an important gap in the subspace-preserving recovery literature.

----

## [474] Training Recurrent Neural Networks via Forward Propagation Through Time

**Authors**: *Anil Kag, Venkatesh Saligrama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kag21a.html](http://proceedings.mlr.press/v139/kag21a.html)

**Abstract**:

Back-propagation through time (BPTT) has been widely used for training Recurrent Neural Networks (RNNs). BPTT updates RNN parameters on an instance by back-propagating the error in time over the entire sequence length, and as a result, leads to poor trainability due to the well-known gradient explosion/decay phenomena. While a number of prior works have proposed to mitigate vanishing/explosion effect through careful RNN architecture design, these RNN variants still train with BPTT. We propose a novel forward-propagation algorithm, FPTT, where at each time, for an instance, we update RNN parameters by optimizing an instantaneous risk function. Our proposed risk is a regularization penalty at time $t$ that evolves dynamically based on previously observed losses, and allows for RNN parameter updates to converge to a stationary solution of the empirical RNN objective. We consider both sequence-to-sequence as well as terminal loss problems. Empirically FPTT outperforms BPTT on a number of well-known benchmark tasks, thus enabling architectures like LSTMs to solve long range dependencies problems.

----

## [475] The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation

**Authors**: *Peter Kairouz, Ziyu Liu, Thomas Steinke*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kairouz21a.html](http://proceedings.mlr.press/v139/kairouz21a.html)

**Abstract**:

We consider training models on private data that are distributed across user devices. To ensure privacy, we add on-device noise and use secure aggregation so that only the noisy sum is revealed to the server. We present a comprehensive end-to-end system, which appropriately discretizes the data and adds discrete Gaussian noise before performing secure aggregation. We provide a novel privacy analysis for sums of discrete Gaussians and carefully analyze the effects of data quantization and modular summation arithmetic. Our theoretical guarantees highlight the complex tension between communication, privacy, and accuracy. Our extensive experimental results demonstrate that our solution is essentially able to match the accuracy to central differential privacy with less than 16 bits of precision per value.

----

## [476] Practical and Private (Deep) Learning Without Sampling or Shuffling

**Authors**: *Peter Kairouz, Brendan McMahan, Shuang Song, Om Thakkar, Abhradeep Thakurta, Zheng Xu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kairouz21b.html](http://proceedings.mlr.press/v139/kairouz21b.html)

**Abstract**:

We consider training models with differential privacy (DP) using mini-batch gradients. The existing state-of-the-art, Differentially Private Stochastic Gradient Descent (DP-SGD), requires \emph{privacy amplification by sampling or shuffling} to obtain the best privacy/accuracy/computation trade-offs. Unfortunately, the precise requirements on exact sampling and shuffling can be hard to obtain in important practical scenarios, particularly federated learning (FL). We design and analyze a DP variant of Follow-The-Regularized-Leader (DP-FTRL) that compares favorably (both theoretically and empirically) to amplified DP-SGD, while allowing for much more flexible data access patterns. DP-FTRL does not use any form of privacy amplification.

----

## [477] A Differentiable Point Process with Its Application to Spiking Neural Networks

**Authors**: *Hiroshi Kajino*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kajino21a.html](http://proceedings.mlr.press/v139/kajino21a.html)

**Abstract**:

This paper is concerned about a learning algorithm for a probabilistic model of spiking neural networks (SNNs). Jimenez Rezende & Gerstner (2014) proposed a stochastic variational inference algorithm to train SNNs with hidden neurons. The algorithm updates the variational distribution using the score function gradient estimator, whose high variance often impedes the whole learning algorithm. This paper presents an alternative gradient estimator for SNNs based on the path-wise gradient estimator. The main technical difficulty is a lack of a general method to differentiate a realization of an arbitrary point process, which is necessary to derive the path-wise gradient estimator. We develop a differentiable point process, which is the technical highlight of this paper, and apply it to derive the path-wise gradient estimator for SNNs. We investigate the effectiveness of our gradient estimator through numerical simulation.

----

## [478] Projection techniques to update the truncated SVD of evolving matrices with applications

**Authors**: *Vasileios Kalantzis, Georgios Kollias, Shashanka Ubaru, Athanasios N. Nikolakopoulos, Lior Horesh, Kenneth L. Clarkson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kalantzis21a.html](http://proceedings.mlr.press/v139/kalantzis21a.html)

**Abstract**:

This submission considers the problem of updating the rank-$k$ truncated Singular Value Decomposition (SVD) of matrices subject to the addition of new rows and/or columns over time. Such matrix problems represent an important computational kernel in applications such as Latent Semantic Indexing and Recommender Systems. Nonetheless, the proposed framework is purely algebraic and targets general updating problems. The algorithm presented in this paper undertakes a projection viewpoint and focuses on building a pair of subspaces which approximate the linear span of the sought singular vectors of the updated matrix. We discuss and analyze two different choices to form the projection subspaces. Results on matrices from real applications suggest that the proposed algorithm can lead to higher accuracy, especially for the singular triplets associated with the largest modulus singular values. Several practical details and key differences with other approaches are also discussed.

----

## [479] Optimal Off-Policy Evaluation from Multiple Logging Policies

**Authors**: *Nathan Kallus, Yuta Saito, Masatoshi Uehara*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kallus21a.html](http://proceedings.mlr.press/v139/kallus21a.html)

**Abstract**:

We study off-policy evaluation (OPE) from multiple logging policies, each generating a dataset of fixed size, i.e., stratified sampling. Previous work noted that in this setting the ordering of the variances of different importance sampling estimators is instance-dependent, which brings up a dilemma as to which importance sampling weights to use. In this paper, we resolve this dilemma by finding the OPE estimator for multiple loggers with minimum variance for any instance, i.e., the efficient one. In particular, we establish the efficiency bound under stratified sampling and propose an estimator achieving this bound when given consistent $q$-estimates. To guard against misspecification of $q$-functions, we also provide a way to choose the control variate in a hypothesis class to minimize variance. Extensive experiments demonstrate the benefits of our methods’ efficiently leveraging of the stratified sampling of off-policy data from multiple loggers.

----

## [480] Efficient Performance Bounds for Primal-Dual Reinforcement Learning from Demonstrations

**Authors**: *Angeliki Kamoutsi, Goran Banjac, John Lygeros*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kamoutsi21a.html](http://proceedings.mlr.press/v139/kamoutsi21a.html)

**Abstract**:

We consider large-scale Markov decision processes with an unknown cost function and address the problem of learning a policy from a finite set of expert demonstrations. 			We assume that the learner is not allowed to interact with the expert and has no access to reinforcement signal of any kind. 			Existing inverse reinforcement learning methods come with strong theoretical guarantees, but are computationally expensive, while state-of-the-art policy optimization algorithms achieve significant empirical success, but are hampered by limited theoretical understanding. 			To bridge the gap between theory and practice, we introduce a novel bilinear saddle-point framework using Lagrangian duality. 			The proposed primal-dual viewpoint allows us to develop a model-free provably efficient algorithm through the lens of stochastic convex optimization. The method enjoys the advantages of simplicity of implementation, low memory requirements, and computational and sample complexities independent of the number of states. We further present an equivalent no-regret online-learning interpretation.

----

## [481] Statistical Estimation from Dependent Data

**Authors**: *Anthimos Vardis Kandiros, Yuval Dagan, Nishanth Dikkala, Surbhi Goel, Constantinos Daskalakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kandiros21a.html](http://proceedings.mlr.press/v139/kandiros21a.html)

**Abstract**:

We consider a general statistical estimation problem wherein binary labels across different observations are not independent conditioning on their feature vectors, but dependent, capturing settings where e.g. these observations are collected on a spatial domain, a temporal domain, or a social network, which induce dependencies. We model these dependencies in the language of Markov Random Fields and, importantly, allow these dependencies to be substantial, i.e. do not assume that the Markov Random Field capturing these dependencies is in high temperature. As our main contribution we provide algorithms and statistically efficient estimation rates for this model, giving several instantiations of our bounds in logistic regression, sparse logistic regression, and neural network regression settings with dependent data. Our estimation guarantees follow from novel results for estimating the parameters (i.e. external fields and interaction strengths) of Ising models from a single sample.

----

## [482] SKIing on Simplices: Kernel Interpolation on the Permutohedral Lattice for Scalable Gaussian Processes

**Authors**: *Sanyam Kapoor, Marc Finzi, Ke Alexander Wang, Andrew Gordon Wilson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kapoor21a.html](http://proceedings.mlr.press/v139/kapoor21a.html)

**Abstract**:

State-of-the-art methods for scalable Gaussian processes use iterative algorithms, requiring fast matrix vector multiplies (MVMs) with the co-variance kernel. The Structured Kernel Interpolation (SKI) framework accelerates these MVMs by performing efficient MVMs on a grid and interpolating back to the original space. In this work, we develop a connection between SKI and the permutohedral lattice used for high-dimensional fast bilateral filtering. Using a sparse simplicial grid instead of a dense rectangular one, we can perform GP inference exponentially faster in the dimension than SKI. Our approach, Simplex-GP, enables scaling SKI to high dimensions, while maintaining strong predictive performance. We additionally provide a CUDA implementation of Simplex-GP, which enables significant GPU acceleration of MVM based inference.

----

## [483] Variational Auto-Regressive Gaussian Processes for Continual Learning

**Authors**: *Sanyam Kapoor, Theofanis Karaletsos, Thang D. Bui*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kapoor21b.html](http://proceedings.mlr.press/v139/kapoor21b.html)

**Abstract**:

Through sequential construction of posteriors on observing data online, Bayes’ theorem provides a natural framework for continual learning. We develop Variational Auto-Regressive Gaussian Processes (VAR-GPs), a principled posterior updating mechanism to solve sequential tasks in continual learning. By relying on sparse inducing point approximations for scalable posteriors, we propose a novel auto-regressive variational distribution which reveals two fruitful connections to existing results in Bayesian inference, expectation propagation and orthogonal inducing points. Mean predictive entropy estimates show VAR-GPs prevent catastrophic forgetting, which is empirically supported by strong performance on modern continual learning benchmarks against competitive baselines. A thorough ablation study demonstrates the efficacy of our modeling choices.

----

## [484] Off-Policy Confidence Sequences

**Authors**: *Nikos Karampatziakis, Paul Mineiro, Aaditya Ramdas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/karampatziakis21a.html](http://proceedings.mlr.press/v139/karampatziakis21a.html)

**Abstract**:

We develop confidence bounds that hold uniformly over time for off-policy evaluation in the contextual bandit setting. These confidence sequences are based on recent ideas from martingale analysis and are non-asymptotic, non-parametric, and valid at arbitrary stopping times. We provide algorithms for computing these confidence sequences that strike a good balance between computational and statistical efficiency. We empirically demonstrate the tightness of our approach in terms of failure probability and width and apply it to the “gated deployment” problem of safely upgrading a production contextual bandit system.

----

## [485] Learning from History for Byzantine Robust Optimization

**Authors**: *Sai Praneeth Karimireddy, Lie He, Martin Jaggi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/karimireddy21a.html](http://proceedings.mlr.press/v139/karimireddy21a.html)

**Abstract**:

Byzantine robustness has received significant attention recently given its importance for distributed and federated learning. In spite of this, we identify severe flaws in existing algorithms even when the data across the participants is identically distributed. First, we show realistic examples where current state of the art robust aggregation rules fail to converge even in the absence of any Byzantine attackers. Secondly, we prove that even if the aggregation rules may succeed in limiting the influence of the attackers in a single round, the attackers can couple their attacks across time eventually leading to divergence. To address these issues, we present two surprisingly simple strategies: a new robust iterative clipping procedure, and incorporating worker momentum to overcome time-coupled attacks. This is the first provably robust method for the standard stochastic optimization setting.

----

## [486] Non-Negative Bregman Divergence Minimization for Deep Direct Density Ratio Estimation

**Authors**: *Masahiro Kato, Takeshi Teshima*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kato21a.html](http://proceedings.mlr.press/v139/kato21a.html)

**Abstract**:

Density ratio estimation (DRE) is at the core of various machine learning tasks such as anomaly detection and domain adaptation. In the DRE literature, existing studies have extensively studied methods based on Bregman divergence (BD) minimization. However, when we apply the BD minimization with highly flexible models, such as deep neural networks, it tends to suffer from what we call train-loss hacking, which is a source of over-fitting caused by a typical characteristic of empirical BD estimators. In this paper, to mitigate train-loss hacking, we propose non-negative correction for empirical BD estimators. Theoretically, we confirm the soundness of the proposed method through a generalization error bound. In our experiments, the proposed methods show favorable performances in inlier-based outlier detection.

----

## [487] Improved Algorithms for Agnostic Pool-based Active Classification

**Authors**: *Julian Katz-Samuels, Jifan Zhang, Lalit Jain, Kevin Jamieson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/katz-samuels21a.html](http://proceedings.mlr.press/v139/katz-samuels21a.html)

**Abstract**:

We consider active learning for binary classification in the agnostic pool-based setting. The vast majority of works in active learning in the agnostic setting are inspired by the CAL algorithm where each query is uniformly sampled from the disagreement region of the current version space. The sample complexity of such algorithms is described by a quantity known as the disagreement coefficient which captures both the geometry of the hypothesis space as well as the underlying probability space. To date, the disagreement coefficient has been justified by minimax lower bounds only, leaving the door open for superior instance dependent sample complexities. In this work we propose an algorithm that, in contrast to uniform sampling over the disagreement region, solves an experimental design problem to determine a distribution over examples from which to request labels. We show that the new approach achieves sample complexity bounds that are never worse than the best disagreement coefficient-based bounds, but in specific cases can be dramatically smaller. From a practical perspective, the proposed algorithm requires no hyperparameters to tune (e.g., to control the aggressiveness of sampling), and is computationally efficient by means of assuming access to an empirical risk minimization oracle (without any constraints). Empirically, we demonstrate that our algorithm is superior to state of the art agnostic active learning algorithms on image classification datasets.

----

## [488] When Does Data Augmentation Help With Membership Inference Attacks?

**Authors**: *Yigitcan Kaya, Tudor Dumitras*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kaya21a.html](http://proceedings.mlr.press/v139/kaya21a.html)

**Abstract**:

Deep learning models often raise privacy concerns as they leak information about their training data. This leakage enables membership inference attacks (MIA) that can identify whether a data point was in a model’s training set. Research shows that some ’data augmentation’ mechanisms may reduce the risk by combatting a key factor increasing the leakage, overfitting. While many mechanisms exist, their effectiveness against MIAs and privacy properties have not been studied systematically. Employing two recent MIAs, we explore the lower bound on the risk in the absence of formal upper bounds. First, we evaluate 7 mechanisms and differential privacy, on three image classification tasks. We find that applying augmentation to increase the model’s utility does not mitigate the risk and protection comes with a utility penalty. Further, we also investigate why popular label smoothing mechanism consistently amplifies the risk. Finally, we propose ’loss-rank-correlation’ (LRC) metric to assess how similar the effects of different mechanisms are. This, for example, reveals the similarity of applying high-intensity augmentation against MIAs to simply reducing the training time. Our findings emphasize the utility-privacy trade-off and provide practical guidelines on using augmentation to manage the trade-off.

----

## [489] Regularized Submodular Maximization at Scale

**Authors**: *Ehsan Kazemi, Shervin Minaee, Moran Feldman, Amin Karbasi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kazemi21a.html](http://proceedings.mlr.press/v139/kazemi21a.html)

**Abstract**:

In this paper, we propose scalable methods for maximizing a regularized submodular function $f \triangleq g-\ell$ expressed as the difference between a monotone submodular function $g$ and a modular function $\ell$. Submodularity is inherently related to the notions of diversity, coverage, and representativeness. In particular, finding the mode (i.e., the most likely configuration) of many popular probabilistic models of diversity, such as determinantal point processes and strongly log-concave distributions, involves maximization of (regularized) submodular functions. Since a regularized function $f$ can potentially take on negative values, the classic theory of submodular maximization, which heavily relies on the non-negativity assumption of submodular functions, is not applicable. To circumvent this challenge, we develop the first one-pass streaming algorithm for maximizing a regularized submodular function subject to a $k$-cardinality constraint. Furthermore, we develop the first distributed algorithm that returns a solution $S$ in $O(1/ \epsilon)$ rounds of MapReduce computation. We highlight that our result, even for the unregularized case where the modular term $\ell$ is zero, improves the memory and communication complexity of the state-of-the-art by a factor of $O(1/ \epsilon)$ while arguably provides a simpler distributed algorithm and a unifying analysis. We empirically study the performance of our scalable methods on a set of real-life applications, including finding the mode of negatively correlated distributions, vertex cover of social networks, and several data summarization tasks.

----

## [490] Prior Image-Constrained Reconstruction using Style-Based Generative Models

**Authors**: *Varun A. Kelkar, Mark A. Anastasio*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kelkar21a.html](http://proceedings.mlr.press/v139/kelkar21a.html)

**Abstract**:

Obtaining a useful estimate of an object from highly incomplete imaging measurements remains a holy grail of imaging science. Deep learning methods have shown promise in learning object priors or constraints to improve the conditioning of an ill-posed imaging inverse problem. In this study, a framework for estimating an object of interest that is semantically related to a known prior image, is proposed. An optimization problem is formulated in the disentangled latent space of a style-based generative model, and semantically meaningful constraints are imposed using the disentangled latent representation of the prior image. Stable recovery from incomplete measurements with the help of a prior image is theoretically analyzed. Numerical experiments demonstrating the superior performance of our approach as compared to related methods are presented.

----

## [491] Self Normalizing Flows

**Authors**: *T. Anderson Keller, Jorn W. T. Peters, Priyank Jaini, Emiel Hoogeboom, Patrick Forré, Max Welling*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/keller21a.html](http://proceedings.mlr.press/v139/keller21a.html)

**Abstract**:

Efficient gradient computation of the Jacobian determinant term is a core problem in many machine learning settings, and especially so in the normalizing flow framework. Most proposed flow models therefore either restrict to a function class with easy evaluation of the Jacobian determinant, or an efficient estimator thereof. However, these restrictions limit the performance of such density models, frequently requiring significant depth to reach desired performance levels. In this work, we propose \emph{Self Normalizing Flows}, a flexible framework for training normalizing flows by replacing expensive terms in the gradient by learned approximate inverses at each layer. This reduces the computational complexity of each layer’s exact update from $\mathcal{O}(D^3)$ to $\mathcal{O}(D^2)$, allowing for the training of flow architectures which were otherwise computationally infeasible, while also providing efficient sampling. We show experimentally that such models are remarkably stable and optimize to similar data likelihood values as their exact gradient counterparts, while training more quickly and surpassing the performance of functionally constrained counterparts.

----

## [492] Interpretable Stability Bounds for Spectral Graph Filters

**Authors**: *Henry Kenlay, Dorina Thanou, Xiaowen Dong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kenlay21a.html](http://proceedings.mlr.press/v139/kenlay21a.html)

**Abstract**:

Graph-structured data arise in a variety of real-world context ranging from sensor and transportation to biological and social networks. As a ubiquitous tool to process graph-structured data, spectral graph filters have been used to solve common tasks such as denoising and anomaly detection, as well as design deep learning architectures such as graph neural networks. Despite being an important tool, there is a lack of theoretical understanding of the stability properties of spectral graph filters, which are important for designing robust machine learning models. In this paper, we study filter stability and provide a novel and interpretable upper bound on the change of filter output, where the bound is expressed in terms of the endpoint degrees of the deleted and newly added edges, as well as the spatial proximity of those edges. This upper bound allows us to reason, in terms of structural properties of the graph, when a spectral graph filter will be stable. We further perform extensive experiments to verify intuition that can be gained from the bound.

----

## [493] Affine Invariant Analysis of Frank-Wolfe on Strongly Convex Sets

**Authors**: *Thomas Kerdreux, Lewis Liu, Simon Lacoste-Julien, Damien Scieur*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kerdreux21a.html](http://proceedings.mlr.press/v139/kerdreux21a.html)

**Abstract**:

It is known that the Frank-Wolfe (FW) algorithm, which is affine covariant, enjoys faster convergence rates than $\mathcal{O}\left(1/K\right)$ when the constraint set is strongly convex. However, these results rely on norm-dependent assumptions, usually incurring non-affine invariant bounds, in contradiction with FW’s affine covariant property. In this work, we introduce new structural assumptions on the problem (such as the directional smoothness) and derive an affine invariant, norm-independent analysis of Frank-Wolfe. We show that our rates are better than any other known convergence rates of FW in this setting. Based on our analysis, we propose an affine invariant backtracking line-search. Interestingly, we show that typical backtracking line-searches using smoothness of the objective function present similar performances than its affine invariant counterpart, despite using affine dependent norms in the step size’s computation.

----

## [494] Markpainting: Adversarial Machine Learning meets Inpainting

**Authors**: *David Khachaturov, Ilia Shumailov, Yiren Zhao, Nicolas Papernot, Ross J. Anderson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/khachaturov21a.html](http://proceedings.mlr.press/v139/khachaturov21a.html)

**Abstract**:

Inpainting is a learned interpolation technique that is based on generative modeling and used to populate masked or missing pieces in an image; it has wide applications in picture editing and retouching. Recently, inpainting started being used for watermark removal, raising concerns. In this paper we study how to manipulate it using our markpainting technique. First, we show how an image owner with access to an inpainting model can augment their image in such a way that any attempt to edit it using that model will add arbitrary visible information. We find that we can target multiple different models simultaneously with our technique. This can be designed to reconstitute a watermark if the editor had been trying to remove it. Second, we show that our markpainting technique is transferable to models that have different architectures or were trained on different datasets, so watermarks created using it are difficult for adversaries to remove. Markpainting is novel and can be used as a manipulation alarm that becomes visible in the event of inpainting. Source code is available at: https://github.com/iliaishacked/markpainting.

----

## [495] Finite-Sample Analysis of Off-Policy Natural Actor-Critic Algorithm

**Authors**: *Sajad Khodadadian, Zaiwei Chen, Siva Theja Maguluri*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/khodadadian21a.html](http://proceedings.mlr.press/v139/khodadadian21a.html)

**Abstract**:

In this paper, we provide finite-sample convergence guarantees for an off-policy variant of the natural actor-critic (NAC) algorithm based on Importance Sampling. In particular, we show that the algorithm converges to a global optimal policy with a sample complexity of $\mathcal{O}(\epsilon^{-3}\log^2(1/\epsilon))$ under an appropriate choice of stepsizes. In order to overcome the issue of large variance due to Importance Sampling, we propose the $Q$-trace algorithm for the critic, which is inspired by the V-trace algorithm (Espeholt et al., 2018). This enables us to explicitly control the bias and variance, and characterize the trade-off between them. As an advantage of off-policy sampling, a major feature of our result is that we do not need any additional assumptions, beyond the ergodicity of the Markov chain induced by the behavior policy.

----

## [496] Functional Space Analysis of Local GAN Convergence

**Authors**: *Valentin Khrulkov, Artem Babenko, Ivan V. Oseledets*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/khrulkov21a.html](http://proceedings.mlr.press/v139/khrulkov21a.html)

**Abstract**:

Recent work demonstrated the benefits of studying continuous-time dynamics governing the GAN training. However, this dynamics is analyzed in the model parameter space, which results in finite-dimensional dynamical systems. We propose a novel perspective where we study the local dynamics of adversarial training in the general functional space and show how it can be represented as a system of partial differential equations. Thus, the convergence properties can be inferred from the eigenvalues of the resulting differential operator. We show that these eigenvalues can be efficiently estimated from the target dataset before training. Our perspective reveals several insights on the practical tricks commonly used to stabilize GANs, such as gradient penalty, data augmentation, and advanced integration schemes. As an immediate practical benefit, we demonstrate how one can a priori select an optimal data augmentation strategy for a particular generation task.

----

## [497] "Hey, that's not an ODE": Faster ODE Adjoints via Seminorms

**Authors**: *Patrick Kidger, Ricky T. Q. Chen, Terry J. Lyons*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kidger21a.html](http://proceedings.mlr.press/v139/kidger21a.html)

**Abstract**:

Neural differential equations may be trained by backpropagating gradients via the adjoint method, which is another differential equation typically solved using an adaptive-step-size numerical differential equation solver. A proposed step is accepted if its error, \emph{relative to some norm}, is sufficiently small; else it is rejected, the step is shrunk, and the process is repeated. Here, we demonstrate that the particular structure of the adjoint equations makes the usual choices of norm (such as $L^2$) unnecessarily stringent. By replacing it with a more appropriate (semi)norm, fewer steps are unnecessarily rejected and the backpropagation is made faster. This requires only minor code modifications. Experiments on a wide range of tasks—including time series, generative modeling, and physical control—demonstrate a median improvement of 40% fewer function evaluations. On some problems we see as much as 62% fewer function evaluations, so that the overall training time is roughly halved.

----

## [498] Neural SDEs as Infinite-Dimensional GANs

**Authors**: *Patrick Kidger, James Foster, Xuechen Li, Terry J. Lyons*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kidger21b.html](http://proceedings.mlr.press/v139/kidger21b.html)

**Abstract**:

Stochastic differential equations (SDEs) are a staple of mathematical modelling of temporal dynamics. However, a fundamental limitation has been that such models have typically been relatively inflexible, which recent work introducing Neural SDEs has sought to solve. Here, we show that the current classical approach to fitting SDEs may be approached as a special case of (Wasserstein) GANs, and in doing so the neural and classical regimes may be brought together. The input noise is Brownian motion, the output samples are time-evolving paths produced by a numerical solver, and by parameterising a discriminator as a Neural Controlled Differential Equation (CDE), we obtain Neural SDEs as (in modern machine learning parlance) continuous-time generative time series models. Unlike previous work on this problem, this is a direct extension of the classical approach without reference to either prespecified statistics or density functions. Arbitrary drift and diffusions are admissible, so as the Wasserstein loss has a unique global minima, in the infinite data limit \textit{any} SDE may be learnt.

----

## [499] GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training

**Authors**: *KrishnaTeja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, Rishabh K. Iyer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/killamsetty21a.html](http://proceedings.mlr.press/v139/killamsetty21a.html)

**Abstract**:

The great success of modern machine learning models on large datasets is contingent on extensive computational resources with high financial and environmental costs. One way to address this is by extracting subsets that generalize on par with the full data. In this work, we propose a general framework, GRAD-MATCH, which finds subsets that closely match the gradient of the \emph{training or validation} set. We find such subsets effectively using an orthogonal matching pursuit algorithm. We show rigorous theoretical and convergence guarantees of the proposed algorithm and, through our extensive experiments on real-world datasets, show the effectiveness of our proposed framework. We show that GRAD-MATCH significantly and consistently outperforms several recent data-selection algorithms and achieves the best accuracy-efficiency trade-off. GRAD-MATCH is available as a part of the CORDS toolkit: \url{https://github.com/decile-team/cords}.

----

## [500] Improving Predictors via Combination Across Diverse Task Categories

**Authors**: *Kwang In Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21a.html](http://proceedings.mlr.press/v139/kim21a.html)

**Abstract**:

Predictor combination is the problem of improving a task predictor using predictors of other tasks when the forms of individual predictors are unknown. Previous work approached this problem by nonparametrically assessing predictor relationships based on their joint evaluations on a shared sample. This limits their application to cases where all predictors are defined on the same task category, e.g. all predictors estimate attributes of shoes. We present a new predictor combination algorithm that overcomes this limitation. Our algorithm aligns the heterogeneous domains of different predictors in a shared latent space to facilitate comparisons of predictors independently of the domains on which they are originally defined. We facilitate this by a new data alignment scheme that matches data distributions across task categories. Based on visual attribute ranking experiments on datasets that span diverse task categories (e.g. shoes and animals), we demonstrate that our approach often significantly improves the performances of the initial predictors.

----

## [501] Self-Improved Retrosynthetic Planning

**Authors**: *Junsu Kim, Sungsoo Ahn, Hankook Lee, Jinwoo Shin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21b.html](http://proceedings.mlr.press/v139/kim21b.html)

**Abstract**:

Retrosynthetic planning is a fundamental problem in chemistry for finding a pathway of reactions to synthesize a target molecule. Recently, search algorithms have shown promising results for solving this problem by using deep neural networks (DNNs) to expand their candidate solutions, i.e., adding new reactions to reaction pathways. However, the existing works on this line are suboptimal; the retrosynthetic planning problem requires the reaction pathways to be (a) represented by real-world reactions and (b) executable using “building block” molecules, yet the DNNs expand reaction pathways without fully incorporating such requirements. Motivated by this, we propose an end-to-end framework for directly training the DNNs towards generating reaction pathways with the desirable properties. Our main idea is based on a self-improving procedure that trains the model to imitate successful trajectories found by itself. We also propose a novel reaction augmentation scheme based on a forward reaction model. Our experiments demonstrate that our scheme significantly improves the success rate of solving the retrosynthetic problem from 86.84% to 96.32% while maintaining the performance of DNN for predicting valid reactions.

----

## [502] Reward Identification in Inverse Reinforcement Learning

**Authors**: *Kuno Kim, Shivam Garg, Kirankumar Shiragur, Stefano Ermon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21c.html](http://proceedings.mlr.press/v139/kim21c.html)

**Abstract**:

We study the problem of reward identifiability in the context of Inverse Reinforcement Learning (IRL). The reward identifiability question is critical to answer when reasoning about the effectiveness of using Markov Decision Processes (MDPs) as computational models of real world decision makers in order to understand complex decision making behavior and perform counterfactual reasoning. While identifiability has been acknowledged as a fundamental theoretical question in IRL, little is known about the types of MDPs for which rewards are identifiable, or even if there exist such MDPs. In this work, we formalize the reward identification problem in IRL and study how identifiability relates to properties of the MDP model. For deterministic MDP models with the MaxEntRL objective, we prove necessary and sufficient conditions for identifiability. Building on these results, we present efficient algorithms for testing whether or not an MDP model is identifiable.

----

## [503] I-BERT: Integer-only BERT Quantization

**Authors**: *Sehoon Kim, Amir Gholami, Zhewei Yao, Michael W. Mahoney, Kurt Keutzer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21d.html](http://proceedings.mlr.press/v139/kim21d.html)

**Abstract**:

Transformer based models, like BERT and RoBERTa, have achieved state-of-the-art results in many Natural Language Processing tasks. However, their memory footprint, inference latency, and power consumption are prohibitive efficient inference at the edge, and even at the data center. While quantization can be a viable solution for this, previous work on quantizing Transformer based models use floating-point arithmetic during inference, which cannot efficiently utilize integer-only logical units such as the recent Turing Tensor Cores, or traditional integer-only ARM processors. In this work, we propose I-BERT, a novel quantization scheme for Transformer based models that quantizes the entire inference with integer-only arithmetic. Based on lightweight integer-only approximation methods for nonlinear operations, e.g., GELU, Softmax, and Layer Normalization, I-BERT performs an end-to-end integer-only BERT inference without any floating point calculation. We evaluate our approach on GLUE downstream tasks using RoBERTa-Base/Large. We show that for both cases, I-BERT achieves similar (and slightly higher) accuracy as compared to the full-precision baseline. Furthermore, our preliminary implementation of I-BERT shows a speedup of 2.4- 4.0x for INT8 inference on a T4 GPU system as compared to FP32 inference. The framework has been developed in PyTorch and has been open-sourced.

----

## [504] Message Passing Adaptive Resonance Theory for Online Active Semi-supervised Learning

**Authors**: *Taehyeong Kim, Injune Hwang, Hyundo Lee, Hyunseo Kim, Won-Seok Choi, Joseph J. Lim, Byoung-Tak Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21e.html](http://proceedings.mlr.press/v139/kim21e.html)

**Abstract**:

Active learning is widely used to reduce labeling effort and training time by repeatedly querying only the most beneficial samples from unlabeled data. In real-world problems where data cannot be stored indefinitely due to limited storage or privacy issues, the query selection and the model update should be performed as soon as a new data sample is observed. Various online active learning methods have been studied to deal with these challenges; however, there are difficulties in selecting representative query samples and updating the model efficiently without forgetting. In this study, we propose Message Passing Adaptive Resonance Theory (MPART) that learns the distribution and topology of input data online. Through message passing on the topological graph, MPART actively queries informative and representative samples, and continuously improves the classification performance using both labeled and unlabeled data. We evaluate our model in stream-based selective sampling scenarios with comparable query selection strategies, showing that MPART significantly outperforms competitive models.

----

## [505] Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

**Authors**: *Jaehyeon Kim, Jungil Kong, Juhee Son*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21f.html](http://proceedings.mlr.press/v139/kim21f.html)

**Abstract**:

Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel sampling have been proposed, but their sample quality does not match that of two-stage TTS systems. In this work, we present a parallel end-to-end TTS method that generates more natural sounding audio than current two-stage models. Our method adopts variational inference augmented with normalizing flows and an adversarial training process, which improves the expressive power of generative modeling. We also propose a stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the natural one-to-many relationship in which a text input can be spoken in multiple ways with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS) on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly available TTS systems and achieves a MOS comparable to ground truth.

----

## [506] A Policy Gradient Algorithm for Learning to Learn in Multiagent Reinforcement Learning

**Authors**: *Dong-Ki Kim, Miao Liu, Matthew Riemer, Chuangchuang Sun, Marwa Abdulhai, Golnaz Habibi, Sebastian Lopez-Cot, Gerald Tesauro, Jonathan P. How*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21g.html](http://proceedings.mlr.press/v139/kim21g.html)

**Abstract**:

A fundamental challenge in multiagent reinforcement learning is to learn beneficial behaviors in a shared environment with other simultaneously learning agents. In particular, each agent perceives the environment as effectively non-stationary due to the changing policies of other agents. Moreover, each agent is itself constantly learning, leading to natural non-stationarity in the distribution of experiences encountered. In this paper, we propose a novel meta-multiagent policy gradient theorem that directly accounts for the non-stationary policy dynamics inherent to multiagent learning settings. This is achieved by modeling our gradient updates to consider both an agent’s own non-stationary policy dynamics and the non-stationary policy dynamics of other agents in the environment. We show that our theoretically grounded approach provides a general solution to the multiagent learning problem, which inherently comprises all key aspects of previous state of the art approaches on this topic. We test our method on a diverse suite of multiagent benchmarks and demonstrate a more efficient ability to adapt to new agents as they learn than baseline methods across the full spectrum of mixed incentive, competitive, and cooperative domains.

----

## [507] Inferring Latent Dynamics Underlying Neural Population Activity via Neural Differential Equations

**Authors**: *Timothy D. Kim, Thomas Z. Luo, Jonathan W. Pillow, Carlos Brody*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21h.html](http://proceedings.mlr.press/v139/kim21h.html)

**Abstract**:

An important problem in systems neuroscience is to identify the latent dynamics underlying neural population activity. Here we address this problem by introducing a low-dimensional nonlinear model for latent neural population dynamics using neural ordinary differential equations (neural ODEs), with noisy sensory inputs and Poisson spike train outputs. We refer to this as the Poisson Latent Neural Differential Equations (PLNDE) model. We apply the PLNDE framework to a variety of synthetic datasets, and show that it accurately infers the phase portraits and fixed points of nonlinear systems augmented to produce spike train data, including the FitzHugh-Nagumo oscillator, a 3-dimensional nonlinear spiral, and a nonlinear sensory decision-making model with attractor dynamics. Our model significantly outperforms existing methods at inferring single-trial neural firing rates and the corresponding latent trajectories that generated them, especially in the regime where the spike counts and number of trials are low. We then apply our model to multi-region neural population recordings from medial frontal cortex of rats performing an auditory decision-making task. Our model provides a general, interpretable framework for investigating the neural mechanisms of decision-making and other cognitive computations through the lens of dynamical systems.

----

## [508] The Lipschitz Constant of Self-Attention

**Authors**: *Hyunjik Kim, George Papamakarios, Andriy Mnih*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21i.html](http://proceedings.mlr.press/v139/kim21i.html)

**Abstract**:

Lipschitz constants of neural networks have been explored in various contexts in deep learning, such as provable adversarial robustness, estimating Wasserstein distance, stabilising training of GANs, and formulating invertible neural networks. Such works have focused on bounding the Lipschitz constant of fully connected or convolutional networks, composed of linear maps and pointwise non-linearities. In this paper, we investigate the Lipschitz constant of self-attention, a non-linear neural network module widely used in sequence modelling. We prove that the standard dot-product self-attention is not Lipschitz for unbounded input domain, and propose an alternative L2 self-attention that is Lipschitz. We derive an upper bound on the Lipschitz constant of L2 self-attention and provide empirical evidence for its asymptotic tightness. To demonstrate the practical relevance of our theoretical work, we formulate invertible self-attention and use it in a Transformer-based architecture for a character-level language modelling task.

----

## [509] Unsupervised Skill Discovery with Bottleneck Option Learning

**Authors**: *Jaekyeom Kim, Seohong Park, Gunhee Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21j.html](http://proceedings.mlr.press/v139/kim21j.html)

**Abstract**:

Having the ability to acquire inherent skills from environments without any external rewards or supervision like humans is an important problem. We propose a novel unsupervised skill discovery method named Information Bottleneck Option Learning (IBOL). On top of the linearization of environments that promotes more various and distant state transitions, IBOL enables the discovery of diverse skills. It provides the abstraction of the skills learned with the information bottleneck framework for the options with improved stability and encouraged disentanglement. We empirically demonstrate that IBOL outperforms multiple state-of-the-art unsupervised skill discovery methods on the information-theoretic evaluations and downstream tasks in MuJoCo environments, including Ant, HalfCheetah, Hopper and D’Kitty. Our code is available at https://vision.snu.ac.kr/projects/ibol.

----

## [510] ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision

**Authors**: *Wonjae Kim, Bokyung Son, Ildoo Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kim21k.html](http://proceedings.mlr.press/v139/kim21k.html)

**Abstract**:

Vision-and-Language Pre-training (VLP) has improved performance on various joint vision-and-language downstream tasks. Current approaches to VLP heavily rely on image feature extraction processes, most of which involve region supervision (e.g., object detection) and the convolutional architecture (e.g., ResNet). Although disregarded in the literature, we find it problematic in terms of both (1) efficiency/speed, that simply extracting input features requires much more computation than the multimodal interaction steps; and (2) expressive power, as it is upper bounded to the expressive power of the visual embedder and its predefined visual vocabulary. In this paper, we present a minimal VLP model, Vision-and-Language Transformer (ViLT), monolithic in the sense that the processing of visual inputs is drastically simplified to just the same convolution-free manner that we process textual inputs. We show that ViLT is up to tens of times faster than previous VLP models, yet with competitive or better downstream task performance. Our code and pre-trained weights are available at https://github.com/dandelin/vilt.

----

## [511] Bias-Robust Bayesian Optimization via Dueling Bandits

**Authors**: *Johannes Kirschner, Andreas Krause*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kirschner21a.html](http://proceedings.mlr.press/v139/kirschner21a.html)

**Abstract**:

We consider Bayesian optimization in settings where observations can be adversarially biased, for example by an uncontrolled hidden confounder. Our first contribution is a reduction of the confounded setting to the dueling bandit model. Then we propose a novel approach for dueling bandits based on information-directed sampling (IDS). Thereby, we obtain the first efficient kernelized algorithm for dueling bandits that comes with cumulative regret guarantees. Our analysis further generalizes a previously proposed semi-parametric linear bandit model to non-linear reward functions, and uncovers interesting links to doubly-robust estimation.

----

## [512] CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients

**Authors**: *Dani Kiyasseh, Tingting Zhu, David A. Clifton*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kiyasseh21a.html](http://proceedings.mlr.press/v139/kiyasseh21a.html)

**Abstract**:

The healthcare industry generates troves of unlabelled physiological data. This data can be exploited via contrastive learning, a self-supervised pre-training method that encourages representations of instances to be similar to one another. We propose a family of contrastive learning methods, CLOCS, that encourages representations across space, time, \textit{and} patients to be similar to one another. We show that CLOCS consistently outperforms the state-of-the-art methods, BYOL and SimCLR, when performing a linear evaluation of, and fine-tuning on, downstream tasks. We also show that CLOCS achieves strong generalization performance with only 25% of labelled training data. Furthermore, our training procedure naturally generates patient-specific representations that can be used to quantify patient-similarity.

----

## [513] Scalable Optimal Transport in High Dimensions for Graph Distances, Embedding Alignment, and More

**Authors**: *Johannes Klicpera, Marten Lienen, Stephan Günnemann*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/klicpera21a.html](http://proceedings.mlr.press/v139/klicpera21a.html)

**Abstract**:

The current best practice for computing optimal transport (OT) is via entropy regularization and Sinkhorn iterations. This algorithm runs in quadratic time as it requires the full pairwise cost matrix, which is prohibitively expensive for large sets of objects. In this work we propose two effective log-linear time approximations of the cost matrix: First, a sparse approximation based on locality-sensitive hashing (LSH) and, second, a Nyström approximation with LSH-based sparse corrections, which we call locally corrected Nyström (LCN). These approximations enable general log-linear time algorithms for entropy-regularized OT that perform well even for the complex, high-dimensional spaces common in deep learning. We analyse these approximations theoretically and evaluate them experimentally both directly and end-to-end as a component for real-world applications. Using our approximations for unsupervised word embedding alignment enables us to speed up a state-of-the-art method by a factor of 3 while also improving the accuracy by 3.1 percentage points without any additional model changes. For graph distance regression we propose the graph transport network (GTN), which combines graph neural networks (GNNs) with enhanced Sinkhorn. GTN outcompetes previous models by 48% and still scales log-linearly in the number of nodes.

----

## [514] Representational aspects of depth and conditioning in normalizing flows

**Authors**: *Frederic Koehler, Viraj Mehta, Andrej Risteski*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/koehler21a.html](http://proceedings.mlr.press/v139/koehler21a.html)

**Abstract**:

Normalizing flows are among the most popular paradigms in generative modeling, especially for images, primarily because we can efficiently evaluate the likelihood of a data point. This is desirable both for evaluating the fit of a model, and for ease of training, as maximizing the likelihood can be done by gradient descent. However, training normalizing flows comes with difficulties as well: models which produce good samples typically need to be extremely deep – which comes with accompanying vanishing/exploding gradient problems. A very related problem is that they are often poorly \emph{conditioned}: since they are parametrized as invertible maps from $\mathbb{R}^d \to \mathbb{R}^d$, and typical training data like images intuitively is lower-dimensional, the learned maps often have Jacobians that are close to being singular. In our paper, we tackle representational aspects around depth and conditioning of normalizing flows: both for general invertible architectures, and for a particular common architecture, affine couplings. We prove that $\Theta(1)$ affine coupling layers suffice to exactly represent a permutation or $1 \times 1$ convolution, as used in GLOW, showing that representationally the choice of partition is not a bottleneck for depth. We also show that shallow affine coupling networks are universal approximators in Wasserstein distance if ill-conditioning is allowed, and experimentally investigate related phenomena involving padding. Finally, we show a depth lower bound for general flow architectures with few neurons per layer and bounded Lipschitz constant.

----

## [515] WILDS: A Benchmark of in-the-Wild Distribution Shifts

**Authors**: *Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, Tony Lee, Etienne David, Ian Stavness, Wei Guo, Berton Earnshaw, Imran S. Haque, Sara M. Beery, Jure Leskovec, Anshul Kundaje, Emma Pierson, Sergey Levine, Chelsea Finn, Percy Liang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/koh21a.html](http://proceedings.mlr.press/v139/koh21a.html)

**Abstract**:

Distribution shifts—where the training distribution differs from the test distribution—can substantially degrade the accuracy of machine learning (ML) systems deployed in the wild. Despite their ubiquity in the real-world deployments, these distribution shifts are under-represented in the datasets widely used in the ML community today. To address this gap, we present WILDS, a curated benchmark of 10 datasets reflecting a diverse range of distribution shifts that naturally arise in real-world applications, such as shifts across hospitals for tumor identification; across camera traps for wildlife monitoring; and across time and location in satellite imaging and poverty mapping. On each dataset, we show that standard training yields substantially lower out-of-distribution than in-distribution performance. This gap remains even with models trained by existing methods for tackling distribution shifts, underscoring the need for new methods for training models that are more robust to the types of distribution shifts that arise in practice. To facilitate method development, we provide an open-source package that automates dataset loading, contains default model architectures and hyperparameters, and standardizes evaluations. The full paper, code, and leaderboards are available at https://wilds.stanford.edu.

----

## [516] One-sided Frank-Wolfe algorithms for saddle problems

**Authors**: *Vladimir Kolmogorov, Thomas Pock*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kolmogorov21a.html](http://proceedings.mlr.press/v139/kolmogorov21a.html)

**Abstract**:

We study a class of convex-concave saddle-point problems of the form $\min_x\max_y ⟨Kx,y⟩+f_{\cal P}(x)-h^*(y)$ where $K$ is a linear operator, $f_{\cal P}$ is the sum of a convex function $f$ with a Lipschitz-continuous gradient and the indicator function of a bounded convex polytope ${\cal P}$, and $h^\ast$ is a convex (possibly nonsmooth) function. Such problem arises, for example, as a Lagrangian relaxation of various discrete optimization problems. Our main assumptions are the existence of an efficient {\em linear minimization oracle} ($lmo$) for $f_{\cal P}$ and an efficient {\em proximal map} ($prox$) for $h^*$ which motivate the solution via a blend of proximal primal-dual algorithms and Frank-Wolfe algorithms. In case $h^*$ is the indicator function of a linear constraint and function $f$ is quadratic, we show a $O(1/n^2)$ convergence rate on the dual objective, requiring $O(n \log n)$ calls of $lmo$. If the problem comes from the constrained optimization problem $\min_{x\in\mathbb R^d}\{f_{\cal P}(x)\:|\:Ax-b=0\}$ then we additionally get bound $O(1/n^2)$ both on the primal gap and on the infeasibility gap. In the most general case, we show a $O(1/n)$ convergence rate of the primal-dual gap again requiring $O(n\log n)$ calls of $lmo$. To the best of our knowledge, this improves on the known convergence rates for the considered class of saddle-point problems. We show applications to labeling problems frequently appearing in machine learning and computer vision.

----

## [517] A Lower Bound for the Sample Complexity of Inverse Reinforcement Learning

**Authors**: *Abi Komanduru, Jean Honorio*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/komanduru21a.html](http://proceedings.mlr.press/v139/komanduru21a.html)

**Abstract**:

Inverse reinforcement learning (IRL) is the task of finding a reward function that generates a desired optimal policy for a given Markov Decision Process (MDP). This paper develops an information-theoretic lower bound for the sample complexity of the finite state, finite action IRL problem. A geometric construction of $\beta$-strict separable IRL problems using spherical codes is considered. Properties of the ensemble size as well as the Kullback-Leibler divergence between the generated trajectories are derived. The resulting ensemble is then used along with Fano’s inequality to derive a sample complexity lower bound of $O(n \log n)$, where $n$ is the number of states in the MDP.

----

## [518] Consensus Control for Decentralized Deep Learning

**Authors**: *Lingjing Kong, Tao Lin, Anastasia Koloskova, Martin Jaggi, Sebastian U. Stich*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kong21a.html](http://proceedings.mlr.press/v139/kong21a.html)

**Abstract**:

Decentralized training of deep learning models enables on-device learning over networks, as well as efficient scaling to large compute clusters. Experiments in earlier works reveal that, even in a data-center setup, decentralized training often suffers from the degradation in the quality of the model: the training and test performance of models trained in a decentralized fashion is in general worse than that of models trained in a centralized fashion, and this performance drop is impacted by parameters such as network size, communication topology and data partitioning. We identify the changing consensus distance between devices as a key parameter to explain the gap between centralized and decentralized training. We show in theory that when the training consensus distance is lower than a critical quantity, decentralized training converges as fast as the centralized counterpart. We empirically validate that the relation between generalization performance and consensus distance is consistent with this theoretical observation. Our empirical insights allow the principled design of better decentralized training schemes that mitigate the performance drop. To this end, we provide practical training guidelines and exemplify its effectiveness on the data-center setup as the important first step.

----

## [519] A Distribution-dependent Analysis of Meta Learning

**Authors**: *Mikhail Konobeev, Ilja Kuzborskij, Csaba Szepesvári*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/konobeev21a.html](http://proceedings.mlr.press/v139/konobeev21a.html)

**Abstract**:

A key problem in the theory of meta-learning is to understand how the task distributions influence transfer risk, the expected error of a meta-learner on a new task drawn from the unknown task distribution. In this paper, focusing on fixed design linear regression with Gaussian noise and a Gaussian task (or parameter) distribution, we give distribution-dependent lower bounds on the transfer risk of any algorithm, while we also show that a novel, weighted version of the so-called biased regularized regression method is able to match these lower bounds up to a fixed constant factor. Notably, the weighting is derived from the covariance of the Gaussian task distribution. Altogether, our results provide a precise characterization of the difficulty of meta-learning in this Gaussian setting. While this problem setting may appear simple, we show that it is rich enough to unify the “parameter sharing” and “representation learning” streams of meta-learning; in particular, representation learning is obtained as the special case when the covariance matrix of the task distribution is unknown. For this case we propose to adopt the EM method, which is shown to enjoy efficient updates in our case. The paper is completed by an empirical study of EM. In particular, our experimental results show that the EM algorithm can attain the lower bound as the number of tasks grows, while the algorithm is also successful in competing with its alternatives when used in a representation learning context.

----

## [520] Evaluating Robustness of Predictive Uncertainty Estimation: Are Dirichlet-based Models Reliable?

**Authors**: *Anna-Kathrin Kopetzki, Bertrand Charpentier, Daniel Zügner, Sandhya Giri, Stephan Günnemann*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kopetzki21a.html](http://proceedings.mlr.press/v139/kopetzki21a.html)

**Abstract**:

Dirichlet-based uncertainty (DBU) models are a recent and promising class of uncertainty-aware models. DBU models predict the parameters of a Dirichlet distribution to provide fast, high-quality uncertainty estimates alongside with class predictions. In this work, we present the first large-scale, in-depth study of the robustness of DBU models under adversarial attacks. Our results suggest that uncertainty estimates of DBU models are not robust w.r.t. three important tasks: (1) indicating correctly and wrongly classified samples; (2) detecting adversarial examples; and (3) distinguishing between in-distribution (ID) and out-of-distribution (OOD) data. Additionally, we explore the first approaches to make DBU mod- els more robust. While adversarial training has a minor effect, our median smoothing based ap- proach significantly increases robustness of DBU models.

----

## [521] Kernel Stein Discrepancy Descent

**Authors**: *Anna Korba, Pierre-Cyril Aubin-Frankowski, Szymon Majewski, Pierre Ablin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/korba21a.html](http://proceedings.mlr.press/v139/korba21a.html)

**Abstract**:

Among dissimilarities between probability distributions, the Kernel Stein Discrepancy (KSD) has received much interest recently. We investigate the properties of its Wasserstein gradient flow to approximate a target probability distribution $\pi$ on $\mathbb{R}^d$, known up to a normalization constant. This leads to a straightforwardly implementable, deterministic score-based method to sample from $\pi$, named KSD Descent, which uses a set of particles to approximate $\pi$. Remarkably, owing to a tractable loss function, KSD Descent can leverage robust parameter-free optimization schemes such as L-BFGS; this contrasts with other popular particle-based schemes such as the Stein Variational Gradient Descent algorithm. We study the convergence properties of KSD Descent and demonstrate its practical relevance. However, we also highlight failure cases by showing that the algorithm can get stuck in spurious local minima.

----

## [522] Boosting the Throughput and Accelerator Utilization of Specialized CNN Inference Beyond Increasing Batch Size

**Authors**: *Jack Kosaian, Amar Phanishayee, Matthai Philipose, Debadeepta Dey, Rashmi Vinayak*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kosaian21a.html](http://proceedings.mlr.press/v139/kosaian21a.html)

**Abstract**:

Datacenter vision systems widely use small, specialized convolutional neural networks (CNNs) trained on specific tasks for high-throughput inference. These settings employ accelerators with massive computational capacity, but which specialized CNNs underutilize due to having low arithmetic intensity. This results in suboptimal application-level throughput and poor returns on accelerator investment. Increasing batch size is the only known way to increase both application-level throughput and accelerator utilization for inference, but yields diminishing returns; specialized CNNs poorly utilize accelerators even with large batch size. We propose FoldedCNNs, a new approach to CNN design that increases inference throughput and utilization beyond large batch size. FoldedCNNs rethink the structure of inputs and layers of specialized CNNs to boost arithmetic intensity: in FoldedCNNs, f images with C channels each are concatenated into a single input with fC channels and jointly classified by a wider CNN. Increased arithmetic intensity in FoldedCNNs increases the throughput and GPU utilization of specialized CNN inference by up to 2.5x and 2.8x, with accuracy close to the original CNN in most cases.

----

## [523] NeRF-VAE: A Geometry Aware 3D Scene Generative Model

**Authors**: *Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Sona Mokrá, Danilo Jimenez Rezende*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kosiorek21a.html](http://proceedings.mlr.press/v139/kosiorek21a.html)

**Abstract**:

We propose NeRF-VAE, a 3D scene generative model that incorporates geometric structure via Neural Radiance Fields (NeRF) and differentiable volume rendering. In contrast to NeRF, our model takes into account shared structure across scenes, and is able to infer the structure of a novel scene—without the need to re-train—using amortized inference. NeRF-VAE’s explicit 3D rendering process further contrasts previous generative models with convolution-based rendering which lacks geometric structure. Our model is a VAE that learns a distribution over radiance fields by conditioning them on a latent scene representation. We show that, once trained, NeRF-VAE is able to infer and render geometrically-consistent scenes from previously unseen 3D environments of synthetic scenes using very few input images. We further demonstrate that NeRF-VAE generalizes well to out-of-distribution cameras, while convolutional models do not. Finally, we introduce and study an attention-based conditioning mechanism of NeRF-VAE’s decoder, which improves model performance.

----

## [524] Active Testing: Sample-Efficient Model Evaluation

**Authors**: *Jannik Kossen, Sebastian Farquhar, Yarin Gal, Tom Rainforth*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kossen21a.html](http://proceedings.mlr.press/v139/kossen21a.html)

**Abstract**:

We introduce a new framework for sample-efficient model evaluation that we call active testing. While approaches like active learning reduce the number of labels needed for model training, existing literature largely ignores the cost of labeling test data, typically unrealistically assuming large test sets for model evaluation. This creates a disconnect to real applications, where test labels are important and just as expensive, e.g. for optimizing hyperparameters. Active testing addresses this by carefully selecting the test points to label, ensuring model evaluation is sample-efficient. To this end, we derive theoretically-grounded and intuitive acquisition strategies that are specifically tailored to the goals of active testing, noting these are distinct to those of active learning. As actively selecting labels introduces a bias; we further show how to remove this bias while reducing the variance of the estimator at the same time. Active testing is easy to implement and can be applied to any supervised machine learning method. We demonstrate its effectiveness on models including WideResNets and Gaussian processes on datasets including Fashion-MNIST and CIFAR-100.

----

## [525] High Confidence Generalization for Reinforcement Learning

**Authors**: *James E. Kostas, Yash Chandak, Scott M. Jordan, Georgios Theocharous, Philip S. Thomas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kostas21a.html](http://proceedings.mlr.press/v139/kostas21a.html)

**Abstract**:

We present several classes of reinforcement learning algorithms that safely generalize to Markov decision processes (MDPs) not seen during training. Specifically, we study the setting in which some set of MDPs is accessible for training. The goal is to generalize safely to MDPs that are sampled from the same distribution, but which may not be in the set accessible for training. For various definitions of safety, our algorithms give probabilistic guarantees that agents can safely generalize to MDPs that are sampled from the same distribution but are not necessarily in the training set. These algorithms are a type of Seldonian algorithm (Thomas et al., 2019), which is a class of machine learning algorithms that return models with probabilistic safety guarantees for user-specified definitions of safety.

----

## [526] Offline Reinforcement Learning with Fisher Divergence Critic Regularization

**Authors**: *Ilya Kostrikov, Rob Fergus, Jonathan Tompson, Ofir Nachum*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kostrikov21a.html](http://proceedings.mlr.press/v139/kostrikov21a.html)

**Abstract**:

Many modern approaches to offline Reinforcement Learning (RL) utilize behavior regularization, typically augmenting a model-free actor critic algorithm with a penalty measuring divergence of the policy from the offline data. In this work, we propose an alternative approach to encouraging the learned policy to stay close to the data, namely parameterizing the critic as the log-behavior-policy, which generated the offline data, plus a state-action value offset term, which can be learned using a neural network. Behavior regularization then corresponds to an appropriate regularizer on the offset term. We propose using a gradient penalty regularizer for the offset term and demonstrate its equivalence to Fisher divergence regularization, suggesting connections to the score matching and generative energy-based model literature. We thus term our resulting algorithm Fisher-BRC (Behavior Regularized Critic). On standard offline RL benchmarks, Fisher-BRC achieves both improved performance and faster convergence over existing state-of-the-art methods.

----

## [527] ADOM: Accelerated Decentralized Optimization Method for Time-Varying Networks

**Authors**: *Dmitry Kovalev, Egor Shulgin, Peter Richtárik, Alexander Rogozin, Alexander V. Gasnikov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kovalev21a.html](http://proceedings.mlr.press/v139/kovalev21a.html)

**Abstract**:

We propose ADOM – an accelerated method for smooth and strongly convex decentralized optimization over time-varying networks. ADOM uses a dual oracle, i.e., we assume access to the gradient of the Fenchel conjugate of the individual loss functions. Up to a constant factor, which depends on the network structure only, its communication complexity is the same as that of accelerated Nesterov gradient method. To the best of our knowledge, only the algorithm of Rogozin et al. (2019) has a convergence rate with similar properties. However, their algorithm converges under the very restrictive assumption that the number of network changes can not be greater than a tiny percentage of the number of iterations. This assumption is hard to satisfy in practice, as the network topology changes usually can not be controlled. In contrast, ADOM merely requires the network to stay connected throughout time.

----

## [528] Revisiting Peng's Q(λ) for Modern Reinforcement Learning

**Authors**: *Tadashi Kozuno, Yunhao Tang, Mark Rowland, Rémi Munos, Steven Kapturowski, Will Dabney, Michal Valko, David Abel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kozuno21a.html](http://proceedings.mlr.press/v139/kozuno21a.html)

**Abstract**:

Off-policy multi-step reinforcement learning algorithms consist of conservative and non-conservative algorithms: the former actively cut traces, whereas the latter do not. Recently, Munos et al. (2016) proved the convergence of conservative algorithms to an optimal Q-function. In contrast, non-conservative algorithms are thought to be unsafe and have a limited or no theoretical guarantee. Nonetheless, recent studies have shown that non-conservative algorithms empirically outperform conservative ones. Motivated by the empirical results and the lack of theory, we carry out theoretical analyses of Peng’s Q($\lambda$), a representative example of non-conservative algorithms. We prove that \emph{it also converges to an optimal policy} provided that the behavior policy slowly tracks a greedy policy in a way similar to conservative policy iteration. Such a result has been conjectured to be true but has not been proven. We also experiment with Peng’s Q($\lambda$) in complex continuous control tasks, confirming that Peng’s Q($\lambda$) often outperforms conservative algorithms despite its simplicity. These results indicate that Peng’s Q($\lambda$), which was thought to be unsafe, is a theoretically-sound and practically effective algorithm.

----

## [529] Adapting to misspecification in contextual bandits with offline regression oracles

**Authors**: *Sanath Kumar Krishnamurthy, Vitor Hadad, Susan Athey*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/krishnamurthy21a.html](http://proceedings.mlr.press/v139/krishnamurthy21a.html)

**Abstract**:

Computationally efficient contextual bandits are often based on estimating a predictive model of rewards given contexts and arms using past data. However, when the reward model is not well-specified, the bandit algorithm may incur unexpected regret, so recent work has focused on algorithms that are robust to misspecification. We propose a simple family of contextual bandit algorithms that adapt to misspecification error by reverting to a good safe policy when there is evidence that misspecification is causing a regret increase. Our algorithm requires only an offline regression oracle to ensure regret guarantees that gracefully degrade in terms of a measure of the average misspecification level. Compared to prior work, we attain similar regret guarantees, but we do no rely on a master algorithm, and do not require more robust oracles like online or constrained regression oracles (e.g., Foster et al. (2020), Krishnamurthy et al. (2020)). This allows us to design algorithms for more general function approximation classes.

----

## [530] Out-of-Distribution Generalization via Risk Extrapolation (REx)

**Authors**: *David Krueger, Ethan Caballero, Jörn-Henrik Jacobsen, Amy Zhang, Jonathan Binas, Dinghuai Zhang, Rémi Le Priol, Aaron C. Courville*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/krueger21a.html](http://proceedings.mlr.press/v139/krueger21a.html)

**Abstract**:

Distributional shift is one of the major obstacles when transferring machine learning prediction systems from the lab to the real world. To tackle this problem, we assume that variation across training domains is representative of the variation we might encounter at test time, but also that shifts at test time may be more extreme in magnitude. In particular, we show that reducing differences in risk across training domains can reduce a model’s sensitivity to a wide range of extreme distributional shifts, including the challenging setting where the input contains both causal and anti-causal elements. We motivate this approach, Risk Extrapolation (REx), as a form of robust optimization over a perturbation set of extrapolated domains (MM-REx), and propose a penalty on the variance of training risks (V-REx) as a simpler variant. We prove that variants of REx can recover the causal mechanisms of the targets, while also providing robustness to changes in the input distribution (“covariate shift”). By appropriately trading-off robustness to causally induced distributional shifts and covariate shift, REx is able to outperform alternative methods such as Invariant Risk Minimization in situations where these types of shift co-occur.

----

## [531] Near-Optimal Confidence Sequences for Bounded Random Variables

**Authors**: *Arun K. Kuchibhotla, Qinqing Zheng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kuchibhotla21a.html](http://proceedings.mlr.press/v139/kuchibhotla21a.html)

**Abstract**:

Many inference problems, such as sequential decision problems like A/B testing, adaptive sampling schemes like bandit selection, are often online in nature. The fundamental problem for online inference is to provide a sequence of confidence intervals that are valid uniformly over the growing-into-infinity sample sizes. To address this question, we provide a near-optimal confidence sequence for bounded random variables by utilizing Bentkus’ concentration results. We show that it improves on the existing approaches that use the Cram{é}r-Chernoff technique such as the Hoeffding, Bernstein, and Bennett inequalities. The resulting confidence sequence is confirmed to be favorable in synthetic coverage problems, adaptive stopping algorithms, and multi-armed bandit problems.

----

## [532] Differentially Private Bayesian Inference for Generalized Linear Models

**Authors**: *Tejas Kulkarni, Joonas Jälkö, Antti Koskela, Samuel Kaski, Antti Honkela*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kulkarni21a.html](http://proceedings.mlr.press/v139/kulkarni21a.html)

**Abstract**:

Generalized linear models (GLMs) such as logistic regression are among the most widely used arms in data analyst’s repertoire and often used on sensitive datasets. A large body of prior works that investigate GLMs under differential privacy (DP) constraints provide only private point estimates of the regression coefficients, and are not able to quantify parameter uncertainty. In this work, with logistic and Poisson regression as running examples, we introduce a generic noise-aware DP Bayesian inference method for a GLM at hand, given a noisy sum of summary statistics. Quantifying uncertainty allows us to determine which of the regression coefficients are statistically significantly different from zero. We provide a previously unknown tight privacy analysis and experimentally demonstrate that the posteriors obtained from our model, while adhering to strong privacy guarantees, are close to the non-private posteriors.

----

## [533] Bayesian Structural Adaptation for Continual Learning

**Authors**: *Abhishek Kumar, Sunabha Chatterjee, Piyush Rai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kumar21a.html](http://proceedings.mlr.press/v139/kumar21a.html)

**Abstract**:

Continual Learning is a learning paradigm where learning systems are trained on a sequence of tasks. The goal here is to perform well on the current task without suffering from a performance drop on the previous tasks. Two notable directions among the recent advances in continual learning with neural networks are (1) variational Bayes based regularization by learning priors from previous tasks, and, (2) learning the structure of deep networks to adapt to new tasks. So far, these two approaches have been largely orthogonal. We present a novel Bayesian framework based on continually learning the structure of deep neural networks, to unify these distinct yet complementary approaches. The proposed framework learns the deep structure for each task by learning which weights to be used, and supports inter-task transfer through the overlapping of different sparse subsets of weights learned by different tasks. An appealing aspect of our proposed continual learning framework is that it is applicable to both discriminative (supervised) and generative (unsupervised) settings. Experimental results on supervised and unsupervised benchmarks demonstrate that our approach performs comparably or better than recent advances in continual learning.

----

## [534] Implicit rate-constrained optimization of non-decomposable objectives

**Authors**: *Abhishek Kumar, Harikrishna Narasimhan, Andrew Cotter*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kumar21b.html](http://proceedings.mlr.press/v139/kumar21b.html)

**Abstract**:

We consider a popular family of constrained optimization problems arising in machine learning that involve optimizing a non-decomposable evaluation metric with a certain thresholded form, while constraining another metric of interest. Examples of such problems include optimizing false negative rate at a fixed false positive rate, optimizing precision at a fixed recall, optimizing the area under the precision-recall or ROC curves, etc. Our key idea is to formulate a rate-constrained optimization that expresses the threshold parameter as a function of the model parameters via the Implicit Function theorem. We show how the resulting optimization problem can be solved using standard gradient based methods. Experiments on benchmark datasets demonstrate the effectiveness of our proposed method over existing state-of-the-art approaches for these problems.

----

## [535] A Scalable Second Order Method for Ill-Conditioned Matrix Completion from Few Samples

**Authors**: *Christian Kümmerle, Claudio Mayrink Verdun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kummerle21a.html](http://proceedings.mlr.press/v139/kummerle21a.html)

**Abstract**:

We propose an iterative algorithm for low-rank matrix completion with that can be interpreted as an iteratively reweighted least squares (IRLS) algorithm, a saddle-escaping smoothing Newton method or a variable metric proximal gradient method applied to a non-convex rank surrogate. It combines the favorable data-efficiency of previous IRLS approaches with an improved scalability by several orders of magnitude. We establish the first local convergence guarantee from a minimal number of samples for that class of algorithms, showing that the method attains a local quadratic convergence rate. Furthermore, we show that the linear systems to be solved are well-conditioned even for very ill-conditioned ground truth matrices. We provide extensive experiments, indicating that unlike many state-of-the-art approaches, our method is able to complete very ill-conditioned matrices with a condition number of up to $10^{10}$ from few samples, while being competitive in its scalability.

----

## [536] Meta-Thompson Sampling

**Authors**: *Branislav Kveton, Mikhail Konobeev, Manzil Zaheer, Chih-Wei Hsu, Martin Mladenov, Craig Boutilier, Csaba Szepesvári*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kveton21a.html](http://proceedings.mlr.press/v139/kveton21a.html)

**Abstract**:

Efficient exploration in bandits is a fundamental online learning problem. We propose a variant of Thompson sampling that learns to explore better as it interacts with bandit instances drawn from an unknown prior. The algorithm meta-learns the prior and thus we call it MetaTS. We propose several efficient implementations of MetaTS and analyze it in Gaussian bandits. Our analysis shows the benefit of meta-learning and is of a broader interest, because we derive a novel prior-dependent Bayes regret bound for Thompson sampling. Our theory is complemented by empirical evaluation, which shows that MetaTS quickly adapts to the unknown prior.

----

## [537] Targeted Data Acquisition for Evolving Negotiation Agents

**Authors**: *Minae Kwon, Siddharth Karamcheti, Mariano-Florentino Cuellar, Dorsa Sadigh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kwon21a.html](http://proceedings.mlr.press/v139/kwon21a.html)

**Abstract**:

Successful negotiators must learn how to balance optimizing for self-interest and cooperation. Yet current artificial negotiation agents often heavily depend on the quality of the static datasets they were trained on, limiting their capacity to fashion an adaptive response balancing self-interest and cooperation. For this reason, we find that these agents can achieve either high utility or cooperation, but not both. To address this, we introduce a targeted data acquisition framework where we guide the exploration of a reinforcement learning agent using annotations from an expert oracle. The guided exploration incentivizes the learning agent to go beyond its static dataset and develop new negotiation strategies. We show that this enables our agents to obtain higher-reward and more Pareto-optimal solutions when negotiating with both simulated and human partners compared to standard supervised learning and reinforcement learning methods. This trend additionally holds when comparing agents using our targeted data acquisition framework to variants of agents trained with a mix of supervised learning and reinforcement learning, or to agents using tailored reward functions that explicitly optimize for utility and Pareto-optimality.

----

## [538] ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks

**Authors**: *Jungmin Kwon, Jeongseop Kim, Hyunseo Park, In Kwon Choi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/kwon21b.html](http://proceedings.mlr.press/v139/kwon21b.html)

**Abstract**:

Recently, learning algorithms motivated from sharpness of loss surface as an effective measure of generalization gap have shown state-of-the-art performances. Nevertheless, sharpness defined in a rigid region with a fixed radius, has a drawback in sensitivity to parameter re-scaling which leaves the loss unaffected, leading to weakening of the connection between sharpness and generalization gap. In this paper, we introduce the concept of adaptive sharpness which is scale-invariant and propose the corresponding generalization bound. We suggest a novel learning method, adaptive sharpness-aware minimization (ASAM), utilizing the proposed generalization bound. Experimental results in various benchmark datasets show that ASAM contributes to significant improvement of model generalization performance.

----

## [539] On the price of explainability for some clustering problems

**Authors**: *Eduardo Sany Laber, Lucas Murtinho*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/laber21a.html](http://proceedings.mlr.press/v139/laber21a.html)

**Abstract**:

The price of explainability for a clustering task can be defined as the unavoidable loss, in terms of the objective function, if we force the final partition to be explainable. Here, we study this price for the following clustering problems: $k$-means, $k$-medians, $k$-centers and maximum-spacing. We provide upper and lower bounds for a natural model where explainability is achieved via decision trees. For the $k$-means and $k$-medians problems our upper bounds improve those obtained by [Dasgupta et. al, ICML 20] for low dimensions. Another contribution is a simple and efficient algorithm for building explainable clusterings for the $k$-means problem. We provide empirical evidence that its performance is better than the current state of the art for decision-tree based explainable clustering.

----

## [540] Adaptive Newton Sketch: Linear-time Optimization with Quadratic Convergence and Effective Hessian Dimensionality

**Authors**: *Jonathan Lacotte, Yifei Wang, Mert Pilanci*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lacotte21a.html](http://proceedings.mlr.press/v139/lacotte21a.html)

**Abstract**:

We propose a randomized algorithm with quadratic convergence rate for convex optimization problems with a self-concordant, composite, strongly convex objective function. Our method is based on performing an approximate Newton step using a random projection of the Hessian. Our first contribution is to show that, at each iteration, the embedding dimension (or sketch size) can be as small as the effective dimension of the Hessian matrix. Leveraging this novel fundamental result, we design an algorithm with a sketch size proportional to the effective dimension and which exhibits a quadratic rate of convergence. This result dramatically improves on the classical linear-quadratic convergence rates of state-of-the-art sub-sampled Newton methods. However, in most practical cases, the effective dimension is not known beforehand, and this raises the question of how to pick a sketch size as small as the effective dimension while preserving a quadratic convergence rate. Our second and main contribution is thus to propose an adaptive sketch size algorithm with quadratic convergence rate and which does not require prior knowledge or estimation of the effective dimension: at each iteration, it starts with a small sketch size, and increases it until quadratic progress is achieved. Importantly, we show that the embedding dimension remains proportional to the effective dimension throughout the entire path and that our method achieves state-of-the-art computational complexity for solving convex optimization programs with a strongly convex component. We discuss and illustrate applications to linear and quadratic programming, as well as logistic regression and other generalized linear models.

----

## [541] Generalization Bounds in the Presence of Outliers: a Median-of-Means Study

**Authors**: *Pierre Laforgue, Guillaume Staerman, Stéphan Clémençon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/laforgue21a.html](http://proceedings.mlr.press/v139/laforgue21a.html)

**Abstract**:

In contrast to the empirical mean, the Median-of-Means (MoM) is an estimator of the mean $\theta$ of a square integrable r.v. Z, around which accurate nonasymptotic confidence bounds can be built, even when Z does not exhibit a sub-Gaussian tail behavior. Thanks to the high confidence it achieves on heavy-tailed data, MoM has found various applications in machine learning, where it is used to design training procedures that are not sensitive to atypical observations. More recently, a new line of work is now trying to characterize and leverage MoM’s ability to deal with corrupted data. In this context, the present work proposes a general study of MoM’s concentration properties under the contamination regime, that provides a clear understanding on the impact of the outlier proportion and the number of blocks chosen. The analysis is extended to (multisample) U-statistics, i.e. averages over tuples of observations, that raise additional challenges due to the dependence induced. Finally, we show that the latter bounds can be used in a straightforward fashion to derive generalization guarantees for pairwise learning in a contaminated setting, and propose an algorithm to compute provably reliable decision functions.

----

## [542] Model Fusion for Personalized Learning

**Authors**: *Thanh Chi Lam, Trong Nghia Hoang, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lam21a.html](http://proceedings.mlr.press/v139/lam21a.html)

**Abstract**:

Production systems operating on a growing domain of analytic services often require generating warm-start solution models for emerging tasks with limited data. One potential approach to address this warm-start challenge is to adopt meta learning to generate a base model that can be adapted to solve unseen tasks with minimal fine-tuning. This however requires the training processes of previous solution models of existing tasks to be synchronized. This is not possible if these models were pre-trained separately on private data owned by different entities and cannot be synchronously re-trained. To accommodate for such scenarios, we develop a new personalized learning framework that synthesizes customized models for unseen tasks via fusion of independently pre-trained models of related tasks. We establish performance guarantee for the proposed framework and demonstrate its effectiveness on both synthetic and real datasets.

----

## [543] Gradient Disaggregation: Breaking Privacy in Federated Learning by Reconstructing the User Participant Matrix

**Authors**: *Maximilian Lam, Gu-Yeon Wei, David Brooks, Vijay Janapa Reddi, Michael Mitzenmacher*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lam21b.html](http://proceedings.mlr.press/v139/lam21b.html)

**Abstract**:

We show that aggregated model updates in federated learning may be insecure. An untrusted central server may disaggregate user updates from sums of updates across participants given repeated observations, enabling the server to recover privileged information about individual users’ private training data via traditional gradient inference attacks. Our method revolves around reconstructing participant information (e.g: which rounds of training users participated in) from aggregated model updates by leveraging summary information from device analytics commonly used to monitor, debug, and manage federated learning systems. Our attack is parallelizable and we successfully disaggregate user updates on settings with up to thousands of participants. We quantitatively and qualitatively demonstrate significant improvements in the capability of various inference attacks on the disaggregated updates. Our attack enables the attribution of learned properties to individual users, violating anonymity, and shows that a determined central server may undermine the secure aggregation protocol to break individual users’ data privacy in federated learning.

----

## [544] Stochastic Multi-Armed Bandits with Unrestricted Delay Distributions

**Authors**: *Tal Lancewicki, Shahar Segal, Tomer Koren, Yishay Mansour*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lancewicki21a.html](http://proceedings.mlr.press/v139/lancewicki21a.html)

**Abstract**:

We study the stochastic Multi-Armed Bandit (MAB) problem with random delays in the feedback received by the algorithm. We consider two settings: the {\it reward dependent} delay setting, where realized delays may depend on the stochastic rewards, and the {\it reward-independent} delay setting. Our main contribution is algorithms that achieve near-optimal regret in each of the settings, with an additional additive dependence on the quantiles of the delay distribution. Our results do not make any assumptions on the delay distributions: in particular, we do not assume they come from any parametric family of distributions and allow for unbounded support and expectation; we further allow for the case of infinite delays where the algorithm might occasionally not observe any feedback.

----

## [545] Discovering symbolic policies with deep reinforcement learning

**Authors**: *Mikel Landajuela, Brenden K. Petersen, Sookyung Kim, Cláudio P. Santiago, Ruben Glatt, T. Nathan Mundhenk, Jacob F. Pettit, Daniel M. Faissol*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/landajuela21a.html](http://proceedings.mlr.press/v139/landajuela21a.html)

**Abstract**:

Deep reinforcement learning (DRL) has proven successful for many difficult control problems by learning policies represented by neural networks. However, the complexity of neural network-based policies{—}involving thousands of composed non-linear operators{—}can render them problematic to understand, trust, and deploy. In contrast, simple policies comprising short symbolic expressions can facilitate human understanding, while also being transparent and exhibiting predictable behavior. To this end, we propose deep symbolic policy, a novel approach to directly search the space of symbolic policies. We use an autoregressive recurrent neural network to generate control policies represented by tractable mathematical expressions, employing a risk-seeking policy gradient to maximize performance of the generated policies. To scale to environments with multi-dimensional action spaces, we propose an "anchoring" algorithm that distills pre-trained neural network-based policies into fully symbolic policies, one action dimension at a time. We also introduce two novel methods to improve exploration in DRL-based combinatorial optimization, building on ideas of entropy regularization and distribution initialization. Despite their dramatically reduced complexity, we demonstrate that discovered symbolic policies outperform seven state-of-the-art DRL algorithms in terms of average rank and average normalized episodic reward across eight benchmark environments.

----

## [546] Graph Cuts Always Find a Global Optimum for Potts Models (With a Catch)

**Authors**: *Hunter Lang, David A. Sontag, Aravindan Vijayaraghavan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lang21a.html](http://proceedings.mlr.press/v139/lang21a.html)

**Abstract**:

We prove that the alpha-expansion algorithm for MAP inference always returns a globally optimal assignment for Markov Random Fields with Potts pairwise potentials, with a catch: the returned assignment is only guaranteed to be optimal for an instance within a small perturbation of the original problem instance. In other words, all local minima with respect to expansion moves are global minima to slightly perturbed versions of the problem. On "real-world" instances, MAP assignments of small perturbations of the problem should be very similar to the MAP assignment(s) of the original problem instance. We design an algorithm that can certify whether this is the case in practice. On several MAP inference problem instances from computer vision, this algorithm certifies that MAP solutions to all of these perturbations are very close to solutions of the original instance. These results taken together give a cohesive explanation for the good performance of "graph cuts" algorithms in practice. Every local expansion minimum is a global minimum in a small perturbation of the problem, and all of these global minima are close to the original solution.

----

## [547] Efficient Message Passing for 0-1 ILPs with Binary Decision Diagrams

**Authors**: *Jan-Hendrik Lange, Paul Swoboda*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lange21a.html](http://proceedings.mlr.press/v139/lange21a.html)

**Abstract**:

We present a message passing method for 0{–}1 integer linear programs. Our algorithm is based on a decomposition of the original problem into subproblems that are represented as binary deci- sion diagrams. The resulting Lagrangean dual is solved iteratively by a series of efficient block coordinate ascent steps. Our method has linear iteration complexity in the size of the decomposi- tion and can be effectively parallelized. The char- acteristics of our approach are desirable towards solving ever larger problems arising in structured prediction. We present experimental results on combinatorial problems from MAP inference for Markov Random Fields, quadratic assignment, discrete tomography and cell tracking for develop- mental biology and show promising performance.

----

## [548] CountSketches, Feature Hashing and the Median of Three

**Authors**: *Kasper Green Larsen, Rasmus Pagh, Jakub Tetek*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/larsen21a.html](http://proceedings.mlr.press/v139/larsen21a.html)

**Abstract**:

In this paper, we revisit the classic CountSketch method, which is a sparse, random projection that transforms a (high-dimensional) Euclidean vector $v$ to a vector of dimension $(2t-1) s$, where $t, s > 0$ are integer parameters. It is known that a CountSketch allows estimating coordinates of $v$ with variance bounded by $\|v\|_2^2/s$. For $t > 1$, the estimator takes the median of $2t-1$ independent estimates, and the probability that the estimate is off by more than $2 \|v\|_2/\sqrt{s}$ is exponentially small in $t$. This suggests choosing $t$ to be logarithmic in a desired inverse failure probability. However, implementations of CountSketch often use a small, constant $t$. Previous work only predicts a constant factor improvement in this setting. Our main contribution is a new analysis of CountSketch, showing an improvement in variance to $O(\min\{\|v\|_1^2/s^2,\|v\|_2^2/s\})$ when $t > 1$. That is, the variance decreases proportionally to $s^{-2}$, asymptotically for large enough $s$.

----

## [549] MorphVAE: Generating Neural Morphologies from 3D-Walks using a Variational Autoencoder with Spherical Latent Space

**Authors**: *Sophie Laturnus, Philipp Berens*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/laturnus21a.html](http://proceedings.mlr.press/v139/laturnus21a.html)

**Abstract**:

For the past century, the anatomy of a neuron has been considered one of its defining features: The shape of a neuron’s dendrites and axon fundamentally determines what other neurons it can connect to. These neurites have been described using mathematical tools e.g. in the context of cell type classification, but generative models of these structures have only rarely been proposed and are often computationally inefficient. Here we propose MorphVAE, a sequence-to-sequence variational autoencoder with spherical latent space as a generative model for neural morphologies. The model operates on walks within the tree structure of a neuron and can incorporate expert annotations on a subset of the data using semi-supervised learning. We develop our model on artificially generated toy data and evaluate its performance on dendrites of excitatory cells and axons of inhibitory cells of mouse motor cortex (M1) and dendrites of retinal ganglion cells. We show that the learned latent feature space allows for better cell type discrimination than other commonly used features. By sampling new walks from the latent space we can easily construct new morphologies with a specified degree of similarity to their reference neuron, providing an efficient generative model for neural morphologies.

----

## [550] Improved Regret Bound and Experience Replay in Regularized Policy Iteration

**Authors**: *Nevena Lazic, Dong Yin, Yasin Abbasi-Yadkori, Csaba Szepesvári*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lazic21a.html](http://proceedings.mlr.press/v139/lazic21a.html)

**Abstract**:

In this work, we study algorithms for learning in infinite-horizon undiscounted Markov decision processes (MDPs) with function approximation. We first show that the regret analysis of the Politex algorithm (a version of regularized policy iteration) can be sharpened from $O(T^{3/4})$ to $O(\sqrt{T})$ under nearly identical assumptions, and instantiate the bound with linear function approximation. Our result provides the first high-probability $O(\sqrt{T})$ regret bound for a computationally efficient algorithm in this setting. The exact implementation of Politex with neural network function approximation is inefficient in terms of memory and computation. Since our analysis suggests that we need to approximate the average of the action-value functions of past policies well, we propose a simple efficient implementation where we train a single Q-function on a replay buffer with past data. We show that this often leads to superior performance over other implementation choices, especially in terms of wall-clock time. Our work also provides a novel theoretical justification for using experience replay within policy iteration algorithms.

----

## [551] LAMDA: Label Matching Deep Domain Adaptation

**Authors**: *Trung Le, Tuan Nguyen, Nhat Ho, Hung Bui, Dinh Phung*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/le21a.html](http://proceedings.mlr.press/v139/le21a.html)

**Abstract**:

Deep domain adaptation (DDA) approaches have recently been shown to perform better than their shallow rivals with better modeling capacity on complex domains (e.g., image, structural data, and sequential data). The underlying idea is to learn domain invariant representations on a latent space that can bridge the gap between source and target domains. Several theoretical studies have established insightful understanding and the benefit of learning domain invariant features; however, they are usually limited to the case where there is no label shift, hence hindering its applicability. In this paper, we propose and study a new challenging setting that allows us to use a Wasserstein distance (WS) to not only quantify the data shift but also to define the label shift directly. We further develop a theory to demonstrate that minimizing the WS of the data shift leads to closing the gap between the source and target data distributions on the latent space (e.g., an intermediate layer of a deep net), while still being able to quantify the label shift with respect to this latent space. Interestingly, our theory can consequently explain certain drawbacks of learning domain invariant features on the latent space. Finally, grounded on the results and guidance of our developed theory, we propose the Label Matching Deep Domain Adaptation (LAMDA) approach that outperforms baselines on real-world datasets for DA problems.

----

## [552] Gaussian Process-Based Real-Time Learning for Safety Critical Applications

**Authors**: *Armin Lederer, Alejandro Jose Ordóñez Conejo, Korbinian Maier, Wenxin Xiao, Jonas Umlauft, Sandra Hirche*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lederer21a.html](http://proceedings.mlr.press/v139/lederer21a.html)

**Abstract**:

The safe operation of physical systems typically relies on high-quality models. Since a continuous stream of data is generated during run-time, such models are often obtained through the application of Gaussian process regression because it provides guarantees on the prediction error. Due to its high computational complexity, Gaussian process regression must be used offline on batches of data, which prevents applications, where a fast adaptation through online learning is necessary to ensure safety. In order to overcome this issue, we propose the LoG-GP. It achieves a logarithmic update and prediction complexity in the number of training points through the aggregation of locally active Gaussian process models. Under weak assumptions on the aggregation scheme, it inherits safety guarantees from exact Gaussian process regression. These theoretical advantages are exemplarily exploited in the design of a safe and data-efficient, online-learning control policy. The efficiency and performance of the proposed real-time learning approach is demonstrated in a comparison to state-of-the-art methods.

----

## [553] Sharing Less is More: Lifelong Learning in Deep Networks with Selective Layer Transfer

**Authors**: *Seungwon Lee, Sima Behpour, Eric Eaton*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21a.html](http://proceedings.mlr.press/v139/lee21a.html)

**Abstract**:

Effective lifelong learning across diverse tasks requires the transfer of diverse knowledge, yet transferring irrelevant knowledge may lead to interference and catastrophic forgetting. In deep networks, transferring the appropriate granularity of knowledge is as important as the transfer mechanism, and must be driven by the relationships among tasks. We first show that the lifelong learning performance of several current deep learning architectures can be significantly improved by transfer at the appropriate layers. We then develop an expectation-maximization (EM) method to automatically select the appropriate transfer configuration and optimize the task network weights. This EM-based selective transfer is highly effective, balancing transfer performance on all tasks with avoiding catastrophic forgetting, as demonstrated on three algorithms in several lifelong object classification scenarios.

----

## [554] Fair Selective Classification Via Sufficiency

**Authors**: *Joshua K. Lee, Yuheng Bu, Deepta Rajan, Prasanna Sattigeri, Rameswar Panda, Subhro Das, Gregory W. Wornell*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21b.html](http://proceedings.mlr.press/v139/lee21b.html)

**Abstract**:

Selective classification is a powerful tool for decision-making in scenarios where mistakes are costly but abstentions are allowed. In general, by allowing a classifier to abstain, one can improve the performance of a model at the cost of reducing coverage and classifying fewer samples. However, recent work has shown, in some cases, that selective classification can magnify disparities between groups, and has illustrated this phenomenon on multiple real-world datasets. We prove that the sufficiency criterion can be used to mitigate these disparities by ensuring that selective classification increases performance on all groups, and introduce a method for mitigating the disparity in precision across the entire coverage scale based on this criterion. We then provide an upper bound on the conditional mutual information between the class label and sensitive attribute, conditioned on the learned features, which can be used as a regularizer to achieve fairer selective classification. The effectiveness of the method is demonstrated on the Adult, CelebA, Civil Comments, and CheXpert datasets.

----

## [555] On-the-fly Rectification for Robust Large-Vocabulary Topic Inference

**Authors**: *Moontae Lee, Sungjun Cho, Kun Dong, David Mimno, David Bindel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21c.html](http://proceedings.mlr.press/v139/lee21c.html)

**Abstract**:

Across many data domains, co-occurrence statistics about the joint appearance of objects are powerfully informative. By transforming unsupervised learning problems into decompositions of co-occurrence statistics, spectral algorithms provide transparent and efficient algorithms for posterior inference such as latent topic analysis and community detection. As object vocabularies grow, however, it becomes rapidly more expensive to store and run inference algorithms on co-occurrence statistics. Rectifying co-occurrence, the key process to uphold model assumptions, becomes increasingly more vital in the presence of rare terms, but current techniques cannot scale to large vocabularies. We propose novel methods that simultaneously compress and rectify co-occurrence statistics, scaling gracefully with the size of vocabulary and the dimension of latent space. We also present new algorithms learning latent variables from the compressed statistics, and verify that our methods perform comparably to previous approaches on both textual and non-textual data.

----

## [556] Unsupervised Embedding Adaptation via Early-Stage Feature Reconstruction for Few-Shot Classification

**Authors**: *Dong Hoon Lee, Sae-Young Chung*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21d.html](http://proceedings.mlr.press/v139/lee21d.html)

**Abstract**:

We propose unsupervised embedding adaptation for the downstream few-shot classification task. Based on findings that deep neural networks learn to generalize before memorizing, we develop Early-Stage Feature Reconstruction (ESFR) — a novel adaptation scheme with feature reconstruction and dimensionality-driven early stopping that finds generalizable features. Incorporating ESFR consistently improves the performance of baseline methods on all standard settings, including the recently proposed transductive method. ESFR used in conjunction with the transductive method further achieves state-of-the-art performance on mini-ImageNet, tiered-ImageNet, and CUB; especially with 1.2% 2.0% improvements in accuracy over the previous best performing method on 1-shot setting.

----

## [557] Continual Learning in the Teacher-Student Setup: Impact of Task Similarity

**Authors**: *Sebastian Lee, Sebastian Goldt, Andrew M. Saxe*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21e.html](http://proceedings.mlr.press/v139/lee21e.html)

**Abstract**:

Continual learning{—}the ability to learn many tasks in sequence{—}is critical for artificial learning systems. Yet standard training methods for deep networks often suffer from catastrophic forgetting, where learning new tasks erases knowledge of the earlier tasks. While catastrophic forgetting labels the problem, the theoretical reasons for interference between tasks remain unclear. Here, we attempt to narrow this gap between theory and practice by studying continual learning in the teacher-student setup. We extend previous analytical work on two-layer networks in the teacher-student setup to multiple teachers. Using each teacher to represent a different task, we investigate how the relationship between teachers affects the amount of forgetting and transfer exhibited by the student when the task switches. In line with recent work, we find that when tasks depend on similar features, intermediate task similarity leads to greatest forgetting. However, feature similarity is only one way in which tasks may be related. The teacher-student approach allows us to disentangle task similarity at the level of \emph{readouts} (hidden-to-output weights) as well as \emph{features} (input-to-hidden weights). We find a complex interplay between both types of similarity, initial transfer/forgetting rates, maximum transfer/forgetting, and the long-time (post-switch) amount of transfer/forgetting. Together, these results help illuminate the diverse factors contributing to catastrophic forgetting.

----

## [558] OptiDICE: Offline Policy Optimization via Stationary Distribution Correction Estimation

**Authors**: *Jongmin Lee, Wonseok Jeon, Byung-Jun Lee, Joelle Pineau, Kee-Eung Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21f.html](http://proceedings.mlr.press/v139/lee21f.html)

**Abstract**:

We consider the offline reinforcement learning (RL) setting where the agent aims to optimize the policy solely from the data without further environment interactions. In offline RL, the distributional shift becomes the primary source of difficulty, which arises from the deviation of the target policy being optimized from the behavior policy used for data collection. This typically causes overestimation of action values, which poses severe problems for model-free algorithms that use bootstrapping. To mitigate the problem, prior offline RL algorithms often used sophisticated techniques that encourage underestimation of action values, which introduces an additional set of hyperparameters that need to be tuned properly. In this paper, we present an offline RL algorithm that prevents overestimation in a more principled way. Our algorithm, OptiDICE, directly estimates the stationary distribution corrections of the optimal policy and does not rely on policy-gradients, unlike previous offline RL algorithms. Using an extensive set of benchmark datasets for offline RL, we show that OptiDICE performs competitively with the state-of-the-art methods.

----

## [559] SUNRISE: A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning

**Authors**: *Kimin Lee, Michael Laskin, Aravind Srinivas, Pieter Abbeel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21g.html](http://proceedings.mlr.press/v139/lee21g.html)

**Abstract**:

Off-policy deep reinforcement learning (RL) has been successful in a range of challenging domains. However, standard off-policy RL algorithms can suffer from several issues, such as instability in Q-learning and balancing exploration and exploitation. To mitigate these issues, we present SUNRISE, a simple unified ensemble method, which is compatible with various off-policy RL algorithms. SUNRISE integrates two key ingredients: (a) ensemble-based weighted Bellman backups, which re-weight target Q-values based on uncertainty estimates from a Q-ensemble, and (b) an inference method that selects actions using the highest upper-confidence bounds for efficient exploration. By enforcing the diversity between agents using Bootstrap with random initialization, we show that these different ideas are largely orthogonal and can be fruitfully integrated, together further improving the performance of existing off-policy RL algorithms, such as Soft Actor-Critic and Rainbow DQN, for both continuous and discrete control tasks on both low-dimensional and high-dimensional environments.

----

## [560] Achieving Near Instance-Optimality and Minimax-Optimality in Stochastic and Adversarial Linear Bandits Simultaneously

**Authors**: *Chung-Wei Lee, Haipeng Luo, Chen-Yu Wei, Mengxiao Zhang, Xiaojin Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21h.html](http://proceedings.mlr.press/v139/lee21h.html)

**Abstract**:

In this work, we develop linear bandit algorithms that automatically adapt to different environments. By plugging a novel loss estimator into the optimization problem that characterizes the instance-optimal strategy, our first algorithm not only achieves nearly instance-optimal regret in stochastic environments, but also works in corrupted environments with additional regret being the amount of corruption, while the state-of-the-art (Li et al., 2019) achieves neither instance-optimality nor the optimal dependence on the corruption amount. Moreover, by equipping this algorithm with an adversarial component and carefully-designed testings, our second algorithm additionally enjoys minimax-optimal regret in completely adversarial environments, which is the first of this kind to our knowledge. Finally, all our guarantees hold with high probability, while existing instance-optimal guarantees only hold in expectation.

----

## [561] PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training

**Authors**: *Kimin Lee, Laura M. Smith, Pieter Abbeel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lee21i.html](http://proceedings.mlr.press/v139/lee21i.html)

**Abstract**:

Conveying complex objectives to reinforcement learning (RL) agents can often be difficult, involving meticulous design of reward functions that are sufficiently informative yet easy enough to provide. Human-in-the-loop RL methods allow practitioners to instead interactively teach agents through tailored feedback; however, such approaches have been challenging to scale since human feedback is very expensive. In this work, we aim to make this process more sample- and feedback-efficient. We present an off-policy, interactive RL algorithm that capitalizes on the strengths of both feedback and off-policy learning. Specifically, we learn a reward model by actively querying a teacher’s preferences between two clips of behavior and use it to train an agent. To enable off-policy learning, we relabel all the agent’s past experience when its reward model changes. We additionally show that pre-training our agents with unsupervised exploration substantially increases the mileage of its queries. We demonstrate that our approach is capable of learning tasks of higher complexity than previously considered by human-in-the-loop methods, including a variety of locomotion and robotic manipulation skills. We also show that our method is able to utilize real-time human feedback to effectively prevent reward exploitation and learn new behaviors that are difficult to specify with standard reward functions.

----

## [562] Near-Optimal Linear Regression under Distribution Shift

**Authors**: *Qi Lei, Wei Hu, Jason D. Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lei21a.html](http://proceedings.mlr.press/v139/lei21a.html)

**Abstract**:

Transfer learning is essential when sufficient data comes from the source domain, with scarce labeled data from the target domain. We develop estimators that achieve minimax linear risk for linear regression problems under distribution shift. Our algorithms cover different transfer learning settings including covariate shift and model shift. We also consider when data are generated from either linear or general nonlinear models. We show that linear minimax estimators are within an absolute constant of the minimax risk even among nonlinear estimators for various source/target distributions.

----

## [563] Stability and Generalization of Stochastic Gradient Methods for Minimax Problems

**Authors**: *Yunwen Lei, Zhenhuan Yang, Tianbao Yang, Yiming Ying*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lei21b.html](http://proceedings.mlr.press/v139/lei21b.html)

**Abstract**:

Many machine learning problems can be formulated as minimax problems such as Generative Adversarial Networks (GANs), AUC maximization and robust estimation, to mention but a few. A substantial amount of studies are devoted to studying the convergence behavior of their stochastic gradient-type algorithms. In contrast, there is relatively little work on understanding their generalization, i.e., how the learning models built from training examples would behave on test examples. In this paper, we provide a comprehensive generalization analysis of stochastic gradient methods for minimax problems under both convex-concave and nonconvex-nonconcave cases through the lens of algorithmic stability. We establish a quantitative connection between stability and several generalization measures both in expectation and with high probability. For the convex-concave setting, our stability analysis shows that stochastic gradient descent ascent attains optimal generalization bounds for both smooth and nonsmooth minimax problems. We also establish generalization bounds for both weakly-convex-weakly-concave and gradient-dominated problems. We report preliminary experimental results to verify our theory.

----

## [564] Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot

**Authors**: *Joel Z. Leibo, Edgar A. Duéñez-Guzmán, Alexander Vezhnevets, John P. Agapiou, Peter Sunehag, Raphael Koster, Jayd Matyas, Charlie Beattie, Igor Mordatch, Thore Graepel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/leibo21a.html](http://proceedings.mlr.press/v139/leibo21a.html)

**Abstract**:

Existing evaluation suites for multi-agent reinforcement learning (MARL) do not assess generalization to novel situations as their primary objective (unlike supervised learning benchmarks). Our contribution, Melting Pot, is a MARL evaluation suite that fills this gap and uses reinforcement learning to reduce the human labor required to create novel test scenarios. This works because one agent’s behavior constitutes (part of) another agent’s environment. To demonstrate scalability, we have created over 80 unique test scenarios covering a broad range of research topics such as social dilemmas, reciprocity, resource sharing, and task partitioning. We apply these test scenarios to standard MARL training algorithms, and demonstrate how Melting Pot reveals weaknesses not apparent from training performance alone.

----

## [565] Better Training using Weight-Constrained Stochastic Dynamics

**Authors**: *Benedict J. Leimkuhler, Tiffany J. Vlaar, Timothée Pouchon, Amos J. Storkey*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/leimkuhler21a.html](http://proceedings.mlr.press/v139/leimkuhler21a.html)

**Abstract**:

We employ constraints to control the parameter space of deep neural networks throughout training. The use of customised, appropriately designed constraints can reduce the vanishing/exploding gradients problem, improve smoothness of classification boundaries, control weight magnitudes and stabilize deep neural networks, and thus enhance the robustness of training algorithms and the generalization capabilities of neural networks. We provide a general approach to efficiently incorporate constraints into a stochastic gradient Langevin framework, allowing enhanced exploration of the loss landscape. We also present specific examples of constrained training methods motivated by orthogonality preservation for weight matrices and explicit weight normalizations. Discretization schemes are provided both for the overdamped formulation of Langevin dynamics and the underdamped form, in which momenta further improve sampling efficiency. These optimisation schemes can be used directly, without needing to adapt neural network architecture design choices or to modify the objective with regularization terms, and see performance improvements in classification tasks.

----

## [566] Globally-Robust Neural Networks

**Authors**: *Klas Leino, Zifan Wang, Matt Fredrikson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/leino21a.html](http://proceedings.mlr.press/v139/leino21a.html)

**Abstract**:

The threat of adversarial examples has motivated work on training certifiably robust neural networks to facilitate efficient verification of local robustness at inference time. We formalize a notion of global robustness, which captures the operational properties of on-line local robustness certification while yielding a natural learning objective for robust training. We show that widely-used architectures can be easily adapted to this objective by incorporating efficient global Lipschitz bounds into the network, yielding certifiably-robust models by construction that achieve state-of-the-art verifiable accuracy. Notably, this approach requires significantly less time and memory than recent certifiable training methods, and leads to negligible costs when certifying points on-line; for example, our evaluation shows that it is possible to train a large robust Tiny-Imagenet model in a matter of hours. Our models effectively leverage inexpensive global Lipschitz bounds for real-time certification, despite prior suggestions that tighter local bounds are needed for good performance; we posit this is possible because our models are specifically trained to achieve tighter global bounds. Namely, we prove that the maximum achievable verifiable accuracy for a given dataset is not improved by using a local bound.

----

## [567] Learning to Price Against a Moving Target

**Authors**: *Renato Paes Leme, Balasubramanian Sivan, Yifeng Teng, Pratik Worah*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/leme21a.html](http://proceedings.mlr.press/v139/leme21a.html)

**Abstract**:

In the Learning to Price setting, a seller posts prices over time with the goal of maximizing revenue while learning the buyer’s valuation. This problem is very well understood when values are stationary (fixed or iid). Here we study the problem where the buyer’s value is a moving target, i.e., they change over time either by a stochastic process or adversarially with bounded variation. In either case, we provide matching upper and lower bounds on the optimal revenue loss. Since the target is moving, any information learned soon becomes out-dated, which forces the algorithms to keep switching between exploring and exploiting phases.

----

## [568] SigGPDE: Scaling Sparse Gaussian Processes on Sequential Data

**Authors**: *Maud Lemercier, Cristopher Salvi, Thomas Cass, Edwin V. Bonilla, Theodoros Damoulas, Terry J. Lyons*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lemercier21a.html](http://proceedings.mlr.press/v139/lemercier21a.html)

**Abstract**:

Making predictions and quantifying their uncertainty when the input data is sequential is a fundamental learning challenge, recently attracting increasing attention. We develop SigGPDE, a new scalable sparse variational inference framework for Gaussian Processes (GPs) on sequential data. Our contribution is twofold. First, we construct inducing variables underpinning the sparse approximation so that the resulting evidence lower bound (ELBO) does not require any matrix inversion. Second, we show that the gradients of the GP signature kernel are solutions of a hyperbolic partial differential equation (PDE). This theoretical insight allows us to build an efficient back-propagation algorithm to optimize the ELBO. We showcase the significant computational gains of SigGPDE compared to existing methods, while achieving state-of-the-art performance for classification tasks on large datasets of up to 1 million multivariate time series.

----

## [569] Strategic Classification Made Practical

**Authors**: *Sagi Levanon, Nir Rosenfeld*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/levanon21a.html](http://proceedings.mlr.press/v139/levanon21a.html)

**Abstract**:

Strategic classification regards the problem of learning in settings where users can strategically modify their features to improve outcomes. This setting applies broadly, and has received much recent attention. But despite its practical significance, work in this space has so far been predominantly theoretical. In this paper we present a learning framework for strategic classification that is practical. Our approach directly minimizes the “strategic” empirical risk, which we achieve by differentiating through the strategic response of users. This provides flexibility that allows us to extend beyond the original problem formulation and towards more realistic learning scenarios. A series of experiments demonstrates the effectiveness of our approach on various learning settings.

----

## [570] Improved, Deterministic Smoothing for L1 Certified Robustness

**Authors**: *Alexander Levine, Soheil Feizi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/levine21a.html](http://proceedings.mlr.press/v139/levine21a.html)

**Abstract**:

Randomized smoothing is a general technique for computing sample-dependent robustness guarantees against adversarial attacks for deep classifiers. Prior works on randomized smoothing against L_1 adversarial attacks use additive smoothing noise and provide probabilistic robustness guarantees. In this work, we propose a non-additive and deterministic smoothing method, Deterministic Smoothing with Splitting Noise (DSSN). To develop DSSN, we first develop SSN, a randomized method which involves generating each noisy smoothing sample by first randomly splitting the input space and then returning a representation of the center of the subdivision occupied by the input sample. In contrast to uniform additive smoothing, the SSN certification does not require the random noise components used to be independent. Thus, smoothing can be done effectively in just one dimension and can therefore be efficiently derandomized for quantized data (e.g., images). To the best of our knowledge, this is the first work to provide deterministic "randomized smoothing" for a norm-based adversarial threat model while allowing for an arbitrary classifier (i.e., a deep model) to be used as a base classifier and without requiring an exponential number of smoothing samples. On CIFAR-10 and ImageNet datasets, we provide substantially larger L_1 robustness certificates compared to prior works, establishing a new state-of-the-art. The determinism of our method also leads to significantly faster certificate computation. Code is available at: https://github.com/alevine0/smoothingSplittingNoise.

----

## [571] BASE Layers: Simplifying Training of Large, Sparse Models

**Authors**: *Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, Luke Zettlemoyer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lewis21a.html](http://proceedings.mlr.press/v139/lewis21a.html)

**Abstract**:

We introduce a new balanced assignment of experts (BASE) layer for large language models that greatly simplifies existing high capacity sparse layers. Sparse layers can dramatically improve the efficiency of training and inference by routing each token to specialized expert modules that contain only a small fraction of the model parameters. However, it can be difficult to learn balanced routing functions that make full use of the available experts; existing approaches typically use routing heuristics or auxiliary expert-balancing loss functions. In contrast, we formulate token-to-expert allocation as a linear assignment problem, allowing an optimal assignment in which each expert receives an equal number of tokens. This optimal assignment scheme improves efficiency by guaranteeing balanced compute loads, and also simplifies training by not requiring any new hyperparameters or auxiliary losses. Code is publicly released.

----

## [572] Run-Sort-ReRun: Escaping Batch Size Limitations in Sliced Wasserstein Generative Models

**Authors**: *José Lezama, Wei Chen, Qiang Qiu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lezama21a.html](http://proceedings.mlr.press/v139/lezama21a.html)

**Abstract**:

When training an implicit generative model, ideally one would like the generator to reproduce all the different modes and subtleties of the target distribution. Naturally, when comparing two empirical distributions, the larger the sample population, the more these statistical nuances can be captured. However, existing objective functions are computationally constrained in the amount of samples they can consider by the memory required to process a batch of samples. In this paper, we build upon recent progress in sliced Wasserstein distances, a family of differentiable metrics for distribution discrepancy based on the Optimal Transport paradigm. We introduce a procedure to train these distances with virtually any batch size, allowing the discrepancy measure to capture richer statistics and better approximating the distance between the underlying continuous distributions. As an example, we demonstrate the matching of the distribution of Inception features with batches of tens of thousands of samples, achieving FID scores that outperform state-of-the-art implicit generative models.

----

## [573] PAGE: A Simple and Optimal Probabilistic Gradient Estimator for Nonconvex Optimization

**Authors**: *Zhize Li, Hongyan Bao, Xiangliang Zhang, Peter Richtárik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21a.html](http://proceedings.mlr.press/v139/li21a.html)

**Abstract**:

In this paper, we propose a novel stochastic gradient estimator—ProbAbilistic Gradient Estimator (PAGE)—for nonconvex optimization. PAGE is easy to implement as it is designed via a small adjustment to vanilla SGD: in each iteration, PAGE uses the vanilla minibatch SGD update with probability $p_t$ or reuses the previous gradient with a small adjustment, at a much lower computational cost, with probability $1-p_t$. We give a simple formula for the optimal choice of $p_t$. Moreover, we prove the first tight lower bound $\Omega(n+\frac{\sqrt{n}}{\epsilon^2})$ for nonconvex finite-sum problems, which also leads to a tight lower bound $\Omega(b+\frac{\sqrt{b}}{\epsilon^2})$ for nonconvex online problems, where $b:= \min\{\frac{\sigma^2}{\epsilon^2}, n\}$. Then, we show that PAGE obtains the optimal convergence results $O(n+\frac{\sqrt{n}}{\epsilon^2})$ (finite-sum) and $O(b+\frac{\sqrt{b}}{\epsilon^2})$ (online) matching our lower bounds for both nonconvex finite-sum and online problems. Besides, we also show that for nonconvex functions satisfying the Polyak-Ł{ojasiewicz} (PL) condition, PAGE can automatically switch to a faster linear convergence rate $O(\cdot\log \frac{1}{\epsilon})$. Finally, we conduct several deep learning experiments (e.g., LeNet, VGG, ResNet) on real datasets in PyTorch showing that PAGE not only converges much faster than SGD in training but also achieves the higher test accuracy, validating the optimal theoretical results and confirming the practical superiority of PAGE.

----

## [574] Tightening the Dependence on Horizon in the Sample Complexity of Q-Learning

**Authors**: *Gen Li, Changxiao Cai, Yuxin Chen, Yuantao Gu, Yuting Wei, Yuejie Chi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21b.html](http://proceedings.mlr.press/v139/li21b.html)

**Abstract**:

Q-learning, which seeks to learn the optimal Q-function of a Markov decision process (MDP) in a model-free fashion, lies at the heart of reinforcement learning. Focusing on the synchronous setting (such that independent samples for all state-action pairs are queried via a generative model in each iteration), substantial progress has been made recently towards understanding the sample efficiency of Q-learning. To yield an entrywise $\varepsilon$-accurate estimate of the optimal Q-function, state-of-the-art theory requires at least an order of $\frac{|S||A|}{(1-\gamma)^5\varepsilon^{2}}$ samples in the infinite-horizon $\gamma$-discounted setting. In this work, we sharpen the sample complexity of synchronous Q-learning to the order of $\frac{|S||A|}{(1-\gamma)^4\varepsilon^2}$ (up to some logarithmic factor) for any $0<\varepsilon <1$, leading to an order-wise improvement in $\frac{1}{1-\gamma}$. Analogous results are derived for finite-horizon MDPs as well. Notably, our sample complexity analysis unveils the effectiveness of vanilla Q-learning, which matches that of speedy Q-learning without requiring extra computation and storage. Our result is obtained by identifying novel error decompositions and recursion relations, which might shed light on how to study other variants of Q-learning.

----

## [575] Winograd Algorithm for AdderNet

**Authors**: *Wenshuo Li, Hanting Chen, Mingqiang Huang, Xinghao Chen, Chunjing Xu, Yunhe Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21c.html](http://proceedings.mlr.press/v139/li21c.html)

**Abstract**:

Adder neural network (AdderNet) is a new kind of deep model that replaces the original massive multiplications in convolutions by additions while preserving the high performance. Since the hardware complexity of additions is much lower than that of multiplications, the overall energy consumption is thus reduced significantly. To further optimize the hardware overhead of using AdderNet, this paper studies the winograd algorithm, which is a widely used fast algorithm for accelerating convolution and saving the computational costs. Unfortunately, the conventional Winograd algorithm cannot be directly applied to AdderNets since the distributive law in multiplication is not valid for the l1-norm. Therefore, we replace the element-wise multiplication in the Winograd equation by additions and then develop a new set of transform matrixes that can enhance the representation ability of output features to maintain the performance. Moreover, we propose the l2-to-l1 training strategy to mitigate the negative impacts caused by formal inconsistency. Experimental results on both FPGA and benchmarks show that the new method can further reduce the energy consumption without affecting the accuracy of the original AdderNet.

----

## [576] A Free Lunch From ANN: Towards Efficient, Accurate Spiking Neural Networks Calibration

**Authors**: *Yuhang Li, Shikuang Deng, Xin Dong, Ruihao Gong, Shi Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21d.html](http://proceedings.mlr.press/v139/li21d.html)

**Abstract**:

Spiking Neural Network (SNN) has been recognized as one of the next generation of neural networks. Conventionally, SNN can be converted from a pre-trained ANN by only replacing the ReLU activation to spike activation while keeping the parameters intact. Perhaps surprisingly, in this work we show that a proper way to calibrate the parameters during the conversion of ANN to SNN can bring significant improvements. We introduce SNN Calibration, a cheap but extraordinarily effective method by leveraging the knowledge within a pre-trained Artificial Neural Network (ANN). Starting by analyzing the conversion error and its propagation through layers theoretically, we propose the calibration algorithm that can correct the error layer-by-layer. The calibration only takes a handful number of training data and several minutes to finish. Moreover, our calibration algorithm can produce SNN with state-of-the-art architecture on the large-scale ImageNet dataset, including MobileNet and RegNet. Extensive experiments demonstrate the effectiveness and efficiency of our algorithm. For example, our advanced pipeline can increase up to 69% top-1 accuracy when converting MobileNet on ImageNet compared to baselines. Codes are released at https://github.com/yhhhli/SNN_Calibration.

----

## [577] Privacy-Preserving Feature Selection with Secure Multiparty Computation

**Authors**: *Xiling Li, Rafael Dowsley, Martine De Cock*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21e.html](http://proceedings.mlr.press/v139/li21e.html)

**Abstract**:

Existing work on privacy-preserving machine learning with Secure Multiparty Computation (MPC) is almost exclusively focused on model training and on inference with trained models, thereby overlooking the important data pre-processing stage. In this work, we propose the first MPC based protocol for private feature selection based on the filter method, which is independent of model training, and can be used in combination with any MPC protocol to rank features. We propose an efficient feature scoring protocol based on Gini impurity to this end. To demonstrate the feasibility of our approach for practical data science, we perform experiments with the proposed MPC protocols for feature selection in a commonly used machine-learning-as-a-service configuration where computations are outsourced to multiple servers, with semi-honest and with malicious adversaries. Regarding effectiveness, we show that secure feature selection with the proposed protocols improves the accuracy of classifiers on a variety of real-world data sets, without leaking information about the feature values or even which features were selected. Regarding efficiency, we document runtimes ranging from several seconds to an hour for our protocols to finish, depending on the size of the data set and the security settings.

----

## [578] Theory of Spectral Method for Union of Subspaces-Based Random Geometry Graph

**Authors**: *Gen Li, Yuantao Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21f.html](http://proceedings.mlr.press/v139/li21f.html)

**Abstract**:

Spectral method is a commonly used scheme to cluster data points lying close to Union of Subspaces, a task known as Subspace Clustering. The typical usage is to construct a Random Geometry Graph first and then apply spectral method to the graph to obtain clustering result. The latter step has been coined the name Spectral Clustering. As far as we know, in spite of the significance of both steps in spectral-method-based Subspace Clustering, all existing theoretical results focus on the first step of constructing the graph, but ignore the final step to correct false connections through spectral clustering. This paper establishes a theory to show the power of this method for the first time, in which we demonstrate the mechanism of spectral clustering by analyzing a simplified algorithm under the widely used semi-random model. Based on this theory, we prove the efficiency of Subspace Clustering in fairly broad conditions. The insights and analysis techniques developed in this paper might also have implications for other random graph problems.

----

## [579] MURAL: Meta-Learning Uncertainty-Aware Rewards for Outcome-Driven Reinforcement Learning

**Authors**: *Kevin Li, Abhishek Gupta, Ashwin Reddy, Vitchyr H. Pong, Aurick Zhou, Justin Yu, Sergey Levine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21g.html](http://proceedings.mlr.press/v139/li21g.html)

**Abstract**:

Exploration in reinforcement learning is, in general, a challenging problem. A common technique to make learning easier is providing demonstrations from a human supervisor, but such demonstrations can be expensive and time-consuming to acquire. In this work, we study a more tractable class of reinforcement learning problems defined simply by examples of successful outcome states, which can be much easier to provide while still making the exploration problem more tractable. In this problem setting, the reward function can be obtained automatically by training a classifier to categorize states as successful or not. However, as we will show, this requires the classifier to make uncertainty-aware predictions that are very difficult using standard techniques for training deep networks. To address this, we propose a novel mechanism for obtaining calibrated uncertainty based on an amortized technique for computing the normalized maximum likelihood (NML) distribution, leveraging tools from meta-learning to make this distribution tractable. We show that the resulting algorithm has a number of intriguing connections to both count-based exploration methods and prior algorithms for learning reward functions, while also providing more effective guidance towards the goal. We demonstrate that our algorithm solves a number of challenging navigation and robotic manipulation tasks which prove difficult or impossible for prior methods.

----

## [580] Ditto: Fair and Robust Federated Learning Through Personalization

**Authors**: *Tian Li, Shengyuan Hu, Ahmad Beirami, Virginia Smith*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21h.html](http://proceedings.mlr.press/v139/li21h.html)

**Abstract**:

Fairness and robustness are two important concerns for federated learning systems. In this work, we identify that robustness to data and model poisoning attacks and fairness, measured as the uniformity of performance across devices, are competing constraints in statistically heterogeneous networks. To address these constraints, we propose employing a simple, general framework for personalized federated learning, Ditto, that can inherently provide fairness and robustness benefits, and develop a scalable solver for it. Theoretically, we analyze the ability of Ditto to achieve fairness and robustness simultaneously on a class of linear problems. Empirically, across a suite of federated datasets, we show that Ditto not only achieves competitive performance relative to recent personalization methods, but also enables more accurate, robust, and fair models relative to state-of-the-art fair or robust baselines.

----

## [581] Quantization Algorithms for Random Fourier Features

**Authors**: *Xiaoyun Li, Ping Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21i.html](http://proceedings.mlr.press/v139/li21i.html)

**Abstract**:

The method of random projection (RP) is the standard technique for dimensionality reduction, approximate near neighbor search, compressed sensing, etc., which provides a simple and effective scheme for approximating pairwise inner products and Euclidean distances in massive data. Closely related to RP, the method of random Fourier features (RFF) has also become popular for approximating the (nonlinear) Gaussian kernel. RFF applies a specific nonlinear transformation on the projected data from RP. In practice, using the Gaussian kernel often leads to better performance than the linear kernel (inner product). After random projections, quantization is an important step for efficient data storage, computation and transmission. Quantization for RP has been extensively studied in the literature. In this paper, we focus on developing quantization algorithms for RFF. The task is in a sense challenging due to the tuning parameter $\gamma$ in the Gaussian kernel. For example, the quantizer and the quantized data might be tied to each specific Gaussian kernel parameter $\gamma$. Our contribution begins with the analysis on the probability distributions of RFF, and an interesting discovery that the marginal distribution of RFF is free of the parameter $\gamma$. This significantly simplifies the design of the Lloyd-Max (LM) quantization scheme for RFF in that there would be only one LM quantizer (regardless of $\gamma$). Detailed theoretical analysis is provided on the kernel estimators and approximation error, and experiments confirm the effectiveness and efficiency of the proposed method.

----

## [582] Approximate Group Fairness for Clustering

**Authors**: *Bo Li, Lijun Li, Ankang Sun, Chenhao Wang, Yingfan Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21j.html](http://proceedings.mlr.press/v139/li21j.html)

**Abstract**:

We incorporate group fairness into the algorithmic centroid clustering problem, where $k$ centers are to be located to serve $n$ agents distributed in a metric space. We refine the notion of proportional fairness proposed in [Chen et al., ICML 2019] as {\em core fairness}. A $k$-clustering is in the core if no coalition containing at least $n/k$ agents can strictly decrease their total distance by deviating to a new center together. Our solution concept is motivated by the situation where agents are able to coordinate and utilities are transferable. A string of existence, hardness and approximability results is provided. Particularly, we propose two dimensions to relax core requirements: one is on the degree of distance improvement, and the other is on the size of deviating coalition. For both relaxations and their combination, we study the extent to which relaxed core fairness can be satisfied in metric spaces including line, tree and general metric space, and design approximation algorithms accordingly. We also conduct experiments on synthetic and real-world data to examine the performance of our algorithms.

----

## [583] Sharper Generalization Bounds for Clustering

**Authors**: *Shaojie Li, Yong Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21k.html](http://proceedings.mlr.press/v139/li21k.html)

**Abstract**:

Existing generalization analysis of clustering mainly focuses on specific instantiations, such as (kernel) $k$-means, and a unified framework for studying clustering performance is still lacking. Besides, the existing excess clustering risk bounds are mostly of order $\mathcal{O}(K/\sqrt{n})$ provided that the underlying distribution has bounded support, where $n$ is the sample size and $K$ is the cluster numbers, or of order $\mathcal{O}(K^2/n)$ under strong assumptions on the underlying distribution, where these assumptions are hard to be verified in general. In this paper, we propose a unified clustering learning framework and investigate its excess risk bounds, obtaining state-of-the-art upper bounds under mild assumptions. Specifically, we derive sharper bounds of order $\mathcal{O}(K^2/n)$ under mild assumptions on the covering number of the hypothesis spaces, where these assumptions are easy to be verified. Moreover, for the hard clustering scheme, such as (kernel) $k$-means, if just assume the hypothesis functions to be bounded, we improve the upper bounds from the order $\mathcal{O}(K/\sqrt{n})$ to $\mathcal{O}(\sqrt{K}/\sqrt{n})$. Furthermore, state-of-the-art bounds of faster order $\mathcal{O}(K/n)$ are obtained with the covering number assumptions.

----

## [584] Provably End-to-end Label-noise Learning without Anchor Points

**Authors**: *Xuefeng Li, Tongliang Liu, Bo Han, Gang Niu, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21l.html](http://proceedings.mlr.press/v139/li21l.html)

**Abstract**:

In label-noise learning, the transition matrix plays a key role in building statistically consistent classifiers. Existing consistent estimators for the transition matrix have been developed by exploiting anchor points. However, the anchor-point assumption is not always satisfied in real scenarios. In this paper, we propose an end-to-end framework for solving label-noise learning without anchor points, in which we simultaneously optimize two objectives: the cross entropy loss between the noisy label and the predicted probability by the neural network, and the volume of the simplex formed by the columns of the transition matrix. Our proposed framework can identify the transition matrix if the clean class-posterior probabilities are sufficiently scattered. This is by far the mildest assumption under which the transition matrix is provably identifiable and the learned classifier is statistically consistent. Experimental results on benchmark datasets demonstrate the effectiveness and robustness of the proposed method.

----

## [585] A Novel Method to Solve Neural Knapsack Problems

**Authors**: *Duanshun Li, Jing Liu, Dongeun Lee, Ali Seyedmazloom, Giridhar Kaushik, Kookjin Lee, Noseong Park*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21m.html](http://proceedings.mlr.press/v139/li21m.html)

**Abstract**:

0-1 knapsack is of fundamental importance across many fields. In this paper, we present a game-theoretic method to solve 0-1 knapsack problems (KPs) where the number of items (products) is large and the values of items are not predetermined but decided by an external value assignment function (e.g., a neural network in our case) during the optimization process. While existing papers are interested in predicting solutions with neural networks for classical KPs whose objective functions are mostly linear functions, we are interested in solving KPs whose objective functions are neural networks. In other words, we choose a subset of items that maximize the sum of the values predicted by neural networks. Its key challenge is how to optimize the neural network-based non-linear KP objective with a budget constraint. Our solution is inspired by game-theoretic approaches in deep learning, e.g., generative adversarial networks. After formally defining our two-player game, we develop an adaptive gradient ascent method to solve it. In our experiments, our method successfully solves two neural network-based non-linear KPs and conventional linear KPs with 1 million items.

----

## [586] Mixed Cross Entropy Loss for Neural Machine Translation

**Authors**: *Haoran Li, Wei Lu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21n.html](http://proceedings.mlr.press/v139/li21n.html)

**Abstract**:

In neural machine translation, Cross Entropy loss (CE) is the standard loss function in two training methods of auto-regressive models, i.e., teacher forcing and scheduled sampling. In this paper, we propose mixed Cross Entropy loss (mixed CE) as a substitute for CE in both training approaches. In teacher forcing, the model trained with CE regards the translation problem as a one-to-one mapping process, while in mixed CE this process can be relaxed to one-to-many. In scheduled sampling, we show that mixed CE has the potential to encourage the training and testing behaviours to be similar to each other, more effectively mitigating the exposure bias problem. We demonstrate the superiority of mixed CE over CE on several machine translation datasets, WMT’16 Ro-En, WMT’16 Ru-En, and WMT’14 En-De in both teacher forcing and scheduled sampling setups. Furthermore, in WMT’14 En-De, we also find mixed CE consistently outperforms CE on a multi-reference set as well as a challenging paraphrased reference set. We also found the model trained with mixed CE is able to provide a better probability distribution defined over the translation output space. Our code is available at https://github.com/haorannlp/mix.

----

## [587] Training Graph Neural Networks with 1000 Layers

**Authors**: *Guohao Li, Matthias Müller, Bernard Ghanem, Vladlen Koltun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21o.html](http://proceedings.mlr.press/v139/li21o.html)

**Abstract**:

Deep graph neural networks (GNNs) have achieved excellent results on various tasks on increasingly large graph datasets with millions of nodes and edges. However, memory complexity has become a major obstacle when training deep GNNs for practical applications due to the immense number of nodes, edges, and intermediate activations. To improve the scalability of GNNs, prior works propose smart graph sampling or partitioning strategies to train GNNs with a smaller set of nodes or sub-graphs. In this work, we study reversible connections, group convolutions, weight tying, and equilibrium models to advance the memory and parameter efficiency of GNNs. We find that reversible connections in combination with deep network architectures enable the training of overparameterized GNNs that significantly outperform existing methods on multiple datasets. Our models RevGNN-Deep (1001 layers with 80 channels each) and RevGNN-Wide (448 layers with 224 channels each) were both trained on a single commodity GPU and achieve an ROC-AUC of 87.74 $\pm$ 0.13 and 88.14 $\pm$ 0.15 on the ogbn-proteins dataset. To the best of our knowledge, RevGNN-Deep is the deepest GNN in the literature by one order of magnitude.

----

## [588] Active Feature Acquisition with Generative Surrogate Models

**Authors**: *Yang Li, Junier Oliva*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21p.html](http://proceedings.mlr.press/v139/li21p.html)

**Abstract**:

Many real-world situations allow for the acquisition of additional relevant information when making an assessment with limited or uncertain data. However, traditional ML approaches either require all features to be acquired beforehand or regard part of them as missing data that cannot be acquired. In this work, we consider models that perform active feature acquisition (AFA) and query the environment for unobserved features to improve the prediction assessments at evaluation time. Our work reformulates the Markov decision process (MDP) that underlies the AFA problem as a generative modeling task and optimizes a policy via a novel model-based approach. We propose learning a generative surrogate model (GSM) that captures the dependencies among input features to assess potential information gain from acquisitions. The GSM is leveraged to provide intermediate rewards and auxiliary information to aid the agent navigate a complicated high-dimensional action space and sparse rewards. Furthermore, we extend AFA in a task we coin active instance recognition (AIR) for the unsupervised case where the target variables are the unobserved features themselves and the goal is to collect information for a particular instance in a cost-efficient way. Empirical results demonstrate that our approach achieves considerably better performance than previous state of the art methods on both supervised and unsupervised tasks.

----

## [589] Partially Observed Exchangeable Modeling

**Authors**: *Yang Li, Junier Oliva*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21q.html](http://proceedings.mlr.press/v139/li21q.html)

**Abstract**:

Modeling dependencies among features is fundamental for many machine learning tasks. Although there are often multiple related instances that may be leveraged to inform conditional dependencies, typical approaches only model conditional dependencies over individual instances. In this work, we propose a novel framework, partially observed exchangeable modeling (POEx) that takes in a set of related partially observed instances and infers the conditional distribution for the unobserved dimensions over multiple elements. Our approach jointly models the intra-instance (among features in a point) and inter-instance (among multiple points in a set) dependencies in data. POEx is a general framework that encompasses many existing tasks such as point cloud expansion and few-shot generation, as well as new tasks like few-shot imputation. Despite its generality, extensive empirical evaluations show that our model achieves state-of-the-art performance across a range of applications.

----

## [590] Testing DNN-based Autonomous Driving Systems under Critical Environmental Conditions

**Authors**: *Zhong Li, Minxue Pan, Tian Zhang, Xuandong Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21r.html](http://proceedings.mlr.press/v139/li21r.html)

**Abstract**:

Due to the increasing usage of Deep Neural Network (DNN) based autonomous driving systems (ADS) where erroneous or unexpected behaviours can lead to catastrophic accidents, testing such systems is of growing importance. Existing approaches often just focus on finding erroneous behaviours and have not thoroughly studied the impact of environmental conditions. In this paper, we propose to test DNN-based ADS under different environmental conditions to identify the critical ones, that is, the environmental conditions under which the ADS are more prone to errors. To tackle the problem of the space of environmental conditions being extremely large, we present a novel approach named TACTIC that employs the search-based method to identify critical environmental conditions generated by an image-to-image translation model. Large-scale experiments show that TACTIC can effectively identify critical environmental conditions and produce realistic testing images, and meanwhile, reveal more erroneous behaviours compared to existing approaches.

----

## [591] The Symmetry between Arms and Knapsacks: A Primal-Dual Approach for Bandits with Knapsacks

**Authors**: *Xiaocheng Li, Chunlin Sun, Yinyu Ye*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21s.html](http://proceedings.mlr.press/v139/li21s.html)

**Abstract**:

In this paper, we study the bandits with knapsacks (BwK) problem and develop a primal-dual based algorithm that achieves a problem-dependent logarithmic regret bound. The BwK problem extends the multi-arm bandit (MAB) problem to model the resource consumption, and the existing BwK literature has been mainly focused on deriving asymptotically optimal distribution-free regret bounds. We first study the primal and dual linear programs underlying the BwK problem. From this primal-dual perspective, we discover symmetry between arms and knapsacks, and then propose a new notion of suboptimality measure for the BwK problem. The suboptimality measure highlights the important role of knapsacks in determining algorithm regret and inspires the design of our two-phase algorithm. In the first phase, the algorithm identifies the optimal arms and the binding knapsacks, and in the second phase, it exhausts the binding knapsacks via playing the optimal arms through an adaptive procedure. Our regret upper bound involves the proposed suboptimality measure and it has a logarithmic dependence on length of horizon $T$ and a polynomial dependence on $m$ (the numbers of arms) and $d$ (the number of knapsacks). To the best of our knowledge, this is the first problem-dependent logarithmic regret bound for solving the general BwK problem.

----

## [592] Distributionally Robust Optimization with Markovian Data

**Authors**: *Mengmeng Li, Tobias Sutter, Daniel Kuhn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21t.html](http://proceedings.mlr.press/v139/li21t.html)

**Abstract**:

We study a stochastic program where the probability distribution of the uncertain problem parameters is unknown and only indirectly observed via finitely many correlated samples generated by an unknown Markov chain with $d$ states. We propose a data-driven distributionally robust optimization model to estimate the problem’s objective function and optimal solution. By leveraging results from large deviations theory, we derive statistical guarantees on the quality of these estimators. The underlying worst-case expectation problem is nonconvex and involves $\mathcal O(d^2)$ decision variables. Thus, it cannot be solved efficiently for large $d$. By exploiting the structure of this problem, we devise a customized Frank-Wolfe algorithm with convex direction-finding subproblems of size $\mathcal O(d)$. We prove that this algorithm finds a stationary point efficiently under mild conditions. The efficiency of the method is predicated on a dimensionality reduction enabled by a dual reformulation. Numerical experiments indicate that our approach has better computational and statistical properties than the state-of-the-art methods.

----

## [593] Communication-Efficient Distributed SVD via Local Power Iterations

**Authors**: *Xiang Li, Shusen Wang, Kun Chen, Zhihua Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21u.html](http://proceedings.mlr.press/v139/li21u.html)

**Abstract**:

We study distributed computing of the truncated singular value decomposition (SVD). We develop an algorithm that we call \texttt{LocalPower} for improving communication efficiency. Specifically, we uniformly partition the dataset among $m$ nodes and alternate between multiple (precisely $p$) local power iterations and one global aggregation. In the aggregation, we propose to weight each local eigenvector matrix with orthogonal Procrustes transformation (OPT). As a practical surrogate of OPT, sign-fixing, which uses a diagonal matrix with $\pm 1$ entries as weights, has better computation complexity and stability in experiments. We theoretically show that under certain assumptions \texttt{LocalPower} lowers the required number of communications by a factor of $p$ to reach a constant accuracy. We also show that the strategy of periodically decaying $p$ helps obtain high-precision solutions. We conduct experiments to demonstrate the effectiveness of \texttt{LocalPower}.

----

## [594] FILTRA: Rethinking Steerable CNN by Filter Transform

**Authors**: *Bo Li, Qili Wang, Gim Hee Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21v.html](http://proceedings.mlr.press/v139/li21v.html)

**Abstract**:

Steerable CNN imposes the prior knowledge of transformation invariance or equivariance in the network architecture to enhance the the network robustness on geometry transformation of data and reduce overfitting. It has been an intuitive and widely used technique to construct a steerable filter by augmenting a filter with its transformed copies in the past decades, which is named as filter transform in this paper. Recently, the problem of steerable CNN has been studied from aspect of group representation theory, which reveals the function space structure of a steerable kernel function. However, it is not yet clear on how this theory is related to the filter transform technique. In this paper, we show that kernel constructed by filter transform can also be interpreted in the group representation theory. This interpretation help complete the puzzle of steerable CNN theory and provides a novel and simple approach to implement steerable convolution operators. Experiments are executed on multiple datasets to verify the feasibility of the proposed approach.

----

## [595] Online Unrelated Machine Load Balancing with Predictions Revisited

**Authors**: *Shi Li, Jiayi Xian*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21w.html](http://proceedings.mlr.press/v139/li21w.html)

**Abstract**:

We study the online load balancing problem with machine learned predictions, and give results that improve upon and extend those in a recent paper by Lattanzi et al. (2020). First, we design deterministic and randomized online rounding algorithms for the problem in the unrelated machine setting, with $O(\frac{\log m}{\log \log m})$- and $O(\frac{\log \log m}{\log \log \log m})$-competitive ratios. They respectively improve upon the previous ratios of $O(\log m)$ and $O(\log^3\log m)$, and match the lower bounds given by Lattanzi et al. Second, we extend their prediction scheme from the identical machine restricted assignment setting to the unrelated machine setting. With the knowledge of two vectors over machines, a dual vector and a weight vector, we can construct a good fractional assignment online, that can be passed to an online rounding algorithm. Finally, we consider the learning model introduced by Lavastida et al. (2020), and show that under the model, the two vectors can be learned efficiently with a few samples of instances.

----

## [596] Asymptotic Normality and Confidence Intervals for Prediction Risk of the Min-Norm Least Squares Estimator

**Authors**: *Zeng Li, Chuanlong Xie, Qinwen Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21x.html](http://proceedings.mlr.press/v139/li21x.html)

**Abstract**:

This paper quantifies the uncertainty of prediction risk for the min-norm least squares estimator in high-dimensional linear regression models. We establish the asymptotic normality of prediction risk when both the sample size and the number of features tend to infinity. Based on the newly established central limit theorems(CLTs), we derive the confidence intervals of the prediction risk under various scenarios. Our results demonstrate the sample-wise non-monotonicity of the prediction risk and confirm “more data hurt" phenomenon. Furthermore, the width of confidence intervals indicates that over-parameterization would enlarge the randomness of prediction performance.

----

## [597] TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models

**Authors**: *Zhuohan Li, Siyuan Zhuang, Shiyuan Guo, Danyang Zhuo, Hao Zhang, Dawn Song, Ion Stoica*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21y.html](http://proceedings.mlr.press/v139/li21y.html)

**Abstract**:

Model parallelism has become a necessity for training modern large-scale deep language models. In this work, we identify a new and orthogonal dimension from existing model parallel approaches: it is possible to perform pipeline parallelism within a single training sequence for Transformer-based language models thanks to its autoregressive property. This enables a more fine-grained pipeline compared with previous work. With this key idea, we design TeraPipe, a high-performance token-level pipeline parallel algorithm for synchronous model-parallel training of Transformer-based language models. We develop a novel dynamic programming-based algorithm to calculate the optimal pipelining execution scheme given a specific model and cluster configuration. We show that TeraPipe can speed up the training by 5.0x for the largest GPT-3 model with 175 billion parameters on an AWS cluster with 48 p3.16xlarge instances compared with state-of-the-art model-parallel methods. The code for reproduction can be found at https://github.com/zhuohan123/terapipe

----

## [598] A Second look at Exponential and Cosine Step Sizes: Simplicity, Adaptivity, and Performance

**Authors**: *Xiaoyu Li, Zhenxun Zhuang, Francesco Orabona*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/li21z.html](http://proceedings.mlr.press/v139/li21z.html)

**Abstract**:

Stochastic Gradient Descent (SGD) is a popular tool in training large-scale machine learning models. Its performance, however, is highly variable, depending crucially on the choice of the step sizes. Accordingly, a variety of strategies for tuning the step sizes have been proposed, ranging from coordinate-wise approaches (a.k.a. “adaptive” step sizes) to sophisticated heuristics to change the step size in each iteration. In this paper, we study two step size schedules whose power has been repeatedly confirmed in practice: the exponential and the cosine step sizes. For the first time, we provide theoretical support for them proving convergence rates for smooth non-convex functions, with and without the Polyak-Ł{}ojasiewicz (PL) condition. Moreover, we show the surprising property that these two strategies are \emph{adaptive} to the noise level in the stochastic gradients of PL functions. That is, contrary to polynomial step sizes, they achieve almost optimal performance without needing to know the noise level nor tuning their hyperparameters based on it. Finally, we conduct a fair and comprehensive empirical evaluation of real-world datasets with deep learning architectures. Results show that, even if only requiring at most two hyperparameters to tune, these two strategies best or match the performance of various finely-tuned state-of-the-art strategies.

----

## [599] Towards Understanding and Mitigating Social Biases in Language Models

**Authors**: *Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, Ruslan Salakhutdinov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liang21a.html](http://proceedings.mlr.press/v139/liang21a.html)

**Abstract**:

As machine learning methods are deployed in real-world settings such as healthcare, legal systems, and social science, it is crucial to recognize how they shape social biases and stereotypes in these sensitive decision-making processes. Among such real-world deployments are large-scale pretrained language models (LMs) that can be potentially dangerous in manifesting undesirable representational biases - harmful biases resulting from stereotyping that propagate negative generalizations involving gender, race, religion, and other social constructs. As a step towards improving the fairness of LMs, we carefully define several sources of representational biases before proposing new benchmarks and metrics to measure them. With these tools, we propose steps towards mitigating social biases during text generation. Our empirical results and human evaluation demonstrate effectiveness in mitigating bias while retaining crucial contextual information for high-fidelity text generation, thereby pushing forward the performance-fairness Pareto frontier.

----



[Go to the previous page](ICML-2021-list02.md)

[Go to the next page](ICML-2021-list04.md)

[Go to the catalog section](README.md)