## [400] Random Matrix Analysis to Balance between Supervised and Unsupervised Learning under the Low Density Separation Assumption

**Authors**: *Vasilii Feofanov, Malik Tiomoko, Aladin Virmaux*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/feofanov23a.html](https://proceedings.mlr.press/v202/feofanov23a.html)

**Abstract**:

We propose a theoretical framework to analyze semi-supervised classification under the low density separation assumption in a high-dimensional regime. In particular, we introduce QLDS, a linear classification model, where the low density separation assumption is implemented via quadratic margin maximization. The algorithm has an explicit solution with rich theoretical properties, and we show that particular cases of our algorithm are the least-square support vector machine in the supervised case, the spectral clustering in the fully unsupervised regime, and a class of semi-supervised graph-based approaches. As such, QLDS establishes a smooth bridge between these supervised and unsupervised learning methods. Using recent advances in the random matrix theory, we formally derive a theoretical evaluation of the classification error in the asymptotic regime. As an application, we derive a hyperparameter selection policy that finds the best balance between the supervised and the unsupervised terms of our learning criterion. Finally, we provide extensive illustrations of our framework, as well as an experimental study on several benchmarks to demonstrate that QLDS, while being computationally more efficient, improves over cross-validation for hyperparameter selection, indicating a high promise of the usage of random matrix theory for semi-supervised model selection.

----

## [401] SurCo: Learning Linear SURrogates for COmbinatorial Nonlinear Optimization Problems

**Authors**: *Aaron M. Ferber, Taoan Huang, Daochen Zha, Martin Schubert, Benoit Steiner, Bistra Dilkina, Yuandong Tian*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ferber23a.html](https://proceedings.mlr.press/v202/ferber23a.html)

**Abstract**:

Optimization problems with nonlinear cost functions and combinatorial constraints appear in many real-world applications but remain challenging to solve efficiently compared to their linear counterparts. To bridge this gap, we propose $\textbf{\emph{\texttt{SurCo}}}$ that learns linear $\underline{\text{Sur}}$rogate costs which can be used in existing $\underline{\text{Co}}$mbinatorial solvers to output good solutions to the original nonlinear combinatorial optimization problem. The surrogate costs are learned end-to-end with nonlinear loss by differentiating through the linear surrogate solver, combining the flexibility of gradient-based methods with the structure of linear combinatorial optimization. We propose three $\texttt{SurCo}$ variants: $\texttt{SurCo}-\texttt{zero}$ for individual nonlinear problems, $\texttt{SurCo}-\texttt{prior}$ for problem distributions, and $\texttt{SurCo}-\texttt{hybrid}$ to combine both distribution and problem-specific information. We give theoretical intuition motivating $\texttt{SurCo}$, and evaluate it empirically. Experiments show that $\texttt{SurCo}$ finds better solutions faster than state-of-the-art and domain expert approaches in real-world optimization problems such as embedding table sharding, inverse photonic design, and nonlinear route planning.

----

## [402] Scaling Laws for Multilingual Neural Machine Translation

**Authors**: *Patrick Fernandes, Behrooz Ghorbani, Xavier Garcia, Markus Freitag, Orhan Firat*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fernandes23a.html](https://proceedings.mlr.press/v202/fernandes23a.html)

**Abstract**:

In this work, we provide a large-scale empirical study of the scaling properties of multilingual neural machine translation models. We examine how increases in the model size affect the model performance and investigate the role of the individual language pair weights on the scaling behavior. We find that these weights only affect the multiplicative factor of the scaling law, and in particular, the scaling exponent is unaffected by them. Through a novel joint scaling law formulation, we compute the effective number of parameters allocated to each language pair and examine the role of language similarity in the scaling behavior of our models. We find little evidence that language similarity has any impact. In contrast, “direction” of the multilinguality plays a significant role, with models translating from multiple languages into English having a larger number of effective parameters per task than their reversed counterparts. Finally, we leverage our observations to predict the performance of multilingual models trained with any language weighting at any scale, greatly reducing efforts required for language balancing in large multilingual models. Our findings apply to both in-domain and out-of-domain test sets and to multiple evaluation metrics, such as ChrF and BLEURT.

----

## [403] Constant Matters: Fine-grained Error Bound on Differentially Private Continual Observation

**Authors**: *Hendrik Fichtenberger, Monika Henzinger, Jalaj Upadhyay*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fichtenberger23a.html](https://proceedings.mlr.press/v202/fichtenberger23a.html)

**Abstract**:

We study fine-grained error bounds for differentially private algorithms for counting under continual observation. Our main insight is that the matrix mechanism when using lower-triangular matrices can be used in the continual observation model. More specifically, we give an explicit factorization for the counting matrix $M_\mathsf{count}$ and upper bound the error explicitly. We also give a fine-grained analysis, specifying the exact constant in the upper bound. Our analysis is based on upper and lower bounds of the completely bounded norm (cb-norm) of $M_\mathsf{count}$. Along the way, we improve the best-known bound of 28 years by Mathias (SIAM Journal on Matrix Analysis and Applications, 1993) on the cb-norm of $M_\mathsf{count}$ for a large range of the dimension of $M_\mathsf{count}$. Furthermore, we are the first to give concrete error bounds for various problems under continual observation such as binary counting, maintaining a histogram, releasing an approximately cut-preserving synthetic graph, many graph-based statistics, and substring and episode counting. Finally, we note that our result can be used to get a fine-grained error bound for non-interactive local learning and the first lower bounds on the additive error for $(\epsilon,\delta)$-differentially-private counting under continual observation. Subsequent to this work, Henzinger et al. (SODA, 2023) showed that our factorization also achieves fine-grained mean-squared error.

----

## [404] Adapting to game trees in zero-sum imperfect information games

**Authors**: *Côme Fiegel, Pierre Ménard, Tadashi Kozuno, Rémi Munos, Vianney Perchet, Michal Valko*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fiegel23a.html](https://proceedings.mlr.press/v202/fiegel23a.html)

**Abstract**:

Imperfect information games (IIG) are games in which each player only partially observes the current game state. We study how to learn $\epsilon$-optimal strategies in a zero-sum IIG through self-play with trajectory feedback. We give a problem-independent lower bound $\widetilde{\mathcal{O}}(H(A_{\mathcal{X}}+B_{\mathcal{Y}})/\epsilon^2)$ on the required number of realizations to learn these strategies with high probability, where $H$ is the length of the game, $A_{\mathcal{X}}$ and $B_{\mathcal{Y}}$ are the total number of actions for the two players. We also propose two Follow the Regularized leader (FTRL) algorithms for this setting: Balanced FTRL which matches this lower bound, but requires the knowledge of the information set structure beforehand to define the regularization; and Adaptive FTRL which needs $\widetilde{\mathcal{O}}(H^2(A_{\mathcal{X}}+B_{\mathcal{Y}})/\epsilon^2)$ realizations without this requirement by progressively adapting the regularization to the observations.

----

## [405] User-defined Event Sampling and Uncertainty Quantification in Diffusion Models for Physical Dynamical Systems

**Authors**: *Marc Anton Finzi, Anudhyan Boral, Andrew Gordon Wilson, Fei Sha, Leonardo Zepeda-Núñez*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/finzi23a.html](https://proceedings.mlr.press/v202/finzi23a.html)

**Abstract**:

Diffusion models are a class of probabilistic generative models that have been widely used as a prior for image processing tasks like text conditional generation and inpainting. We demonstrate that these models can be adapted to make predictions and provide uncertainty quantification for chaotic dynamical systems. In these applications, diffusion models can implicitly represent knowledge about outliers and extreme events; however, querying that knowledge through conditional sampling or measuring probabilities is surprisingly difficult. Existing methods for conditional sampling at inference time seek mainly to enforce the constraints, which is insufficient to match the statistics of the distribution or compute the probability of the chosen events. To achieve these ends, optimally one would use the conditional score function, but its computation is typically intractable. In this work, we develop a probabilistic approximation scheme for the conditional score function which provably converges to the true distribution as the noise level decreases. With this scheme we are able to sample conditionally on nonlinear user-defined events at inference time, and matches data statistics even when sampling from the tails of the distribution.

----

## [406] ACAT: Adversarial Counterfactual Attention for Classification and Detection in Medical Imaging

**Authors**: *Alessandro Fontanella, Antreas Antoniou, Wenwen Li, Joanna M. Wardlaw, Grant Mair, Emanuele Trucco, Amos J. Storkey*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fontanella23a.html](https://proceedings.mlr.press/v202/fontanella23a.html)

**Abstract**:

In some medical imaging tasks and other settings where only small parts of the image are informative for the classification task, traditional CNNs can sometimes struggle to generalise. Manually annotated Regions of Interest (ROI) are often used to isolate the most informative parts of the image. However, these are expensive to collect and may vary significantly across annotators. To overcome these issues, we propose a framework that employs saliency maps to obtain soft spatial attention masks that modulate the image features at different scales. We refer to our method as Adversarial Counterfactual Attention (ACAT). ACAT increases the baseline classification accuracy of lesions in brain CT scans from $71.39 %$ to $72.55 %$ and of COVID-19 related findings in lung CT scans from $67.71 %$ to $70.84 %$ and exceeds the performance of competing methods. We investigate the best way to generate the saliency maps employed in our architecture and propose a way to obtain them from adversarially generated counterfactual images. They are able to isolate the area of interest in brain and lung CT scans without using any manual annotations. In the task of localising the lesion location out of 6 possible regions, they obtain a score of $65.05 %$ on brain CT scans, improving the score of $61.29 %$ obtained with the best competing method.

----

## [407] Explainable Data-Driven Optimization: From Context to Decision and Back Again

**Authors**: *Alexandre Forel, Axel Parmentier, Thibaut Vidal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/forel23a.html](https://proceedings.mlr.press/v202/forel23a.html)

**Abstract**:

Data-driven optimization uses contextual information and machine learning algorithms to find solutions to decision problems with uncertain parameters. While a vast body of work is dedicated to interpreting machine learning models in the classification setting, explaining decision pipelines involving learning algorithms remains unaddressed. This lack of interpretability can block the adoption of data-driven solutions as practitioners may not understand or trust the recommended decisions. We bridge this gap by introducing a counterfactual explanation methodology tailored to explain solutions to data-driven problems. We introduce two classes of explanations and develop methods to find nearest explanations of random forest and nearest-neighbor predictors. We demonstrate our approach by explaining key problems in operations management such as inventory management and routing.

----

## [408] Hardness of Independent Learning and Sparse Equilibrium Computation in Markov Games

**Authors**: *Dylan J. Foster, Noah Golowich, Sham M. Kakade*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/foster23a.html](https://proceedings.mlr.press/v202/foster23a.html)

**Abstract**:

We consider the problem of decentralized multi-agent reinforcement learning in Markov games. A fundamental question is whether there exist algorithms that, when run independently by all agents, lead to no-regret for each player, analogous to celebrated convergence results for no-regret learning in normal-form games. While recent work has shown that such algorithms exist for restricted settings (notably, when regret is defined with respect to deviations to Markov policies), the question of whether independent no-regret learning can be achieved in the standard Markov game framework was open. We provide a decisive negative resolution to this problem, both from a computational and statistical perspective. We show that: • Under the complexity-theoretic assumption that PPAD $\neq$ P, there is no polynomial-time algorithm that attains no-regret in two-player general-sum Markov games when executed independently by all players, even when the game is known to the algorithm designer. • When the game is unknown, no algorithm, efficient or otherwise, can achieve no-regret without observing exponentially many episodes in the number of players. These results are proven via lower bounds for a simpler problem we refer to as SparseCCE, in which the goal is to compute a coarse correlated equilibrium that is “sparse” in the sense that it can be represented as a mixture of a small number of product policies.

----

## [409] Disentangled Generative Models for Robust Prediction of System Dynamics

**Authors**: *Stathi Fotiadis, Mario Lino Valencia, Shunlong Hu, Stef Garasto, Chris D. Cantwell, Anil Anthony Bharath*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fotiadis23a.html](https://proceedings.mlr.press/v202/fotiadis23a.html)

**Abstract**:

The use of deep neural networks for modelling system dynamics is increasingly popular, but long-term prediction accuracy and out-of-distribution generalization still present challenges. In this study, we address these challenges by considering the parameters of dynamical systems as factors of variation of the data and leverage their ground-truth values to disentangle the representations learned by generative models. Our experimental results in phase-space and observation-space dynamics, demonstrate the effectiveness of latent-space supervision in producing disentangled representations, leading to improved long-term prediction accuracy and out-of-distribution robustness.

----

## [410] Can Forward Gradient Match Backpropagation?

**Authors**: *Louis Fournier, Stéphane Rivaud, Eugene Belilovsky, Michael Eickenberg, Edouard Oyallon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fournier23a.html](https://proceedings.mlr.press/v202/fournier23a.html)

**Abstract**:

Forward Gradients - the idea of using directional derivatives in forward differentiation mode - have recently been shown to be utilizable for neural network training while avoiding problems generally associated with backpropagation gradient computation, such as locking and memorization requirements. The cost is the requirement to guess the step direction, which is hard in high dimensions. While current solutions rely on weighted averages over isotropic guess vector distributions, we propose to strongly bias our gradient guesses in directions that are much more promising, such as feedback obtained from small, local auxiliary networks. For a standard computer vision neural network, we conduct a rigorous study systematically covering a variety of combinations of gradient targets and gradient guesses, including those previously presented in the literature. We find that using gradients obtained from a local loss as a candidate direction drastically improves on random noise in Forward Gradient methods.

----

## [411] Last Switch Dependent Bandits with Monotone Payoff Functions

**Authors**: *Ayoub Foussoul, Vineet Goyal, Orestis Papadigenopoulos, Assaf Zeevi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/foussoul23a.html](https://proceedings.mlr.press/v202/foussoul23a.html)

**Abstract**:

In a recent work, Laforgue et al. introduce the model of last switch dependent (LSD) bandits, in an attempt to capture nonstationary phenomena induced by the interaction between the player and the environment. Examples include satiation, where consecutive plays of the same action lead to decreased performance, or deprivation, where the payoff of an action increases after an interval of inactivity. In this work, we take a step towards understanding the approximability of planning LSD bandits, namely, the (NP-hard) problem of computing an optimal arm-pulling strategy under complete knowledge of the model. In particular, we design the first efficient constant approximation algorithm for the problem and show that, under a natural monotonicity assumption on the payoffs, its approximation guarantee (almost) matches the state-of-the-art for the special and well-studied class of recharging bandits (also known as delay-dependent). In this attempt, we develop new tools and insights for this class of problems, including a novel higher-dimensional relaxation and the technique of mirroring the evolution of virtual states. We believe that these novel elements could potentially be used for approaching richer classes of action-induced nonstationary bandits (e.g., special instances of restless bandits). In the case where the model parameters are initially unknown, we develop an online learning adaptation of our algorithm for which we provide sublinear regret guarantees against its full-information counterpart.

----

## [412] A Theoretical Analysis of the Learning Dynamics under Class Imbalance

**Authors**: *Emanuele Francazi, Marco Baity-Jesi, Aurélien Lucchi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/francazi23a.html](https://proceedings.mlr.press/v202/francazi23a.html)

**Abstract**:

Data imbalance is a common problem in machine learning that can have a critical effect on the performance of a model. Various solutions exist but their impact on the convergence of the learning dynamics is not understood. Here, we elucidate the significant negative impact of data imbalance on learning, showing that the learning curves for minority and majority classes follow sub-optimal trajectories when training with a gradient-based optimizer. This slowdown is related to the imbalance ratio and can be traced back to a competition between the optimization of different classes. Our main contribution is the analysis of the convergence of full-batch (GD) and stochastic gradient descent (SGD), and of variants that renormalize the contribution of each per-class gradient. We find that GD is not guaranteed to decrease the loss for each class but that this problem can be addressed by performing a per-class normalization of the gradient. With SGD, class imbalance has an additional effect on the direction of the gradients: the minority class suffers from a higher directional noise, which reduces the effectiveness of the per-class gradient normalization. Our findings not only allow us to understand the potential and limitations of strategies involving the per-class gradients, but also the reason for the effectiveness of previously used solutions for class imbalancesuch as oversampling.

----

## [413] SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot

**Authors**: *Elias Frantar, Dan Alistarh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/frantar23a.html](https://proceedings.mlr.press/v202/frantar23a.html)

**Abstract**:

We show for the first time that large-scale generative pretrained transformer (GPT) family models can be pruned to at least 50% sparsity in one-shot, without any retraining, at minimal loss of accuracy. This is achieved via a new pruning method called SparseGPT, specifically designed to work efficiently and accurately on massive GPT-family models. We can execute SparseGPT on the largest available open-source models, OPT-175B and BLOOM-176B, in under 4.5 hours, and can reach 60% unstructured sparsity with negligible increase in perplexity: remarkably, more than 100 billion weights from these models can be ignored at inference time. SparseGPT generalizes to semi-structured (2:4 and 4:8) patterns, and is compatible with weight quantization approaches. The code is available at: https://github.com/IST-DASLab/sparsegpt.

----

## [414] Learning Temporally AbstractWorld Models without Online Experimentation

**Authors**: *Benjamin Freed, Siddarth Venkatraman, Guillaume Adrien Sartoretti, Jeff Schneider, Howie Choset*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/freed23a.html](https://proceedings.mlr.press/v202/freed23a.html)

**Abstract**:

Agents that can build temporally abstract representations of their environment are better able to understand their world and make plans on extended time scales, with limited computational power and modeling capacity. However, existing methods for automatically learning temporally abstract world models usually require millions of online environmental interactions and incentivize agents to reach every accessible environmental state, which is infeasible for most real-world robots both in terms of data efficiency and hardware safety. In this paper, we present an approach for simultaneously learning sets of skills and temporally abstract, skill-conditioned world models purely from offline data, enabling agents to perform zero-shot online planning of skill sequences for new tasks. We show that our approach performs comparably to or better than a wide array of state-of-the-art offline RL algorithms on a number of simulated robotics locomotion and manipulation benchmarks, while offering a higher degree of adaptability to new goals. Finally, we show that our approach offers a much higher degree of robustness to perturbations in environmental dynamics, compared to policy-based methods.

----

## [415] A Coupled Flow Approach to Imitation Learning

**Authors**: *Gideon Joseph Freund, Elad Sarafian, Sarit Kraus*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/freund23a.html](https://proceedings.mlr.press/v202/freund23a.html)

**Abstract**:

In reinforcement learning and imitation learning, an object of central importance is the state distribution induced by the policy. It plays a crucial role in the policy gradient theorem, and references to it–along with the related state-action distribution–can be found all across the literature. Despite its importance, the state distribution is mostly discussed indirectly and theoretically, rather than being modeled explicitly. The reason being an absence of appropriate density estimation tools. In this work, we investigate applications of a normalizing flow based model for the aforementioned distributions. In particular, we use a pair of flows coupled through the optimality point of the Donsker-Varadhan representation of the Kullback-Leibler (KL) divergence, for distribution matching based imitation learning. Our algorithm, Coupled Flow Imitation Learning (CFIL), achieves state-of-the-art performance on benchmark tasks with a single expert trajectory and extends naturally to a variety of other settings, including the subsampled and state-only regimes.

----

## [416] Simple Hardware-Efficient Long Convolutions for Sequence Modeling

**Authors**: *Daniel Y. Fu, Elliot L. Epstein, Eric Nguyen, Armin W. Thomas, Michael Zhang, Tri Dao, Atri Rudra, Christopher Ré*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23a.html](https://proceedings.mlr.press/v202/fu23a.html)

**Abstract**:

State space models (SSMs) have high performance on long sequence modeling but require sophisticated initialization techniques and specialized implementations for high quality and runtime performance. We study whether a simple alternative can match SSMs in performance and efficiency: directly learning long convolutions over the sequence. We find that a key requirement to achieving high performance is keeping the convolution kernels smooth. We find that simple interventions-such as squashing the kernel weights-result in smooth kernels and recover SSM performance on a range of tasks including the long range arena, image classification, language modeling, and brain data modeling. Next, we develop FlashButterfly, an IO-aware algorithm to improve the runtime performance of long convolutions. FlashButterfly appeals to classic Butterfly decompositions of the convolution to reduce GPU memory IO and increase FLOP utilization. FlashButterfly speeds up convolutions by 2.2$\times$, and allows us to train on Path256, a challenging task with sequence length 64K, where we set state-of-the-art by 29.1 points while training 7.2$\times$ faster than prior work. Lastly, we introduce an extension to FlashButterfly that learns the coefficients of the Butterfly decomposition, increasing expressivity without increasing runtime. Using this extension, we outperform a Transformer on WikiText103 by 0.2 PPL with 30% fewer parameters.

----

## [417] MonoNeRF: Learning Generalizable NeRFs from Monocular Videos without Camera Poses

**Authors**: *Yang Fu, Ishan Misra, Xiaolong Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23b.html](https://proceedings.mlr.press/v202/fu23b.html)

**Abstract**:

We propose a generalizable neural radiance fields - MonoNeRF, that can be trained on large-scale monocular videos of moving in static scenes without any ground-truth annotations of depth and camera poses. MonoNeRF follows an Autoencoder-based architecture, where the encoder estimates the monocular depth and the camera pose, and the decoder constructs a Multiplane NeRF representation based on the depth encoder feature, and renders the input frames with the estimated camera. The learning is supervised by the reconstruction error. Once the model is learned, it can be applied to multiple applications including depth estimation, camera pose estimation, and single-image novel view synthesis. More qualitative results are available at: https://oasisyang.github.io/mononerf.

----

## [418] Go Beyond Imagination: Maximizing Episodic Reachability with World Models

**Authors**: *Yao Fu, Run Peng, Honglak Lee*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23c.html](https://proceedings.mlr.press/v202/fu23c.html)

**Abstract**:

Efficient exploration is a challenging topic in reinforcement learning, especially for sparse reward tasks. To deal with the reward sparsity, people commonly apply intrinsic rewards to motivate agents to explore the state space efficiently. In this paper, we introduce a new intrinsic reward design called GoBI - Go Beyond Imagination, which combines the traditional lifelong novelty motivation with an episodic intrinsic reward that is designed to maximize the stepwise reachability expansion. More specifically, we apply learned world models to generate predicted future states with random actions. States with more unique predictions that are not in episodic memory are assigned high intrinsic rewards. Our method greatly outperforms previous state-of-the-art methods on 12 of the most challenging Minigrid navigation tasks and improves the sample efficiency on locomotion tasks from DeepMind Control Suite.

----

## [419] Specializing Smaller Language Models towards Multi-Step Reasoning

**Authors**: *Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, Tushar Khot*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23d.html](https://proceedings.mlr.press/v202/fu23d.html)

**Abstract**:

The surprising ability of Large Language Models (LLMs) to perform well on complex reasoning with only few-shot chain-of-thought prompts is believed to emerge only in very large-scale models. We show that such abilities can, in fact, be distilled down from GPT-3.5 (≥ 175B) to T5 variants (≤ 11B). We propose model specialization, to specialize the model’s ability towards a target task. The hypothesis is that large models (commonly viewed as larger than 100B) have strong modeling power such that they can perform a large spectrum of tasks. Small models (commonly viewed as smaller than 10B) have limited model capacity, but if we specialize their capacity towards a target task, the model can achieve decent performance improvements. We use multi-step math reasoning as our testbed because it is a very typical emergent ability. We show two important aspects of model abilities: (1) balancing language model’s performance on multiple tasks is a delicate matter, as improvements on one task may compromise other tasks; (2) yet by intentionally paying the price of decreased generic ability, we can clearly improve across different model scales smaller than 10B towards a specialized multi-step math reasoning ability. We further give comprehensive discussions about important design choices for better generalization, including the data format mixture and the start model checkpoint. We hope our practice and discoveries can serve as an important attempt towards specialized smaller models in the new research paradigm set by LLMs.

----

## [420] Accelerated Stochastic Optimization Methods under Quasar-convexity

**Authors**: *Qiang Fu, Dongchu Xu, Ashia Camage Wilson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23e.html](https://proceedings.mlr.press/v202/fu23e.html)

**Abstract**:

Non-convex optimization plays a key role in a growing number of machine learning applications. This motivates the identification of specialized structure that enables sharper theoretical analysis. One such identified structure is quasar-convexity, a non-convex generalization of convexity that subsumes convex functions. Existing algorithms for minimizing quasar-convex functions in the stochastic setting have either high complexity or slow convergence, which prompts us to derive a new class of stochastic methods for optimizing smooth quasar-convex functions. We demonstrate that our algorithms have fast convergence and outperform existing algorithms on several examples, including the classical problem of learning linear dynamical systems. We also present a unified analysis of our newly proposed algorithms and a previously studied deterministic algorithm.

----

## [421] Meta-learning Parameterized Skills

**Authors**: *Haotian Fu, Shangqun Yu, Saket Tiwari, Michael Littman, George Konidaris*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23f.html](https://proceedings.mlr.press/v202/fu23f.html)

**Abstract**:

We propose a novel parameterized skill-learning algorithm that aims to learn transferable parameterized skills and synthesize them into a new action space that supports efficient learning in long-horizon tasks. We propose to leverage off-policy Meta-RL combined with a trajectory-centric smoothness term to learn a set of parameterized skills. Our agent can use these learned skills to construct a three-level hierarchical framework that models a Temporally-extended Parameterized Action Markov Decision Process. We empirically demonstrate that the proposed algorithms enable an agent to solve a set of highly difficult long-horizon (obstacle-course and robot manipulation) tasks.

----

## [422] NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations

**Authors**: *Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Celine Lin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/fu23g.html](https://proceedings.mlr.press/v202/fu23g.html)

**Abstract**:

Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks’ adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF’s adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool$^+$, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool$^+$ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.

----

## [423] Hierarchies of Reward Machines

**Authors**: *Daniel Furelos-Blanco, Mark Law, Anders Jonsson, Krysia Broda, Alessandra Russo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/furelos-blanco23a.html](https://proceedings.mlr.press/v202/furelos-blanco23a.html)

**Abstract**:

Reward machines (RMs) are a recent formalism for representing the reward function of a reinforcement learning task through a finite-state machine whose edges encode subgoals of the task using high-level events. The structure of RMs enables the decomposition of a task into simpler and independently solvable subtasks that help tackle long-horizon and/or sparse reward tasks. We propose a formalism for further abstracting the subtask structure by endowing an RM with the ability to call other RMs, thus composing a hierarchy of RMs (HRM). We exploit HRMs by treating each call to an RM as an independently solvable subtask using the options framework, and describe a curriculum-based method to learn HRMs from traces observed by the agent. Our experiments reveal that exploiting a handcrafted HRM leads to faster convergence than with a flat HRM, and that learning an HRM is feasible in cases where its equivalent flat representation is not.

----

## [424] Why Random Pruning Is All We Need to Start Sparse

**Authors**: *Advait Harshal Gadhikar, Sohom Mukherjee, Rebekka Burkholz*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gadhikar23a.html](https://proceedings.mlr.press/v202/gadhikar23a.html)

**Abstract**:

Random masks define surprisingly effective sparse neural network models, as has been shown empirically. The resulting sparse networks can often compete with dense architectures and state-of-the-art lottery ticket pruning algorithms, even though they do not rely on computationally expensive prune-train iterations and can be drawn initially without significant computational overhead. We offer a theoretical explanation of how random masks can approximate arbitrary target networks if they are wider by a logarithmic factor in the inverse sparsity $1 / \log(1/\text{sparsity})$. This overparameterization factor is necessary at least for 3-layer random networks, which elucidates the observed degrading performance of random networks at higher sparsity. At moderate to high sparsity levels, however, our results imply that sparser networks are contained within random source networks so that any dense-to-sparse training scheme can be turned into a computationally more efficient sparse-to-sparse one by constraining the search to a fixed random mask. We demonstrate the feasibility of this approach in experiments for different pruning methods and propose particularly effective choices of initial layer-wise sparsity ratios of the random source network. As a special case, we show theoretically and experimentally that random source networks also contain strong lottery tickets.

----

## [425] Cell-Free Latent Go-Explore

**Authors**: *Quentin Gallouédec, Emmanuel Dellandréa*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gallouedec23a.html](https://proceedings.mlr.press/v202/gallouedec23a.html)

**Abstract**:

In this paper, we introduce Latent Go-Explore (LGE), a simple and general approach based on the Go-Explore paradigm for exploration in reinforcement learning (RL). Go-Explore was initially introduced with a strong domain knowledge constraint for partitioning the state space into cells. However, in most real-world scenarios, drawing domain knowledge from raw observations is complex and tedious. If the cell partitioning is not informative enough, Go-Explore can completely fail to explore the environment. We argue that the Go-Explore approach can be generalized to any environment without domain knowledge and without cells by exploiting a learned latent representation. Thus, we show that LGE can be flexibly combined with any strategy for learning a latent representation. Our results indicate that LGE, although simpler than Go-Explore, is more robust and outperforms state-of-the-art algorithms in terms of pure exploration on multiple hard-exploration environments including Montezuma’s Revenge. The LGE implementation is available as open-source at https://github.com/qgallouedec/lge.

----

## [426] Graph Reinforcement Learning for Network Control via Bi-Level Optimization

**Authors**: *Daniele Gammelli, James Harrison, Kaidi Yang, Marco Pavone, Filipe Rodrigues, Francisco C. Pereira*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gammelli23a.html](https://proceedings.mlr.press/v202/gammelli23a.html)

**Abstract**:

Optimization problems over dynamic networks have been extensively studied and widely used in the past decades to formulate numerous real-world problems. However, (1) traditional optimization-based approaches do not scale to large networks, and (2) the design of good heuristics or approximation algorithms often requires significant manual trial-and-error. In this work, we argue that data-driven strategies can automate this process and learn efficient algorithms without compromising optimality. To do so, we present network control problems through the lens of reinforcement learning and propose a graph network-based framework to handle a broad class of problems. Instead of naively computing actions over high-dimensional graph elements, e.g., edges, we propose a bi-level formulation where we (1) specify a desired next state via RL, and (2) solve a convex program to best achieve it, leading to drastically improved scalability and performance. We further highlight a collection of desirable features to system designers, investigate design decisions, and present experiments on real-world control problems showing the utility, scalability, and flexibility of our framework.

----

## [427] Why Is Public Pretraining Necessary for Private Model Training?

**Authors**: *Arun Ganesh, Mahdi Haghifam, Milad Nasr, Sewoong Oh, Thomas Steinke, Om Thakkar, Abhradeep Guha Thakurta, Lun Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ganesh23a.html](https://proceedings.mlr.press/v202/ganesh23a.html)

**Abstract**:

In the privacy-utility tradeoff of a model trained on benchmark language and vision tasks, remarkable improvements have been widely reported when the model is pretrained on public data. Some gain is expected as these models inherit the benefits of transfer learning, which is the standard motivation in non-private settings. However, the stark contrast in the gain of pretraining between non-private and private machine learning suggests that the gain in the latter is rooted in a fundamentally different cause. To explain this phenomenon, we hypothesize that the non-convex loss landscape of a model training necessitates the optimization algorithm to go through two phases. In the first, the algorithm needs to select a good “basin” in the loss landscape. In the second, the algorithm solves an easy optimization within that basin. The former is a harder problem to solve with private data, while the latter is harder to solve with public data due to a distribution shift or data scarcity. Guided by this intuition, we provide theoretical constructions that provably demonstrate the separation between private training with and without public pretraining. Further, systematic experiments on CIFAR10 and Librispeech provide supporting evidence for our hypothesis.

----

## [428] Do Perceptually Aligned Gradients Imply Robustness?

**Authors**: *Roy Ganz, Bahjat Kawar, Michael Elad*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ganz23a.html](https://proceedings.mlr.press/v202/ganz23a.html)

**Abstract**:

Adversarially robust classifiers possess a trait that non-robust models do not - Perceptually Aligned Gradients (PAG). Their gradients with respect to the input align well with human perception. Several works have identified PAG as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether Perceptually Aligned Gradients imply Robustness. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on multiple datasets and architectures validate that models with aligned gradients exhibit significant robustness, exposing the surprising bidirectional connection between PAG and robustness. Lastly, we show that better gradient alignment leads to increased robustness and harness this observation to boost the robustness of existing adversarial training techniques.

----

## [429] Solving Linear Programs with Fast Online Learning Algorithms

**Authors**: *Wenzhi Gao, Dongdong Ge, Chunlin Sun, Yinyu Ye*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23a.html](https://proceedings.mlr.press/v202/gao23a.html)

**Abstract**:

This paper presents fast first-order methods for solving linear programs (LPs) approximately. We adapt online linear programming algorithms to offline LPs and obtain algorithms that avoid any matrix multiplication. We also introduce a variable-duplication technique that copies each variable $K$ times and reduces the optimality gap and constraint violation by a factor of $\sqrt{K}$. Furthermore, we show how online algorithms can be effectively integrated into sifting, a column generation scheme for large-scale LPs. Numerical experiments demonstrate that our methods can serve as either an approximate direct solver, or an initialization subroutine for exact LP solving.

----

## [430] Gradient Descent Finds the Global Optima of Two-Layer Physics-Informed Neural Networks

**Authors**: *Yihang Gao, Yiqi Gu, Michael Ng*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23b.html](https://proceedings.mlr.press/v202/gao23b.html)

**Abstract**:

The main aim of this paper is to conduct the convergence analysis of the gradient descent for two-layer physics-informed neural networks (PINNs). Here, the loss function involves derivatives of neural network outputs with respect to its inputs, so the interaction between the trainable parameters is more complicated compared with simple regression and classification tasks. We first develop the positive definiteness of Gram matrices and prove that the gradient flow finds the global optima of the empirical loss under over-parameterization. Then, we demonstrate that the standard gradient descent converges to the global optima of the loss with proper choices of learning rates. The framework of our analysis works for various categories of PDEs (e.g., linear second-order PDEs) and common types of network initialization (LecunUniform etc.). Our theoretical results do not need a very strict hypothesis for training samples and have a looser requirement on the network width compared with some previous works.

----

## [431] Generalizing Neural Wave Functions

**Authors**: *Nicholas Gao, Stephan Günnemann*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23c.html](https://proceedings.mlr.press/v202/gao23c.html)

**Abstract**:

Recent neural network-based wave functions have achieved state-of-the-art accuracies in modeling ab-initio ground-state potential energy surface. However, these networks can only solve different spatial arrangements of the same set of atoms. To overcome this limitation, we present Graph-learned orbital embeddings (Globe), a neural network-based reparametrization method that can adapt neural wave functions to different molecules. Globe learns representations of local electronic structures that generalize across molecules via spatial message passing by connecting molecular orbitals to covalent bonds. Further, we propose a size-consistent wave function Ansatz, the Molecular orbital network (Moon), tailored to jointly solve Schrödinger equations of different molecules. In our experiments, we find Moon converging in 4.5 times fewer steps to similar accuracy as previous methods or to lower energies given the same time. Further, our analysis shows that Moon’s energy estimate scales additively with increased system sizes, unlike previous work where we observe divergence. In both computational chemistry and machine learning, we are the first to demonstrate that a single wave function can solve the Schrödinger equation of molecules with different atoms jointly.

----

## [432] On the Impact of Algorithmic Recourse on Social Segregation

**Authors**: *Ruijiang Gao, Himabindu Lakkaraju*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23d.html](https://proceedings.mlr.press/v202/gao23d.html)

**Abstract**:

As predictive models seep into several real-world applications, it has become critical to ensure that individuals who are negatively impacted by the outcomes of these models are provided with a means for recourse. To this end, there has been a growing body of research on algorithmic recourse in recent years. While recourses can be extremely beneficial to affected individuals, their implementation at a large scale can lead to potential data distribution shifts and other unintended consequences. However, there is little to no research on understanding the impact of algorithmic recourse after implementation. In this work, we address the aforementioned gaps by making one of the first attempts at analyzing the delayed societal impact of algorithmic recourse. To this end, we theoretically and empirically analyze the recourses output by state-of-the-art algorithms. Our analysis demonstrates that large-scale implementation of recourses by end users may exacerbate social segregation. To address this problem, we propose novel algorithms which leverage implicit and explicit conditional generative models to not only minimize the chance of segregation but also provide realistic recourses. Extensive experimentation with real-world datasets demonstrates the efficacy of the proposed approaches.

----

## [433] DDGR: Continual Learning with Deep Diffusion-based Generative Replay

**Authors**: *Rui Gao, Weiwei Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23e.html](https://proceedings.mlr.press/v202/gao23e.html)

**Abstract**:

Popular deep-learning models in the field of image classification suffer from catastrophic forgetting—models will forget previously acquired skills when learning new ones. Generative replay (GR), which typically consists of a generator and a classifier, is an efficient way to mitigate catastrophic forgetting. However, conventional GR methods only focus on a single instruction relationship (generator-to-classifier), where the generator synthesizes samples for previous tasks to instruct the training of the classifier, while ignoring the ways in which the classifier can benefit the generator. In addition, most generative replay methods typically reuse the generated samples to update the generator, which causes the samples regenerated by the generator deviating from the distribution of previous tasks. To overcome these two issues, we propose a novel approach, called deep diffusion-based generative replay (DDGR), which adopts a diffusion model as the generator and calculates an instruction-operator through the classifier to instruct the generation of samples. Extensive experiments in class incremental (CI) and class incremental with repetition (CIR) settings demonstrate the advantages of DDGR. Our code is available at https://github.com/xiaocangshengGR/DDGR.

----

## [434] PAL: Program-aided Language Models

**Authors**: *Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, Graham Neubig*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23f.html](https://proceedings.mlr.press/v202/gao23f.html)

**Abstract**:

Large language models (LLMs) have demonstrated an impressive ability to perform arithmetic and symbolic reasoning tasks, when provided with a few examples at test time ("few-shot prompting"). Much of this success can be attributed to prompting methods such as "chain-of-thought", which employ LLMs for both understanding the problem description by decomposing it into steps, as well as solving each step of the problem. While LLMs seem to be adept at this sort of step-by-step decomposition, LLMs often make logical and arithmetic mistakes in the solution part, even when the problem is decomposed correctly. In this paper, we present Program-Aided Language models (PAL): a novel approach that uses the LLM to read natural language problems and generate programs as the intermediate reasoning steps, but offloads the solution step to a runtime such as a Python interpreter. With PAL, decomposing the natural language problem into runnable steps remains the only learning task for the LLM, while solving is delegated to the interpreter. We demonstrate this synergy between a neural LLM and a symbolic interpreter across 13 mathematical, symbolic, and algorithmic reasoning tasks from BIG-Bench Hard and others. In all these natural language reasoning tasks, generating code using an LLM and reasoning using a Python interpreter leads to more accurate results than much larger models. For example, PAL using Codex achieves state-of-the-art few-shot accuracy on GSM8K, surpassing PaLM which uses chain-of-thought by absolute 15% top-1.

----

## [435] Out-of-Domain Robustness via Targeted Augmentations

**Authors**: *Irena Gao, Shiori Sagawa, Pang Wei Koh, Tatsunori Hashimoto, Percy Liang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23g.html](https://proceedings.mlr.press/v202/gao23g.html)

**Abstract**:

Models trained on one set of domains often suffer performance drops on unseen domains, e.g., when wildlife monitoring models are deployed in new camera locations. In this work, we study principles for designing data augmentations for out-of-domain (OOD) generalization. In particular, we focus on real-world scenarios in which some domain-dependent features are robust, i.e., some features that vary across domains are predictive OOD. For example, in the wildlife monitoring application above, image backgrounds vary across camera locations but indicate habitat type, which helps predict the species of photographed animals. Motivated by theoretical analysis on a linear setting, we propose targeted augmentations, which selectively randomize spurious domain-dependent features while preserving robust ones. We prove that targeted augmentations improve OOD performance, allowing models to generalize better with fewer domains. In contrast, existing approaches such as generic augmentations, which fail to randomize domain-dependent features, and domain-invariant augmentations, which randomize all domain-dependent features, both perform poorly OOD. In experiments on three real-world datasets, we show that targeted augmentations set new states-of-the-art for OOD performance by 3.2-15.2%.

----

## [436] Scaling Laws for Reward Model Overoptimization

**Authors**: *Leo Gao, John Schulman, Jacob Hilton*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gao23h.html](https://proceedings.mlr.press/v202/gao23h.html)

**Abstract**:

In reinforcement learning from human feedback, it is common to optimize against a reward model trained to predict human preferences. Because the reward model is an imperfect proxy, optimizing its value too much can hinder ground truth performance, in accordance with Goodhart’s law. This effect has been frequently observed, but not carefully measured due to the expense of collecting human preference data. In this work, we use a synthetic setup in which a fixed “gold-standard” reward model plays the role of humans, providing labels used to train a proxy reward model. We study how the gold reward model score changes as we optimize against the proxy reward model using either reinforcement learning or best-of-$n$ sampling. We find that this relationship follows a different functional form depending on the method of optimization, and that in both cases its coefficients scale smoothly with the number of reward model parameters. We also study the effect on this relationship of the size of the reward model dataset, the number of reward model and policy parameters, and the coefficient of the KL penalty added to the reward in the reinforcement learning setup. We explore the implications of these empirical results for theoretical considerations in AI alignment.

----

## [437] The Unreasonable Effectiveness of Few-shot Learning for Machine Translation

**Authors**: *Xavier Garcia, Yamini Bansal, Colin Cherry, George F. Foster, Maxim Krikun, Melvin Johnson, Orhan Firat*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/garcia23a.html](https://proceedings.mlr.press/v202/garcia23a.html)

**Abstract**:

We demonstrate the potential of few-shot translation systems, trained with unpaired language data, for both high and low-resource language pairs. We show that with only 5 examples of high-quality translation data shown at inference, a transformer decoder-only model trained solely with self-supervised learning, is able to match specialized supervised state-of-the-art models as well as more general commercial translation systems. In particular, we outperform the best performing system on the WMT’21 English-Chinese news translation task by only using five examples of English-Chinese parallel data at inference. Furthermore, the resulting models are two orders of magnitude smaller than state-of-the-art language models. We then analyze the factors which impact the performance of few-shot translation systems, and highlight that the quality of the few-shot demonstrations heavily determines the quality of the translations generated by our models. Finally, we show that the few-shot paradigm also provides a way to control certain attributes of the translation — we show that we are able to control for regional varieties and formality using only a five examples at inference, paving the way towards controllable machine translation systems.

----

## [438] RLSbench: Domain Adaptation Under Relaxed Label Shift

**Authors**: *Saurabh Garg, Nick Erickson, James Sharpnack, Alex Smola, Sivaraman Balakrishnan, Zachary Chase Lipton*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/garg23a.html](https://proceedings.mlr.press/v202/garg23a.html)

**Abstract**:

Despite the emergence of principled methods for domain adaptation under label shift, their sensitivity to shifts in class conditional distributions is precariously under explored. Meanwhile, popular deep domain adaptation heuristics tend to falter when faced with label proportions shifts. While several papers modify these heuristics in attempts to handle label proportions shifts, inconsistencies in evaluation standards, datasets, and baselines make it difficult to gauge the current best practices. In this paper, we introduce RLSbench, a large-scale benchmark for relaxed label shift, consisting of $>$500 distribution shift pairs spanning vision, tabular, and language modalities, with varying label proportions. Unlike existing benchmarks, which primarily focus on shifts in class-conditional $p(x|y)$, our benchmark also focuses on label marginal shifts. First, we assess 13 popular domain adaptation methods, demonstrating more widespread failures under label proportion shifts than were previously known. Next, we develop an effective two-step meta-algorithm that is compatible with most domain adaptation heuristics: (i) pseudo-balance the data at each epoch; and (ii) adjust the final classifier with target label distribution estimate. The meta-algorithm improves existing domain adaptation heuristics under large label proportion shifts, often by 2–10% accuracy points, while conferring minimal effect ($<$0.5%) when label proportions do not shift. We hope that these findings and the availability of RLSbench will encourage researchers to rigorously evaluate proposed methods in relaxed label shift settings. Code is publicly available at https://github.com/acmi-lab/RLSbench.

----

## [439] RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank

**Authors**: *Quentin Garrido, Randall Balestriero, Laurent Najman, Yann LeCun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/garrido23a.html](https://proceedings.mlr.press/v202/garrido23a.html)

**Abstract**:

Joint-Embedding Self Supervised Learning (JE-SSL) has seen a rapid development, with the emergence of many method variations but only few principled guidelines that would help practitioners to successfully deploy them. The main reason for that pitfall comes from JE-SSL’s core principle of not employing any input reconstruction therefore lacking visual cues of unsuccessful training. Adding non informative loss values to that, it becomes difficult to deploy SSL on a new dataset for which no labels can help to judge the quality of the learned representation. In this study, we develop a simple unsupervised criterion that is indicative of the quality of the learned JE-SSL representations: their effective rank. Albeit simple and computationally friendly, this method —coined RankMe— allows one to assess the performance of JE-SSL representations, even on different downstream datasets, without requiring any labels. A further benefit of RankMe is that it does not have any training or hyper-parameters to tune. Through thorough empirical experiments involving hundreds of training episodes, we demonstrate how RankMe can be used for hyperparameter selection with nearly no reduction in final performance compared to the current selection method that involve a dataset’s labels. We hope that RankMe will facilitate the deployment of JE-SSL towards domains that do not have the opportunity to rely on labels for representations’ quality assessment.

----

## [440] Self-supervised learning of Split Invariant Equivariant representations

**Authors**: *Quentin Garrido, Laurent Najman, Yann LeCun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/garrido23b.html](https://proceedings.mlr.press/v202/garrido23b.html)

**Abstract**:

Recent progress has been made towards learning invariant or equivariant representations with self-supervised learning. While invariant methods are evaluated on large scale datasets, equivariant ones are evaluated in smaller, more controlled, settings. We aim at bridging the gap between the two in order to learn more diverse representations that are suitable for a wide range of tasks. We start by introducing a dataset called 3DIEBench, consisting of renderings from 3D models over 55 classes and more than 2.5 million images where we have full control on the transformations applied to the objects. We further introduce a predictor architecture based on hypernetworks to learn equivariant representations with no possible collapse to invariance. We introduce SIE (Split Invariant-Equivariant) which combines the hypernetwork-based predictor with representations split in two parts, one invariant, the other equivariant, to learn richer representations. We demonstrate significant performance gains over existing methods on equivariance related tasks from both a qualitative and quantitative point of view. We further analyze our introduced predictor and show how it steers the learned latent space. We hope that both our introduced dataset and approach will enable learning richer representations without supervision in more complex scenarios. Code and data are available at https://github.com/garridoq/SIE.

----

## [441] Federated Heavy Hitter Recovery under Linear Sketching

**Authors**: *Adrià Gascón, Peter Kairouz, Ziteng Sun, Ananda Theertha Suresh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gascon23a.html](https://proceedings.mlr.press/v202/gascon23a.html)

**Abstract**:

Motivated by real-life deployments of multi-round federated analytics with secure aggregation, we investigate the fundamental communication-accuracy tradeoffs of the heavy hitter discovery and approximate (open-domain) histogram problems under a linear sketching constraint. We propose efficient algorithms based on local subsampling and invertible bloom look-up tables (IBLTs). We also show that our algorithms are information-theoretically optimal for a broad class of interactive schemes. The results show that the linear sketching constraint does increase the communication cost for both tasks by introducing an extra linear dependence on the number of users in a round. Moreover, our results also establish a separation between the communication cost for heavy hitter discovery and approximate histogram in the multi-round setting. The dependence on the number of rounds $R$ is at most logarithmic for heavy hitter discovery whereas that of approximate histogram is $\Theta(\sqrt{R})$. We also empirically demonstrate our findings.

----

## [442] On the Global Convergence of Fitted Q-Iteration with Two-layer Neural Network Parametrization

**Authors**: *Mudit Gaur, Vaneet Aggarwal, Mridul Agarwal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gaur23a.html](https://proceedings.mlr.press/v202/gaur23a.html)

**Abstract**:

Deep Q-learning based algorithms have been applied successfully in many decision making problems, while their theoretical foundations are not as well understood. In this paper, we study a Fitted Q-Iteration with two-layer ReLU neural network parameterization, and find the sample complexity guarantees for the algorithm. Our approach estimates the Q-function in each iteration using a convex optimization problem. We show that this approach achieves a sample complexity of $\tilde{\mathcal{O}}(1/\epsilon^{2})$, which is order-optimal. This result holds for a countable state-spaces and does not require any assumptions such as a linear or low rank structure on the MDP.

----

## [443] A Reinforcement Learning Framework for Dynamic Mediation Analysis

**Authors**: *Lin Ge, Jitao Wang, Chengchun Shi, Zhenke Wu, Rui Song*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ge23a.html](https://proceedings.mlr.press/v202/ge23a.html)

**Abstract**:

Mediation analysis learns the causal effect transmitted via mediator variables between treatments and outcomes, and receives increasing attention in various scientific domains to elucidate causal relations. Most existing works focus on point-exposure studies where each subject only receives one treatment at a single time point. However, there are a number of applications (e.g., mobile health) where the treatments are sequentially assigned over time and the dynamic mediation effects are of primary interest. Proposing a reinforcement learning (RL) framework, we are the first to evaluate dynamic mediation effects in settings with infinite horizons. We decompose the average treatment effect into an immediate direct effect, an immediate mediation effect, a delayed direct effect, and a delayed mediation effect. Upon the identification of each effect component, we further develop robust and semi-parametrically efficient estimators under the RL framework to infer these causal effects. The superior performance of the proposed method is demonstrated through extensive numerical studies, theoretical results, and an analysis of a mobile health dataset. A Python implementation of the proposed procedure is available at https://github.com/linlinlin97/MediationRL.

----

## [444] Compositional Score Modeling for Simulation-Based Inference

**Authors**: *Tomas Geffner, George Papamakarios, Andriy Mnih*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/geffner23a.html](https://proceedings.mlr.press/v202/geffner23a.html)

**Abstract**:

Neural Posterior Estimation methods for simulation-based inference can be ill-suited for dealing with posterior distributions obtained by conditioning on multiple observations, as they tend to require a large number of simulator calls to learn accurate approximations. In contrast, Neural Likelihood Estimation methods can handle multiple observations at inference time after learning from individual observations, but they rely on standard inference methods, such as MCMC or variational inference, which come with certain performance drawbacks. We introduce a new method based on conditional score modeling that enjoys the benefits of both approaches. We model the scores of the (diffused) posterior distributions induced by individual observations, and introduce a way of combining the learned scores to approximately sample from the target posterior distribution. Our approach is sample-efficient, can naturally aggregate multiple observations at inference time, and avoids the drawbacks of standard inference methods.

----

## [445] Cramming: Training a Language Model on a single GPU in one day

**Authors**: *Jonas Geiping, Tom Goldstein*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/geiping23a.html](https://proceedings.mlr.press/v202/geiping23a.html)

**Abstract**:

Recent trends in language modeling have focused on increasing performance through scaling, and have resulted in an environment where training language models is out of reach for most researchers and practitioners. While most in the community are asking how to push the limits of extreme computation, we ask the opposite question: How far can we get with a single GPU in just one day? We investigate the downstream performance achievable with a transformer-based language model trained completely from scratch with masked language modeling for a single day on a single consumer GPU. Aside from re-analyzing nearly all components of the pretraining pipeline for this scenario and providing a modified pipeline with performance close to BERT, we investigate why scaling down is hard, and which modifications actually improve performance in this scenario. We provide evidence that even in this constrained setting, performance closely follows scaling laws observed in large-compute settings. Through the lens of scaling laws, we categorize a range of recent improvements to training and architecture and discuss their merit and practical applicability (or lack thereof) for the limited compute setting. We provide code to reproduce all experiments at github.com/JonasGeiping/cramming .

----

## [446] Transformers Meet Directed Graphs

**Authors**: *Simon Geisler, Yujia Li, Daniel J. Mankowitz, Ali Taylan Cemgil, Stephan Günnemann, Cosmin Paduraru*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/geisler23a.html](https://proceedings.mlr.press/v202/geisler23a.html)

**Abstract**:

Transformers were originally proposed as a sequence-to-sequence model for text but have become vital for a wide range of modalities, including images, audio, video, and undirected graphs. However, transformers for directed graphs are a surprisingly underexplored topic, despite their applicability to ubiquitous domains, including source code and logic circuits. In this work, we propose two direction- and structure-aware positional encodings for directed graphs: (1) the eigenvectors of the Magnetic Laplacian — a direction-aware generalization of the combinatorial Laplacian; (2) directional random walk encodings. Empirically, we show that the extra directionality information is useful in various downstream tasks, including correctness testing of sorting networks and source code understanding. Together with a data-flow-centric graph construction, our model outperforms the prior state of the art on the Open Graph Benchmark Code2 relatively by 14.7%.

----

## [447] Memory-Based Meta-Learning on Non-Stationary Distributions

**Authors**: *Tim Genewein, Grégoire Delétang, Anian Ruoss, Li Kevin Wenliang, Elliot Catt, Vincent Dutordoir, Jordi Grau-Moya, Laurent Orseau, Marcus Hutter, Joel Veness*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/genewein23a.html](https://proceedings.mlr.press/v202/genewein23a.html)

**Abstract**:

Memory-based meta-learning is a technique for approximating Bayes-optimal predictors. Under fairly general conditions, minimizing sequential prediction error, measured by the log loss, leads to implicit meta-learning. The goal of this work is to investigate how far this interpretation can be realized by current sequence prediction models and training regimes. The focus is on piecewise stationary sources with unobserved switching-points, which arguably capture an important characteristic of natural language and action-observation sequences in partially observable environments. We show that various types of memory-based neural models, including Transformers, LSTMs, and RNNs can learn to accurately approximate known Bayes-optimal algorithms and behave as if performing Bayesian inference over the latent switching-points and the latent parameters governing the data distribution within each segment.

----

## [448] Towards Reliable Neural Specifications

**Authors**: *Chuqin Geng, Nham Le, Xiaojie Xu, Zhaoyue Wang, Arie Gurfinkel, Xujie Si*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/geng23a.html](https://proceedings.mlr.press/v202/geng23a.html)

**Abstract**:

Having reliable specifications is an unavoidable challenge in achieving verifiable correctness, robustness, and interpretability of AI systems. Existing specifications for neural networks are in the paradigm of data as specification. That is, the local neighborhood centering around a reference input is considered to be correct (or robust). While existing specifications contribute to verifying adversarial robustness, a significant problem in many research domains, our empirical study shows that those verified regions are somewhat tight, and thus fail to allow verification of test set inputs, making them impractical for some real-world applications. To this end, we propose a new family of specifications called neural representation as specification. This form of specifications uses the intrinsic information of neural networks, specifically neural activation patterns (NAPs), rather than input data to specify the correctness and/or robustness of neural network predictions. We present a simple statistical approach to mining neural activation patterns. To show the effectiveness of discovered NAPs, we formally verify several important properties, such as various types of misclassifications will never happen for a given NAP, and there is no ambiguity between different NAPs. We show that by using NAP, we can verify a significant region of the input space, while still recalling 84% of the data on MNIST. Moreover, we can push the verifiable bound to 10 times larger on the CIFAR10 benchmark. Thus, we argue that NAPs can potentially be used as a more reliable and extensible specification for neural network verification.

----

## [449] Oracles & Followers: Stackelberg Equilibria in Deep Multi-Agent Reinforcement Learning

**Authors**: *Matthias Gerstgrasser, David C. Parkes*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gerstgrasser23a.html](https://proceedings.mlr.press/v202/gerstgrasser23a.html)

**Abstract**:

Stackelberg equilibria arise naturally in a range of popular learning problems, such as in security games or indirect mechanism design, and have received increasing attention in the reinforcement learning literature. We present a general framework for implementing Stackelberg equilibria search as a multi-agent RL problem, allowing a wide range of algorithmic design choices. We discuss how previous approaches can be seen as specific instantiations of this framework. As a key insight, we note that the design space allows for approaches not previously seen in the literature, for instance by leveraging multitask and meta-RL techniques for follower convergence. We propose one such approach using contextual policies, and evaluate it experimentally on both standard and novel benchmark domains, showing greatly improved sample efficiency compared to previous approaches. Finally, we explore the effect of adopting algorithm designs outside the borders of our framework.

----

## [450] Approximately Optimal Core Shapes for Tensor Decompositions

**Authors**: *Mehrdad Ghadiri, Matthew Fahrbach, Gang Fu, Vahab Mirrokni*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghadiri23a.html](https://proceedings.mlr.press/v202/ghadiri23a.html)

**Abstract**:

This work studies the combinatorial optimization problem of finding an optimal core tensor shape, also called multilinear rank, for a size-constrained Tucker decomposition. We give an algorithm with provable approximation guarantees for its reconstruction error via connections to higher-order singular values. Specifically, we introduce a novel Tucker packing problem, which we prove is NP-hard, and give a polynomial-time approximation scheme based on a reduction to the 2-dimensional knapsack problem with a matroid constraint. We also generalize our techniques to tree tensor network decompositions. We implement our algorithm using an integer programming solver, and show that its solution quality is competitive with (and sometimes better than) the greedy algorithm that uses the true Tucker decomposition loss at each step, while also running up to 1000x faster.

----

## [451] GAT: Guided Adversarial Training with Pareto-optimal Auxiliary Tasks

**Authors**: *Salah Ghamizi, Jingfeng Zhang, Maxime Cordy, Mike Papadakis, Masashi Sugiyama, Yves Le Traon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghamizi23a.html](https://proceedings.mlr.press/v202/ghamizi23a.html)

**Abstract**:

While leveraging additional training data is well established to improve adversarial robustness, it incurs the unavoidable cost of data collection and the heavy computation to train models. To mitigate the costs, we propose *Guided Adversarial Training * (GAT), a novel adversarial training technique that exploits auxiliary tasks under a limited set of training data. Our approach extends single-task models into multi-task models during the min-max optimization of adversarial training, and drives the loss optimization with a regularization of the gradient curvature across multiple tasks. GAT leverages two types of auxiliary tasks: self-supervised tasks, where the labels are generated automatically, and domain-knowledge tasks, where human experts provide additional labels. Experimentally, under limited data, GAT increases the robust accuracy on CIFAR-10 up to four times (from 11% to 42% robust accuracy) and the robust AUC of CheXpert medical imaging dataset from 50% to 83%. On the full CIFAR-10 dataset, GAT outperforms eight state-of-the-art adversarial training strategies. Our large study across five datasets and six tasks demonstrates that task augmentation is an efficient alternative to data augmentation, and can be key to achieving both clean and robust performances.

----

## [452] On User-Level Private Convex Optimization

**Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, Chiyuan Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghazi23a.html](https://proceedings.mlr.press/v202/ghazi23a.html)

**Abstract**:

We introduce a new mechanism for stochastic convex optimization (SCO) with user-level differential privacy guarantees. The convergence rates of this mechanism are similar to those in the prior work of Levy et al. 2021 and Narayanan et al. 2022, but with two important improvements. Our mechanism does not require any smoothness assumptions on the loss. Furthermore, our bounds are also the first where the minimum number of users needed for user-level privacy has no dependence on the dimension and only a logarithmic dependence on the desired excess error. The main idea underlying the new mechanism is to show that the optimizers of strongly convex losses have low local deletion sensitivity, along with a new output perturbation method for functions with low local deletion sensitivity, which could be of independent interest.

----

## [453] Contextual Reliability: When Different Features Matter in Different Contexts

**Authors**: *Gaurav Rohit Ghosal, Amrith Setlur, Daniel S. Brown, Anca D. Dragan, Aditi Raghunathan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghosal23a.html](https://proceedings.mlr.press/v202/ghosal23a.html)

**Abstract**:

Deep neural networks often fail catastrophically by relying on spurious correlations. Most prior work assumes a clear dichotomy into spurious and reliable features; however, this is often unrealistic. For example, most of the time we do not want an autonomous car to simply copy the speed of surrounding cars—we don’t want our car to run a red light if a neighboring car does so. However, we cannot simply enforce invariance to next-lane speed, since it could provide valuable information about an unobservable pedestrian at a crosswalk. Thus, universally ignoring features that are sometimes (but not always) reliable can lead to non-robust performance. We formalize a new setting called contextual reliability which accounts for the fact that the "right" features to use may vary depending on the context. We propose and analyze a two-stage framework called Explicit Non-spurious feature Prediction (ENP) which first identifies the relevant features to use for a given context, then trains a model to rely exclusively on these features. Our work theoretically and empirically demonstrates the advantages of ENP over existing methods and provides new benchmarks for contextual reliability.

----

## [454] Reinforcement Learning from Passive Data via Latent Intentions

**Authors**: *Dibya Ghosh, Chethan Anand Bhateja, Sergey Levine*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghosh23a.html](https://proceedings.mlr.press/v202/ghosh23a.html)

**Abstract**:

Passive observational data, such as human videos, is abundant and rich in information, yet remains largely untapped by current RL methods. Perhaps surprisingly, we show that passive data, despite not having reward or action labels, can still be used to learn features that accelerate downstream RL. Our approach learns from passive data by modeling intentions: measuring how the likelihood of future outcomes change when the agent acts to achieve a particular task. We propose a temporal difference learning objective to learn about intentions, resulting in an algorithm similar to conventional RL, but which learns entirely from passive data. When optimizing this objective, our agent simultaneously learns representations of states, of policies, and of possible outcomes in an environment, all from raw observational data. Both theoretically and empirically, this scheme learns features amenable for value prediction for downstream tasks, and our experiments demonstrate the ability to learn from many forms of passive data, including cross-embodiment video data and YouTube videos.

----

## [455] Harmonic Neural Networks

**Authors**: *Atiyo Ghosh, Antonio Andrea Gentile, Mario Dagrada, Chul Lee, Seong-Hyok Sean Kim, Hyukgeun Cha, Yunjun Choi, Dongho Kim, Jeong-Il Kye, Vincent Emanuel Elfving*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghosh23b.html](https://proceedings.mlr.press/v202/ghosh23b.html)

**Abstract**:

Harmonic functions are abundant in nature, appearing in limiting cases of Maxwell’s, Navier-Stokes equations, the heat and the wave equation. Consequently, there are many applications of harmonic functions from industrial process optimisation to robotic path planning and the calculation of first exit times of random walks. Despite their ubiquity and relevance, there have been few attempts to incorporate inductive biases towards harmonic functions in machine learning contexts. In this work, we demonstrate effective means of representing harmonic functions in neural networks and extend such results also to quantum neural networks to demonstrate the generality of our approach. We benchmark our approaches against (quantum) physics-informed neural networks, where we show favourable performance.

----

## [456] Dividing and Conquering a BlackBox to a Mixture of Interpretable Models: Route, Interpret, Repeat

**Authors**: *Shantanu Ghosh, Ke Yu, Forough Arabshahi, Kayhan Batmanghelich*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ghosh23c.html](https://proceedings.mlr.press/v202/ghosh23c.html)

**Abstract**:

ML model design either starts with an interpretable model or a Blackbox and explains it post hoc. Blackbox models are flexible but difficult to explain, while interpretable models are inherently explainable. Yet, interpretable models require extensive ML knowledge and tend to be less flexible, potentially underperforming than their Blackbox equivalents. This paper aims to blur the distinction between a post hoc explanation of a Blackbox and constructing interpretable models. Beginning with a Blackbox, we iteratively carve out a mixture of interpretable models and a residual network. The interpretable models identify a subset of samples and explain them using First Order Logic (FOL), providing basic reasoning on concepts from the Blackbox. We route the remaining samples through a flexible residual. We repeat the method on the residual network until all the interpretable models explain the desired proportion of data. Our extensive experiments show that our route, interpret, and repeat approach (1) identifies a richer diverse set of instance-specific concepts with high concept completeness via interpretable models by specializing in various subsets of data without compromising in performance, (2) identifies the relatively “harder” samples to explain via residuals, (3) outperforms the interpretable by-design models by significant margins during test-time interventions, (4) can be used to fix the shortcut learned by the original Blackbox.

----

## [457] Looped Transformers as Programmable Computers

**Authors**: *Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, Dimitris Papailiopoulos*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/giannou23a.html](https://proceedings.mlr.press/v202/giannou23a.html)

**Abstract**:

We present a framework for using transformer networks as universal computers by programming them with specific weights and placing them in a loop. Our input sequence acts as a punchcard, consisting of instructions and memory for data read/writes. We demonstrate that a constant number of encoder layers can emulate basic computing blocks, including lexicographic operations, non-linear functions, function calls, program counters, and conditional branches. Using this framework, we emulate a computer using a simple instruction-set architecture, which allows us to map iterative algorithms to programs that can be executed by a constant depth looped transformer network. We show how a single frozen transformer, instructed by its input, can emulate a basic calculator, a basic linear algebra library, and even a full backpropagation, in-context learning algorithm. Our findings reveal the potential of transformer networks as programmable compute units and offer insight into the mechanics of attention.

----

## [458] Generalized Disparate Impact for Configurable Fairness Solutions in ML

**Authors**: *Luca Giuliani, Eleonora Misino, Michele Lombardi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/giuliani23a.html](https://proceedings.mlr.press/v202/giuliani23a.html)

**Abstract**:

We make two contributions in the field of AI fairness over continuous protected attributes. First, we show that the Hirschfeld-Gebelein-Renyi (HGR) indicator (the only one currently available for such a case) is valuable but subject to a few crucial limitations regarding semantics, interpretability, and robustness. Second, we introduce a family of indicators that are: 1) complementary to HGR in terms of semantics; 2) fully interpretable and transparent; 3) robust over finite samples; 4) configurable to suit specific applications. Our approach also allows us to define fine-grained constraints to permit certain types of dependence and forbid others selectively. By expanding the available options for continuous protected attributes, our approach represents a significant contribution to the area of fair artificial intelligence.

----

## [459] Multicalibration as Boosting for Regression

**Authors**: *Ira Globus-Harris, Declan Harrison, Michael Kearns, Aaron Roth, Jessica Sorrell*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/globus-harris23a.html](https://proceedings.mlr.press/v202/globus-harris23a.html)

**Abstract**:

We study the connection between multicalibration and boosting for squared error regression. First we prove a useful characterization of multicalibration in terms of a “swap regret” like condition on squared error. Using this characterization, we give an exceedingly simple algorithm that can be analyzed both as a boosting algorithm for regression and as a multicalibration algorithm for a class $\mathcal{H}$ that makes use only of a standard squared error regression oracle for $\mathcal{H}$. We give a weak learning assumption on $\mathcal{H}$ that ensures convergence to Bayes optimality without the need to make any realizability assumptions — giving us an agnostic boosting algorithm for regression. We then show that our weak learning assumption on $\mathcal{H}$ is both necessary and sufficient for multicalibration with respect to $\mathcal{H}$ to imply Bayes optimality, answering an open question. We also show that if $\mathcal{H}$ satisfies our weak learning condition relative to another class $\mathcal{C}$ then multicalibration with respect to $\mathcal{H}$ implies multicalibration with respect to $\mathcal{C}$. Finally we investigate the empirical performance of our algorithm experimentally.

----

## [460] Adversarial robustness of amortized Bayesian inference

**Authors**: *Manuel Glöckler, Michael Deistler, Jakob H. Macke*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gloeckler23a.html](https://proceedings.mlr.press/v202/gloeckler23a.html)

**Abstract**:

Bayesian inference usually requires running potentially costly inference procedures separately for every new observation. In contrast, the idea of amortized Bayesian inference is to initially invest computational cost in training an inference network on simulated data, which can subsequently be used to rapidly perform inference (i.e., to return estimates of posterior distributions) for new observations. This approach has been applied to many real-world models in the sciences and engineering, but it is unclear how robust the approach is to adversarial perturbations in the observed data. Here, we study the adversarial robustness of amortized Bayesian inference, focusing on simulation-based estimation of multi-dimensional posterior distributions. We show that almost unrecognizable, targeted perturbations of the observations can lead to drastic changes in the predicted posterior and highly unrealistic posterior predictive samples, across several benchmark tasks and a real-world example from neuroscience. We propose a computationally efficient regularization scheme based on penalizing the Fisher information of the conditional density estimator, and show how it improves the adversarial robustness of amortized Bayesian inference.

----

## [461] Efficient RL via Disentangled Environment and Agent Representations

**Authors**: *Kevin Gmelin, Shikhar Bahl, Russell Mendonca, Deepak Pathak*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gmelin23a.html](https://proceedings.mlr.press/v202/gmelin23a.html)

**Abstract**:

Agents that are aware of the separation between the environments and themselves can leverage this understanding to form effective representations of visual input. We propose an approach for learning such structured representations for RL algorithms, using visual knowledge of the agent, which is often inexpensive to obtain, such as its shape or mask. This is incorporated into the RL objective using a simple auxiliary loss. We show that our method, SEAR (Structured Environment-Agent Representations), outperforms state-of-the-art model-free approaches over 18 different challenging visual simulation environments spanning 5 different robots.

----

## [462] Aligning Language Models with Preferences through f-divergence Minimization

**Authors**: *Dongyoung Go, Tomasz Korbak, Germán Kruszewski, Jos Rozen, Nahyeon Ryu, Marc Dymetman*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/go23a.html](https://proceedings.mlr.press/v202/go23a.html)

**Abstract**:

Aligning language models with preferences can be posed as approximating a target distribution representing some desired behavior. Existing approaches differ both in the functional form of the target distribution and the algorithm used to approximate it. For instance, Reinforcement Learning from Human Feedback (RLHF) corresponds to minimizing a reverse KL from an implicit target distribution arising from a KL penalty in the objective. On the other hand, Generative Distributional Control (GDC) has an explicit target distribution and minimizes a forward KL from it using the Distributional Policy Gradient (DPG) algorithm. In this paper, we propose a new approach, $f$-DPG, which allows the use of any $f$-divergence to approximate any target distribution that can be evaluated. $f$-DPG unifies both frameworks (RLHF, GDC) and the approximation methods (DPG, RL with KL penalties). We show the practical benefits of various choices of divergence objectives and demonstrate that there is no universally optimal objective but that different divergences present different alignment and diversity trade-offs. We show that Jensen-Shannon divergence strikes a good balance between these objectives, and frequently outperforms forward KL divergence by a wide margin, leading to significant improvements over prior work. These distinguishing characteristics between divergences persist as the model size increases, highlighting the importance of selecting appropriate divergence objectives.

----

## [463] Robust Consensus in Ranking Data Analysis: Definitions, Properties and Computational Issues

**Authors**: *Morgane Goibert, Clément Calauzènes, Ekhine Irurozki, Stéphan Clémençon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/goibert23a.html](https://proceedings.mlr.press/v202/goibert23a.html)

**Abstract**:

As the issue of robustness in AI systems becomes vital, statistical learning techniques that are reliable even in presence of partly contaminated data have to be developed. Preference data, in the form of (complete) rankings in the simplest situations, are no exception and the demand for appropriate concepts and tools is all the more pressing given that technologies fed by or producing this type of data ($\textit{e.g.}$ search engines, recommending systems) are now massively deployed. However, the lack of vector space structure for the set of rankings ($\textit{i.e.}$ the symmetric group $\mathfrak{S}_n$) and the complex nature of statistics considered in ranking data analysis make the formulation of robustness objectives in this domain challenging. In this paper, we introduce notions of robustness, together with dedicated statistical methods, for $\textit{Consensus Ranking}$ the flagship problem in ranking data analysis, aiming at summarizing a probability distribution on $\mathfrak{S}_n$ by a $\textit{median}$ ranking. Precisely, we propose specific extensions of the popular concept of breakdown point, tailored to consensus ranking, and address the related computational issues. Beyond the theoretical contributions, the relevance of the approach proposed is supported by an experimental study.

----

## [464] Learning Distributions over Quantum Measurement Outcomes

**Authors**: *Weiyuan Gong, Scott Aaronson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gong23a.html](https://proceedings.mlr.press/v202/gong23a.html)

**Abstract**:

Shadow tomography for quantum states provides a sample efficient approach for predicting the measurement outcomes of quantum systems. However, these shadow tomography procedures yield poor bounds if there are more than two outcomes per measurement. In this paper, we consider a general problem of learning properties from quantum states: given an unknown $d$-dimensional quantum state $\rho$ and $M$ unknown quantum measurements $\mathcal{M}_1,...,\mathcal{M}_M$ with $K\geq 2$ outcomes, estimating the probability distribution for applying $\mathcal{M}_i$ on $\rho$ to within total variation distance $\epsilon$. Compared to the special case when $K=2$, we have to learn unknown distributions instead of values. Here, we propose an online shadow tomography procedure that solves this problem with high success probability requiring $\tilde{O}(K\log^2M\log d/\epsilon^4)$ copies of $\rho$. We further prove an information-theoretic lower bound showing that at least $\Omega(\min\{d^2,K+\log M\}/\epsilon^2)$ copies of $\rho$ are required to solve this problem with high success probability. Our shadow tomography procedure requires sample complexity with only logarithmic dependence on $M$ and $d$ and is sample-optimal concerning the dependence on $K$.

----

## [465] Convergence of Proximal Point and Extragradient-Based Methods Beyond Monotonicity: the Case of Negative Comonotonicity

**Authors**: *Eduard Gorbunov, Adrien B. Taylor, Samuel Horváth, Gauthier Gidel*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gorbunov23a.html](https://proceedings.mlr.press/v202/gorbunov23a.html)

**Abstract**:

Algorithms for min-max optimization and variational inequalities are often studied under monotonicity assumptions. Motivated by non-monotone machine learning applications, we follow the line of works (Diakonikolas et al., 2021; Lee & Kim, 2021; Pethick et al., 2022; Bohm,2022) aiming at going beyond monotonicity by considering the weaker negative comonotonicity assumption. In this work, we provide tight complexity analyses for the Proximal Point (PP), Extragradient (EG), and Optimistic Gradient (OG) methods in this setup, closing several questions on their working guarantees beyond monotonicity. In particular, we derive the first non-asymptotic convergence rates for PP under negative comonotonicity and star-negative comonotonicity and show their tightness via constructing worst-case examples; we also relax the assumptions for the last-iterate convergence guarantees for EG and OG and prove the tightness of the existing best-iterate guarantees for EG and OG via constructing counter-examples.

----

## [466] Adaptive Annealed Importance Sampling with Constant Rate Progress

**Authors**: *Shirin Goshtasbpour, Victor Cohen, Fernando Pérez-Cruz*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/goshtasbpour23a.html](https://proceedings.mlr.press/v202/goshtasbpour23a.html)

**Abstract**:

Annealed Importance Sampling (AIS) synthesizes weighted samples from an intractable distribution given its unnormalized density function. This algorithm relies on a sequence of interpolating distributions bridging the target to an initial tractable distribution such as the well-known geometric mean path of unnormalized distributions which is assumed to be suboptimal in general. In this paper, we prove that the geometric annealing corresponds to the distribution path that minimizes the KL divergence between the current particle distribution and the desired target when the feasible change in the particle distribution is constrained. Following this observation, we derive the constant rate discretization schedule for this annealing sequence, which adjusts the schedule to the difficulty of moving samples between the initial and the target distributions. We further extend our results to $f$-divergences and present the respective dynamics of annealing sequences based on which we propose the Constant Rate AIS (CR-AIS) algorithm and its efficient implementation for $\alpha$-divergences. We empirically show that CR-AIS performs well on multiple benchmark distributions while avoiding the computationally expensive tuning loop in existing Adaptive AIS.

----

## [467] Formalizing Preferences Over Runtime Distributions

**Authors**: *Devon R. Graham, Kevin Leyton-Brown, Tim Roughgarden*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/graham23a.html](https://proceedings.mlr.press/v202/graham23a.html)

**Abstract**:

When trying to solve a computational problem, we are often faced with a choice between algorithms that are guaranteed to return the right answer but differ in their runtime distributions (e.g., SAT solvers, sorting algorithms). This paper aims to lay theoretical foundations for such choices by formalizing preferences over runtime distributions. It might seem that we should simply prefer the algorithm that minimizes expected runtime. However, such preferences would be driven by exactly how slow our algorithm is on bad inputs, whereas in practice we are typically willing to cut off occasional, sufficiently long runs before they finish. We propose a principled alternative, taking a utility-theoretic approach to characterize the scoring functions that describe preferences over algorithms. These functions depend on the way our value for solving our problem decreases with time and on the distribution from which captimes are drawn. We describe examples of realistic utility functions and show how to leverage a maximum-entropy approach for modeling underspecified captime distributions. Finally, we show how to efficiently estimate an algorithm’s expected utility from runtime samples.

----

## [468] Topological Point Cloud Clustering

**Authors**: *Vincent Peter Grande, Michael T. Schaub*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/grande23a.html](https://proceedings.mlr.press/v202/grande23a.html)

**Abstract**:

We present Topological Point Cloud Clustering (TPCC), a new method to cluster points in an arbitrary point cloud based on their contribution to global topological features. TPCC synthesizes desirable features from spectral clustering and topological data analysis and is based on considering the spectral properties of a simplicial complex associated to the considered point cloud. As it is based on considering sparse eigenvector computations, TPCC is similarly easy to interpret and implement as spectral clustering. However, by focusing not just on a single matrix associated to a graph created from the point cloud data, but on a whole set of Hodge-Laplacians associated to an appropriately constructed simplicial complex, we can leverage a far richer set of topological features to characterize the data points within the point cloud and benefit from the relative robustness of topological techniques against noise. We test the performance of TPCC on both synthetic and real-world data and compare it with classical spectral clustering.

----

## [469] On Sampling with Approximate Transport Maps

**Authors**: *Louis Grenioux, Alain Oliviero Durmus, Eric Moulines, Marylou Gabrié*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/grenioux23a.html](https://proceedings.mlr.press/v202/grenioux23a.html)

**Abstract**:

Transport maps can ease the sampling of distributions with non-trivial geometries by transforming them into distributions that are easier to handle. The potential of this approach has risen with the development of Normalizing Flows (NF) which are maps parameterized with deep neural networks trained to push a reference distribution towards a target. NF-enhanced samplers recently proposed blend (Markov chain) Monte Carlo methods with either (i) proposal draws from the flow or (ii) a flow-based reparametrization. In both cases, the quality of the learned transport conditions performance. The present work clarifies for the first time the relative strengths and weaknesses of these two approaches. Our study concludes that multimodal targets can be reliably handled with flow-based proposals up to moderately high dimensions. In contrast, methods relying on reparametrization struggle with multimodality but are more robust otherwise in high-dimensional settings and under poor training. To further illustrate the influence of target-proposal adequacy, we also derive a new quantitative bound for the mixing time of the Independent Metropolis-Hastings sampler.

----

## [470] Hidden Symmetries of ReLU Networks

**Authors**: *J. Elisenda Grigsby, Kathryn Lindsey, David Rolnick*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/grigsby23a.html](https://proceedings.mlr.press/v202/grigsby23a.html)

**Abstract**:

The parameter space for any fixed architecture of feedforward ReLU neural networks serves as a proxy during training for the associated class of functions - but how faithful is this representation? It is known that many different parameter settings $\theta$ can determine the same function $f$. Moreover, the degree of this redundancy is inhomogeneous: for some networks, the only symmetries are permutation of neurons in a layer and positive scaling of parameters at a neuron, while other networks admit additional hidden symmetries. In this work, we prove that, for any network architecture where no layer is narrower than the input, there exist parameter settings with no hidden symmetries. We also describe a number of mechanisms through which hidden symmetries can arise, and empirically approximate the functional dimension of different network architectures at initialization. These experiments indicate that the probability that a network has no hidden symmetries decreases towards 0 as depth increases, while increasing towards 1 as width and input dimension increase.

----

## [471] EF21-P and Friends: Improved Theoretical Communication Complexity for Distributed Optimization with Bidirectional Compression

**Authors**: *Kaja Gruntkowska, Alexander Tyurin, Peter Richtárik*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gruntkowska23a.html](https://proceedings.mlr.press/v202/gruntkowska23a.html)

**Abstract**:

In this work we focus our attention on distributed optimization problems in the context where the communication time between the server and the workers is non-negligible. We obtain novel methods supporting bidirectional compression (both from the server to the workers and vice versa) that enjoy new state-of-the-art theoretical communication complexity for convex and nonconvex problems. Our bounds are the first that manage to decouple the variance/error coming from the workers-to-server and server-to-workers compression, transforming a multiplicative dependence to an additive one. Moreover, in the convex regime, we obtain the first bounds that match the theoretical communication complexity of gradient descent. Even in this convex regime, our algorithms work with biased gradient estimators, which is non-standard and requires new proof techniques that may be of independent interest. Finally, our theoretical results are corroborated through suitable experiments.

----

## [472] NerfDiff: Single-image View Synthesis with NeRF-guided Distillation from 3D-aware Diffusion

**Authors**: *Jiatao Gu, Alex Trevithick, Kai-En Lin, Joshua M. Susskind, Christian Theobalt, Lingjie Liu, Ravi Ramamoorthi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gu23a.html](https://proceedings.mlr.press/v202/gu23a.html)

**Abstract**:

Novel view synthesis from a single image requires inferring occluded regions of objects and scenes whilst simultaneously maintaining semantic and physical consistency with the input. Existing approaches condition neural radiance fields (NeRF) on local image features, projecting points to the input image plane, and aggregating 2D features to perform volume rendering. However, under severe occlusion, this projection fails to resolve uncertainty, resulting in blurry renderings that lack details. In this work, we propose NerfDiff, which addresses this issue by distilling the knowledge of a 3D-aware conditional diffusion model (CDM) into NeRF through synthesizing and refining a set of virtual views at test-time. We further propose a novel NeRF-guided distillation algorithm that simultaneously generates 3D consistent virtual views from the CDM samples, and finetunes the NeRF based on the improved virtual views. Our approach significantly outperforms existing NeRF-based and geometry-free approaches on challenging datasets including ShapeNet, ABO, and Clevr3D.

----

## [473] DecompDiff: Diffusion Models with Decomposed Priors for Structure-Based Drug Design

**Authors**: *Jiaqi Guan, Xiangxin Zhou, Yuwei Yang, Yu Bao, Jian Peng, Jianzhu Ma, Qiang Liu, Liang Wang, Quanquan Gu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guan23a.html](https://proceedings.mlr.press/v202/guan23a.html)

**Abstract**:

Designing 3D ligands within a target binding site is a fundamental task in drug discovery. Existing structured-based drug design methods treat all ligand atoms equally, which ignores different roles of atoms in the ligand for drug design and can be less efficient for exploring the large drug-like molecule space. In this paper, inspired by the convention in pharmaceutical practice, we decompose the ligand molecule into two parts, namely arms and scaffold, and propose a new diffusion model, DecompDiff, with decomposed priors over arms and scaffold. In order to facilitate the decomposed generation and improve the properties of the generated molecules, we incorporate both bond diffusion in the model and additional validity guidance in the sampling phase. Extensive experiments on CrossDocked2020 show that our approach achieves state-of-the-art performance in generating high-affinity molecules while maintaining proper molecular properties and conformational stability, with up to $-8.39$ Avg. Vina Dock score and $24.5%$ Success Rate. The code is provided at https://github.com/bytedance/DecompDiff

----

## [474] On Excess Mass Behavior in Gaussian Mixture Models with Orlicz-Wasserstein Distances

**Authors**: *Aritra Guha, Nhat Ho, XuanLong Nguyen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guha23a.html](https://proceedings.mlr.press/v202/guha23a.html)

**Abstract**:

Dirichlet Process mixture models (DPMM) in combination with Gaussian kernels have been an important modeling tool for numerous data domains arising from biological, physical, and social sciences. However, this versatility in applications does not extend to strong theoretical guarantees for the underlying parameter estimates, for which only a logarithmic rate is achieved. In this work, we (re)introduce and investigate a metric, named Orlicz-Wasserstein distance, in the study of the Bayesian contraction behavior for the parameters. We show that despite the overall slow convergence guarantees for all the parameters, posterior contraction for parameters happens at almost polynomial rates in outlier regions of the parameter space. Our theoretical results provide new insight in understanding the convergence behavior of parameters arising from various settings of hierarchical Bayesian nonparametric models. In addition, we provide an algorithm to compute the metric by leveraging Sinkhorn divergences and validate our findings through a simulation study.

----

## [475] Conformalization of Sparse Generalized Linear Models

**Authors**: *Etash Kumar Guha, Eugène Ndiaye, Xiaoming Huo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guha23b.html](https://proceedings.mlr.press/v202/guha23b.html)

**Abstract**:

Given a sequence of observable variables $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, the conformal prediction method estimates a confidence set for $y_{n+1}$ given $x_{n+1}$ that is valid for any finite sample size by merely assuming that the joint distribution of the data is permutation invariant. Although attractive, computing such a set is computationally infeasible in most regression problems. Indeed, in these cases, the unknown variable $y_{n+1}$ can take an infinite number of possible candidate values, and generating conformal sets requires retraining a predictive model for each candidate. In this paper, we focus on a sparse linear model with only a subset of variables for prediction and use numerical continuation techniques to approximate the solution path efficiently. The critical property we exploit is that the set of selected variables is invariant under a small perturbation of the input data. Therefore, it is sufficient to enumerate and refit the model only at the change points of the set of active features and smoothly interpolate the rest of the solution via a Predictor-Corrector mechanism. We show how our path-following algorithm accurately approximates conformal prediction sets and illustrate its performance using synthetic and real data examples.

----

## [476] Privacy-Aware Compression for Federated Learning Through Numerical Mechanism Design

**Authors**: *Chuan Guo, Kamalika Chaudhuri, Pierre Stock, Michael G. Rabbat*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23a.html](https://proceedings.mlr.press/v202/guo23a.html)

**Abstract**:

In private federated learning (FL), a server aggregates differentially private updates from a large number of clients in order to train a machine learning model. The main challenge in this setting is balancing privacy with both classification accuracy of the learnt model as well as the number of bits communicated between the clients and server. Prior work has achieved a good trade-off by designing a privacy-aware compression mechanism, called the minimum variance unbiased (MVU) mechanism, that numerically solves an optimization problem to determine the parameters of the mechanism. This paper builds upon it by introducing a new interpolation procedure in the numerical design process that allows for a far more efficient privacy analysis. The result is the new Interpolated MVU mechanism that is more scalable, has a better privacy-utility trade-off, and provides SOTA results on communication-efficient private FL on a variety of datasets.

----

## [477] Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships

**Authors**: *Yaming Guo, Kai Guo, Xiaofeng Cao, Tieru Wu, Yi Chang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23b.html](https://proceedings.mlr.press/v202/guo23b.html)

**Abstract**:

Out-of-distribution generalization is challenging for non-participating clients of federated learning under distribution shifts. A proven strategy is to explore those invariant relationships between input and target variables, working equally well for non-participating clients. However, learning invariant relationships is often in an explicit manner from data, representation, and distribution, which violates the federated principles of privacy-preserving and limited communication. In this paper, we propose FedIIR, which implicitly learns invariant relationships from parameter for out-of-distribution generalization, adhering to the above principles. Specifically, we utilize the prediction disagreement to quantify invariant relationships and implicitly reduce it through inter-client gradient alignment. Theoretically, we demonstrate the range of non-participating clients to which FedIIR is expected to generalize and present the convergence results for FedIIR in the massively distributed with limited communication. Extensive experiments show that FedIIR significantly outperforms relevant baselines in terms of out-of-distribution generalization of federated learning.

----

## [478] FeDXL: Provable Federated Learning for Deep X-Risk Optimization

**Authors**: *Zhishuai Guo, Rong Jin, Jiebo Luo, Tianbao Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23c.html](https://proceedings.mlr.press/v202/guo23c.html)

**Abstract**:

In this paper, we tackle a novel federated learning (FL) problem for optimizing a family of X-risks, to which no existing FL algorithms are applicable. In particular, the objective has the form of $\mathbb{E}_{\mathbf{z}\sim \mathcal{S}_1} f(\mathbb{E}_{\mathbf{z}’\sim\mathcal{S}_2} \ell(\mathbf{w}; \mathbf{z}, \mathbf{z}’))$, where two sets of data $\mathcal S_1, \mathcal S_2$ are distributed over multiple machines, $\ell(\cdot; \cdot,\cdot)$ is a pairwise loss that only depends on the prediction outputs of the input data pairs $(\mathbf{z}, \mathbf{z}’)$. This problem has important applications in machine learning, e.g., AUROC maximization with a pairwise loss, and partial AUROC maximization with a compositional loss. The challenges for designing an FL algorithm for X-risks lie in the non-decomposability of the objective over multiple machines and the interdependency between different machines. To this end, we propose an active-passive decomposition framework that decouples the gradient’s components with two types, namely active parts and passive parts, where the active parts depend on local data that are computed with the local model and the passive parts depend on other machines that are communicated/computed based on historical models and samples. Under this framework, we design two FL algorithms (FeDXL) for handling linear and nonlinear $f$, respectively, based on federated averaging and merging and develop a novel theoretical analysis to combat the latency of the passive parts and the interdependency between the local model parameters and the involved data for computing local gradient estimators. We establish both iteration and communication complexities and show that using the historical samples and models for computing the passive parts do not degrade the complexities. We conduct empirical studies of FeDXL for deep AUROC and partial AUROC maximization, and demonstrate their performance compared with several baselines.

----

## [479] Provably Efficient Representation Learning with Tractable Planning in Low-Rank POMDP

**Authors**: *Jiacheng Guo, Zihao Li, Huazheng Wang, Mengdi Wang, Zhuoran Yang, Xuezhou Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23d.html](https://proceedings.mlr.press/v202/guo23d.html)

**Abstract**:

In this paper, we study representation learning in partially observable Markov Decision Processes (POMDPs), where the agent learns a decoder function that maps a series of high-dimensional raw observations to a compact representation and uses it for more efficient exploration and planning. We focus our attention on the sub-classes of $\gamma$-observable and decodable POMDPs, for which it has been shown that statistically tractable learning is possible, but there has not been any computationally efficient algorithm. We first present an algorithm for decodable PMMDPs that combines maximum likelihood estimation (MLE) and optimism in the face of uncertainty (OFU) to perform representation learning and achieve efficient sample complexity, while only calling supervised learning computational oracles. We then show how to adapt this algorithm to also work in the broader class of $\gamma$-observable POMDPs.

----

## [480] Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano

**Authors**: *Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23e.html](https://proceedings.mlr.press/v202/guo23e.html)

**Abstract**:

Differential privacy (DP) is by far the most widely accepted framework for mitigating privacy risks in machine learning. However, exactly how small the privacy parameter $\epsilon$ needs to be to protect against certain privacy risks in practice is still not well-understood. In this work, we study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. For a learning algorithm satisfying $(\alpha, \epsilon)$-Renyi DP, we utilize different variants of the celebrated Fano’s inequality to upper bound the attack advantage of a data reconstruction adversary. Our bound can be numerically computed to relate the parameter $\epsilon$ to the desired level of privacy protection in practice, and complements the empirical evidence for the effectiveness of DP against data reconstruction attacks even at relatively large values of $\epsilon$.

----

## [481] Linkless Link Prediction via Relational Distillation

**Authors**: *Zhichun Guo, William Shiao, Shichang Zhang, Yozen Liu, Nitesh V. Chawla, Neil Shah, Tong Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23f.html](https://proceedings.mlr.press/v202/guo23f.html)

**Abstract**:

Graph Neural Networks (GNNs) have shown exceptional performance in the task of link prediction. Despite their effectiveness, the high latency brought by non-trivial neighborhood data dependency limits GNNs in practical deployments. Conversely, the known efficient MLPs are much less effective than GNNs due to the lack of relational knowledge. In this work, to combine the advantages of GNNs and MLPs, we start with exploring direct knowledge distillation (KD) methods for link prediction, i.e., predicted logit-based matching and node representation-based matching. Upon observing direct KD analogs do not perform well for link prediction, we propose a relational KD framework, Linkless Link Prediction (LLP), to distill knowledge for link prediction with MLPs. Unlike simple KD methods that match independent link logits or node representations, LLP distills relational knowledge that is centered around each (anchor) node to the student MLP. Specifically, we propose rank-based matching and distribution-based matching strategies that complement each other. Extensive experiments demonstrate that LLP boosts the link prediction performance of MLPs with significant margins and even outperforms the teacher GNNs on 7 out of 8 benchmarks. LLP also achieves a 70.68x speedup in link prediction inference compared to GNNs on the large-scale OGB dataset.

----

## [482] FedBR: Improving Federated Learning on Heterogeneous Data via Local Learning Bias Reduction

**Authors**: *Yongxin Guo, Xiaoying Tang, Tao Lin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23g.html](https://proceedings.mlr.press/v202/guo23g.html)

**Abstract**:

Federated Learning (FL) is a way for machines to learn from data that is kept locally, in order to protect the privacy of clients. This is typically done using local SGD, which helps to improve communication efficiency. However, such a scheme is currently constrained by slow and unstable convergence due to the variety of data on different clients’ devices. In this work, we identify three under-explored phenomena of biased local learning that may explain these challenges caused by local updates in supervised FL. As a remedy, we propose FedBR, a novel unified algorithm that reduces the local learning bias on features and classifiers to tackle these challenges. FedBR has two components. The first component helps to reduce bias in local classifiers by balancing the output of the models. The second component helps to learn local features that are similar to global features, but different from those learned from other data sources. We conducted several experiments to test FedBR and found that it consistently outperforms other SOTA FL methods. Both of its components also individually show performance gains. Our code is available at https://github.com/lins-lab/fedbr.

----

## [483] Hierarchical Grammar-Induced Geometry for Data-Efficient Molecular Property Prediction

**Authors**: *Minghao Guo, Veronika Thost, Samuel W. Song, Adithya Balachandran, Payel Das, Jie Chen, Wojciech Matusik*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23h.html](https://proceedings.mlr.press/v202/guo23h.html)

**Abstract**:

The prediction of molecular properties is a crucial task in the field of material and drug discovery. The potential benefits of using deep learning techniques are reflected in the wealth of recent literature. Still, these techniques are faced with a common challenge in practice: Labeled data are limited by the cost of manual extraction from literature and laborious experimentation. In this work, we propose a data-efficient property predictor by utilizing a learnable hierarchical molecular grammar that can generate molecules from grammar production rules. Such a grammar induces an explicit geometry of the space of molecular graphs, which provides an informative prior on molecular structural similarity. The property prediction is performed using graph neural diffusion over the grammar-induced geometry. On both small and large datasets, our evaluation shows that this approach outperforms a wide spectrum of baselines, including supervised and pre-trained graph neural networks. We include a detailed ablation study and further analysis of our solution, showing its effectiveness in cases with extremely limited data.

----

## [484] Graph Neural Networks with Learnable and Optimal Polynomial Bases

**Authors**: *Yuhe Guo, Zhewei Wei*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23i.html](https://proceedings.mlr.press/v202/guo23i.html)

**Abstract**:

Polynomial filters, a kind of Graph Neural Networks, typically use a predetermined polynomial basis and learn the coefficients from the training data. It has been observed that the effectiveness of the model is highly dependent on the property of the polynomial basis. Consequently, two natural and fundamental questions arise: Can we learn a suitable polynomial basis from the training data? Can we determine the optimal polynomial basis for a given graph and node features? In this paper, we propose two spectral GNN models that provide positive answers to the questions posed above. First, inspired by Favard’s Theorem, we propose the FavardGNN model, which learns a polynomial basis from the space of all possible orthonormal bases. Second, we examine the supposedly unsolvable definition of optimal polynomial basis from Wang et al. (2022) and propose a simple model, OptBasisGNN, which computes the optimal basis for a given graph structure and graph signal. Extensive experiments are conducted to demonstrate the effectiveness of our proposed models. Our code is available at https://github.com/yuziGuo/FarOptBasis.

----

## [485] LongCoder: A Long-Range Pre-trained Language Model for Code Completion

**Authors**: *Daya Guo, Canwen Xu, Nan Duan, Jian Yin, Julian J. McAuley*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23j.html](https://proceedings.mlr.press/v202/guo23j.html)

**Abstract**:

In this paper, we introduce a new task for code completion that focuses on handling long code input and propose a sparse Transformer model, called LongCoder, to address this task. LongCoder employs a sliding window mechanism for self-attention and introduces two types of globally accessible tokens - bridge tokens and memory tokens - to improve performance and efficiency. Bridge tokens are inserted throughout the input sequence to aggregate local information and facilitate global interaction, while memory tokens are included to highlight important statements that may be invoked later and need to be memorized, such as package imports and definitions of classes, functions, or structures. We conduct experiments on a newly constructed dataset that contains longer code context and the publicly available CodeXGLUE benchmark. Experimental results demonstrate that LongCoder achieves superior performance on code completion tasks compared to previous models while maintaining comparable efficiency in terms of computational resources during inference.

----

## [486] Estimating Heterogeneous Treatment Effects: Mutual Information Bounds and Learning Algorithms

**Authors**: *Xingzhuo Guo, Yuchen Zhang, Jianmin Wang, Mingsheng Long*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23k.html](https://proceedings.mlr.press/v202/guo23k.html)

**Abstract**:

Estimating heterogeneous treatment effects (HTE) from observational studies is rising in importance due to the widespread accumulation of data in many fields. Due to the selection bias behind the inaccessibility of counterfactual data, the problem differs fundamentally from supervised learning in a challenging way. However, existing works on modeling selection bias and corresponding algorithms do not naturally generalize to non-binary treatment spaces. To address this limitation, we propose to use mutual information to describe selection bias in estimating HTE and derive a novel error bound using the mutual information between the covariates and the treatments, which is the first error bound to cover general treatment schemes including multinoulli or continuous spaces. We then bring forth theoretically justified algorithms, the Mutual Information Treatment Network (MitNet), using adversarial optimization to reduce selection bias and obtain more accurate HTE estimations. Our algorithm reaches remarkable performance in both simulation study and empirical evaluation.

----

## [487] Identifying Useful Learnwares for Heterogeneous Label Spaces

**Authors**: *Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li, Zhi-Hua Zhou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guo23l.html](https://proceedings.mlr.press/v202/guo23l.html)

**Abstract**:

The learnware paradigm aims to build a learnware market containing numerous learnwares, each of which is a well-performing machine learning model with a corresponding specification to describe its functionality so that future users can identify useful models for reuse according to their own requirements. With the learnware paradigm, model developers can spontaneously submit models to the market without leaking data privacy, and users can leverage models in the market to accomplish different machine learning tasks without having to build models from scratch. Recent studies have attempted to realize the model specification through Reduced Kernel Mean Embedding (RKME). In this paper, we make an attempt to improve the effectiveness of RKME specification for heterogeneous label spaces, where the learnware market does not contain a model that has the same label space as the user’s task, by considering a class-specific model specification explicitly, along with a class-wise learnware identification method. Both theoretical and empirical analyses show that our proposal can quickly and accurately find useful learnwares that satisfy users’ requirements. Moreover, we find that for a specific task, reusing a small model identified via the specification performs better than directly reusing a pre-trained generic big model.

----

## [488] High-dimensional Location Estimation via Norm Concentration for Subgamma Vectors

**Authors**: *Shivam Gupta, Jasper C. H. Lee, Eric Price*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gupta23a.html](https://proceedings.mlr.press/v202/gupta23a.html)

**Abstract**:

In location estimation, we are given $n$ samples from a known distribution $f$ shifted by an unknown translation $\lambda$, and want to estimate $\lambda$ as precisely as possible. Asymptotically, the maximum likelihood estimate achieves the Cramér-Rao bound of error $\mathcal N(0, \frac{1}{n\mathcal I})$, where $\mathcal I$ is the Fisher information of $f$. However, the $n$ required for convergence depends on $f$, and may be arbitrarily large. We build on the theory using smoothed estimators to bound the error for finite $n$ in terms of $\mathcal I_r$, the Fisher information of the $r$-smoothed distribution. As $n \to \infty$, $r \to 0$ at an explicit rate and this converges to the Cramér-Rao bound. We (1) improve the prior work for 1-dimensional $f$ to converge for constant failure probability in addition to high probability, and (2) extend the theory to high-dimensional distributions. In the process, we prove a new bound on the norm of a high-dimensional random variable whose 1-dimensional projections are subgamma, which may be of independent interest.

----

## [489] GRAFENNE: Learning on Graphs with Heterogeneous and Dynamic Feature Sets

**Authors**: *Shubham Gupta, Sahil Manchanda, Sayan Ranu, Srikanta J. Bedathur*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gupta23b.html](https://proceedings.mlr.press/v202/gupta23b.html)

**Abstract**:

Graph neural networks (GNNs), in general, are built on the assumption of a static set of features characterizing each node in a graph. This assumption is often violated in practice. Existing methods partly address this issue through feature imputation. However, these techniques (i) assume uniformity of feature set across nodes, (ii) are transductive by nature, and (iii) fail to work when features are added or removed over time. In this work, we address these limitations through a novel GNN framework called GRAFENNE. GRAFENNE performs a novel allotropic transformation on the original graph, wherein the nodes and features are decoupled through a bipartite encoding. Through a carefully chosen message passing framework on the allotropic transformation, we make the model parameter size independent of the number of features and thereby inductive to both unseen nodes and features. We prove that GRAFENNE is at least as expressive as any of the existing message-passing GNNs in terms of Weisfeiler-Leman tests, and therefore, the additional inductivity to unseen features does not come at the cost of expressivity. In addition, as demonstrated over four real-world graphs, GRAFENNE empowers the underlying GNN with high empirical efficacy and the ability to learn in continual fashion over streaming feature sets.

----

## [490] Online Platt Scaling with Calibeating

**Authors**: *Chirag Gupta, Aaditya Ramdas*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gupta23c.html](https://proceedings.mlr.press/v202/gupta23c.html)

**Abstract**:

We present an online post-hoc calibration method, called Online Platt Scaling (OPS), which combines the Platt scaling technique with online logistic regression. We demonstrate that OPS smoothly adapts between i.i.d. and non-i.i.d. settings with distribution drift. Further, in scenarios where the best Platt scaling model is itself miscalibrated, we enhance OPS by incorporating a recently developed technique called calibeating to make it more robust. Theoretically, our resulting OPS+calibeating method is guaranteed to be calibrated for adversarial outcome sequences. Empirically, it is effective on a range of synthetic and real-world datasets, with and without distribution drifts, achieving superior performance without hyperparameter tuning. Finally, we extend all OPS ideas to the beta scaling method.

----

## [491] Multi-Task Structural Learning using Local Task Similarity induced Neuron Creation and Removal

**Authors**: *NareshKumar Gurulingan, Bahram Zonooz, Elahe Arani*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gurulingan23a.html](https://proceedings.mlr.press/v202/gurulingan23a.html)

**Abstract**:

Multi-task learning has the potential to improve generalization by maximizing positive transfer between tasks while reducing task interference. Fully achieving this potential is hindered by manually designed architectures that remain static throughout training. On the contrary, learning in the brain occurs through structural changes that are in tandem with changes in synaptic strength. Thus, we propose Multi-Task Structural Learning (MTSL) that simultaneously learns the multi-task architecture and its parameters. MTSL begins with an identical single-task network for each task and alternates between a task-learning phase and a structural-learning phase. In the task learning phase, each network specializes in the corresponding task. In each of the structural learning phases, starting from the earliest layer, locally similar task layers first transfer their knowledge to a newly created group layer before being removed. MTSL then uses the group layer in place of the corresponding removed task layers and moves on to the next layers. Our empirical results show that MTSL achieves competitive generalization with various baselines and improves robustness to out-of-distribution data.

----

## [492] Conditionally Strongly Log-Concave Generative Models

**Authors**: *Florentin Guth, Etienne Lempereur, Joan Bruna, Stéphane Mallat*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guth23a.html](https://proceedings.mlr.press/v202/guth23a.html)

**Abstract**:

There is a growing gap between the impressive results of deep image generative models and classical algorithms that offer theoretical guarantees. The former suffer from mode collapse or memorization issues, limiting their application to scientific data. The latter require restrictive assumptions such as log-concavity to escape the curse of dimensionality. We partially bridge this gap by introducing conditionally strongly log-concave (CSLC) models, which factorize the data distribution into a product of conditional probability distributions that are strongly log-concave. This factorization is obtained with orthogonal projectors adapted to the data distribution. It leads to efficient parameter estimation and sampling algorithms, with theoretical guarantees, although the data distribution is not globally log-concave. We show that several challenging multiscale processes are conditionally log-concave using wavelet packet orthogonal projectors. Numerical results are shown for physical fields such as the $\varphi^4$ model and weak lensing convergence maps with higher resolution than in previous works.

----

## [493] DRew: Dynamically Rewired Message Passing with Delay

**Authors**: *Benjamin Gutteridge, Xiaowen Dong, Michael M. Bronstein, Francesco Di Giovanni*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/gutteridge23a.html](https://proceedings.mlr.press/v202/gutteridge23a.html)

**Abstract**:

Message passing neural networks (MPNNs) have been shown to suffer from the phenomenon of over-squashing that causes poor performance for tasks relying on long-range interactions. This can be largely attributed to message passing only occurring locally, over a node’s immediate neighbours. Rewiring approaches attempting to make graphs ’more connected’, and supposedly better suited to long-range tasks, often lose the inductive bias provided by distance on the graph since they make distant nodes communicate instantly at every layer. In this paper we propose a framework, applicable to any MPNN architecture, that performs a layer-dependent rewiring to ensure gradual densification of the graph. We also propose a delay mechanism that permits skip connections between nodes depending on the layer and their mutual distance. We validate our approach on several long-range tasks and show that it outperforms graph Transformers and multi-hop MPNNs.

----

## [494] Kernel Logistic Regression Approximation of an Understandable ReLU Neural Network

**Authors**: *Marie Guyomard, Susana Barbosa, Lionel Fillatre*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/guyomard23a.html](https://proceedings.mlr.press/v202/guyomard23a.html)

**Abstract**:

This paper proposes an understandable neural network whose score function is modeled as an additive sum of univariate spline functions. It extends usual understandable models like generative additive models, spline-based models, and neural additive models. It is shown that this neural network can be approximated by a logistic regression whose inputs are obtained with a non-linear preprocessing of input data. This preprocessing depends on the neural network initialization but this paper establishes that it can be replaced by a non random kernel-based preprocessing that no longer depends on the initialization. Hence, the convergence of the training process is guaranteed and the solution is unique for a given training dataset.

----

## [495] Conformal Prediction Sets for Graph Neural Networks

**Authors**: *Soroush H. Zargarbashi, Simone Antonelli, Aleksandar Bojchevski*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/h-zargarbashi23a.html](https://proceedings.mlr.press/v202/h-zargarbashi23a.html)

**Abstract**:

Despite the widespread use of graph neural networks (GNNs) we lack methods to reliably quantify their uncertainty. We propose a conformal procedure to equip GNNs with prediction sets that come with distribution-free guarantees – the output set contains the true label with arbitrarily high probability. Our post-processing procedure can wrap around any (pretrained) GNN, and unlike existing methods, results in meaningful sets even when the model provides only the top class. The key idea is to diffuse the node-wise conformity scores to incorporate neighborhood information. By leveraging the network homophily we construct sets with comparable or better efficiency (average size) and significantly improved singleton hit ratio (correct sets of size one). In addition to an extensive empirical evaluation, we investigate the theoretical conditions under which smoothing provably improves efficiency.

----

## [496] Social learning spontaneously emerges by searching optimal heuristics with deep reinforcement learning

**Authors**: *Seungwoong Ha, Hawoong Jeong*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ha23a.html](https://proceedings.mlr.press/v202/ha23a.html)

**Abstract**:

How have individuals of social animals in nature evolved to learn from each other, and what would be the optimal strategy for such learning in a specific environment? Here, we address both problems by employing a deep reinforcement learning model to optimize the social learning strategies (SLSs) of agents in a cooperative game in a multi-dimensional landscape. Throughout the training for maximizing the overall payoff, we find that the agent spontaneously learns various concepts of social learning, such as copying, focusing on frequent and well-performing neighbors, self-comparison, long-term cooperation between agents, and the importance of balancing between individual and social learning, without any explicit guidance or prior knowledge about the system. The SLS from a fully trained agent outperforms all of the traditional, baseline SLSs in terms of mean payoff. We demonstrate the superior performance of the reinforcement learning agent in various environments, including temporally changing environments and real social networks, which also verifies the adaptability of our framework to different social settings.

----

## [497] Convex Geometry of ReLU-layers, Injectivity on the Ball and Local Reconstruction

**Authors**: *Daniel Haider, Martin Ehler, Péter Balázs*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/haider23a.html](https://proceedings.mlr.press/v202/haider23a.html)

**Abstract**:

The paper uses a frame-theoretic setting to study the injectivity of a ReLU-layer on the closed ball of $\mathbb{R}^n$ and its non-negative part. In particular, the interplay between the radius of the ball and the bias vector is emphasized. Together with a perspective from convex geometry, this leads to a computationally feasible method of verifying the injectivity of a ReLU-layer under reasonable restrictions in terms of an upper bound of the bias vector. Explicit reconstruction formulas are provided, inspired by the duality concept from frame theory. All this gives rise to the possibility of quantifying the invertibility of a ReLU-layer and a concrete reconstruction algorithm for any input vector on the ball.

----

## [498] Robust Counterfactual Explanations for Neural Networks With Probabilistic Guarantees

**Authors**: *Faisal Hamman, Erfaun Noorani, Saumitra Mishra, Daniele Magazzeni, Sanghamitra Dutta*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hamman23a.html](https://proceedings.mlr.press/v202/hamman23a.html)

**Abstract**:

There is an emerging interest in generating robust counterfactual explanations that would remain valid if the model is updated or changed even slightly. Towards finding robust counterfactuals, existing literature often assumes that the original model $m$ and the new model $M$ are bounded in the parameter space, i.e., $\|\text{Params}(M){-}\text{Params}(m)\|{<}\Delta$. However, models can often change significantly in the parameter space with little to no change in their predictions or accuracy on the given dataset. In this work, we introduce a mathematical abstraction termed naturally-occurring model change, which allows for arbitrary changes in the parameter space such that the change in predictions on points that lie on the data manifold is limited. Next, we propose a measure – that we call Stability – to quantify the robustness of counterfactuals to potential model changes for differentiable models, e.g., neural networks. Our main contribution is to show that counterfactuals with sufficiently high value of Stability as defined by our measure will remain valid after potential “naturally-occurring” model changes with high probability (leveraging concentration bounds for Lipschitz function of independent Gaussians). Since our quantification depends on the local Lipschitz constant around a data point which is not always available, we also examine practical relaxations of our proposed measure and demonstrate experimentally how they can be incorporated to find robust counterfactuals for neural networks that are close, realistic, and remain valid after potential model changes.

----

## [499] Wrapped Cauchy Distributed Angular Softmax for Long-Tailed Visual Recognition

**Authors**: *Boran Han*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/han23a.html](https://proceedings.mlr.press/v202/han23a.html)

**Abstract**:

Addressing imbalanced or long-tailed data is a major challenge in visual recognition tasks due to disparities between training and testing distributions and issues with data noise. We propose the Wrapped Cauchy Distributed Angular Softmax (WCDAS), a novel softmax function that incorporates data-wise Gaussian-based kernels into the angular correlation between feature representations and classifier weights, effectively mitigating noise and sparse sampling concerns. The class-wise distribution of angular representation becomes a sum of these kernels. Our theoretical analysis reveals that the wrapped Cauchy distribution excels the Gaussian distribution in approximating mixed distributions. Additionally, WCDAS uses trainable concentration parameters to dynamically adjust the compactness and margin of each class. Empirical results confirm label-aware behavior in these parameters and demonstrate WCDAS’s superiority over other state-of-the-art softmax-based methods in handling long-tailed visual recognition across multiple benchmark datasets. The code is public available.

----

## [500] On the Impact of Knowledge Distillation for Model Interpretability

**Authors**: *Hyeongrok Han, Siwon Kim, Hyun-Soo Choi, Sungroh Yoon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/han23b.html](https://proceedings.mlr.press/v202/han23b.html)

**Abstract**:

Several recent studies have elucidated why knowledge distillation (KD) improves model performance. However, few have researched the other advantages of KD in addition to its improving model performance. In this study, we have attempted to show that KD enhances the interpretability as well as the accuracy of models. We measured the number of concept detectors identified in network dissection for a quantitative comparison of model interpretability. We attributed the improvement in interpretability to the class-similarity information transferred from the teacher to student models. First, we confirmed the transfer of class-similarity information from the teacher to student model via logit distillation. Then, we analyzed how class-similarity information affects model interpretability in terms of its presence or absence and degree of similarity information. We conducted various quantitative and qualitative experiments and examined the results on different datasets, different KD methods, and according to different measures of interpretability. Our research showed that KD models by large models could be used more reliably in various fields. The code is available at https://github.com/Rok07/KD_XAI.git.

----

## [501] Alternately Optimized Graph Neural Networks

**Authors**: *Haoyu Han, Xiaorui Liu, Haitao Mao, MohamadAli Torkamani, Feng Shi, Victor Lee, Jiliang Tang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/han23c.html](https://proceedings.mlr.press/v202/han23c.html)

**Abstract**:

Graph Neural Networks (GNNs) have greatly advanced the semi-supervised node classification task on graphs. The majority of existing GNNs are trained in an end-to-end manner that can be viewed as tackling a bi-level optimization problem. This process is often inefficient in computation and memory usage. In this work, we propose a new optimization framework for semi-supervised learning on graphs from a multi-view learning perspective. The proposed framework can be conveniently solved by the alternating optimization algorithms, resulting in significantly improved efficiency. Extensive experiments demonstrate that the proposed method can achieve comparable or better performance with state-of-the-art baselines while it has significantly better computation and memory efficiency.

----

## [502] System Identification of Neural Systems: If We Got It Right, Would We Know?

**Authors**: *Yena Han, Tomaso A. Poggio, Brian Cheung*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/han23d.html](https://proceedings.mlr.press/v202/han23d.html)

**Abstract**:

Artificial neural networks are being proposed as models of parts of the brain. The networks are compared to recordings of biological neurons, and good performance in reproducing neural responses is considered to support the model’s validity. A key question is how much this system identification approach tells us about brain computation. Does it validate one model architecture over another? We evaluate the most commonly used comparison techniques, such as a linear encoding model and centered kernel alignment, to correctly identify a model by replacing brain recordings with known ground truth models. System identification performance is quite variable; it also depends significantly on factors independent of the ground truth architecture, such as stimuli images. In addition, we show the limitations of using functional similarity scores in identifying higher-level architectural motifs.

----

## [503] Total Variation Graph Neural Networks

**Authors**: *Jonas Berg Hansen, Filippo Maria Bianchi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hansen23a.html](https://proceedings.mlr.press/v202/hansen23a.html)

**Abstract**:

Recently proposed Graph Neural Networks (GNNs) for vertex clustering are trained with an unsupervised minimum cut objective, approximated by a Spectral Clustering (SC) relaxation. However, the SC relaxation is loose and, while it offers a closed-form solution, it also yields overly smooth cluster assignments that poorly separate the vertices. In this paper, we propose a GNN model that computes cluster assignments by optimizing a tighter relaxation of the minimum cut based on graph total variation (GTV). The cluster assignments can be used directly to perform vertex clustering or to implement graph pooling in a graph classification framework. Our model consists of two core components: i) a message-passing layer that minimizes the $\ell_1$ distance in the features of adjacent vertices, which is key to achieving sharp transitions between clusters; ii) an unsupervised loss function that minimizes the GTV of the cluster assignments while ensuring balanced partitions. Experimental results show that our model outperforms other GNNs for vertex clustering and graph classification.

----

## [504] Learning Physical Models that Can Respect Conservation Laws

**Authors**: *Derek Hansen, Danielle C. Maddix, Shima Alizadeh, Gaurav Gupta, Michael W. Mahoney*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hansen23b.html](https://proceedings.mlr.press/v202/hansen23b.html)

**Abstract**:

Recent work in scientific machine learning (SciML) has focused on incorporating partial differential equation (PDE) information into the learning process. Much of this work has focused on relatively "easy” PDE operators (e.g., elliptic and parabolic), with less emphasis on relatively “hard” PDE operators (e.g., hyperbolic). Within numerical PDEs, the latter problem class requires control of a type of volume element or conservation constraint, which is known to be challenging. Delivering on the promise of SciML requires seamlessly incorporating both types of problems into the learning process. To address this issue, we propose ProbConserv, a framework for incorporating constraints into a generic SciML architecture. To do so, ProbConserv combines the integral form of a conservation law with a Bayesian update. We provide a detailed analysis of ProbConserv on learning with the Generalized Porous Medium Equation (GPME), a widely-applicable parameterized family of PDEs that illustrates the qualitative properties of both easier and harder PDEs. ProbConserv is effective for easy GPME variants, performing well with state-of-the-art competitors; and for harder GPME variants it outperforms other approaches that do not guarantee volume conservation. ProbConserv seamlessly enforces physical conservation constraints, maintains probabilistic uncertainty quantification (UQ), and deals well with shocks and heteroscedasticity. In each case, it achieves superior predictive performance on downstream tasks.

----

## [505] On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline

**Authors**: *Nicklas Hansen, Zhecheng Yuan, Yanjie Ze, Tongzhou Mu, Aravind Rajeswaran, Hao Su, Huazhe Xu, Xiaolong Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hansen23c.html](https://proceedings.mlr.press/v202/hansen23c.html)

**Abstract**:

In this paper, we examine the effectiveness of pre-training for visuo-motor control tasks. We revisit a simple Learning-from-Scratch (LfS) baseline that incorporates data augmentation and a shallow ConvNet, and find that this baseline is surprisingly competitive with recent approaches (PVR, MVP, R3M) that leverage frozen visual representations trained on large-scale vision datasets – across a variety of algorithms, task domains, and metrics in simulation and on a real robot. Our results demonstrate that these methods are hindered by a significant domain gap between the pre-training datasets and current benchmarks for visuo-motor control, which is alleviated by finetuning. Based on our findings, we provide recommendations for future research in pre-training for control and hope that our simple yet strong baseline will aid in accurately benchmarking progress in this area. Code: https://github.com/gemcollector/learning-from-scratch.

----

## [506] Leveraging Demonstrations to Improve Online Learning: Quality Matters

**Authors**: *Botao Hao, Rahul Jain, Tor Lattimore, Benjamin Van Roy, Zheng Wen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hao23a.html](https://proceedings.mlr.press/v202/hao23a.html)

**Abstract**:

We investigate the extent to which offline demonstration data can improve online learning. It is natural to expect some improvement, but the question is how, and by how much? We show that the degree of improvement must depend on the quality of the demonstration data. To generate portable insights, we focus on Thompson sampling (TS) applied to a multi-armed bandit as a prototypical online learning algorithm and model. The demonstration data is generated by an expert with a given competence level, a notion we introduce. We propose an informed TS algorithm that utilizes the demonstration data in a coherent way through Bayes’ rule and derive a prior-dependent Bayesian regret bound. This offers insight into how pretraining can greatly improve online performance and how the degree of improvement increases with the expert’s competence level. We also develop a practical, approximate informed TS algorithm through Bayesian bootstrapping and show substantial empirical regret reduction through experiments.

----

## [507] Coupled Variational Autoencoder

**Authors**: *Xiaoran Hao, Patrick Shafto*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hao23b.html](https://proceedings.mlr.press/v202/hao23b.html)

**Abstract**:

Variational auto-encoders are powerful probabilistic models in generative tasks but suffer from generating low-quality samples which are caused by the holes in the prior. We propose the Coupled Variational Auto-Encoder (C-VAE), which formulates the VAE problem as one of Optimal Transport (OT) between the prior and data distributions. The C-VAE allows greater flexibility in priors and natural resolution of the prior hole problem by enforcing coupling between the prior and the data distribution and enables flexible optimization through the primal, dual, and semi-dual formulations of entropic OT. Simulations on synthetic and real data show that the C-VAE outperforms alternatives including VAE, WAE, and InfoVAE in fidelity to the data, quality of the latent representation, and in quality of generated samples.

----

## [508] GNOT: A General Neural Operator Transformer for Operator Learning

**Authors**: *Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, Jun Zhu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hao23c.html](https://proceedings.mlr.press/v202/hao23c.html)

**Abstract**:

Learning partial differential equations’ (PDEs) solution operators is an essential problem in machine learning. However, there are several challenges for learning operators in practical applications like the irregular mesh, multiple input functions, and complexity of the PDEs’ solution. To address these challenges, we propose a general neural operator transformer (GNOT), a scalable and effective transformer-based framework for learning operators. By designing a novel heterogeneous normalized attention layer, our model is highly flexible to handle multiple input functions and irregular meshes. Besides, we introduce a geometric gating mechanism which could be viewed as a soft domain decomposition to solve the multi-scale problems. The large model capacity of the transformer architecture grants our model the possibility to scale to large datasets and practical problems. We conduct extensive experiments on multiple challenging datasets from different domains and achieve a remarkable improvement compared with alternative methods. Our code and data are publicly available at https://github.com/thu-ml/GNOT.

----

## [509] Algorithmic Collective Action in Machine Learning

**Authors**: *Moritz Hardt, Eric Mazumdar, Celestine Mendler-Dünner, Tijana Zrnic*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hardt23a.html](https://proceedings.mlr.press/v202/hardt23a.html)

**Abstract**:

We initiate a principled study of algorithmic collective action on digital platforms that deploy machine learning algorithms. We propose a simple theoretical model of a collective interacting with a firm’s learning algorithm. The collective pools the data of participating individuals and executes an algorithmic strategy by instructing participants how to modify their own data to achieve a collective goal. We investigate the consequences of this model in three fundamental learning-theoretic settings: nonparametric optimal learning, parametric risk minimization, and gradient-based optimization. In each setting, we come up with coordinated algorithmic strategies and characterize natural success criteria as a function of the collective’s size. Complementing our theory, we conduct systematic experiments on a skill classification task involving tens of thousands of resumes from a gig platform for freelancers. Through more than two thousand model training runs of a BERT-like language model, we see a striking correspondence emerge between our empirical observations and the predictions made by our theory. Taken together, our theory and experiments broadly support the conclusion that algorithmic collectives of exceedingly small fractional size can exert significant control over a platform’s learning algorithm.

----

## [510] Gaussian Process Priors for Systems of Linear Partial Differential Equations with Constant Coefficients

**Authors**: *Marc Härkönen, Markus Lange-Hegermann, Bogdan Raita*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/harkonen23a.html](https://proceedings.mlr.press/v202/harkonen23a.html)

**Abstract**:

Partial differential equations (PDEs) are important tools to model physical systems and including them into machine learning models is an important way of incorporating physical knowledge. Given any system of linear PDEs with constant coefficients, we propose a family of Gaussian process (GP) priors, which we call EPGP, such that all realizations are exact solutions of this system. We apply the Ehrenpreis-Palamodov fundamental principle, which works as a non-linear Fourier transform, to construct GP kernels mirroring standard spectral methods for GPs. Our approach can infer probable solutions of linear PDE systems from any data such as noisy measurements, or pointwise defined initial and boundary conditions. Constructing EPGP-priors is algorithmic, generally applicable, and comes with a sparse version (S-EPGP) that learns the relevant spectral frequencies and works better for big data sets. We demonstrate our approach on three families of systems of PDEs, the heat equation, wave equation, and Maxwell’s equations, where we improve upon the state of the art in computation time and precision, in some experiments by several orders of magnitude.

----

## [511] Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting

**Authors**: *Hilaf Hasson, Danielle C. Maddix, Bernie Wang, Gaurav Gupta, Youngsuk Park*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hasson23a.html](https://proceedings.mlr.press/v202/hasson23a.html)

**Abstract**:

Ensembling is among the most popular tools in machine learning (ML) due to its effectiveness in minimizing variance and thus improving generalization. Most ensembling methods for black-box base learners fall under the umbrella of "stacked generalization," namely training an ML algorithm that takes the inferences from the base learners as input. While stacking has been widely applied in practice, its theoretical properties are poorly understood. In this paper, we prove a novel result, showing that choosing the best stacked generalization from a (finite or finite-dimensional) family of stacked generalizations based on cross-validated performance does not perform "much worse" than the oracle best. Our result strengthens and significantly extends the results in Van der Laan et al. (2007). Inspired by the theoretical analysis, we further propose a particular family of stacked generalizations in the context of probabilistic forecasting, each one with a different sensitivity for how much the ensemble weights are allowed to vary across items, timestamps in the forecast horizon, and quantiles. Experimental results demonstrate the performance gain of the proposed method.

----

## [512] Global Context Vision Transformers

**Authors**: *Ali Hatamizadeh, Hongxu Yin, Greg Heinrich, Jan Kautz, Pavlo Molchanov*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hatamizadeh23a.html](https://proceedings.mlr.press/v202/hatamizadeh23a.html)

**Abstract**:

We propose global context vision transformer (GC ViT), a novel architecture that enhances parameter and compute utilization for computer vision. Our method leverages global context self-attention modules, joint with standard local self-attention, to effectively and efficiently model both long and short-range spatial interactions, without the need for expensive operations such as computing attention masks or shifting local windows. In addition, we address the lack of the inductive bias in ViTs, and propose to leverage a modified fused inverted residual blocks in our architecture. Our proposed GC ViT achieves state-of-the-art results across image classification, object detection and semantic segmentation tasks. On ImageNet-1K dataset for classification, the variants of GC ViT with 51M, 90M and 201M parameters achieve 84.3%, 85.0% and 85.7% Top-1 accuracy, respectively, at 224 image resolution and without any pre-training, hence surpassing comparably-sized prior art such as CNN-based ConvNeXt and ViT-based MaxViT and Swin Transformer by a large margin. Pre-trained GC ViT backbones in downstream tasks of object detection, instance segmentation, and semantic segmentation using MS COCO and ADE20K datasets outperform prior work consistently. Specifically, GC ViT with a 4-scale DINO detection head achieves a box AP of 58.3 on MS COCO dataset.

----

## [513] Counterfactual Analysis in Dynamic Latent State Models

**Authors**: *Martin B. Haugh, Raghav Singal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/haugh23a.html](https://proceedings.mlr.press/v202/haugh23a.html)

**Abstract**:

We provide an optimization-based framework to perform counterfactual analysis in a dynamic model with hidden states. Our framework is grounded in the “abduction, action, and prediction” approach to answer counterfactual queries and handles two key challenges where (1) the states are hidden and (2) the model is dynamic. Recognizing the lack of knowledge on the underlying causal mechanism and the possibility of infinitely many such mechanisms, we optimize over this space and compute upper and lower bounds on the counterfactual quantity of interest. Our work brings together ideas from causality, state-space models, simulation, and optimization, and we apply it on a breast cancer case study. To the best of our knowledge, we are the first to compute lower and upper bounds on a counterfactual query in a dynamic latent-state model.

----

## [514] Sampling-based Nyström Approximation and Kernel Quadrature

**Authors**: *Satoshi Hayakawa, Harald Oberhauser, Terry J. Lyons*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hayakawa23a.html](https://proceedings.mlr.press/v202/hayakawa23a.html)

**Abstract**:

We analyze the Nyström approximation of a positive definite kernel associated with a probability measure. We first prove an improved error bound for the conventional Nyström approximation with i.i.d. sampling and singular-value decomposition in the continuous regime; the proof techniques are borrowed from statistical learning theory. We further introduce a refined selection of subspaces in Nyström approximation with theoretical guarantees that is applicable to non-i.i.d. landmark points. Finally, we discuss their application to convex kernel quadrature and give novel theoretical guarantees as well as numerical observations.

----

## [515] Width and Depth Limits Commute in Residual Networks

**Authors**: *Soufiane Hayou, Greg Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hayou23a.html](https://proceedings.mlr.press/v202/hayou23a.html)

**Abstract**:

We show that taking the width and depth to infinity in a deep neural network with skip connections, when branches are scaled by $1/\sqrt{depth}$, result in the same covariance structure no matter how that limit is taken. This explains why the standard infinite-width-then-depth approach provides practical insights even for networks with depth of the same order as width. We also demonstrate that the pre-activations, in this case, have Gaussian distributions which has direct applications in Bayesian deep learning. We conduct extensive simulations that show an excellent match with our theoretical findings.

----

## [516] A Generalization of ViT/MLP-Mixer to Graphs

**Authors**: *Xiaoxin He, Bryan Hooi, Thomas Laurent, Adam Perold, Yann LeCun, Xavier Bresson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/he23a.html](https://proceedings.mlr.press/v202/he23a.html)

**Abstract**:

Graph Neural Networks (GNNs) have shown great potential in the field of graph representation learning. Standard GNNs define a local message-passing mechanism which propagates information over the whole graph domain by stacking multiple layers. This paradigm suffers from two major limitations, over-squashing and poor long-range dependencies, that can be solved using global attention but significantly increases the computational cost to quadratic complexity. In this work, we propose an alternative approach to overcome these structural limitations by leveraging the ViT/MLP-Mixer architectures introduced in computer vision. We introduce a new class of GNNs, called Graph ViT/MLP-Mixer, that holds three key properties. First, they capture long-range dependency and mitigate the issue of over-squashing as demonstrated on Long Range Graph Benchmark and TreeNeighbourMatch datasets. Second, they offer better speed and memory efficiency with a complexity linear to the number of nodes and edges, surpassing the related Graph Transformer and expressive GNN models. Third, they show high expressivity in terms of graph isomorphism as they can distinguish at least 3-WL non-isomorphic graphs. We test our architecture on 4 simulated datasets and 7 real-world benchmarks, and show highly competitive results on all of them. The source code is available for reproducibility at: https://github.com/XiaoxinHe/Graph-ViT-MLPMixer.

----

## [517] Domain Adaptation for Time Series Under Feature and Label Shifts

**Authors**: *Huan He, Owen Queen, Teddy Koker, Consuelo Cuevas, Theodoros Tsiligkaridis, Marinka Zitnik*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/he23b.html](https://proceedings.mlr.press/v202/he23b.html)

**Abstract**:

Unsupervised domain adaptation (UDA) enables the transfer of models trained on source domains to unlabeled target domains. However, transferring complex time series models presents challenges due to the dynamic temporal structure variations across domains. This leads to feature shifts in the time and frequency representations. Additionally, the label distributions of tasks in the source and target domains can differ significantly, posing difficulties in addressing label shifts and recognizing labels unique to the target domain. Effectively transferring complex time series models remains a formidable problem. We present RAINCOAT, the first model for both closed-set and universal domain adaptation on complex time series. RAINCOAT addresses feature and label shifts by considering both temporal and frequency features, aligning them across domains, and correcting for misalignments to facilitate the detection of private labels. Additionally, RAINCOAT improves transferability by identifying label shifts in target domains. Our experiments with 5 datasets and 13 state-of-the-art UDA methods demonstrate that RAINCOAT can improve transfer learning performance by up to 16.33% and can handle both closed-set and universal domain adaptation.

----

## [518] Contrastive Learning Meets Homophily: Two Birds with One Stone

**Authors**: *Dongxiao He, Jitao Zhao, Rui Guo, Zhiyong Feng, Di Jin, Yuxiao Huang, Zhen Wang, Weixiong Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/he23c.html](https://proceedings.mlr.press/v202/he23c.html)

**Abstract**:

Graph Contrastive Learning (GCL) has recently enjoyed great success as an efficient self-supervised representation learning approach. However, the existing methods have focused on designing of contrastive modes and used data augmentation with a rigid and inefficient one-to-one sampling strategy. We adopted node neighborhoods to extend positive samplings and made avoided resorting to data augmentation to create different views. We also considered the homophily problem in Graph Neural Networks (GNNs) between the inter-class node pairs. The key novelty of our method hinged upon analyzing this GNNs problem and integrating the GCL sampling strategy with homophily discrimination, where we solved these two significant problems using one approach. We introduced a new parameterized neighbor sampling component to replace the conventional sub-optimal samplings. By keeping and updating the neighbor sets, both the positive sampling of GCL and the message passing of GNNs can be optimized. Moreover, we theoretically proved that the new method provided a lower bound of mutual information for unsupervised semantic learning, and it can also keep the lower bound with downstream tasks. In essence, our method is a new self-supervised approach, which we refer to as group discrimination, and it can make the downstream fine-tuning efficient. Our extensive empirical results demonstrate that the new method can significantly outperform the existing GCL methods because the former can solve the homophily problem in a self-supervised way with the new group discrimination method used.

----

## [519] Nearly Minimax Optimal Reinforcement Learning for Linear Markov Decision Processes

**Authors**: *Jiafan He, Heyang Zhao, Dongruo Zhou, Quanquan Gu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/he23d.html](https://proceedings.mlr.press/v202/he23d.html)

**Abstract**:

We study reinforcement learning (RL) with linear function approximation. For episodic time-inhomogeneous linear Markov decision processes (linear MDPs) whose transition probability can be parameterized as a linear function of a given feature mapping, we propose the first computationally efficient algorithm that achieves the nearly minimax optimal regret $\tilde O(d\sqrt{H^3K})$, where $d$ is the dimension of the feature mapping, $H$ is the planning horizon, and $K$ is the number of episodes. Our algorithm is based on a weighted linear regression scheme with a carefully designed weight, which depends on a new variance estimator that (1) directly estimates the variance of the optimal value function, (2) monotonically decreases with respect to the number of episodes to ensure a better estimation accuracy, and (3) uses a rare-switching policy to update the value function estimator to control the complexity of the estimated value function class. Our work provides a complete answer to optimal RL with linear MDPs, and the developed algorithm and theoretical tools may be of independent interest.

----

## [520] CRISP: Curriculum based Sequential neural decoders for Polar code family

**Authors**: *S. Ashwin Hebbar, Viraj Vivek Nadkarni, Ashok Vardhan Makkuva, Suma Bhat, Sewoong Oh, Pramod Viswanath*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hebbar23a.html](https://proceedings.mlr.press/v202/hebbar23a.html)

**Abstract**:

Polar codes are widely used state-of-the-art codes for reliable communication that have recently been included in the $5^{\text{th}}$ generation wireless standards ($5$G). However, there still remains room for design of polar decoders that are both efficient and reliable in the short blocklength regime. Motivated by recent successes of data-driven channel decoders, we introduce a novel $\textbf{ C}$ur${\textbf{RI}}$culum based $\textbf{S}$equential neural decoder for $\textbf{P}$olar codes (CRISP). We design a principled curriculum, guided by information-theoretic insights, to train CRISP and show that it outperforms the successive-cancellation (SC) decoder and attains near-optimal reliability performance on the $\text{Polar}(32,16)$ and $\text{Polar}(64,22)$ codes. The choice of the proposed curriculum is critical in achieving the accuracy gains of CRISP, as we show by comparing against other curricula. More notably, CRISP can be readily extended to Polarization-Adjusted-Convolutional (PAC) codes, where existing SC decoders are significantly less reliable. To the best of our knowledge, CRISP constructs the first data-driven decoder for PAC codes and attains near-optimal performance on the $\text{PAC}(32,16)$ code.

----

## [521] Sketch-Flip-Merge: Mergeable Sketches for Private Distinct Counting

**Authors**: *Jonathan Hehir, Daniel Ting, Graham Cormode*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hehir23a.html](https://proceedings.mlr.press/v202/hehir23a.html)

**Abstract**:

Data sketching is a critical tool for distinct counting, enabling multisets to be represented by compact summaries that admit fast cardinality estimates. Because sketches may be merged to summarize multiset unions, they are a basic building block in data warehouses. Although many practical sketches for cardinality estimation exist, none provide privacy when merging. We propose the first practical cardinality sketches that are simultaneously mergeable, differentially private (DP), and have low empirical errors. These introduce a novel randomized algorithm for performing logical operations on noisy bits, a tight privacy analysis, and provably optimal estimation. Our sketches dramatically outperform existing theoretical solutions in simulations and on real-world data.

----

## [522] Functional Neural Networks: Shift invariant models for functional data with applications to EEG classification

**Authors**: *Florian Heinrichs, Mavin Heim, Corinna Weber*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/heinrichs23a.html](https://proceedings.mlr.press/v202/heinrichs23a.html)

**Abstract**:

It is desirable for statistical models to detect signals of interest independently of their position. If the data is generated by some smooth process, this additional structure should be taken into account. We introduce a new class of neural networks that are shift invariant and preserve smoothness of the data: functional neural networks (FNNs). For this, we use methods from functional data analysis (FDA) to extend multi-layer perceptrons and convolutional neural networks to functional data. We propose different model architectures, show that the models outperform a benchmark model from FDA in terms of accuracy and successfully use FNNs to classify electroencephalography (EEG) data.

----

## [523] Distance Weighted Supervised Learning for Offline Interaction Data

**Authors**: *Joey Hejna, Jensen Gao, Dorsa Sadigh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hejna23a.html](https://proceedings.mlr.press/v202/hejna23a.html)

**Abstract**:

Sequential decision making algorithms often struggle to leverage different sources of unstructured offline interaction data. Imitation learning (IL) methods based on supervised learning are robust, but require optimal demonstrations, which are hard to collect. Offline goal-conditioned reinforcement learning (RL) algorithms promise to learn from sub-optimal data, but face optimization challenges especially with high-dimensional data. To bridge the gap between IL and RL, we introduce Distance Weighted Supervised Learning or DWSL, a supervised method for learning goal-conditioned policies from offline data. DWSL models the entire distribution of time-steps between states in offline data with only supervised learning, and uses this distribution to approximate shortest path distances. To extract a policy, we weight actions by their reduction in distance estimates. Theoretically, DWSL converges to an optimal policy constrained to the data distribution, an attractive property for offline learning, without any bootstrapping. Across all datasets we test, DWSL empirically maintains behavior cloning as a lower bound while still exhibiting policy improvement. In high-dimensional image domains, DWSL surpasses the performance of both prior goal-conditioned IL and RL algorithms. Visualizations and code can be found at https://sites.google.com/view/dwsl/home.

----

## [524] Group Equivariant Fourier Neural Operators for Partial Differential Equations

**Authors**: *Jacob Helwig, Xuan Zhang, Cong Fu, Jerry Kurtin, Stephan Wojtowytsch, Shuiwang Ji*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/helwig23a.html](https://proceedings.mlr.press/v202/helwig23a.html)

**Abstract**:

We consider solving partial differential equations (PDEs) with Fourier neural operators (FNOs), which operate in the frequency domain. Since the laws of physics do not depend on the coordinate system used to describe them, it is desirable to encode such symmetries in the neural operator architecture for better performance and easier learning. While encoding symmetries in the physical domain using group theory has been studied extensively, how to capture symmetries in the frequency domain is under-explored. In this work, we extend group convolutions to the frequency domain and design Fourier layers that are equivariant to rotations, translations, and reflections by leveraging the equivariance property of the Fourier transform. The resulting $G$-FNO architecture generalizes well across input resolutions and performs well in settings with varying levels of symmetry. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).

----

## [525] Training-Free Neural Active Learning with Initialization-Robustness Guarantees

**Authors**: *Apivich Hemachandra, Zhongxiang Dai, Jasraj Singh, See-Kiong Ng, Bryan Kian Hsiang Low*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hemachandra23a.html](https://proceedings.mlr.press/v202/hemachandra23a.html)

**Abstract**:

Existing neural active learning algorithms have aimed to optimize the predictive performance of neural networks (NNs) by selecting data for labelling. However, other than a good predictive performance, being robust against random parameter initializations is also a crucial requirement in safety-critical applications. To this end, we introduce our expected variance with Gaussian processes (EV-GP) criterion for neural active learning, which is theoretically guaranteed to select data points which lead to trained NNs with both (a) good predictive performances and (b) initialization robustness. Importantly, our EV-GP criterion is training-free, i.e., it does not require any training of the NN during data selection, which makes it computationally efficient. We empirically demonstrate that our EV-GP criterion is highly correlated with both initialization robustness and generalization performance, and show that it consistently outperforms baseline methods in terms of both desiderata, especially in situations with limited initial data or large batch sizes.

----

## [526] A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs

**Authors**: *Mikael Henaff, Minqi Jiang, Roberta Raileanu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/henaff23a.html](https://proceedings.mlr.press/v202/henaff23a.html)

**Abstract**:

Exploration in environments which differ across episodes has received increasing attention in recent years. Current methods use some combination of global novelty bonuses, computed using the agent’s entire training experience, and episodic novelty bonuses, computed using only experience from the current episode. However, the use of these two types of bonuses has been ad-hoc and poorly understood. In this work, we shed light on the behavior of these two types of bonuses through controlled experiments on easily interpretable tasks as well as challenging pixel-based settings. We find that the two types of bonuses succeed in different settings, with episodic bonuses being most effective when there is little shared structure across episodes and global bonuses being effective when more structure is shared. We develop a conceptual framework which makes this notion of shared structure precise by considering the variance of the value function across contexts, and which provides a unifying explanation of our empirical results. We furthermore find that combining the two bonuses can lead to more robust performance across different degrees of shared structure, and investigate different algorithmic choices for defining and combining global and episodic bonuses based on function approximation. This results in an algorithm which sets a new state of the art across 16 tasks from the MiniHack suite used in prior work, and also performs robustly on Habitat and Montezuma’s Revenge.

----

## [527] Robust Camera Pose Refinement for Multi-Resolution Hash Encoding

**Authors**: *Hwan Heo, Taekyung Kim, Jiyoung Lee, Jaewon Lee, Soohyun Kim, Hyunwoo J. Kim, Jin-Hwa Kim*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/heo23a.html](https://proceedings.mlr.press/v202/heo23a.html)

**Abstract**:

Multi-resolution hash encoding has recently been proposed to reduce the computational cost of neural renderings, such as NeRF. This method requires accurate camera poses for the neural renderings of given scenes. However, contrary to previous methods jointly optimizing camera poses and 3D scenes, the naive gradient-based camera pose refinement method using multi-resolution hash encoding severely deteriorates performance. We propose a joint optimization algorithm to calibrate the camera pose and learn a geometric representation using efficient multi-resolution hash encoding. Showing that the oscillating gradient flows of hash encoding interfere with the registration of camera poses, our method addresses the issue by utilizing smooth interpolation weighting to stabilize the gradient oscillation for the ray samplings across hash grids. Moreover, the curriculum training procedure helps to learn the level-wise hash encoding, further increasing the pose refinement. Experiments on the novel-view synthesis datasets validate that our learning frameworks achieve state-of-the-art performance and rapid convergence of neural rendering.

----

## [528] Generalized Teacher Forcing for Learning Chaotic Dynamics

**Authors**: *Florian Hess, Zahra Monfared, Manuel Brenner, Daniel Durstewitz*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hess23a.html](https://proceedings.mlr.press/v202/hess23a.html)

**Abstract**:

Chaotic dynamical systems (DS) are ubiquitous in nature and society. Often we are interested in reconstructing such systems from observed time series for prediction or mechanistic insight, where by reconstruction we mean learning geometrical and invariant temporal properties of the system in question (like attractors). However, training reconstruction algorithms like recurrent neural networks (RNNs) on such systems by gradient-descent based techniques faces severe challenges. This is mainly due to exploding gradients caused by the exponential divergence of trajectories in chaotic systems. Moreover, for (scientific) interpretability we wish to have as low dimensional reconstructions as possible, preferably in a model which is mathematically tractable. Here we report that a surprisingly simple modification of teacher forcing leads to provably strictly all-time bounded gradients in training on chaotic systems, and, when paired with a simple architectural rearrangement of a tractable RNN design, piecewise-linear RNNs (PLRNNs), allows for faithful reconstruction in spaces of at most the dimensionality of the observed system. We show on several DS that with these amendments we can reconstruct DS better than current SOTA algorithms, in much lower dimensions. Performance differences were particularly compelling on real world data with which most other methods severely struggled. This work thus led to a simple yet powerful DS reconstruction algorithm which is highly interpretable at the same time.

----

## [529] Causal Modeling of Policy Interventions From Treatment-Outcome Sequences

**Authors**: *Caglar Hizli, S. T. John, Anne Tuulikki Juuti, Tuure Tapani Saarinen, Kirsi Hannele Pietiläinen, Pekka Marttinen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hizli23a.html](https://proceedings.mlr.press/v202/hizli23a.html)

**Abstract**:

A treatment policy defines when and what treatments are applied to affect some outcome of interest. Data-driven decision-making requires the ability to predict what happens if a policy is changed. Existing methods that predict how the outcome evolves under different scenarios assume that the tentative sequences of future treatments are fixed in advance, while in practice the treatments are determined stochastically by a policy and may depend, for example, on the efficiency of previous treatments. Therefore, the current methods are not applicable if the treatment policy is unknown or a counterfactual analysis is needed. To handle these limitations, we model the treatments and outcomes jointly in continuous time, by combining Gaussian processes and point processes. Our model enables the estimation of a treatment policy from observational sequences of treatments and outcomes, and it can predict the interventional and counterfactual progression of the outcome after an intervention on the treatment policy (in contrast with the causal effect of a single treatment). We show with real-world and semi-synthetic data on blood glucose progression that our method can answer causal queries more accurately than existing alternatives.

----

## [530] Monotonicity and Double Descent in Uncertainty Estimation with Gaussian Processes

**Authors**: *Liam Hodgkinson, Christopher van der Heide, Fred Roosta, Michael W. Mahoney*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hodgkinson23a.html](https://proceedings.mlr.press/v202/hodgkinson23a.html)

**Abstract**:

Despite their importance for assessing reliability of predictions, uncertainty quantification (UQ) measures in machine learning models have only recently begun to be rigorously characterized. One prominent issue is the curse of dimensionality: it is commonly believed that the marginal likelihood should be reminiscent of cross-validation metrics and both should deteriorate with larger input dimensions. However, we prove that by tuning hyperparameters to maximize marginal likelihood (the empirical Bayes procedure), performance, as measured by the marginal likelihood, improves monotonically with the input dimension. On the other hand, cross-validation metrics exhibit qualitatively different behavior that is characteristic of double descent. Cold posteriors, which have recently attracted interest due to their improved performance in certain settings, appear to exacerbate these phenomena. We verify empirically that our results hold for real data, beyond our considered assumptions, and we explore consequences involving synthetic covariates.

----

## [531] AdaBoost is not an Optimal Weak to Strong Learner

**Authors**: *Mikael Møller Høgsgaard, Kasper Green Larsen, Martin Ritzert*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hogsgaard23a.html](https://proceedings.mlr.press/v202/hogsgaard23a.html)

**Abstract**:

AdaBoost is a classic boosting algorithm for combining multiple inaccurate classifiers produced by a weak learner, to produce a strong learner with arbitrarily high accuracy when given enough training data. Determining the optimal number of samples necessary to obtain a given accuracy of the strong learner, is a basic learning theoretic question. Larsen and Ritzert (NeurIPS’22) recently presented the first provably optimal weak-to-strong learner. However, their algorithm is somewhat complicated and it remains an intriguing question whether the prototypical boosting algorithm AdaBoost also makes optimal use of training samples. In this work, we answer this question in the negative. Concretely, we show that the sample complexity of AdaBoost, and other classic variations thereof, are sub-optimal by at least one logarithmic factor in the desired accuracy of the strong learner.

----

## [532] Dual Propagation: Accelerating Contrastive Hebbian Learning with Dyadic Neurons

**Authors**: *Rasmus Kjær Høier, D. Staudt, Christopher Zach*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hoier23a.html](https://proceedings.mlr.press/v202/hoier23a.html)

**Abstract**:

Activity difference based learning algorithms—such as contrastive Hebbian learning and equilibrium propagation—have been proposed as biologically plausible alternatives to error back-propagation. However, on traditional digital chips these algorithms suffer from having to solve a costly inference problem twice, making these approaches more than two orders of magnitude slower than back-propagation. In the analog realm equilibrium propagation may be promising for fast and energy efficient learning, but states still need to be inferred and stored twice. Inspired by lifted neural networks and compartmental neuron models we propose a simple energy based compartmental neuron model, termed dual propagation, in which each neuron is a dyad with two intrinsic states. At inference time these intrinsic states encode the error/activity duality through their difference and their mean respectively. The advantage of this method is that only a single inference phase is needed and that inference can be solved in layerwise closed-form. Experimentally we show on common computer vision datasets, including Imagenet32x32, that dual propagation performs equivalently to back-propagation both in terms of accuracy and runtime.

----

## [533] Multi-Task Off-Policy Learning from Bandit Feedback

**Authors**: *Joey Hong, Branislav Kveton, Manzil Zaheer, Sumeet Katariya, Mohammad Ghavamzadeh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hong23a.html](https://proceedings.mlr.press/v202/hong23a.html)

**Abstract**:

Many practical problems involve solving similar tasks. In recommender systems, the tasks can be users with similar preferences; in search engines, the tasks can be items with similar affinities. To learn statistically efficiently, the tasks can be organized in a hierarchy, where the task affinity is captured using an unknown latent parameter. We study the problem of off-policy learning for similar tasks from logged bandit feedback. To solve the problem, we propose a hierarchical off-policy optimization algorithm HierOPO. The key idea is to estimate the task parameters using the hierarchy and then act pessimistically with respect to them. To analyze the algorithm, we develop novel Bayesian error bounds. Our bounds are the first in off-policy learning that improve with a more informative prior and capture statistical gains due to hierarchical models. Therefore, they are of a general interest. HierOPO also performs well in practice. Our experiments demonstrate the benefits of using the hierarchy over solving each task independently.

----

## [534] Constrained Optimization via Exact Augmented Lagrangian and Randomized Iterative Sketching

**Authors**: *Ilgee Hong, Sen Na, Michael W. Mahoney, Mladen Kolar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hong23b.html](https://proceedings.mlr.press/v202/hong23b.html)

**Abstract**:

We consider solving equality-constrained nonlinear, nonconvex optimization problems. This class of problems appears widely in a variety of applications in machine learning and engineering, ranging from constrained deep neural networks, to optimal control, to PDE-constrained optimization. We develop an adaptive inexact Newton method for this problem class. In each iteration, we solve the Lagrangian Newton system inexactly via a randomized iterative sketching solver, and select a suitable stepsize by performing line search on an exact augmented Lagrangian merit function. The randomized solvers have advantages over deterministic linear system solvers by significantly reducing per-iteration flops complexity and storage cost, when equipped with suitable sketching matrices. Our method adaptively controls the accuracy of the randomized solver and the penalty parameters of the exact augmented Lagrangian, to ensure that the inexact Newton direction is a descent direction of the exact augmented Lagrangian. This allows us to establish a global almost sure convergence. We also show that a unit stepsize is admissible locally, so that our method exhibits a local linear convergence. Furthermore, we prove that the linear convergence can be strengthened to superlinear convergence if we gradually sharpen the adaptive accuracy condition on the randomized solver. We demonstrate the superior performance of our method on benchmark nonlinear problems in CUTEst test set, constrained logistic regression with data from LIBSVM, and a PDE-constrained problem.

----

## [535] Revisiting Data-Free Knowledge Distillation with Poisoned Teachers

**Authors**: *Junyuan Hong, Yi Zeng, Shuyang Yu, Lingjuan Lyu, Ruoxi Jia, Jiayu Zhou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hong23c.html](https://proceedings.mlr.press/v202/hong23c.html)

**Abstract**:

Data-free knowledge distillation (KD) helps transfer knowledge from a pre-trained model (known as the teacher model) to a smaller model (known as the student model) without access to the original training data used for training the teacher model. However, the security of the synthetic or out-of-distribution (OOD) data required in data-free KD is largely unknown and under-explored. In this work, we make the first effort to uncover the security risk of data-free KD w.r.t. untrusted pre-trained models. We then propose Anti-Backdoor Data-Free KD (ABD), the first plug-in defensive method for data-free KD methods to mitigate the chance of potential backdoors being transferred. We empirically evaluate the effectiveness of our proposed ABD in diminishing transferred backdoor knowledge while maintaining compatible downstream performances as the vanilla KD. We envision this work as a milestone for alarming and mitigating the potential backdoors in data-free KD. Codes are released at https://github.com/illidanlab/ABD .

----

## [536] simple diffusion: End-to-end diffusion for high resolution images

**Authors**: *Emiel Hoogeboom, Jonathan Heek, Tim Salimans*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hoogeboom23a.html](https://proceedings.mlr.press/v202/hoogeboom23a.html)

**Abstract**:

Currently, applying diffusion models in pixel space of high resolution images is difficult. Instead, existing approaches focus on diffusion in lower dimensional spaces (latent diffusion), or have multiple super-resolution levels of generation referred to as cascades. The downside is that these approaches add additional complexity to the diffusion framework. This paper aims to improve denoising diffusion for high resolution images while keeping the model as simple as possible. The paper is centered around the research question: How can one train a standard denoising diffusion models on high resolution images, and still obtain performance comparable to these alternate approaches? The four main findings are: 1) the noise schedule should be adjusted for high resolution images, 2) It is sufficient to scale only a particular part of the architecture, 3) dropout should be added at specific locations in the architecture, and 4) downsampling is an effective strategy to avoid high resolution feature maps. Combining these simple yet effective techniques, we achieve state-of-the-art on image generation among diffusion models without sampling modifiers on ImageNet.

----

## [537] Causal Strategic Classification: A Tale of Two Shifts

**Authors**: *Guy Horowitz, Nir Rosenfeld*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/horowitz23a.html](https://proceedings.mlr.press/v202/horowitz23a.html)

**Abstract**:

When users can benefit from certain predictive outcomes, they may be prone to act to achieve those outcome, e.g., by strategically modifying their features. The goal in strategic classification is therefore to train predictive models that are robust to such behavior. However, the conventional framework assumes that changing features does not change actual outcomes, which depicts users as "gaming" the system. Here we remove this assumption, and study learning in a causal strategic setting where true outcomes do change. Focusing on accuracy as our primary objective, we show how strategic behavior and causal effects underlie two complementing forms of distribution shift. We characterize these shifts, and propose a learning algorithm that balances between these two forces and over time, and permits end-to-end training. Experiments on synthetic and semi-synthetic data demonstrate the utility of our approach.

----

## [538] Fair and Accurate Decision Making through Group-Aware Learning

**Authors**: *Ramtin Hosseini, Li Zhang, Bhanu Garg, Pengtao Xie*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hosseini23a.html](https://proceedings.mlr.press/v202/hosseini23a.html)

**Abstract**:

The integration of machine learning models in various real-world applications is becoming more prevalent to assist humans in their daily decision-making tasks as a result of recent advancements in this field. However, it has been discovered that there is a tradeoff between the accuracy and fairness of these decision-making tasks. In some cases, these AI systems can be unfair by exhibiting bias or discrimination against certain social groups, which can have severe consequences in real life. Inspired by one of the most well-known human learning skills called grouping, we address this issue by proposing a novel machine learning (ML) framework where the ML model learns to group a diverse set of problems into distinct subgroups to solve each subgroup using its specific sub-model. Our proposed framework involves three stages of learning, which are formulated as a three-level optimization problem: 1) grouping problems into subgroups, 2) learning group-specific sub-models for problem-solving, and 3) updating group assignments of training examples by minimizing validation loss. These three learning stages are performed end-to-end in a joint manner using gradient descent. To improve fairness and accuracy, we develop an efficient optimization algorithm to solve this three-level optimization problem. To further decrease the risk of overfitting in small datasets using our LBG method, we incorporate domain adaptation techniques in the second stage of training. We further apply our method to differentiable neural architecture search (NAS) methods.

----

## [539] Approximation Algorithms for Fair Range Clustering

**Authors**: *Sèdjro Salomon Hotegni, Sepideh Mahabadi, Ali Vakilian*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hotegni23a.html](https://proceedings.mlr.press/v202/hotegni23a.html)

**Abstract**:

This paper studies the fair range clustering problem in which the data points are from different demographic groups and the goal is to pick $k$ centers with the minimum clustering cost such that each group is at least minimally represented in the centers set and no group dominates the centers set. More precisely, given a set of $n$ points in a metric space $(P, d)$ where each point belongs to one of the $\ell$ different demographics (i.e., $P = P_1 \uplus P_2 \uplus \cdots \uplus P_\ell$) and a set of $\ell$ intervals $[\alpha_1, \beta_1], \cdots, [\alpha_\ell, \beta_\ell]$ on desired number of centers from each group, the goal is to pick a set of $k$ centers $C$ with minimum $\ell_p$-clustering cost (i.e., $(\sum_{v\in P} d(v,C)^p)^{1/p}$) such that for each group $i\in \ell$, $|C\cap P_i| \in [\alpha_i, \beta_i]$. In particular, the fair range $\ell_p$-clustering captures fair range $k$-center, $k$-median and $k$-means as its special cases. In this work, we provide an efficient constant factor approximation algorithm for the fair range $\ell_p$-clustering for all values of $p\in [1,\infty)$.

----

## [540] Decoding Layer Saliency in Language Transformers

**Authors**: *Elizabeth Mary Hou, Gregory David Castañón*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hou23a.html](https://proceedings.mlr.press/v202/hou23a.html)

**Abstract**:

In this paper, we introduce a strategy for identifying textual saliency in large-scale language models applied to classification tasks. In visual networks where saliency is more well-studied, saliency is naturally localized through the convolutional layers of the network; however, the same is not true in modern transformer-stack networks used to process natural language. We adapt gradient-based saliency methods for these networks, propose a method for evaluating the degree of semantic coherence of each layer, and demonstrate consistent improvement over numerous other methods for textual saliency on multiple benchmark classification datasets. Our approach requires no additional training or access to labelled data, and is comparatively very computationally efficient.

----

## [541] PromptBoosting: Black-Box Text Classification with Ten Forward Passes

**Authors**: *Bairu Hou, Joe O'Connor, Jacob Andreas, Shiyu Chang, Yang Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hou23b.html](https://proceedings.mlr.press/v202/hou23b.html)

**Abstract**:

We describe PromptBoosting, a query-efficient procedure for building a text classifier from a neural language model (LM) without access to the LM’s parameters, gradients, or hidden representations. This form of "black-box" classifier training has become increasingly important as the cost of training and inference in large-scale LMs has grown. But existing black-box LM classifier learning approaches are themselves computationally inefficient, typically specializing LMs to the target task by searching in a large space of (discrete or continuous) prompts using zeroth-order optimization methods. Instead of directly optimizing in prompt space, PromptBoosting obtains a small pool of prompts via a gradient-free approach and then constructs a large pool of weak learners by pairing these prompts with different elements of the LM’s output distribution. These weak learners are then ensembled using the AdaBoost algorithm. The entire learning process requires only a small number of forward passes and no backward pass. Experiments show that PromptBoosting achieves state-of-the-art performance in multiple black-box few-shot classification tasks, and matches or outperforms full fine-tuning in both few-shot and standard learning paradigms, while training 10x faster than existing black-box methods.

----

## [542] Sparse Learning of Dynamical Systems in RKHS: An Operator-Theoretic Approach

**Authors**: *Boya Hou, Sina Sanjari, Nathan Dahlin, Subhonmesh Bose, Umesh Vaidya*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hou23c.html](https://proceedings.mlr.press/v202/hou23c.html)

**Abstract**:

Transfer operators provide a rich framework for representing the dynamics of very general, nonlinear dynamical systems. When interacting with reproducing kernel Hilbert spaces (RKHS), descriptions of dynamics often incur prohibitive data storage requirements, motivating dataset sparsification as a precursory step to computation. Further, in practice, data is available in the form of trajectories, introducing correlation between samples. In this work, we present a method for sparse learning of transfer operators from $\beta$-mixing stochastic processes, in both discrete and continuous time, and provide sample complexity analysis extending existing theoretical guarantees for learning from non-sparse, i.i.d. data. In addressing continuous-time settings, we develop precise descriptions using covariance-type operators for the infinitesimal generator that aids in the sample complexity analysis. We empirically illustrate the efficacy of our sparse embedding approach through deterministic and stochastic nonlinear system examples.

----

## [543] Probably Anytime-Safe Stochastic Combinatorial Semi-Bandits

**Authors**: *Yunlong Hou, Vincent Y. F. Tan, Zixin Zhong*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hou23d.html](https://proceedings.mlr.press/v202/hou23d.html)

**Abstract**:

Motivated by concerns about making online decisions that incur undue amount of risk at each time step, in this paper, we formulate the probably anytime-safe stochastic combinatorial semi-bandits problem. In this problem, the agent is given the option to select a subset of size at most $K$ from a set of $L$ ground items. Each item is associated to a certain mean reward as well as a variance that represents its risk. To mitigate the risk that the agent incurs, we require that with probability at least $1-\delta$, over the entire horizon of time $T$, each of the choices that the agent makes should contain items whose sum of variances does not exceed a certain variance budget. We call this probably anytime-safe constraint. Under this constraint, we design and analyze an algorithm PASCombUCB that minimizes the regret over the horizon of time $T$. By developing accompanying information-theoretic lower bounds, we show that under both the problem-dependent and problem-independent paradigms, PASCombUCB is almost asymptotically optimal. Experiments are conducted to corroborate our theoretical findings. Our problem setup, the proposed PASCombUCB algorithm, and novel analyses are applicable to domains such as recommendation systems and transportation in which an agent is allowed to choose multiple items at a single time step and wishes to control the risk over the whole time horizon.

----

## [544] Automatic Data Augmentation via Invariance-Constrained Learning

**Authors**: *Ignacio Hounie, Luiz F. O. Chamon, Alejandro Ribeiro*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hounie23a.html](https://proceedings.mlr.press/v202/hounie23a.html)

**Abstract**:

Underlying data structures, such as symmetries or invariance to transformations, are often exploited to improve the solution of learning tasks. However, embedding these properties in models or learning algorithms can be challenging and computationally intensive. Data augmentation, on the other hand, induces these symmetries during training by applying multiple transformations to the input data. Despite its ubiquity, its effectiveness depends on the choices of which transformations to apply, when to do so, and how often. In fact, there is both empirical and theoretical evidence that the indiscriminate use of data augmentation can introduce biases that outweigh its benefits. This work tackles these issues by automatically adapting the data augmentation while solving the learning task. To do so, it formulates data augmentation as an invariance constrained learning problem and leverages Monte Carlo Markov Chain (MCMC) sampling to solve it. The result is an algorithm that not only does away with a priori searches for augmentation distributions, but also dynamically controls if and when data augmentation is applied. We validate empirically our theoretical developments in automatic data augmentation benchmarks for CIFAR and ImageNet-100 datasets. Furthermore, our experiments show how this approach can be used to gather insights on the actual symmetries underlying a learning task.

----

## [545] Thompson Sampling with Diffusion Generative Prior

**Authors**: *Yu-Guan Hsieh, Shiva Prasad Kasiviswanathan, Branislav Kveton, Patrick Blöbaum*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hsieh23a.html](https://proceedings.mlr.press/v202/hsieh23a.html)

**Abstract**:

In this work, we initiate the idea of using denoising diffusion models to learn priors for online decision making problems. We specifically focus on bandit meta-learning, aiming to learn a policy that performs well across bandit tasks of a same class. To this end, we train a diffusion model that learns the underlying task distribution and combine Thompson sampling with the learned prior to deal with new tasks at test time. Our posterior sampling algorithm carefully balances between the learned prior and the noisy observations that come from the learner’s interaction with the environment. To capture realistic bandit scenarios, we propose a novel diffusion model training procedure that trains from incomplete and noisy data, which could be of independent interest. Finally, our extensive experiments clearly demonstrate the potential of the proposed approach.

----

## [546] Tighter Analysis for ProxSkip

**Authors**: *Zhengmian Hu, Heng Huang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23a.html](https://proceedings.mlr.press/v202/hu23a.html)

**Abstract**:

In this paper, we provide a tighter analysis for ProxSkip, an algorithm that allows fewer proximal operator computations to solve composite optimization problems. We improve the existing decreasing speed of Lyapunov function from $\mathcal{O}(p^2)$ to $\mathcal{O}(p)$, when $p$, the frequency of the proximal operators is small enough. Our theoretical analysis also reveals the drawbacks of using large step sizes for gradient descent in ProxSkip when the proximal operator part is the bottleneck. Our main motivation comes from the continuous limit in which the original analysis of ProxSkip fails to guarantee convergence when both the step size $\gamma$ and frequency $p$ tend to zero. We construct a counterexample to demonstrate why such counterintuitive behavior occurs for the original analysis and then propose a novel Lyapunov function variant to construct a tighter analysis, avoiding the problem of the old one. Such a new Lyapunov function can be directly extended to many other variants of ProxSkip. When applied to stochastic gradient setup, our analysis leads to an improved proximal operator complexity for SProxSkip from $\mathcal{O}(\sqrt{\frac{1}{\varepsilon\mu^2}}\log(\frac{1}{\varepsilon}))$ to $\mathcal{O}(\sqrt{\kappa}\log(\frac{1}{\varepsilon}))$.

----

## [547] Omnipredictors for Constrained Optimization

**Authors**: *Lunjia Hu, Inbal Rachel Livni Navon, Omer Reingold, Chutong Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23b.html](https://proceedings.mlr.press/v202/hu23b.html)

**Abstract**:

The notion of omnipredictors (Gopalan, Kalai, Reingold, Sharan and Wieder ITCS 2022), suggested a new paradigm for loss minimization. Rather than learning a predictor based on a known loss function, omnipredictors can easily be post-processed to minimize any one of a rich family of loss functions compared with the loss of hypotheses in a class $\mathcal C$. It has been shown that such omnipredictors exist and are implied (for all convex and Lipschitz loss functions) by the notion of multicalibration from the algorithmic fairness literature. In this paper, we introduce omnipredictors for constrained optimization and study their complexity and implications. The notion that we introduce allows the learner to be unaware of the loss function that will be later assigned as well as the constraints that will be later imposed, as long as the subpopulations that are used to define these constraints are known. We show how to obtain omnipredictors for constrained optimization problems, relying on appropriate variants of multicalibration. We also investigate the implications of this notion when the constraints used are so-called group fairness notions.

----

## [548] GFlowNet-EM for Learning Compositional Latent Variable Models

**Authors**: *Edward J. Hu, Nikolay Malkin, Moksh Jain, Katie E. Everett, Alexandros Graikos, Yoshua Bengio*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23c.html](https://proceedings.mlr.press/v202/hu23c.html)

**Abstract**:

Latent variable models (LVMs) with discrete compositional latents are an important but challenging setting due to a combinatorially large number of possible configurations of the latents. A key tradeoff in modeling the posteriors over latents is between expressivity and tractable optimization. For algorithms based on expectation-maximization (EM), the E-step is often intractable without restrictive approximations to the posterior. We propose the use of GFlowNets, algorithms for sampling from an unnormalized density by learning a stochastic policy for sequential construction of samples, for this intractable E-step. By training GFlowNets to sample from the posterior over latents, we take advantage of their strengths as amortized variational inference algorithms for complex distributions over discrete structures. Our approach, GFlowNet-EM, enables the training of expressive LVMs with discrete compositional latents, as shown by experiments on non-context-free grammar induction and on images using discrete variational autoencoders (VAEs) without conditional independence enforced in the encoder.

----

## [549] Blockwise Stochastic Variance-Reduced Methods with Parallel Speedup for Multi-Block Bilevel Optimization

**Authors**: *Quanqi Hu, Zi-Hao Qiu, Zhishuai Guo, Lijun Zhang, Tianbao Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23d.html](https://proceedings.mlr.press/v202/hu23d.html)

**Abstract**:

In this paper, we consider non-convex multi-block bilevel optimization (MBBO) problems, which involve $m\gg 1$ lower level problems and have important applications in machine learning. Designing a stochastic gradient and controlling its variance is more intricate due to the hierarchical sampling of blocks and data and the unique challenge of estimating hyper-gradient. We aim to achieve three nice properties for our algorithm: (a) matching the state-of-the-art complexity of standard BO problems with a single block; (b) achieving parallel speedup by sampling $I$ blocks and sampling $B$ samples for each sampled block per-iteration; (c) avoiding the computation of the inverse of a high-dimensional Hessian matrix estimator. However, it is non-trivial to achieve all of these by observing that existing works only achieve one or two of these properties. To address the involved challenges for achieving (a, b, c), we propose two stochastic algorithms by using advanced blockwise variance-reduction techniques for tracking the Hessian matrices (for low-dimensional problems) or the Hessian-vector products (for high-dimensional problems), and prove an iteration complexity of $O(\frac{m\epsilon^{-3}\mathbb{I}(I \textless m)}{I\sqrt{I}}+\frac{m\epsilon^{-3}}{I\sqrt{B}})$ for finding an $\epsilon$-stationary point under appropriate conditions. We also conduct experiments to verify the effectiveness of the proposed algorithms comparing with existing MBBO algorithms.

----

## [550] Language Instructed Reinforcement Learning for Human-AI Coordination

**Authors**: *Hengyuan Hu, Dorsa Sadigh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23e.html](https://proceedings.mlr.press/v202/hu23e.html)

**Abstract**:

One of the fundamental quests of AI is to produce agents that coordinate well with humans. This problem is challenging, especially in domains that lack high quality human behavioral data, because multi-agent reinforcement learning (RL) often converges to different equilibria from the ones that humans prefer. We propose a novel framework, instructRL, that enables humans to specify what kind of strategies they expect from their AI partners through natural language instructions. We use pretrained large language models to generate a prior policy conditioned on the human instruction and use the prior to regularize the RL objective. This leads to the RL agent converging to equilibria that are aligned with human preferences. We show that instructRL converges to human-like policies that satisfy the given instructions in a proof-of-concept environment as well as the challenging Hanabi benchmark. Finally, we show that knowing the language instruction significantly boosts human-AI coordination performance in human evaluations in Hanabi.

----

## [551] Surface Snapping Optimization Layer for Single Image Object Shape Reconstruction

**Authors**: *Yuan-Ting Hu, Alexander G. Schwing, Raymond A. Yeh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23f.html](https://proceedings.mlr.press/v202/hu23f.html)

**Abstract**:

Reconstructing the 3D shape of objects observed in a single image is a challenging task. Recent approaches rely on visual cues extracted from a given image learned from a deep net. In this work, we leverage recent advances in monocular scene understanding to incorporate an additional geometric cue of surface normals. For this, we proposed a novel optimization layer that encourages the face normals of the reconstructed shape to be aligned with estimated surface normals. We develop a computationally efficient conjugate-gradient-based method that avoids the computation of a high-dimensional sparse matrix. We show this framework to achieve compelling shape reconstruction results on the challenging Pix3D and ShapeNet datasets.

----

## [552] Learning to Learn from APIs: Black-Box Data-Free Meta-Learning

**Authors**: *Zixuan Hu, Li Shen, Zhenyi Wang, Baoyuan Wu, Chun Yuan, Dacheng Tao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23g.html](https://proceedings.mlr.press/v202/hu23g.html)

**Abstract**:

Data-free meta-learning (DFML) aims to enable efficient learning of new tasks by meta-learning from a collection of pre-trained models without access to the training data. Existing DFML work can only meta-learn from (i) white-box and (ii) small-scale pre-trained models (iii) with the same architecture, neglecting the more practical setting where the users only have inference access to the APIs with arbitrary model architectures and model scale inside. To solve this issue, we propose a Bi-level Data-free Meta Knowledge Distillation (BiDf-MKD) framework to transfer more general meta knowledge from a collection of black-box APIs to one single meta model. Specifically, by just querying APIs, we inverse each API to recover its training data via a zero-order gradient estimator and then perform meta-learning via a novel bi-level meta knowledge distillation structure, in which we design a boundary query set recovery technique to recover a more informative query set near the decision boundary. In addition, to encourage better generalization within the setting of limited API budgets, we propose task memory replay to diversify the underlying task distribution by covering more interpolated tasks. Extensive experiments in various real-world scenarios show the superior performance of our BiDf-MKD framework.

----

## [553] For Pre-Trained Vision Models in Motor Control, Not All Policy Learning Methods are Created Equal

**Authors**: *Yingdong Hu, Renhao Wang, Li Erran Li, Yang Gao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23h.html](https://proceedings.mlr.press/v202/hu23h.html)

**Abstract**:

In recent years, increasing attention has been directed to leveraging pre-trained vision models for motor control. While existing works mainly emphasize the importance of this pre-training phase, the arguably equally important role played by downstream policy learning during control-specific fine-tuning is often neglected. It thus remains unclear if pre-trained vision models are consistent in their effectiveness under different control policies. To bridge this gap in understanding, we conduct a comprehensive study on 14 pre-trained vision models using 3 distinct classes of policy learning methods, including reinforcement learning (RL), imitation learning through behavior cloning (BC), and imitation learning with a visual reward function (VRF). Our study yields a series of intriguing results, including the discovery that the effectiveness of pre-training is highly dependent on the choice of the downstream policy learning algorithm. We show that conventionally accepted evaluation based on RL methods is highly variable and therefore unreliable, and further advocate for using more robust methods like VRF and BC. To facilitate more universal evaluations of pre-trained models and their policy learning methods in the future, we also release a benchmark of 21 tasks across 3 different environments alongside our work.

----

## [554] Beyond Lipschitz Smoothness: A Tighter Analysis for Nonconvex Optimization

**Authors**: *Zhengmian Hu, Xidong Wu, Heng Huang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23i.html](https://proceedings.mlr.press/v202/hu23i.html)

**Abstract**:

Negative and positive curvatures affect optimization in different ways. However, a lot of existing optimization theories are based on the Lipschitz smoothness assumption, which cannot differentiate between the two. In this paper, we propose to use two separate assumptions for positive and negative curvatures, so that we can study the different implications of the two. We analyze the Lookahead and Local SGD methods as concrete examples. Both of them require multiple copies of model parameters and communication among them for every certain period of time in order to prevent divergence. We show that the minimum communication frequency is inversely proportional to the negative curvature, and when the negative curvature becomes zero, we recover the existing theory results for convex optimization. Finally, both experimentally and theoretically, we demonstrate that modern neural networks have highly unbalanced positive/negative curvatures. Thus, an analysis based on separate positive and negative curvatures is more pertinent.

----

## [555] Understanding the Impact of Adversarial Robustness on Accuracy Disparity

**Authors**: *Yuzheng Hu, Fan Wu, Hongyang Zhang, Han Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hu23j.html](https://proceedings.mlr.press/v202/hu23j.html)

**Abstract**:

While it has long been empirically observed that adversarial robustness may be at odds with standard accuracy and may have further disparate impacts on different classes, it remains an open question to what extent such observations hold and how the class imbalance plays a role within. In this paper, we attempt to understand this question of accuracy disparity by taking a closer look at linear classifiers under a Gaussian mixture model. We decompose the impact of adversarial robustness into two parts: an inherent effect that will degrade the standard accuracy on all classes due to the robustness constraint, and the other caused by the class imbalance ratio, which will increase the accuracy disparity compared to standard training. Furthermore, we also show that such effects extend beyond the Gaussian mixture model, by generalizing our data model to the general family of stable distributions. More specifically, we demonstrate that while the constraint of adversarial robustness consistently degrades the standard accuracy in the balanced class setting, the class imbalance ratio plays a fundamentally different role in accuracy disparity compared to the Gaussian case, due to the heavy tail of the stable distribution. We additionally perform experiments on both synthetic and real-world datasets to corroborate our theoretical findings. Our empirical results also suggest that the implications may extend to nonlinear models over real-world datasets. Our code is publicly available on GitHub at https://github.com/Accuracy-Disparity/AT-on-AD.

----

## [556] Reinforcement Learning in Low-rank MDPs with Density Features

**Authors**: *Audrey Huang, Jinglin Chen, Nan Jiang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23a.html](https://proceedings.mlr.press/v202/huang23a.html)

**Abstract**:

MDPs with low-rank transitions—that is, the transition matrix can be factored into the product of two matrices, left and right—is a highly representative structure that enables tractable learning. The left matrix enables expressive function approximation for value-based learning and has been studied extensively. In this work, we instead investigate sample-efficient learning with density features, i.e., the right matrix, which induce powerful models for state-occupancy distributions. This setting not only sheds light on leveraging unsupervised learning in RL, but also enables plug-in solutions for settings like convex RL. In the offline setting, we propose an algorithm for off-policy estimation of occupancies that can handle non-exploratory data. Using this as a subroutine, we further devise an online algorithm that constructs exploratory data distributions in a level-by-level manner. As a central technical challenge, the additive error of occupancy estimation is incompatible with the multiplicative definition of data coverage. In the absence of strong assumptions like reachability, this incompatibility easily leads to exponential error blow-up, which we overcome via novel technical tools. Our results also readily extend to the representation learning setting, when the density features are unknown and must be learned from an exponentially large candidate set.

----

## [557] Composer: Creative and Controllable Image Synthesis with Composable Conditions

**Authors**: *Lianghua Huang, Di Chen, Yu Liu, Yujun Shen, Deli Zhao, Jingren Zhou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23b.html](https://proceedings.mlr.press/v202/huang23b.html)

**Abstract**:

Recent large-scale generative models learned on big data are capable of synthesizing incredible images yet suffer from limited controllability. This work offers a new generation paradigm that allows flexible control of the output image, such as spatial layout and palette, while maintaining the synthesis quality and model creativity. With compositionality as the core idea, we first decompose an image into representative factors, and then train a diffusion model with all these factors as the conditions to recompose the input. At the inference stage, the rich intermediate representations work as composable elements, leading to a huge design space (i.e., exponentially proportional to the number of decomposed factors) for customizable content creation. It is noteworthy that our approach, which we call Composer, supports various levels of conditions, such as text description as the global information, depth map and sketch as the local guidance, color histogram for low-level details, etc. Besides improving controllability, we confirm that Composer serves as a general framework and facilitates a wide range of classical generative tasks without retraining. Code and models will be made available.

----

## [558] Model-Aware Contrastive Learning: Towards Escaping the Dilemmas

**Authors**: *Zizheng Huang, Haoxing Chen, Ziqi Wen, Chao Zhang, Huaxiong Li, Bo Wang, Chunlin Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23c.html](https://proceedings.mlr.press/v202/huang23c.html)

**Abstract**:

Contrastive learning (CL) continuously achieves significant breakthroughs across multiple domains. However, the most common InfoNCE-based methods suffer from some dilemmas, such as uniformity-tolerance dilemma (UTD) and gradient reduction, both of which are related to a $\mathcal{P}_{ij}$ term. It has been identified that UTD can lead to unexpected performance degradation. We argue that the fixity of temperature is to blame for UTD. To tackle this challenge, we enrich the CL loss family by presenting a Model-Aware Contrastive Learning (MACL) strategy, whose temperature is adaptive to the magnitude of alignment that reflects the basic confidence of the instance discrimination task, then enables CL loss to adjust the penalty strength for hard negatives adaptively. Regarding another dilemma, the gradient reduction issue, we derive the limits of an involved gradient scaling factor, which allows us to explain from a unified perspective why some recent approaches are effective with fewer negative samples, and summarily present a gradient reweighting to escape this dilemma. Extensive remarkable empirical results in vision, sentence, and graph modality validate our approach’s general improvement for representation learning and downstream tasks.

----

## [559] High-dimensional Clustering onto Hamiltonian Cycle

**Authors**: *Tianyi Huang, Shenghui Cheng, Stan Z. Li, Zhengjun Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23d.html](https://proceedings.mlr.press/v202/huang23d.html)

**Abstract**:

Clustering aims to group unlabelled samples based on their similarities and is widespread in high-dimensional data analysis. However, most of the clustering methods merely generate pseudo labels and thus are unable to simultaneously present the similarities between different clusters and outliers. This paper proposes a new framework called High-dimensional Clustering onto Hamiltonian Cycle (HCHC) to solve the above problems. First, HCHC combines global structure with local structure in one objective function for deep clustering, improving the labels as relative probabilities, to mine the similarities between different clusters while keeping the local structure in each cluster. Then, the anchors of different clusters are sorted on the optimal Hamiltonian cycle generated by the cluster similarities and mapped on the circumference of a circle. Finally, a sample with a higher probability of a cluster will be mapped closer to the corresponding anchor. In this way, our framework allows us to appreciate three aspects visually and simultaneously - clusters (formed by samples with high probabilities), cluster similarities (represented as circular distances), and outliers (recognized as dots far away from all clusters). The theoretical analysis and experiments illustrate the superiority of HCHC.

----

## [560] Banker Online Mirror Descent: A Universal Approach for Delayed Online Bandit Learning

**Authors**: *Jiatai Huang, Yan Dai, Longbo Huang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23e.html](https://proceedings.mlr.press/v202/huang23e.html)

**Abstract**:

We propose Banker Online Mirror Descent (Banker-OMD), a novel framework generalizing the classical Online Mirror Descent (OMD) technique in the online learning literature. The Banker-OMD framework almost completely decouples feedback delay handling and the task-specific OMD algorithm design, thus facilitating the design of new algorithms capable of efficiently and robustly handling feedback delays. Specifically, it offers a general methodology for achieving $\widetilde{\mathcal O}(\sqrt{T} + \sqrt{D})$-style regret bounds in online bandit learning tasks with delayed feedback, where $T$ is the number of rounds and $D$ is the total feedback delay. We demonstrate the power of Banker-OMD by applications to two important bandit learning scenarios with delayed feedback, including delayed scale-free adversarial Multi-Armed Bandits (MAB) and delayed adversarial linear bandits. Banker-OMD leads to the first delayed scale-free adversarial MAB algorithm achieving $\widetilde{\mathcal O}(\sqrt{K}L(\sqrt T+\sqrt D))$ regret and the first delayed adversarial linear bandit algorithm achieving $\widetilde{\mathcal O}(\text{poly}(n)(\sqrt{T} + \sqrt{D}))$ regret. As a corollary, the first application also implies $\widetilde{\mathcal O}(\sqrt{KT}L)$ regret for non-delayed scale-free adversarial MABs, which is the first to match the $\Omega(\sqrt{KT}L)$ lower bound up to logarithmic factors and can be of independent interest.

----

## [561] Fast Algorithms for Distributed k-Clustering with Outliers

**Authors**: *Junyu Huang, Qilong Feng, Ziyun Huang, Jinhui Xu, Jianxin Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23f.html](https://proceedings.mlr.press/v202/huang23f.html)

**Abstract**:

In this paper, we study the $k$-clustering problems with outliers in distributed setting. The current best results for the distributed $k$-center problem with outliers have quadratic local running time with communication cost dependent on the aspect ratio $\Delta$ of the given instance, which may constraint the scalability of the algorithms for handling large-scale datasets. To achieve better communication cost for the problem with faster local running time, we propose an inliers-recalling sampling method, which avoids guessing the optimal radius of the given instance, and can achieve a 4-round bi-criteria $(14(1+\epsilon),1+\epsilon)$-approximation with linear local running time in the data size and communication cost independent of the aspect ratio. To obtain a more practical algorithm for the problem, we propose another space-narrowing sampling method, which automatically adjusts the sample size to adapt to different outliers distributions on each machine, and can achieve a 2-round bi-criteria $(14(1+\epsilon),1+\epsilon)$-approximation with communication cost independent of the number of outliers. We show that, if the data points are randomly partitioned across machines, our proposed sampling-based methods can be extended to the $k$-median/means problems with outliers, and can achieve $(O(\frac{1}{\epsilon^2}),1+\epsilon)$-approximation with communication cost independent of the number of outliers. Empirical experiments suggest that the proposed 2-round distributed algorithms outperform other state-of-the-art algorithms.

----

## [562] Searching Large Neighborhoods for Integer Linear Programs with Contrastive Learning

**Authors**: *Taoan Huang, Aaron M. Ferber, Yuandong Tian, Bistra Dilkina, Benoit Steiner*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23g.html](https://proceedings.mlr.press/v202/huang23g.html)

**Abstract**:

Integer Linear Programs (ILPs) are powerful tools for modeling and solving a large number of combinatorial optimization problems. Recently, it has been shown that Large Neighborhood Search (LNS), as a heuristic algorithm, can find high-quality solutions to ILPs faster than Branch and Bound. However, how to find the right heuristics to maximize the performance of LNS remains an open problem. In this paper, we propose a novel approach, CL-LNS, that delivers state-of-the-art anytime performance on several ILP benchmarks measured by metrics including the primal gap, the primal integral, survival rates and the best performing rate. Specifically, CL-LNS collects positive and negative solution samples from an expert heuristic that is slow to compute and learns a more efficient one with contrastive learning. We use graph attention networks and a richer set of features to further improve its performance.

----

## [563] On Coresets for Clustering in Small Dimensional Euclidean spaces

**Authors**: *Lingxiao Huang, Ruiyuan Huang, Zengfeng Huang, Xuan Wu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23h.html](https://proceedings.mlr.press/v202/huang23h.html)

**Abstract**:

We consider the problem of constructing small coresets for $k$-Median in Euclidean spaces. Given a large set of data points $P\subset \mathbb{R}^d$, a coreset is a much smaller set $S\subset \mathbb{R}^d$, so that the $k$-Median costs of any $k$ centers w.r.t. $P$ and $S$ are close. Existing literature mainly focuses on the high-dimension case and there has been great success in obtaining dimension-independent bounds, whereas the case for small $d$ is largely unexplored. Considering many applications of Euclidean clustering algorithms are in small dimensions and the lack of systematic studies in the current literature, this paper investigates coresets for $k$-Median in small dimensions. For small $d$, a natural question is whether existing near-optimal dimension-independent bounds can be significantly improved. We provide affirmative answers to this question for a range of parameters. Moreover, new lower bound results are also proved, which are the highest for small $d$. In particular, we completely settle the coreset size bound for $1$-d $k$-Median (up to log factors). Interestingly, our results imply a strong separation between $1$-d $1$-Median and $1$-d $2$-Median. As far as we know, this is the first such separation between $k=1$ and $k=2$ in any dimension.

----

## [564] Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models

**Authors**: *Rongjie Huang, Jiawei Huang, Dongchao Yang, Yi Ren, Luping Liu, Mingze Li, Zhenhui Ye, Jinglin Liu, Xiang Yin, Zhou Zhao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23i.html](https://proceedings.mlr.press/v202/huang23i.html)

**Abstract**:

Large-scale multimodal generative modeling has created milestones in text-to-image and text-to-video generation. Its application to audio still lags behind for two main reasons: the lack of large-scale datasets with high-quality text-audio pairs, and the complexity of modeling long continuous audio data. In this work, we propose Make-An-Audio with a prompt-enhanced diffusion model that addresses these gaps by 1) introducing pseudo prompt enhancement with a distill-then-reprogram approach, it alleviates data scarcity with orders of magnitude concept compositions by using language-free audios; 2) leveraging spectrogram autoencoder to predict the self-supervised audio representation instead of waveforms. Together with robust contrastive language-audio pretraining (CLAP) representations, Make-An-Audio achieves state-of-the-art results in both objective and subjective benchmark evaluation. Moreover, we present its controllability and generalization for X-to-Audio with "No Modality Left Behind", for the first time unlocking the ability to generate high-definition, high-fidelity audios given a user-defined modality input. Audio samples are available at https://Make-An-Audio.github.io

----

## [565] The Power of Uniform Sampling for k-Median

**Authors**: *Lingxiao Huang, Shaofeng H.-C. Jiang, Jianing Lou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23j.html](https://proceedings.mlr.press/v202/huang23j.html)

**Abstract**:

We study the power of uniform sampling for $k$-Median in various metric spaces. We relate the query complexity for approximating $k$-Median, to a key parameter of the dataset, called the balancedness $\beta \in (0, 1]$ (with $1$ being perfectly balanced). We show that any algorithm must make $\Omega(1 / \beta)$ queries to the point set in order to achieve $O(1)$-approximation for $k$-Median. This particularly implies existing constructions of coresets, a popular data reduction technique, cannot be query-efficient. On the other hand, we show a simple uniform sample of $\mathrm{poly}(k \epsilon^{-1} \beta^{-1})$ points suffices for $(1 + \epsilon)$-approximation for $k$-Median for various metric spaces, which nearly matches the lower bound. We conduct experiments to verify that in many real datasets, the balancedness parameter is usually well bounded, and that the uniform sampling performs consistently well even for the case with moderately large balancedness, which justifies that uniform sampling is indeed a viable approach for solving $k$-Median.

----

## [566] Reparameterized Policy Learning for Multimodal Trajectory Optimization

**Authors**: *Zhiao Huang, Litian Liang, Zhan Ling, Xuanlin Li, Chuang Gan, Hao Su*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23k.html](https://proceedings.mlr.press/v202/huang23k.html)

**Abstract**:

We investigate the challenge of parametrizing policies for reinforcement learning (RL) in high-dimensional continuous action spaces. Our objective is to develop a multimodal policy that overcomes limitations inherent in the commonly-used Gaussian parameterization. To achieve this, we propose a principled framework that models the continuous RL policy as a generative model of optimal trajectories. By conditioning the policy on a latent variable, we derive a novel variational bound as the optimization objective, which promotes exploration of the environment. We then present a practical model-based RL method, called Reparameterized Policy Gradient (RPG), which leverages the multimodal policy parameterization and learned world model to achieve strong exploration capabilities and high data efficiency. Empirical results demonstrate that our method can help agents evade local optima in tasks with dense rewards and solve challenging sparse-reward environments by incorporating an object-centric intrinsic reward. Our method consistently outperforms previous approaches across a range of tasks. Code and supplementary materials are available on the project page https://haosulab.github.io/RPG/

----

## [567] Theoretical Bounds on the Network Community Profile from Low-rank Semi-definite Programming

**Authors**: *Yufan Huang, C. Seshadhri, David F. Gleich*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23l.html](https://proceedings.mlr.press/v202/huang23l.html)

**Abstract**:

We study a new connection between a technical measure called $\mu$-conductance that arises in the study of Markov chains for sampling convex bodies and the network community profile that characterizes size-resolved properties of clusters and communities in social and information networks. The idea of $\mu$-conductance is similar to the traditional graph conductance, but disregards sets with small volume. We derive a sequence of optimization problems including a low-rank semi-definite program from which we can derive a lower bound on the optimal $\mu$-conductance value. These ideas give the first theoretically sound bound on the behavior of the network community profile for a wide range of cluster sizes. The algorithm scales up to graphs with hundreds of thousands of nodes and we demonstrate how our framework validates the predicted structures of real-world graphs.

----

## [568] NeuralStagger: Accelerating Physics-constrained Neural PDE Solver with Spatial-temporal Decomposition

**Authors**: *Xinquan Huang, Wenlei Shi, Qi Meng, Yue Wang, Xiaotian Gao, Jia Zhang, Tie-Yan Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23m.html](https://proceedings.mlr.press/v202/huang23m.html)

**Abstract**:

Neural networks have shown great potential in accelerating the solution of partial differential equations (PDEs). Recently, there has been a growing interest in introducing physics constraints into training neural PDE solvers to reduce the use of costly data and improve the generalization ability. However, these physics constraints, based on certain finite dimensional approximations over the function space, must resolve the smallest scaled physics to ensure the accuracy and stability of the simulation, resulting in high computational costs from large input, output, and neural networks. This paper proposes a general acceleration methodology called NeuralStagger by spatially and temporally decomposing the original learning tasks into several coarser-resolution subtasks. We define a coarse-resolution neural solver for each subtask, which requires fewer computational resources, and jointly train them with the vanilla physics-constrained loss by simply arranging their outputs to reconstruct the original solution. Due to the perfect parallelism between them, the solution is achieved as fast as a coarse-resolution neural solver. In addition, the trained solvers bring the flexibility of simulating with multiple levels of resolution. We demonstrate the successful application of NeuralStagger on 2D and 3D fluid dynamics simulations, which leads to an additional $10\sim100\times$ speed-up. Moreover, the experiment also shows that the learned model could be well used for optimal control.

----

## [569] Policy Contrastive Imitation Learning

**Authors**: *Jialei Huang, Zhao-Heng Yin, Yingdong Hu, Yang Gao*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23n.html](https://proceedings.mlr.press/v202/huang23n.html)

**Abstract**:

Adversarial imitation learning (AIL) is a popular method that has recently achieved much success. However, the performance of AIL is still unsatisfactory on the more challenging tasks. We find that one of the major reasons is due to the low quality of AIL discriminator representation. Since the AIL discriminator is trained via binary classification that does not necessarily discriminate the policy from the expert in a meaningful way, the resulting reward might not be meaningful either. We propose a new method called Policy Contrastive Imitation Learning (PCIL) to resolve this issue. PCIL learns a contrastive representation space by anchoring on different policies and uses a smooth cosine-similarity-based reward to encourage imitation learning. Our proposed representation learning objective can be viewed as a stronger version of the AIL objective and provide a more meaningful comparison between the agent and the policy. From a theoretical perspective, we show the validity of our method using the apprenticeship learning framework. Furthermore, our empirical evaluation on the DeepMind Control suite demonstrates that PCIL can achieve state-of-the-art performance. Finally, qualitative results suggest that PCIL builds a smoother and more meaningful representation space for imitation learning.

----

## [570] Are Large Kernels Better Teachers than Transformers for ConvNets?

**Authors**: *Tianjin Huang, Lu Yin, Zhenyu Zhang, Li Shen, Meng Fang, Mykola Pechenizkiy, Zhangyang Wang, Shiwei Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23o.html](https://proceedings.mlr.press/v202/huang23o.html)

**Abstract**:

This paper reveals a new appeal of the recently emerged large-kernel Convolutional Neural Networks (ConvNets): as the teacher in Knowledge Distillation (KD) for small-kernel ConvNets. While Transformers have led state-of-the-art (SOTA) performance in various fields with ever-larger models and labeled data, small-kernel ConvNets are considered more suitable for resource-limited applications due to the efficient convolution operation and compact weight sharing. KD is widely used to boost the performance of small-kernel ConvNets. However, previous research shows that it is not quite effective to distill knowledge (e.g., global information) from Transformers to small-kernel ConvNets, presumably due to their disparate architectures. We hereby carry out a first-of-its-kind study unveiling that modern large-kernel ConvNets, a compelling competitor to Vision Transformers, are remarkably more effective teachers for small-kernel ConvNets, due to more similar architectures. Our findings are backed up by extensive experiments on both logit-level and feature-level KD "out of the box", with no dedicated architectural nor training recipe modifications. Notably, we obtain the best-ever pure ConvNet under 30M parameters with 83.1% top-1 accuracy on ImageNet, outperforming current SOTA methods including ConvNeXt V2 and Swin V2. We also find that beneficial characteristics of large-kernel ConvNets, e.g., larger effective receptive fields, can be seamlessly transferred to students through this large-to-small kernel distillation. Code is available at: https://github.com/VITA-Group/SLaK.

----

## [571] Achieving Linear Speedup in Non-IID Federated Bilevel Learning

**Authors**: *Minhui Huang, Dewei Zhang, Kaiyi Ji*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23p.html](https://proceedings.mlr.press/v202/huang23p.html)

**Abstract**:

Federated bilevel learning has received increasing attention in various emerging machine learning and communication applications. Recently, several Hessian-vector-based algorithms have been proposed to solve the federated bilevel optimization problem. However, several important properties in federated learning such as the partial client participation and the linear speedup for convergence (i.e., the convergence rate and complexity are improved linearly with respect to the number of sampled clients) in the presence of non-i.i.d. datasets, still remain open. In this paper, we fill these gaps by proposing a new federated bilevel algorithm named FedMBO with a novel client sampling scheme in the federated hypergradient estimation. We show that FedMBO achieves a convergence rate of $\mathcal{O}\big(\frac{1}{\sqrt{nK}}+\frac{1}{K}+\frac{\sqrt{n}}{K^{3/2}}\big)$ on non-i.i.d. datasets, where $n$ is the number of participating clients in each round, and $K$ is the total number of iteration. This is the first theoretical linear speedup result for non-i.i.d. federated bilevel optimization. Extensive experiments validate our theoretical results and demonstrate the effectiveness of our proposed method.

----

## [572] Federated Linear Contextual Bandits with User-level Differential Privacy

**Authors**: *Ruiquan Huang, Huanyu Zhang, Luca Melis, Milan Shen, Meisam Hejazinia, Jing Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huang23q.html](https://proceedings.mlr.press/v202/huang23q.html)

**Abstract**:

This paper studies federated linear contextual bandits under the notion of user-level differential privacy (DP). We first introduce a unified federated bandits framework that can accommodate various definitions of DP in the sequential decision-making setting. We then formally introduce user-level central DP (CDP) and local DP (LDP) in the federated bandits framework, and investigate the fundamental trade-offs between the learning regrets and the corresponding DP guarantees in a federated linear contextual bandits model. For CDP, we propose a federated algorithm termed as $\texttt{ROBIN}$ and show that it is near-optimal in terms of the number of clients $M$ and the privacy budget $\varepsilon$ by deriving nearly-matching upper and lower regret bounds when user-level DP is satisfied. For LDP, we obtain several lower bounds, indicating that learning under user-level $(\varepsilon,\delta)$-LDP must suffer a regret blow-up factor at least $\min\{1/\varepsilon,M\}$ or $\min\{1/\sqrt{\varepsilon},\sqrt{M}\}$ under different conditions.

----

## [573] Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks

**Authors**: *Minyoung Huh, Brian Cheung, Pulkit Agrawal, Phillip Isola*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huh23a.html](https://proceedings.mlr.press/v202/huh23a.html)

**Abstract**:

This work examines the challenges of training neural networks using vector quantization using straight-through estimation. We find that the main cause of training instability is the discrepancy between the model embedding and the code-vector distribution. We identify the factors that contribute to this issue, including the codebook gradient sparsity and the asymmetric nature of the commitment loss, which leads to misaligned code-vector assignments. We propose to address this issue via affine re-parameterization of the code vectors. Additionally, we introduce an alternating optimization to reduce the gradient error introduced by the straight-through estimation. Moreover, we propose an improvement to the commitment loss to ensure better alignment between the codebook representation and the model embedding. These optimization methods improve the mathematical approximation of the straight-through estimation and, ultimately, the model performance. We demonstrate the effectiveness of our methods on several common model architectures, such as AlexNet, ResNet, and ViT, across various tasks, including image classification and generative modeling.

----

## [574] Cut your Losses with Squentropy

**Authors**: *Like Hui, Mikhail Belkin, Stephen Wright*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hui23a.html](https://proceedings.mlr.press/v202/hui23a.html)

**Abstract**:

Nearly all practical neural models for classification are trained using the cross-entropy loss. Yet this ubiquitous choice is supported by little theoretical or empirical evidence. Recent work (Hui & Belkin, 2020) suggests that training using the (rescaled) square loss is often superior in terms of the classification accuracy. In this paper we propose the "squentropy" loss, which is the sum of two terms: the cross-entropy loss and the average square loss over the incorrect classes. We provide an extensive set of experiment on multi-class classification problems showing that the squentropy loss outperforms both the pure cross-entropy and rescaled square losses in terms of the classification accuracy. We also demonstrate that it provides significantly better model calibration than either of these alternative losses and, furthermore, has less variance with respect to the random initialization. Additionally, in contrast to the square loss, squentropy loss can frequently be trained using exactly the same optimization parameters, including the learning rate, as the standard cross-entropy loss, making it a true ”plug-and-play” replacement. Finally, unlike the rescaled square loss, multiclass squentropy contains no parameters that need to be adjusted.

----

## [575] SOM-CPC: Unsupervised Contrastive Learning with Self-Organizing Maps for Structured Representations of High-Rate Time Series

**Authors**: *Iris A. M. Huijben, Arthur Andreas Nijdam, Sebastiaan Overeem, Merel M. van Gilst, Ruud van Sloun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/huijben23a.html](https://proceedings.mlr.press/v202/huijben23a.html)

**Abstract**:

Continuous monitoring with an ever-increasing number of sensors has become ubiquitous across many application domains. However, acquired time series are typically high-dimensional and difficult to interpret. Expressive deep learning (DL) models have gained popularity for dimensionality reduction, but the resulting latent space often remains difficult to interpret. In this work we propose SOM-CPC, a model that visualizes data in an organized 2D manifold, while preserving higher-dimensional information. We address a largely unexplored and challenging set of scenarios comprising high-rate time series, and show on both synthetic and real-life data (physiological data and audio recordings) that SOM-CPC outperforms strong baselines like DL-based feature extraction, followed by conventional dimensionality reduction techniques, and models that jointly optimize a DL model and a Self-Organizing Map (SOM). SOM-CPC has great potential to acquire a better understanding of latent patterns in high-rate data streams.

----

## [576] One-Shot Federated Conformal Prediction

**Authors**: *Pierre Humbert, Batiste Le Bars, Aurélien Bellet, Sylvain Arlot*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/humbert23a.html](https://proceedings.mlr.press/v202/humbert23a.html)

**Abstract**:

In this paper, we present a Conformal Prediction method that computes prediction sets in a one-shot Federated Learning (FL) setting. More specifically, we introduce a novel quantile-of-quantiles estimator and prove that for any distribution, it is possible to compute prediction sets with desired coverage in only one round of communication. To mitigate privacy issues, we also describe a locally differentially private version of our estimator. Finally, over a wide range of experiments, we show that our method returns prediction sets with coverage and length very similar to those obtained in a centralized setting. These results demonstrate that our method is well-suited for one-shot Federated Learning.

----

## [577] The Impact of Exploration on Convergence and Performance of Multi-Agent Q-Learning Dynamics

**Authors**: *Aamal Abbas Hussain, Francesco Belardinelli, Dario Paccagnan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hussain23a.html](https://proceedings.mlr.press/v202/hussain23a.html)

**Abstract**:

Understanding the impact of exploration on the behaviour of multi-agent learning has, so far, benefited from the restriction to potential, or network zero-sum games in which convergence to an equilibrium can be shown. Outside of these classes, learning dynamics rarely converge and little is known about the effect of exploration in the face of non-convergence. To progress this front, we study the smooth Q- Learning dynamics. We show that, in any network game, exploration by agents results in the convergence of Q-Learning to a neighbourhood of an equilibrium. This holds independently of whether the dynamics reach the equilibrium or display complex behaviours. We show that increasing the exploration rate decreases the size of this neighbourhood and also decreases the ability of all agents to improve their payoffs. Furthermore, in a broad class of games, the payoff performance of Q-Learning dynamics, measured by Social Welfare, decreases when the exploration rate increases. Our experiments show this to be a general phenomenon, namely that exploration leads to improved convergence of Q-Learning, at the cost of payoff performance.

----

## [578] Combinatorial Neural Bandits

**Authors**: *Taehyun Hwang, Kyuwook Chai, Min Hwan Oh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hwang23a.html](https://proceedings.mlr.press/v202/hwang23a.html)

**Abstract**:

We consider a contextual combinatorial bandit problem where in each round a learning agent selects a subset of arms and receives feedback on the selected arms according to their scores. The score of an arm is an unknown function of the arm’s feature. Approximating this unknown score function with deep neural networks, we propose algorithms: Combinatorial Neural UCB ($\texttt{CN-UCB}$) and Combinatorial Neural Thompson Sampling ($\texttt{CN-TS}$). We prove that $\texttt{CN-UCB}$ achieves $\tilde{\mathcal{O}}(\tilde{d} \sqrt{T})$ or $\tilde{\mathcal{O}}(\sqrt{\tilde{d} T K})$ regret, where $\tilde{d}$ is the effective dimension of a neural tangent kernel matrix, $K$ is the size of a subset of arms, and $T$ is the time horizon. For $\texttt{CN-TS}$, we adapt an optimistic sampling technique to ensure the optimism of the sampled combinatorial action, achieving a worst-case (frequentist) regret of $\tilde{\mathcal{O}}(\tilde{d} \sqrt{TK})$. To the best of our knowledge, these are the first combinatorial neural bandit algorithms with regret performance guarantees. In particular, $\texttt{CN-TS}$ is the first Thompson sampling algorithm with the worst-case regret guarantees for the general contextual combinatorial bandit problem. The numerical experiments demonstrate the superior performances of our proposed algorithms.

----

## [579] MAGANet: Achieving Combinatorial Generalization by Modeling a Group Action

**Authors**: *Geonho Hwang, Jaewoong Choi, Hyunsoo Cho, Myungjoo Kang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hwang23b.html](https://proceedings.mlr.press/v202/hwang23b.html)

**Abstract**:

Combinatorial generalization refers to the ability to collect and assemble various attributes from diverse data to generate novel unexperienced data. This ability is considered a necessary passing point for achieving human-level intelligence. To achieve this ability, previous unsupervised approaches mainly focused on learning the disentangled representation, such as the variational autoencoder. However, recent studies discovered that the disentangled representation is insufficient for combinatorial generalization and is not even correlated. In this regard, we propose a novel framework for data generation that can robustly generalize under these distribution shift situations. Instead of representing each data, our model discovers the fundamental transformation between a pair of data by simulating a group action. To test the combinatorial generalizability, we evaluated our model in two settings: Recombination-to-Element and Recombination-to-Range. The experiments demonstrated that our method has quantitatively and qualitatively superior generalizability and generates better images than traditional models.

----

## [580] Information-Theoretic State Space Model for Multi-View Reinforcement Learning

**Authors**: *HyeongJoo Hwang, Seokin Seo, Youngsoo Jang, Sungyoon Kim, Geon-Hyeong Kim, Seunghoon Hong, Kee-Eung Kim*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/hwang23c.html](https://proceedings.mlr.press/v202/hwang23c.html)

**Abstract**:

Multi-View Reinforcement Learning (MVRL) seeks to find an optimal control for an agent given multi-view observations from various sources. Despite recent advances in multi-view learning that aim to extract the latent representation from multi-view data, it is not straightforward to apply them to control tasks, especially when the observations are temporally dependent on one another. The problem can be even more challenging if the observations are intermittently missing for a subset of views. In this paper, we introduce Fuse2Control (F2C), an information-theoretic approach to capturing the underlying state space model from the sequences of multi-view observations. We conduct an extensive set of experiments in various control tasks showing that our method is highly effective in aggregating task-relevant information across many views, that scales linearly with the number of views while retaining robustness to arbitrary missing view scenarios.

----

## [581] Under-Counted Tensor Completion with Neural Incorporation of Attributes

**Authors**: *Shahana Ibrahim, Xiao Fu, Rebecca A. Hutchinson, Eugene Seo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ibrahim23a.html](https://proceedings.mlr.press/v202/ibrahim23a.html)

**Abstract**:

Systematic under-counting effects are observed in data collected across many disciplines, e.g., epidemiology and ecology. Under-counted tensor completion (UC-TC) is well-motivated for many data analytics tasks, e.g., inferring the case numbers of infectious diseases at unobserved locations from under-counted case numbers in neighboring regions. However, existing methods for similar problems often lack supports in theory, making it hard to understand the underlying principles and conditions beyond empirical successes. In this work, a low-rank Poisson tensor model with an expressive unknown nonlinear side information extractor is proposed for under-counted multi-aspect data. A joint low-rank tensor completion and neural network learning algorithm is designed to recover the model. Moreover, the UC-TC formulation is supported by theoretical analysis showing that the fully counted entries of the tensor and each entry’s under-counting probability can be provably recovered from partial observations—under reasonable conditions. To our best knowledge, the result is the first to offer theoretical supports for under-counted multi-aspect data completion. Simulations and real-data experiments corroborate the theoretical claims.

----

## [582] On the Identifiability and Estimation of Causal Location-Scale Noise Models

**Authors**: *Alexander Immer, Christoph Schultheiss, Julia E. Vogt, Bernhard Schölkopf, Peter Bühlmann, Alexander Marx*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/immer23a.html](https://proceedings.mlr.press/v202/immer23a.html)

**Abstract**:

We study the class of location-scale or heteroscedastic noise models (LSNMs), in which the effect $Y$ can be written as a function of the cause $X$ and a noise source $N$ independent of $X$, which may be scaled by a positive function $g$ over the cause, i.e., $Y = f(X) + g(X)N$. Despite the generality of the model class, we show the causal direction is identifiable up to some pathological cases. To empirically validate these theoretical findings, we propose two estimators for LSNMs: an estimator based on (non-linear) feature maps, and one based on neural networks. Both model the conditional distribution of $Y$ given $X$ as a Gaussian parameterized by its natural parameters. When the feature maps are correctly specified, we prove that our estimator is jointly concave, and a consistent estimator for the cause-effect identification task. Although the the neural network does not inherit those guarantees, it can fit functions of arbitrary complexity, and reaches state-of-the-art performance across benchmarks.

----

## [583] Stochastic Marginal Likelihood Gradients using Neural Tangent Kernels

**Authors**: *Alexander Immer, Tycho F. A. van der Ouderaa, Mark van der Wilk, Gunnar Rätsch, Bernhard Schölkopf*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/immer23b.html](https://proceedings.mlr.press/v202/immer23b.html)

**Abstract**:

Selecting hyperparameters in deep learning greatly impacts its effectiveness but requires manual effort and expertise. Recent works show that Bayesian model selection with Laplace approximations can allow to optimize such hyperparameters just like standard neural network parameters using gradients and on the training data. However, estimating a single hyperparameter gradient requires a pass through the entire dataset, limiting the scalability of such algorithms. In this work, we overcome this issue by introducing lower bounds to the linearized Laplace approximation of the marginal likelihood. In contrast to previous estimators, these bounds are amenable to stochastic-gradient-based optimization and allow to trade off estimation accuracy against computational complexity. We derive them using the function-space form of the linearized Laplace, which can be estimated using the neural tangent kernel. Experimentally, we show that the estimators can significantly accelerate gradient-based hyperparameter optimization.

----

## [584] Differentially Private Hierarchical Clustering with Provable Approximation Guarantees

**Authors**: *Jacob Imola, Alessandro Epasto, Mohammad Mahdian, Vincent Cohen-Addad, Vahab Mirrokni*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/imola23a.html](https://proceedings.mlr.press/v202/imola23a.html)

**Abstract**:

Hierarchical Clustering is a popular unsupervised machine learning method with decades of history and numerous applications. We initiate the study of differentially-private approximation algorithms for hierarchical clustering under the rigorous framework introduced by Dasgupta (2016). We show strong lower bounds for the problem: that any $\epsilon$-DP algorithm must exhibit $O(|V|^2/ \epsilon)$-additive error for an input dataset $V$. Then, we exhibit a polynomial-time approximation algorithm with $O(|V|^{2.5}/ \epsilon)$-additive error, and an exponential-time algorithm that meets the lower bound. To overcome the lower bound, we focus on the stochastic block model, a popular model of graphs, and, with a separation assumption on the blocks, propose a private $1+o(1)$ approximation algorithm which also recovers the blocks exactly. Finally, we perform an empirical study of our algorithms and validate their performance.

----

## [585] Neural Network Accelerated Implicit Filtering: Integrating Neural Network Surrogates With Provably Convergent Derivative Free Optimization Methods

**Authors**: *Brian Irwin, Eldad Haber, Raviv Gal, Avi Ziv*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/irwin23a.html](https://proceedings.mlr.press/v202/irwin23a.html)

**Abstract**:

In this paper, we introduce neural network accelerated implicit filtering (NNAIF), a novel family of methods for solving noisy derivative free (i.e. black box, zeroth order) optimization problems. NNAIF intelligently combines the established literature on implicit filtering (IF) optimization methods with a neural network (NN) surrogate model of the objective function, resulting in accelerated derivative free methods for unconstrained optimization problems. The NN surrogate model consists of a fixed number of parameters, which can be as few as $\approx 1.3 \times 10^{4}$, that are updated as NNAIF progresses. We show that NNAIF directly inherits the convergence properties of IF optimization methods, and thus NNAIF is guaranteed to converge towards a critical point of the objective function under appropriate assumptions. Numerical experiments with $31$ noisy problems from the CUTEst optimization benchmark set demonstrate the benefits and costs associated with NNAIF. These benefits include NNAIF’s ability to minimize structured functions of several thousand variables much more rapidly than well-known alternatives, such as Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and finite difference based variants of gradient descent (GD) and BFGS, as well as its namesake IF.

----

## [586] Principled Offline RL in the Presence of Rich Exogenous Information

**Authors**: *Riashat Islam, Manan Tomar, Alex Lamb, Yonathan Efroni, Hongyu Zang, Aniket Rajiv Didolkar, Dipendra Misra, Xin Li, Harm van Seijen, Remi Tachet des Combes, John Langford*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/islam23a.html](https://proceedings.mlr.press/v202/islam23a.html)

**Abstract**:

Learning to control an agent from offline data collected in a rich pixel-based visual observation space is vital for real-world applications of reinforcement learning (RL). A major challenge in this setting is the presence of input information that is hard to model and irrelevant to controlling the agent. This problem has been approached by the theoretical RL community through the lens of exogenous information, i.e., any control-irrelevant information contained in observations. For example, a robot navigating in busy streets needs to ignore irrelevant information, such as other people walking in the background, textures of objects, or birds in the sky. In this paper, we focus on the setting with visually detailed exogenous information and introduce new offline RL benchmarks that offer the ability to study this problem. We find that contemporary representation learning techniques can fail on datasets where the noise is a complex and time-dependent process, which is prevalent in practical applications. To address these, we propose to use multi-step inverse models to learn Agent-Centric Representations for Offline-RL (ACRO). Despite being simple and reward-free, we show theoretically and empirically that the representation created by this objective greatly outperforms baselines.

----

## [587] Unveiling the Latent Space Geometry of Push-Forward Generative Models

**Authors**: *Thibaut Issenhuth, Ugo Tanielian, Jérémie Mary, David Picard*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/issenhuth23a.html](https://proceedings.mlr.press/v202/issenhuth23a.html)

**Abstract**:

Many deep generative models are defined as a push-forward of a Gaussian measure by a continuous generator, such as Generative Adversarial Networks (GANs) or Variational Auto-Encoders (VAEs). This work explores the latent space of such deep generative models. A key issue with these models is their tendency to output samples outside of the support of the target distribution when learning disconnected distributions. We investigate the relationship between the performance of these models and the geometry of their latent space. Building on recent developments in geometric measure theory, we prove a sufficient condition for optimality in the case where the dimension of the latent space is larger than the number of modes. Through experiments on GANs, we demonstrate the validity of our theoretical results and gain new insights into the latent space geometry of these models. Additionally, we propose a truncation method that enforces a simplicial cluster structure in the latent space and improves the performance of GANs.

----

## [588] CO-BED: Information-Theoretic Contextual Optimization via Bayesian Experimental Design

**Authors**: *Desi R. Ivanova, Joel Jennings, Tom Rainforth, Cheng Zhang, Adam Foster*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ivanova23a.html](https://proceedings.mlr.press/v202/ivanova23a.html)

**Abstract**:

We formalize the problem of contextual optimization through the lens of Bayesian experimental design and propose CO-BED—a general, model-agnostic framework for designing contextual experiments using information-theoretic principles. After formulating a suitable information-based objective, we employ black-box variational methods to simultaneously estimate it and optimize the designs in a single stochastic gradient scheme. In addition, to accommodate discrete actions within our framework, we propose leveraging continuous relaxation schemes, which can naturally be integrated into our variational objective. As a result, CO-BED provides a general and automated solution to a wide range of contextual optimization problems. We illustrate its effectiveness in a number of experiments, where CO-BED demonstrates competitive performance even when compared to bespoke, model-specific alternatives.

----

## [589] DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule

**Authors**: *Maor Ivgi, Oliver Hinder, Yair Carmon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ivgi23a.html](https://proceedings.mlr.press/v202/ivgi23a.html)

**Abstract**:

We propose a tuning-free dynamic SGD step size formula, which we call Distance over Gradients (DoG). The DoG step sizes depend on simple empirical quantities (distance from the initial point and norms of gradients) and have no “learning rate” parameter. Theoretically, we show that, for stochastic convex optimization, a slight variation of the DoG formula enjoys strong, high-probability parameter-free convergence guarantees and iterate movement bounds. Empirically, we consider a broad range of vision and language transfer learning tasks, and show that DoG’s performance is close to that of SGD with tuned learning rate. We also propose a per-layer variant of DoG that generally outperforms tuned SGD, approaching the performance of tuned Adam. A PyTorch implementation of our algorithms is available at https://github.com/formll/dog.

----

## [590] Maximal Initial Learning Rates in Deep ReLU Networks

**Authors**: *Gaurav Iyer, Boris Hanin, David Rolnick*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/iyer23a.html](https://proceedings.mlr.press/v202/iyer23a.html)

**Abstract**:

Training a neural network requires choosing a suitable learning rate, which involves a trade-off between speed and effectiveness of convergence. While there has been considerable theoretical and empirical analysis of how large the learning rate can be, most prior work focuses only on late-stage training. In this work, we introduce the maximal initial learning rate $\eta^{\ast}$ - the largest learning rate at which a randomly initialized neural network can successfully begin training and achieve (at least) a given threshold accuracy. Using a simple approach to estimate $\eta^{\ast}$, we observe that in constant-width fully-connected ReLU networks, $\eta^{\ast}$ behaves differently from the maximum learning rate later in training. Specifically, we find that $\eta^{\ast}$ is well predicted as a power of depth $\times$ width, provided that (i) the width of the network is sufficiently large compared to the depth, and (ii) the input layer is trained at a relatively small learning rate. We further analyze the relationship between $\eta^{\ast}$ and the sharpness $\lambda_{1}$ of the network at initialization, indicating they are closely though not inversely related. We formally prove bounds for $\lambda_{1}$ in terms of depth $\times$ width that align with our empirical results.

----

## [591] Data-Driven Subgroup Identification for Linear Regression

**Authors**: *Zachary Izzo, Ruishan Liu, James Zou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/izzo23a.html](https://proceedings.mlr.press/v202/izzo23a.html)

**Abstract**:

Medical studies frequently require to extract the relationship between each covariate and the outcome with statistical confidence measures. To do this, simple parametric models are frequently used (e.g. coefficients of linear regression) but always fitted on the whole dataset. However, it is common that the covariates may not have a uniform effect over the whole population and thus a unified simple model can miss the heterogeneous signal. For example, a linear model may be able to explain a subset of the data but fail on the rest due to the nonlinearity and heterogeneity in the data. In this paper, we propose DDGroup (data-driven group discovery), a data-driven method to effectively identify subgroups in the data with a uniform linear relationship between the features and the label. DDGroup outputs an interpretable region in which the linear model is expected to hold. It is simple to implement and computationally tractable for use. We show theoretically that, given a large enough sample, DDGroup recovers a region where a single linear model with low variance is well-specified (if one exists), and experiments on real-world medical datasets confirm that it can discover regions where a local linear model has improved performance. Our experiments also show that DDGroup can uncover subgroups with qualitatively different relationships which are missed by simply applying parametric approaches to the whole dataset.

----

## [592] Efficient Training of Language Models using Few-Shot Learning

**Authors**: *Sashank J. Reddi, Sobhan Miryoosefi, Stefani Karp, Shankar Krishnan, Satyen Kale, Seungyeon Kim, Sanjiv Kumar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/j-reddi23a.html](https://proceedings.mlr.press/v202/j-reddi23a.html)

**Abstract**:

Large deep learning models have achieved state-of-the-art performance across various natural language processing (NLP) tasks and demonstrated remarkable few-shot learning performance. However, training them is often challenging and resource-intensive. In this paper, we study an efficient approach to train language models using few-shot learners. We show that, by leveraging the fast learning nature of few-shot learners, one can train language models efficiently in a stagewise manner. Our main insight is that stacking a good few-shot learner on a good small language model provides a good initializer for a larger language model. Using this insight and building upon progressive stacking approaches, we develop novel approaches for training such networks in a stagewise manner. Furthermore, we also provide a theoretical framework and accompanying empirical studies to support our insights, thereby creating a theoretical foundation for progressive stacking. Finally, we provide empirical results to demonstrate the effectiveness of our approach in reducing the training time of few-shot learners.

----

## [593] Scalable Adaptive Computation for Iterative Generation

**Authors**: *Allan Jabri, David J. Fleet, Ting Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jabri23a.html](https://proceedings.mlr.press/v202/jabri23a.html)

**Abstract**:

Natural data is redundant yet predominant architectures tile computation uniformly across their input and output space. We propose the Recurrent Interface Network (RIN), an attention-based architecture that decouples its core computation from the dimensionality of the data, enabling adaptive computation for more scalable generation of high-dimensional data. RINs focus the bulk of computation (i.e. global self-attention) on a set of latent tokens, using cross-attention to read and write (i.e. route) information between latent and data tokens. Stacking RIN blocks allows bottom-up (data to latent) and top-down (latent to data) feedback, leading to deeper and more expressive routing. While this routing introduces challenges, this is less problematic in recurrent computation settings where the task (and routing problem) changes gradually, such as iterative generation with diffusion models. We show how to leverage recurrence by conditioning the latent tokens at each forward pass of the reverse diffusion process with those from prior computation, i.e. latent self-conditioning. RINs yield state-of-the-art pixel diffusion models for image and video generation, scaling to1024×1024 images without cascades or guidance, while being domain-agnostic and up to 10× more efficient than 2D and 3D U-Nets.

----

## [594] Unconstrained Online Learning with Unbounded Losses

**Authors**: *Andrew Jacobsen, Ashok Cutkosky*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jacobsen23a.html](https://proceedings.mlr.press/v202/jacobsen23a.html)

**Abstract**:

Algorithms for online learning typically require one or more boundedness assumptions: that the domain is bounded, that the losses are Lipschitz, or both. In this paper, we develop a new setting for online learning with unbounded domains and non-Lipschitz losses. For this setting we provide an algorithm which guarantees $R_{T}(u)\le \tilde O(G\|u\|\sqrt{T}+L\|u\|^{2}\sqrt{T})$ regret on any problem where the subgradients satisfy $\|g_{t}\|\le G+L\|w_{t}\|$, and show that this bound is unimprovable without further assumptions. We leverage this algorithm to develop new saddle-point optimization algorithms that converge in duality gap in unbounded domains, even in the absence of meaningful curvature. Finally, we provide the first algorithm achieving non-trivial dynamic regret in an unbounded domain for non-Lipschitz losses, as well as a matching lower bound. The regret of our dynamic regret algorithm automatically improves to a novel $L^{*}$ bound when the losses are smooth.

----

## [595] Multi-Objective GFlowNets

**Authors**: *Moksh Jain, Sharath Chandra Raparthy, Alex Hernández-García, Jarrid Rector-Brooks, Yoshua Bengio, Santiago Miret, Emmanuel Bengio*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jain23a.html](https://proceedings.mlr.press/v202/jain23a.html)

**Abstract**:

We study the problem of generating diverse candidates in the context of Multi-Objective Optimization. In many applications of machine learning such as drug discovery and material design, the goal is to generate candidates which simultaneously optimize a set of potentially conflicting objectives. Moreover, these objectives are often imperfect evaluations of some underlying property of interest, making it important to generate diverse candidates to have multiple options for expensive downstream evaluations. We propose Multi-Objective GFlowNets (MOGFNs), a novel method for generating diverse Pareto optimal solutions, based on GFlowNets. We introduce two variants of MOGFNs: MOGFN-PC, which models a family of independent sub-problems defined by a scalarization function, with reward-conditional GFlowNets, and MOGFN-AL, which solves a sequence of sub-problems defined by an acquisition function in an active learning loop. Our experiments on wide variety of synthetic and benchmark tasks demonstrate advantages of the proposed methods in terms of the Pareto performance and importantly, improved candidate diversity, which is the main contribution of this work.

----

## [596] The Price of Differential Privacy under Continual Observation

**Authors**: *Palak Jain, Sofya Raskhodnikova, Satchit Sivakumar, Adam D. Smith*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jain23b.html](https://proceedings.mlr.press/v202/jain23b.html)

**Abstract**:

We study the accuracy of differentially private mechanisms in the continual release model. A continual release mechanism receives a sensitive dataset as a stream of $T$ inputs and produces, after receiving each input, an output that is accurate for all the inputs received so far. We provide the first strong lower bounds on the error of continual release mechanisms. In particular, for two fundamental problems that are closely related to empirical risk minimization and widely studied and used in the standard (batch) model, we prove that the worst case error of every continual release algorithm is $\tilde \Omega(T^{1/3})$ times larger than that of the best batch algorithm. Previous work shows only a $\Omega(\log T)$ gap between the worst case error achievable in these two models. We also formulate a model that allows for adaptively selected inputs, thus capturing dependencies that arise in many applications of continual release. Even though, in general, both privacy and accuracy are harder to attain in this model, we show that our lower bounds are matched by the error of simple algorithms that work even for adaptively selected inputs.

----

## [597] Graph Ladling: Shockingly Simple Parallel GNN Training without Intermediate Communication

**Authors**: *Ajay Kumar Jaiswal, Shiwei Liu, Tianlong Chen, Ying Ding, Zhangyang Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jaiswal23a.html](https://proceedings.mlr.press/v202/jaiswal23a.html)

**Abstract**:

Graphs are omnipresent and GNNs are a powerful family of neural networks for learning over graphs. Despite their popularity, scaling GNNs either by deepening or widening suffers from prevalent issues of $\textit{unhealthy gradients, over-smoothening, information squashing}$, which often lead to sub-standard performance. In this work, we are interested in exploring a principled way to scale GNNs capacity without deepening or widening, which can improve its performance across multiple small and large graphs. Motivated by the recent intriguing phenomenon of model soups, which suggest that fine-tuned weights of multiple large-language pre-trained models can be merged to a better minima, we argue to exploit the fundamentals of model soups to mitigate the aforementioned issues of memory bottleneck and trainability during GNNs scaling. More specifically, we propose not to deepen or widen current GNNs, but instead present $\textbf{first data-centric perspective}$ of model soups to build powerful GNNs by dividing giant graph data to build independently and parallelly trained multiple comparatively weaker GNNs without any intermediate communication, and $\textit{combining their strength}$ using a greedy interpolation soup procedure to achieve state-of-the-art performance. Moreover, we provide a wide variety of model soup preparation techniques by leveraging state-of-the-art graph sampling and graph partitioning approaches that can handle large graph data structures. Our extensive experiments across many real-world small and large graphs, illustrate the effectiveness of our approach and point towards a promising orthogonal direction for GNN scaling. Codes are available at: https://github.com/VITA-Group/graph_ladling

----

## [598] Instant Soup: Cheap Pruning Ensembles in A Single Pass Can Draw Lottery Tickets from Large Models

**Authors**: *Ajay Kumar Jaiswal, Shiwei Liu, Tianlong Chen, Ying Ding, Zhangyang Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jaiswal23b.html](https://proceedings.mlr.press/v202/jaiswal23b.html)

**Abstract**:

Large pre-trained transformers have been receiving explosive attention in the past few years, due to their acculturation for numerous downstream applications via fine-tuning, but their exponentially increasing parameter counts are becoming a primary hurdle to even just fine-tune them without industry-standard hardware. Recently, Lottery Ticket Hypothesis (LTH) and its variants, have been exploited to prune these large pre-trained models generating subnetworks which can achieve similar performance as their dense counterparts, but LTH pragmatism is enormously inhibited by repetitive full training and pruning routine of iterative magnitude pruning (IMP) which worsens with increasing model size. Motivated by the recent observations of model soups, which suggest that fine-tuned weights of multiple models can be merged to a better minima, we propose Instant Soup Pruning (ISP) to generate lottery ticket quality subnetworks, using a fraction of the original IMP cost by replacing the expensive intermediate pruning stages of IMP with computationally efficient weak mask generation and aggregation routine. More specifically, during the mask generation stage, ISP takes a small handful of iterations using varying training protocols and data subsets to generate many weak and noisy subnetworks, and superpose them to average out the noise creating a high-quality denoised subnetwork. Our extensive experiments and ablation on two popular large-scale pre-trained models: $\texttt{CLIP} (unexplored in pruning till date)$ and $\texttt{BERT}$ across multiple benchmark vision $\texttt{\{MNIST, SVHN, Cars, GTSRB, CIFAR-10, CIFAR-100\}}$ and language datasets $\texttt{\{MNLI, QNLI, QQP, SST, ...\}}$ validate the effectiveness of ISP compared to several state-of-the-art pruning methods. Additionally, we show that ISP can be easily modified with minimal overhead to produce benefits comparable to model soups, without the prerequisite to generate multiple candidates fine-tuned models. Codes are available at: https://github.com/VITA-Group/instant_soup.

----

## [599] Exploring the Benefits of Training Expert Language Models over Instruction Tuning

**Authors**: *Joel Jang, Seungone Kim, Seonghyeon Ye, Doyoung Kim, Lajanugen Logeswaran, Moontae Lee, Kyungjae Lee, Minjoon Seo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/jang23a.html](https://proceedings.mlr.press/v202/jang23a.html)

**Abstract**:

Recently, Language Models (LMs) instruction-tuned on multiple tasks, also known as multitask-prompted fine-tuning (MT), have shown capabilities to generalize to unseen tasks. Previous work has shown that scaling the number of finetuning datasets and instructions is the key component in making stronger MT LMs. In this work, we report surprising findings that show an expert LM trained on just a single task can outperform an MT LM trained with 300+ different tasks on 11 different unseen datasets and on 13 datasets of the BIG-bench benchmark by an average of 3.20% and 1.29%, respectively. This finding casts doubt on the previously held belief that simply scaling the number of tasks makes stronger MT LMs. Leveraging this finding, we further show that this distributed approach of training multiple expert LMs instead of a single MT LM for zero-shot inference possesses many benefits including (1) avoiding negative task transfer that often occurs during instruction tuning, (2) being able to continually learn new tasks without having to re-train on previous tasks to avoid catastrophic forgetting, and (3) showing compositional capabilities when merging individual experts together.

----



[Go to the previous page](ICML-2023-list02.md)

[Go to the next page](ICML-2023-list04.md)

[Go to the catalog section](README.md)