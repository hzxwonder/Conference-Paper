## [800] DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

**Authors**: *Samyam Rajbhandari, Conglong Li, Zhewei Yao, Minjia Zhang, Reza Yazdani Aminabadi, Ammar Ahmad Awan, Jeff Rasley, Yuxiong He*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rajbhandari22a.html](https://proceedings.mlr.press/v162/rajbhandari22a.html)

**Abstract**:

As the training of giant dense models hits the boundary on the availability and capability of the hardware resources today, Mixture-of-Experts (MoE) models have become one of the most promising model architectures due to their significant training cost reduction compared to quality-equivalent dense models. Their training cost saving is demonstrated from encoder-decoder models (prior works) to a 5x saving for auto-aggressive language models (this work). However, due to the much larger model size and unique architecture, how to provide fast MoE model inference remains challenging and unsolved, limiting their practical usage. To tackle this, we present DeepSpeed-MoE, an end-to-end MoE training and inference solution, including novel MoE architecture designs and model compression techniques that reduce MoE model size by up to 3.7x, and a highly optimized inference system that provides 7.3x better latency and cost compared to existing MoE inference solutions. DeepSpeed-MoE offers an unprecedented scale and efficiency to serve massive MoE models with up to 4.5x faster and 9x cheaper inference compared to quality-equivalent dense models. We hope our innovations and systems help open a promising path to new directions in the large model landscape, a shift from dense to sparse MoE models, where training and deploying higher-quality models with fewer resources becomes more widely possible.

----

## [801] Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization

**Authors**: *Alexandre Ramé, Corentin Dancette, Matthieu Cord*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rame22a.html](https://proceedings.mlr.press/v162/rame22a.html)

**Abstract**:

Learning robust models that generalize well under changes in the data distribution is critical for real-world applications. To this end, there has been a growing surge of interest to learn simultaneously from multiple training domains - while enforcing different types of invariance across those domains. Yet, all existing approaches fail to show systematic benefits under controlled evaluation protocols. In this paper, we introduce a new regularization - named Fishr - that enforces domain invariance in the space of the gradients of the loss: specifically, the domain-level variances of gradients are matched across training domains. Our approach is based on the close relations between the gradient covariance, the Fisher Information and the Hessian of the loss: in particular, we show that Fishr eventually aligns the domain-level loss landscapes locally around the final weights. Extensive experiments demonstrate the effectiveness of Fishr for out-of-distribution generalization. Notably, Fishr improves the state of the art on the DomainBed benchmark and performs consistently better than Empirical Risk Minimization. Our code is available at https://github.com/alexrame/fishr.

----

## [802] A Closer Look at Smoothness in Domain Adversarial Training

**Authors**: *Harsh Rangwani, Sumukh K. Aithal, Mayank Mishra, Arihant Jain, Venkatesh Babu Radhakrishnan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rangwani22a.html](https://proceedings.mlr.press/v162/rangwani22a.html)

**Abstract**:

Domain adversarial training has been ubiquitous for achieving invariant representations and is used widely for various domain adaptation tasks. In recent times, methods converging to smooth optima have shown improved generalization for supervised learning tasks like classification. In this work, we analyze the effect of smoothness enhancing formulations on domain adversarial training, the objective of which is a combination of task loss (eg. classification, regression etc.) and adversarial terms. We find that converging to a smooth minima with respect to (w.r.t.) task loss stabilizes the adversarial training leading to better performance on target domain. In contrast to task loss, our analysis shows that converging to smooth minima w.r.t. adversarial loss leads to sub-optimal generalization on the target domain. Based on the analysis, we introduce the Smooth Domain Adversarial Training (SDAT) procedure, which effectively enhances the performance of existing domain adversarial methods for both classification and object detection tasks. Our analysis also provides insight into the extensive usage of SGD over Adam in the community for domain adversarial training.

----

## [803] Linear Adversarial Concept Erasure

**Authors**: *Shauli Ravfogel, Michael Twiton, Yoav Goldberg, Ryan Cotterell*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ravfogel22a.html](https://proceedings.mlr.press/v162/ravfogel22a.html)

**Abstract**:

Modern neural models trained on textual data rely on pre-trained representations that emerge without direct supervision. As these representations are increasingly being used in real-world applications, the inability to control their content becomes an increasingly important problem. In this work, we formulate the problem of identifying a linear subspace that corresponds to a given concept, and removing it from the representation. We formulate this problem as a constrained, linear minimax game, and show that existing solutions are generally not optimal for this task. We derive a closed-form solution for certain objectives, and propose a convex relaxation that works well for others. When evaluated in the context of binary gender removal, the method recovers a low-dimensional subspace whose removal mitigates bias by intrinsic and extrinsic evaluation. Surprisingly, we show that the method—despite being linear—is highly expressive, effectively mitigating bias in the output layers of deep, nonlinear classifiers while maintaining tractability and interpretability.

----

## [804] Implicit Regularization in Hierarchical Tensor Factorization and Deep Convolutional Neural Networks

**Authors**: *Noam Razin, Asaf Maman, Nadav Cohen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/razin22a.html](https://proceedings.mlr.press/v162/razin22a.html)

**Abstract**:

In the pursuit of explaining implicit regularization in deep learning, prominent focus was given to matrix and tensor factorizations, which correspond to simplified neural networks. It was shown that these models exhibit an implicit tendency towards low matrix and tensor ranks, respectively. Drawing closer to practical deep learning, the current paper theoretically analyzes the implicit regularization in hierarchical tensor factorization, a model equivalent to certain deep convolutional neural networks. Through a dynamical systems lens, we overcome challenges associated with hierarchy, and establish implicit regularization towards low hierarchical tensor rank. This translates to an implicit regularization towards locality for the associated convolutional networks. Inspired by our theory, we design explicit regularization discouraging locality, and demonstrate its ability to improve the performance of modern convolutional networks on non-local tasks, in defiance of conventional wisdom by which architectural changes are needed. Our work highlights the potential of enhancing neural networks via theoretical analysis of their implicit regularization.

----

## [805] One-Pass Algorithms for MAP Inference of Nonsymmetric Determinantal Point Processes

**Authors**: *Aravind Reddy, Ryan A. Rossi, Zhao Song, Anup B. Rao, Tung Mai, Nedim Lipka, Gang Wu, Eunyee Koh, Nesreen K. Ahmed*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/reddy22a.html](https://proceedings.mlr.press/v162/reddy22a.html)

**Abstract**:

In this paper, we initiate the study of one-pass algorithms for solving the maximum-a-posteriori (MAP) inference problem for Non-symmetric Determinantal Point Processes (NDPPs). In particular, we formulate streaming and online versions of the problem and provide one-pass algorithms for solving these problems. In our streaming setting, data points arrive in an arbitrary order and the algorithms are constrained to use a single-pass over the data as well as sub-linear memory, and only need to output a valid solution at the end of the stream. Our online setting has an additional requirement of maintaining a valid solution at any point in time. We design new one-pass algorithms for these problems and show that they perform comparably to (or even better than) the offline greedy algorithm while using substantially lower memory.

----

## [806] Universality of Winning Tickets: A Renormalization Group Perspective

**Authors**: *William T. Redman, Tianlong Chen, Zhangyang Wang, Akshunna S. Dogra*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/redman22a.html](https://proceedings.mlr.press/v162/redman22a.html)

**Abstract**:

Foundational work on the Lottery Ticket Hypothesis has suggested an exciting corollary: winning tickets found in the context of one task can be transferred to similar tasks, possibly even across different architectures. This has generated broad interest, but methods to study this universality are lacking. We make use of renormalization group theory, a powerful tool from theoretical physics, to address this need. We find that iterative magnitude pruning, the principal algorithm used for discovering winning tickets, is a renormalization group scheme, and can be viewed as inducing a flow in parameter space. We demonstrate that ResNet-50 models with transferable winning tickets have flows with common properties, as would be expected from the theory. Similar observations are made for BERT models, with evidence that their flows are near fixed points. Additionally, we leverage our framework to study winning tickets transferred across ResNet architectures, observing that smaller models have flows with more uniform properties than larger models, complicating transfer between them.

----

## [807] The dynamics of representation learning in shallow, non-linear autoencoders

**Authors**: *Maria Refinetti, Sebastian Goldt*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/refinetti22a.html](https://proceedings.mlr.press/v162/refinetti22a.html)

**Abstract**:

Autoencoders are the simplest neural network for unsupervised learning, and thus an ideal framework for studying feature learning. While a detailed understanding of the dynamics of linear autoencoders has recently been obtained, the study of non-linear autoencoders has been hindered by the technical difficulty of handling training data with non-trivial correlations {–} a fundamental prerequisite for feature extraction. Here, we study the dynamics of feature learning in non-linear, shallow autoencoders. We derive a set of asymptotically exact equations that describe the generalisation dynamics of autoencoders trained with stochastic gradient descent (SGD) in the limit of high-dimensional inputs. These equations reveal that autoencoders learn the leading principal components of their inputs sequentially. An analysis of the long-time dynamics explains the failure of sigmoidal autoencoders to learn with tied weights, and highlights the importance of training the bias in ReLU autoencoders. Building on previous results for linear networks, we analyse a modification of the vanilla SGD algorithm which allows learning of the exact principal components. Finally, we show that our equations accurately describe the generalisation dynamics of non-linear autoencoders on realistic datasets such as CIFAR10.

----

## [808] Proximal Exploration for Model-guided Protein Sequence Design

**Authors**: *Zhizhou Ren, Jiahan Li, Fan Ding, Yuan Zhou, Jianzhu Ma, Jian Peng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ren22a.html](https://proceedings.mlr.press/v162/ren22a.html)

**Abstract**:

Designing protein sequences with a particular biological function is a long-lasting challenge for protein engineering. Recent advances in machine-learning-guided approaches focus on building a surrogate sequence-function model to reduce the burden of expensive in-lab experiments. In this paper, we study the exploration mechanism of model-guided sequence design. We leverage a natural property of protein fitness landscape that a concise set of mutations upon the wild-type sequence are usually sufficient to enhance the desired function. By utilizing this property, we propose Proximal Exploration (PEX) algorithm that prioritizes the evolutionary search for high-fitness mutants with low mutation counts. In addition, we develop a specialized model architecture, called Mutation Factorization Network (MuFacNet), to predict low-order mutational effects, which further improves the sample efficiency of model-guided evolution. In experiments, we extensively evaluate our method on a suite of in-silico protein sequence design tasks and demonstrate substantial improvement over baseline algorithms.

----

## [809] Towards Theoretical Analysis of Transformation Complexity of ReLU DNNs

**Authors**: *Jie Ren, Mingjie Li, Meng Zhou, Shih-Han Chan, Quanshi Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ren22b.html](https://proceedings.mlr.press/v162/ren22b.html)

**Abstract**:

This paper aims to theoretically analyze the complexity of feature transformations encoded in piecewise linear DNNs with ReLU layers. We propose metrics to measure three types of complexities of transformations based on the information theory. We further discover and prove the strong correlation between the complexity and the disentanglement of transformations. Based on the proposed metrics, we analyze two typical phenomena of the change of the transformation complexity during the training process, and explore the ceiling of a DNN’s complexity. The proposed metrics can also be used as a loss to learn a DNN with the minimum complexity, which also controls the over-fitting level of the DNN and influences adversarial robustness, adversarial transferability, and knowledge consistency. Comprehensive comparative studies have provided new perspectives to understand the DNN. The code is released at https://github.com/sjtu-XAI-lab/transformation-complexity.

----

## [810] Benchmarking and Analyzing Point Cloud Classification under Corruptions

**Authors**: *Jiawei Ren, Liang Pan, Ziwei Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ren22c.html](https://proceedings.mlr.press/v162/ren22c.html)

**Abstract**:

3D perception, especially point cloud classification, has achieved substantial progress. However, in real-world deployment, point cloud corruptions are inevitable due to the scene complexity, sensor inaccuracy, and processing imprecision. In this work, we aim to rigorously benchmark and analyze point cloud classification under corruptions. To conduct a systematic investigation, we first provide a taxonomy of common 3D corruptions and identify the atomic corruptions. Then, we perform a comprehensive evaluation on a wide range of representative point cloud models to understand their robustness and generalizability. Our benchmark results show that although point cloud classification performance improves over time, the state-of-the-art methods are on the verge of being less robust. Based on the obtained observations, we propose several effective techniques to enhance point cloud classifier robustness. We hope our comprehensive benchmark, in-depth analysis, and proposed techniques could spark future research in robust 3D perception.

----

## [811] A Unified View on PAC-Bayes Bounds for Meta-Learning

**Authors**: *Arezou Rezazadeh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rezazadeh22a.html](https://proceedings.mlr.press/v162/rezazadeh22a.html)

**Abstract**:

Meta learning automatically infers an inductive bias, that includes the hyperparameter of the baselearning algorithm, by observing data from a finite number of related tasks. This paper studies PAC-Bayes bounds on meta generalization gap. The meta-generalization gap comprises two sources of generalization gaps: the environmentlevel and task-level gaps resulting from observation of a finite number of tasks and data samples per task, respectively. In this paper, by upper bounding arbitrary convex functions, which link the expected and empirical losses at the environment and also per-task levels, we obtain new PAC-Bayes bounds. Using these bounds, we develop new PAC-Bayes meta-learning algorithms. Numerical examples demonstrate the merits of the proposed novel bounds and algorithm in comparison to prior PAC-Bayes bounds for meta-learning

----

## [812] 3PC: Three Point Compressors for Communication-Efficient Distributed Training and a Better Theory for Lazy Aggregation

**Authors**: *Peter Richtárik, Igor Sokolov, Elnur Gasanov, Ilyas Fatkhullin, Zhize Li, Eduard Gorbunov*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/richtarik22a.html](https://proceedings.mlr.press/v162/richtarik22a.html)

**Abstract**:

We propose and study a new class of gradient compressors for communication-efficient training—three point compressors (3PC)—as well as efficient distributed nonconvex optimization algorithms that can take advantage of them. Unlike most established approaches, which rely on a static compressor choice (e.g., TopK), our class allows the compressors to evolve throughout the training process, with the aim of improving the theoretical communication complexity and practical efficiency of the underlying methods. We show that our general approach can recover the recently proposed state-of-the-art error feedback mechanism EF21 (Richtárik et al, 2021) and its theoretical properties as a special case, but also leads to a number of new efficient methods. Notably, our approach allows us to improve upon the state-of-the-art in the algorithmic and theoretical foundations of the lazy aggregation literature (Liu et al, 2017; Lan et al, 2017). As a by-product that may be of independent interest, we provide a new and fundamental link between the lazy aggregation and error feedback literature. A special feature of our work is that we do not require the compressors to be unbiased.

----

## [813] Robust SDE-Based Variational Formulations for Solving Linear PDEs via Deep Learning

**Authors**: *Lorenz Richter, Julius Berner*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/richter22a.html](https://proceedings.mlr.press/v162/richter22a.html)

**Abstract**:

The combination of Monte Carlo methods and deep learning has recently led to efficient algorithms for solving partial differential equations (PDEs) in high dimensions. Related learning problems are often stated as variational formulations based on associated stochastic differential equations (SDEs), which allow the minimization of corresponding losses using gradient-based optimization methods. In respective numerical implementations it is therefore crucial to rely on adequate gradient estimators that exhibit low variance in order to reach convergence accurately and swiftly. In this article, we rigorously investigate corresponding numerical aspects that appear in the context of linear Kolmogorov PDEs. In particular, we systematically compare existing deep learning approaches and provide theoretical explanations for their performances. Subsequently, we suggest novel methods that can be shown to be more robust both theoretically and numerically, leading to substantial performance improvements.

----

## [814] Probabilistically Robust Learning: Balancing Average and Worst-case Performance

**Authors**: *Alexander Robey, Luiz F. O. Chamon, George J. Pappas, Hamed Hassani*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/robey22a.html](https://proceedings.mlr.press/v162/robey22a.html)

**Abstract**:

Many of the successes of machine learning are based on minimizing an averaged loss function. However, it is well-known that this paradigm suffers from robustness issues that hinder its applicability in safety-critical domains. These issues are often addressed by training against worst-case perturbations of data, a technique known as adversarial training. Although empirically effective, adversarial training can be overly conservative, leading to unfavorable trade-offs between nominal performance and robustness. To this end, in this paper we propose a framework called probabilistic robustness that bridges the gap between the accurate, yet brittle average case and the robust, yet conservative worst case by enforcing robustness to most rather than to all perturbations. From a theoretical point of view, this framework overcomes the trade-offs between the performance and the sample-complexity of worst-case and average-case learning. From a practical point of view, we propose a novel algorithm based on risk-aware optimization that effectively balances average- and worst-case performance at a considerably lower computational cost relative to adversarial training. Our results on MNIST, CIFAR-10, and SVHN illustrate the advantages of this framework on the spectrum from average- to worst-case robustness. Our code is available at: https://github.com/arobey1/advbench.

----

## [815] LyaNet: A Lyapunov Framework for Training Neural ODEs

**Authors**: *Ivan Dario Jimenez Rodriguez, Aaron D. Ames, Yisong Yue*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rodriguez22a.html](https://proceedings.mlr.press/v162/rodriguez22a.html)

**Abstract**:

We propose a method for training ordinary differential equations by using a control-theoretic Lyapunov condition for stability. Our approach, called LyaNet, is based on a novel Lyapunov loss formulation that encourages the inference dynamics to converge quickly to the correct prediction. Theoretically, we show that minimizing Lyapunov loss guarantees exponential convergence to the correct solution and enables a novel robustness guarantee. We also provide practical algorithms, including one that avoids the cost of backpropagating through a solver or using the adjoint method. Relative to standard Neural ODE training, we empirically find that LyaNet can offer improved prediction performance, faster convergence of inference dynamics, and improved adversarial robustness. Our code is available at https://github.com/ivandariojr/LyapunovLearning.

----

## [816] Short-Term Plasticity Neurons Learning to Learn and Forget

**Authors**: *Hector Garcia Rodriguez, Qinghai Guo, Timoleon Moraitis*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rodriguez22b.html](https://proceedings.mlr.press/v162/rodriguez22b.html)

**Abstract**:

Short-term plasticity (STP) is a mechanism that stores decaying memories in synapses of the cerebral cortex. In computing practice, STP has been used, but mostly in the niche of spiking neurons, even though theory predicts that it is the optimal solution to certain dynamic tasks. Here we present a new type of recurrent neural unit, the STP Neuron (STPN), which indeed turns out strikingly powerful. Its key mechanism is that synapses have a state, propagated through time by a self-recurrent connection-within-the-synapse. This formulation enables training the plasticity with backpropagation through time, resulting in a form of learning to learn and forget in the short term. The STPN outperforms all tested alternatives, i.e. RNNs, LSTMs, other models with fast weights, and differentiable plasticity. We confirm this in both supervised and reinforcement learning (RL), and in tasks such as Associative Retrieval, Maze Exploration, Atari video games, and MuJoCo robotics. Moreover, we calculate that, in neuromorphic or biological circuits, the STPN minimizes energy consumption across models, as it depresses individual synapses dynamically. Based on these, biological STP may have been a strong evolutionary attractor that maximizes both efficiency and computational power. The STPN now brings these neuromorphic advantages also to a broad spectrum of machine learning practice. Code is available in https://github.com/NeuromorphicComputing/stpn.

----

## [817] Function-space Inference with Sparse Implicit Processes

**Authors**: *Simón Rodríguez Santana, Bryan Zaldivar, Daniel Hernández-Lobato*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rodri-guez-santana22a.html](https://proceedings.mlr.press/v162/rodri-guez-santana22a.html)

**Abstract**:

Implicit Processes (IPs) represent a flexible framework that can be used to describe a wide variety of models, from Bayesian neural networks, neural samplers and data generators to many others. IPs also allow for approximate inference in function-space. This change of formulation solves intrinsic degenerate problems of parameter-space approximate inference concerning the high number of parameters and their strong dependencies in large models. For this, previous works in the literature have attempted to employ IPs both to set up the prior and to approximate the resulting posterior. However, this has proven to be a challenging task. Existing methods that can tune the prior IP result in a Gaussian predictive distribution, which fails to capture important data patterns. By contrast, methods producing flexible predictive distributions by using another IP to approximate the posterior process cannot tune the prior IP to the observed data. We propose here the first method that can accomplish both goals. For this, we rely on an inducing-point representation of the prior IP, as often done in the context of sparse Gaussian processes. The result is a scalable method for approximate inference with IPs that can tune the prior IP parameters to the data, and that provides accurate non-Gaussian predictive distributions.

----

## [818] Score Matching Enables Causal Discovery of Nonlinear Additive Noise Models

**Authors**: *Paul Rolland, Volkan Cevher, Matthäus Kleindessner, Chris Russell, Dominik Janzing, Bernhard Schölkopf, Francesco Locatello*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rolland22a.html](https://proceedings.mlr.press/v162/rolland22a.html)

**Abstract**:

This paper demonstrates how to recover causal graphs from the score of the data distribution in non-linear additive (Gaussian) noise models. Using score matching algorithms as a building block, we show how to design a new generation of scalable causal discovery methods. To showcase our approach, we also propose a new efficient method for approximating the score’s Jacobian, enabling to recover the causal graph. Empirically, we find that the new algorithm, called SCORE, is competitive with state-of-the-art causal discovery methods while being significantly faster.

----

## [819] Dual Decomposition of Convex Optimization Layers for Consistent Attention in Medical Images

**Authors**: *Tom Ron, Tamir Hazan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/ron22a.html](https://proceedings.mlr.press/v162/ron22a.html)

**Abstract**:

A key concern in integrating machine learning models in medicine is the ability to interpret their reasoning. Popular explainability methods have demonstrated satisfactory results in natural image recognition, yet in medical image analysis, many of these approaches provide partial and noisy explanations. Recently, attention mechanisms have shown compelling results both in their predictive performance and in their interpretable qualities. A fundamental trait of attention is that it leverages salient parts of the input which contribute to the model’s prediction. To this end, our work focuses on the explanatory value of attention weight distributions. We propose a multi-layer attention mechanism that enforces consistent interpretations between attended convolutional layers using convex optimization. We apply duality to decompose the consistency constraints between the layers by reparameterizing their attention probability distributions. We further suggest learning the dual witness by optimizing with respect to our objective; thus, our implementation uses standard back-propagation, hence it is highly efficient. While preserving predictive performance, our proposed method leverages weakly annotated medical imaging data and provides complete and faithful explanations to the model’s prediction.

----

## [820] A Consistent and Efficient Evaluation Strategy for Attribution Methods

**Authors**: *Yao Rong, Tobias Leemann, Vadim Borisov, Gjergji Kasneci, Enkelejda Kasneci*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rong22a.html](https://proceedings.mlr.press/v162/rong22a.html)

**Abstract**:

With a variety of local feature attribution methods being proposed in recent years, follow-up work suggested several evaluation strategies. To assess the attribution quality across different attribution techniques, the most popular among these evaluation strategies in the image domain use pixel perturbations. However, recent advances discovered that different evaluation strategies produce conflicting rankings of attribution methods and can be prohibitively expensive to compute. In this work, we present an information-theoretic analysis of evaluation strategies based on pixel perturbations. Our findings reveal that the results are strongly affected by information leakage through the shape of the removed pixels as opposed to their actual values. Using our theoretical insights, we propose a novel evaluation framework termed Remove and Debias (ROAD) which offers two contributions: First, it mitigates the impact of the confounders, which entails higher consistency among evaluation strategies. Second, ROAD does not require the computationally expensive retraining step and saves up to 99% in computational costs compared to the state-of-the-art. We release our source code at https://github.com/tleemann/road_evaluation.

----

## [821] Efficiently Learning the Topology and Behavior of a Networked Dynamical System Via Active Queries

**Authors**: *Daniel J. Rosenkrantz, Abhijin Adiga, Madhav V. Marathe, Zirou Qiu, S. S. Ravi, Richard Edwin Stearns, Anil Vullikanti*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rosenkrantz22a.html](https://proceedings.mlr.press/v162/rosenkrantz22a.html)

**Abstract**:

Using a discrete dynamical system model, many papers have addressed the problem of learning the behavior (i.e., the local function at each node) of a networked system through active queries, assuming that the network topology is known. We address the problem of inferring both the topology of the network and the behavior of a discrete dynamical system through active queries. We consider two query models studied in the literature, namely the batch model (where all the queries must be submitted together) and the adaptive model (where responses to previous queries can be used in formulating a new query). Our results are for systems where the state of each node is from {0,1} and the local functions are Boolean. We present algorithms to learn the topology and the behavior under both batch and adaptive query models for several classes of dynamical systems. These algorithms use only a polynomial number of queries. We also present experimental results obtained by running our query generation algorithms on synthetic and real-world networks.

----

## [822] Learning to Infer Structures of Network Games

**Authors**: *Emanuele Rossi, Federico Monti, Yan Leng, Michael M. Bronstein, Xiaowen Dong*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rossi22a.html](https://proceedings.mlr.press/v162/rossi22a.html)

**Abstract**:

Strategic interactions between a group of individuals or organisations can be modelled as games played on networks, where a player’s payoff depends not only on their actions but also on those of their neighbours. Inferring the network structure from observed game outcomes (equilibrium actions) is an important problem with numerous potential applications in economics and social sciences. Existing methods mostly require the knowledge of the utility function associated with the game, which is often unrealistic to obtain in real-world scenarios. We adopt a transformer-like architecture which correctly accounts for the symmetries of the problem and learns a mapping from the equilibrium actions to the network structure of the game without explicit knowledge of the utility function. We test our method on three different types of network games using both synthetic and real-world data, and demonstrate its effectiveness in network structure inference and superior performance over existing methods.

----

## [823] Direct Behavior Specification via Constrained Reinforcement Learning

**Authors**: *Julien Roy, Roger Girgis, Joshua Romoff, Pierre-Luc Bacon, Christopher J. Pal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/roy22a.html](https://proceedings.mlr.press/v162/roy22a.html)

**Abstract**:

The standard formulation of Reinforcement Learning lacks a practical way of specifying what are admissible and forbidden behaviors. Most often, practitioners go about the task of behavior specification by manually engineering the reward function, a counter-intuitive process that requires several iterations and is prone to reward hacking by the agent. In this work, we argue that constrained RL, which has almost exclusively been used for safe RL, also has the potential to significantly reduce the amount of work spent for reward specification in applied RL projects. To this end, we propose to specify behavioral preferences in the CMDP framework and to use Lagrangian methods to automatically weigh each of these behavioral constraints. Specifically, we investigate how CMDPs can be adapted to solve goal-based tasks while adhering to several constraints simultaneously. We evaluate this framework on a set of continuous control tasks relevant to the application of Reinforcement Learning for NPC design in video games.

----

## [824] Constraint-based graph network simulator

**Authors**: *Yulia Rubanova, Alvaro Sanchez-Gonzalez, Tobias Pfaff, Peter W. Battaglia*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rubanova22a.html](https://proceedings.mlr.press/v162/rubanova22a.html)

**Abstract**:

In the area of physical simulations, nearly all neural-network-based methods directly predict future states from the input states. However, many traditional simulation engines instead model the constraints of the system and select the state which satisfies them. Here we present a framework for constraint-based learned simulation, where a scalar constraint function is implemented as a graph neural network, and future predictions are computed by solving the optimization problem defined by the learned constraint. Our model achieves comparable or better accuracy to top learned simulators on a variety of challenging physical domains, and offers several unique advantages. We can improve the simulation accuracy on a larger system by applying more solver iterations at test time. We also can incorporate novel hand-designed constraints at test time and simulate new dynamics which were not present in the training data. Our constraint-based framework shows how key techniques from traditional simulation and numerical methods can be leveraged as inductive biases in machine learning simulators.

----

## [825] Continual Learning via Sequential Function-Space Variational Inference

**Authors**: *Tim G. J. Rudner, Freddie Bickford Smith, Qixuan Feng, Yee Whye Teh, Yarin Gal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rudner22a.html](https://proceedings.mlr.press/v162/rudner22a.html)

**Abstract**:

Sequential Bayesian inference over predictive functions is a natural framework for continual learning from streams of data. However, applying it to neural networks has proved challenging in practice. Addressing the drawbacks of existing techniques, we propose an optimization objective derived by formulating continual learning as sequential function-space variational inference. In contrast to existing methods that regularize neural network parameters directly, this objective allows parameters to vary widely during training, enabling better adaptation to new tasks. Compared to objectives that directly regularize neural network predictions, the proposed objective allows for more flexible variational distributions and more effective regularization. We demonstrate that, across a range of task sequences, neural networks trained via sequential function-space variational inference achieve better predictive accuracy than networks trained with related methods while depending less on maintaining a set of representative points from previous tasks.

----

## [826] Graph-Coupled Oscillator Networks

**Authors**: *T. Konstantin Rusch, Ben Chamberlain, James Rowbottom, Siddhartha Mishra, Michael M. Bronstein*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rusch22a.html](https://proceedings.mlr.press/v162/rusch22a.html)

**Abstract**:

We propose Graph-Coupled Oscillator Networks (GraphCON), a novel framework for deep learning on graphs. It is based on discretizations of a second-order system of ordinary differential equations (ODEs), which model a network of nonlinear controlled and damped oscillators, coupled via the adjacency structure of the underlying graph. The flexibility of our framework permits any basic GNN layer (e.g. convolutional or attentional) as the coupling function, from which a multi-layer deep neural network is built up via the dynamics of the proposed ODEs. We relate the oversmoothing problem, commonly encountered in GNNs, to the stability of steady states of the underlying ODE and show that zero-Dirichlet energy steady states are not stable for our proposed ODEs. This demonstrates that the proposed framework mitigates the oversmoothing problem. Moreover, we prove that GraphCON mitigates the exploding and vanishing gradients problem to facilitate training of deep multi-layer GNNs. Finally, we show that our approach offers competitive performance with respect to the state-of-the-art on a variety of graph-based learning tasks.

----

## [827] Hindering Adversarial Attacks with Implicit Neural Representations

**Authors**: *Andrei A. Rusu, Dan Andrei Calian, Sven Gowal, Raia Hadsell*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/rusu22a.html](https://proceedings.mlr.press/v162/rusu22a.html)

**Abstract**:

We introduce the Lossy Implicit Network Activation Coding (LINAC) defence, an input transformation which successfully hinders several common adversarial attacks on CIFAR-10 classifiers for perturbations up to 8/255 in Linf norm and 0.5 in L2 norm. Implicit neural representations are used to approximately encode pixel colour intensities in 2D images such that classifiers trained on transformed data appear to have robustness to small perturbations without adversarial training or large drops in performance. The seed of the random number generator used to initialise and train the implicit neural representation turns out to be necessary information for stronger generic attacks, suggesting its role as a private key. We devise a Parametric Bypass Approximation (PBA) attack strategy for key-based defences, which successfully invalidates an existing method in this category. Interestingly, our LINAC defence also hinders some transfer and adaptive attacks, including our novel PBA strategy. Our results emphasise the importance of a broad range of customised attacks despite apparent robustness according to standard evaluations.

----

## [828] Exploiting Independent Instruments: Identification and Distribution Generalization

**Authors**: *Sorawit Saengkyongam, Leonard Henckel, Niklas Pfister, Jonas Peters*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/saengkyongam22a.html](https://proceedings.mlr.press/v162/saengkyongam22a.html)

**Abstract**:

Instrumental variable models allow us to identify a causal function between covariates $X$ and a response $Y$, even in the presence of unobserved confounding. Most of the existing estimators assume that the error term in the response $Y$ and the hidden confounders are uncorrelated with the instruments $Z$. This is often motivated by a graphical separation, an argument that also justifies independence. Positing an independence restriction, however, leads to strictly stronger identifiability results. We connect to the existing literature in econometrics and provide a practical method called HSIC-X for exploiting independence that can be combined with any gradient-based learning procedure. We see that even in identifiable settings, taking into account higher moments may yield better finite sample results. Furthermore, we exploit the independence for distribution generalization. We prove that the proposed estimator is invariant to distributional shifts on the instruments and worst-case optimal whenever these shifts are sufficiently strong. These results hold even in the under-identified case where the instruments are not sufficiently rich to identify the causal function.

----

## [829] FedNL: Making Newton-Type Methods Applicable to Federated Learning

**Authors**: *Mher Safaryan, Rustem Islamov, Xun Qian, Peter Richtárik*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/safaryan22a.html](https://proceedings.mlr.press/v162/safaryan22a.html)

**Abstract**:

Inspired by recent work of Islamov et al (2021), we propose a family of Federated Newton Learn (\algname{FedNL}) methods, which we believe is a marked step in the direction of making second-order methods applicable to FL. In contrast to the aforementioned work, \algname{FedNL} employs a different Hessian learning technique which i) enhances privacy as it does not rely on the training data to be revealed to the coordinating server, ii) makes it applicable beyond generalized linear models, and iii) provably works with general contractive compression operators for compressing the local Hessians, such as Top-$K$ or Rank-$R$, which are vastly superior in practice. Notably, we do not need to rely on error feedback for our methods to work with contractive compressors. Moreover, we develop \algname{FedNL-PP}, \algname{FedNL-CR} and \algname{FedNL-LS}, which are variants of \algname{FedNL} that support partial participation, and globalization via cubic regularization and line search, respectively, and \algname{FedNL-BC}, which is a variant that can further benefit from bidirectional compression of gradients and models, i.e., smart uplink gradient and smart downlink model compression. We prove local convergence rates that are independent of the condition number, the number of training data points, and compression variance. Our communication efficient Hessian learning technique provably learns the Hessian at the optimum. Finally, we perform a variety of numerical experiments that show that our \algname{FedNL} methods have state-of-the-art communication complexity when compared to key baselines.

----

## [830] Versatile Dueling Bandits: Best-of-both World Analyses for Learning from Relative Preferences

**Authors**: *Aadirupa Saha, Pierre Gaillard*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/saha22a.html](https://proceedings.mlr.press/v162/saha22a.html)

**Abstract**:

We study the problem of $K$-armed dueling bandit for both stochastic and adversarial environments, where the goal of the learner is to aggregate information through relative preferences of pair of decision points queried in an online sequential manner. We first propose a novel reduction from any (general) dueling bandits to multi-armed bandits which allows us to improve many existing results in dueling bandits. In particular, we give the first best-of-both world result for the dueling bandits regret minimization problem—a unified framework that is guaranteed to perform optimally for both stochastic and adversarial preferences simultaneously. Moreover, our algorithm is also the first to achieve an optimal $O(\sum_{i = 1}^K \frac{\log T}{\Delta_i})$ regret bound against the Condorcet-winner benchmark, which scales optimally both in terms of the arm-size $K$ and the instance-specific suboptimality gaps $\{\Delta_i\}_{i = 1}^K$. This resolves the long standing problem of designing an instancewise gap-dependent order optimal regret algorithm for dueling bandits (with matching lower bounds up to small constant factors). We further justify the robustness of our proposed algorithm by proving its optimal regret rate under adversarially corrupted preferences—this outperforms the existing state-of-the-art corrupted dueling results by a large margin. In summary, we believe our reduction idea will find a broader scope in solving a diverse class of dueling bandits setting, which are otherwise studied separately from multi-armed bandits with often more complex solutions and worse guarantees. The efficacy of our proposed algorithms are empirically corroborated against state-of-the art dueling bandit methods.

----

## [831] Optimal and Efficient Dynamic Regret Algorithms for Non-Stationary Dueling Bandits

**Authors**: *Aadirupa Saha, Shubham Gupta*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/saha22b.html](https://proceedings.mlr.press/v162/saha22b.html)

**Abstract**:

We study the problem of dynamic regret minimization in $K$-armed Dueling Bandits under non-stationary or time-varying preferences. This is an online learning setup where the agent chooses a pair of items at each round and observes only a relative binary ‘win-loss’ feedback for this pair sampled from an underlying preference matrix at that round. We first study the problem of static-regret minimization for adversarial preference sequences and design an efficient algorithm with $O(\sqrt{KT})$ regret bound. We next use similar algorithmic ideas to propose an efficient and provably optimal algorithm for dynamic-regret minimization under two notions of non-stationarities. In particular, we show $\tO(\sqrt{SKT})$ and $\tO({V_T^{1/3}K^{1/3}T^{2/3}})$ dynamic-regret guarantees, respectively, with $S$ being the total number of ‘effective-switches’ in the underlying preference relations and $V_T$ being a measure of ‘continuous-variation’ non-stationarity. These rates are provably optimal as justified with matching lower bound guarantees. Moreover, our proposed algorithms are flexible as they can be easily ‘blackboxed’ to yield dynamic regret guarantees for other notions of dueling bandits regret, including condorcet regret, best-response bounds, and Borda regret. The complexity of these problems have not been studied prior to this work despite the practicality of non-stationary environments. Our results are corroborated with extensive simulations.

----

## [832] Unraveling Attention via Convex Duality: Analysis and Interpretations of Vision Transformers

**Authors**: *Arda Sahiner, Tolga Ergen, Batu Ozturkler, John M. Pauly, Morteza Mardani, Mert Pilanci*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sahiner22a.html](https://proceedings.mlr.press/v162/sahiner22a.html)

**Abstract**:

Vision transformers using self-attention or its proposed alternatives have demonstrated promising results in many image related tasks. However, the underpinning inductive bias of attention is not well understood. To address this issue, this paper analyzes attention through the lens of convex duality. For the non-linear dot-product self-attention, and alternative mechanisms such as MLP-mixer and Fourier Neural Operator (FNO), we derive equivalent finite-dimensional convex problems that are interpretable and solvable to global optimality. The convex programs lead to block nuclear-norm regularization that promotes low rank in the latent feature and token dimensions. In particular, we show how self-attention networks implicitly clusters the tokens, based on their latent similarity. We conduct experiments for transferring a pre-trained transformer backbone for CIFAR-100 classification by fine-tuning a variety of convex attention heads. The results indicate the merits of the bias induced by attention compared with the existing MLP or linear heads.

----

## [833] Off-Policy Evaluation for Large Action Spaces via Embeddings

**Authors**: *Yuta Saito, Thorsten Joachims*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/saito22a.html](https://proceedings.mlr.press/v162/saito22a.html)

**Abstract**:

Off-policy evaluation (OPE) in contextual bandits has seen rapid adoption in real-world systems, since it enables offline evaluation of new policies using only historic log data. Unfortunately, when the number of actions is large, existing OPE estimators – most of which are based on inverse propensity score weighting – degrade severely and can suffer from extreme bias and variance. This foils the use of OPE in many applications from recommender systems to language models. To overcome this issue, we propose a new OPE estimator that leverages marginalized importance weights when action embeddings provide structure in the action space. We characterize the bias, variance, and mean squared error of the proposed estimator and analyze the conditions under which the action embedding provides statistical benefits over conventional estimators. In addition to the theoretical analysis, we find that the empirical performance improvement can be substantial, enabling reliable OPE even when existing estimators collapse due to a large number of actions.

----

## [834] Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training

**Authors**: *Charbel Sakr, Steve Dai, Rangharajan Venkatesan, Brian Zimmer, William J. Dally, Brucek Khailany*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sakr22a.html](https://proceedings.mlr.press/v162/sakr22a.html)

**Abstract**:

Data clipping is crucial in reducing noise in quantization operations and improving the achievable accuracy of quantization-aware training (QAT). Current practices rely on heuristics to set clipping threshold scalars and cannot be shown to be optimal. We propose Optimally Clipped Tensors And Vectors (OCTAV), a recursive algorithm to determine MSE-optimal clipping scalars. Derived from the fast Newton-Raphson method, OCTAV finds optimal clipping scalars on the fly, for every tensor, at every iteration of the QAT routine. Thus, the QAT algorithm is formulated with provably minimum quantization noise at each step. In addition, we reveal limitations in common gradient estimation techniques in QAT and propose magnitude-aware differentiation as a remedy to further improve accuracy. Experimentally, OCTAV-enabled QAT achieves state-of-the-art accuracy on multiple tasks. These include training-from-scratch and retraining ResNets and MobileNets on ImageNet, and Squad fine-tuning using BERT models, where OCTAV-enabled QAT consistently preserves accuracy at low precision (4-to-6-bits). Our results require no modifications to the baseline training recipe, except for the insertion of quantization operations where appropriate.

----

## [835] A Convergence Theory for SVGD in the Population Limit under Talagrand's Inequality T1

**Authors**: *Adil Salim, Lukang Sun, Peter Richtárik*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/salim22a.html](https://proceedings.mlr.press/v162/salim22a.html)

**Abstract**:

Stein Variational Gradient Descent (SVGD) is an algorithm for sampling from a target density which is known up to a multiplicative constant. Although SVGD is a popular algorithm in practice, its theoretical study is limited to a few recent works. We study the convergence of SVGD in the population limit, (i.e., with an infinite number of particles) to sample from a non-logconcave target distribution satisfying Talagrand’s inequality T1. We first establish the convergence of the algorithm. Then, we establish a dimension-dependent complexity bound in terms of the Kernelized Stein Discrepancy (KSD). Unlike existing works, we do not assume that the KSD is bounded along the trajectory of the algorithm. Our approach relies on interpreting SVGD as a gradient descent over a space of probability measures.

----

## [836] FITNESS: (Fine Tune on New and Similar Samples) to detect anomalies in streams with drift and outliers

**Authors**: *Abishek Sankararaman, Balakrishnan Narayanaswamy, Vikramank Y. Singh, Zhao Song*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sankararaman22a.html](https://proceedings.mlr.press/v162/sankararaman22a.html)

**Abstract**:

Technology improvements have made it easier than ever to collect diverse telemetry at high resolution from any cyber or physical system, for both monitoring and control. In the domain of monitoring, anomaly detection has become an important problem in many research areas ranging from IoT and sensor networks to devOps. These systems operate in real, noisy and non-stationary environments. A fundamental question is then, ‘How to quickly spot anomalies in a data-stream, and differentiate them from either sudden or gradual drifts in the normal behaviour?’ Although several heuristics have been proposed for detecting anomalies on streams, no known method has formalized the desiderata and rigorously proven that they can be achieved. We begin by formalizing the problem as a sequential estimation task. We propose \name, (\textbf{Fi}ne \textbf{T}une on \textbf{Ne}w and \textbf{S}imilar \textbf{S}amples), a flexible framework for detecting anomalies on data streams. We show that in the case when the data stream has a gaussian distribution, FITNESS is provably both robust and adaptive. The core of our method is to fine-tune the anomaly detection system only on recent, similar examples, before predicting an anomaly score. We prove that this is sufficient for robustness and adaptivity. We further experimentally demonstrate that \name;{is} flexible in practice, i.e., it can convert existing offline AD algorithms in to robust and adaptive online ones.

----

## [837] The Algebraic Path Problem for Graph Metrics

**Authors**: *Enrique Fita Sanmartín, Sebastian Damrich, Fred A. Hamprecht*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sanmarti-n22a.html](https://proceedings.mlr.press/v162/sanmarti-n22a.html)

**Abstract**:

Finding paths with optimal properties is a foundational problem in computer science. The notions of shortest paths (minimal sum of edge costs), minimax paths (minimal maximum edge weight), reliability of a path and many others all arise as special cases of the "algebraic path problem" (APP). Indeed, the APP formalizes the relation between different semirings such as min-plus, min-max and the distances they induce. We here clarify, for the first time, the relation between the potential distance and the log-semiring. We also define a new unifying family of algebraic structures that include all above-mentioned path problems as well as the commute cost and others as special or limiting cases. The family comprises not only semirings but also strong bimonoids (that is, semirings without distributivity). We call this new and very general distance the "log-norm distance". Finally, we derive some sufficient conditions which ensure that the APP associated with a semiring defines a metric over an arbitrary graph.

----

## [838] LSB: Local Self-Balancing MCMC in Discrete Spaces

**Authors**: *Emanuele Sansone*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sansone22a.html](https://proceedings.mlr.press/v162/sansone22a.html)

**Abstract**:

We present the Local Self-Balancing sampler (LSB), a local Markov Chain Monte Carlo (MCMC) method for sampling in purely discrete domains, which is able to autonomously adapt to the target distribution and to reduce the number of target evaluations required to converge. LSB is based on (i) a parametrization of locally balanced proposals, (ii) an objective function based on mutual information and (iii) a self-balancing learning procedure, which minimises the proposed objective to update the proposal parameters. Experiments on energy-based models and Markov networks show that LSB converges using a smaller number of queries to the oracle distribution compared to recent local MCMC samplers.

----

## [839] PoF: Post-Training of Feature Extractor for Improving Generalization

**Authors**: *Ikuro Sato, Ryota Yamada, Masayuki Tanaka, Nakamasa Inoue, Rei Kawakami*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sato22a.html](https://proceedings.mlr.press/v162/sato22a.html)

**Abstract**:

It has been intensively investigated that the local shape, especially flatness, of the loss landscape near a minimum plays an important role for generalization of deep models. We developed a training algorithm called PoF: Post-Training of Feature Extractor that updates the feature extractor part of an already-trained deep model to search a flatter minimum. The characteristics are two-fold: 1) Feature extractor is trained under parameter perturbations in the higher-layer parameter space, based on observations that suggest flattening higher-layer parameter space, and 2) the perturbation range is determined in a data-driven manner aiming to reduce a part of test loss caused by the positive loss curvature. We provide a theoretical analysis that shows the proposed algorithm implicitly reduces the target Hessian components as well as the loss. Experimental results show that PoF improved model performance against baseline methods on both CIFAR-10 and CIFAR-100 datasets for only 10-epoch post-training, and on SVHN dataset for 50-epoch post-training.

----

## [840] Re-evaluating Word Mover's Distance

**Authors**: *Ryoma Sato, Makoto Yamada, Hisashi Kashima*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sato22b.html](https://proceedings.mlr.press/v162/sato22b.html)

**Abstract**:

The word mover’s distance (WMD) is a fundamental technique for measuring the similarity of two documents. As the crux of WMD, it can take advantage of the underlying geometry of the word space by employing an optimal transport formulation. The original study on WMD reported that WMD outperforms classical baselines such as bag-of-words (BOW) and TF-IDF by significant margins in various datasets. In this paper, we point out that the evaluation in the original study could be misleading. We re-evaluate the performances of WMD and the classical baselines and find that the classical baselines are competitive with WMD if we employ an appropriate preprocessing, i.e., L1 normalization. In addition, we introduce an analogy between WMD and L1-normalized BOW and find that not only the performance of WMD but also the distance values resemble those of BOW in high dimensional spaces.

----

## [841] Understanding Contrastive Learning Requires Incorporating Inductive Biases

**Authors**: *Nikunj Saunshi, Jordan T. Ash, Surbhi Goel, Dipendra Misra, Cyril Zhang, Sanjeev Arora, Sham M. Kakade, Akshay Krishnamurthy*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/saunshi22a.html](https://proceedings.mlr.press/v162/saunshi22a.html)

**Abstract**:

Contrastive learning is a popular form of self-supervised learning that encourages augmentations (views) of the same input to have more similar representations compared to augmentations of different inputs. Recent attempts to theoretically explain the success of contrastive learning on downstream classification tasks prove guarantees depending on properties of augmentations and the value of contrastive loss of representations. We demonstrate that such analyses, that ignore inductive biases of the function class and training algorithm, cannot adequately explain the success of contrastive learning, even provably leading to vacuous guarantees in some settings. Extensive experiments on image and text domains highlight the ubiquity of this problem – different function classes and algorithms behave very differently on downstream tasks, despite having the same augmentations and contrastive losses. Theoretical analysis is presented for the class of linear representations, where incorporating inductive biases of the function class allows contrastive learning to work with less stringent conditions compared to prior analyses.

----

## [842] The Neural Race Reduction: Dynamics of Abstraction in Gated Networks

**Authors**: *Andrew M. Saxe, Shagun Sodhani, Sam Jay Lewallen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/saxe22a.html](https://proceedings.mlr.press/v162/saxe22a.html)

**Abstract**:

Our theoretical understanding of deep learning has not kept pace with its empirical success. While network architecture is known to be critical, we do not yet understand its effect on learned representations and network behavior, or how this architecture should reflect task structure.In this work, we begin to address this gap by introducing the Gated Deep Linear Network framework that schematizes how pathways of information flow impact learning dynamics within an architecture. Crucially, because of the gating, these networks can compute nonlinear functions of their input. We derive an exact reduction and, for certain cases, exact solutions to the dynamics of learning. Our analysis demonstrates that the learning dynamics in structured networks can be conceptualized as a neural race with an implicit bias towards shared representations, which then govern the model’s ability to systematically generalize, multi-task, and transfer. We validate our key insights on naturalistic datasets and with relaxed assumptions. Taken together, our work gives rise to general hypotheses relating neural architecture to learning and provides a mathematical approach towards understanding the design of more complex architectures and the role of modularity and compositionality in solving real-world problems. The code and results are available at https://www.saxelab.org/gated-dln.

----

## [843] Convergence Rates of Non-Convex Stochastic Gradient Descent Under a Generic Lojasiewicz Condition and Local Smoothness

**Authors**: *Kevin Scaman, Cédric Malherbe, Ludovic Dos Santos*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/scaman22a.html](https://proceedings.mlr.press/v162/scaman22a.html)

**Abstract**:

Training over-parameterized neural networks involves the empirical minimization of highly non-convex objective functions. Recently, a large body of works provided theoretical evidence that, despite this non-convexity, properly initialized over-parameterized networks can converge to a zero training loss through the introduction of the Polyak-Lojasiewicz condition. However, these analyses are restricted to quadratic losses such as mean square error, and tend to indicate fast exponential convergence rates that are seldom observed in practice. In this work, we propose to extend these results by analyzing stochastic gradient descent under more generic Lojasiewicz conditions that are applicable to any convex loss function, thus extending the current theory to a larger panel of losses commonly used in practice such as cross-entropy. Moreover, our analysis provides high-probability bounds on the approximation error under sub-Gaussian gradient noise and only requires the local smoothness of the objective function, thus making it applicable to deep neural networks in realistic settings.

----

## [844] An Asymptotic Test for Conditional Independence using Analytic Kernel Embeddings

**Authors**: *Meyer Scetbon, Laurent Meunier, Yaniv Romano*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/scetbon22a.html](https://proceedings.mlr.press/v162/scetbon22a.html)

**Abstract**:

We propose a new conditional dependence measure and a statistical test for conditional independence. The measure is based on the difference between analytic kernel embeddings of two well-suited distributions evaluated at a finite set of locations. We obtain its asymptotic distribution under the null hypothesis of conditional independence and design a consistent statistical test from it. We conduct a series of experiments showing that our new test outperforms state-of-the-art methods both in terms of type-I and type-II errors even in the high dimensional setting.

----

## [845] Linear-Time Gromov Wasserstein Distances using Low Rank Couplings and Costs

**Authors**: *Meyer Scetbon, Gabriel Peyré, Marco Cuturi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/scetbon22b.html](https://proceedings.mlr.press/v162/scetbon22b.html)

**Abstract**:

The ability to align points across two related yet incomparable point clouds (e.g. living in different spaces) plays an important role in machine learning. The Gromov-Wasserstein (GW) framework provides an increasingly popular answer to such problems, by seeking a low-distortion, geometry-preserving assignment between these points. As a non-convex, quadratic generalization of optimal transport (OT), GW is NP-hard. While practitioners often resort to solving GW approximately as a nested sequence of entropy-regularized OT problems, the cubic complexity (in the number $n$ of samples) of that approach is a roadblock. We show in this work how a recent variant of the OT problem that restricts the set of admissible couplings to those having a low-rank factorization is remarkably well suited to the resolution of GW: when applied to GW, we show that this approach is not only able to compute a stationary point of the GW problem in time $O(n^2)$, but also uniquely positioned to benefit from the knowledge that the initial cost matrices are low-rank, to yield a linear time $O(n)$ GW approximation. Our approach yields similar results, yet orders of magnitude faster computation than the SoTA entropic GW approaches, on both simulated and real data.

----

## [846] Streaming Inference for Infinite Feature Models

**Authors**: *Rylan Schaeffer, Yilun Du, Gabrielle K. Liu, Ila Fiete*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/schaeffer22a.html](https://proceedings.mlr.press/v162/schaeffer22a.html)

**Abstract**:

Unsupervised learning from a continuous stream of data is arguably one of the most common and most challenging problems facing intelligent agents. One class of unsupervised models, collectively termed feature models, attempts unsupervised discovery of latent features underlying the data and includes common models such as PCA, ICA, and NMF. However, if the data arrives in a continuous stream, determining the number of features is a significant challenge and the number may grow with time. In this work, we make feature models significantly more applicable to streaming data by imbuing them with the ability to create new features, online, in a probabilistic and principled manner. To achieve this, we derive a novel recursive form of the Indian Buffet Process, which we term the Recursive IBP (R-IBP). We demonstrate that R-IBP can be be used as a prior for feature models to efficiently infer a posterior over an unbounded number of latent features, with quasilinear average time complexity and logarithmic average space complexity. We compare R-IBP to existing offline sampling and variational baselines in two feature models (Linear Gaussian and Factor Analysis) and demonstrate on synthetic and real data that R-IBP achieves comparable or better performance in significantly less time.

----

## [847] Modeling Irregular Time Series with Continuous Recurrent Units

**Authors**: *Mona Schirmer, Mazin Eltayeb, Stefan Lessmann, Maja Rudolph*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/schirmer22a.html](https://proceedings.mlr.press/v162/schirmer22a.html)

**Abstract**:

Recurrent neural networks (RNNs) are a popular choice for modeling sequential data. Modern RNN architectures assume constant time-intervals between observations. However, in many datasets (e.g. medical records) observation times are irregular and can carry important information. To address this challenge, we propose continuous recurrent units (CRUs) {–} a neural architecture that can naturally handle irregular intervals between observations. The CRU assumes a hidden state, which evolves according to a linear stochastic differential equation and is integrated into an encoder-decoder framework. The recursive computations of the CRU can be derived using the continuous-discrete Kalman filter and are in closed form. The resulting recurrent architecture has temporal continuity between hidden states and a gating mechanism that can optimally integrate noisy observations. We derive an efficient parameterization scheme for the CRU that leads to a fast implementation f-CRU. We empirically study the CRU on a number of challenging datasets and find that it can interpolate irregular time series better than methods based on neural ordinary differential equations.

----

## [848] Structure Preserving Neural Networks: A Case Study in the Entropy Closure of the Boltzmann Equation

**Authors**: *Steffen Schotthöfer, Tianbai Xiao, Martin Frank, Cory D. Hauck*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/schotthofer22a.html](https://proceedings.mlr.press/v162/schotthofer22a.html)

**Abstract**:

In this paper, we explore applications of deep learning in statistical physics. We choose the Boltzmann equation as a typical example, where neural networks serve as a closure to its moment system. We present two types of neural networks to embed the convexity of entropy and to preserve the minimum entropy principle and intrinsic mathematical structures of the moment system of the Boltzmann equation. We derive an error bound for the generalization gap of convex neural networks which are trained in Sobolev norm and use the results to construct data sampling methods for neural network training. Numerical experiments demonstrate that the neural entropy closure is significantly faster than classical optimizers while maintaining sufficient accuracy.

----

## [849] Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification

**Authors**: *Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Bjoern M. Eskofier, Dario Zanca*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/schwinn22a.html](https://proceedings.mlr.press/v162/schwinn22a.html)

**Abstract**:

The reliability of neural networks is essential for their use in safety-critical applications. Existing approaches generally aim at improving the robustness of neural networks to either real-world distribution shifts (e.g., common corruptions and perturbations, spatial transformations, and natural adversarial examples) or worst-case distribution shifts (e.g., optimized adversarial examples). In this work, we propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface. Furthermore, we propose an implementation using targeted and untargeted adversarial attacks. An extensive empirical evaluation shows that DRQ increases the robustness of adversarially and non-adversarially trained models against real-world and worst-case distribution shifts on several computer vision benchmark datasets.

----

## [850] Symmetric Machine Theory of Mind

**Authors**: *Melanie Sclar, Graham Neubig, Yonatan Bisk*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sclar22a.html](https://proceedings.mlr.press/v162/sclar22a.html)

**Abstract**:

Theory of mind, the ability to model others’ thoughts and desires, is a cornerstone of human social intelligence. This makes it an important challenge for the machine learning community, but previous works mainly attempt to design agents that model the "mental state" of others as passive observers or in specific predefined roles, such as in speaker-listener scenarios. In contrast, we propose to model machine theory of mind in a more general symmetric scenario. We introduce a multi-agent environment SymmToM where, like in real life, all agents can speak, listen, see other agents, and move freely through the world. Effective strategies to maximize an agent’s reward require it to develop a theory of mind. We show that reinforcement learning agents that model the mental states of others achieve significant performance improvements over agents with no such theory of mind model. Importantly, our best agents still fail to achieve performance comparable to agents with access to the gold-standard mental state of other agents, demonstrating that the modeling of theory of mind in multi-agent scenarios is very much an open challenge.

----

## [851] Data-SUITE: Data-centric identification of in-distribution incongruous examples

**Authors**: *Nabeel Seedat, Jonathan Crabbé, Mihaela van der Schaar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/seedat22a.html](https://proceedings.mlr.press/v162/seedat22a.html)

**Abstract**:

Systematic quantification of data quality is critical for consistent model performance. Prior works have focused on out-of-distribution data. Instead, we tackle an understudied yet equally important problem of characterizing incongruous regions of in-distribution (ID) data, which may arise from feature space heterogeneity. To this end, we propose a paradigm shift with Data-SUITE: a data-centric AI framework to identify these regions, independent of a task-specific model. Data-SUITE leverages copula modeling, representation learning, and conformal prediction to build feature-wise confidence interval estimators based on a set of training instances. These estimators can be used to evaluate the congruence of test instances with respect to the training set, to answer two practically useful questions: (1) which test instances will be reliably predicted by a model trained with the training instances? and (2) can we identify incongruous regions of the feature space so that data owners understand the data’s limitations or guide future data collection? We empirically validate Data-SUITE’s performance and coverage guarantees and demonstrate on cross-site medical data, biased data, and data with concept drift, that Data-SUITE best identifies ID regions where a downstream model may be reliable (independent of said model). We also illustrate how these identified regions can provide insights into datasets and highlight their limitations.

----

## [852] Continuous-Time Modeling of Counterfactual Outcomes Using Neural Controlled Differential Equations

**Authors**: *Nabeel Seedat, Fergus Imrie, Alexis Bellot, Zhaozhi Qian, Mihaela van der Schaar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/seedat22b.html](https://proceedings.mlr.press/v162/seedat22b.html)

**Abstract**:

Estimating counterfactual outcomes over time has the potential to unlock personalized healthcare by assisting decision-makers to answer "what-if" questions. Existing causal inference approaches typically consider regular, discrete-time intervals between observations and treatment decisions and hence are unable to naturally model irregularly sampled data, which is the common setting in practice. To handle arbitrary observation patterns, we interpret the data as samples from an underlying continuous-time process and propose to model its latent trajectory explicitly using the mathematics of controlled differential equations. This leads to a new approach, the Treatment Effect Neural Controlled Differential Equation (TE-CDE), that allows the potential outcomes to be evaluated at any time point. In addition, adversarial training is used to adjust for time-dependent confounding which is critical in longitudinal settings and is an added challenge not encountered in conventional time series. To assess solutions to this problem, we propose a controllable simulation environment based on a model of tumor growth for a range of scenarios with irregular sampling reflective of a variety of clinical scenarios. TE-CDE consistently outperforms existing approaches in all scenarios with irregular sampling.

----

## [853] Neural Tangent Kernel Beyond the Infinite-Width Limit: Effects of Depth and Initialization

**Authors**: *Mariia Seleznova, Gitta Kutyniok*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/seleznova22a.html](https://proceedings.mlr.press/v162/seleznova22a.html)

**Abstract**:

Neural Tangent Kernel (NTK) is widely used to analyze overparametrized neural networks due to the famous result by Jacot et al. (2018): in the infinite-width limit, the NTK is deterministic and constant during training. However, this result cannot explain the behavior of deep networks, since it generally does not hold if depth and width tend to infinity simultaneously. In this paper, we study the NTK of fully-connected ReLU networks with depth comparable to width. We prove that the NTK properties depend significantly on the depth-to-width ratio and the distribution of parameters at initialization. In fact, our results indicate the importance of the three phases in the hyperparameter space identified in Poole et al. (2016): ordered, chaotic and the edge of chaos (EOC). We derive exact expressions for the NTK dispersion in the infinite-depth-and-width limit in all three phases and conclude that the NTK variability grows exponentially with depth at the EOC and in the chaotic phase but not in the ordered phase. We also show that the NTK of deep networks may stay constant during training only in the ordered phase and discuss how the structure of the NTK matrix changes during training.

----

## [854] Reinforcement Learning with Action-Free Pre-Training from Videos

**Authors**: *Younggyo Seo, Kimin Lee, Stephen L. James, Pieter Abbeel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/seo22a.html](https://proceedings.mlr.press/v162/seo22a.html)

**Abstract**:

Recent unsupervised pre-training methods have shown to be effective on language and vision domains by learning useful representations for multiple downstream tasks. In this paper, we investigate if such unsupervised pre-training methods can also be effective for vision-based reinforcement learning (RL). To this end, we introduce a framework that learns representations useful for understanding the dynamics via generative pre-training on videos. Our framework consists of two phases: we pre-train an action-free latent video prediction model, and then utilize the pre-trained representations for efficiently learning action-conditional world models on unseen environments. To incorporate additional action inputs during fine-tuning, we introduce a new architecture that stacks an action-conditional latent prediction model on top of the pre-trained action-free prediction model. Moreover, for better exploration, we propose a video-based intrinsic bonus that leverages pre-trained representations. We demonstrate that our framework significantly improves both final performances and sample-efficiency of vision-based RL in a variety of manipulation and locomotion tasks. Code is available at \url{https://github.com/younggyoseo/apv}.

----

## [855] Efficient Model-based Multi-agent Reinforcement Learning via Optimistic Equilibrium Computation

**Authors**: *Pier Giuseppe Sessa, Maryam Kamgarpour, Andreas Krause*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sessa22a.html](https://proceedings.mlr.press/v162/sessa22a.html)

**Abstract**:

We consider model-based multi-agent reinforcement learning, where the environment transition model is unknown and can only be learned via expensive interactions with the environment. We propose H-MARL (Hallucinated Multi-Agent Reinforcement Learning), a novel sample-efficient algorithm that can efficiently balance exploration, i.e., learning about the environment, and exploitation, i.e., achieve good equilibrium performance in the underlying general-sum Markov game. H-MARL builds high-probability confidence intervals around the unknown transition model and sequentially updates them based on newly observed data. Using these, it constructs an optimistic hallucinated game for the agents for which equilibrium policies are computed at each round. We consider general statistical models (e.g., Gaussian processes, deep ensembles, etc.) and policy classes (e.g., deep neural networks), and theoretically analyze our approach by bounding the agents’ dynamic regret. Moreover, we provide a convergence rate to the equilibria of the underlying Markov game. We demonstrate our approach experimentally on an autonomous driving simulation benchmark. H-MARL learns successful equilibrium policies after a few interactions with the environment and can significantly improve the performance compared to non-optimistic exploration methods.

----

## [856] Selective Regression under Fairness Criteria

**Authors**: *Abhin Shah, Yuheng Bu, Joshua K. Lee, Subhro Das, Rameswar Panda, Prasanna Sattigeri, Gregory W. Wornell*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shah22a.html](https://proceedings.mlr.press/v162/shah22a.html)

**Abstract**:

Selective regression allows abstention from prediction if the confidence to make an accurate prediction is not sufficient. In general, by allowing a reject option, one expects the performance of a regression model to increase at the cost of reducing coverage (i.e., by predicting on fewer samples). However, as we show, in some cases, the performance of a minority subgroup can decrease while we reduce the coverage, and thus selective regression can magnify disparities between different sensitive subgroups. Motivated by these disparities, we propose new fairness criteria for selective regression requiring the performance of every subgroup to improve with a decrease in coverage. We prove that if a feature representation satisfies the sufficiency criterion or is calibrated for mean and variance, then the proposed fairness criteria is met. Further, we introduce two approaches to mitigate the performance disparity across subgroups: (a) by regularizing an upper bound of conditional mutual information under a Gaussian assumption and (b) by regularizing a contrastive loss for conditional mean and conditional variance prediction. The effectiveness of these approaches is demonstrated on synthetic and real-world datasets.

----

## [857] Utility Theory for Sequential Decision Making

**Authors**: *Mehran Shakerinava, Siamak Ravanbakhsh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shakerinava22a.html](https://proceedings.mlr.press/v162/shakerinava22a.html)

**Abstract**:

The von Neumann-Morgenstern (VNM) utility theorem shows that under certain axioms of rationality, decision-making is reduced to maximizing the expectation of some utility function. We extend these axioms to increasingly structured sequential decision making settings and identify the structure of the corresponding utility functions. In particular, we show that memoryless preferences lead to a utility in the form of a per transition reward and multiplicative factor on the future return. This result motivates a generalization of Markov Decision Processes (MDPs) with this structure on the agent’s returns, which we call Affine-Reward MDPs. A stronger constraint on preferences is needed to recover the commonly used cumulative sum of scalar rewards in MDPs. A yet stronger constraint simplifies the utility function for goal-seeking agents in the form of a difference in some function of states that we call potential functions. Our necessary and sufficient conditions demystify the reward hypothesis that underlies the design of rational agents in reinforcement learning by adding an axiom to the VNM rationality axioms and motivates new directions for AI research involving sequential decision making.

----

## [858] Translating Robot Skills: Learning Unsupervised Skill Correspondences Across Robots

**Authors**: *Tanmay Shankar, Yixin Lin, Aravind Rajeswaran, Vikash Kumar, Stuart Anderson, Jean Oh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shankar22a.html](https://proceedings.mlr.press/v162/shankar22a.html)

**Abstract**:

In this paper, we explore how we can endow robots with the ability to learn correspondences between their own skills, and those of morphologically different robots in different domains, in an entirely unsupervised manner. We make the insight that different morphological robots use similar task strategies to solve similar tasks. Based on this insight, we frame learning skill correspondences as a problem of matching distributions of sequences of skills across robots. We then present an unsupervised objective that encourages a learnt skill translation model to match these distributions across domains, inspired by recent advances in unsupervised machine translation. Our approach is able to learn semantically meaningful correspondences between skills across multiple robot-robot and human-robot domain pairs despite being completely unsupervised. Further, the learnt correspondences enable the transfer of task strategies across robots and domains. We present dynamic visualizations of our results at https://sites.google.com/view/translatingrobotskills/home.

----

## [859] A State-Distribution Matching Approach to Non-Episodic Reinforcement Learning

**Authors**: *Archit Sharma, Rehaan Ahmad, Chelsea Finn*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sharma22a.html](https://proceedings.mlr.press/v162/sharma22a.html)

**Abstract**:

While reinforcement learning (RL) provides a framework for learning through trial and error, translating RL algorithms into the real world has remained challenging. A major hurdle to real-world application arises from the development of algorithms in an episodic setting where the environment is reset after every trial, in contrast with the continual and non-episodic nature of the real-world encountered by embodied agents such as humans and robots. Enabling agents to learn behaviors autonomously in such non-episodic environments requires that the agent to be able to conduct its own trials. Prior works have considered an alternating approach where a forward policy learns to solve the task and the backward policy learns to reset the environment, but what initial state distribution should the backward policy reset the agent to? Assuming access to a few demonstrations, we propose a new method, MEDAL, that trains the backward policy to match the state distribution in the provided demonstrations. This keeps the agent close to the task-relevant states, allowing for a mix of easy and difficult starting states for the forward policy. Our experiments show that MEDAL matches or outperforms prior methods on three sparse-reward continuous control tasks from the EARL benchmark, with 40% gains on the hardest task, while making fewer assumptions than prior works.

----

## [860] Content Addressable Memory Without Catastrophic Forgetting by Heteroassociation with a Fixed Scaffold

**Authors**: *Sugandha Sharma, Sarthak Chandra, Ila R. Fiete*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sharma22b.html](https://proceedings.mlr.press/v162/sharma22b.html)

**Abstract**:

Content-addressable memory (CAM) networks, so-called because stored items can be recalled by partial or corrupted versions of the items, exhibit near-perfect recall of a small number of information-dense patterns below capacity and a ’memory cliff’ beyond, such that inserting a single additional pattern results in catastrophic loss of all stored patterns. We propose a novel CAM architecture, Memory Scaffold with Heteroassociation (MESH), that factorizes the problems of internal attractor dynamics and association with external content to generate a CAM continuum without a memory cliff: Small numbers of patterns are stored with complete information recovery matching standard CAMs, while inserting more patterns still results in partial recall of every pattern, with a graceful trade-off between pattern number and pattern richness. Motivated by the architecture of the Entorhinal-Hippocampal memory circuit in the brain, MESH is a tripartite architecture with pairwise interactions that uses a predetermined set of internally stabilized states together with heteroassociation between the internal states and arbitrary external patterns. We show analytically and experimentally that for any number of stored patterns, MESH nearly saturates the total information bound (given by the number of synapses) for CAM networks, outperforming all existing CAM models.

----

## [861] Federated Minimax Optimization: Improved Convergence Analyses and Algorithms

**Authors**: *Pranay Sharma, Rohan Panda, Gauri Joshi, Pramod K. Varshney*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sharma22c.html](https://proceedings.mlr.press/v162/sharma22c.html)

**Abstract**:

In this paper, we consider nonconvex minimax optimization, which is gaining prominence in many modern machine learning applications, such as GANs. Large-scale edge-based collection of training data in these applications calls for communication-efficient distributed optimization algorithms, such as those used in federated learning, to process the data. In this paper, we analyze local stochastic gradient descent ascent (SGDA), the local-update version of the SGDA algorithm. SGDA is the core algorithm used in minimax optimization, but it is not well-understood in a distributed setting. We prove that Local SGDA has order-optimal sample complexity for several classes of nonconvex-concave and nonconvex-nonconcave minimax problems, and also enjoys linear speedup with respect to the number of clients. We provide a novel and tighter analysis, which improves the convergence and communication guarantees in the existing literature. For nonconvex-PL and nonconvex-one-point-concave functions, we improve the existing complexity results for centralized minimax problems. Furthermore, we propose a momentum-based local-update algorithm, which has the same convergence guarantees, but outperforms Local SGDA as demonstrated in our experiments.

----

## [862] DNS: Determinantal Point Process Based Neural Network Sampler for Ensemble Reinforcement Learning

**Authors**: *Hassam Sheikh, Kizza Frisbee, Mariano Phielipp*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sheikh22a.html](https://proceedings.mlr.press/v162/sheikh22a.html)

**Abstract**:

The application of an ensemble of neural networks is becoming an imminent tool for advancing state-of-the-art deep reinforcement learning algorithms. However, training these large numbers of neural networks in the ensemble has an exceedingly high computation cost which may become a hindrance in training large-scale systems. In this paper, we propose DNS: a Determinantal Point Process based Neural Network Sampler that specifically uses k-DPP to sample a subset of neural networks for backpropagation at every training step thus significantly reducing the training time and computation cost. We integrated DNS in REDQ for continuous control tasks and evaluated on MuJoCo environments. Our experiments show that DNS augmented REDQ matches the baseline REDQ in terms of average cumulative reward and achieves this using less than 50% computation when measured in FLOPS. The code is available at https://github.com/IntelLabs/DNS

----

## [863] Instance Dependent Regret Analysis of Kernelized Bandits

**Authors**: *Shubhanshu Shekhar, Tara Javidi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shekhar22a.html](https://proceedings.mlr.press/v162/shekhar22a.html)

**Abstract**:

We study the problem of designing an adaptive strategy for querying a noisy zeroth-order-oracle to efficiently learn about the optimizer of an unknown function $f$. To make the problem tractable, we assume that $f$ lies in the reproducing kernel Hilbert space (RKHS) associated with a known kernel $K$, with its norm bounded by $M<\infty$. Prior results, working in a minimax framework, have characterized the worst-case (over all functions in the problem class) limits on regret achievable by any algorithm, and have constructed algorithms with matching (modulo polylogarithmic factors) worst-case performance for the Matern family of kernels. 	These results suffer from two drawbacks. First, the minimax lower bound gives limited information about the limits of regret achievable by commonly used algorithms on a specific problem instance $f$. Second, the existing upper bound analysis fails to adapt to easier problem instances within the function class. Our work takes steps to address both these issues. First, we derive instance-dependent regret lower bounds for algorithms with uniformly (over the function class) vanishing normalized cumulative regret. Our result, valid for several practically relevant kernelized bandits algorithms, such as, GP-UCB, GP-TS and SupKernelUCB, identifies a fundamental complexity measure associated with every problem instance. We then address the second issue, by proposing a new minimax near-optimal algorithm that also adapts to easier problem instances.

----

## [864] Data Augmentation as Feature Manipulation

**Authors**: *Ruoqi Shen, Sébastien Bubeck, Suriya Gunasekar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22a.html](https://proceedings.mlr.press/v162/shen22a.html)

**Abstract**:

Data augmentation is a cornerstone of the machine learning pipeline, yet its theoretical underpinnings remain unclear. Is it merely a way to artificially augment the data set size? Or is it about encouraging the model to satisfy certain invariances? In this work we consider another angle, and we study the effect of data augmentation on the dynamic of the learning process. We find that data augmentation can alter the relative importance of various features, effectively making certain informative but hard to learn features more likely to be captured in the learning process. Importantly, we show that this effect is more pronounced for non-linear models, such as neural networks. Our main contribution is a detailed analysis of data augmentation on the learning dynamic for a two layer convolutional neural network in the recently proposed multi-view model by Z. Allen-Zhu and Y. Li. We complement this analysis with further experimental evidence that data augmentation can be viewed as a form of feature manipulation.

----

## [865] Metric-Fair Active Learning

**Authors**: *Jie Shen, Nan Cui, Jing Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22b.html](https://proceedings.mlr.press/v162/shen22b.html)

**Abstract**:

Active learning has become a prevalent technique for designing label-efficient algorithms, where the central principle is to only query and fit “informative” labeled instances. It is, however, known that an active learning algorithm may incur unfairness due to such instance selection procedure. In this paper, we henceforth study metric-fair active learning of homogeneous halfspaces, and show that under the distribution-dependent PAC learning model, fairness and label efficiency can be achieved simultaneously. We further propose two extensions of our main results: 1) we show that it is possible to make the algorithm robust to the adversarial noise – one of the most challenging noise models in learning theory; and 2) it is possible to significantly improve the label complexity when the underlying halfspace is sparse.

----

## [866] PDO-s3DCNNs: Partial Differential Operator Based Steerable 3D CNNs

**Authors**: *Zhengyang Shen, Tao Hong, Qi She, Jinwen Ma, Zhouchen Lin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22c.html](https://proceedings.mlr.press/v162/shen22c.html)

**Abstract**:

Steerable models can provide very general and flexible equivariance by formulating equivariance requirements in the language of representation theory and feature fields, which has been recognized to be effective for many vision tasks. However, deriving steerable models for 3D rotations is much more difficult than that in the 2D case, due to more complicated mathematics of 3D rotations. In this work, we employ partial differential operators (PDOs) to model 3D filters, and derive general steerable 3D CNNs, which are called PDO-s3DCNNs. We prove that the equivariant filters are subject to linear constraints, which can be solved efficiently under various conditions. As far as we know, PDO-s3DCNNs are the most general steerable CNNs for 3D rotations, in the sense that they cover all common subgroups of SO(3) and their representations, while existing methods can only be applied to specific groups and representations. Extensive experiments show that our models can preserve equivariance well in the discrete domain, and outperform previous works on SHREC’17 retrieval and ISBI 2012 segmentation tasks with a low network complexity.

----

## [867] Connect, Not Collapse: Explaining Contrastive Learning for Unsupervised Domain Adaptation

**Authors**: *Kendrick Shen, Robbie M. Jones, Ananya Kumar, Sang Michael Xie, Jeff Z. HaoChen, Tengyu Ma, Percy Liang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22d.html](https://proceedings.mlr.press/v162/shen22d.html)

**Abstract**:

We consider unsupervised domain adaptation (UDA), where labeled data from a source domain (e.g., photos) and unlabeled data from a target domain (e.g., sketches) are used to learn a classifier for the target domain. Conventional UDA methods (e.g., domain adversarial training) learn domain-invariant features to generalize from the source domain to the target domain. In this paper, we show that contrastive pre-training, which learns features on unlabeled source and target data and then fine-tunes on labeled source data, is competitive with strong UDA methods. However, we find that contrastive pre-training does not learn domain-invariant features, diverging from conventional UDA intuitions. We show theoretically that contrastive pre-training can learn features that vary subtantially across domains but still generalize to the target domain, by disentangling domain and class information. We empirically validate our theory on benchmark vision datasets.

----

## [868] Constrained Optimization with Dynamic Bound-scaling for Effective NLP Backdoor Defense

**Authors**: *Guangyu Shen, Yingqi Liu, Guanhong Tao, Qiuling Xu, Zhuo Zhang, Shengwei An, Shiqing Ma, Xiangyu Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22e.html](https://proceedings.mlr.press/v162/shen22e.html)

**Abstract**:

Modern language models are vulnerable to backdoor attacks. An injected malicious token sequence (i.e., a trigger) can cause the compromised model to misbehave, raising security concerns. Trigger inversion is a widely-used technique for scanning backdoors in vision models. It can- not be directly applied to NLP models due to their discrete nature. In this paper, we develop a novel optimization method for NLP backdoor inversion. We leverage a dynamically reducing temperature coefficient in the softmax function to provide changing loss landscapes to the optimizer such that the process gradually focuses on the ground truth trigger, which is denoted as a one-hot value in a convex hull. Our method also features a temperature rollback mechanism to step away from local optimals, exploiting the observation that local optimals can be easily determined in NLP trigger inversion (while not in general optimization). We evaluate the technique on over 1600 models (with roughly half of them having injected backdoors) on 3 prevailing NLP tasks, with 4 different backdoor attacks and 7 architectures. Our results show that the technique is able to effectively and efficiently detect and remove backdoors, outperforming 5 baseline methods. The code is available at https: //github.com/PurduePAML/DBS.

----

## [869] Staged Training for Transformer Language Models

**Authors**: *Sheng Shen, Pete Walsh, Kurt Keutzer, Jesse Dodge, Matthew E. Peters, Iz Beltagy*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22f.html](https://proceedings.mlr.press/v162/shen22f.html)

**Abstract**:

The current standard approach to scaling transformer language models trains each model size from a different random initialization. As an alternative, we consider a staged training setup that begins with a small model and incrementally increases the amount of compute used for training by applying a "growth operator" to increase the model depth and width. By initializing each stage with the output of the previous one, the training process effectively re-uses the compute from prior stages and becomes more efficient. Our growth operators each take as input the entire training state (including model parameters, optimizer state, learning rate schedule, etc.) and output a new training state from which training continues. We identify two important properties of these growth operators, namely that they preserve both the loss and the “training dynamics” after applying the operator. While the loss-preserving property has been discussed previously, to the best of our knowledge this work is the first to identify the importance of preserving the training dynamics (the rate of decrease of the loss during training). To find the optimal schedule for stages, we use the scaling laws from (Kaplan et al., 2020) to find a precise schedule that gives the most compute saving by starting a new stage when training efficiency starts decreasing. We empirically validate our growth operators and staged training for autoregressive language models, showing up to 22% compute savings compared to a strong baseline trained from scratch. Our code is available at https://github.com/allenai/staged-training.

----

## [870] Deep Network Approximation in Terms of Intrinsic Parameters

**Authors**: *Zuowei Shen, Haizhao Yang, Shijun Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shen22g.html](https://proceedings.mlr.press/v162/shen22g.html)

**Abstract**:

One of the arguments to explain the success of deep learning is the powerful approximation capacity of deep neural networks. Such capacity is generally accompanied by the explosive growth of the number of parameters, which, in turn, leads to high computational costs. It is of great interest to ask whether we can achieve successful deep learning with a small number of learnable parameters adapting to the target function. From an approximation perspective, this paper shows that the number of parameters that need to be learned can be significantly smaller than people typically expect. First, we theoretically design ReLU networks with a few learnable parameters to achieve an attractive approximation. We prove by construction that, for any Lipschitz continuous function $f$ on $[0,1]^d$ with a Lipschitz constant $\lambda>0$, a ReLU network with $n+2$ intrinsic parameters (those depending on $f$) can approximate $f$ with an exponentially small error $5 \lambda \sqrt{d} \, 2^{-n}$. Such a result is generalized to generic continuous functions. Furthermore, we show that the idea of learning a small number of parameters to achieve a good approximation can be numerically observed. We conduct several experiments to verify that training a small part of parameters can also achieve good results for classification problems if other parameters are pre-specified or pre-trained from a related problem.

----

## [871] Gradient-Free Method for Heavily Constrained Nonconvex Optimization

**Authors**: *Wanli Shi, Hongchang Gao, Bin Gu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22a.html](https://proceedings.mlr.press/v162/shi22a.html)

**Abstract**:

Zeroth-order (ZO) method has been shown to be a powerful method for solving the optimization problem where explicit expression of the gradients is difficult or infeasible to obtain. Recently, due to the practical value of the constrained problems, a lot of ZO Frank-Wolfe or projected ZO methods have been proposed. However, in many applications, we may have a very large number of nonconvex white/black-box constraints, which makes the existing zeroth-order methods extremely inefficient (or even not working) since they need to inquire function value of all the constraints and project the solution to the complicated feasible set. In this paper, to solve the nonconvex problem with a large number of white/black-box constraints, we proposed a doubly stochastic zeroth-order gradient method (DSZOG) with momentum method and adaptive step size. Theoretically, we prove DSZOG can converge to the $\epsilon$-stationary point of the constrained problem. Experimental results in two applications demonstrate the superiority of our method in terms of training time and accuracy compared with other ZO methods for the constrained problem.

----

## [872] Global Optimization of K-Center Clustering

**Authors**: *Mingfei Shi, Kaixun Hua, Jiayang Ren, Yankai Cao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22b.html](https://proceedings.mlr.press/v162/shi22b.html)

**Abstract**:

$k$-center problem is a well-known clustering method and can be formulated as a mixed-integer nonlinear programming problem. This work provides a practical global optimization algorithm for this task based on a reduced-space spatial branch and bound scheme. This algorithm can guarantee convergence to the global optimum by only branching on the centers of clusters, which is independent of the dataset’s cardinality. In addition, a set of feasibility-based bounds tightening techniques are proposed to narrow down the domain of centers and significantly accelerate the convergence. To demonstrate the capacity of this algorithm, we present computational results on 32 datasets. Notably, for the dataset with 14 million samples and 3 features, the serial implementation of the algorithm can converge to an optimality gap of 0.1% within 2 hours. Compared with a heuristic method, the global optimum obtained by our algorithm can reduce the objective function on average by 30.4%.

----

## [873] Pessimistic Q-Learning for Offline Reinforcement Learning: Towards Optimal Sample Complexity

**Authors**: *Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Yuejie Chi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22c.html](https://proceedings.mlr.press/v162/shi22c.html)

**Abstract**:

Offline or batch reinforcement learning seeks to learn a near-optimal policy using history data without active exploration of the environment. To counter the insufficient coverage and sample scarcity of many offline datasets, the principle of pessimism has been recently introduced to mitigate high bias of the estimated values. While pessimistic variants of model-based algorithms (e.g., value iteration with lower confidence bounds) have been theoretically investigated, their model-free counterparts — which do not require explicit model estimation — have not been adequately studied, especially in terms of sample efficiency. To address this inadequacy, we study a pessimistic variant of Q-learning in the context of finite-horizon Markov decision processes, and characterize its sample complexity under the single-policy concentrability assumption which does not require the full coverage of the state-action space. In addition, a variance-reduced pessimistic Q-learning algorithm is proposed to achieve near-optimal sample complexity. Altogether, this work highlights the efficiency of model-free algorithms in offline RL when used in conjunction with pessimism and variance reduction.

----

## [874] Adversarial Masking for Self-Supervised Learning

**Authors**: *Yuge Shi, N. Siddharth, Philip H. S. Torr, Adam R. Kosiorek*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22d.html](https://proceedings.mlr.press/v162/shi22d.html)

**Abstract**:

We propose ADIOS, a masked image model (MIM) framework for self-supervised learning, which simultaneously learns a masking function and an image encoder using an adversarial objective. The image encoder is trained to minimise the distance between representations of the original and that of a masked image. The masking function, conversely, aims at maximising this distance. ADIOS consistently improves on state-of-the-art self-supervised learning (SSL) methods on a variety of tasks and datasets—including classification on ImageNet100 and STL10, transfer learning on CIFAR10/100, Flowers102 and iNaturalist, as well as robustness evaluated on the backgrounds challenge (Xiao et al., 2021)—while generating semantically meaningful masks. Unlike modern MIM models such as MAE, BEiT and iBOT, ADIOS does not rely on the image-patch tokenisation construction of Vision Transformers, and can be implemented with convolutional backbones. We further demonstrate that the masks learned by ADIOS are more effective in improving representation learning of SSL methods than masking schemes used in popular MIM models.

----

## [875] Visual Attention Emerges from Recurrent Sparse Reconstruction

**Authors**: *Baifeng Shi, Yale Song, Neel Joshi, Trevor Darrell, Xin Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22e.html](https://proceedings.mlr.press/v162/shi22e.html)

**Abstract**:

Visual attention helps achieve robust perception under noise, corruption, and distribution shifts in human vision, which are areas where modern neural networks still fall short. We present VARS, Visual Attention from Recurrent Sparse reconstruction, a new attention formulation built on two prominent features of the human visual attention mechanism: recurrency and sparsity. Related features are grouped together via recurrent connections between neurons, with salient objects emerging via sparse regularization. VARS adopts an attractor network with recurrent connections that converges toward a stable pattern over time. Network layers are represented as ordinary differential equations (ODEs), formulating attention as a recurrent attractor network that equivalently optimizes the sparse reconstruction of input using a dictionary of “templates” encoding underlying patterns of data. We show that self-attention is a special case of VARS with a single-step optimization and no sparsity constraint. VARS can be readily used as a replacement for self-attention in popular vision transformers, consistently improving their robustness across various benchmarks.

----

## [876] A Minimax Learning Approach to Off-Policy Evaluation in Confounded Partially Observable Markov Decision Processes

**Authors**: *Chengchun Shi, Masatoshi Uehara, Jiawei Huang, Nan Jiang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22f.html](https://proceedings.mlr.press/v162/shi22f.html)

**Abstract**:

We consider off-policy evaluation (OPE) in Partially Observable Markov Decision Processes (POMDPs), where the evaluation policy depends only on observable variables and the behavior policy depends on unobservable latent variables. Existing works either assume no unmeasured confounders, or focus on settings where both the observation and the state spaces are tabular. In this work, we first propose novel identification methods for OPE in POMDPs with latent confounders, by introducing bridge functions that link the target policy’s value and the observed data distribution. We next propose minimax estimation methods for learning these bridge functions, and construct three estimators based on these estimated bridge functions, corresponding to a value function-based estimator, a marginalized importance sampling estimator, and a doubly-robust estimator. Our proposal permits general function approximation and is thus applicable to settings with continuous or large observation/state spaces. The nonasymptotic and asymptotic properties of the proposed estimators are investigated in detail. A Python implementation of our proposal is available at https://github.com/jiaweihhuang/ Confounded-POMDP-Exp.

----

## [877] Robust Group Synchronization via Quadratic Programming

**Authors**: *Yunpeng Shi, Cole M. Wyeth, Gilad Lerman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shi22g.html](https://proceedings.mlr.press/v162/shi22g.html)

**Abstract**:

We propose a novel quadratic programming formulation for estimating the corruption levels in group synchronization, and use these estimates to solve this problem. Our objective function exploits the cycle consistency of the group and we thus refer to our method as detection and estimation of structural consistency (DESC). This general framework can be extended to other algebraic and geometric structures. Our formulation has the following advantages: it can tolerate corruption as high as the information-theoretic bound, it does not require a good initialization for the estimates of group elements, it has a simple interpretation, and under some mild conditions the global minimum of our objective function exactly recovers the corruption levels. We demonstrate the competitive accuracy of our approach on both synthetic and real data experiments of rotation averaging.

----

## [878] Log-Euclidean Signatures for Intrinsic Distances Between Unaligned Datasets

**Authors**: *Tal Shnitzer, Mikhail Yurochkin, Kristjan H. Greenewald, Justin M. Solomon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shnitzer22a.html](https://proceedings.mlr.press/v162/shnitzer22a.html)

**Abstract**:

The need for efficiently comparing and representing datasets with unknown alignment spans various fields, from model analysis and comparison in machine learning to trend discovery in collections of medical datasets. We use manifold learning to compare the intrinsic geometric structures of different datasets by comparing their diffusion operators, symmetric positive-definite (SPD) matrices that relate to approximations of the continuous Laplace-Beltrami operator from discrete samples. Existing methods typically assume known data alignment and compare such operators in a pointwise manner. Instead, we exploit the Riemannian geometry of SPD matrices to compare these operators and define a new theoretically-motivated distance based on a lower bound of the log-Euclidean metric. Our framework facilitates comparison of data manifolds expressed in datasets with different sizes, numbers of features, and measurement modalities. Our log-Euclidean signature (LES) distance recovers meaningful structural differences, outperforming competing methods in various application domains.

----

## [879] Scalable Computation of Causal Bounds

**Authors**: *Madhumitha Shridharan, Garud Iyengar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shridharan22a.html](https://proceedings.mlr.press/v162/shridharan22a.html)

**Abstract**:

We consider the problem of computing bounds for causal inference problems with unobserved confounders, where identifiability does not hold. Existing non-parametric approaches for computing such bounds use linear programming (LP) formulations that quickly become intractable for existing solvers because the size of the LP grows exponentially in the number of edges in the underlying causal graph. We show that this LP can be significantly pruned by carefully considering the structure of the causal query, allowing us to compute bounds for significantly larger causal inference problems as compared to what is possible using existing techniques. This pruning procedure also allows us to compute the bounds in closed form for a special class of causal graphs and queries, which includes a well-studied family of problems where multiple confounded treatments influence an outcome. We also propose a very efficient greedy heuristic that produces very high quality bounds, and scales to problems that are several orders of magnitude larger than those for which the pruned LP can be solved.

----

## [880] Bit Prioritization in Variational Autoencoders via Progressive Coding

**Authors**: *Rui Shu, Stefano Ermon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shu22a.html](https://proceedings.mlr.press/v162/shu22a.html)

**Abstract**:

The hierarchical variational autoencoder (HVAE) is a popular generative model used for many representation learning tasks. However, its application to image synthesis often yields models with poor sample quality. In this work, we treat image synthesis itself as a hierarchical representation learning problem and regularize an HVAE toward representations that improve the model’s image synthesis performance. We do so by leveraging the progressive coding hypothesis, which claims hierarchical latent variable models that are good at progressive lossy compression will generate high-quality samples. To test this hypothesis, we first show empirically that conventionally-trained HVAEs are not good progressive coders. We then propose a simple method that constrains the hierarchical representations to prioritize the encoding of information beneficial for lossy compression, and show that this modification leads to improved sample quality. Our work lends further support to the progressive coding hypothesis and demonstrates that this hypothesis should be exploited when designing variational autoencoders.

----

## [881] Fair Representation Learning through Implicit Path Alignment

**Authors**: *Changjian Shui, Qi Chen, Jiaqi Li, Boyu Wang, Christian Gagné*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/shui22a.html](https://proceedings.mlr.press/v162/shui22a.html)

**Abstract**:

We consider a fair representation learning perspective, where optimal predictors, on top of the data representation, are ensured to be invariant with respect to different sub-groups. Specifically, we formulate this intuition as a bi-level optimization, where the representation is learned in the outer-loop, and invariant optimal group predictors are updated in the inner-loop. Moreover, the proposed bi-level objective is demonstrated to fulfill the sufficiency rule, which is desirable in various practical scenarios but was not commonly studied in the fair learning. Besides, to avoid the high computational and memory cost of differentiating in the inner-loop of bi-level objective, we propose an implicit path alignment algorithm, which only relies on the solution of inner optimization and the implicit differentiation rather than the exact optimization path. We further analyze the error gap of the implicit approach and empirically validate the proposed method in both classification and regression settings. Experimental results show the consistently better trade-off in prediction performance and fairness measurement.

----

## [882] Faster Algorithms for Learning Convex Functions

**Authors**: *Ali Siahkamari, Durmus Alp Emre Acar, Christopher Liao, Kelly L. Geyer, Venkatesh Saligrama, Brian Kulis*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/siahkamari22a.html](https://proceedings.mlr.press/v162/siahkamari22a.html)

**Abstract**:

The task of approximating an arbitrary convex function arises in several learning problems such as convex regression, learning with a difference of convex (DC) functions, and learning Bregman or $f$-divergences. In this paper, we develop and analyze an approach for solving a broad range of convex function learning problems that is faster than state-of-the-art approaches. Our approach is based on a 2-block ADMM method where each block can be computed in closed form. For the task of convex Lipschitz regression, we establish that our proposed algorithm converges with iteration complexity of $ O(n\sqrt{d}/\epsilon)$ for a dataset $\bm X \in \mathbb R^{n\times d}$ and $\epsilon > 0$. Combined with per-iteration computation complexity, our method converges with the rate $O(n^3 d^{1.5}/\epsilon+n^2 d^{2.5}/\epsilon+n d^3/\epsilon)$. This new rate improves the state of the art rate of $O(n^5d^2/\epsilon)$ if $d = o( n^4)$. Further we provide similar solvers for DC regression and Bregman divergence learning. Unlike previous approaches, our method is amenable to the use of GPUs. We demonstrate on regression and metric learning experiments that our approach is over 100 times faster than existing approaches on some data sets, and produces results that are comparable to state of the art.

----

## [883] Coin Flipping Neural Networks

**Authors**: *Yuval Sieradzki, Nitzan Hodos, Gal Yehuda, Assaf Schuster*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sieradzki22a.html](https://proceedings.mlr.press/v162/sieradzki22a.html)

**Abstract**:

We show that neural networks with access to randomness can outperform deterministic networks by using amplification. We call such networks Coin-Flipping Neural Networks, or CFNNs. We show that a CFNN can approximate the indicator of a d-dimensional ball to arbitrary accuracy with only 2 layers and O(1) neurons, where a 2-layer deterministic network was shown to require Omega(e^d) neurons, an exponential improvement. We prove a highly non-trivial result, that for almost any classification problem, there exists a trivially simple network that solves it given a sufficiently powerful generator for the network’s weights. Combining these results we conjecture that for most classification problems, there is a CFNN which solves them with higher accuracy or fewer neurons than any deterministic network. Finally, we verify our proofs experimentally using novel CFNN architectures on CIFAR10 and CIFAR100, reaching an improvement of 9.25% from the baseline.

----

## [884] Reverse Engineering the Neural Tangent Kernel

**Authors**: *James Benjamin Simon, Sajant Anand, Michael Robert DeWeese*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/simon22a.html](https://proceedings.mlr.press/v162/simon22a.html)

**Abstract**:

The development of methods to guide the design of neural networks is an important open challenge for deep learning theory. As a paradigm for principled neural architecture design, we propose the translation of high-performing kernels, which are better-understood and amenable to first-principles design, into equivalent network architectures, which have superior efficiency, flexibility, and feature learning. To this end, we constructively prove that, with just an appropriate choice of activation function, any positive-semidefinite dot-product kernel can be realized as either the NNGP or neural tangent kernel of a fully-connected neural network with only one hidden layer. We verify our construction numerically and demonstrate its utility as a design tool for finite fully-connected networks in several experiments.

----

## [885] Demystifying the Adversarial Robustness of Random Transformation Defenses

**Authors**: *Chawin Sitawarin, Zachary J. Golan-Strieb, David A. Wagner*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sitawarin22a.html](https://proceedings.mlr.press/v162/sitawarin22a.html)

**Abstract**:

Neural networks’ lack of robustness against attacks raises concerns in security-sensitive settings such as autonomous vehicles. While many countermeasures may look promising, only a few withstand rigorous evaluation. Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT’s evaluation is ineffective and likely overestimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense. Our new attack vastly outperforms the baseline, reducing the accuracy by 83% compared to the 19% reduction by the commonly used EoT attack ($4.3\times$ improvement). Our result indicates that the RT defense on the Imagenette dataset (a ten-class subset of ImageNet) is not robust against adversarial examples. Extending the study further, we use our new attack to adversarially train RT defense (called AdvRT), resulting in a large robustness gain. Code is available at https://github.com/wagnergroup/demystify-random-transform.

----

## [886] Smoothed Adversarial Linear Contextual Bandits with Knapsacks

**Authors**: *Vidyashankar Sivakumar, Shiliang Zuo, Arindam Banerjee*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sivakumar22a.html](https://proceedings.mlr.press/v162/sivakumar22a.html)

**Abstract**:

Many bandit problems are characterized by the learner making decisions under constraints. The learner in Linear Contextual Bandits with Knapsacks (LinCBwK) receives a resource consumption vector in addition to a scalar reward in each time step which are both linear functions of the context corresponding to the chosen arm. For a fixed time horizon $T$, the goal of the learner is to maximize rewards while ensuring resource consumptions do not exceed a pre-specified budget. We present algorithms and characterize regret for LinCBwK in the smoothed setting where base context vectors are assumed to be perturbed by Gaussian noise. We consider both the stochastic and adversarial settings for the base contexts, and our analysis of stochastic LinCBwK can be viewed as a warm-up to the more challenging adversarial LinCBwK. For the stochastic setting, we obtain $O(\sqrt{T})$ additive regret bounds compared to the best context dependent fixed policy. The analysis combines ideas for greedy parameter estimation in \cite{kmrw18, siwb20} and the primal-dual paradigm first explored in \cite{agde17, agde14}. Our main contribution is an algorithm with $O(\log T)$ competitive ratio relative to the best context dependent fixed policy for the adversarial setting. The algorithm for the adversarial setting employs ideas from the primal-dual framework \cite{agde17, agde14} and a novel adaptation of the doubling trick \cite{isss19}.

----

## [887] GenLabel: Mixup Relabeling using Generative Models

**Authors**: *Jy-yong Sohn, Liang Shang, Hongxu Chen, Jaekyun Moon, Dimitris S. Papailiopoulos, Kangwook Lee*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sohn22a.html](https://proceedings.mlr.press/v162/sohn22a.html)

**Abstract**:

Mixup is a data augmentation method that generates new data points by mixing a pair of input data. While mixup generally improves the prediction performance, it sometimes degrades the performance. In this paper, we first identify the main causes of this phenomenon by theoretically and empirically analyzing the mixup algorithm. To resolve this, we propose GenLabel, a simple yet effective relabeling algorithm designed for mixup. In particular, GenLabel helps the mixup algorithm correctly label mixup samples by learning the class-conditional data distribution using generative models. Via theoretical and empirical analysis, we show that mixup, when used together with GenLabel, can effectively resolve the aforementioned phenomenon, improving the accuracy of mixup-trained model.

----

## [888] Communicating via Markov Decision Processes

**Authors**: *Samuel Sokota, Christian A. Schröder de Witt, Maximilian Igl, Luisa M. Zintgraf, Philip H. S. Torr, Martin Strohmeier, J. Zico Kolter, Shimon Whiteson, Jakob N. Foerster*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sokota22a.html](https://proceedings.mlr.press/v162/sokota22a.html)

**Abstract**:

We consider the problem of communicating exogenous information by means of Markov decision process trajectories. This setting, which we call a Markov coding game (MCG), generalizes both source coding and a large class of referential games. MCGs also isolate a problem that is important in decentralized control settings in which cheap-talk is not available—namely, they require balancing communication with the associated cost of communicating. We contribute a theoretically grounded approach to MCGs based on maximum entropy reinforcement learning and minimum entropy coupling that we call MEME. Due to recent breakthroughs in approximation algorithms for minimum entropy coupling, MEME is not merely a theoretical algorithm, but can be applied to practical settings. Empirically, we show both that MEME is able to outperform a strong baseline on small MCGs and that MEME is able to achieve strong performance on extremely large MCGs. To the latter point, we demonstrate that MEME is able to losslessly communicate binary images via trajectories of Cartpole and Pong, while simultaneously achieving the maximal or near maximal expected returns, and that it is even capable of performing well in the presence of actuator noise.

----

## [889] The Multivariate Community Hawkes Model for Dependent Relational Events in Continuous-time Networks

**Authors**: *Hadeel Soliman, Lingfei Zhao, Zhipeng Huang, Subhadeep Paul, Kevin S. Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/soliman22a.html](https://proceedings.mlr.press/v162/soliman22a.html)

**Abstract**:

The stochastic block model (SBM) is one of the most widely used generative models for network data. Many continuous-time dynamic network models are built upon the same assumption as the SBM: edges or events between all pairs of nodes are conditionally independent given the block or community memberships, which prevents them from reproducing higher-order motifs such as triangles that are commonly observed in real networks. We propose the multivariate community Hawkes (MULCH) model, an extremely flexible community-based model for continuous-time networks that introduces dependence between node pairs using structured multivariate Hawkes processes. We fit the model using a spectral clustering and likelihood-based local refinement procedure. We find that our proposed MULCH model is far more accurate than existing models both for predictive and generative tasks.

----

## [890] Disentangling Sources of Risk for Distributional Multi-Agent Reinforcement Learning

**Authors**: *Kyunghwan Son, Junsu Kim, Sungsoo Ahn, Roben Delos Reyes, Yung Yi, Jinwoo Shin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/son22a.html](https://proceedings.mlr.press/v162/son22a.html)

**Abstract**:

In cooperative multi-agent reinforcement learning, the outcomes of agent-wise policies are highly stochastic due to the two sources of risk: (a) random actions taken by teammates and (b) random transition and rewards. Although the two sources have very distinct characteristics, existing frameworks are insufficient to control the risk-sensitivity of agent-wise policies in a disentangled manner. To this end, we propose Disentangled RIsk-sensitive Multi-Agent reinforcement learning (DRIMA) to separately access the risk sources. For example, our framework allows an agent to be optimistic with respect to teammates (who can prosocially adapt) but more risk-neutral with respect to the environment (which does not adapt). Our experiments demonstrate that DRIMA significantly outperforms prior state-of-the-art methods across various scenarios in the StarCraft Multi-agent Challenge environment. Notably, DRIMA shows robust performance where prior methods learn only a highly suboptimal policy, regardless of reward shaping, exploration scheduling, and noisy (random or adversarial) agents.

----

## [891] TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification

**Authors**: *Jaeyun Song, Joonhyung Park, Eunho Yang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/song22a.html](https://proceedings.mlr.press/v162/song22a.html)

**Abstract**:

Learning unbiased node representations under class-imbalanced graph data is challenging due to interactions between adjacent nodes. Existing studies have in common that they compensate the minor class nodes ‘as a group’ according to their overall quantity (ignoring node connections in graph), which inevitably increase the false positive cases for major nodes. We hypothesize that the increase in these false positive cases is highly affected by the label distribution around each node and confirm it experimentally. In addition, in order to handle this issue, we propose Topology-Aware Margin (TAM) to reflect local topology on the learning objective. Our method compares the connectivity pattern of each node with the class-averaged counter-part and adaptively adjusts the margin accordingly based on that. Our method consistently exhibits superiority over the baselines on various node classification benchmark datasets with representative GNN architectures.

----

## [892] A General Recipe for Likelihood-free Bayesian Optimization

**Authors**: *Jiaming Song, Lantao Yu, Willie Neiswanger, Stefano Ermon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/song22b.html](https://proceedings.mlr.press/v162/song22b.html)

**Abstract**:

The acquisition function, a critical component in Bayesian optimization (BO), can often be written as the expectation of a utility function under a surrogate model. However, to ensure that acquisition functions are tractable to optimize, restrictions must be placed on the surrogate model and utility function. To extend BO to a broader class of models and utilities, we propose likelihood-free BO (LFBO), an approach based on likelihood-free inference. LFBO directly models the acquisition function without having to separately perform inference with a probabilistic surrogate model. We show that computing the acquisition function in LFBO can be reduced to optimizing a weighted classification problem, which extends an existing likelihood-free density ratio estimation method related to probability of improvement (PI). By choosing the utility function for expected improvement (EI), LFBO outperforms the aforementioned method, as well as various state-of-the-art black-box optimization methods on several real-world optimization problems. LFBO can also leverage composite structures of the objective function, which further improves its regret by several orders of magnitude.

----

## [893] Fully-Connected Network on Noncompact Symmetric Space and Ridgelet Transform based on Helgason-Fourier Analysis

**Authors**: *Sho Sonoda, Isao Ishikawa, Masahiro Ikeda*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sonoda22a.html](https://proceedings.mlr.press/v162/sonoda22a.html)

**Abstract**:

Neural network on Riemannian symmetric space such as hyperbolic space and the manifold of symmetric positive definite (SPD) matrices is an emerging subject of research in geometric deep learning. Based on the well-established framework of the Helgason-Fourier transform on the noncompact symmetric space, we present a fully-connected network and its associated ridgelet transform on the noncompact symmetric space, covering the hyperbolic neural network (HNN) and the SPDNet as special cases. The ridgelet transform is an analysis operator of a depth-2 continuous network spanned by neurons, namely, it maps an arbitrary given function to the weights of a network. Thanks to the coordinate-free reformulation, the role of nonlinear activation functions is revealed to be a wavelet function. Moreover, the reconstruction formula is applied to present a constructive proof of the universality of finite networks on symmetric spaces.

----

## [894] Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation

**Authors**: *Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang, David Henry Mguni, Jun Wang, Haitham Ammar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sootla22a.html](https://proceedings.mlr.press/v162/sootla22a.html)

**Abstract**:

Satisfying safety constraints almost surely (or with probability one) can be critical for the deployment of Reinforcement Learning (RL) in real-life applications. For example, plane landing and take-off should ideally occur with probability one. We address the problem by introducing Safety Augmented (Saute) Markov Decision Processes (MDPs), where the safety constraints are eliminated by augmenting them into the state-space and reshaping the objective. We show that Saute MDP satisfies the Bellman equation and moves us closer to solving Safe RL with constraints satisfied almost surely. We argue that Saute MDP allows viewing the Safe RL problem from a different perspective enabling new features. For instance, our approach has a plug-and-play nature, i.e., any RL algorithm can be "Sauteed”. Additionally, state augmentation allows for policy generalization across safety constraints. We finally show that Saute RL algorithms can outperform their state-of-the-art counterparts when constraint satisfaction is of high importance.

----

## [895] Lightweight Projective Derivative Codes for Compressed Asynchronous Gradient Descent

**Authors**: *Pedro J. Soto, Ilia Ilmer, Haibin Guan, Jun Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/soto22a.html](https://proceedings.mlr.press/v162/soto22a.html)

**Abstract**:

Coded distributed computation has become common practice for performing gradient descent on large datasets to mitigate stragglers and other faults. This paper proposes a novel algorithm that encodes the partial derivatives themselves and furthermore optimizes the codes by performing lossy compression on the derivative codewords by maximizing the information contained in the codewords while minimizing the information between the codewords. The utility of this application of coding theory is a geometrical consequence of the observed fact in optimization research that noise is tolerable, sometimes even helpful, in gradient descent based learning algorithms since it helps avoid overfitting and local minima. This stands in contrast with much current conventional work on distributed coded computation which focuses on recovering all of the data from the workers. A second further contribution is that the low-weight nature of the coding scheme allows for asynchronous gradient updates since the code can be iteratively decoded; i.e., a worker’s task can immediately be updated into the larger gradient. The directional derivative is always a linear function of the direction vectors; thus, our framework is robust since it can apply linear coding techniques to general machine learning frameworks such as deep neural networks.

----

## [896] Accelerating Bayesian Optimization for Biological Sequence Design with Denoising Autoencoders

**Authors**: *Samuel Stanton, Wesley J. Maddox, Nate Gruver, Phillip M. Maffettone, Emily Delaney, Peyton Greenside, Andrew Gordon Wilson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/stanton22a.html](https://proceedings.mlr.press/v162/stanton22a.html)

**Abstract**:

Bayesian optimization (BayesOpt) is a gold standard for query-efficient continuous optimization. However, its adoption for drug design has been hindered by the discrete, high-dimensional nature of the decision variables. We develop a new approach (LaMBO) which jointly trains a denoising autoencoder with a discriminative multi-task Gaussian process head, allowing gradient-based optimization of multi-objective acquisition functions in the latent space of the autoencoder. These acquisition functions allow LaMBO to balance the explore-exploit tradeoff over multiple design rounds, and to balance objective tradeoffs by optimizing sequences at many different points on the Pareto frontier. We evaluate LaMBO on two small-molecule design tasks, and introduce new tasks optimizing in silico and in vitro properties of large-molecule fluorescent proteins. In our experiments LaMBO outperforms genetic optimizers and does not require a large pretraining corpus, demonstrating that BayesOpt is practical and effective for biological sequence design.

----

## [897] 3D Infomax improves GNNs for Molecular Property Prediction

**Authors**: *Hannes Stärk, Dominique Beaini, Gabriele Corso, Prudencio Tossou, Christian Dallago, Stephan Günnemann, Pietro Lió*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/stark22a.html](https://proceedings.mlr.press/v162/stark22a.html)

**Abstract**:

Molecular property prediction is one of the fastest-growing applications of deep learning with critical real-world impacts. Although the 3D molecular graph structure is necessary for models to achieve strong performance on many tasks, it is infeasible to obtain 3D structures at the scale required by many real-world applications. To tackle this issue, we propose to use existing 3D molecular datasets to pre-train a model to reason about the geometry of molecules given only their 2D molecular graphs. Our method, called 3D Infomax, maximizes the mutual information between learned 3D summary vectors and the representations of a graph neural network (GNN). During fine-tuning on molecules with unknown geometry, the GNN is still able to produce implicit 3D information and uses it for downstream tasks. We show that 3D Infomax provides significant improvements for a wide range of properties, including a 22% average MAE reduction on QM9 quantum mechanical properties. Moreover, the learned representations can be effectively transferred between datasets in different molecular spaces.

----

## [898] EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction

**Authors**: *Hannes Stärk, Octavian Ganea, Lagnajit Pattanaik, Regina Barzilay, Tommi S. Jaakkola*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/stark22b.html](https://proceedings.mlr.press/v162/stark22b.html)

**Abstract**:

Predicting how a drug-like molecule binds to a specific protein target is a core problem in drug discovery. An extremely fast computational binding method would enable key applications such as fast virtual screening or drug engineering. Existing methods are computationally expensive as they rely on heavy candidate sampling coupled with scoring, ranking, and fine-tuning steps. We challenge this paradigm with EquiBind, an SE(3)-equivariant geometric deep learning model performing direct-shot prediction of both i) the receptor binding location (blind docking) and ii) the ligand’s bound pose and orientation. EquiBind achieves significant speed-ups and better quality compared to traditional and recent baselines. Further, we show extra improvements when coupling it with existing fine-tuning techniques at the cost of increased running time. Finally, we propose a novel and fast fine-tuning model that adjusts torsion angles of a ligand’s rotatable bonds based on closed form global minima of the von Mises angular distance to a given input atomic point cloud, avoiding previous expensive differential evolution strategies for energy minimization.

----

## [899] Plug & Play Attacks: Towards Robust and Flexible Model Inversion Attacks

**Authors**: *Lukas Struppek, Dominik Hintersdorf, Antonio De Almeida Correia, Antonia Adler, Kristian Kersting*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/struppek22a.html](https://proceedings.mlr.press/v162/struppek22a.html)

**Abstract**:

Model inversion attacks (MIAs) aim to create synthetic images that reflect the class-wise characteristics from a target classifier’s private training data by exploiting the model’s learned knowledge. Previous research has developed generative MIAs that use generative adversarial networks (GANs) as image priors tailored to a specific target model. This makes the attacks time- and resource-consuming, inflexible, and susceptible to distributional shifts between datasets. To overcome these drawbacks, we present Plug & Play Attacks, which relax the dependency between the target model and image prior, and enable the use of a single GAN to attack a wide range of targets, requiring only minor adjustments to the attack. Moreover, we show that powerful MIAs are possible even with publicly available pre-trained GANs and under strong distributional shifts, for which previous approaches fail to produce meaningful results. Our extensive evaluation confirms the improved robustness and flexibility of Plug & Play Attacks and their ability to create high-quality images revealing sensitive class characteristics.

----

## [900] Scaling-up Diverse Orthogonal Convolutional Networks by a Paraunitary Framework

**Authors**: *Jiahao Su, Wonmin Byeon, Furong Huang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/su22a.html](https://proceedings.mlr.press/v162/su22a.html)

**Abstract**:

Enforcing orthogonality in convolutional neural networks is a remedy for gradient vanishing/exploding problems and sensitivity to perturbation. Many previous approaches for orthogonal convolutions enforce orthogonality on its flattened kernel, which, however, do not lead to the orthogonality of the operation. Some recent approaches consider orthogonality for standard convolutional layers and propose specific classes of their realizations. In this work, we propose a theoretical framework that establishes the equivalence between diverse orthogonal convolutional layers in the spatial domain and the paraunitary systems in the spectral domain. Since 1D paraunitary systems admit a complete factorization, we can parameterize any separable orthogonal convolution as a composition of spatial filters. As a result, our framework endows high expressive power to various convolutional layers while maintaining their exact orthogonality. Furthermore, our layers are memory and computationally efficient for deep networks compared to previous designs. Our versatile framework, for the first time, enables the study of architectural designs for deep orthogonal networks, such as choices of skip connection, initialization, stride, and dilation. Consequently, we scale up orthogonal networks to deep architectures, including ResNet and ShuffleNet, substantially outperforming their shallower counterparts. Finally, we show how to construct residual flows, a flow-based generative model that requires strict Lipschitzness, using our orthogonal networks. Our code will be publicly available at https://github.com/umd-huang-lab/ortho-conv

----

## [901] Divergence-Regularized Multi-Agent Actor-Critic

**Authors**: *Kefan Su, Zongqing Lu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/su22b.html](https://proceedings.mlr.press/v162/su22b.html)

**Abstract**:

Entropy regularization is a popular method in reinforcement learning (RL). Although it has many advantages, it alters the RL objective and makes the converged policy deviate from the optimal policy of the original Markov Decision Process (MDP). Though divergence regularization has been proposed to settle this problem, it cannot be trivially applied to cooperative multi-agent reinforcement learning (MARL). In this paper, we investigate divergence regularization in cooperative MARL and propose a novel off-policy cooperative MARL framework, divergence-regularized multi-agent actor-critic (DMAC). Theoretically, we derive the update rule of DMAC which is naturally off-policy, guarantees the monotonic policy improvement and convergence in both the original MDP and the divergence-regularized MDP, and is not biased by the regularization. We also give a bound of the discrepancy between the converged policy and the optimal policy in the original MDP. DMAC is a flexible framework and can be combined with many existing MARL algorithms. Empirically, we evaluate DMAC in a didactic stochastic game and StarCraft Multi-Agent Challenge and show that DMAC substantially improves the performance of existing MARL algorithms.

----

## [902] Influence-Augmented Local Simulators: a Scalable Solution for Fast Deep RL in Large Networked Systems

**Authors**: *Miguel Suau, Jinke He, Matthijs T. J. Spaan, Frans A. Oliehoek*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/suau22a.html](https://proceedings.mlr.press/v162/suau22a.html)

**Abstract**:

Learning effective policies for real-world problems is still an open challenge for the field of reinforcement learning (RL). The main limitation being the amount of data needed and the pace at which that data can be obtained. In this paper, we study how to build lightweight simulators of complicated systems that can run sufficiently fast for deep RL to be applicable. We focus on domains where agents interact with a reduced portion of a larger environment while still being affected by the global dynamics. Our method combines the use of local simulators with learned models that mimic the influence of the global system. The experiments reveal that incorporating this idea into the deep RL workflow can considerably accelerate the training process and presents several opportunities for the future.

----

## [903] Improved StyleGAN-v2 based Inversion for Out-of-Distribution Images

**Authors**: *Rakshith Subramanyam, Vivek Sivaraman Narayanaswamy, Mark Naufel, Andreas Spanias, Jayaraman J. Thiagarajan*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/subramanyam22a.html](https://proceedings.mlr.press/v162/subramanyam22a.html)

**Abstract**:

Inverting an image onto the latent space of pre-trained generators, e.g., StyleGAN-v2, has emerged as a popular strategy to leverage strong image priors for ill-posed restoration. Several studies have showed that this approach is effective at inverting images similar to the data used for training. However, with out-of-distribution (OOD) data that the generator has not been exposed to, existing inversion techniques produce sub-optimal results. In this paper, we propose SPHInX (StyleGAN with Projection Heads for Inverting X), an approach for accurately embedding OOD images onto the StyleGAN latent space. SPHInX optimizes a style projection head using a novel training strategy that imposes a vicinal regularization in the StyleGAN latent space. To further enhance OOD inversion, SPHInX can additionally optimize a content projection head and noise variables in every layer. Our empirical studies on a suite of OOD data show that, in addition to producing higher quality reconstructions over the state-of-the-art inversion techniques, SPHInX is effective for ill-posed restoration tasks while offering semantic editing capabilities.

----

## [904] Continuous-Time Analysis of Accelerated Gradient Methods via Conservation Laws in Dilated Coordinate Systems

**Authors**: *Jaewook J. Suh, Gyumin Roh, Ernest K. Ryu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/suh22a.html](https://proceedings.mlr.press/v162/suh22a.html)

**Abstract**:

We analyze continuous-time models of accelerated gradient methods through deriving conservation laws in dilated coordinate systems. Namely, instead of analyzing the dynamics of $X(t)$, we analyze the dynamics of $W(t)=t^\alpha(X(t)-X_c)$ for some $\alpha$ and $X_c$ and derive a conserved quantity, analogous to physical energy, in this dilated coordinate system. Through this methodology, we recover many known continuous-time analyses in a streamlined manner and obtain novel continuous-time analyses for OGM-G, an acceleration mechanism for efficiently reducing gradient magnitude that is distinct from that of Nesterov. Finally, we show that a semi-second-order symplectic Euler discretization in the dilated coordinate system leads to an $\mathcal{O}(1/k^2)$ rate on the standard setup of smooth convex minimization, without any further assumptions such as infinite differentiability.

----

## [905] Do Differentiable Simulators Give Better Policy Gradients?

**Authors**: *Hyung Ju Suh, Max Simchowitz, Kaiqing Zhang, Russ Tedrake*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/suh22b.html](https://proceedings.mlr.press/v162/suh22b.html)

**Abstract**:

Differentiable simulators promise faster computation time for reinforcement learning by replacing zeroth-order gradient estimates of a stochastic objective with an estimate based on first-order gradients. However, it is yet unclear what factors decide the performance of the two estimators on complex landscapes that involve long-horizon planning and control on physical systems, despite the crucial relevance of this question for the utility of differentiable simulators. We show that characteristics of certain physical systems, such as stiffness or discontinuities, may compromise the efficacy of the first-order estimator, and analyze this phenomenon through the lens of bias and variance. We additionally propose an $\alpha$-order gradient estimator, with $\alpha \in [0,1]$, which correctly utilizes exact gradients to combine the efficiency of first-order estimates with the robustness of zero-order methods. We demonstrate the pitfalls of traditional estimators and the advantages of the $\alpha$-order estimator on some numerical examples.

----

## [906] Intriguing Properties of Input-Dependent Randomized Smoothing

**Authors**: *Peter Súkeník, Aleksei Kuvshinov, Stephan Günnemann*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sukeni-k22a.html](https://proceedings.mlr.press/v162/sukeni-k22a.html)

**Abstract**:

Randomized smoothing is currently considered the state-of-the-art method to obtain certifiably robust classifiers. Despite its remarkable performance, the method is associated with various serious problems such as “certified accuracy waterfalls”, certification vs. accuracy trade-off, or even fairness issues. Input-dependent smoothing approaches have been proposed with intention of overcoming these flaws. However, we demonstrate that these methods lack formal guarantees and so the resulting certificates are not justified. We show that in general, the input-dependent smoothing suffers from the curse of dimensionality, forcing the variance function to have low semi-elasticity. On the other hand, we provide a theoretical and practical framework that enables the usage of input-dependent smoothing even in the presence of the curse of dimensionality, under strict restrictions. We present one concrete design of the smoothing variance function and test it on CIFAR10 and MNIST. Our design mitigates some of the problems of classical smoothing and is formally underlined, yet further improvement of the design is still necessary.

----

## [907] Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning Environments

**Authors**: *Ryan Sullivan, Jordan K. Terry, Benjamin Black, John P. Dickerson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sullivan22a.html](https://proceedings.mlr.press/v162/sullivan22a.html)

**Abstract**:

Visualizing optimization landscapes has resulted in many fundamental insights in numeric optimization, specifically regarding novel improvements to optimization techniques. However, visualizations of the objective that reinforcement learning optimizes (the "reward surface") have only ever been generated for a small number of narrow contexts. This work presents reward surfaces and related visualizations of 27 of the most widely used reinforcement learning environments in Gym for the first time. We also explore reward surfaces in the policy gradient direction and show for the first time that many popular reinforcement learning environments have frequent "cliffs" (sudden large drops in expected reward). We demonstrate that A2C often "dives off" these cliffs into low reward regions of the parameter space while PPO avoids them, confirming a popular intuition for PPO’s improved performance over previous methods. We additionally introduce a highly extensible library that allows researchers to easily generate these visualizations in the future. Our findings provide new intuition to explain the successes and failures of modern RL methods, and our visualizations concretely characterize several failure modes of reinforcement learning agents in novel ways.

----

## [908] AGNAS: Attention-Guided Micro and Macro-Architecture Search

**Authors**: *Zihao Sun, Yu Hu, Shun Lu, Longxing Yang, Jilin Mei, Yinhe Han, Xiaowei Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sun22a.html](https://proceedings.mlr.press/v162/sun22a.html)

**Abstract**:

Micro- and macro-architecture search have emerged as two popular NAS paradigms recently. Existing methods leverage different search strategies for searching micro- and macro- architectures. When using architecture parameters to search for micro-structure such as normal cell and reduction cell, the architecture parameters can not fully reflect the corresponding operation importance. When searching for the macro-structure chained by pre-defined blocks, many sub-networks need to be sampled for evaluation, which is very time-consuming. To address the two issues, we propose a new search paradigm, that is, leverage the attention mechanism to guide the micro- and macro-architecture search, namely AGNAS. Specifically, we introduce an attention module and plug it behind each candidate operation or each candidate block. We utilize the attention weights to represent the importance of the relevant operations for the micro search or the importance of the relevant blocks for the macro search. Experimental results show that AGNAS can achieve 2.46% test error on CIFAR-10 in the DARTS search space, and 23.4% test error when directly searching on ImageNet in the ProxylessNAS search space. AGNAS also achieves optimal performance on NAS-Bench-201, outperforming state-of-the-art approaches. The source code can be available at https://github.com/Sunzh1996/AGNAS.

----

## [909] Adaptive Random Walk Gradient Descent for Decentralized Optimization

**Authors**: *Tao Sun, Dongsheng Li, Bao Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sun22b.html](https://proceedings.mlr.press/v162/sun22b.html)

**Abstract**:

In this paper, we study the adaptive step size random walk gradient descent with momentum for decentralized optimization, in which the training samples are drawn dependently with each other. We establish theoretical convergence rates of the adaptive step size random walk gradient descent with momentum for both convex and nonconvex settings. In particular, we prove that adaptive random walk algorithms perform as well as the non-adaptive method for dependent data in general cases but achieve acceleration when the stochastic gradients are “sparse”. Moreover, we study the zeroth-order version of adaptive random walk gradient descent and provide corresponding convergence results. All assumptions used in this paper are mild and general, making our results applicable to many machine learning problems.

----

## [910] MAE-DET: Revisiting Maximum Entropy Principle in Zero-Shot NAS for Efficient Object Detection

**Authors**: *Zhenhong Sun, Ming Lin, Xiuyu Sun, Zhiyu Tan, Hao Li, Rong Jin*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sun22c.html](https://proceedings.mlr.press/v162/sun22c.html)

**Abstract**:

In object detection, the detection backbone consumes more than half of the overall inference cost. Recent researches attempt to reduce this cost by optimizing the backbone architecture with the help of Neural Architecture Search (NAS). However, existing NAS methods for object detection require hundreds to thousands of GPU hours of searching, making them impractical in fast-paced research and development. In this work, we propose a novel zero-shot NAS method to address this issue. The proposed method, named MAE-DET, automatically designs efficient detection backbones via the Maximum Entropy Principle without training network parameters, reducing the architecture design cost to nearly zero yet delivering the state-of-the-art (SOTA) performance. Under the hood, MAE-DET maximizes the differential entropy of detection backbones, leading to a better feature extractor for object detection under the same computational budgets. After merely one GPU day of fully automatic design, MAE-DET innovates SOTA detection backbones on multiple detection benchmark datasets with little human intervention. Comparing to ResNet-50 backbone, MAE-DET is $+2.0%$ better in mAP when using the same amount of FLOPs/parameters, and is $1.54$ times faster on NVIDIA V100 at the same mAP. Code and pre-trained models are available here (https://github.com/alibaba/lightweight-neural-architecture-search).

----

## [911] Out-of-Distribution Detection with Deep Nearest Neighbors

**Authors**: *Yiyou Sun, Yifei Ming, Xiaojin Zhu, Yixuan Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sun22d.html](https://proceedings.mlr.press/v162/sun22d.html)

**Abstract**:

Out-of-distribution (OOD) detection is a critical task for deploying machine learning models in the open world. Distance-based methods have demonstrated promise, where testing samples are detected as OOD if they are relatively far away from in-distribution (ID) data. However, prior methods impose a strong distributional assumption of the underlying feature space, which may not always hold. In this paper, we explore the efficacy of non-parametric nearest-neighbor distance for OOD detection, which has been largely overlooked in the literature. Unlike prior works, our method does not impose any distributional assumption, hence providing stronger flexibility and generality. We demonstrate the effectiveness of nearest-neighbor-based OOD detection on several benchmarks and establish superior performance. Under the same model trained on ImageNet-1k, our method substantially reduces the false positive rate (FPR@TPR95) by 24.77% compared to a strong baseline SSD+, which uses a parametric approach Mahalanobis distance in detection. Code is available: https://github.com/deeplearning-wisc/knn-ood.

----

## [912] Black-Box Tuning for Language-Model-as-a-Service

**Authors**: *Tianxiang Sun, Yunfan Shao, Hong Qian, Xuanjing Huang, Xipeng Qiu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sun22e.html](https://proceedings.mlr.press/v162/sun22e.html)

**Abstract**:

Extremely large pre-trained language models (PTMs) such as GPT-3 are usually released as a service. It allows users to design task-specific prompts to query the PTMs through some black-box APIs. In such a scenario, which we call Language-Model-as-a-Service (LMaaS), the gradients of PTMs are usually unavailable. Can we optimize the task prompts by only accessing the model inference APIs? This paper proposes the black-box tuning framework to optimize the continuous prompt prepended to the input text via derivative-free optimization. Instead of optimizing in the original high-dimensional prompt space, which is intractable for traditional derivative-free optimization, we perform optimization in a randomly generated subspace due to the low intrinsic dimensionality of large PTMs. The experimental results show that the black-box tuning with RoBERTa on a few labeled samples not only significantly outperforms manual prompt and GPT-3’s in-context learning, but also surpasses the gradient-based counterparts, i.e., prompt tuning and full model tuning.

----

## [913] Correlated Quantization for Distributed Mean Estimation and Optimization

**Authors**: *Ananda Theertha Suresh, Ziteng Sun, Jae Ro, Felix X. Yu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/suresh22a.html](https://proceedings.mlr.press/v162/suresh22a.html)

**Abstract**:

We study the problem of distributed mean estimation and optimization under communication constraints. We propose a correlated quantization protocol whose error guarantee depends on the deviation of data points instead of their absolute range. The design doesn’t need any prior knowledge on the concentration property of the dataset, which is required to get such dependence in previous works. We show that applying the proposed protocol as a sub-routine in distributed optimization algorithms leads to better convergence rates. We also prove the optimality of our protocol under mild assumptions. Experimental results show that our proposed algorithm outperforms existing mean estimation protocols on a diverse set of tasks.

----

## [914] Causal Imitation Learning under Temporally Correlated Noise

**Authors**: *Gokul Swamy, Sanjiban Choudhury, Drew Bagnell, Steven Wu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/swamy22a.html](https://proceedings.mlr.press/v162/swamy22a.html)

**Abstract**:

We develop algorithms for imitation learning from policy data that was corrupted by temporally correlated noise in expert actions. When noise affects multiple timesteps of recorded data, it can manifest as spurious correlations between states and actions that a learner might latch on to, leading to poor policy performance. To break up these spurious correlations, we apply modern variants of the instrumental variable regression (IVR) technique of econometrics, enabling us to recover the underlying policy without requiring access to an interactive expert. In particular, we present two techniques, one of a generative-modeling flavor (DoubIL) that can utilize access to a simulator, and one of a game-theoretic flavor (ResiduIL) that can be run entirely offline. We find both of our algorithms compare favorably to behavioral cloning on simulated control tasks.

----

## [915] Being Properly Improper

**Authors**: *Tyler Sypherd, Richard Nock, Lalitha Sankar*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/sypherd22a.html](https://proceedings.mlr.press/v162/sypherd22a.html)

**Abstract**:

Properness for supervised losses stipulates that the loss function shapes the learning algorithm towards the true posterior of the data generating distribution. Unfortunately, data in modern machine learning can be corrupted or twisted in many ways. Hence, optimizing a proper loss function on twisted data could perilously lead the learning algorithm towards the twisted posterior, rather than to the desired clean posterior. Many papers cope with specific twists (e.g., label/feature/adversarial noise), but there is a growing need for a unified and actionable understanding atop properness. Our chief theoretical contribution is a generalization of the properness framework with a notion called twist-properness, which delineates loss functions with the ability to "untwist" the twisted posterior into the clean posterior. Notably, we show that a nontrivial extension of a loss function called alpha-loss, which was first introduced in information theory, is twist-proper. We study the twist-proper alpha-loss under a novel boosting algorithm, called PILBoost, and provide formal and experimental results for this algorithm. Our overarching practical conclusion is that the twist-proper alpha-loss outperforms the proper log-loss on several variants of twisted data.

----

## [916] Distributionally-Aware Kernelized Bandit Problems for Risk Aversion

**Authors**: *Sho Takemori*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/takemori22a.html](https://proceedings.mlr.press/v162/takemori22a.html)

**Abstract**:

The kernelized bandit problem is a theoretically justified framework and has solid applications to various fields. Recently, there is a growing interest in generalizing the problem to the optimization of risk-averse metrics such as Conditional Value-at-Risk (CVaR) or Mean-Variance (MV). However, due to the model assumption, most existing methods need explicit design of environment random variables and can incur large regret because of possible high dimensionality of them. To address the issues, in this paper, we model environments using a family of the output distributions (or more precisely, probability kernel) and Kernel Mean Embeddings (KME), and provide novel UCB-type algorithms for CVaR and MV. Moreover, we provide algorithm-independent lower bounds for CVaR in the case of Matérn kernels, and propose a nearly optimal algorithm. Furthermore, we empirically verify our theoretical result in synthetic environments, and demonstrate that our proposed method significantly outperforms a baseline in many cases.

----

## [917] Sequential and Parallel Constrained Max-value Entropy Search via Information Lower Bound

**Authors**: *Shion Takeno, Tomoyuki Tamura, Kazuki Shitara, Masayuki Karasuyama*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/takeno22a.html](https://proceedings.mlr.press/v162/takeno22a.html)

**Abstract**:

Max-value entropy search (MES) is one of the state-of-the-art approaches in Bayesian optimization (BO). In this paper, we propose a novel variant of MES for constrained problems, called Constrained MES via Information lower BOund (CMES-IBO), that is based on a Monte Carlo (MC) estimator of a lower bound of a mutual information (MI). Unlike existing studies, our MI is defined so that uncertainty with respect to feasibility can be incorporated. We derive a lower bound of the MI that guarantees non-negativity, while a constrained counterpart of conventional MES can be negative. We further provide theoretical analysis that assures the low-variability of our estimator which has never been investigated for any existing information-theoretic BO. Moreover, using the conditional MI, we extend CMES-IBO to the parallel setting while maintaining the desirable properties. We demonstrate the effectiveness of CMES-IBO by several benchmark functions and real-world problems.

----

## [918] SQ-VAE: Variational Bayes on Discrete Representation with Self-annealed Stochastic Quantization

**Authors**: *Yuhta Takida, Takashi Shibuya, Wei-Hsiang Liao, Chieh-Hsin Lai, Junki Ohmura, Toshimitsu Uesaka, Naoki Murata, Shusuke Takahashi, Toshiyuki Kumakura, Yuki Mitsufuji*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/takida22a.html](https://proceedings.mlr.press/v162/takida22a.html)

**Abstract**:

One noted issue of vector-quantized variational autoencoder (VQ-VAE) is that the learned discrete representation uses only a fraction of the full capacity of the codebook, also known as codebook collapse. We hypothesize that the training scheme of VQ-VAE, which involves some carefully designed heuristics, underlies this issue. In this paper, we propose a new training scheme that extends the standard VAE via novel stochastic dequantization and quantization, called stochastically quantized variational autoencoder (SQ-VAE). In SQ-VAE, we observe a trend that the quantization is stochastic at the initial stage of the training but gradually converges toward a deterministic quantization, which we call self-annealing. Our experiments show that SQ-VAE improves codebook utilization without using common heuristics. Furthermore, we empirically show that SQ-VAE is superior to VAE and VQ-VAE in vision- and speech-related tasks.

----

## [919] A Tree-based Model Averaging Approach for Personalized Treatment Effect Estimation from Heterogeneous Data Sources

**Authors**: *Xiaoqing Tan, Chung-Chou H. Chang, Ling Zhou, Lu Tang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tan22a.html](https://proceedings.mlr.press/v162/tan22a.html)

**Abstract**:

Accurately estimating personalized treatment effects within a study site (e.g., a hospital) has been challenging due to limited sample size. Furthermore, privacy considerations and lack of resources prevent a site from leveraging subject-level data from other sites. We propose a tree-based model averaging approach to improve the estimation accuracy of conditional average treatment effects (CATE) at a target site by leveraging models derived from other potentially heterogeneous sites, without them sharing subject-level data. To our best knowledge, there is no established model averaging approach for distributed data with a focus on improving the estimation of treatment effects. Specifically, under distributed data networks, our framework provides an interpretable tree-based ensemble of CATE estimators that joins models across study sites, while actively modeling the heterogeneity in data sources through site partitioning. The performance of this approach is demonstrated by a real-world study of the causal effects of oxygen therapy on hospital survival rate and backed up by comprehensive simulation results.

----

## [920] N-Penetrate: Active Learning of Neural Collision Handler for Complex 3D Mesh Deformations

**Authors**: *Qingyang Tan, Zherong Pan, Breannan Smith, Takaaki Shiratori, Dinesh Manocha*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tan22b.html](https://proceedings.mlr.press/v162/tan22b.html)

**Abstract**:

We present a robust learning algorithm to detect and handle collisions in 3D deforming meshes. We first train a neural network to detect collisions and then use a numerical optimization algorithm to resolve penetrations guided by the network. Our learned collision handler can resolve collisions for unseen, high-dimensional meshes with thousands of vertices. To obtain stable network performance in such large and unseen spaces, we apply active learning by progressively inserting new collision data based on the network inferences. We automatically label these new data using an analytical collision detector and progressively fine-tune our detection networks. We evaluate our method for collision handling of complex, 3D meshes coming from several datasets with different shapes and topologies, including datasets corresponding to dressed and undressed human poses, cloth simulations, and human hand poses acquired using multi-view capture systems.

----

## [921] Biased Gradient Estimate with Drastic Variance Reduction for Meta Reinforcement Learning

**Authors**: *Yunhao Tang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tang22a.html](https://proceedings.mlr.press/v162/tang22a.html)

**Abstract**:

Despite the empirical success of meta reinforcement learning (meta-RL), there are still a number poorly-understood discrepancies between theory and practice. Critically, biased gradient estimates are almost always implemented in practice, whereas prior theory on meta-RL only establishes convergence under unbiased gradient estimates. In this work, we investigate such a discrepancy. In particular, (1) We show that unbiased gradient estimates have variance $\Theta(N)$ which linearly depends on the sample size $N$ of the inner loop updates; (2) We propose linearized score function (LSF) gradient estimates, which have bias $\mathcal{O}(1/\sqrt{N})$ and variance $\mathcal{O}(1/N)$; (3) We show that most empirical prior work in fact implements variants of the LSF gradient estimates. This implies that practical algorithms "accidentally" introduce bias to achieve better performance; (4) We establish theoretical guarantees for the LSF gradient estimates in meta-RL regarding its convergence to stationary points, showing better dependency on $N$ than prior work when $N$ is large.

----

## [922] Rethinking Graph Neural Networks for Anomaly Detection

**Authors**: *Jianheng Tang, Jiajin Li, Ziqi Gao, Jia Li*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tang22b.html](https://proceedings.mlr.press/v162/tang22b.html)

**Abstract**:

Graph Neural Networks (GNNs) are widely applied for graph anomaly detection. As one of the key components for GNN design is to select a tailored spectral filter, we take the first step towards analyzing anomalies via the lens of the graph spectrum. Our crucial observation is the existence of anomalies will lead to the ‘right-shift’ phenomenon, that is, the spectral energy distribution concentrates less on low frequencies and more on high frequencies. This fact motivates us to propose the Beta Wavelet Graph Neural Network (BWGNN). Indeed, BWGNN has spectral and spatial localized band-pass filters to better handle the ‘right-shift’ phenomenon in anomalies. We demonstrate the effectiveness of BWGNN on four large-scale anomaly detection datasets. Our code and data are released at https://github.com/squareRoot3/Rethinking-Anomaly-Detection.

----

## [923] Deep Safe Incomplete Multi-view Clustering: Theorem and Algorithm

**Authors**: *Huayi Tang, Yong Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tang22c.html](https://proceedings.mlr.press/v162/tang22c.html)

**Abstract**:

Incomplete multi-view clustering is a significant but challenging task. Although jointly imputing incomplete samples and conducting clustering has been shown to achieve promising performance, learning from both complete and incomplete data may be worse than learning only from complete data, particularly when imputed views are semantic inconsistent with missing views. To address this issue, we propose a novel framework to reduce the clustering performance degradation risk from semantic inconsistent imputed views. Concretely, by the proposed bi-level optimization framework, missing views are dynamically imputed from the learned semantic neighbors, and imputed samples are automatically selected for training. In theory, the empirical risk of the model is no higher than learning only from complete data, and the model is never worse than learning only from complete data in terms of expected risk with high probability. Comprehensive experiments demonstrate that the proposed method achieves superior performance and efficient safe incomplete multi-view clustering.

----

## [924] Virtual Homogeneity Learning: Defending against Data Heterogeneity in Federated Learning

**Authors**: *Zhenheng Tang, Yonggang Zhang, Shaohuai Shi, Xin He, Bo Han, Xiaowen Chu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tang22d.html](https://proceedings.mlr.press/v162/tang22d.html)

**Abstract**:

In federated learning (FL), model performance typically suffers from client drift induced by data heterogeneity, and mainstream works focus on correcting client drift. We propose a different approach named virtual homogeneity learning (VHL) to directly “rectify” the data heterogeneity. In particular, VHL conducts FL with a virtual homogeneous dataset crafted to satisfy two conditions: containing no private information and being separable. The virtual dataset can be generated from pure noise shared across clients, aiming to calibrate the features from the heterogeneous clients. Theoretically, we prove that VHL can achieve provable generalization performance on the natural distribution. Empirically, we demonstrate that VHL endows FL with drastically improved convergence speed and generalization performance. VHL is the first attempt towards using a virtual dataset to address data heterogeneity, offering new and effective means to FL.

----

## [925] Cross-Space Active Learning on Graph Convolutional Networks

**Authors**: *Yufei Tao, Hao Wu, Shiyuan Deng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tao22a.html](https://proceedings.mlr.press/v162/tao22a.html)

**Abstract**:

This paper formalizes cross-space active learning on a graph convolutional network (GCN). The objective is to attain the most accurate hypothesis available in any of the instance spaces generated by the GCN. Subject to the objective, the challenge is to minimize the label cost, measured in the number of vertices whose labels are requested. Our study covers both budget algorithms which terminate after a designated number of label requests, and verifiable algorithms which terminate only after having found an accurate hypothesis. A new separation in label complexity between the two algorithm types is established. The separation is unique to GCNs.

----

## [926] FedNest: Federated Bilevel, Minimax, and Compositional Optimization

**Authors**: *Davoud Ataee Tarzanagh, Mingchen Li, Christos Thrampoulidis, Samet Oymak*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tarzanagh22a.html](https://proceedings.mlr.press/v162/tarzanagh22a.html)

**Abstract**:

Standard federated optimization methods successfully apply to stochastic problems with single-level structure. However, many contemporary ML problems - including adversarial robustness, hyperparameter tuning, actor-critic - fall under nested bilevel programming that subsumes minimax and compositional optimization. In this work, we propose FedNest: A federated alternating stochastic gradient method to address general nested problems. We establish provable convergence rates for FedNest in the presence of heterogeneous data and introduce variations for bilevel, minimax, and compositional optimization. FedNest introduces multiple innovations including federated hypergradient computation and variance reduction to address inner-level heterogeneity. We complement our theory with experiments on hyperparameter & hyper-representation learning and minimax optimization that demonstrate the benefits of our method in practice.

----

## [927] Efficient Distributionally Robust Bayesian Optimization with Worst-case Sensitivity

**Authors**: *Sebastian Shenghong Tay, Chuan Sheng Foo, Daisuke Urano, Richalynn Leong, Bryan Kian Hsiang Low*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tay22a.html](https://proceedings.mlr.press/v162/tay22a.html)

**Abstract**:

In distributionally robust Bayesian optimization (DRBO), an exact computation of the worst-case expected value requires solving an expensive convex optimization problem. We develop a fast approximation of the worst-case expected value based on the notion of worst-case sensitivity that caters to arbitrary convex distribution distances. We provide a regret bound for our novel DRBO algorithm with the fast approximation, and empirically show it is competitive with that using the exact worst-case expected value while incurring significantly less computation time. In order to guide the choice of distribution distance to be used with DRBO, we show that our approximation implicitly optimizes an objective close to an interpretable risk-sensitive value.

----

## [928] LIDL: Local Intrinsic Dimension Estimation Using Approximate Likelihood

**Authors**: *Piotr Tempczyk, Rafal Michaluk, Lukasz Garncarek, Przemyslaw Spurek, Jacek Tabor, Adam Golinski*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tempczyk22a.html](https://proceedings.mlr.press/v162/tempczyk22a.html)

**Abstract**:

Most of the existing methods for estimating the local intrinsic dimension of a data distribution do not scale well to high dimensional data. Many of them rely on a non-parametric nearest neighbours approach which suffers from the curse of dimensionality. We attempt to address that challenge by proposing a novel approach to the problem: Local Intrinsic Dimension estimation using approximate Likelihood (LIDL). Our method relies on an arbitrary density estimation method as its subroutine, and hence tries to sidestep the dimensionality challenge by making use of the recent progress in parametric neural methods for likelihood estimation. We carefully investigate the empirical properties of the proposed method, compare them with our theoretical predictions, show that LIDL yields competitive results on the standard benchmarks for this problem, and that it scales to thousands of dimensions. What is more, we anticipate this approach to improve further with the continuing advances in the density estimation literature.

----

## [929] LCANets: Lateral Competition Improves Robustness Against Corruption and Attack

**Authors**: *Michael A. Teti, Garrett T. Kenyon, Ben Migliori, Juston Moore*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/teti22a.html](https://proceedings.mlr.press/v162/teti22a.html)

**Abstract**:

Although Convolutional Neural Networks (CNNs) achieve high accuracy on image recognition tasks, they lack robustness against realistic corruptions and fail catastrophically when deliberately attacked. Previous CNNs with representations similar to primary visual cortex (V1) were more robust to adversarial attacks on images than current adversarial defense techniques, but they required training on large-scale neural recordings or handcrafting neuroscientific models. Motivated by evidence that neural activity in V1 is sparse, we develop a class of hybrid CNNs, called LCANets, which feature a frontend that performs sparse coding via local lateral competition. We demonstrate that LCANets achieve competitive clean accuracy to standard CNNs on action and image recognition tasks and significantly greater accuracy under various image corruptions. We also perform the first adversarial attacks with full knowledge of a sparse coding CNN layer by attacking LCANets with white-box and black-box attacks, and we show that, contrary to previous hypotheses, sparse coding layers are not very robust to white-box attacks. Finally, we propose a way to use sparse coding layers as a plug-and-play robust frontend by showing that they significantly increase the robustness of adversarially-trained CNNs over corruptions and attacks.

----

## [930] Reverse Engineering ℓp attacks: A block-sparse optimization approach with recovery guarantees

**Authors**: *Darshan Thaker, Paris Giampouras, René Vidal*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/thaker22a.html](https://proceedings.mlr.press/v162/thaker22a.html)

**Abstract**:

Deep neural network-based classifiers have been shown to be vulnerable to imperceptible perturbations to their input, such as $\ell_p$-bounded norm adversarial attacks. This has motivated the development of many defense methods, which are then broken by new attacks, and so on. This paper focuses on a different but related problem of reverse engineering adversarial attacks. Specifically, given an attacked signal, we study conditions under which one can determine the type of attack ($\ell_1$, $\ell_2$ or $\ell_\infty$) and recover the clean signal. We pose this problem as a block-sparse recovery problem, where both the signal and the attack are assumed to lie in a union of subspaces that includes one subspace per class and one subspace per attack type. We derive geometric conditions on the subspaces under which any attacked signal can be decomposed as the sum of a clean signal plus an attack. In addition, by determining the subspaces that contain the signal and the attack, we can also classify the signal and determine the attack type. Experiments on digit and face classification demonstrate the effectiveness of the proposed approach.

----

## [931] Generalised Policy Improvement with Geometric Policy Composition

**Authors**: *Shantanu Thakoor, Mark Rowland, Diana Borsa, Will Dabney, Rémi Munos, André Barreto*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/thakoor22a.html](https://proceedings.mlr.press/v162/thakoor22a.html)

**Abstract**:

We introduce a method for policy improvement that interpolates between the greedy approach of value-based reinforcement learning (RL) and the full planning approach typical of model-based RL. The new method builds on the concept of a geometric horizon model (GHM, also known as a \gamma-model), which models the discounted state-visitation distribution of a given policy. We show that we can evaluate any non-Markov policy that switches between a set of base Markov policies with fixed probability by a careful composition of the base policy GHMs, without any additional learning. We can then apply generalised policy improvement (GPI) to collections of such non-Markov policies to obtain a new Markov policy that will in general outperform its precursors. We provide a thorough theoretical analysis of this approach, develop applications to transfer and standard RL, and empirically demonstrate its effectiveness over standard GPI on a challenging deep RL continuous control task. We also provide an analysis of GHM training methods, proving a novel convergence result regarding previously proposed methods and showing how to train these models stably in deep RL settings.

----

## [932] Algorithms for the Communication of Samples

**Authors**: *Lucas Theis, Noureldin Y. Ahmed*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/theis22a.html](https://proceedings.mlr.press/v162/theis22a.html)

**Abstract**:

The efficient communication of noisy data has applications in several areas of machine learning, such as neural compression or differential privacy, and is also known as reverse channel coding or the channel simulation problem. Here we propose two new coding schemes with practical advantages over existing approaches. First, we introduce ordered random coding (ORC) which uses a simple trick to reduce the coding cost of previous approaches. This scheme further illuminates a connection between schemes based on importance sampling and the so-called Poisson functional representation. Second, we describe a hybrid coding scheme which uses dithered quantization to more efficiently communicate samples from distributions with bounded support.

----

## [933] Consistent Polyhedral Surrogates for Top-k Classification and Variants

**Authors**: *Anish Thilagar, Rafael M. Frongillo, Jessica Finocchiaro, Emma Goodwill*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/thilagar22a.html](https://proceedings.mlr.press/v162/thilagar22a.html)

**Abstract**:

Top-$k$ classification is a generalization of multiclass classification used widely in information retrieval, image classification, and other extreme classification settings. Several hinge-like (piecewise-linear) surrogates have been proposed for the problem, yet all are either non-convex or inconsistent. For the proposed hinge-like surrogates that are convex (i.e., polyhedral), we apply the recent embedding framework of Finocchiaro et al. (2019; 2022) to determine the prediction problem for which the surrogate is consistent. These problems can all be interpreted as variants of top-$k$ classification, which may be better aligned with some applications. We leverage this analysis to derive constraints on the conditional label distributions under which these proposed surrogates become consistent for top-$k$. It has been further suggested that every convex hinge-like surrogate must be inconsistent for top-$k$. Yet, we use the same embedding framework to give the first consistent polyhedral surrogate for this problem.

----

## [934] On the Finite-Time Complexity and Practical Computation of Approximate Stationarity Concepts of Lipschitz Functions

**Authors**: *Lai Tian, Kaiwen Zhou, Anthony Man-Cho So*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tian22a.html](https://proceedings.mlr.press/v162/tian22a.html)

**Abstract**:

We report a practical finite-time algorithmic scheme to compute approximately stationary points for nonconvex nonsmooth Lipschitz functions. In particular, we are interested in two kinds of approximate stationarity notions for nonconvex nonsmooth problems, i.e., Goldstein approximate stationarity (GAS) and near-approximate stationarity (NAS). For GAS, our scheme removes the unrealistic subgradient selection oracle assumption in (Zhang et al., 2020, Assumption 1) and computes GAS with the same finite-time complexity. For NAS, Davis & Drusvyatskiy (2019) showed that $\rho$-weakly convex functions admit finite-time computation, while Tian & So (2021) provided the matching impossibility results of dimension-free finite-time complexity for first-order methods. Complement to these developments, in this paper, we isolate a new class of functions that could be Clarke irregular (and thus not weakly convex anymore) and show that our new algorithmic scheme can compute NAS points for functions in that class within finite time. To demonstrate the wide applicability of our new theoretical framework, we show that $\rho$-margin SVM, $1$-layer, and $2$-layer ReLU neural networks, all being Clarke irregular, satisfy our new conditions.

----

## [935] From Dirichlet to Rubin: Optimistic Exploration in RL without Bonuses

**Authors**: *Daniil Tiapkin, Denis Belomestny, Eric Moulines, Alexey Naumov, Sergey Samsonov, Yunhao Tang, Michal Valko, Pierre Ménard*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tiapkin22a.html](https://proceedings.mlr.press/v162/tiapkin22a.html)

**Abstract**:

We propose the Bayes-UCBVI algorithm for reinforcement learning in tabular, stage-dependent, episodic Markov decision process: a natural extension of the Bayes-UCB algorithm by Kaufmann et al. 2012 for multi-armed bandits. Our method uses the quantile of a Q-value function posterior as upper confidence bound on the optimal Q-value function. For Bayes-UCBVI, we prove a regret bound of order $\widetilde{\mathcal{O}}(\sqrt{H^3SAT})$ where $H$ is the length of one episode, $S$ is the number of states, $A$ the number of actions, $T$ the number of episodes, that matches the lower-bound of $\Omega(\sqrt{H^3SAT})$ up to poly-$\log$ terms in $H,S,A,T$ for a large enough $T$. To the best of our knowledge, this is the first algorithm that obtains an optimal dependence on the horizon $H$ (and $S$) without the need of an involved Bernstein-like bonus or noise. Crucial to our analysis is a new fine-grained anti-concentration bound for a weighted Dirichlet sum that can be of independent interest. We then explain how Bayes-UCBVI can be easily extended beyond the tabular setting, exhibiting a strong link between our algorithm and Bayesian bootstrap (Rubin,1981).

----

## [936] Nonparametric Sparse Tensor Factorization with Hierarchical Gamma Processes

**Authors**: *Conor Tillinghast, Zheng Wang, Shandian Zhe*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tillinghast22a.html](https://proceedings.mlr.press/v162/tillinghast22a.html)

**Abstract**:

We propose a nonparametric factorization approach for sparsely observed tensors. The sparsity does not mean zero-valued entries are massive or dominated. Rather, it implies the observed entries are very few, and even fewer with the growth of the tensor; this is ubiquitous in practice. Compared with the existent works, our model not only leverages the structural information underlying the observed entry indices, but also provides extra interpretability and flexibility {—} it can simultaneously estimate a set of location factors about the intrinsic properties of the tensor nodes, and another set of sociability factors reflecting their extrovert activity in interacting with others; users are free to choose a trade-off between the two types of factors. Specifically, we use hierarchical Gamma processes and Poisson random measures to construct a tensor-valued process, which can freely sample the two types of factors to generate tensors and always guarantees an asymptotic sparsity. We then normalize the tensor process to obtain hierarchical Dirichlet processes to sample each observed entry index, and use a Gaussian process to sample the entry value as a nonlinear function of the factors, so as to capture both the sparse structure properties and complex node relationships. For efficient inference, we use Dirichlet process properties over finite sample partitions, density transformations, and random features to develop a stochastic variational estimation algorithm. We demonstrate the advantage of our method in several benchmark datasets.

----

## [937] Deciphering Lasso-based Classification Through a Large Dimensional Analysis of the Iterative Soft-Thresholding Algorithm

**Authors**: *Malik Tiomoko, Ekkehard Schnoor, Mohamed El Amine Seddik, Igor Colin, Aladin Virmaux*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tiomoko22a.html](https://proceedings.mlr.press/v162/tiomoko22a.html)

**Abstract**:

This paper proposes a theoretical analysis of a Lasso-based classification algorithm. Leveraging on a realistic regime where the dimension of the data $p$ and their number $n$ are of the same order of magnitude, the theoretical classification error is derived as a function of the data statistics. As a result, insights into the functioning of the Lasso in classification and its differences with competing algorithms are highlighted. Our work is based on an original novel analysis of the Iterative Soft-Thresholding Algorithm (ISTA), which may be of independent interest beyond the particular problem studied here and may be adapted to similar iterative schemes. A theoretical optimization of the model’s hyperparameters is also provided, which allows for the data- and time-consuming cross-validation to be avoided. Finally, several applications on synthetic and real data are provided to validate the theoretical study and justify its impact in the design and understanding of algorithms of practical interest.

----

## [938] Extended Unconstrained Features Model for Exploring Deep Neural Collapse

**Authors**: *Tom Tirer, Joan Bruna*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tirer22a.html](https://proceedings.mlr.press/v162/tirer22a.html)

**Abstract**:

The modern strategy for training deep neural networks for classification tasks includes optimizing the network’s weights even after the training error vanishes to further push the training loss toward zero. Recently, a phenomenon termed “neural collapse" (NC) has been empirically observed in this training procedure. Specifically, it has been shown that the learned features (the output of the penultimate layer) of within-class samples converge to their mean, and the means of different classes exhibit a certain tight frame structure, which is also aligned with the last layer’s weights. Recent papers have shown that minimizers with this structure emerge when optimizing a simplified “unconstrained features model" (UFM) with a regularized cross-entropy loss. In this paper, we further analyze and extend the UFM. First, we study the UFM for the regularized MSE loss, and show that the minimizers’ features can have a more delicate structure than in the cross-entropy case. This affects also the structure of the weights. Then, we extend the UFM by adding another layer of weights as well as ReLU nonlinearity to the model and generalize our previous results. Finally, we empirically demonstrate the usefulness of our nonlinear extended UFM in modeling the NC phenomenon that occurs with practical networks.

----

## [939] Object Permanence Emerges in a Random Walk along Memory

**Authors**: *Pavel Tokmakov, Allan Jabri, Jie Li, Adrien Gaidon*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tokmakov22a.html](https://proceedings.mlr.press/v162/tokmakov22a.html)

**Abstract**:

This paper proposes a self-supervised objective for learning representations that localize objects under occlusion - a property known as object permanence. A central question is the choice of learning signal in cases of total occlusion. Rather than directly supervising the locations of invisible objects, we propose a self-supervised objective that requires neither human annotation, nor assumptions about object dynamics. We show that object permanence can emerge by optimizing for temporal coherence of memory: we fit a Markov walk along a space-time graph of memories, where the states in each time step are non-Markovian features from a sequence encoder. This leads to a memory representation that stores occluded objects and predicts their motion, to better localize them. The resulting model outperforms existing approaches on several datasets of increasing complexity and realism, despite requiring minimal supervision, and hence being broadly applicable.

----

## [940] Generic Coreset for Scalable Learning of Monotonic Kernels: Logistic Regression, Sigmoid and more

**Authors**: *Elad Tolochinsky, Ibrahim Jubran, Dan Feldman*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tolochinksy22a.html](https://proceedings.mlr.press/v162/tolochinksy22a.html)

**Abstract**:

Coreset (or core-set) is a small weighted subset $Q$ of an input set $P$ with respect to a given monotonic function $f:\mathbb{R}\to\mathbb{R}$ that provably approximates its fitting loss $\sum_{p\in P}f(p\cdot x)$ to any given $x\in\mathbb{R}^d$. Using $Q$ we can obtain an approximation of $x^*$ that minimizes this loss, by running existing optimization algorithms on $Q$. In this work we provide: (i) A lower bound which proves that there are sets with no coresets smaller than $n=|P|$ for general monotonic loss functions. (ii) A proof that, with an additional common regularization term and under a natural assumption that holds e.g. for logistic regression and the sigmoid activation functions, a small coreset exists for any input $P$. (iii) A generic coreset construction algorithm that computes such a small coreset $Q$ in $O(nd+n\log n)$ time, and (iv) Experimental results with open-source code which demonstrate that our coresets are effective and are much smaller in practice than predicted in theory.

----

## [941] Failure and success of the spectral bias prediction for Laplace Kernel Ridge Regression: the case of low-dimensional data

**Authors**: *Umberto M. Tomasini, Antonio Sclocchi, Matthieu Wyart*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tomasini22a.html](https://proceedings.mlr.press/v162/tomasini22a.html)

**Abstract**:

Recently, several theories including the replica method made predictions for the generalization error of Kernel Ridge Regression. In some regimes, they predict that the method has a ‘spectral bias’: decomposing the true function $f^*$ on the eigenbasis of the kernel, it fits well the coefficients associated with the O(P) largest eigenvalues, where $P$ is the size of the training set. This prediction works very well on benchmark data sets such as images, yet the assumptions these approaches make on the data are never satisfied in practice. To clarify when the spectral bias prediction holds, we first focus on a one-dimensional model where rigorous results are obtained and then use scaling arguments to generalize and test our findings in higher dimensions. Our predictions include the classification case $f(x)=$sign$(x_1)$ with a data distribution that vanishes at the decision boundary $p(x)\sim x_1^{\chi}$. For $\chi>0$ and a Laplace kernel, we find that (i) there exists a cross-over ridge $\lambda^*_{d,\chi}(P)\sim P^{-\frac{1}{d+\chi}}$ such that for $\lambda\gg \lambda^*_{d,\chi}(P)$, the replica method applies, but not for $\lambda\ll\lambda^*_{d,\chi}(P)$, (ii) in the ridge-less case, spectral bias predicts the correct training curve exponent only in the limit $d\rightarrow\infty$.

----

## [942] Quantifying and Learning Linear Symmetry-Based Disentanglement

**Authors**: *Loek Tonnaer, Luis Armando Pérez Rey, Vlado Menkovski, Mike Holenderski, Jim Portegies*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tonnaer22a.html](https://proceedings.mlr.press/v162/tonnaer22a.html)

**Abstract**:

The definition of Linear Symmetry-Based Disentanglement (LSBD) formalizes the notion of linearly disentangled representations, but there is currently no metric to quantify LSBD. Such a metric is crucial to evaluate LSBD methods and to compare them to previous understandings of disentanglement. We propose D_LSBD, a mathematically sound metric to quantify LSBD, and provide a practical implementation for SO(2) groups. Furthermore, from this metric we derive LSBD-VAE, a semi-supervised method to learn LSBD representations. We demonstrate the utility of our metric by showing that (1) common VAE-based disentanglement methods don’t learn LSBD representations, (2) LSBD-VAE, as well as other recent methods, can learn LSBD representations needing only limited supervision on transformations, and (3) various desirable properties expressed by existing disentanglement metrics are also achieved by LSBD representations.

----

## [943] A Temporal-Difference Approach to Policy Gradient Estimation

**Authors**: *Samuele Tosatto, Andrew Patterson, Martha White, Rupam Mahmood*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tosatto22a.html](https://proceedings.mlr.press/v162/tosatto22a.html)

**Abstract**:

The policy gradient theorem (Sutton et al., 2000) prescribes the usage of a cumulative discounted state distribution under the target policy to approximate the gradient. Most algorithms based on this theorem, in practice, break this assumption, introducing a distribution shift that can cause the convergence to poor solutions. In this paper, we propose a new approach of reconstructing the policy gradient from the start state without requiring a particular sampling strategy. The policy gradient calculation in this form can be simplified in terms of a gradient critic, which can be recursively estimated due to a new Bellman equation of gradients. By using temporal-difference updates of the gradient critic from an off-policy data stream, we develop the first estimator that side-steps the distribution shift issue in a model-free way. We prove that, under certain realizability conditions, our estimator is unbiased regardless of the sampling strategy. We empirically show that our technique achieves a superior bias-variance trade-off and performance in presence of off-policy samples.

----

## [944] Simple and near-optimal algorithms for hidden stratification and multi-group learning

**Authors**: *Christopher J. Tosh, Daniel Hsu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tosh22a.html](https://proceedings.mlr.press/v162/tosh22a.html)

**Abstract**:

Multi-group agnostic learning is a formal learning criterion that is concerned with the conditional risks of predictors within subgroups of a population. The criterion addresses recent practical concerns such as subgroup fairness and hidden stratification. This paper studies the structure of solutions to the multi-group learning problem, and provides simple and near-optimal algorithms for the learning problem.

----

## [945] Design-Bench: Benchmarks for Data-Driven Offline Model-Based Optimization

**Authors**: *Brandon Trabucco, Xinyang Geng, Aviral Kumar, Sergey Levine*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/trabucco22a.html](https://proceedings.mlr.press/v162/trabucco22a.html)

**Abstract**:

Black-box model-based optimization (MBO) problems, where the goal is to find a design input that maximizes an unknown objective function, are ubiquitous in a wide range of domains, such as the design of proteins, DNA sequences, aircraft, and robots. Solving model-based optimization problems typically requires actively querying the unknown objective function on design proposals, which means physically building the candidate molecule, aircraft, or robot, testing it, and storing the result. This process can be expensive and time consuming, and one might instead prefer to optimize for the best design using only the data one already has. This setting—called offline MBO—poses substantial and different algorithmic challenges than more commonly studied online techniques. A number of recent works have demonstrated success with offline MBO for high-dimensional optimization problems using high-capacity deep neural networks. However, the lack of standardized benchmarks in this emerging field is making progress difficult to track. To address this, we present Design-Bench, a benchmark for offline MBO with a unified evaluation protocol and reference implementations of recent methods. Our benchmark includes a suite of diverse and realistic tasks derived from real-world optimization problems in biology, materials science, and robotics that present distinct challenges for offline MBO. Our benchmark and reference implementations are released at github.com/rail-berkeley/design-bench and github.com/rail-berkeley/design-baselines.

----

## [946] AnyMorph: Learning Transferable Polices By Inferring Agent Morphology

**Authors**: *Brandon Trabucco, Mariano Phielipp, Glen Berseth*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/trabucco22b.html](https://proceedings.mlr.press/v162/trabucco22b.html)

**Abstract**:

The prototypical approach to reinforcement learning involves training policies tailored to a particular agent from scratch for every new morphology. Recent work aims to eliminate the re-training of policies by investigating whether a morphology-agnostic policy, trained on a diverse set of agents with similar task objectives, can be transferred to new agents with unseen morphologies without re-training. This is a challenging problem that required previous approaches to use hand-designed descriptions of the new agent’s morphology. Instead of hand-designing this description, we propose a data-driven method that learns a representation of morphology directly from the reinforcement learning objective. Ours is the first reinforcement learning algorithm that can train a policy to generalize to new agent morphologies without requiring a description of the agent’s morphology in advance. We evaluate our approach on the standard benchmark for agent-agnostic control, and improve over the current state of the art in zero-shot generalization to new agents. Importantly, our method attains good performance without an explicit description of morphology.

----

## [947] Detecting Adversarial Examples Is (Nearly) As Hard As Classifying Them

**Authors**: *Florian Tramèr*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tramer22a.html](https://proceedings.mlr.press/v162/tramer22a.html)

**Abstract**:

Making classifiers robust to adversarial examples is challenging. Thus, many works tackle the seemingly easier task of detecting perturbed inputs. We show a barrier towards this goal. We prove a hardness reduction between detection and classification of adversarial examples: given a robust detector for attacks at distance $\epsilon$ (in some metric), we show how to build a similarly robust (but inefficient) classifier for attacks at distance $\epsilon/2$. Our reduction is computationally inefficient, but preserves the data complexity of the original detector. The reduction thus cannot be directly used to build practical classifiers. Instead, it is a useful sanity check to test whether empirical detection results imply something much stronger than the authors presumably anticipated (namely a highly robust and data-efficient classifier). To illustrate, we revisit $14$ empirical detector defenses published over the past years. For $12/14$ defenses, we show that the claimed detection results imply an inefficient classifier with robustness far beyond the state-of-the-art— thus casting some doubts on the results’ validity. Finally, we show that our reduction applies in both directions: a robust classifier for attacks at distance $\epsilon/2$ implies an inefficient robust detector at distance $\epsilon$. Thus, we argue that robust classification and robust detection should be regarded as (near)-equivalent problems, if we disregard their computational complexity.

----

## [948] Nesterov Accelerated Shuffling Gradient Method for Convex Optimization

**Authors**: *Trang H. Tran, Katya Scheinberg, Lam M. Nguyen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tran22a.html](https://proceedings.mlr.press/v162/tran22a.html)

**Abstract**:

In this paper, we propose Nesterov Accelerated Shuffling Gradient (NASG), a new algorithm for the convex finite-sum minimization problems. Our method integrates the traditional Nesterov’s acceleration momentum with different shuffling sampling schemes. We show that our algorithm has an improved rate of $\Ocal(1/T)$ using unified shuffling schemes, where $T$ is the number of epochs. This rate is better than that of any other shuffling gradient methods in convex regime. Our convergence analysis does not require an assumption on bounded domain or a bounded gradient condition. For randomized shuffling schemes, we improve the convergence bound further. When employing some initial condition, we show that our method converges faster near the small neighborhood of the solution. Numerical simulations demonstrate the efficiency of our algorithm.

----

## [949] A Completely Tuning-Free and Robust Approach to Sparse Precision Matrix Estimation

**Authors**: *Chau Tran, Guo Yu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tran22b.html](https://proceedings.mlr.press/v162/tran22b.html)

**Abstract**:

Despite the vast literature on sparse Gaussian graphical models, current methods either are asymptotically tuning-free (which still require fine-tuning in practice) or hinge on computationally expensive methods (e.g., cross-validation) to determine the proper level of regularization. We propose a completely tuning-free approach for estimating sparse Gaussian graphical models. Our method uses model-agnostic regularization parameters to estimate each column of the target precision matrix and enjoys several desirable properties. Computationally, our estimator can be computed efficiently by linear programming. Theoretically, the proposed estimator achieves minimax optimal convergence rates under various norms. We further propose a second-stage enhancement with non-convex penalties which possesses the strong oracle property. Through comprehensive numerical studies, our methods demonstrate favorable statistical performance. Remarkably, our methods exhibit strong robustness to the violation of the Gaussian assumption and significantly outperform competing methods in the heavy-tailed settings.

----

## [950] Tackling covariate shift with node-based Bayesian neural networks

**Authors**: *Trung Q. Trinh, Markus Heinonen, Luigi Acerbi, Samuel Kaski*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/trinh22a.html](https://proceedings.mlr.press/v162/trinh22a.html)

**Abstract**:

Bayesian neural networks (BNNs) promise improved generalization under covariate shift by providing principled probabilistic representations of epistemic uncertainty. However, weight-based BNNs often struggle with high computational complexity of large-scale architectures and datasets. Node-based BNNs have recently been introduced as scalable alternatives, which induce epistemic uncertainty by multiplying each hidden node with latent random variables, while learning a point-estimate of the weights. In this paper, we interpret these latent noise variables as implicit representations of simple and domain-agnostic data perturbations during training, producing BNNs that perform well under covariate shift due to input corruptions. We observe that the diversity of the implicit corruptions depends on the entropy of the latent variables, and propose a straightforward approach to increase the entropy of these variables during training. We evaluate the method on out-of-distribution image classification benchmarks, and show improved uncertainty estimation of node-based BNNs under covariate shift due to input perturbations. As a side effect, the method also provides robustness against noisy training labels.

----

## [951] Fenrir: Physics-Enhanced Regression for Initial Value Problems

**Authors**: *Filip Tronarp, Nathanael Bosch, Philipp Hennig*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tronarp22a.html](https://proceedings.mlr.press/v162/tronarp22a.html)

**Abstract**:

We show how probabilistic numerics can be used to convert an initial value problem into a Gauss–Markov process parametrised by the dynamics of the initial value problem. Consequently, the often difficult problem of parameter estimation in ordinary differential equations is reduced to hyper-parameter estimation in Gauss–Markov regression, which tends to be considerably easier. The method’s relation and benefits in comparison to classical numerical integration and gradient matching approaches is elucidated. In particular, the method can, in contrast to gradient matching, handle partial observations, and has certain routes for escaping local optima not available to classical numerical integration. Experimental results demonstrate that the method is on par or moderately better than competing approaches.

----

## [952] Interpretable Off-Policy Learning via Hyperbox Search

**Authors**: *Daniel Tschernutter, Tobias Hatt, Stefan Feuerriegel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tschernutter22a.html](https://proceedings.mlr.press/v162/tschernutter22a.html)

**Abstract**:

Personalized treatment decisions have become an integral part of modern medicine. Thereby, the aim is to make treatment decisions based on individual patient characteristics. Numerous methods have been developed for learning such policies from observational data that achieve the best outcome across a certain policy class. Yet these methods are rarely interpretable. However, interpretability is often a prerequisite for policy learning in clinical practice. In this paper, we propose an algorithm for interpretable off-policy learning via hyperbox search. In particular, our policies can be represented in disjunctive normal form (i.e., OR-of-ANDs) and are thus intelligible. We prove a universal approximation theorem that shows that our policy class is flexible enough to approximate any measurable function arbitrarily well. For optimization, we develop a tailored column generation procedure within a branch-and-bound framework. Using a simulation study, we demonstrate that our algorithm outperforms state-of-the-art methods from interpretable off-policy learning in terms of regret. Using real-word clinical data, we perform a user study with actual clinical experts, who rate our policies as highly interpretable.

----

## [953] FriendlyCore: Practical Differentially Private Aggregation

**Authors**: *Eliad Tsfadia, Edith Cohen, Haim Kaplan, Yishay Mansour, Uri Stemmer*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tsfadia22a.html](https://proceedings.mlr.press/v162/tsfadia22a.html)

**Abstract**:

Differentially private algorithms for common metric aggregation tasks, such as clustering or averaging, often have limited practicality due to their complexity or to the large number of data points that is required for accurate results. We propose a simple and practical tool $\mathsf{FriendlyCore}$ that takes a set of points ${\cal D}$ from an unrestricted (pseudo) metric space as input. When ${\cal D}$ has effective diameter $r$, $\mathsf{FriendlyCore}$ returns a “stable” subset ${\cal C} \subseteq {\cal D}$ that includes all points, except possibly few outliers, and is guaranteed to have diameter $r$. $\mathsf{FriendlyCore}$ can be used to preprocess the input before privately aggregating it, potentially simplifying the aggregation or boosting its accuracy. Surprisingly, $\mathsf{FriendlyCore}$ is light-weight with no dependence on the dimension. We empirically demonstrate its advantages in boosting the accuracy of mean estimation and clustering tasks such as $k$-means and $k$-GMM, outperforming tailored methods.

----

## [954] Pairwise Conditional Gradients without Swap Steps and Sparser Kernel Herding

**Authors**: *Kazuma Tsuji, Ken'ichiro Tanaka, Sebastian Pokutta*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tsuji22a.html](https://proceedings.mlr.press/v162/tsuji22a.html)

**Abstract**:

The Pairwise Conditional Gradients (PCG) algorithm is a powerful extension of the Frank-Wolfe algorithm leading to particularly sparse solutions, which makes PCG very appealing for problems such as sparse signal recovery, sparse regression, and kernel herding. Unfortunately, PCG exhibits so-called swap steps that might not provide sufficient primal progress. The number of these bad steps is bounded by a function in the dimension and as such known guarantees do not generalize to the infinite-dimensional case, which would be needed for kernel herding. We propose a new variant of PCG, the so-called Blended Pairwise Conditional Gradients (BPCG). This new algorithm does not exhibit any swap steps, is very easy to implement, and does not require any internal gradient alignment procedures. The convergence rate of BPCG is basically that of PCG if no drop steps would occur and as such is no worse than PCG but improves and provides new rates in many cases. Moreover, we observe in the numerical experiments that BPCG’s solutions are much sparser than those of PCG. We apply BPCG to the kernel herding setting, where we derive nice quadrature rules and provide numerical results demonstrating the performance of our method.

----

## [955] Prototype Based Classification from Hierarchy to Fairness

**Authors**: *Mycal Tucker, Julie A. Shah*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/tucker22a.html](https://proceedings.mlr.press/v162/tucker22a.html)

**Abstract**:

Artificial neural nets can represent and classify many types of high-dimensional data but are often tailored to particular applications – e.g., for “fair” or “hierarchical” classification. Once an architecture has been selected, it is often difficult for humans to adjust models for a new task; for example, a hierarchical classifier cannot be easily transformed into a fair classifier that shields a protected field. Our contribution in this work is a new neural network architecture, the concept subspace network (CSN), which generalizes existing specialized classifiers to produce a unified model capable of learning a spectrum of multi-concept relationships. We demonstrate that CSNs reproduce state-of-the-art results in fair classification when enforcing concept independence, may be transformed into hierarchical classifiers, or may even reconcile fairness and hierarchy within a single classifier. The CSN is inspired by and matches the performance of existing prototype-based classifiers that promote interpretability.

----

## [956] Consensus Multiplicative Weights Update: Learning to Learn using Projector-based Game Signatures

**Authors**: *Nelson Vadori, Rahul Savani, Thomas Spooner, Sumitra Ganesh*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vadori22a.html](https://proceedings.mlr.press/v162/vadori22a.html)

**Abstract**:

Cheung and Piliouras (2020) recently showed that two variants of the Multiplicative Weights Update method - OMWU and MWU - display opposite convergence properties depending on whether the game is zero-sum or cooperative. Inspired by this work and the recent literature on learning to optimize for single functions, we introduce a new framework for learning last-iterate convergence to Nash Equilibria in games, where the update rule’s coefficients (learning rates) along a trajectory are learnt by a reinforcement learning policy that is conditioned on the nature of the game: the game signature. We construct the latter using a new decomposition of two-player games into eight components corresponding to commutative projection operators, generalizing and unifying recent game concepts studied in the literature. We compare the performance of various update rules when their coefficients are learnt, and show that the RL policy is able to exploit the game signature across a wide range of game types. In doing so, we introduce CMWU, a new algorithm that extends consensus optimization to the constrained case, has local convergence guarantees for zero-sum bimatrix games, and show that it enjoys competitive performance on both zero-sum games with constant coefficients and across a spectrum of games when its coefficients are learnt.

----

## [957] Self-Supervised Models of Audio Effectively Explain Human Cortical Responses to Speech

**Authors**: *Aditya R. Vaidya, Shailee Jain, Alexander Huth*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vaidya22a.html](https://proceedings.mlr.press/v162/vaidya22a.html)

**Abstract**:

Self-supervised language models are very effective at predicting high-level cortical responses during language comprehension. However, the best current models of lower-level auditory processing in the human brain rely on either hand-constructed acoustic filters or representations from supervised audio neural networks. In this work, we capitalize on the progress of self-supervised speech representation learning (SSL) to create new state-of-the-art models of the human auditory system. Compared against acoustic baselines, phonemic features, and supervised models, representations from the middle layers of self-supervised models (APC, wav2vec, wav2vec 2.0, and HuBERT) consistently yield the best prediction performance for fMRI recordings within the auditory cortex (AC). Brain areas involved in low-level auditory processing exhibit a preference for earlier SSL model layers, whereas higher-level semantic areas prefer later layers. We show that these trends are due to the models’ ability to encode information at multiple linguistic levels (acoustic, phonetic, and lexical) along their representation depth. Overall, these results show that self-supervised models effectively capture the hierarchy of information relevant to different stages of speech processing in human cortex.

----

## [958] Path-Gradient Estimators for Continuous Normalizing Flows

**Authors**: *Lorenz Vaitl, Kim Andrea Nicoli, Shinichi Nakajima, Pan Kessel*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vaitl22a.html](https://proceedings.mlr.press/v162/vaitl22a.html)

**Abstract**:

Recent work has established a path-gradient estimator for simple variational Gaussian distributions and has argued that the path-gradient is particularly beneficial in the regime in which the variational distribution approaches the exact target distribution. In many applications, this regime can however not be reached by a simple Gaussian variational distribution. In this work, we overcome this crucial limitation by proposing a path-gradient estimator for the considerably more expressive variational family of continuous normalizing flows. We outline an efficient algorithm to calculate this estimator and establish its superior performance empirically.

----

## [959] Improved Convergence Rates for Sparse Approximation Methods in Kernel-Based Learning

**Authors**: *Sattar Vakili, Jonathan Scarlett, Da-Shan Shiu, Alberto Bernacchia*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vakili22a.html](https://proceedings.mlr.press/v162/vakili22a.html)

**Abstract**:

Kernel-based models such as kernel ridge regression and Gaussian processes are ubiquitous in machine learning applications for regression and optimization. It is well known that a major downside for kernel-based models is the high computational cost; given a dataset of $n$ samples, the cost grows as $\mathcal{O}(n^3)$. Existing sparse approximation methods can yield a significant reduction in the computational cost, effectively reducing the actual cost down to as low as $\mathcal{O}(n)$ in certain cases. Despite this remarkable empirical success, significant gaps remain in the existing results for the analytical bounds on the error due to approximation. In this work, we provide novel confidence intervals for the Nyström method and the sparse variational Gaussian process approximation method, which we establish using novel interpretations of the approximate (surrogate) posterior variance of the models. Our confidence intervals lead to improved performance bounds in both regression and optimization problems.

----

## [960] EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning

**Authors**: *Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben-Itzhak, Michael Mitzenmacher*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vargaftik22a.html](https://proceedings.mlr.press/v162/vargaftik22a.html)

**Abstract**:

Distributed Mean Estimation (DME) is a central building block in federated learning, where clients send local gradients to a parameter server for averaging and updating the model. Due to communication constraints, clients often use lossy compression techniques to compress the gradients, resulting in estimation inaccuracies. DME is more challenging when clients have diverse network conditions, such as constrained communication budgets and packet losses. In such settings, DME techniques often incur a significant increase in the estimation error leading to degraded learning performance. In this work, we propose a robust DME technique named EDEN that naturally handles heterogeneous communication budgets and packet losses. We derive appealing theoretical guarantees for EDEN and evaluate it empirically. Our results demonstrate that EDEN consistently improves over state-of-the-art DME techniques.

----

## [961] Towards Noise-adaptive, Problem-adaptive (Accelerated) Stochastic Gradient Descent

**Authors**: *Sharan Vaswani, Benjamin Dubois-Taine, Reza Babanezhad*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vaswani22a.html](https://proceedings.mlr.press/v162/vaswani22a.html)

**Abstract**:

We aim to make stochastic gradient descent (SGD) adaptive to (i) the noise $\sigma^2$ in the stochastic gradients and (ii) problem-dependent constants. When minimizing smooth, strongly-convex functions with condition number $\kappa$, we prove that $T$ iterations of SGD with exponentially decreasing step-sizes and knowledge of the smoothness can achieve an $\tilde{O} \left(\exp \left( \nicefrac{-T}{\kappa} \right) + \nicefrac{\sigma^2}{T} \right)$ rate, without knowing $\sigma^2$. In order to be adaptive to the smoothness, we use a stochastic line-search (SLS) and show (via upper and lower-bounds) that SGD with SLS converges at the desired rate, but only to a neighbourhood of the solution. On the other hand, we prove that SGD with an offline estimate of the smoothness converges to the minimizer. However, its rate is slowed down proportional to the estimation error. Next, we prove that SGD with Nesterov acceleration and exponential step-sizes (referred to as ASGD) can achieve the near-optimal $\tilde{O} \left(\exp \left( \nicefrac{-T}{\sqrt{\kappa}} \right) + \nicefrac{\sigma^2}{T} \right)$ rate, without knowledge of $\sigma^2$. When used with offline estimates of the smoothness and strong-convexity, ASGD still converges to the solution, albeit at a slower rate. Finally, we empirically demonstrate the effectiveness of exponential step-sizes coupled with a novel variant of SLS.

----

## [962] Correlation Clustering via Strong Triadic Closure Labeling: Fast Approximation Algorithms and Practical Lower Bounds

**Authors**: *Nate Veldt*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/veldt22a.html](https://proceedings.mlr.press/v162/veldt22a.html)

**Abstract**:

Correlation clustering is a widely studied framework for clustering based on pairwise similarity and dissimilarity scores, but its best approximation algorithms rely on impractical linear programming relaxations. We present faster approximation algorithms that avoid these relaxations, for two well-studied special cases: cluster editing and cluster deletion. We accomplish this by drawing new connections to edge labeling problems related to the principle of strong triadic closure. This leads to faster and more practical linear programming algorithms, as well as extremely scalable combinatorial techniques, including the first combinatorial approximation algorithm for cluster deletion. In practice, our algorithms produce approximate solutions that nearly match the best algorithms in quality, while scaling to problems that are orders of magnitude larger.

----

## [963] The CLRS Algorithmic Reasoning Benchmark

**Authors**: *Petar Velickovic, Adrià Puigdomènech Badia, David Budden, Razvan Pascanu, Andrea Banino, Misha Dashevskiy, Raia Hadsell, Charles Blundell*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/velickovic22a.html](https://proceedings.mlr.press/v162/velickovic22a.html)

**Abstract**:

Learning representations of algorithms is an emerging area of machine learning, seeking to bridge concepts from neural networks with classical algorithms. Several important works have investigated whether neural networks can effectively reason like algorithms, typically by learning to execute them. The common trend in the area, however, is to generate targeted kinds of algorithmic data to evaluate specific hypotheses, making results hard to transfer across publications, and increasing the barrier of entry. To consolidate progress and work towards unified evaluation, we propose the CLRS Algorithmic Reasoning Benchmark, covering classical algorithms from the Introduction to Algorithms textbook. Our benchmark spans a variety of algorithmic reasoning procedures, including sorting, searching, dynamic programming, graph algorithms, string algorithms and geometric algorithms. We perform extensive experiments to demonstrate how several popular algorithmic reasoning baselines perform on these tasks, and consequently, highlight links to several open challenges. Our library is readily available at https://github.com/deepmind/clrs.

----

## [964] Bregman Power k-Means for Clustering Exponential Family Data

**Authors**: *Adithya Vellal, Saptarshi Chakraborty, Jason Q. Xu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vellal22a.html](https://proceedings.mlr.press/v162/vellal22a.html)

**Abstract**:

Recent progress in center-based clustering algorithms combats poor local minima by implicit annealing through a family of generalized means. These methods are variations of Lloyd’s celebrated k-means algorithm, and are most appropriate for spherical clusters such as those arising from Gaussian data. In this paper, we bridge these new algorithmic advances to classical work on hard clustering under Bregman divergences, which enjoy a bijection to exponential family distributions and are thus well-suited for clustering objects arising from a breadth of data generating mechanisms. The elegant properties of Bregman divergences allow us to maintain closed form updates in a simple and transparent algorithm, and moreover lead to new theoretical arguments for establishing finite sample bounds that relax the bounded support assumption made in the existing state of the art. Additionally, we consider thorough empirical analyses on simulated experiments and a case study on rainfall data, finding that the proposed method outperforms existing peer methods in a variety of non-Gaussian data settings.

----

## [965] Estimation in Rotationally Invariant Generalized Linear Models via Approximate Message Passing

**Authors**: *Ramji Venkataramanan, Kevin Kögler, Marco Mondelli*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/venkataramanan22a.html](https://proceedings.mlr.press/v162/venkataramanan22a.html)

**Abstract**:

We consider the problem of signal estimation in generalized linear models defined via rotationally invariant design matrices. Since these matrices can have an arbitrary spectral distribution, this model is well suited for capturing complex correlation structures which often arise in applications. We propose a novel family of approximate message passing (AMP) algorithms for signal estimation, and rigorously characterize their performance in the high-dimensional limit via a state evolution recursion. Our rotationally invariant AMP has complexity of the same order as the existing AMP derived under the restrictive assumption of a Gaussian design; our algorithm also recovers this existing AMP as a special case. Numerical results showcase a performance close to Vector AMP (which is conjectured to be Bayes-optimal in some settings), but obtained with a much lower complexity, as the proposed algorithm does not require a computationally expensive singular value decomposition.

----

## [966] Bayesian Optimization under Stochastic Delayed Feedback

**Authors**: *Arun Verma, Zhongxiang Dai, Bryan Kian Hsiang Low*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/verma22a.html](https://proceedings.mlr.press/v162/verma22a.html)

**Abstract**:

Bayesian optimization (BO) is a widely-used sequential method for zeroth-order optimization of complex and expensive-to-compute black-box functions. The existing BO methods assume that the function evaluation (feedback) is available to the learner immediately or after a fixed delay. Such assumptions may not be practical in many real-life problems like online recommendations, clinical trials, and hyperparameter tuning where feedback is available after a random delay. To benefit from the experimental parallelization in these problems, the learner needs to start new function evaluations without waiting for delayed feedback. In this paper, we consider the BO under stochastic delayed feedback problem. We propose algorithms with sub-linear regret guarantees that efficiently address the dilemma of selecting new function queries while waiting for randomly delayed feedback. Building on our results, we also make novel contributions to batch BO and contextual Gaussian process bandits. Experiments on synthetic and real-life datasets verify the performance of our algorithms.

----

## [967] VarScene: A Deep Generative Model for Realistic Scene Graph Synthesis

**Authors**: *Tathagat Verma, Abir De, Yateesh Agrawal, Vishwa Vinay, Soumen Chakrabarti*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/verma22b.html](https://proceedings.mlr.press/v162/verma22b.html)

**Abstract**:

Scene graphs are powerful abstractions that capture relationships between objects in images by modeling objects as nodes and relationships as edges. Generation of realistic synthetic scene graphs has applications like scene synthesis and data augmentation for supervised learning. Existing graph generative models are predominantly targeted toward molecular graphs, leveraging the limited vocabulary of atoms and bonds and also the well-defined semantics of chemical compounds. In contrast, scene graphs have much larger object and relation vocabularies, and their semantics are latent. To address this challenge, we propose a variational autoencoder for scene graphs, which is optimized for the maximum mean discrepancy (MMD) between the ground truth scene graph distribution and distribution of the generated scene graphs. Our method views a scene graph as a collection of star graphs and encodes it into a latent representation of the underlying stars. The decoder generates scene graphs by learning to sample the component stars and edges between them. Our experiments show that our method is able to mimic the underlying scene graph generative process more accurately than several state-of-the-art baselines.

----

## [968] Calibrated Learning to Defer with One-vs-All Classifiers

**Authors**: *Rajeev Verma, Eric T. Nalisnick*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/verma22c.html](https://proceedings.mlr.press/v162/verma22c.html)

**Abstract**:

The learning to defer (L2D) framework has the potential to make AI systems safer. For a given input, the system can defer the decision to a human if the human is more likely than the model to take the correct action. We study the calibration of L2D systems, investigating if the probabilities they output are sound. We find that Mozannar & Sontag’s (2020) multiclass framework is not calibrated with respect to expert correctness. Moreover, it is not even guaranteed to produce valid probabilities due to its parameterization being degenerate for this purpose. We propose an L2D system based on one-vs-all classifiers that is able to produce calibrated probabilities of expert correctness. Furthermore, our loss function is also a consistent surrogate for multiclass L2D, like Mozannar & Sontag’s (2020). Our experiments verify that not only is our system calibrated, but this benefit comes at no cost to accuracy. Our model’s accuracy is always comparable (and often superior) to Mozannar & Sontag’s (2020) model’s in tasks ranging from hate speech detection to galaxy classification to diagnosis of skin lesions.

----

## [969] Regret Bounds for Stochastic Shortest Path Problems with Linear Function Approximation

**Authors**: *Daniel Vial, Advait Parulekar, Sanjay Shakkottai, R. Srikant*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vial22a.html](https://proceedings.mlr.press/v162/vial22a.html)

**Abstract**:

We propose an algorithm that uses linear function approximation (LFA) for stochastic shortest path (SSP). Under minimal assumptions, it obtains sublinear regret, is computationally efficient, and uses stationary policies. To our knowledge, this is the first such algorithm in the LFA literature (for SSP or other formulations). Our algorithm is a special case of a more general one, which achieves regret square root in the number of episodes given access to a computation oracle.

----

## [970] On Implicit Bias in Overparameterized Bilevel Optimization

**Authors**: *Paul Vicol, Jonathan P. Lorraine, Fabian Pedregosa, David Duvenaud, Roger B. Grosse*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vicol22a.html](https://proceedings.mlr.press/v162/vicol22a.html)

**Abstract**:

Many problems in machine learning involve bilevel optimization (BLO), including hyperparameter optimization, meta-learning, and dataset distillation. Bilevel problems involve inner and outer parameters, each optimized for its own objective. Often, at least one of the two levels is underspecified and there are multiple ways to choose among equivalent optima. Inspired by recent studies of the implicit bias induced by optimization algorithms in single-level optimization, we investigate the implicit bias of different gradient-based algorithms for jointly optimizing the inner and outer parameters. We delineate two standard BLO methods—cold-start and warm-start BLO—and show that the converged solution or long-run behavior depends to a large degree on these and other algorithmic choices, such as the hypergradient approximation. We also show that the solutions from warm-start BLO can encode a surprising amount of information about the outer objective, even when the outer optimization variables are low-dimensional. We believe that implicit bias deserves as central a role in the study of bilevel optimization as it has attained in the study of single-level neural net optimization.

----

## [971] Multiclass learning with margin: exponential rates with no bias-variance trade-off

**Authors**: *Stefano Vigogna, Giacomo Meanti, Ernesto De Vito, Lorenzo Rosasco*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vigogna22a.html](https://proceedings.mlr.press/v162/vigogna22a.html)

**Abstract**:

We study the behavior of error bounds for multiclass classification under suitable margin conditions. For a wide variety of methods we prove that the classification error under a hard-margin condition decreases exponentially fast without any bias-variance trade-off. Different convergence rates can be obtained in correspondence of different margin assumptions. With a self-contained and instructive analysis we are able to generalize known results from the binary to the multiclass setting.

----

## [972] Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning

**Authors**: *Adam R. Villaflor, Zhe Huang, Swapnil Pande, John M. Dolan, Jeff Schneider*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/villaflor22a.html](https://proceedings.mlr.press/v162/villaflor22a.html)

**Abstract**:

Impressive results in natural language processing (NLP) based on the Transformer neural network architecture have inspired researchers to explore viewing offline reinforcement learning (RL) as a generic sequence modeling problem. Recent works based on this paradigm have achieved state-of-the-art results in several of the mostly deterministic offline Atari and D4RL benchmarks. However, because these methods jointly model the states and actions as a single sequencing problem, they struggle to disentangle the effects of the policy and world dynamics on the return. Thus, in adversarial or stochastic environments, these methods lead to overly optimistic behavior that can be dangerous in safety-critical systems like autonomous driving. In this work, we propose a method that addresses this optimism bias by explicitly disentangling the policy and world models, which allows us at test time to search for policies that are robust to multiple possible futures in the environment. We demonstrate our method’s superior performance on a variety of autonomous driving tasks in simulation.

----

## [973] Bayesian Nonparametrics for Offline Skill Discovery

**Authors**: *Valentin Villecroze, Harry J. Braviner, Panteha Naderian, Chris J. Maddison, Gabriel Loaiza-Ganem*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/villecroze22a.html](https://proceedings.mlr.press/v162/villecroze22a.html)

**Abstract**:

Skills or low-level policies in reinforcement learning are temporally extended actions that can speed up learning and enable complex behaviours. Recent work in offline reinforcement learning and imitation learning has proposed several techniques for skill discovery from a set of expert trajectories. While these methods are promising, the number K of skills to discover is always a fixed hyperparameter, which requires either prior knowledge about the environment or an additional parameter search to tune it. We first propose a method for offline learning of options (a particular skill framework) exploiting advances in variational inference and continuous relaxations. We then highlight an unexplored connection between Bayesian nonparametrics and offline skill discovery, and show how to obtain a nonparametric version of our model. This version is tractable thanks to a carefully structured approximate posterior with a dynamically-changing number of options, removing the need to specify K. We also show how our nonparametric extension can be applied in other skill frameworks, and empirically demonstrate that our method can outperform state-of-the-art offline skill learning algorithms across a variety of environments.

----

## [974] Hermite Polynomial Features for Private Data Generation

**Authors**: *Margarita Vinaroz, Mohammad-Amin Charusaie, Frederik Harder, Kamil Adamczewski, Mijung Park*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vinaroz22a.html](https://proceedings.mlr.press/v162/vinaroz22a.html)

**Abstract**:

Kernel mean embedding is a useful tool to compare probability measures. Despite its usefulness, kernel mean embedding considers infinite-dimensional features, which are challenging to handle in the context of differentially private data generation. A recent work, DP-MERF (Harder et al., 2021), proposes to approximate the kernel mean embedding of data distribution using finite-dimensional random features, which yields an analytically tractable sensitivity of approximate kernel mean embedding. However, the required number of random features in DP-MERF is excessively high, often ten thousand to a hundred thousand, which worsens the sensitivity of the approximate kernel mean embedding. To improve the sensitivity, we propose to replace random features with Hermite polynomial features. Unlike the random features, the Hermite polynomial features are ordered, where the features at the low orders contain more information on the distribution than those at the high orders. Hence, a relatively low order of Hermite polynomial features can more accurately approximate the mean embedding of the data distribution compared to a significantly higher number of random features. As a result, the Hermite polynomial features help us to improve the privacy-accuracy trade-off compared to DP-MERF, as demonstrated on several heterogeneous tabular datasets, as well as several image benchmark datasets.

----

## [975] What Can Linear Interpolation of Neural Network Loss Landscapes Tell Us?

**Authors**: *Tiffany J. Vlaar, Jonathan Frankle*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vlaar22a.html](https://proceedings.mlr.press/v162/vlaar22a.html)

**Abstract**:

Studying neural network loss landscapes provides insights into the nature of the underlying optimization problems. Unfortunately, loss landscapes are notoriously difficult to visualize in a human-comprehensible fashion. One common way to address this problem is to plot linear slices of the landscape, for example from the initial state of the network to the final state after optimization. On the basis of this analysis, prior work has drawn broader conclusions about the difficulty of the optimization problem. In this paper, we put inferences of this kind to the test, systematically evaluating how linear interpolation and final performance vary when altering the data, choice of initialization, and other optimizer and architecture design choices. Further, we use linear interpolation to study the role played by individual layers and substructures of the network. We find that certain layers are more sensitive to the choice of initialization, but that the shape of the linear path is not indicative of the changes in test accuracy of the model. Our results cast doubt on the broader intuition that the presence or absence of barriers when interpolating necessarily relates to the success of optimization.

----

## [976] Multirate Training of Neural Networks

**Authors**: *Tiffany J. Vlaar, Benedict J. Leimkuhler*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/vlaar22b.html](https://proceedings.mlr.press/v162/vlaar22b.html)

**Abstract**:

We propose multirate training of neural networks: partitioning neural network parameters into "fast" and "slow" parts which are trained on different time scales, where slow parts are updated less frequently. By choosing appropriate partitionings we can obtain substantial computational speed-up for transfer learning tasks. We show for applications in vision and NLP that we can fine-tune deep neural networks in almost half the time, without reducing the generalization performance of the resulting models. We analyze the convergence properties of our multirate scheme and draw a comparison with vanilla SGD. We also discuss splitting choices for the neural network parameters which could enhance generalization performance when neural networks are trained from scratch. A multirate approach can be used to learn different features present in the data and as a form of regularization. Our paper unlocks the potential of using multirate techniques for neural network training and provides several starting points for future work in this area.

----

## [977] Provably Adversarially Robust Nearest Prototype Classifiers

**Authors**: *Václav Vorácek, Matthias Hein*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/voracek22a.html](https://proceedings.mlr.press/v162/voracek22a.html)

**Abstract**:

Nearest prototype classifiers (NPCs) assign to each input point the label of the nearest prototype with respect to a chosen distance metric. A direct advantage of NPCs is that the decisions are interpretable. Previous work could provide lower bounds on the minimal adversarial perturbation in the $\ell_p$-threat model when using the same $\ell_p$-distance for the NPCs. In this paper we provide a complete discussion on the complexity when using $\ell_p$-distances for decision and $\ell_q$-threat models for certification for $p,q \in \{1,2,\infty\}$. In particular we provide scalable algorithms for the exact computation of the minimal adversarial perturbation when using $\ell_2$-distance and improved lower bounds in other cases. Using efficient improved lower bounds we train our \textbf{P}rovably adversarially robust \textbf{NPC} (PNPC), for MNIST which have better $\ell_2$-robustness guarantees than neural networks. Additionally, we show up to our knowledge the first certification results w.r.t. to the LPIPS perceptual metric which has been argued to be a more realistic threat model for image classification than $\ell_p$-balls. Our PNPC has on CIFAR10 higher certified robust accuracy than the empirical robust accuracy reported in \cite{laidlaw2021perceptual}. The code is available in our \href{https://github.com/vvoracek/Provably-Adversarially-Robust-Nearest-Prototype-Classifiers}{repository}.

----

## [978] First-Order Regret in Reinforcement Learning with Linear Function Approximation: A Robust Estimation Approach

**Authors**: *Andrew J. Wagenmaker, Yifang Chen, Max Simchowitz, Simon S. Du, Kevin G. Jamieson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wagenmaker22a.html](https://proceedings.mlr.press/v162/wagenmaker22a.html)

**Abstract**:

Obtaining first-order regret bounds—regret bounds scaling not as the worst-case but with some measure of the performance of the optimal policy on a given instance—is a core question in sequential decision-making. While such bounds exist in many settings, they have proven elusive in reinforcement learning with large state spaces. In this work we address this gap, and show that it is possible to obtain regret scaling as $\widetilde{\mathcal{O}}(\sqrt{d^3 H^3 \cdot V_1^\star \cdot K} + d^{3.5}H^3\log K )$ in reinforcement learning with large state spaces, namely the linear MDP setting. Here $V_1^\star$ is the value of the optimal policy and $K$ is the number of episodes. We demonstrate that existing techniques based on least squares estimation are insufficient to obtain this result, and instead develop a novel robust self-normalized concentration bound based on the robust Catoni mean estimator, which may be of independent interest.

----

## [979] Reward-Free RL is No Harder Than Reward-Aware RL in Linear Markov Decision Processes

**Authors**: *Andrew J. Wagenmaker, Yifang Chen, Max Simchowitz, Simon S. Du, Kevin G. Jamieson*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wagenmaker22b.html](https://proceedings.mlr.press/v162/wagenmaker22b.html)

**Abstract**:

Reward-free reinforcement learning (RL) considers the setting where the agent does not have access to a reward function during exploration, but must propose a near-optimal policy for an arbitrary reward function revealed only after exploring. In the the tabular setting, it is well known that this is a more difficult problem than reward-aware (PAC) RL—where the agent has access to the reward function during exploration—with optimal sample complexities in the two settings differing by a factor of $|\mathcal{S}|$, the size of the state space. We show that this separation does not exist in the setting of linear MDPs. We first develop a computationally efficient algorithm for reward-free RL in a $d$-dimensional linear MDP with sample complexity scaling as $\widetilde{\mathcal{O}}(d^2 H^5/\epsilon^2)$. We then show a lower bound with matching dimension-dependence of $\Omega(d^2 H^2/\epsilon^2)$, which holds for the reward-aware RL setting. To our knowledge, our approach is the first computationally efficient algorithm to achieve optimal $d$ dependence in linear MDPs, even in the single-reward PAC setting. Our algorithm relies on a novel procedure which efficiently traverses a linear MDP, collecting samples in any given “feature direction”, and enjoys a sample complexity scaling optimally in the (linear MDP equivalent of the) maximal state visitation probability. We show that this exploration procedure can also be applied to solve the problem of obtaining “well-conditioned” covariates in linear MDPs.

----

## [980] Training Characteristic Functions with Reinforcement Learning: XAI-methods play Connect Four

**Authors**: *Stephan Wäldchen, Sebastian Pokutta, Felix Huber*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/waldchen22a.html](https://proceedings.mlr.press/v162/waldchen22a.html)

**Abstract**:

Characteristic functions (from cooperative game theory) are able to evaluate partial inputs and form the basis for attribution methods like Shapley values. These attribution methods allow us to measure how important each input component is for the function output—one of the goals of explainable AI (XAI). Given a standard classifier function, it is unclear how partial input should be realised. Instead, most XAI-methods for black-box classifiers like neural networks consider counterfactual inputs that generally lie off-manifold, which makes them hard to evaluate and easy to manipulate. We propose a setup to directly train characteristic functions in the form of neural networks to play simple two-player games. We apply this to the game of Connect Four by randomly hiding colour information from our agents during training. This has three advantages for comparing XAI-methods: It alleviates the ambiguity about how to realise partial input, makes off-manifold evaluation unnecessary and allows us to compare the methods by letting them play against each other.

----

## [981] Retroformer: Pushing the Limits of End-to-end Retrosynthesis Transformer

**Authors**: *Yue Wan, Chang-Yu Hsieh, Ben Liao, Shengyu Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wan22a.html](https://proceedings.mlr.press/v162/wan22a.html)

**Abstract**:

Retrosynthesis prediction is one of the fundamental challenges in organic synthesis. The task is to predict the reactants given a core product. With the advancement of machine learning, computer-aided synthesis planning has gained increasing interest. Numerous methods were proposed to solve this problem with different levels of dependency on additional chemical knowledge. In this paper, we propose Retroformer, a novel Transformer-based architecture for retrosynthesis prediction without relying on any cheminformatics tools for molecule editing. Via the proposed local attention head, the model can jointly encode the molecular sequence and graph, and efficiently exchange information between the local reactive region and the global reaction context. Retroformer reaches the new state-of-the-art accuracy for the end-to-end template-free retrosynthesis, and improves over many strong baselines on better molecule and reaction validity. In addition, its generative procedure is highly interpretable and controllable. Overall, Retroformer pushes the limits of the reaction reasoning ability of deep generative models.

----

## [982] Safe Exploration for Efficient Policy Evaluation and Comparison

**Authors**: *Runzhe Wan, Branislav Kveton, Rui Song*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wan22b.html](https://proceedings.mlr.press/v162/wan22b.html)

**Abstract**:

High-quality data plays a central role in ensuring the accuracy of policy evaluation. This paper initiates the study of efficient and safe data collection for bandit policy evaluation. We formulate the problem and investigate its several representative variants. For each variant, we analyze its statistical properties, derive the corresponding exploration policy, and design an efficient algorithm for computing it. Both theoretical analysis and experiments support the usefulness of the proposed methods.

----

## [983] Greedy based Value Representation for Optimal Coordination in Multi-agent Reinforcement Learning

**Authors**: *Lipeng Wan, Zeyang Liu, Xingyu Chen, Xuguang Lan, Nanning Zheng*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wan22c.html](https://proceedings.mlr.press/v162/wan22c.html)

**Abstract**:

Due to the representation limitation of the joint Q value function, multi-agent reinforcement learning methods with linear value decomposition (LVD) or monotonic value decomposition (MVD) suffer from relative overgeneralization. As a result, they can not ensure optimal consistency (i.e., the correspondence between individual greedy actions and the best team performance). In this paper, we derive the expression of the joint Q value function of LVD and MVD. According to the expression, we draw a transition diagram, where each self-transition node (STN) is a possible convergence. To ensure the optimal consistency, the optimal node is required to be the unique STN. Therefore, we propose the greedy-based value representation (GVR), which turns the optimal node into an STN via inferior target shaping and eliminates the non-optimal STNs via superior experience replay. Theoretical proofs and empirical results demonstrate that given the true Q values, GVR ensures the optimal consistency under sufficient exploration. Besides, in tasks where the true Q values are unavailable, GVR achieves an adaptive trade-off between optimality and stability. Our method outperforms state-of-the-art baselines in experiments on various benchmarks.

----

## [984] Towards Evaluating Adaptivity of Model-Based Reinforcement Learning Methods

**Authors**: *Yi Wan, Ali Rahimi-Kalahroudi, Janarthanan Rajendran, Ida Momennejad, Sarath Chandar, Harm van Seijen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wan22d.html](https://proceedings.mlr.press/v162/wan22d.html)

**Abstract**:

In recent years, a growing number of deep model-based reinforcement learning (RL) methods have been introduced. The interest in deep model-based RL is not surprising, given its many potential benefits, such as higher sample efficiency and the potential for fast adaption to changes in the environment. However, we demonstrate, using an improved version of the recently introduced Local Change Adaptation (LoCA) setup, that well-known model-based methods such as PlaNet and DreamerV2 perform poorly in their ability to adapt to local environmental changes. Combined with prior work that made a similar observation about the other popular model-based method, MuZero, a trend appears to emerge, suggesting that current deep model-based methods have serious limitations. We dive deeper into the causes of this poor performance, by identifying elements that hurt adaptive behavior and linking these to underlying techniques frequently used in deep model-based RL. We empirically validate these insights in the case of linear function approximation by demonstrating that a modified version of linear Dyna achieves effective adaptation to local changes. Furthermore, we provide detailed insights into the challenges of building an adaptive nonlinear model-based method, by experimenting with a nonlinear version of Dyna.

----

## [985] Fast Lossless Neural Compression with Integer-Only Discrete Flows

**Authors**: *Siyu Wang, Jianfei Chen, Chongxuan Li, Jun Zhu, Bo Zhang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22a.html](https://proceedings.mlr.press/v162/wang22a.html)

**Abstract**:

By applying entropy codecs with learned data distributions, neural compressors have significantly outperformed traditional codecs in terms of compression ratio. However, the high inference latency of neural networks hinders the deployment of neural compressors in practical applications. In this work, we propose Integer-only Discrete Flows (IODF) an efficient neural compressor with integer-only arithmetic. Our work is built upon integer discrete flows, which consists of invertible transformations between discrete random variables. We propose efficient invertible transformations with integer-only arithmetic based on 8-bit quantization. Our invertible transformation is equipped with learnable binary gates to remove redundant filters during inference. We deploy IODF with TensorRT on GPUs, achieving $10\times$ inference speedup compared to the fastest existing neural compressors, while retaining the high compression rates on ImageNet32 and ImageNet64.

----

## [986] Accelerating Shapley Explanation via Contributive Cooperator Selection

**Authors**: *Guanchu Wang, Yu-Neng Chuang, Mengnan Du, Fan Yang, Quan Zhou, Pushkar Tripathi, Xuanting Cai, Xia Ben Hu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22b.html](https://proceedings.mlr.press/v162/wang22b.html)

**Abstract**:

Even though Shapley value provides an effective explanation for a DNN model prediction, the computation relies on the enumeration of all possible input feature coalitions, which leads to the exponentially growing complexity. To address this problem, we propose a novel method SHEAR to significantly accelerate the Shapley explanation for DNN models, where only a few coalitions of input features are involved in the computation. The selection of the feature coalitions follows our proposed Shapley chain rule to minimize the absolute error from the ground-truth Shapley values, such that the computation can be both efficient and accurate. To demonstrate the effectiveness, we comprehensively evaluate SHEAR across multiple metrics including the absolute error from the ground-truth Shapley value, the faithfulness of the explanations, and running speed. The experimental results indicate SHEAR consistently outperforms state-of-the-art baseline methods across different evaluation metrics, which demonstrates its potentials in real-world applications where the computational resource is limited.

----

## [987] Denoised MDPs: Learning World Models Better Than the World Itself

**Authors**: *Tongzhou Wang, Simon S. Du, Antonio Torralba, Phillip Isola, Amy Zhang, Yuandong Tian*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22c.html](https://proceedings.mlr.press/v162/wang22c.html)

**Abstract**:

The ability to separate signal from noise, and reason with clean abstractions, is critical to intelligence. With this ability, humans can efficiently perform real world tasks without considering all possible nuisance factors. How can artificial agents do the same? What kind of information can agents safely discard as noises? In this work, we categorize information out in the wild into four types based on controllability and relation with reward, and formulate useful information as that which is both controllable and reward-relevant. This framework clarifies the kinds information removed by various prior work on representation learning in reinforcement learning (RL), and leads to our proposed approach of learning a Denoised MDP that explicitly factors out certain noise distractors. Extensive experiments on variants of DeepMind Control Suite and RoboDesk demonstrate superior performance of our denoised world model over using raw observations alone, and over prior works, across policy optimization control tasks as well as the non-control task of joint position regression. Project Page: https://ssnl.github.io/denoised_mdp/ Code: https://github.com/facebookresearch/denoised_mdp/

----

## [988] Neural Implicit Dictionary Learning via Mixture-of-Expert Training

**Authors**: *Peihao Wang, Zhiwen Fan, Tianlong Chen, Zhangyang Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22d.html](https://proceedings.mlr.press/v162/wang22d.html)

**Abstract**:

Representing visual signals by coordinate-based deep fully-connected networks has been shown advantageous in fitting complex details and solving inverse problems than discrete grid-based representation. However, acquiring such a continuous Implicit Neural Representation (INR) requires tedious per-scene training on tons of signal measurements, which limits its practicality. In this paper, we present a generic INR framework that achieves both data and training efficiency by learning a Neural Implicit Dictionary (NID) from a data collection and representing INR as a functional combination of wavelets sampled from the dictionary. Our NID assembles a group of coordinate-based subnetworks which are tuned to span the desired function space. After training, one can instantly and robustly acquire an unseen scene representation by solving the coding coefficients. To parallelly optimize a large group of networks, we borrow the idea from Mixture-of-Expert (MoE) to design and train our network with a sparse gating mechanism. Our experiments show that, NID can improve reconstruction of 2D images or 3D scenes by 2 orders of magnitude faster with up to 98% less input data. We further demonstrate various applications of NID in image inpainting and occlusion removal, which are considered to be challenging with vanilla INR. Our codes are available in https://github.com/VITA-Group/Neural-Implicit-Dict.

----

## [989] Robust Models Are More Interpretable Because Attributions Look Normal

**Authors**: *Zifan Wang, Matt Fredrikson, Anupam Datta*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22e.html](https://proceedings.mlr.press/v162/wang22e.html)

**Abstract**:

Recent work has found that adversarially-robust deep networks used for image classification are more interpretable: their feature attributions tend to be sharper, and are more concentrated on the objects associated with the image’s ground- truth class. We show that smooth decision boundaries play an important role in this enhanced interpretability, as the model’s input gradients around data points will more closely align with boundaries’ normal vectors when they are smooth. Thus, because robust models have smoother boundaries, the results of gradient- based attribution methods, like Integrated Gradients and DeepLift, will capture more accurate information about nearby decision boundaries. This understanding of robust interpretability leads to our second contribution: boundary attributions, which aggregate information about the normal vectors of local decision bound- aries to explain a classification outcome. We show that by leveraging the key fac- tors underpinning robust interpretability, boundary attributions produce sharper, more concentrated visual explanations{—}even on non-robust models.

----

## [990] Disentangling Disease-related Representation from Obscure for Disease Prediction

**Authors**: *Chu-ran Wang, Fei Gao, Fandong Zhang, Fangwei Zhong, Yizhou Yu, Yizhou Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22f.html](https://proceedings.mlr.press/v162/wang22f.html)

**Abstract**:

Disease-related representations play a crucial role in image-based disease prediction such as cancer diagnosis, due to its considerable generalization capacity. However, it is still a challenge to identify lesion characteristics in obscured images, as many lesions are obscured by other tissues. In this paper, to learn the representations for identifying obscured lesions, we propose a disentanglement learning strategy under the guidance of alpha blending generation in an encoder-decoder framework (DAB-Net). Specifically, we take mammogram mass benign/malignant classification as an example. In our framework, composite obscured mass images are generated by alpha blending and then explicitly disentangled into disease-related mass features and interference glands features. To achieve disentanglement learning, features of these two parts are decoded to reconstruct the mass and the glands with corresponding reconstruction losses, and only disease-related mass features are fed into the classifier for disease prediction. Experimental results on one public dataset DDSM and three in-house datasets demonstrate that the proposed strategy can achieve state-of-the-art performance. DAB-Net achieves substantial improvements of 3.9%~4.4% AUC in obscured cases. Besides, the visualization analysis shows the model can better disentangle the mass and glands in the obscured image, suggesting the effectiveness of our solution in exploring the hidden characteristics in this challenging problem.

----

## [991] Solving Stackelberg Prediction Game with Least Squares Loss via Spherically Constrained Least Squares Reformulation

**Authors**: *Jiali Wang, Wen Huang, Rujun Jiang, Xudong Li, Alex L. Wang*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22g.html](https://proceedings.mlr.press/v162/wang22g.html)

**Abstract**:

The Stackelberg prediction game (SPG) is popular in characterizing strategic interactions between a learner and an attacker. As an important special case, the SPG with least squares loss (SPG-LS) has recently received much research attention. Although initially formulated as a difficult bi-level optimization problem, SPG-LS admits tractable reformulations which can be polynomially globally solved by semidefinite programming or second order cone programming. However, all the available approaches are not well-suited for handling large-scale datasets, especially those with huge numbers of features. In this paper, we explore an alternative reformulation of the SPG-LS. By a novel nonlinear change of variables, we rewrite the SPG-LS as a spherically constrained least squares (SCLS) problem. Theoretically, we show that an $\epsilon$ optimal solutions to the SCLS (and the SPG-LS) can be achieved in $\tilde O(N/\sqrt{\epsilon})$ floating-point operations, where $N$ is the number of nonzero entries in the data matrix. Practically, we apply two well-known methods for solving this new reformulation, i.e., the Krylov subspace method and the Riemannian trust region method. Both algorithms are factorization free so that they are suitable for solving large scale problems. Numerical results on both synthetic and real-world datasets indicate that the SPG-LS, equipped with the SCLS reformulation, can be solved orders of magnitude faster than the state of the art.

----

## [992] VLMixer: Unpaired Vision-Language Pre-training via Cross-Modal CutMix

**Authors**: *Teng Wang, Wenhao Jiang, Zhichao Lu, Feng Zheng, Ran Cheng, Chengguo Yin, Ping Luo*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22h.html](https://proceedings.mlr.press/v162/wang22h.html)

**Abstract**:

Existing vision-language pre-training (VLP) methods primarily rely on paired image-text datasets, which are either annotated by enormous human labors or crawled from the internet followed by elaborate data cleaning techniques. To reduce the dependency on well-aligned image-text pairs, it is promising to directly leverage the large-scale text-only and image-only corpora. This paper proposes a data augmentation method, namely cross-modal CutMix (CMC), for implicit cross-modal alignment learning in unpaired VLP. Specifically, CMC transforms natural sentences in the textual view into a multi-modal view, where visually-grounded words in a sentence are randomly replaced by diverse image patches with similar semantics. There are several appealing proprieties of the proposed CMC. First, it enhances the data diversity while keeping the semantic meaning intact for tackling problems where the aligned data are scarce; Second, by attaching cross-modal noise on uni-modal data, it guides models to learn token-level interactions across modalities for better denoising. Furthermore, we present a new unpaired VLP method, dubbed as VLMixer, that integrates CMC with contrastive learning to pull together the uni-modal and multi-modal views for better instance-level alignments among different modalities. Extensive experiments on five downstream tasks show that VLMixer could surpass previous state-of-the-art unpaired VLP methods.

----

## [993] DynaMixer: A Vision MLP Architecture with Dynamic Mixing

**Authors**: *Ziyu Wang, Wenhao Jiang, Yiming Zhu, Li Yuan, Yibing Song, Wei Liu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22i.html](https://proceedings.mlr.press/v162/wang22i.html)

**Abstract**:

Recently, MLP-like vision models have achieved promising performances on mainstream visual recognition tasks. In contrast with vision transformers and CNNs, the success of MLP-like models shows that simple information fusion operations among tokens and channels can yield a good representation power for deep recognition models. However, existing MLP-like models fuse tokens through static fusion operations, lacking adaptability to the contents of the tokens to be mixed. Thus, customary information fusion procedures are not effective enough. To this end, this paper presents an efficient MLP-like network architecture, dubbed DynaMixer, resorting to dynamic information fusion. Critically, we propose a procedure, on which the DynaMixer model relies, to dynamically generate mixing matrices by leveraging the contents of all the tokens to be mixed. To reduce the time complexity and improve the robustness, a dimensionality reduction technique and a multi-segment fusion mechanism are adopted. Our proposed DynaMixer model (97M parameters) achieves 84.3% top-1 accuracy on the ImageNet-1K dataset without extra training data, performing favorably against the state-of-the-art vision MLP models. When the number of parameters is reduced to 26M, it still achieves 82.7% top-1 accuracy, surpassing the existing MLP-like models with a similar capacity. The code is available at \url{https://github.com/ziyuwwang/DynaMixer}.

----

## [994] Improving Screening Processes via Calibrated Subset Selection

**Authors**: *Lequn Wang, Thorsten Joachims, Manuel Gomez Rodriguez*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22j.html](https://proceedings.mlr.press/v162/wang22j.html)

**Abstract**:

Many selection processes such as finding patients qualifying for a medical trial or retrieval pipelines in search engines consist of multiple stages, where an initial screening stage focuses the resources on shortlisting the most promising candidates. In this paper, we investigate what guarantees a screening classifier can provide, independently of whether it is constructed manually or trained. We find that current solutions do not enjoy distribution-free theoretical guarantees and we show that, in general, even for a perfectly calibrated classifier, there always exist specific pools of candidates for which its shortlist is suboptimal. Then, we develop a distribution-free screening algorithm—called Calibrated Subsect Selection (CSS)—that, given any classifier and some amount of calibration data, finds near-optimal shortlists of candidates that contain a desired number of qualified candidates in expectation. Moreover, we show that a variant of CSS that calibrates a given classifier multiple times across specific groups can create shortlists with provable diversity guarantees. Experiments on US Census survey data validate our theoretical results and show that the shortlists provided by our algorithm are superior to those provided by several competitive baselines.

----

## [995] The Geometry of Robust Value Functions

**Authors**: *Kaixin Wang, Navdeep Kumar, Kuangqi Zhou, Bryan Hooi, Jiashi Feng, Shie Mannor*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22k.html](https://proceedings.mlr.press/v162/wang22k.html)

**Abstract**:

The space of value functions is a fundamental concept in reinforcement learning. Characterizing its geometric properties may provide insights for optimization and representation. Existing works mainly focus on the value space for Markov Decision Processes (MDPs). In this paper, we study the geometry of the robust value space for the more general Robust MDPs (RMDPs) setting, where transition uncertainties are considered. Specifically, since we find it hard to directly adapt prior approaches to RMDPs, we start with revisiting the non-robust case, and introduce a new perspective that enables us to characterize both the non-robust and robust value space in a similar fashion. The key of this perspective is to decompose the value space, in a state-wise manner, into unions of hypersurfaces. Through our analysis, we show that the robust value space is determined by a set of conic hypersurfaces, each of which contains the robust values of all policies that agree on one state. Furthermore, we find that taking only extreme points in the uncertainty set is sufficient to determine the robust value space. Finally, we discuss some other aspects about the robust value space, including its non-convexity and policy agreement on multiple states.

----

## [996] What Dense Graph Do You Need for Self-Attention?

**Authors**: *Yuxin Wang, Chu-Tak Lee, Qipeng Guo, Zhangyue Yin, Yunhua Zhou, Xuanjing Huang, Xipeng Qiu*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22l.html](https://proceedings.mlr.press/v162/wang22l.html)

**Abstract**:

Transformers have made progress in miscellaneous tasks, but suffer from quadratic computational and memory complexities. Recent works propose sparse transformers with attention on sparse graphs to reduce complexity and remain strong performance. While effective, the crucial parts of how dense a graph needs to be to perform well are not fully explored. In this paper, we propose Normalized Information Payload (NIP), a graph scoring function measuring information transfer on graph, which provides an analysis tool for trade-offs between performance and complexity. Guided by this theoretical analysis, we present Hypercube Transformer, a sparse transformer that models token interactions in a hypercube and shows comparable or even better results with vanilla transformer while yielding $O(N\log N)$ complexity with sequence length $N$. Experiments on tasks requiring various sequence lengths lay validation for our graph function well.

----

## [997] Improved Certified Defenses against Data Poisoning with (Deterministic) Finite Aggregation

**Authors**: *Wenxiao Wang, Alexander Levine, Soheil Feizi*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22m.html](https://proceedings.mlr.press/v162/wang22m.html)

**Abstract**:

Data poisoning attacks aim at manipulating model behaviors through distorting training data. Previously, an aggregation-based certified defense, Deep Partition Aggregation (DPA), was proposed to mitigate this threat. DPA predicts through an aggregation of base classifiers trained on disjoint subsets of data, thus restricting its sensitivity to dataset distortions. In this work, we propose an improved certified defense against general poisoning attacks, namely Finite Aggregation. In contrast to DPA, which directly splits the training set into disjoint subsets, our method first splits the training set into smaller disjoint subsets and then combines duplicates of them to build larger (but not disjoint) subsets for training base classifiers. This reduces the worst-case impacts of poison samples and thus improves certified robustness bounds. In addition, we offer an alternative view of our method, bridging the designs of deterministic and stochastic aggregation-based certified defenses. Empirically, our proposed Finite Aggregation consistently improves certificates on MNIST, CIFAR-10, and GTSRB, boosting certified fractions by up to 3.05%, 3.87% and 4.77%, respectively, while keeping the same clean accuracies as DPA’s, effectively establishing a new state of the art in (pointwise) certified robustness against data poisoning.

----

## [998] Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path and Beyond

**Authors**: *Haoxiang Wang, Bo Li, Han Zhao*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22n.html](https://proceedings.mlr.press/v162/wang22n.html)

**Abstract**:

The vast majority of existing algorithms for unsupervised domain adaptation (UDA) focus on adapting from a labeled source domain to an unlabeled target domain directly in a one-off way. Gradual domain adaptation (GDA), on the other hand, assumes a path of $(T-1)$ unlabeled intermediate domains bridging the source and target, and aims to provide better generalization in the target domain by leveraging the intermediate ones. Under certain assumptions, Kumar et al. (2020) proposed a simple algorithm, Gradual Self-Training, along with a generalization bound in the order of $e^{O(T)} \left(\varepsilon_0+O\left(\sqrt{log(T)/n}\right)\right)$ for the target domain error, where $\varepsilon_0$ is the source domain error and $n$ is the data size of each domain. Due to the exponential factor, this upper bound becomes vacuous when $T$ is only moderately large. In this work, we analyze gradual self-training under more general and relaxed assumptions, and prove a significantly improved generalization bound as $\widetilde{O}\left(\varepsilon_0 + T\Delta + T/\sqrt{n} + 1/\sqrt{nT}\right)$, where $\Delta$ is the average distributional distance between consecutive domains. Compared with the existing bound with an exponential dependency on $T$ as a multiplicative factor, our bound only depends on $T$ linearly and additively. Perhaps more interestingly, our result implies the existence of an optimal choice of $T$ that minimizes the generalization error, and it also naturally suggests an optimal way to construct the path of intermediate domains so as to minimize the accumulative path length $T\Delta$ between the source and target. To corroborate the implications of our theory, we examine gradual self-training on multiple semi-synthetic and real datasets, which confirms our findings. We believe our insights provide a path forward toward the design of future GDA algorithms.

----

## [999] Communication-Efficient Adaptive Federated Learning

**Authors**: *Yujia Wang, Lu Lin, Jinghui Chen*

**Conference**: *icml 2022*

**URL**: [https://proceedings.mlr.press/v162/wang22o.html](https://proceedings.mlr.press/v162/wang22o.html)

**Abstract**:

Federated learning is a machine learning training paradigm that enables clients to jointly train models without sharing their own localized data. However, the implementation of federated learning in practice still faces numerous challenges, such as the large communication overhead due to the repetitive server-client synchronization and the lack of adaptivity by SGD-based model updates. Despite that various methods have been proposed for reducing the communication cost by gradient compression or quantization, and the federated versions of adaptive optimizers such as FedAdam are proposed to add more adaptivity, the current federated learning framework still cannot solve the aforementioned challenges all at once. In this paper, we propose a novel communication-efficient adaptive federated learning method (FedCAMS) with theoretical convergence guarantees. We show that in the nonconvex stochastic optimization setting, our proposed FedCAMS achieves the same convergence rate of $O(\frac{1}{\sqrt{TKm}})$ as its non-compressed counterparts. Extensive experiments on various benchmarks verify our theoretical analysis.

----



[Go to the previous page](ICML-2022-list04.md)

[Go to the next page](ICML-2022-list06.md)

[Go to the catalog section](README.md)