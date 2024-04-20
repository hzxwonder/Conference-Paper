## [0] A New Representation of Successor Features for Transfer across Dissimilar Environments

**Authors**: *Majid Abdolshah, Hung Le, Thommen George Karimpanal, Sunil Gupta, Santu Rana, Svetha Venkatesh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/abdolshah21a.html](http://proceedings.mlr.press/v139/abdolshah21a.html)

**Abstract**:

Transfer in reinforcement learning is usually achieved through generalisation across tasks. Whilst many studies have investigated transferring knowledge when the reward function changes, they have assumed that the dynamics of the environments remain consistent. Many real-world RL problems require transfer among environments with different dynamics. To address this problem, we propose an approach based on successor features in which we model successor feature functions with Gaussian Processes permitting the source successor features to be treated as noisy measurements of the target successor feature function. Our theoretical analysis proves the convergence of this approach as well as the bounded error on modelling successor feature functions with Gaussian Processes in environments with both different dynamics and rewards. We demonstrate our method on benchmark datasets and show that it outperforms current baselines.

----

## [1] Massively Parallel and Asynchronous Tsetlin Machine Architecture Supporting Almost Constant-Time Scaling

**Authors**: *Kuruge Darshana Abeyrathna, Bimal Bhattarai, Morten Goodwin, Saeed Rahimi Gorji, Ole-Christoffer Granmo, Lei Jiao, Rupsa Saha, Rohan Kumar Yadav*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/abeyrathna21a.html](http://proceedings.mlr.press/v139/abeyrathna21a.html)

**Abstract**:

Using logical clauses to represent patterns, Tsetlin Machine (TM) have recently obtained competitive performance in terms of accuracy, memory footprint, energy, and learning speed on several benchmarks. Each TM clause votes for or against a particular class, with classification resolved using a majority vote. While the evaluation of clauses is fast, being based on binary operators, the voting makes it necessary to synchronize the clause evaluation, impeding parallelization. In this paper, we propose a novel scheme for desynchronizing the evaluation of clauses, eliminating the voting bottleneck. In brief, every clause runs in its own thread for massive native parallelism. For each training example, we keep track of the class votes obtained from the clauses in local voting tallies. The local voting tallies allow us to detach the processing of each clause from the rest of the clauses, supporting decentralized learning. This means that the TM most of the time will operate on outdated voting tallies. We evaluated the proposed parallelization across diverse learning tasks and it turns out that our decentralized TM learning algorithm copes well with working on outdated data, resulting in no significant loss in learning accuracy. Furthermore, we show that the approach provides up to 50 times faster learning. Finally, learning time is almost constant for reasonable clause amounts (employing from 20 to 7,000 clauses on a Tesla V100 GPU). For sufficiently large clause numbers, computation time increases approximately proportionally. Our parallel and asynchronous architecture thus allows processing of more massive datasets and operating with more clauses for higher accuracy.

----

## [2] Debiasing Model Updates for Improving Personalized Federated Training

**Authors**: *Durmus Alp Emre Acar, Yue Zhao, Ruizhao Zhu, Ramon Matas Navarro, Matthew Mattina, Paul N. Whatmough, Venkatesh Saligrama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/acar21a.html](http://proceedings.mlr.press/v139/acar21a.html)

**Abstract**:

We propose a novel method for federated learning that is customized specifically to the objective of a given edge device. In our proposed method, a server trains a global meta-model by collaborating with devices without actually sharing data. The trained global meta-model is then personalized locally by each device to meet its specific objective. Different from the conventional federated learning setting, training customized models for each device is hindered by both the inherent data biases of the various devices, as well as the requirements imposed by the federated architecture. We propose gradient correction methods leveraging prior works, and explicitly de-bias the meta-model in the distributed heterogeneous data setting to learn personalized device models. We present convergence guarantees of our method for strongly convex, convex and nonconvex meta objectives. We empirically evaluate the performance of our method on benchmark datasets and demonstrate significant communication savings.

----

## [3] Memory Efficient Online Meta Learning

**Authors**: *Durmus Alp Emre Acar, Ruizhao Zhu, Venkatesh Saligrama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/acar21b.html](http://proceedings.mlr.press/v139/acar21b.html)

**Abstract**:

We propose a novel algorithm for online meta learning where task instances are sequentially revealed with limited supervision and a learner is expected to meta learn them in each round, so as to allow the learner to customize a task-specific model rapidly with little task-level supervision. A fundamental concern arising in online meta-learning is the scalability of memory as more tasks are viewed over time. Heretofore, prior works have allowed for perfect recall leading to linear increase in memory with time. Different from prior works, in our method, prior task instances are allowed to be deleted. We propose to leverage prior task instances by means of a fixed-size state-vector, which is updated sequentially. Our theoretical analysis demonstrates that our proposed memory efficient online learning (MOML) method suffers sub-linear regret with convex loss functions and sub-linear local regret for nonconvex losses. On benchmark datasets we show that our method can outperform prior works even though they allow for perfect recall.

----

## [4] Robust Testing and Estimation under Manipulation Attacks

**Authors**: *Jayadev Acharya, Ziteng Sun, Huanyu Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/acharya21a.html](http://proceedings.mlr.press/v139/acharya21a.html)

**Abstract**:

We study robust testing and estimation of discrete distributions in the strong contamination model. Our results cover both centralized setting and distributed setting with general local information constraints including communication and LDP constraints. Our technique relates the strength of manipulation attacks to the earth-mover distance using Hamming distance as the metric between messages (samples) from the users. In the centralized setting, we provide optimal error bounds for both learning and testing. Our lower bounds under local information constraints build on the recent lower bound methods in distributed inference. In the communication constrained setting, we develop novel algorithms based on random hashing and an L1-L1 isometry.

----

## [5] GP-Tree: A Gaussian Process Classifier for Few-Shot Incremental Learning

**Authors**: *Idan Achituve, Aviv Navon, Yochai Yemini, Gal Chechik, Ethan Fetaya*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/achituve21a.html](http://proceedings.mlr.press/v139/achituve21a.html)

**Abstract**:

Gaussian processes (GPs) are non-parametric, flexible, models that work well in many tasks. Combining GPs with deep learning methods via deep kernel learning (DKL) is especially compelling due to the strong representational power induced by the network. However, inference in GPs, whether with or without DKL, can be computationally challenging on large datasets. Here, we propose GP-Tree, a novel method for multi-class classification with Gaussian processes and DKL. We develop a tree-based hierarchical model in which each internal node of the tree fits a GP to the data using the P{รณ}lya-Gamma augmentation scheme. As a result, our method scales well with both the number of classes and data size. We demonstrate the effectiveness of our method against other Gaussian process training baselines, and we show how our general GP approach achieves improved accuracy on standard incremental few-shot learning benchmarks.

----

## [6] f-Domain Adversarial Learning: Theory and Algorithms

**Authors**: *David Acuna, Guojun Zhang, Marc T. Law, Sanja Fidler*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/acuna21a.html](http://proceedings.mlr.press/v139/acuna21a.html)

**Abstract**:

Unsupervised domain adaptation is used in many machine learning applications where, during training, a model has access to unlabeled data in the target domain, and a related labeled dataset. In this paper, we introduce a novel and general domain-adversarial framework. Specifically, we derive a novel generalization bound for domain adaptation that exploits a new measure of discrepancy between distributions based on a variational characterization of f-divergences. It recovers the theoretical results from Ben-David et al. (2010a) as a special case and supports divergences used in practice. Based on this bound, we derive a new algorithmic framework that introduces a key correction in the original adversarial training method of Ganin et al. (2016). We show that many regularizers and ad-hoc objectives introduced over the last years in this framework are then not required to achieve performance comparable to (if not better than) state-of-the-art domain-adversarial methods. Experimental analysis conducted on real-world natural language and computer vision datasets show that our framework outperforms existing baselines, and obtains the best results for f-divergences that were not considered previously in domain-adversarial learning.

----

## [7] Towards Rigorous Interpretations: a Formalisation of Feature Attribution

**Authors**: *Darius Afchar, Vincent Guigue, Romain Hennequin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/afchar21a.html](http://proceedings.mlr.press/v139/afchar21a.html)

**Abstract**:

Feature attribution is often loosely presented as the process of selecting a subset of relevant features as a rationale of a prediction. Task-dependent by nature, precise definitions of "relevance" encountered in the literature are however not always consistent. This lack of clarity stems from the fact that we usually do not have access to any notion of ground-truth attribution and from a more general debate on what good interpretations are. In this paper we propose to formalise feature selection/attribution based on the concept of relaxed functional dependence. In particular, we extend our notions to the instance-wise setting and derive necessary properties for candidate selection solutions, while leaving room for task-dependence. By computing ground-truth attributions on synthetic datasets, we evaluate many state-of-the-art attribution methods and show that, even when optimised, some fail to verify the proposed properties and provide wrong solutions.

----

## [8] Acceleration via Fractal Learning Rate Schedules

**Authors**: *Naman Agarwal, Surbhi Goel, Cyril Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/agarwal21a.html](http://proceedings.mlr.press/v139/agarwal21a.html)

**Abstract**:

In practical applications of iterative first-order optimization, the learning rate schedule remains notoriously difficult to understand and expensive to tune. We demonstrate the presence of these subtleties even in the innocuous case when the objective is a convex quadratic. We reinterpret an iterative algorithm from the numerical analysis literature as what we call the Chebyshev learning rate schedule for accelerating vanilla gradient descent, and show that the problem of mitigating instability leads to a fractal ordering of step sizes. We provide some experiments to challenge conventional beliefs about stable learning rates in deep learning: the fractal schedule enables training to converge with locally unstable updates which make negative progress on the objective.

----

## [9] A Regret Minimization Approach to Iterative Learning Control

**Authors**: *Naman Agarwal, Elad Hazan, Anirudha Majumdar, Karan Singh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/agarwal21b.html](http://proceedings.mlr.press/v139/agarwal21b.html)

**Abstract**:

We consider the setting of iterative learning control, or model-based policy learning in the presence of uncertain, time-varying dynamics. In this setting, we propose a new performance metric, planning regret, which replaces the standard stochastic uncertainty assumptions with worst case regret. Based on recent advances in non-stochastic control, we design a new iterative algorithm for minimizing planning regret that is more robust to model mismatch and uncertainty. We provide theoretical and empirical evidence that the proposed algorithm outperforms existing methods on several benchmarks.

----

## [10] Towards the Unification and Robustness of Perturbation and Gradient Based Explanations

**Authors**: *Sushant Agarwal, Shahin Jabbari, Chirag Agarwal, Sohini Upadhyay, Steven Wu, Himabindu Lakkaraju*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/agarwal21c.html](http://proceedings.mlr.press/v139/agarwal21c.html)

**Abstract**:

As machine learning black boxes are increasingly being deployed in critical domains such as healthcare and criminal justice, there has been a growing emphasis on developing techniques for explaining these black boxes in a post hoc manner. In this work, we analyze two popular post hoc interpretation techniques: SmoothGrad which is a gradient based method, and a variant of LIME which is a perturbation based method. More specifically, we derive explicit closed form expressions for the explanations output by these two methods and show that they both converge to the same explanation in expectation, i.e., when the number of perturbed samples used by these methods is large. We then leverage this connection to establish other desirable properties, such as robustness, for these techniques. We also derive finite sample complexity bounds for the number of perturbations required for these methods to converge to their expected explanation. Finally, we empirically validate our theory using extensive experimentation on both synthetic and real-world datasets.

----

## [11] Label Inference Attacks from Log-loss Scores

**Authors**: *Abhinav Aggarwal, Shiva Prasad Kasiviswanathan, Zekun Xu, Oluwaseyi Feyisetan, Nathanael Teissier*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/aggarwal21a.html](http://proceedings.mlr.press/v139/aggarwal21a.html)

**Abstract**:

Log-loss (also known as cross-entropy loss) metric is ubiquitously used across machine learning applications to assess the performance of classification algorithms. In this paper, we investigate the problem of inferring the labels of a dataset from single (or multiple) log-loss score(s), without any other access to the dataset. Surprisingly, we show that for any finite number of label classes, it is possible to accurately infer the labels of the dataset from the reported log-loss score of a single carefully constructed prediction vector if we allow arbitrary precision arithmetic. Additionally, we present label inference algorithms (attacks) that succeed even under addition of noise to the log-loss scores and under limited precision arithmetic. All our algorithms rely on ideas from number theory and combinatorics and require no model training. We run experimental simulations on some real datasets to demonstrate the ease of running these attacks in practice.

----

## [12] Deep Kernel Processes

**Authors**: *Laurence Aitchison, Adam X. Yang, Sebastian W. Ober*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/aitchison21a.html](http://proceedings.mlr.press/v139/aitchison21a.html)

**Abstract**:

We define deep kernel processes in which positive definite Gram matrices are progressively transformed by nonlinear kernel functions and by sampling from (inverse) Wishart distributions. Remarkably, we find that deep Gaussian processes (DGPs), Bayesian neural networks (BNNs), infinite BNNs, and infinite BNNs with bottlenecks can all be written as deep kernel processes. For DGPs the equivalence arises because the Gram matrix formed by the inner product of features is Wishart distributed, and as we show, standard isotropic kernels can be written entirely in terms of this Gram matrix — we do not need knowledge of the underlying features. We define a tractable deep kernel process, the deep inverse Wishart process, and give a doubly-stochastic inducing-point variational inference scheme that operates on the Gram matrices, not on the features, as in DGPs. We show that the deep inverse Wishart process gives superior performance to DGPs and infinite BNNs on fully-connected baselines.

----

## [13] How Does Loss Function Affect Generalization Performance of Deep Learning? Application to Human Age Estimation

**Authors**: *Ali Akbari, Muhammad Awais, Manijeh Bashar, Josef Kittler*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/akbari21a.html](http://proceedings.mlr.press/v139/akbari21a.html)

**Abstract**:

Good generalization performance across a wide variety of domains caused by many external and internal factors is the fundamental goal of any machine learning algorithm. This paper theoretically proves that the choice of loss function matters for improving the generalization performance of deep learning-based systems. By deriving the generalization error bound for deep neural models trained by stochastic gradient descent, we pinpoint the characteristics of the loss function that is linked to the generalization error and can therefore be used for guiding the loss function selection process. In summary, our main statement in this paper is: choose a stable loss function, generalize better. Focusing on human age estimation from the face which is a challenging topic in computer vision, we then propose a novel loss function for this learning problem. We theoretically prove that the proposed loss function achieves stronger stability, and consequently a tighter generalization error bound, compared to the other common loss functions for this problem. We have supported our findings theoretically, and demonstrated the merits of the guidance process experimentally, achieving significant improvements.

----

## [14] On Learnability via Gradient Method for Two-Layer ReLU Neural Networks in Teacher-Student Setting

**Authors**: *Shunta Akiyama, Taiji Suzuki*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/akiyama21a.html](http://proceedings.mlr.press/v139/akiyama21a.html)

**Abstract**:

Deep learning empirically achieves high performance in many applications, but its training dynamics has not been fully understood theoretically. In this paper, we explore theoretical analysis on training two-layer ReLU neural networks in a teacher-student regression model, in which a student network learns an unknown teacher network through its outputs. We show that with a specific regularization and sufficient over-parameterization, the student network can identify the parameters of the teacher network with high probability via gradient descent with a norm dependent stepsize even though the objective function is highly non-convex. The key theoretical tool is the measure representation of the neural networks and a novel application of a dual certificate argument for sparse estimation on a measure space. We analyze the global minima and global convergence property in the measure space.

----

## [15] Slot Machines: Discovering Winning Combinations of Random Weights in Neural Networks

**Authors**: *Maxwell Mbabilla Aladago, Lorenzo Torresani*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/aladago21a.html](http://proceedings.mlr.press/v139/aladago21a.html)

**Abstract**:

In contrast to traditional weight optimization in a continuous space, we demonstrate the existence of effective random networks whose weights are never updated. By selecting a weight among a fixed set of random values for each individual connection, our method uncovers combinations of random weights that match the performance of traditionally-trained networks of the same capacity. We refer to our networks as "slot machines" where each reel (connection) contains a fixed set of symbols (random values). Our backpropagation algorithm "spins" the reels to seek "winning" combinations, i.e., selections of random weight values that minimize the given loss. Quite surprisingly, we find that allocating just a few random values to each connection (e.g., 8 values per connection) yields highly competitive combinations despite being dramatically more constrained compared to traditionally learned weights. Moreover, finetuning these combinations often improves performance over the trained baselines. A randomly initialized VGG-19 with 8 values per connection contains a combination that achieves 91% test accuracy on CIFAR-10. Our method also achieves an impressive performance of 98.2% on MNIST for neural networks containing only random weights.

----

## [16] A large-scale benchmark for few-shot program induction and synthesis

**Authors**: *Ferran Alet, Javier Lopez-Contreras, James Koppel, Maxwell I. Nye, Armando Solar-Lezama, Tomás Lozano-Pérez, Leslie Pack Kaelbling, Joshua B. Tenenbaum*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/alet21a.html](http://proceedings.mlr.press/v139/alet21a.html)

**Abstract**:

A landmark challenge for AI is to learn flexible, powerful representations from small numbers of examples. On an important class of tasks, hypotheses in the form of programs provide extreme generalization capabilities from surprisingly few examples. However, whereas large natural few-shot learning image benchmarks have spurred progress in meta-learning for deep networks, there is no comparably big, natural program-synthesis dataset that can play a similar role. This is because, whereas images are relatively easy to label from internet meta-data or annotated by non-experts, generating meaningful input-output examples for program induction has proven hard to scale. In this work, we propose a new way of leveraging unit tests and natural inputs for small programs as meaningful input-output examples for each sub-program of the overall program. This allows us to create a large-scale naturalistic few-shot program-induction benchmark and propose new challenges in this domain. The evaluation of multiple program induction and synthesis algorithms points to shortcomings of current methods and suggests multiple avenues for future work.

----

## [17] Robust Pure Exploration in Linear Bandits with Limited Budget

**Authors**: *Ayya Alieva, Ashok Cutkosky, Abhimanyu Das*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/alieva21a.html](http://proceedings.mlr.press/v139/alieva21a.html)

**Abstract**:

We consider the pure exploration problem in the fixed-budget linear bandit setting. We provide a new algorithm that identifies the best arm with high probability while being robust to unknown levels of observation noise as well as to moderate levels of misspecification in the linear model. Our technique combines prior approaches to pure exploration in the multi-armed bandit problem with optimal experimental design algorithms to obtain both problem dependent and problem independent bounds. Our success probability is never worse than that of an algorithm that ignores the linear structure, but seamlessly takes advantage of such structure when possible. Furthermore, we only need the number of samples to scale with the dimension of the problem rather than the number of arms. We complement our theoretical results with empirical validation.

----

## [18] Communication-Efficient Distributed Optimization with Quantized Preconditioners

**Authors**: *Foivos Alimisis, Peter Davies, Dan Alistarh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/alimisis21a.html](http://proceedings.mlr.press/v139/alimisis21a.html)

**Abstract**:

We investigate fast and communication-efficient algorithms for the classic problem of minimizing a sum of strongly convex and smooth functions that are distributed among $n$ different nodes, which can communicate using a limited number of bits. Most previous communication-efficient approaches for this problem are limited to first-order optimization, and therefore have \emph{linear} dependence on the condition number in their communication complexity. We show that this dependence is not inherent: communication-efficient methods can in fact have sublinear dependence on the condition number. For this, we design and analyze the first communication-efficient distributed variants of preconditioned gradient descent for Generalized Linear Models, and for Newton’s method. Our results rely on a new technique for quantizing both the preconditioner and the descent direction at each step of the algorithms, while controlling their convergence rate. We also validate our findings experimentally, showing faster convergence and reduced communication relative to previous methods.

----

## [19] Non-Exponentially Weighted Aggregation: Regret Bounds for Unbounded Loss Functions

**Authors**: *Pierre Alquier*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/alquier21a.html](http://proceedings.mlr.press/v139/alquier21a.html)

**Abstract**:

We tackle the problem of online optimization with a general, possibly unbounded, loss function. It is well known that when the loss is bounded, the exponentially weighted aggregation strategy (EWA) leads to a regret in $\sqrt{T}$ after $T$ steps. In this paper, we study a generalized aggregation strategy, where the weights no longer depend exponentially on the losses. Our strategy is based on Follow The Regularized Leader (FTRL): we minimize the expected losses plus a regularizer, that is here a $\phi$-divergence. When the regularizer is the Kullback-Leibler divergence, we obtain EWA as a special case. Using alternative divergences enables unbounded losses, at the cost of a worst regret bound in some cases.

----

## [20] Dataset Dynamics via Gradient Flows in Probability Space

**Authors**: *David Alvarez-Melis, Nicolò Fusi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/alvarez-melis21a.html](http://proceedings.mlr.press/v139/alvarez-melis21a.html)

**Abstract**:

Various machine learning tasks, from generative modeling to domain adaptation, revolve around the concept of dataset transformation and manipulation. While various methods exist for transforming unlabeled datasets, principled methods to do so for labeled (e.g., classification) datasets are missing. In this work, we propose a novel framework for dataset transformation, which we cast as optimization over data-generating joint probability distributions. We approach this class of problems through Wasserstein gradient flows in probability space, and derive practical and efficient particle-based methods for a flexible but well-behaved class of objective functions. Through various experiments, we show that this framework can be used to impose constraints on classification datasets, adapt them for transfer learning, or to re-purpose fixed or black-box models to classify {—}with high accuracy{—} previously unseen datasets.

----

## [21] Submodular Maximization subject to a Knapsack Constraint: Combinatorial Algorithms with Near-optimal Adaptive Complexity

**Authors**: *Georgios Amanatidis, Federico Fusco, Philip Lazos, Stefano Leonardi, Alberto Marchetti-Spaccamela, Rebecca Reiffenhäuser*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/amanatidis21a.html](http://proceedings.mlr.press/v139/amanatidis21a.html)

**Abstract**:

The growing need to deal with massive instances motivates the design of algorithms balancing the quality of the solution with applicability. For the latter, an important measure is the \emph{adaptive complexity}, capturing the number of sequential rounds of parallel computation needed. In this work we obtain the first \emph{constant factor} approximation algorithm for non-monotone submodular maximization subject to a knapsack constraint with \emph{near-optimal} $O(\log n)$ adaptive complexity. Low adaptivity by itself, however, is not enough: one needs to account for the total number of function evaluations (or value queries) as well. Our algorithm asks $\tilde{O}(n^2)$ value queries, but can be modified to run with only $\tilde{O}(n)$ instead, while retaining a low adaptive complexity of $O(\log^2n)$. Besides the above improvement in adaptivity, this is also the first \emph{combinatorial} approach with sublinear adaptive complexity for the problem and yields algorithms comparable to the state-of-the-art even for the special cases of cardinality constraints or monotone objectives. Finally, we showcase our algorithms’ applicability on real-world datasets.

----

## [22] Safe Reinforcement Learning with Linear Function Approximation

**Authors**: *Sanae Amani, Christos Thrampoulidis, Lin Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/amani21a.html](http://proceedings.mlr.press/v139/amani21a.html)

**Abstract**:

Safety in reinforcement learning has become increasingly important in recent years. Yet, existing solutions either fail to strictly avoid choosing unsafe actions, which may lead to catastrophic results in safety-critical systems, or fail to provide regret guarantees for settings where safety constraints need to be learned. In this paper, we address both problems by first modeling safety as an unknown linear cost function of states and actions, which must always fall below a certain threshold. We then present algorithms, termed SLUCB-QVI and RSLUCB-QVI, for episodic Markov decision processes (MDPs) with linear function approximation. We show that SLUCB-QVI and RSLUCB-QVI, while with \emph{no safety violation}, achieve a $\tilde{\mathcal{O}}\left(\kappa\sqrt{d^3H^3T}\right)$ regret, nearly matching that of state-of-the-art unsafe algorithms, where $H$ is the duration of each episode, $d$ is the dimension of the feature mapping, $\kappa$ is a constant characterizing the safety constraints, and $T$ is the total number of action plays. We further present numerical simulations that corroborate our theoretical findings.

----

## [23] Automatic variational inference with cascading flows

**Authors**: *Luca Ambrogioni, Gianluigi Silvestri, Marcel van Gerven*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ambrogioni21a.html](http://proceedings.mlr.press/v139/ambrogioni21a.html)

**Abstract**:

The automation of probabilistic reasoning is one of the primary aims of machine learning. Recently, the confluence of variational inference and deep learning has led to powerful and flexible automatic inference methods that can be trained by stochastic gradient descent. In particular, normalizing flows are highly parameterized deep models that can fit arbitrarily complex posterior densities. However, normalizing flows struggle in highly structured probabilistic programs as they need to relearn the forward-pass of the program. Automatic structured variational inference (ASVI) remedies this problem by constructing variational programs that embed the forward-pass. Here, we combine the flexibility of normalizing flows and the prior-embedding property of ASVI in a new family of variational programs, which we named cascading flows. A cascading flows program interposes a newly designed highway flow architecture in between the conditional distributions of the prior program such as to steer it toward the observed data. These programs can be constructed automatically from an input probabilistic program and can also be amortized automatically. We evaluate the performance of the new variational programs in a series of structured inference problems. We find that cascading flows have much higher performance than both normalizing flows and ASVI in a large set of structured inference problems.

----

## [24] Sparse Bayesian Learning via Stepwise Regression

**Authors**: *Sebastian E. Ament, Carla P. Gomes*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ament21a.html](http://proceedings.mlr.press/v139/ament21a.html)

**Abstract**:

Sparse Bayesian Learning (SBL) is a powerful framework for attaining sparsity in probabilistic models. Herein, we propose a coordinate ascent algorithm for SBL termed Relevance Matching Pursuit (RMP) and show that, as its noise variance parameter goes to zero, RMP exhibits a surprising connection to Stepwise Regression. Further, we derive novel guarantees for Stepwise Regression algorithms, which also shed light on RMP. Our guarantees for Forward Regression improve on deterministic and probabilistic results for Orthogonal Matching Pursuit with noise. Our analysis of Backward Regression culminates in a bound on the residual of the optimal solution to the subset selection problem that, if satisfied, guarantees the optimality of the result. To our knowledge, this bound is the first that can be computed in polynomial time and depends chiefly on the smallest singular value of the matrix. We report numerical experiments using a variety of feature selection algorithms. Notably, RMP and its limiting variant are both efficient and maintain strong performance with correlated features.

----

## [25] Locally Persistent Exploration in Continuous Control Tasks with Sparse Rewards

**Authors**: *Susan Amin, Maziar Gomrokchi, Hossein Aboutalebi, Harsh Satija, Doina Precup*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/amin21a.html](http://proceedings.mlr.press/v139/amin21a.html)

**Abstract**:

A major challenge in reinforcement learning is the design of exploration strategies, especially for environments with sparse reward structures and continuous state and action spaces. Intuitively, if the reinforcement signal is very scarce, the agent should rely on some form of short-term memory in order to cover its environment efficiently. We propose a new exploration method, based on two intuitions: (1) the choice of the next exploratory action should depend not only on the (Markovian) state of the environment, but also on the agent’s trajectory so far, and (2) the agent should utilize a measure of spread in the state space to avoid getting stuck in a small region. Our method leverages concepts often used in statistical physics to provide explanations for the behavior of simplified (polymer) chains in order to generate persistent (locally self-avoiding) trajectories in state space. We discuss the theoretical properties of locally self-avoiding walks and their ability to provide a kind of short-term memory through a decaying temporal correlation within the trajectory. We provide empirical evaluations of our approach in a simulated 2D navigation task, as well as higher-dimensional MuJoCo continuous control locomotion tasks with sparse rewards.

----

## [26] Preferential Temporal Difference Learning

**Authors**: *Nishanth V. Anand, Doina Precup*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/anand21a.html](http://proceedings.mlr.press/v139/anand21a.html)

**Abstract**:

Temporal-Difference (TD) learning is a general and very useful tool for estimating the value function of a given policy, which in turn is required to find good policies. Generally speaking, TD learning updates states whenever they are visited. When the agent lands in a state, its value can be used to compute the TD-error, which is then propagated to other states. However, it may be interesting, when computing updates, to take into account other information than whether a state is visited or not. For example, some states might be more important than others (such as states which are frequently seen in a successful trajectory). Or, some states might have unreliable value estimates (for example, due to partial observability or lack of data), making their values less desirable as targets. We propose an approach to re-weighting states used in TD updates, both when they are the input and when they provide the target for the update. We prove that our approach converges with linear function approximation and illustrate its desirable empirical behaviour compared to other TD-style methods.

----

## [27] Unitary Branching Programs: Learnability and Lower Bounds

**Authors**: *Fidel Ernesto Diaz Andino, Maria Kokkou, Mateus de Oliveira Oliveira, Farhad Vadiee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/andino21a.html](http://proceedings.mlr.press/v139/andino21a.html)

**Abstract**:

Bounded width branching programs are a formalism that can be used to capture the notion of non-uniform constant-space computation. In this work, we study a generalized version of bounded width branching programs where instructions are defined by unitary matrices of bounded dimension. We introduce a new learning framework for these branching programs that leverages on a combination of local search techniques with gradient descent over Riemannian manifolds. We also show that gapped, read-once branching programs of bounded dimension can be learned with a polynomial number of queries in the presence of a teacher. Finally, we provide explicit near-quadratic size lower-bounds for bounded-dimension unitary branching programs, and exponential size lower-bounds for bounded-dimension read-once gapped unitary branching programs. The first lower bound is proven using a combination of Neciporuk’s lower bound technique with classic results from algebraic geometry. The second lower bound is proven within the framework of communication complexity theory.

----

## [28] The Logical Options Framework

**Authors**: *Brandon Araki, Xiao Li, Kiran Vodrahalli, Jonathan A. DeCastro, Micah J. Fry, Daniela Rus*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/araki21a.html](http://proceedings.mlr.press/v139/araki21a.html)

**Abstract**:

Learning composable policies for environments with complex rules and tasks is a challenging problem. We introduce a hierarchical reinforcement learning framework called the Logical Options Framework (LOF) that learns policies that are satisfying, optimal, and composable. LOF efficiently learns policies that satisfy tasks by representing the task as an automaton and integrating it into learning and planning. We provide and prove conditions under which LOF will learn satisfying, optimal policies. And lastly, we show how LOF’s learned policies can be composed to satisfy unseen tasks with only 10-50 retraining steps on our benchmarks. We evaluate LOF on four tasks in discrete and continuous domains, including a 3D pick-and-place environment.

----

## [29] Annealed Flow Transport Monte Carlo

**Authors**: *Michael Arbel, Alexander G. de G. Matthews, Arnaud Doucet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/arbel21a.html](http://proceedings.mlr.press/v139/arbel21a.html)

**Abstract**:

Annealed Importance Sampling (AIS) and its Sequential Monte Carlo (SMC) extensions are state-of-the-art methods for estimating normalizing constants of probability distributions. We propose here a novel Monte Carlo algorithm, Annealed Flow Transport (AFT), that builds upon AIS and SMC and combines them with normalizing flows (NFs) for improved performance. This method transports a set of particles using not only importance sampling (IS), Markov chain Monte Carlo (MCMC) and resampling steps - as in SMC, but also relies on NFs which are learned sequentially to push particles towards the successive annealed targets. We provide limit theorems for the resulting Monte Carlo estimates of the normalizing constant and expectations with respect to the target distribution. Additionally, we show that a continuous-time scaling limit of the population version of AFT is given by a Feynman–Kac measure which simplifies to the law of a controlled diffusion for expressive NFs. We demonstrate experimentally the benefits and limitations of our methodology on a variety of applications.

----

## [30] Permutation Weighting

**Authors**: *David Arbour, Drew Dimmery, Arjun Sondhi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/arbour21a.html](http://proceedings.mlr.press/v139/arbour21a.html)

**Abstract**:

A commonly applied approach for estimating causal effects from observational data is to apply weights which render treatments independent of observed pre-treatment covariates. Recently emphasis has been placed on deriving balancing weights which explicitly target this independence condition. In this work we introduce permutation weighting, a method for estimating balancing weights using a standard binary classifier (regardless of cardinality of treatment). A large class of probabilistic classifiers may be used in this method; the choice of loss for the classifier implies the particular definition of balance. We bound bias and variance in terms of the excess risk of the classifier, show that these disappear asymptotically, and demonstrate that our classification problem directly minimizes imbalance. Additionally, hyper-parameter tuning and model selection can be performed with standard cross-validation methods. Empirical evaluations indicate that permutation weighting provides favorable performance in comparison to existing methods.

----

## [31] Analyzing the tree-layer structure of Deep Forests

**Authors**: *Ludovic Arnould, Claire Boyer, Erwan Scornet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/arnould21a.html](http://proceedings.mlr.press/v139/arnould21a.html)

**Abstract**:

Random forests on the one hand, and neural networks on the other hand, have met great success in the machine learning community for their predictive performance. Combinations of both have been proposed in the literature, notably leading to the so-called deep forests (DF) (Zhou & Feng,2019). In this paper, our aim is not to benchmark DF performances but to investigate instead their underlying mechanisms. Additionally, DF architecture can be generally simplified into more simple and computationally efficient shallow forest networks. Despite some instability, the latter may outperform standard predictive tree-based methods. We exhibit a theoretical framework in which a shallow tree network is shown to enhance the performance of classical decision trees. In such a setting, we provide tight theoretical lower and upper bounds on its excess risk. These theoretical results show the interest of tree-network architectures for well-structured data provided that the first layer, acting as a data encoder, is rich enough.

----

## [32] Dropout: Explicit Forms and Capacity Control

**Authors**: *Raman Arora, Peter L. Bartlett, Poorya Mianjy, Nathan Srebro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/arora21a.html](http://proceedings.mlr.press/v139/arora21a.html)

**Abstract**:

We investigate the capacity control provided by dropout in various machine learning problems. First, we study dropout for matrix completion, where it induces a distribution-dependent regularizer that equals the weighted trace-norm of the product of the factors. In deep learning, we show that the distribution-dependent regularizer due to dropout directly controls the Rademacher complexity of the underlying class of deep neural networks. These developments enable us to give concrete generalization error bounds for the dropout algorithm in both matrix completion as well as training deep neural networks.

----

## [33] Tighter Bounds on the Log Marginal Likelihood of Gaussian Process Regression Using Conjugate Gradients

**Authors**: *Artem Artemev, David R. Burt, Mark van der Wilk*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/artemev21a.html](http://proceedings.mlr.press/v139/artemev21a.html)

**Abstract**:

We propose a lower bound on the log marginal likelihood of Gaussian process regression models that can be computed without matrix factorisation of the full kernel matrix. We show that approximate maximum likelihood learning of model parameters by maximising our lower bound retains many benefits of the sparse variational approach while reducing the bias introduced into hyperparameter learning. The basis of our bound is a more careful analysis of the log-determinant term appearing in the log marginal likelihood, as well as using the method of conjugate gradients to derive tight lower bounds on the term involving a quadratic form. Our approach is a step forward in unifying methods relying on lower bound maximisation (e.g. variational methods) and iterative approaches based on conjugate gradients for training Gaussian processes. In experiments, we show improved predictive performance with our model for a comparable amount of training time compared to other conjugate gradient based approaches.

----

## [34] Deciding What to Learn: A Rate-Distortion Approach

**Authors**: *Dilip Arumugam, Benjamin Van Roy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/arumugam21a.html](http://proceedings.mlr.press/v139/arumugam21a.html)

**Abstract**:

Agents that learn to select optimal actions represent a prominent focus of the sequential decision-making literature. In the face of a complex environment or constraints on time and resources, however, aiming to synthesize such an optimal policy can become infeasible. These scenarios give rise to an important trade-off between the information an agent must acquire to learn and the sub-optimality of the resulting policy. While an agent designer has a preference for how this trade-off is resolved, existing approaches further require that the designer translate these preferences into a fixed learning target for the agent. In this work, leveraging rate-distortion theory, we automate this process such that the designer need only express their preferences via a single hyperparameter and the agent is endowed with the ability to compute its own learning targets that best achieve the desired trade-off. We establish a general bound on expected discounted regret for an agent that decides what to learn in this manner along with computational experiments that illustrate the expressiveness of designer preferences and even show improvements over Thompson sampling in identifying an optimal policy.

----

## [35] Private Adaptive Gradient Methods for Convex Optimization

**Authors**: *Hilal Asi, John C. Duchi, Alireza Fallah, Omid Javidbakht, Kunal Talwar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/asi21a.html](http://proceedings.mlr.press/v139/asi21a.html)

**Abstract**:

We study adaptive methods for differentially private convex optimization, proposing and analyzing differentially private variants of a Stochastic Gradient Descent (SGD) algorithm with adaptive stepsizes, as well as the AdaGrad algorithm. We provide upper bounds on the regret of both algorithms and show that the bounds are (worst-case) optimal. As a consequence of our development, we show that our private versions of AdaGrad outperform adaptive SGD, which in turn outperforms traditional SGD in scenarios with non-isotropic gradients where (non-private) Adagrad provably outperforms SGD. The major challenge is that the isotropic noise typically added for privacy dominates the signal in gradient geometry for high-dimensional problems; approaches to this that effectively optimize over lower-dimensional subspaces simply ignore the actual problems that varying gradient geometries introduce. In contrast, we study non-isotropic clipping and noise addition, developing a principled theoretical approach; the consequent procedures also enjoy significantly stronger empirical performance than prior approaches.

----

## [36] Private Stochastic Convex Optimization: Optimal Rates in L1 Geometry

**Authors**: *Hilal Asi, Vitaly Feldman, Tomer Koren, Kunal Talwar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/asi21b.html](http://proceedings.mlr.press/v139/asi21b.html)

**Abstract**:

Stochastic convex optimization over an $\ell_1$-bounded domain is ubiquitous in machine learning applications such as LASSO but remains poorly understood when learning with differential privacy. We show that, up to logarithmic factors the optimal excess population loss of any $(\epsilon,\delta)$-differentially private optimizer is $\sqrt{\log(d)/n} + \sqrt{d}/\epsilon n.$ The upper bound is based on a new algorithm that combines the iterative localization approach of Feldman et al. (2020) with a new analysis of private regularized mirror descent. It applies to $\ell_p$ bounded domains for $p\in [1,2]$ and queries at most $n^{3/2}$ gradients improving over the best previously known algorithm for the $\ell_2$ case which needs $n^2$ gradients. Further, we show that when the loss functions satisfy additional smoothness assumptions, the excess loss is upper bounded (up to logarithmic factors) by $\sqrt{\log(d)/n} + (\log(d)/\epsilon n)^{2/3}.$ This bound is achieved by a new variance-reduced version of the Frank-Wolfe algorithm that requires just a single pass over the data. We also show that the lower bound in this case is the minimum of the two rates mentioned above.

----

## [37] Combinatorial Blocking Bandits with Stochastic Delays

**Authors**: *Alexia Atsidakou, Orestis Papadigenopoulos, Soumya Basu, Constantine Caramanis, Sanjay Shakkottai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/atsidakou21a.html](http://proceedings.mlr.press/v139/atsidakou21a.html)

**Abstract**:

Recent work has considered natural variations of the {\em multi-armed bandit} problem, where the reward distribution of each arm is a special function of the time passed since its last pulling. In this direction, a simple (yet widely applicable) model is that of {\em blocking bandits}, where an arm becomes unavailable for a deterministic number of rounds after each play. In this work, we extend the above model in two directions: (i) We consider the general combinatorial setting where more than one arms can be played at each round, subject to feasibility constraints. (ii) We allow the blocking time of each arm to be stochastic. We first study the computational/unconditional hardness of the above setting and identify the necessary conditions for the problem to become tractable (even in an approximate sense). Based on these conditions, we provide a tight analysis of the approximation guarantee of a natural greedy heuristic that always plays the maximum expected reward feasible subset among the available (non-blocked) arms. When the arms’ expected rewards are unknown, we adapt the above heuristic into a bandit algorithm, based on UCB, for which we provide sublinear (approximate) regret guarantees, matching the theoretical lower bounds in the limiting case of absence of delays.

----

## [38] Dichotomous Optimistic Search to Quantify Human Perception

**Authors**: *Julien Audiffren*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/audiffren21a.html](http://proceedings.mlr.press/v139/audiffren21a.html)

**Abstract**:

In this paper we address a variant of the continuous multi-armed bandits problem, called the threshold estimation problem, which is at the heart of many psychometric experiments. Here, the objective is to estimate the sensitivity threshold for an unknown psychometric function Psi, which is assumed to be non decreasing and continuous. Our algorithm, Dichotomous Optimistic Search (DOS), efficiently solves this task by taking inspiration from hierarchical multi-armed bandits and Black-box optimization. Compared to previous approaches, DOS is model free and only makes minimal assumption on Psi smoothness, while having strong theoretical guarantees that compares favorably to recent methods from both Psychophysics and Global Optimization. We also empirically evaluate DOS and show that it significantly outperforms these methods, both in experiments that mimics the conduct of a psychometric experiment, and in tests with large pulls budgets that illustrate the faster convergence rate.

----

## [39] Federated Learning under Arbitrary Communication Patterns

**Authors**: *Dmitrii Avdiukhin, Shiva Prasad Kasiviswanathan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/avdiukhin21a.html](http://proceedings.mlr.press/v139/avdiukhin21a.html)

**Abstract**:

Federated Learning is a distributed learning setting where the goal is to train a centralized model with training data distributed over a large number of heterogeneous clients, each with unreliable and relatively slow network connections. A common optimization approach used in federated learning is based on the idea of local SGD: each client runs some number of SGD steps locally and then the updated local models are averaged to form the updated global model on the coordinating server. In this paper, we investigate the performance of an asynchronous version of local SGD wherein the clients can communicate with the server at arbitrary time intervals. Our main result shows that for smooth strongly convex and smooth nonconvex functions we achieve convergence rates that match the synchronous version that requires all clients to communicate simultaneously.

----

## [40] Asynchronous Distributed Learning : Adapting to Gradient Delays without Prior Knowledge

**Authors**: *Rotem Zamir Aviv, Ido Hakimi, Assaf Schuster, Kfir Yehuda Levy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/aviv21a.html](http://proceedings.mlr.press/v139/aviv21a.html)

**Abstract**:

We consider stochastic convex optimization problems, where several machines act asynchronously in parallel while sharing a common memory. We propose a robust training method for the constrained setting and derive non asymptotic convergence guarantees that do not depend on prior knowledge of update delays, objective smoothness, and gradient variance. Conversely, existing methods for this setting crucially rely on this prior knowledge, which render them unsuitable for essentially all shared-resources computational environments, such as clouds and data centers. Concretely, existing approaches are unable to accommodate changes in the delays which result from dynamic allocation of the machines, while our method implicitly adapts to such changes.

----

## [41] Decomposable Submodular Function Minimization via Maximum Flow

**Authors**: *Kyriakos Axiotis, Adam Karczmarz, Anish Mukherjee, Piotr Sankowski, Adrian Vladu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/axiotis21a.html](http://proceedings.mlr.press/v139/axiotis21a.html)

**Abstract**:

This paper bridges discrete and continuous optimization approaches for decomposable submodular function minimization, in both the standard and parametric settings. We provide improved running times for this problem by reducing it to a number of calls to a maximum flow oracle. When each function in the decomposition acts on O(1) elements of the ground set V and is polynomially bounded, our running time is up to polylogarithmic factors equal to that of solving maximum flow in a sparse graph with O(|V|) vertices and polynomial integral capacities. We achieve this by providing a simple iterative method which can optimize to high precision any convex function defined on the submodular base polytope, provided we can efficiently minimize it on the base polytope corresponding to the cut function of a certain graph that we construct. We solve this minimization problem by lifting the solutions of a parametric cut problem, which we obtain via a new efficient combinatorial reduction to maximum flow. This reduction is of independent interest and implies some previously unknown bounds for the parametric minimum s,t-cut problem in multiple settings.

----

## [42] Differentially Private Query Release Through Adaptive Projection

**Authors**: *Sergül Aydöre, William Brown, Michael Kearns, Krishnaram Kenthapadi, Luca Melis, Aaron Roth, Amaresh Ankit Siva*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/aydore21a.html](http://proceedings.mlr.press/v139/aydore21a.html)

**Abstract**:

We propose, implement, and evaluate a new algo-rithm for releasing answers to very large numbersof statistical queries likek-way marginals, sub-ject to differential privacy. Our algorithm makesadaptive use of a continuous relaxation of thePro-jection Mechanism, which answers queries on theprivate dataset using simple perturbation, and thenattempts to find the synthetic dataset that mostclosely matches the noisy answers. We use a con-tinuous relaxation of the synthetic dataset domainwhich makes the projection loss differentiable,and allows us to use efficient ML optimizationtechniques and tooling. Rather than answering allqueries up front, we make judicious use of ourprivacy budget by iteratively finding queries forwhich our (relaxed) synthetic data has high error,and then repeating the projection. Randomizedrounding allows us to obtain synthetic data in theoriginal schema. We perform experimental evalu-ations across a range of parameters and datasets,and find that our method outperforms existingalgorithms on large query classes.

----

## [43] On the Implicit Bias of Initialization Shape: Beyond Infinitesimal Mirror Descent

**Authors**: *Shahar Azulay, Edward Moroshko, Mor Shpigel Nacson, Blake E. Woodworth, Nathan Srebro, Amir Globerson, Daniel Soudry*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/azulay21a.html](http://proceedings.mlr.press/v139/azulay21a.html)

**Abstract**:

Recent work has highlighted the role of initialization scale in determining the structure of the solutions that gradient methods converge to. In particular, it was shown that large initialization leads to the neural tangent kernel regime solution, whereas small initialization leads to so called “rich regimes”. However, the initialization structure is richer than the overall scale alone and involves relative magnitudes of different weights and layers in the network. Here we show that these relative scales, which we refer to as initialization shape, play an important role in determining the learned model. We develop a novel technique for deriving the inductive bias of gradient-flow and use it to obtain closed-form implicit regularizers for multiple cases of interest.

----

## [44] On-Off Center-Surround Receptive Fields for Accurate and Robust Image Classification

**Authors**: *Zahra Babaiee, Ramin M. Hasani, Mathias Lechner, Daniela Rus, Radu Grosu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/babaiee21a.html](http://proceedings.mlr.press/v139/babaiee21a.html)

**Abstract**:

Robustness to variations in lighting conditions is a key objective for any deep vision system. To this end, our paper extends the receptive field of convolutional neural networks with two residual components, ubiquitous in the visual processing system of vertebrates: On-center and off-center pathways, with an excitatory center and inhibitory surround; OOCS for short. The On-center pathway is excited by the presence of a light stimulus in its center, but not in its surround, whereas the Off-center pathway is excited by the absence of a light stimulus in its center, but not in its surround. We design OOCS pathways via a difference of Gaussians, with their variance computed analytically from the size of the receptive fields. OOCS pathways complement each other in their response to light stimuli, ensuring this way a strong edge-detection capability, and as a result an accurate and robust inference under challenging lighting conditions. We provide extensive empirical evidence showing that networks supplied with OOCS pathways gain accuracy and illumination-robustness from the novel edge representation, compared to other baselines.

----

## [45] Uniform Convergence, Adversarial Spheres and a Simple Remedy

**Authors**: *Gregor Bachmann, Seyed-Mohsen Moosavi-Dezfooli, Thomas Hofmann*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bachmann21a.html](http://proceedings.mlr.press/v139/bachmann21a.html)

**Abstract**:

Previous work has cast doubt on the general framework of uniform convergence and its ability to explain generalization in neural networks. By considering a specific dataset, it was observed that a neural network completely misclassifies a projection of the training data (adversarial set), rendering any existing generalization bound based on uniform convergence vacuous. We provide an extensive theoretical investigation of the previously studied data setting through the lens of infinitely-wide models. We prove that the Neural Tangent Kernel (NTK) also suffers from the same phenomenon and we uncover its origin. We highlight the important role of the output bias and show theoretically as well as empirically how a sensible choice completely mitigates the problem. We identify sharp phase transitions in the accuracy on the adversarial set and study its dependency on the training sample size. As a result, we are able to characterize critical sample sizes beyond which the effect disappears. Moreover, we study decompositions of a neural network into a clean and noisy part by considering its canonical decomposition into its different eigenfunctions and show empirically that for too small bias the adversarial phenomenon still persists.

----

## [46] Faster Kernel Matrix Algebra via Density Estimation

**Authors**: *Arturs Backurs, Piotr Indyk, Cameron Musco, Tal Wagner*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/backurs21a.html](http://proceedings.mlr.press/v139/backurs21a.html)

**Abstract**:

We study fast algorithms for computing basic properties of an n x n positive semidefinite kernel matrix K corresponding to n points x_1,...,x_n in R^d. In particular, we consider the estimating the sum of kernel matrix entries, along with its top eigenvalue and eigenvector. These are some of the most basic problems defined over kernel matrices. We show that the sum of matrix entries can be estimated up to a multiplicative factor of 1+\epsilon in time sublinear in n and linear in d for many popular kernel functions, including the Gaussian, exponential, and rational quadratic kernels. For these kernels, we also show that the top eigenvalue (and a witnessing approximate eigenvector) can be approximated to a multiplicative factor of 1+\epsilon in time sub-quadratic in n and linear in d. Our algorithms represent significant advances in the best known runtimes for these problems. They leverage the positive definiteness of the kernel matrix, along with a recent line of work on efficient kernel density estimation.

----

## [47] Robust Reinforcement Learning using Least Squares Policy Iteration with Provable Performance Guarantees

**Authors**: *Kishan Panaganti Badrinath, Dileep Kalathil*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/badrinath21a.html](http://proceedings.mlr.press/v139/badrinath21a.html)

**Abstract**:

This paper addresses the problem of model-free reinforcement learning for Robust Markov Decision Process (RMDP) with large state spaces. The goal of the RMDPs framework is to find a policy that is robust against the parameter uncertainties due to the mismatch between the simulator model and real-world settings. We first propose the Robust Least Squares Policy Evaluation algorithm, which is a multi-step online model-free learning algorithm for policy evaluation. We prove the convergence of this algorithm using stochastic approximation techniques. We then propose Robust Least Squares Policy Iteration (RLSPI) algorithm for learning the optimal robust policy. We also give a general weighted Euclidean norm bound on the error (closeness to optimality) of the resulting policy. Finally, we demonstrate the performance of our RLSPI algorithm on some benchmark problems from OpenAI Gym.

----

## [48] Skill Discovery for Exploration and Planning using Deep Skill Graphs

**Authors**: *Akhil Bagaria, Jason K. Senthil, George Konidaris*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bagaria21a.html](http://proceedings.mlr.press/v139/bagaria21a.html)

**Abstract**:

We introduce a new skill-discovery algorithm that builds a discrete graph representation of large continuous MDPs, where nodes correspond to skill subgoals and the edges to skill policies. The agent constructs this graph during an unsupervised training phase where it interleaves discovering skills and planning using them to gain coverage over ever-increasing portions of the state-space. Given a novel goal at test time, the agent plans with the acquired skill graph to reach a nearby state, then switches to learning to reach the goal. We show that the resulting algorithm, Deep Skill Graphs, outperforms both flat and existing hierarchical reinforcement learning methods on four difficult continuous control tasks.

----

## [49] Locally Adaptive Label Smoothing Improves Predictive Churn

**Authors**: *Dara Bahri, Heinrich Jiang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bahri21a.html](http://proceedings.mlr.press/v139/bahri21a.html)

**Abstract**:

Training modern neural networks is an inherently noisy process that can lead to high \emph{prediction churn}– disagreements between re-trainings of the same model due to factors such as randomization in the parameter initialization and mini-batches– even when the trained models all attain similar accuracies. Such prediction churn can be very undesirable in practice. In this paper, we present several baselines for reducing churn and show that training on soft labels obtained by adaptively smoothing each example’s label based on the example’s neighboring labels often outperforms the baselines on churn while improving accuracy on a variety of benchmark classification tasks and model architectures.

----

## [50] How Important is the Train-Validation Split in Meta-Learning?

**Authors**: *Yu Bai, Minshuo Chen, Pan Zhou, Tuo Zhao, Jason D. Lee, Sham M. Kakade, Huan Wang, Caiming Xiong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bai21a.html](http://proceedings.mlr.press/v139/bai21a.html)

**Abstract**:

Meta-learning aims to perform fast adaptation on a new task through learning a “prior” from multiple existing tasks. A common practice in meta-learning is to perform a train-validation split (\emph{train-val method}) where the prior adapts to the task on one split of the data, and the resulting predictor is evaluated on another split. Despite its prevalence, the importance of the train-validation split is not well understood either in theory or in practice, particularly in comparison to the more direct \emph{train-train method}, which uses all the per-task data for both training and evaluation. We provide a detailed theoretical study on whether and when the train-validation split is helpful in the linear centroid meta-learning problem. In the agnostic case, we show that the expected loss of the train-val method is minimized at the optimal prior for meta testing, and this is not the case for the train-train method in general without structural assumptions on the data. In contrast, in the realizable case where the data are generated from linear models, we show that both the train-val and train-train losses are minimized at the optimal prior in expectation. Further, perhaps surprisingly, our main result shows that the train-train method achieves a \emph{strictly better} excess loss in this realizable case, even when the regularization parameter and split ratio are optimally tuned for both methods. Our results highlight that sample splitting may not always be preferable, especially when the data is realizable by the model. We validate our theories by experimentally showing that the train-train method can indeed outperform the train-val method, on both simulations and real meta-learning tasks.

----

## [51] Stabilizing Equilibrium Models by Jacobian Regularization

**Authors**: *Shaojie Bai, Vladlen Koltun, J. Zico Kolter*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bai21b.html](http://proceedings.mlr.press/v139/bai21b.html)

**Abstract**:

Deep equilibrium networks (DEQs) are a new class of models that eschews traditional depth in favor of finding the fixed point of a single non-linear layer. These models have been shown to achieve performance competitive with the state-of-the-art deep networks while using significantly less memory. Yet they are also slower, brittle to architectural choices, and introduce potential instability to the model. In this paper, we propose a regularization scheme for DEQ models that explicitly regularizes the Jacobian of the fixed-point update equations to stabilize the learning of equilibrium models. We show that this regularization adds only minimal computational cost, significantly stabilizes the fixed-point convergence in both forward and backward passes, and scales well to high-dimensional, realistic domains (e.g., WikiText-103 language modeling and ImageNet classification). Using this method, we demonstrate, for the first time, an implicit-depth model that runs with approximately the same speed and level of performance as popular conventional deep networks such as ResNet-101, while still maintaining the constant memory footprint and architectural simplicity of DEQs. Code is available https://github.com/locuslab/deq.

----

## [52] Don't Just Blame Over-parametrization for Over-confidence: Theoretical Analysis of Calibration in Binary Classification

**Authors**: *Yu Bai, Song Mei, Huan Wang, Caiming Xiong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bai21c.html](http://proceedings.mlr.press/v139/bai21c.html)

**Abstract**:

Modern machine learning models with high accuracy are often miscalibrated—the predicted top probability does not reflect the actual accuracy, and tends to be \emph{over-confident}. It is commonly believed that such over-confidence is mainly due to \emph{over-parametrization}, in particular when the model is large enough to memorize the training data and maximize the confidence. In this paper, we show theoretically that over-parametrization is not the only reason for over-confidence. We prove that \emph{logistic regression is inherently over-confident}, in the realizable, under-parametrized setting where the data is generated from the logistic model, and the sample size is much larger than the number of parameters. Further, this over-confidence happens for general well-specified binary classification problems as long as the activation is symmetric and concave on the positive part. Perhaps surprisingly, we also show that over-confidence is not always the case—there exists another activation function (and a suitable loss function) under which the learned classifier is \emph{under-confident} at some probability values. Overall, our theory provides a precise characterization of calibration in realizable binary classification, which we verify on simulations and real data experiments.

----

## [53] Principled Exploration via Optimistic Bootstrapping and Backward Induction

**Authors**: *Chenjia Bai, Lingxiao Wang, Lei Han, Jianye Hao, Animesh Garg, Peng Liu, Zhaoran Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bai21d.html](http://proceedings.mlr.press/v139/bai21d.html)

**Abstract**:

One principled approach for provably efficient exploration is incorporating the upper confidence bound (UCB) into the value function as a bonus. However, UCB is specified to deal with linear and tabular settings and is incompatible with Deep Reinforcement Learning (DRL). In this paper, we propose a principled exploration method for DRL through Optimistic Bootstrapping and Backward Induction (OB2I). OB2I constructs a general-purpose UCB-bonus through non-parametric bootstrap in DRL. The UCB-bonus estimates the epistemic uncertainty of state-action pairs for optimistic exploration. We build theoretical connections between the proposed UCB-bonus and the LSVI-UCB in linear setting. We propagate future uncertainty in a time-consistent manner through episodic backward update, which exploits the theoretical advantage and empirically improves the sample-efficiency. Our experiments in MNIST maze and Atari suit suggest that OB2I outperforms several state-of-the-art exploration approaches.

----

## [54] GLSearch: Maximum Common Subgraph Detection via Learning to Search

**Authors**: *Yunsheng Bai, Derek Xu, Yizhou Sun, Wei Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bai21e.html](http://proceedings.mlr.press/v139/bai21e.html)

**Abstract**:

Detecting the Maximum Common Subgraph (MCS) between two input graphs is fundamental for applications in drug synthesis, malware detection, cloud computing, etc. However, MCS computation is NP-hard, and state-of-the-art MCS solvers rely on heuristic search algorithms which in practice cannot find good solution for large graph pairs given a limited computation budget. We propose GLSearch, a Graph Neural Network (GNN) based learning to search model. Our model is built upon the branch and bound algorithm, which selects one pair of nodes from the two input graphs to expand at a time. We propose a novel GNN-based Deep Q-Network (DQN) to select the node pair, making the search process much faster. Experiments on synthetic and real-world graph pairs demonstrate that our model learns a search strategy that is able to detect significantly larger common subgraphs than existing MCS solvers given the same computation budget. GLSearch can be potentially extended to solve many other combinatorial problems with constraints on graphs.

----

## [55] Breaking the Limits of Message Passing Graph Neural Networks

**Authors**: *Muhammet Balcilar, Pierre Héroux, Benoit Gaüzère, Pascal Vasseur, Sébastien Adam, Paul Honeine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/balcilar21a.html](http://proceedings.mlr.press/v139/balcilar21a.html)

**Abstract**:

Since the Message Passing (Graph) Neural Networks (MPNNs) have a linear complexity with respect to the number of nodes when applied to sparse graphs, they have been widely implemented and still raise a lot of interest even though their theoretical expressive power is limited to the first order Weisfeiler-Lehman test (1-WL). In this paper, we show that if the graph convolution supports are designed in spectral-domain by a non-linear custom function of eigenvalues and masked with an arbitrary large receptive field, the MPNN is theoretically more powerful than the 1-WL test and experimentally as powerful as a 3-WL existing models, while remaining spatially localized. Moreover, by designing custom filter functions, outputs can have various frequency components that allow the convolution process to learn different relationships between a given input graph signal and its associated properties. So far, the best 3-WL equivalent graph neural networks have a computational complexity in $\mathcal{O}(n^3)$ with memory usage in $\mathcal{O}(n^2)$, consider non-local update mechanism and do not provide the spectral richness of output profile. The proposed method overcomes all these aforementioned problems and reaches state-of-the-art results in many downstream tasks.

----

## [56] Instance Specific Approximations for Submodular Maximization

**Authors**: *Eric Balkanski, Sharon Qian, Yaron Singer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/balkanski21a.html](http://proceedings.mlr.press/v139/balkanski21a.html)

**Abstract**:

The predominant measure for the performance of an algorithm is its worst-case approximation guarantee. While worst-case approximations give desirable robustness guarantees, they can differ significantly from the performance of an algorithm in practice. For the problem of monotone submodular maximization under a cardinality constraint, the greedy algorithm is known to obtain a 1-1/e approximation guarantee, which is optimal for a polynomial-time algorithm. However, very little is known about the approximation achieved by greedy and other submodular maximization algorithms on real instances. We develop an algorithm that gives an instance-specific approximation for any solution of an instance of monotone submodular maximization under a cardinality constraint. This algorithm uses a novel dual approach to submodular maximization. In particular, it relies on the construction of a lower bound to the dual objective that can also be exactly minimized. We use this algorithm to show that on a wide variety of real-world datasets and objectives, greedy and other algorithms find solutions that approximate the optimal solution significantly better than the 1-1/e   0.63 worst-case approximation guarantee, often exceeding 0.9.

----

## [57] Augmented World Models Facilitate Zero-Shot Dynamics Generalization From a Single Offline Environment

**Authors**: *Philip J. Ball, Cong Lu, Jack Parker-Holder, Stephen J. Roberts*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ball21a.html](http://proceedings.mlr.press/v139/ball21a.html)

**Abstract**:

Reinforcement learning from large-scale offline datasets provides us with the ability to learn policies without potentially unsafe or impractical exploration. Significant progress has been made in the past few years in dealing with the challenge of correcting for differing behavior between the data collection and learned policies. However, little attention has been paid to potentially changing dynamics when transferring a policy to the online setting, where performance can be up to 90% reduced for existing methods. In this paper we address this problem with Augmented World Models (AugWM). We augment a learned dynamics model with simple transformations that seek to capture potential changes in physical properties of the robot, leading to more robust policies. We not only train our policy in this new setting, but also provide it with the sampled augmentation as a context, allowing it to adapt to changes in the environment. At test time we learn the context in a self-supervised fashion by approximating the augmentation which corresponds to the new environment. We rigorously evaluate our approach on over 100 different changed dynamics settings, and show that this simple approach can significantly improve the zero-shot generalization of a recent state-of-the-art baseline, often achieving successful policies where the baseline fails.

----

## [58] Regularized Online Allocation Problems: Fairness and Beyond

**Authors**: *Santiago R. Balseiro, Haihao Lu, Vahab S. Mirrokni*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/balseiro21a.html](http://proceedings.mlr.press/v139/balseiro21a.html)

**Abstract**:

Online allocation problems with resource constraints have a rich history in computer science and operations research. In this paper, we introduce the regularized online allocation problem, a variant that includes a non-linear regularizer acting on the total resource consumption. In this problem, requests repeatedly arrive over time and, for each request, a decision maker needs to take an action that generates a reward and consumes resources. The objective is to simultaneously maximize total rewards and the value of the regularizer subject to the resource constraints. Our primary motivation is the online allocation of internet advertisements wherein firms seek to maximize additive objectives such as the revenue or efficiency of the allocation. By introducing a regularizer, firms can account for the fairness of the allocation or, alternatively, punish under-delivery of advertisements—two common desiderata in internet advertising markets. We design an algorithm when arrivals are drawn independently from a distribution that is unknown to the decision maker. Our algorithm is simple, fast, and attains the optimal order of sub-linear regret compared to the optimal allocation with the benefit of hindsight. Numerical experiments confirm the effectiveness of the proposed algorithm and of the regularizers in an internet advertising application.

----

## [59] Predict then Interpolate: A Simple Algorithm to Learn Stable Classifiers

**Authors**: *Yujia Bao, Shiyu Chang, Regina Barzilay*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bao21a.html](http://proceedings.mlr.press/v139/bao21a.html)

**Abstract**:

We propose Predict then Interpolate (PI), a simple algorithm for learning correlations that are stable across environments. The algorithm follows from the intuition that when using a classifier trained on one environment to make predictions on examples from another environment, its mistakes are informative as to which correlations are unstable. In this work, we prove that by interpolating the distributions of the correct predictions and the wrong predictions, we can uncover an oracle distribution where the unstable correlation vanishes. Since the oracle interpolation coefficients are not accessible, we use group distributionally robust optimization to minimize the worst-case risk across all such interpolations. We evaluate our method on both text classification and image classification. Empirical results demonstrate that our algorithm is able to learn robust classifiers (outperforms IRM by 23.85% on synthetic environments and 12.41% on natural environments). Our code and data are available at https://github.com/YujiaBao/ Predict-then-Interpolate.

----

## [60] Variational (Gradient) Estimate of the Score Function in Energy-based Latent Variable Models

**Authors**: *Fan Bao, Kun Xu, Chongxuan Li, Lanqing Hong, Jun Zhu, Bo Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bao21b.html](http://proceedings.mlr.press/v139/bao21b.html)

**Abstract**:

This paper presents new estimates of the score function and its gradient with respect to the model parameters in a general energy-based latent variable model (EBLVM). The score function and its gradient can be expressed as combinations of expectation and covariance terms over the (generally intractable) posterior of the latent variables. New estimates are obtained by introducing a variational posterior to approximate the true posterior in these terms. The variational posterior is trained to minimize a certain divergence (e.g., the KL divergence) between itself and the true posterior. Theoretically, the divergence characterizes upper bounds of the bias of the estimates. In principle, our estimates can be applied to a wide range of objectives, including kernelized Stein discrepancy (KSD), score matching (SM)-based methods and exact Fisher divergence with a minimal model assumption. In particular, these estimates applied to SM-based methods outperform existing methods in learning EBLVMs on several image datasets.

----

## [61] Compositional Video Synthesis with Action Graphs

**Authors**: *Amir Bar, Roei Herzig, Xiaolong Wang, Anna Rohrbach, Gal Chechik, Trevor Darrell, Amir Globerson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bar21a.html](http://proceedings.mlr.press/v139/bar21a.html)

**Abstract**:

Videos of actions are complex signals containing rich compositional structure in space and time. Current video generation methods lack the ability to condition the generation on multiple coordinated and potentially simultaneous timed actions. To address this challenge, we propose to represent the actions in a graph structure called Action Graph and present the new "Action Graph To Video" synthesis task. Our generative model for this task (AG2Vid) disentangles motion and appearance features, and by incorporating a scheduling mechanism for actions facilitates a timely and coordinated video generation. We train and evaluate AG2Vid on CATER and Something-Something V2 datasets, which results in videos that have better visual quality and semantic consistency compared to baselines. Finally, our model demonstrates zero-shot abilities by synthesizing novel compositions of the learned actions.

----

## [62] Approximating a Distribution Using Weight Queries

**Authors**: *Nadav Barak, Sivan Sabato*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/barak21a.html](http://proceedings.mlr.press/v139/barak21a.html)

**Abstract**:

We consider a novel challenge: approximating a distribution without the ability to randomly sample from that distribution. We study how such an approximation can be obtained using *weight queries*. Given some data set of examples, a weight query presents one of the examples to an oracle, which returns the probability, according to the target distribution, of observing examples similar to the presented example. This oracle can represent, for instance, counting queries to a database of the target population, or an interface to a search engine which returns the number of results that match a given search. We propose an interactive algorithm that iteratively selects data set examples and performs corresponding weight queries. The algorithm finds a reweighting of the data set that approximates the weights according to the target distribution, using a limited number of weight queries. We derive an approximation bound on the total variation distance between the reweighting found by the algorithm and the best achievable reweighting. Our algorithm takes inspiration from the UCB approach common in multi-armed bandits problems, and combines it with a new discrepancy estimator and a greedy iterative procedure. In addition to our theoretical guarantees, we demonstrate in experiments the advantages of the proposed algorithm over several baselines. A python implementation of the proposed algorithm and of all the experiments can be found at https://github.com/Nadav-Barak/AWP.

----

## [63] Graph Convolution for Semi-Supervised Classification: Improved Linear Separability and Out-of-Distribution Generalization

**Authors**: *Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/baranwal21a.html](http://proceedings.mlr.press/v139/baranwal21a.html)

**Abstract**:

Recently there has been increased interest in semi-supervised classification in the presence of graphical information. A new class of learning models has emerged that relies, at its most basic level, on classifying the data after first applying a graph convolution. To understand the merits of this approach, we study the classification of a mixture of Gaussians, where the data corresponds to the node attributes of a stochastic block model. We show that graph convolution extends the regime in which the data is linearly separable by a factor of roughly $1/\sqrt{D}$, where $D$ is the expected degree of a node, as compared to the mixture model data on its own. Furthermore, we find that the linear classifier obtained by minimizing the cross-entropy loss after the graph convolution generalizes to out-of-distribution data where the unseen data can have different intra- and inter-class edge probabilities from the training data.

----

## [64] Training Quantized Neural Networks to Global Optimality via Semidefinite Programming

**Authors**: *Burak Bartan, Mert Pilanci*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bartan21a.html](http://proceedings.mlr.press/v139/bartan21a.html)

**Abstract**:

Neural networks (NNs) have been extremely successful across many tasks in machine learning. Quantization of NN weights has become an important topic due to its impact on their energy efficiency, inference time and deployment on hardware. Although post-training quantization is well-studied, training optimal quantized NNs involves combinatorial non-convex optimization problems which appear intractable. In this work, we introduce a convex optimization strategy to train quantized NNs with polynomial activations. Our method leverages hidden convexity in two-layer neural networks from the recent literature, semidefinite lifting, and Grothendieck’s identity. Surprisingly, we show that certain quantized NN problems can be solved to global optimality provably in polynomial time in all relevant parameters via tight semidefinite relaxations. We present numerical examples to illustrate the effectiveness of our method.

----

## [65] Beyond log2(T) regret for decentralized bandits in matching markets

**Authors**: *Soumya Basu, Karthik Abinav Sankararaman, Abishek Sankararaman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/basu21a.html](http://proceedings.mlr.press/v139/basu21a.html)

**Abstract**:

We design decentralized algorithms for regret minimization in the two sided matching market with one-sided bandit feedback that significantly improves upon the prior works (Liu et al.\,2020a, Sankararaman et al.\,2020, Liu et al.\,2020b). First, for general markets, for any $\varepsilon > 0$, we design an algorithm that achieves a $O(\log^{1+\varepsilon}(T))$ regret to the agent-optimal stable matching, with unknown time horizon $T$, improving upon the $O(\log^{2}(T))$ regret achieved in (Liu et al.\,2020b). Second, we provide the optimal $\Theta(\log(T))$ agent-optimal regret for markets satisfying {\em uniqueness consistency} – markets where leaving participants don’t alter the original stable matching. Previously, $\Theta(\log(T))$ regret was achievable (Sankararaman et al.\,2020, Liu et al.\,2020b) in the much restricted {\em serial dictatorship} setting, when all arms have the same preference over the agents. We propose a phase based algorithm, where in each phase, besides deleting the globally communicated dominated arms the agents locally delete arms with which they collide often. This \emph{local deletion} is pivotal in breaking deadlocks arising from rank heterogeneity of agents across arms. We further demonstrate superiority of our algorithm over existing works through simulations.

----

## [66] Optimal Thompson Sampling strategies for support-aware CVaR bandits

**Authors**: *Dorian Baudry, Romain Gautron, Emilie Kaufmann, Odalric Maillard*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/baudry21a.html](http://proceedings.mlr.press/v139/baudry21a.html)

**Abstract**:

In this paper we study a multi-arm bandit problem in which the quality of each arm is measured by the Conditional Value at Risk (CVaR) at some level alpha of the reward distribution. While existing works in this setting mainly focus on Upper Confidence Bound algorithms, we introduce a new Thompson Sampling approach for CVaR bandits on bounded rewards that is flexible enough to solve a variety of problems grounded on physical resources. Building on a recent work by Riou & Honda (2020), we introduce B-CVTS for continuous bounded rewards and M-CVTS for multinomial distributions. On the theoretical side, we provide a non-trivial extension of their analysis that enables to theoretically bound their CVaR regret minimization performance. Strikingly, our results show that these strategies are the first to provably achieve asymptotic optimality in CVaR bandits, matching the corresponding asymptotic lower bounds for this setting. Further, we illustrate empirically the benefit of Thompson Sampling approaches both in a realistic environment simulating a use-case in agriculture and on various synthetic examples.

----

## [67] On Limited-Memory Subsampling Strategies for Bandits

**Authors**: *Dorian Baudry, Yoan Russac, Olivier Cappé*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/baudry21b.html](http://proceedings.mlr.press/v139/baudry21b.html)

**Abstract**:

There has been a recent surge of interest in non-parametric bandit algorithms based on subsampling. One drawback however of these approaches is the additional complexity required by random subsampling and the storage of the full history of rewards. Our first contribution is to show that a simple deterministic subsampling rule, proposed in the recent work of \citet{baudry2020sub} under the name of “last-block subsampling”, is asymptotically optimal in one-parameter exponential families. In addition, we prove that these guarantees also hold when limiting the algorithm memory to a polylogarithmic function of the time horizon. These findings open up new perspectives, in particular for non-stationary scenarios in which the arm distributions evolve over time. We propose a variant of the algorithm in which only the most recent observations are used for subsampling, achieving optimal regret guarantees under the assumption of a known number of abrupt changes. Extensive numerical simulations highlight the merits of this approach, particularly when the changes are not only affecting the means of the rewards.

----

## [68] Generalized Doubly Reparameterized Gradient Estimators

**Authors**: *Matthias Bauer, Andriy Mnih*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bauer21a.html](http://proceedings.mlr.press/v139/bauer21a.html)

**Abstract**:

Efficient low-variance gradient estimation enabled by the reparameterization trick (RT) has been essential to the success of variational autoencoders. Doubly-reparameterized gradients (DReGs) improve on the RT for multi-sample variational bounds by applying reparameterization a second time for an additional reduction in variance. Here, we develop two generalizations of the DReGs estimator and show that they can be used to train conditional and hierarchical VAEs on image modelling tasks more effectively. We first extend the estimator to hierarchical models with several stochastic layers by showing how to treat additional score function terms due to the hierarchical variational posterior. We then generalize DReGs to score functions of arbitrary distributions instead of just those of the sampling distribution, which makes the estimator applicable to the parameters of the prior in addition to those of the posterior.

----

## [69] Directional Graph Networks

**Authors**: *Dominique Beaini, Saro Passaro, Vincent Létourneau, William L. Hamilton, Gabriele Corso, Pietro Lió*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/beani21a.html](http://proceedings.mlr.press/v139/beani21a.html)

**Abstract**:

The lack of anisotropic kernels in graph neural networks (GNNs) strongly limits their expressiveness, contributing to well-known issues such as over-smoothing. To overcome this limitation, we propose the first globally consistent anisotropic kernels for GNNs, allowing for graph convolutions that are defined according to topologicaly-derived directional flows. First, by defining a vector field in the graph, we develop a method of applying directional derivatives and smoothing by projecting node-specific messages into the field. Then, we propose the use of the Laplacian eigenvectors as such vector field. We show that the method generalizes CNNs on an

----

## [70] Policy Analysis using Synthetic Controls in Continuous-Time

**Authors**: *Alexis Bellot, Mihaela van der Schaar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bellot21a.html](http://proceedings.mlr.press/v139/bellot21a.html)

**Abstract**:

Counterfactual estimation using synthetic controls is one of the most successful recent methodological developments in causal inference. Despite its popularity, the current description only considers time series aligned across units and synthetic controls expressed as linear combinations of observed control units. We propose a continuous-time alternative that models the latent counterfactual path explicitly using the formalism of controlled differential equations. This model is directly applicable to the general setting of irregularly-aligned multivariate time series and may be optimized in rich function spaces – thereby improving on some limitations of existing approaches.

----

## [71] Loss Surface Simplexes for Mode Connecting Volumes and Fast Ensembling

**Authors**: *Gregory W. Benton, Wesley J. Maddox, Sanae Lotfi, Andrew Gordon Wilson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/benton21a.html](http://proceedings.mlr.press/v139/benton21a.html)

**Abstract**:

With a better understanding of the loss surfaces for multilayer networks, we can build more robust and accurate training procedures. Recently it was discovered that independently trained SGD solutions can be connected along one-dimensional paths of near-constant training loss. In this paper, we in fact demonstrate the existence of mode-connecting simplicial complexes that form multi-dimensional manifolds of low loss, connecting many independently trained models. Building on this discovery, we show how to efficiently construct simplicial complexes for fast ensembling, outperforming independently trained deep ensembles in accuracy, calibration, and robustness to dataset shift. Notably, our approach is easy to apply and only requires a few training epochs to discover a low-loss simplex.

----

## [72] TFix: Learning to Fix Coding Errors with a Text-to-Text Transformer

**Authors**: *Berkay Berabi, Jingxuan He, Veselin Raychev, Martin T. Vechev*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/berabi21a.html](http://proceedings.mlr.press/v139/berabi21a.html)

**Abstract**:

The problem of fixing errors in programs has attracted substantial interest over the years. The key challenge for building an effective code fixing tool is to capture a wide range of errors and meanwhile maintain high accuracy. In this paper, we address this challenge and present a new learning-based system, called TFix. TFix works directly on program text and phrases the problem of code fixing as a text-to-text task. In turn, this enables it to leverage a powerful Transformer based model pre-trained on natural language and fine-tuned to generate code fixes (via a large, high-quality dataset obtained from GitHub commits). TFix is not specific to a particular programming language or class of defects and, in fact, improved its precision by simultaneously fine-tuning on 52 different error types reported by a popular static analyzer. Our evaluation on a massive dataset of JavaScript programs shows that TFix is practically effective: it is able to synthesize code that fixes the error in  67 percent of cases and significantly outperforms existing learning-based approaches.

----

## [73] Learning Queueing Policies for Organ Transplantation Allocation using Interpretable Counterfactual Survival Analysis

**Authors**: *Jeroen Berrevoets, Ahmed M. Alaa, Zhaozhi Qian, James Jordon, Alexander E. S. Gimson, Mihaela van der Schaar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/berrevoets21a.html](http://proceedings.mlr.press/v139/berrevoets21a.html)

**Abstract**:

Organ transplantation is often the last resort for treating end-stage illnesses, but managing transplant wait-lists is challenging because of organ scarcity and the complexity of assessing donor-recipient compatibility. In this paper, we develop a data-driven model for (real-time) organ allocation using observational data for transplant outcomes. Our model integrates a queuing-theoretic framework with unsupervised learning to cluster the organs into “organ types”, and then construct priority queues (associated with each organ type) wherein incoming patients are assigned. To reason about organ allocations, the model uses synthetic controls to infer a patient’s survival outcomes under counterfactual allocations to the different organ types{–} the model is trained end-to-end to optimise the trade-off between patient waiting time and expected survival time. The usage of synthetic controls enable patient-level interpretations of allocation decisions that can be presented and understood by clinicians. We test our model on multiple data sets, and show that it outperforms other organ-allocation policies in terms of added life-years, and death count. Furthermore, we introduce a novel organ-allocation simulator to accurately test new policies.

----

## [74] Learning from Biased Data: A Semi-Parametric Approach

**Authors**: *Patrice Bertail, Stéphan Clémençon, Yannick Guyonvarch, Nathan Noiry*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bertail21a.html](http://proceedings.mlr.press/v139/bertail21a.html)

**Abstract**:

We consider risk minimization problems where the (source) distribution $P_S$ of the training observations $Z_1, \ldots, Z_n$ differs from the (target) distribution $P_T$ involved in the risk that one seeks to minimize. Under the natural assumption that $P_S$ dominates $P_T$, \textit{i.e.} $P_T< \! \!
Cite this Paper

 BibTeX 

@InProceedings{pmlr-v139-bertail21a,
  title = 	 {Learning from Biased Data: A Semi-Parametric Approach},
  author =       {Bertail, Patrice and Cl{\'e}men{\c{c}}on, Stephan and Guyonvarch, Yannick and Noiry, Nathan},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {803--812},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/bertail21a/bertail21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/bertail21a.html},
  abstract = 	 {We consider risk minimization problems where the (source) distribution $P_S$ of the training observations $Z_1, \ldots, Z_n$ differs from the (target) distribution $P_T$ involved in the risk that one seeks to minimize. Under the natural assumption that $P_S$ dominates $P_T$, \textit{i.e.} $P_T< \! \!
Copy to ClipboardDownload
 Endnote 
%0 Conference Paper
%T Learning from Biased Data: A Semi-Parametric Approach
%A Patrice Bertail
%A Stephan Clémençon
%A Yannick Guyonvarch
%A Nathan Noiry
%B Proceedings of the 38th International Conference on Machine Learning
%C Proceedings of Machine Learning Research
%D 2021
%E Marina Meila
%E Tong Zhang	
%F pmlr-v139-bertail21a
%I PMLR
%P 803--812
%U https://proceedings.mlr.press/v139/bertail21a.html
%V 139
%X We consider risk minimization problems where the (source) distribution $P_S$ of the training observations $Z_1, \ldots, Z_n$ differs from the (target) distribution $P_T$ involved in the risk that one seeks to minimize. Under the natural assumption that $P_S$ dominates $P_T$, \textit{i.e.} $P_T< \! \!
Copy to ClipboardDownload
 APA 

Bertail, P., Clémençon, S., Guyonvarch, Y. & Noiry, N.. (2021). Learning from Biased Data: A Semi-Parametric Approach. Proceedings of the 38th International Conference on Machine Learning, in Proceedings of Machine Learning Research 139:803-812 Available from https://proceedings.mlr.press/v139/bertail21a.html.


Copy to ClipboardDownload

Related Material


Download PDF
Supplementary ZIP

----

## [75] Is Space-Time Attention All You Need for Video Understanding?

**Authors**: *Gedas Bertasius, Heng Wang, Lorenzo Torresani*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bertasius21a.html](http://proceedings.mlr.press/v139/bertasius21a.html)

**Abstract**:

We present a convolution-free approach to video classification built exclusively on self-attention over space and time. Our method, named “TimeSformer,” adapts the standard Transformer architecture to video by enabling spatiotemporal feature learning directly from a sequence of frame-level patches. Our experimental study compares different self-attention schemes and suggests that “divided attention,” where temporal attention and spatial attention are separately applied within each block, leads to the best video classification accuracy among the design choices considered. Despite the radically new design, TimeSformer achieves state-of-the-art results on several action recognition benchmarks, including the best reported accuracy on Kinetics-400 and Kinetics-600. Finally, compared to 3D convolutional networks, our model is faster to train, it can achieve dramatically higher test efficiency (at a small drop in accuracy), and it can also be applied to much longer video clips (over one minute long). Code and models are available at: https://github.com/facebookresearch/TimeSformer.

----

## [76] Confidence Scores Make Instance-dependent Label-noise Learning Possible

**Authors**: *Antonin Berthon, Bo Han, Gang Niu, Tongliang Liu, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/berthon21a.html](http://proceedings.mlr.press/v139/berthon21a.html)

**Abstract**:

In learning with noisy labels, for every instance, its label can randomly walk to other classes following a transition distribution which is named a noise model. Well-studied noise models are all instance-independent, namely, the transition depends only on the original label but not the instance itself, and thus they are less practical in the wild. Fortunately, methods based on instance-dependent noise have been studied, but most of them have to rely on strong assumptions on the noise models. To alleviate this issue, we introduce confidence-scored instance-dependent noise (CSIDN), where each instance-label pair is equipped with a confidence score. We find that with the help of confidence scores, the transition distribution of each instance can be approximately estimated. Similarly to the powerful forward correction for instance-independent noise, we propose a novel instance-level forward correction for CSIDN. We demonstrate the utility and effectiveness of our method through multiple experiments on datasets with synthetic label noise and real-world unknown noise.

----

## [77] Size-Invariant Graph Representations for Graph Classification Extrapolations

**Authors**: *Beatrice Bevilacqua, Yangze Zhou, Bruno Ribeiro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bevilacqua21a.html](http://proceedings.mlr.press/v139/bevilacqua21a.html)

**Abstract**:

In general, graph representation learning methods assume that the train and test data come from the same distribution. In this work we consider an underexplored area of an otherwise rapidly developing field of graph representation learning: The task of out-of-distribution (OOD) graph classification, where train and test data have different distributions, with test data unavailable during training. Our work shows it is possible to use a causal model to learn approximately invariant representations that better extrapolate between train and test data. Finally, we conclude with synthetic and real-world dataset experiments showcasing the benefits of representations that are invariant to train/test distribution shifts.

----

## [78] Principal Bit Analysis: Autoencoding with Schur-Concave Loss

**Authors**: *Sourbh Bhadane, Aaron B. Wagner, Jayadev Acharya*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bhadane21a.html](http://proceedings.mlr.press/v139/bhadane21a.html)

**Abstract**:

We consider a linear autoencoder in which the latent variables are quantized, or corrupted by noise, and the constraint is Schur-concave in the set of latent variances. Although finding the optimal encoder/decoder pair for this setup is a nonconvex optimization problem, we show that decomposing the source into its principal components is optimal. If the constraint is strictly Schur-concave and the empirical covariance matrix has only simple eigenvalues, then any optimal encoder/decoder must decompose the source in this way. As one application, we consider a strictly Schur-concave constraint that estimates the number of bits needed to represent the latent variables under fixed-rate encoding, a setup that we call \emph{Principal Bit Analysis (PBA)}. This yields a practical, general-purpose, fixed-rate compressor that outperforms existing algorithms. As a second application, we show that a prototypical autoencoder-based variable-rate compressor is guaranteed to decompose the source into its principal components.

----

## [79] Lower Bounds on Cross-Entropy Loss in the Presence of Test-time Adversaries

**Authors**: *Arjun Nitin Bhagoji, Daniel Cullina, Vikash Sehwag, Prateek Mittal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bhagoji21a.html](http://proceedings.mlr.press/v139/bhagoji21a.html)

**Abstract**:

Understanding the fundamental limits of robust supervised learning has emerged as a problem of immense interest, from both practical and theoretical standpoints. In particular, it is critical to determine classifier-agnostic bounds on the training loss to establish when learning is possible. In this paper, we determine optimal lower bounds on the cross-entropy loss in the presence of test-time adversaries, along with the corresponding optimal classification outputs. Our formulation of the bound as a solution to an optimization problem is general enough to encompass any loss function depending on soft classifier outputs. We also propose and provide a proof of correctness for a bespoke algorithm to compute this lower bound efficiently, allowing us to determine lower bounds for multiple practical datasets of interest. We use our lower bounds as a diagnostic tool to determine the effectiveness of current robust training methods and find a gap from optimality at larger budgets. Finally, we investigate the possibility of using of optimal classification outputs as soft labels to empirically improve robust training.

----

## [80] Additive Error Guarantees for Weighted Low Rank Approximation

**Authors**: *Aditya Bhaskara, Aravinda Kanchana Ruwanpathirana, Maheshakya Wijewardena*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bhaskara21a.html](http://proceedings.mlr.press/v139/bhaskara21a.html)

**Abstract**:

Low-rank approximation is a classic tool in data analysis, where the goal is to approximate a matrix $A$ with a low-rank matrix $L$ so as to minimize the error $\norm{A - L}_F^2$. However in many applications, approximating some entries is more important than others, which leads to the weighted low rank approximation problem. However, the addition of weights makes the low-rank approximation problem intractable. Thus many works have obtained efficient algorithms under additional structural assumptions on the weight matrix (such as low rank, and appropriate block structure). We study a natural greedy algorithm for weighted low rank approximation and develop a simple condition under which it yields bi-criteria approximation up to a small additive factor in the error. The algorithm involves iteratively computing the top singular vector of an appropriately varying matrix, and is thus easy to implement at scale. Our methods also allow us to study the problem of low rank approximation under $\ell_p$ norm error.

----

## [81] Sample Complexity of Robust Linear Classification on Separated Data

**Authors**: *Robi Bhattacharjee, Somesh Jha, Kamalika Chaudhuri*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bhattacharjee21a.html](http://proceedings.mlr.press/v139/bhattacharjee21a.html)

**Abstract**:

We consider the sample complexity of learning with adversarial robustness. Most prior theoretical results for this problem have considered a setting where different classes in the data are close together or overlapping. We consider, in contrast, the well-separated case where there exists a classifier with perfect accuracy and robustness, and show that the sample complexity narrates an entirely different story. Specifically, for linear classifiers, we show a large class of well-separated distributions where the expected robust loss of any algorithm is at least $\Omega(\frac{d}{n})$, whereas the max margin algorithm has expected standard loss $O(\frac{1}{n})$. This shows a gap in the standard and robust losses that cannot be obtained via prior techniques. Additionally, we present an algorithm that, given an instance where the robustness radius is much smaller than the gap between the classes, gives a solution with expected robust loss is $O(\frac{1}{n})$. This shows that for very well-separated data, convergence rates of $O(\frac{1}{n})$ are achievable, which is not the case otherwise. Our results apply to robustness measured in any $\ell_p$ norm with $p > 1$ (including $p = \infty$).

----

## [82] Finding k in Latent k- polytope

**Authors**: *Chiranjib Bhattacharyya, Ravindran Kannan, Amit Kumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bhattacharyya21a.html](http://proceedings.mlr.press/v139/bhattacharyya21a.html)

**Abstract**:

The recently introduced Latent $k-$ Polytope($\LkP$) encompasses several stochastic Mixed Membership models including Topic Models. The problem of finding $k$, the number of extreme points of $\LkP$, is a fundamental challenge and includes several important open problems such as determination of number of components in Ad-mixtures. This paper addresses this challenge by introducing Interpolative Convex Rank(\INR) of a matrix defined as the minimum number of its columns whose convex hull is within Hausdorff distance $\varepsilon$ of the convex hull of all columns. The first important contribution of this paper is to show that under \emph{standard assumptions} $k$ equals the \INR of a \emph{subset smoothed data matrix} defined from Data generated from an $\LkP$. The second important contribution of the paper is a polynomial time algorithm for finding $k$ under standard assumptions. An immediate corollary is the first polynomial time algorithm for finding the \emph{inner dimension} in Non-negative matrix factorisation(NMF) with assumptions which are qualitatively different than existing ones such as \emph{Separability}. %An immediate corollary is the first polynomial time algorithm for finding the \emph{inner dimension} in Non-negative matrix factorisation(NMF) with assumptions considerably weaker than \emph{Separability}.

----

## [83] Non-Autoregressive Electron Redistribution Modeling for Reaction Prediction

**Authors**: *Hangrui Bi, Hengyi Wang, Chence Shi, Connor W. Coley, Jian Tang, Hongyu Guo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bi21a.html](http://proceedings.mlr.press/v139/bi21a.html)

**Abstract**:

Reliably predicting the products of chemical reactions presents a fundamental challenge in synthetic chemistry. Existing machine learning approaches typically produce a reaction product by sequentially forming its subparts or intermediate molecules. Such autoregressive methods, however, not only require a pre-defined order for the incremental construction but preclude the use of parallel decoding for efficient computation. To address these issues, we devise a non-autoregressive learning paradigm that predicts reaction in one shot. Leveraging the fact that chemical reactions can be described as a redistribution of electrons in molecules, we formulate a reaction as an arbitrary electron flow and predict it with a novel multi-pointer decoding network. Experiments on the USPTO-MIT dataset show that our approach has established a new state-of-the-art top-1 accuracy and achieves at least 27 times inference speedup over the state-of-the-art methods. Also, our predictions are easier for chemists to interpret owing to predicting the electron flows.

----

## [84] TempoRL: Learning When to Act

**Authors**: *André Biedenkapp, Raghu Rajan, Frank Hutter, Marius Lindauer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/biedenkapp21a.html](http://proceedings.mlr.press/v139/biedenkapp21a.html)

**Abstract**:

Reinforcement learning is a powerful approach to learn behaviour through interactions with an environment. However, behaviours are usually learned in a purely reactive fashion, where an appropriate action is selected based on an observation. In this form, it is challenging to learn when it is necessary to execute new decisions. This makes learning inefficient especially in environments that need various degrees of fine and coarse control. To address this, we propose a proactive setting in which the agent not only selects an action in a state but also for how long to commit to that action. Our TempoRL approach introduces skip connections between states and learns a skip-policy for repeating the same action along these skips. We demonstrate the effectiveness of TempoRL on a variety of traditional and deep RL environments, showing that our approach is capable of learning successful policies up to an order of magnitude faster than vanilla Q-learning.

----

## [85] Follow-the-Regularized-Leader Routes to Chaos in Routing Games

**Authors**: *Jakub Bielawski, Thiparat Chotibut, Fryderyk Falniowski, Grzegorz Kosiorowski, Michal Misiurewicz, Georgios Piliouras*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bielawski21a.html](http://proceedings.mlr.press/v139/bielawski21a.html)

**Abstract**:

We study the emergence of chaotic behavior of Follow-the-Regularized Leader (FoReL) dynamics in games. We focus on the effects of increasing the population size or the scale of costs in congestion games, and generalize recent results on unstable, chaotic behaviors in the Multiplicative Weights Update dynamics to a much larger class of FoReL dynamics. We establish that, even in simple linear non-atomic congestion games with two parallel links and \emph{any} fixed learning rate, unless the game is fully symmetric, increasing the population size or the scale of costs causes learning dynamics to becomes unstable and eventually chaotic, in the sense of Li-Yorke and positive topological entropy. Furthermore, we prove the existence of novel non-standard phenomena such as the coexistence of stable Nash equilibria and chaos in the same game. We also observe the simultaneous creation of a chaotic attractor as another chaotic attractor gets destroyed. Lastly, although FoReL dynamics can be strange and non-equilibrating, we prove that the time average still converges to an \emph{exact} equilibrium for any choice of learning rate and any scale of costs.

----

## [86] Neural Symbolic Regression that scales

**Authors**: *Luca Biggio, Tommaso Bendinelli, Alexander Neitz, Aurélien Lucchi, Giambattista Parascandolo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/biggio21a.html](http://proceedings.mlr.press/v139/biggio21a.html)

**Abstract**:

Symbolic equations are at the core of scientific discovery. The task of discovering the underlying equation from a set of input-output pairs is called symbolic regression. Traditionally, symbolic regression methods use hand-designed strategies that do not improve with experience. In this paper, we introduce the first symbolic regression method that leverages large scale pre-training. We procedurally generate an unbounded set of equations, and simultaneously pre-train a Transformer to predict the symbolic equation from a corresponding set of input-output-pairs. At test time, we query the model on a new set of points and use its output to guide the search for the equation. We show empirically that this approach can re-discover a set of well-known physical equations, and that it improves over time with more data and compute.

----

## [87] Model Distillation for Revenue Optimization: Interpretable Personalized Pricing

**Authors**: *Max Biggs, Wei Sun, Markus Ettl*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/biggs21a.html](http://proceedings.mlr.press/v139/biggs21a.html)

**Abstract**:

Data-driven pricing strategies are becoming increasingly common, where customers are offered a personalized price based on features that are predictive of their valuation of a product. It is desirable for this pricing policy to be simple and interpretable, so it can be verified, checked for fairness, and easily implemented. However, efforts to incorporate machine learning into a pricing framework often lead to complex pricing policies that are not interpretable, resulting in slow adoption in practice. We present a novel, customized, prescriptive tree-based algorithm that distills knowledge from a complex black-box machine learning algorithm, segments customers with similar valuations and prescribes prices in such a way that maximizes revenue while maintaining interpretability. We quantify the regret of a resulting policy and demonstrate its efficacy in applications with both synthetic and real-world datasets.

----

## [88] Scalable Normalizing Flows for Permutation Invariant Densities

**Authors**: *Marin Bilos, Stephan Günnemann*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bilos21a.html](http://proceedings.mlr.press/v139/bilos21a.html)

**Abstract**:

Modeling sets is an important problem in machine learning since this type of data can be found in many domains. A promising approach defines a family of permutation invariant densities with continuous normalizing flows. This allows us to maximize the likelihood directly and sample new realizations with ease. In this work, we demonstrate how calculating the trace, a crucial step in this method, raises issues that occur both during training and inference, limiting its practicality. We propose an alternative way of defining permutation equivariant transformations that give closed form trace. This leads not only to improvements while training, but also to better final performance. We demonstrate the benefits of our approach on point processes and general set modeling.

----

## [89] Online Learning for Load Balancing of Unknown Monotone Resource Allocation Games

**Authors**: *Ilai Bistritz, Nicholas Bambos*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bistritz21a.html](http://proceedings.mlr.press/v139/bistritz21a.html)

**Abstract**:

Consider N players that each uses a mixture of K resources. Each of the players’ reward functions includes a linear pricing term for each resource that is controlled by the game manager. We assume that the game is strongly monotone, so if each player runs gradient descent, the dynamics converge to a unique Nash equilibrium (NE). Unfortunately, this NE can be inefficient since the total load on a given resource can be very high. In principle, we can control the total loads by tuning the coefficients of the pricing terms. However, finding pricing coefficients that balance the loads requires knowing the players’ reward functions and their action sets. Obtaining this game structure information is infeasible in a large-scale network and violates the users’ privacy. To overcome this, we propose a simple algorithm that learns to shift the NE of the game to meet the total load constraints by adjusting the pricing coefficients in an online manner. Our algorithm only requires the total load per resource as feedback and does not need to know the reward functions or the action sets. We prove that our algorithm guarantees convergence in L2 to a NE that meets target total load constraints. Simulations show the effectiveness of our approach when applied to smart grid demand-side management or power control in wireless networks.

----

## [90] Low-Precision Reinforcement Learning: Running Soft Actor-Critic in Half Precision

**Authors**: *Johan Björck, Xiangyu Chen, Christopher De Sa, Carla P. Gomes, Kilian Q. Weinberger*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bjorck21a.html](http://proceedings.mlr.press/v139/bjorck21a.html)

**Abstract**:

Low-precision training has become a popular approach to reduce compute requirements, memory footprint, and energy consumption in supervised learning. In contrast, this promising approach has not yet enjoyed similarly widespread adoption within the reinforcement learning (RL) community, partly because RL agents can be notoriously hard to train even in full precision. In this paper we consider continuous control with the state-of-the-art SAC agent and demonstrate that a naïve adaptation of low-precision methods from supervised learning fails. We propose a set of six modifications, all straightforward to implement, that leaves the underlying agent and its hyperparameters unchanged but improves the numerical stability dramatically. The resulting modified SAC agent has lower memory and compute requirements while matching full-precision rewards, demonstrating that low-precision training can substantially accelerate state-of-the-art RL without parameter tuning.

----

## [91] Multiplying Matrices Without Multiplying

**Authors**: *Davis W. Blalock, John V. Guttag*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/blalock21a.html](http://proceedings.mlr.press/v139/blalock21a.html)

**Abstract**:

Multiplying matrices is among the most fundamental and most computationally demanding operations in machine learning and scientific computing. Consequently, the task of efficiently approximating matrix products has received significant attention. We introduce a learning-based algorithm for this task that greatly outperforms existing methods. Experiments using hundreds of matrices from diverse domains show that it often runs 10x faster than alternatives at a given level of error, as well as 100x faster than exact matrix multiplication. In the common case that one matrix is known ahead of time, our method also has the interesting property that it requires zero multiply-adds. These results suggest that a mixture of hashing, averaging, and byte shuffling{—}the core operations of our method{—}could be a more promising building block for machine learning than the sparsified, factorized, and/or scalar quantized matrix products that have recently been the focus of substantial research and hardware investment.

----

## [92] One for One, or All for All: Equilibria and Optimality of Collaboration in Federated Learning

**Authors**: *Avrim Blum, Nika Haghtalab, Richard Lanas Phillips, Han Shao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/blum21a.html](http://proceedings.mlr.press/v139/blum21a.html)

**Abstract**:

In recent years, federated learning has been embraced as an approach for bringing about collaboration across large populations of learning agents. However, little is known about how collaboration protocols should take agents’ incentives into account when allocating individual resources for communal learning in order to maintain such collaborations. Inspired by game theoretic notions, this paper introduces a framework for incentive-aware learning and data sharing in federated learning. Our stable and envy-free equilibria capture notions of collaboration in the presence of agents interested in meeting their learning objectives while keeping their own sample collection burden low. For example, in an envy-free equilibrium, no agent would wish to swap their sampling burden with any other agent and in a stable equilibrium, no agent would wish to unilaterally reduce their sampling burden. In addition to formalizing this framework, our contributions include characterizing the structural properties of such equilibria, proving when they exist, and showing how they can be computed. Furthermore, we compare the sample complexity of incentive-aware collaboration with that of optimal collaboration when one ignores agents’ incentives.

----

## [93] Black-box density function estimation using recursive partitioning

**Authors**: *Erik Bodin, Zhenwen Dai, Neill W. Campbell, Carl Henrik Ek*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bodin21a.html](http://proceedings.mlr.press/v139/bodin21a.html)

**Abstract**:

We present a novel approach to Bayesian inference and general Bayesian computation that is defined through a sequential decision loop. Our method defines a recursive partitioning of the sample space. It neither relies on gradients nor requires any problem-specific tuning, and is asymptotically exact for any density function with a bounded domain. The output is an approximation to the whole density function including the normalisation constant, via partitions organised in efficient data structures. Such approximations may be used for evidence estimation or fast posterior sampling, but also as building blocks to treat a larger class of estimation problems. The algorithm shows competitive performance to recent state-of-the-art methods on synthetic and real-world problems including parameter inference for gravitational-wave physics.

----

## [94] Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks

**Authors**: *Cristian Bodnar, Fabrizio Frasca, Yuguang Wang, Nina Otter, Guido F. Montúfar, Pietro Lió, Michael M. Bronstein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bodnar21a.html](http://proceedings.mlr.press/v139/bodnar21a.html)

**Abstract**:

The pairwise interaction paradigm of graph machine learning has predominantly governed the modelling of relational systems. However, graphs alone cannot capture the multi-level interactions present in many complex systems and the expressive power of such schemes was proven to be limited. To overcome these limitations, we propose Message Passing Simplicial Networks (MPSNs), a class of models that perform message passing on simplicial complexes (SCs). To theoretically analyse the expressivity of our model we introduce a Simplicial Weisfeiler-Lehman (SWL) colouring procedure for distinguishing non-isomorphic SCs. We relate the power of SWL to the problem of distinguishing non-isomorphic graphs and show that SWL and MPSNs are strictly more powerful than the WL test and not less powerful than the 3-WL test. We deepen the analysis by comparing our model with traditional graph neural networks (GNNs) with ReLU activations in terms of the number of linear regions of the functions they can represent. We empirically support our theoretical claims by showing that MPSNs can distinguish challenging strongly regular graphs for which GNNs fail and, when equipped with orientation equivariant layers, they can improve classification accuracy in oriented SCs compared to a GNN baseline.

----

## [95] The Hintons in your Neural Network: a Quantum Field Theory View of Deep Learning

**Authors**: *Roberto Bondesan, Max Welling*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bondesan21a.html](http://proceedings.mlr.press/v139/bondesan21a.html)

**Abstract**:

In this work we develop a quantum field theory formalism for deep learning, where input signals are encoded in Gaussian states, a generalization of Gaussian processes which encode the agent’s uncertainty about the input signal. We show how to represent linear and non-linear layers as unitary quantum gates, and interpret the fundamental excitations of the quantum model as particles, dubbed “Hintons”. On top of opening a new perspective and techniques for studying neural networks, the quantum formulation is well suited for optical quantum computing, and provides quantum deformations of neural networks that can be run efficiently on those devices. Finally, we discuss a semi-classical limit of the quantum deformed models which is amenable to classical simulation.

----

## [96] Offline Contextual Bandits with Overparameterized Models

**Authors**: *David Brandfonbrener, William F. Whitney, Rajesh Ranganath, Joan Bruna*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/brandfonbrener21a.html](http://proceedings.mlr.press/v139/brandfonbrener21a.html)

**Abstract**:

Recent results in supervised learning suggest that while overparameterized models have the capacity to overfit, they in fact generalize quite well. We ask whether the same phenomenon occurs for offline contextual bandits. Our results are mixed. Value-based algorithms benefit from the same generalization behavior as overparameterized supervised learning, but policy-based algorithms do not. We show that this discrepancy is due to the \emph{action-stability} of their objectives. An objective is action-stable if there exists a prediction (action-value vector or action distribution) which is optimal no matter which action is observed. While value-based objectives are action-stable, policy-based objectives are unstable. We formally prove upper bounds on the regret of overparameterized value-based learning and lower bounds on the regret for policy-based algorithms. In our experiments with large neural networks, this gap between action-stable value-based objectives and unstable policy-based objectives leads to significant performance differences.

----

## [97] High-Performance Large-Scale Image Recognition Without Normalization

**Authors**: *Andy Brock, Soham De, Samuel L. Smith, Karen Simonyan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/brock21a.html](http://proceedings.mlr.press/v139/brock21a.html)

**Abstract**:

Batch normalization is a key component of most image classification models, but it has many undesirable properties stemming from its dependence on the batch size and interactions between examples. Although recent work has succeeded in training deep ResNets without normalization layers, these models do not match the test accuracies of the best batch-normalized networks, and are often unstable for large learning rates or strong data augmentations. In this work, we develop an adaptive gradient clipping technique which overcomes these instabilities, and design a significantly improved class of Normalizer-Free ResNets. Our smaller models match the test accuracy of an EfficientNet-B7 on ImageNet while being up to 8.7x faster to train, and our largest models attain a new state-of-the-art top-1 accuracy of 86.5%. In addition, Normalizer-Free models attain significantly better performance than their batch-normalized counterparts when fine-tuning on ImageNet after large-scale pre-training on a dataset of 300 million labeled images, with our best models obtaining an accuracy of 89.2%.

----

## [98] Evaluating the Implicit Midpoint Integrator for Riemannian Hamiltonian Monte Carlo

**Authors**: *James A. Brofos, Roy R. Lederman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/brofos21a.html](http://proceedings.mlr.press/v139/brofos21a.html)

**Abstract**:

Riemannian manifold Hamiltonian Monte Carlo is traditionally carried out using the generalized leapfrog integrator. However, this integrator is not the only choice and other integrators yielding valid Markov chain transition operators may be considered. In this work, we examine the implicit midpoint integrator as an alternative to the generalized leapfrog integrator. We discuss advantages and disadvantages of the implicit midpoint integrator for Hamiltonian Monte Carlo, its theoretical properties, and an empirical assessment of the critical attributes of such an integrator for Hamiltonian Monte Carlo: energy conservation, volume preservation, and reversibility. Empirically, we find that while leapfrog iterations are faster, the implicit midpoint integrator has better energy conservation, leading to higher acceptance rates, as well as better conservation of volume and better reversibility, arguably yielding a more accurate sampling procedure.

----

## [99] Reinforcement Learning of Implicit and Explicit Control Flow Instructions

**Authors**: *Ethan A. Brooks, Janarthanan Rajendran, Richard L. Lewis, Satinder Singh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/brooks21a.html](http://proceedings.mlr.press/v139/brooks21a.html)

**Abstract**:

Learning to flexibly follow task instructions in dynamic environments poses interesting challenges for reinforcement learning agents. We focus here on the problem of learning control flow that deviates from a strict step-by-step execution of instructions{—}that is, control flow that may skip forward over parts of the instructions or return backward to previously completed or skipped steps. Demand for such flexible control arises in two fundamental ways: explicitly when control is specified in the instructions themselves (such as conditional branching and looping) and implicitly when stochastic environment dynamics require re-completion of instructions whose effects have been perturbed, or opportunistic skipping of instructions whose effects are already present. We formulate an attention-based architecture that meets these challenges by learning, from task reward only, to flexibly attend to and condition behavior on an internal encoding of the instructions. We test the architecture’s ability to learn both explicit and implicit control in two illustrative domains—one inspired by Minecraft and the other by StarCraft—and show that the architecture exhibits zero-shot generalization to novel instructions of length greater than those in a training set, at a performance level unmatched by three baseline recurrent architectures and one ablation architecture.

----

## [100] Machine Unlearning for Random Forests

**Authors**: *Jonathan Brophy, Daniel Lowd*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/brophy21a.html](http://proceedings.mlr.press/v139/brophy21a.html)

**Abstract**:

Responding to user data deletion requests, removing noisy examples, or deleting corrupted training data are just a few reasons for wanting to delete instances from a machine learning (ML) model. However, efficiently removing this data from an ML model is generally difficult. In this paper, we introduce data removal-enabled (DaRE) forests, a variant of random forests that enables the removal of training data with minimal retraining. Model updates for each DaRE tree in the forest are exact, meaning that removing instances from a DaRE model yields exactly the same model as retraining from scratch on updated data. DaRE trees use randomness and caching to make data deletion efficient. The upper levels of DaRE trees use random nodes, which choose split attributes and thresholds uniformly at random. These nodes rarely require updates because they only minimally depend on the data. At the lower levels, splits are chosen to greedily optimize a split criterion such as Gini index or mutual information. DaRE trees cache statistics at each node and training data at each leaf, so that only the necessary subtrees are updated as data is removed. For numerical attributes, greedy nodes optimize over a random subset of thresholds, so that they can maintain statistics while approximating the optimal threshold. By adjusting the number of thresholds considered for greedy nodes, and the number of random nodes, DaRE trees can trade off between more accurate predictions and more efficient updates. In experiments on 13 real-world datasets and one synthetic dataset, we find DaRE forests delete data orders of magnitude faster than retraining from scratch while sacrificing little to no predictive power.

----

## [101] Value Alignment Verification

**Authors**: *Daniel S. Brown, Jordan Schneider, Anca D. Dragan, Scott Niekum*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/brown21a.html](http://proceedings.mlr.press/v139/brown21a.html)

**Abstract**:

As humans interact with autonomous agents to perform increasingly complicated, potentially risky tasks, it is important to be able to efficiently evaluate an agent’s performance and correctness. In this paper we formalize and theoretically analyze the problem of efficient value alignment verification: how to efficiently test whether the behavior of another agent is aligned with a human’s values? The goal is to construct a kind of "driver’s test" that a human can give to any agent which will verify value alignment via a minimal number of queries. We study alignment verification problems with both idealized humans that have an explicit reward function as well as problems where they have implicit values. We analyze verification of exact value alignment for rational agents, propose and test heuristics for value alignment verification in gridworlds and a continuous autonomous driving domain, and prove that there exist sufficient conditions such that we can verify epsilon-alignment in any environment via a constant-query-complexity alignment test.

----

## [102] Model-Free and Model-Based Policy Evaluation when Causality is Uncertain

**Authors**: *David Bruns-Smith*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bruns-smith21a.html](http://proceedings.mlr.press/v139/bruns-smith21a.html)

**Abstract**:

When decision-makers can directly intervene, policy evaluation algorithms give valid causal estimates. In off-policy evaluation (OPE), there may exist unobserved variables that both impact the dynamics and are used by the unknown behavior policy. These “confounders” will introduce spurious correlations and naive estimates for a new policy will be biased. We develop worst-case bounds to assess sensitivity to these unobserved confounders in finite horizons when confounders are drawn iid each period. We demonstrate that a model-based approach with robust MDPs gives sharper lower bounds by exploiting domain knowledge about the dynamics. Finally, we show that when unobserved confounders are persistent over time, OPE is far more difficult and existing techniques produce extremely conservative bounds.

----

## [103] Narrow Margins: Classification, Margins and Fat Tails

**Authors**: *Francois Buet-Golfouse*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/buet-golfouse21a.html](http://proceedings.mlr.press/v139/buet-golfouse21a.html)

**Abstract**:

It is well-known that, for separable data, the regularised two-class logistic regression or support vector machine re-normalised estimate converges to the maximal margin classifier as the regularisation hyper-parameter $\lambda$ goes to 0. The fact that different loss functions may lead to the same solution is of theoretical and practical relevance as margin maximisation allows more straightforward considerations in terms of generalisation and geometric interpretation. We investigate the case where this convergence property is not guaranteed to hold and show that it can be fully characterised by the distribution of error terms in the latent variable interpretation of linear classifiers. In particular, if errors follow a regularly varying distribution, then the regularised and re-normalised estimate does not converge to the maximal margin classifier. This shows that classification with fat tails has a qualitatively different behaviour, which should be taken into account when considering real-life data.

----

## [104] Differentially Private Correlation Clustering

**Authors**: *Mark Bun, Marek Eliás, Janardhan Kulkarni*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/bun21a.html](http://proceedings.mlr.press/v139/bun21a.html)

**Abstract**:

Correlation clustering is a widely used technique in unsupervised machine learning. Motivated by applications where individual privacy is a concern, we initiate the study of differentially private correlation clustering. We propose an algorithm that achieves subquadratic additive error compared to the optimal cost. In contrast, straightforward adaptations of existing non-private algorithms all lead to a trivial quadratic error. Finally, we give a lower bound showing that any pure differentially private algorithm for correlation clustering requires additive error $\Omega$(n).

----

## [105] Disambiguation of Weak Supervision leading to Exponential Convergence rates

**Authors**: *Vivien A. Cabannes, Francis R. Bach, Alessandro Rudi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cabannnes21a.html](http://proceedings.mlr.press/v139/cabannnes21a.html)

**Abstract**:

Machine learning approached through supervised learning requires expensive annotation of data. This motivates weakly supervised learning, where data are annotated with incomplete yet discriminative information. In this paper, we focus on partial labelling, an instance of weak supervision where, from a given input, we are given a set of potential targets. We review a disambiguation principle to recover full supervision from weak supervision, and propose an empirical disambiguation algorithm. We prove exponential convergence rates of our algorithm under classical learnability assumptions, and we illustrate the usefulness of our method on practical examples.

----

## [106] Finite mixture models do not reliably learn the number of components

**Authors**: *Diana Cai, Trevor Campbell, Tamara Broderick*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cai21a.html](http://proceedings.mlr.press/v139/cai21a.html)

**Abstract**:

Scientists and engineers are often interested in learning the number of subpopulations (or components) present in a data set. A common suggestion is to use a finite mixture model (FMM) with a prior on the number of components. Past work has shown the resulting FMM component-count posterior is consistent; that is, the posterior concentrates on the true, generating number of components. But consistency requires the assumption that the component likelihoods are perfectly specified, which is unrealistic in practice. In this paper, we add rigor to data-analysis folk wisdom by proving that under even the slightest model misspecification, the FMM component-count posterior diverges: the posterior probability of any particular finite number of components converges to 0 in the limit of infinite data. Contrary to intuition, posterior-density consistency is not sufficient to establish this result. We develop novel sufficient conditions that are more realistic and easily checkable than those common in the asymptotics literature. We illustrate practical consequences of our theory on simulated and real data.

----

## [107] A Theory of Label Propagation for Subpopulation Shift

**Authors**: *Tianle Cai, Ruiqi Gao, Jason D. Lee, Qi Lei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cai21b.html](http://proceedings.mlr.press/v139/cai21b.html)

**Abstract**:

One of the central problems in machine learning is domain adaptation. Different from past theoretical works, we consider a new model of subpopulation shift in the input or representation space. In this work, we propose a provably effective framework based on label propagation by using an input consistency loss. In our analysis we used a simple but realistic “expansion” assumption, which has been proposed in \citet{wei2021theoretical}. It turns out that based on a teacher classifier on the source domain, the learned classifier can not only propagate to the target domain but also improve upon the teacher. By leveraging existing generalization bounds, we also obtain end-to-end finite-sample guarantees on deep neural networks. In addition, we extend our theoretical framework to a more general setting of source-to-target transfer based on an additional unlabeled dataset, which can be easily applied to various learning scenarios. Inspired by our theory, we adapt consistency-based semi-supervised learning methods to domain adaptation settings and gain significant improvements.

----

## [108] Lenient Regret and Good-Action Identification in Gaussian Process Bandits

**Authors**: *Xu Cai, Selwyn Gomes, Jonathan Scarlett*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cai21c.html](http://proceedings.mlr.press/v139/cai21c.html)

**Abstract**:

In this paper, we study the problem of Gaussian process (GP) bandits under relaxed optimization criteria stating that any function value above a certain threshold is “good enough”. On the theoretical side, we study various {\em lenient regret} notions in which all near-optimal actions incur zero penalty, and provide upper bounds on the lenient regret for GP-UCB and an elimination algorithm, circumventing the usual $O(\sqrt{T})$ term (with time horizon $T$) resulting from zooming extremely close towards the function maximum. In addition, we complement these upper bounds with algorithm-independent lower bounds. On the practical side, we consider the problem of finding a single “good action” according to a known pre-specified threshold, and introduce several good-action identification algorithms that exploit knowledge of the threshold. We experimentally find that such algorithms can typically find a good action faster than standard optimization-based approaches.

----

## [109] A Zeroth-Order Block Coordinate Descent Algorithm for Huge-Scale Black-Box Optimization

**Authors**: *HanQin Cai, Yuchen Lou, Daniel McKenzie, Wotao Yin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cai21d.html](http://proceedings.mlr.press/v139/cai21d.html)

**Abstract**:

We consider the zeroth-order optimization problem in the huge-scale setting, where the dimension of the problem is so large that performing even basic vector operations on the decision variables is infeasible. In this paper, we propose a novel algorithm, coined ZO-BCD, that exhibits favorable overall query complexity and has a much smaller per-iteration computational complexity. In addition, we discuss how the memory footprint of ZO-BCD can be reduced even further by the clever use of circulant measurement matrices. As an application of our new method, we propose the idea of crafting adversarial attacks on neural network based classifiers in a wavelet domain, which can result in problem dimensions of over one million. In particular, we show that crafting adversarial examples to audio classifiers in a wavelet domain can achieve the state-of-the-art attack success rate of 97.9% with significantly less distortion.

----

## [110] GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training

**Authors**: *Tianle Cai, Shengjie Luo, Keyulu Xu, Di He, Tie-Yan Liu, Liwei Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cai21e.html](http://proceedings.mlr.press/v139/cai21e.html)

**Abstract**:

Normalization is known to help the optimization of deep neural networks. Curiously, different architectures require specialized normalization methods. In this paper, we study what normalization is effective for Graph Neural Networks (GNNs). First, we adapt and evaluate the existing methods from other domains to GNNs. Faster convergence is achieved with InstanceNorm compared to BatchNorm and LayerNorm. We provide an explanation by showing that InstanceNorm serves as a preconditioner for GNNs, but such preconditioning effect is weaker with BatchNorm due to the heavy batch noise in graph datasets. Second, we show that the shift operation in InstanceNorm results in an expressiveness degradation of GNNs for highly regular graphs. We address this issue by proposing GraphNorm with a learnable shift. Empirically, GNNs with GraphNorm converge faster compared to GNNs using other normalization. GraphNorm also improves the generalization of GNNs, achieving better performance on graph classification benchmarks.

----

## [111] On Lower Bounds for Standard and Robust Gaussian Process Bandit Optimization

**Authors**: *Xu Cai, Jonathan Scarlett*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cai21f.html](http://proceedings.mlr.press/v139/cai21f.html)

**Abstract**:

In this paper, we consider algorithm independent lower bounds for the problem of black-box optimization of functions having a bounded norm is some Reproducing Kernel Hilbert Space (RKHS), which can be viewed as a non-Bayesian Gaussian process bandit problem. In the standard noisy setting, we provide a novel proof technique for deriving lower bounds on the regret, with benefits including simplicity, versatility, and an improved dependence on the error probability. In a robust setting in which the final point is perturbed by an adversary, we strengthen an existing lower bound that only holds for target success probabilities very close to one, by allowing for arbitrary target success probabilities in (0, 1). Furthermore, in a distinct robust setting in which every sampled point may be perturbed by a constrained adversary, we provide a novel lower bound for deterministic strategies, demonstrating an inevitable joint dependence of the cumulative regret on the corruption level and the time horizon, in contrast with existing lower bounds that only characterize the individual dependencies.

----

## [112] High-dimensional Experimental Design and Kernel Bandits

**Authors**: *Romain Camilleri, Kevin Jamieson, Julian Katz-Samuels*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/camilleri21a.html](http://proceedings.mlr.press/v139/camilleri21a.html)

**Abstract**:

In recent years methods from optimal linear experimental design have been leveraged to obtain state of the art results for linear bandits. A design returned from an objective such as G-optimal design is actually a probability distribution over a pool of potential measurement vectors. Consequently, one nuisance of the approach is the task of converting this continuous probability distribution into a discrete assignment of N measurements. While sophisticated rounding techniques have been proposed, in d dimensions they require N to be at least d, d log(log(d)), or d^2 based on the sub-optimality of the solution. In this paper we are interested in settings where N may be much less than d, such as in experimental design in an RKHS where d may be effectively infinite. In this work, we propose a rounding procedure that frees N of any dependence on the dimension d, while achieving nearly the same performance guarantees of existing rounding procedures. We evaluate the procedure against a baseline that projects the problem to a lower dimensional space and performs rounding there, which requires N to just be at least a notion of the effective dimension. We also leverage our new approach in a new algorithm for kernelized bandits to obtain state of the art results for regret minimization and pure exploration. An advantage of our approach over existing UCB-like approaches is that our kernel bandit algorithms are provably robust to model misspecification.

----

## [113] A Gradient Based Strategy for Hamiltonian Monte Carlo Hyperparameter Optimization

**Authors**: *Andrew Campbell, Wenlong Chen, Vincent Stimper, José Miguel Hernández-Lobato, Yichuan Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/campbell21a.html](http://proceedings.mlr.press/v139/campbell21a.html)

**Abstract**:

Hamiltonian Monte Carlo (HMC) is one of the most successful sampling methods in machine learning. However, its performance is significantly affected by the choice of hyperparameter values. Existing approaches for optimizing the HMC hyperparameters either optimize a proxy for mixing speed or consider the HMC chain as an implicit variational distribution and optimize a tractable lower bound that can be very loose in practice. Instead, we propose to optimize an objective that quantifies directly the speed of convergence to the target distribution. Our objective can be easily optimized using stochastic gradient descent. We evaluate our proposed method and compare to baselines on a variety of problems including sampling from synthetic 2D distributions, reconstructing sparse signals, learning deep latent variable models and sampling molecular configurations from the Boltzmann distribution of a 22 atom molecule. We find that our method is competitive with or improves upon alternative baselines in all these experiments.

----

## [114] Asymmetric Heavy Tails and Implicit Bias in Gaussian Noise Injections

**Authors**: *Alexander Camuto, Xiaoyu Wang, Lingjiong Zhu, Chris C. Holmes, Mert Gürbüzbalaban, Umut Simsekli*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/camuto21a.html](http://proceedings.mlr.press/v139/camuto21a.html)

**Abstract**:

Gaussian noise injections (GNIs) are a family of simple and widely-used regularisation methods for training neural networks, where one injects additive or multiplicative Gaussian noise to the network activations at every iteration of the optimisation algorithm, which is typically chosen as stochastic gradient descent (SGD). In this paper, we focus on the so-called ‘implicit effect’ of GNIs, which is the effect of the injected noise on the dynamics of SGD. We show that this effect induces an \emph{asymmetric heavy-tailed noise} on SGD gradient updates. In order to model this modified dynamics, we first develop a Langevin-like stochastic differential equation that is driven by a general family of \emph{asymmetric} heavy-tailed noise. Using this model we then formally prove that GNIs induce an ‘implicit bias’, which varies depending on the heaviness of the tails and the level of asymmetry. Our empirical results confirm that different types of neural networks trained with GNIs are well-modelled by the proposed dynamics and that the implicit effect of these injections induces a bias that degrades the performance of networks.

----

## [115] Fold2Seq: A Joint Sequence(1D)-Fold(3D) Embedding-based Generative Model for Protein Design

**Authors**: *Yue Cao, Payel Das, Vijil Chenthamarakshan, Pin-Yu Chen, Igor Melnyk, Yang Shen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cao21a.html](http://proceedings.mlr.press/v139/cao21a.html)

**Abstract**:

Designing novel protein sequences for a desired 3D topological fold is a fundamental yet non-trivial task in protein engineering. Challenges exist due to the complex sequence–fold relationship, as well as the difficulties to capture the diversity of the sequences (therefore structures and functions) within a fold. To overcome these challenges, we propose Fold2Seq, a novel transformer-based generative framework for designing protein sequences conditioned on a specific target fold. To model the complex sequence–structure relationship, Fold2Seq jointly learns a sequence embedding using a transformer and a fold embedding from the density of secondary structural elements in 3D voxels. On test sets with single, high-resolution and complete structure inputs for individual folds, our experiments demonstrate improved or comparable performance of Fold2Seq in terms of speed, coverage, and reliability for sequence design, when compared to existing state-of-the-art methods that include data-driven deep generative models and physics-based RosettaDesign. The unique advantages of fold-based Fold2Seq, in comparison to a structure-based deep model and RosettaDesign, become more evident on three additional real-world challenges originating from low-quality, incomplete, or ambiguous input structures. Source code and data are available at https://github.com/IBM/fold2seq.

----

## [116] Learning from Similarity-Confidence Data

**Authors**: *Yuzhou Cao, Lei Feng, Yitian Xu, Bo An, Gang Niu, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cao21b.html](http://proceedings.mlr.press/v139/cao21b.html)

**Abstract**:

Weakly supervised learning has drawn considerable attention recently to reduce the expensive time and labor consumption of labeling massive data. In this paper, we investigate a novel weakly supervised learning problem of learning from similarity-confidence (Sconf) data, where only unlabeled data pairs equipped with confidence that illustrates their degree of similarity (two examples are similar if they belong to the same class) are needed for training a discriminative binary classifier. We propose an unbiased estimator of the classification risk that can be calculated from only Sconf data and show that the estimation error bound achieves the optimal convergence rate. To alleviate potential overfitting when flexible models are used, we further employ a risk correction scheme on the proposed risk estimator. Experimental results demonstrate the effectiveness of the proposed methods.

----

## [117] Parameter-free Locally Accelerated Conditional Gradients

**Authors**: *Alejandro Carderera, Jelena Diakonikolas, Cheuk Yin Lin, Sebastian Pokutta*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/carderera21a.html](http://proceedings.mlr.press/v139/carderera21a.html)

**Abstract**:

Projection-free conditional gradient (CG) methods are the algorithms of choice for constrained optimization setups in which projections are often computationally prohibitive but linear optimization over the constraint set remains computationally feasible. Unlike in projection-based methods, globally accelerated convergence rates are in general unattainable for CG. However, a very recent work on Locally accelerated CG (LaCG) has demonstrated that local acceleration for CG is possible for many settings of interest. The main downside of LaCG is that it requires knowledge of the smoothness and strong convexity parameters of the objective function. We remove this limitation by introducing a novel, Parameter-Free Locally accelerated CG (PF-LaCG) algorithm, for which we provide rigorous convergence guarantees. Our theoretical results are complemented by numerical experiments, which demonstrate local acceleration and showcase the practical improvements of PF-LaCG over non-accelerated algorithms, both in terms of iteration count and wall-clock time.

----

## [118] Optimizing persistent homology based functions

**Authors**: *Mathieu Carrière, Frédéric Chazal, Marc Glisse, Yuichi Ike, Hariprasad Kannan, Yuhei Umeda*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/carriere21a.html](http://proceedings.mlr.press/v139/carriere21a.html)

**Abstract**:

Solving optimization tasks based on functions and losses with a topological flavor is a very active and growing field of research in data science and Topological Data Analysis, with applications in non-convex optimization, statistics and machine learning. However, the approaches proposed in the literature are usually anchored to a specific application and/or topological construction, and do not come with theoretical guarantees. To address this issue, we study the differentiability of a general map associated with the most common topological construction, that is, the persistence map. Building on real analytic geometry arguments, we propose a general framework that allows us to define and compute gradients for persistence-based functions in a very simple way. We also provide a simple, explicit and sufficient condition for convergence of stochastic subgradient methods for such functions. This result encompasses all the constructions and applications of topological optimization in the literature. Finally, we provide associated code, that is easy to handle and to mix with other non-topological methods and constraints, as well as some experiments showcasing the versatility of our approach.

----

## [119] Online Policy Gradient for Model Free Learning of Linear Quadratic Regulators with √T Regret

**Authors**: *Asaf B. Cassel, Tomer Koren*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cassel21a.html](http://proceedings.mlr.press/v139/cassel21a.html)

**Abstract**:

We consider the task of learning to control a linear dynamical system under fixed quadratic costs, known as the Linear Quadratic Regulator (LQR) problem. While model-free approaches are often favorable in practice, thus far only model-based methods, which rely on costly system identification, have been shown to achieve regret that scales with the optimal dependence on the time horizon T. We present the first model-free algorithm that achieves similar regret guarantees. Our method relies on an efficient policy gradient scheme, and a novel and tighter analysis of the cost of exploration in policy space in this setting.

----

## [120] Multi-Receiver Online Bayesian Persuasion

**Authors**: *Matteo Castiglioni, Alberto Marchesi, Andrea Celli, Nicola Gatti*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/castiglioni21a.html](http://proceedings.mlr.press/v139/castiglioni21a.html)

**Abstract**:

Bayesian persuasion studies how an informed sender should partially disclose information to influence the behavior of a self-interested receiver. Classical models make the stringent assumption that the sender knows the receiver’s utility. This can be relaxed by considering an online learning framework in which the sender repeatedly faces a receiver of an unknown, adversarially selected type. We study, for the first time, an online Bayesian persuasion setting with multiple receivers. We focus on the case with no externalities and binary actions, as customary in offline models. Our goal is to design no-regret algorithms for the sender with polynomial per-iteration running time. First, we prove a negative result: for any 0 < $\alpha$ $\leq$ 1, there is no polynomial-time no-$\alpha$-regret algorithm when the sender’s utility function is supermodular or anonymous. Then, we focus on the setting of submodular sender’s utility functions and we show that, in this case, it is possible to design a polynomial-time no-(1-1/e)-regret algorithm. To do so, we introduce a general online gradient descent framework to handle online learning problems with a finite number of possible loss functions. This requires the existence of an approximate projection oracle. We show that, in our setting, there exists one such projection oracle which can be implemented in polynomial time.

----

## [121] Marginal Contribution Feature Importance - an Axiomatic Approach for Explaining Data

**Authors**: *Amnon Catav, Boyang Fu, Yazeed Zoabi, Ahuva Weiss-Meilik, Noam Shomron, Jason Ernst, Sriram Sankararaman, Ran Gilad-Bachrach*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/catav21a.html](http://proceedings.mlr.press/v139/catav21a.html)

**Abstract**:

In recent years, methods were proposed for assigning feature importance scores to measure the contribution of individual features. While in some cases the goal is to understand a specific model, in many cases the goal is to understand the contribution of certain properties (features) to a real-world phenomenon. Thus, a distinction has been made between feature importance scores that explain a model and scores that explain the data. When explaining the data, machine learning models are used as proxies in settings where conducting many real-world experiments is expensive or prohibited. While existing feature importance scores show great success in explaining models, we demonstrate their limitations when explaining the data, especially in the presence of correlations between features. Therefore, we develop a set of axioms to capture properties expected from a feature importance score when explaining data and prove that there exists only one score that satisfies all of them, the Marginal Contribution Feature Importance (MCI). We analyze the theoretical properties of this score function and demonstrate its merits empirically.

----

## [122] Disentangling syntax and semantics in the brain with deep networks

**Authors**: *Charlotte Caucheteux, Alexandre Gramfort, Jean-Remi King*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/caucheteux21a.html](http://proceedings.mlr.press/v139/caucheteux21a.html)

**Abstract**:

The activations of language transformers like GPT-2 have been shown to linearly map onto brain activity during speech comprehension. However, the nature of these activations remains largely unknown and presumably conflate distinct linguistic classes. Here, we propose a taxonomy to factorize the high-dimensional activations of language models into four combinatorial classes: lexical, compositional, syntactic, and semantic representations. We then introduce a statistical method to decompose, through the lens of GPT-2’s activations, the brain activity of 345 subjects recorded with functional magnetic resonance imaging (fMRI) during the listening of  4.6 hours of narrated text. The results highlight two findings. First, compositional representations recruit a more widespread cortical network than lexical ones, and encompass the bilateral temporal, parietal and prefrontal cortices. Second, contrary to previous claims, syntax and semantics are not associated with separated modules, but, instead, appear to share a common and distributed neural substrate. Overall, this study introduces a versatile framework to isolate, in the brain activity, the distributed representations of linguistic constructs.

----

## [123] Fair Classification with Noisy Protected Attributes: A Framework with Provable Guarantees

**Authors**: *L. Elisa Celis, Lingxiao Huang, Vijay Keswani, Nisheeth K. Vishnoi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/celis21a.html](http://proceedings.mlr.press/v139/celis21a.html)

**Abstract**:

We present an optimization framework for learning a fair classifier in the presence of noisy perturbations in the protected attributes. Compared to prior work, our framework can be employed with a very general class of linear and linear-fractional fairness constraints, can handle multiple, non-binary protected attributes, and outputs a classifier that comes with provable guarantees on both accuracy and fairness. Empirically, we show that our framework can be used to attain either statistical rate or false positive rate fairness guarantees with a minimal loss in accuracy, even when the noise is large, in two real-world datasets.

----

## [124] Best Model Identification: A Rested Bandit Formulation

**Authors**: *Leonardo Cella, Massimiliano Pontil, Claudio Gentile*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cella21a.html](http://proceedings.mlr.press/v139/cella21a.html)

**Abstract**:

We introduce and analyze a best arm identification problem in the rested bandit setting, wherein arms are themselves learning algorithms whose expected losses decrease with the number of times the arm has been played. The shape of the expected loss functions is similar across arms, and is assumed to be available up to unknown parameters that have to be learned on the fly. We define a novel notion of regret for this problem, where we compare to the policy that always plays the arm having the smallest expected loss at the end of the game. We analyze an arm elimination algorithm whose regret vanishes as the time horizon increases. The actual rate of convergence depends in a detailed way on the postulated functional form of the expected losses. We complement our analysis with lower bounds, indicating strengths and limitations of the proposed solution.

----

## [125] Revisiting Rainbow: Promoting more insightful and inclusive deep reinforcement learning research

**Authors**: *Johan Samir Obando-Ceron, Pablo Samuel Castro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ceron21a.html](http://proceedings.mlr.press/v139/ceron21a.html)

**Abstract**:

Since the introduction of DQN, a vast majority of reinforcement learning research has focused on reinforcement learning with deep neural networks as function approximators. New methods are typically evaluated on a set of environments that have now become standard, such as Atari 2600 games. While these benchmarks help standardize evaluation, their computational cost has the unfortunate side effect of widening the gap between those with ample access to computational resources, and those without. In this work we argue that, despite the community’s emphasis on large-scale environments, the traditional small-scale environments can still yield valuable scientific insights and can help reduce the barriers to entry for underprivileged communities. To substantiate our claims, we empirically revisit the paper which introduced the Rainbow algorithm [Hessel et al., 2018] and present some new insights into the algorithms used by Rainbow.

----

## [126] Learning Routines for Effective Off-Policy Reinforcement Learning

**Authors**: *Edoardo Cetin, Oya Çeliktutan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cetin21a.html](http://proceedings.mlr.press/v139/cetin21a.html)

**Abstract**:

The performance of reinforcement learning depends upon designing an appropriate action space, where the effect of each action is measurable, yet, granular enough to permit flexible behavior. So far, this process involved non-trivial user choices in terms of the available actions and their execution frequency. We propose a novel framework for reinforcement learning that effectively lifts such constraints. Within our framework, agents learn effective behavior over a routine space: a new, higher-level action space, where each routine represents a set of ’equivalent’ sequences of granular actions with arbitrary length. Our routine space is learned end-to-end to facilitate the accomplishment of underlying off-policy reinforcement learning objectives. We apply our framework to two state-of-the-art off-policy algorithms and show that the resulting agents obtain relevant performance improvements while requiring fewer interactions with the environment per episode, improving computational efficiency.

----

## [127] Learning Node Representations Using Stationary Flow Prediction on Large Payment and Cash Transaction Networks

**Authors**: *Ciwan Ceylan, Salla Franzén, Florian T. Pokorny*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ceylan21a.html](http://proceedings.mlr.press/v139/ceylan21a.html)

**Abstract**:

Banks are required to analyse large transaction datasets as a part of the fight against financial crime. Today, this analysis is either performed manually by domain experts or using expensive feature engineering. Gradient flow analysis allows for basic representation learning as node potentials can be inferred directly from network transaction data. However, the gradient model has a fundamental limitation: it cannot represent all types of of network flows. Furthermore, standard methods for learning the gradient flow are not appropriate for flow signals that span multiple orders of magnitude and contain outliers, i.e. transaction data. In this work, the gradient model is extended to a gated version and we prove that it, unlike the gradient model, is a universal approximator for flows on graphs. To tackle the mentioned challenges of transaction data, we propose a multi-scale and outlier robust loss function based on the Student-t log-likelihood. Ethereum transaction data is used for evaluation and the gradient models outperform MLP models using hand-engineered and node2vec features in terms of relative error. These results extend to 60 synthetic datasets, with experiments also showing that the gated gradient model learns qualitative information about the underlying synthetic generative flow distributions.

----

## [128] GRAND: Graph Neural Diffusion

**Authors**: *Ben Chamberlain, James Rowbottom, Maria I. Gorinova, Michael M. Bronstein, Stefan Webb, Emanuele Rossi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chamberlain21a.html](http://proceedings.mlr.press/v139/chamberlain21a.html)

**Abstract**:

We present Graph Neural Diffusion (GRAND) that approaches deep learning on graphs as a continuous diffusion process and treats Graph Neural Networks (GNNs) as discretisations of an underlying PDE. In our model, the layer structure and topology correspond to the discretisation choices of temporal and spatial operators. Our approach allows a principled development of a broad new class of GNNs that are able to address the common plights of graph learning models such as depth, oversmoothing, and bottlenecks. Key to the success of our models are stability with respect to perturbations in the data and this is addressed for both implicit and explicit discretisation schemes. We develop linear and nonlinear versions of GRAND, which achieve competitive results on many standard graph benchmarks.

----

## [129] HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical Projections

**Authors**: *Ines Chami, Albert Gu, Dat Nguyen, Christopher Ré*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chami21a.html](http://proceedings.mlr.press/v139/chami21a.html)

**Abstract**:

This paper studies Principal Component Analysis (PCA) for data lying in hyperbolic spaces. Given directions, PCA relies on: (1) a parameterization of subspaces spanned by these directions, (2) a method of projection onto subspaces that preserves information in these directions, and (3) an objective to optimize, namely the variance explained by projections. We generalize each of these concepts to the hyperbolic space and propose HoroPCA, a method for hyperbolic dimensionality reduction. By focusing on the core problem of extracting principal directions, HoroPCA theoretically better preserves information in the original data such as distances, compared to previous generalizations of PCA. Empirically, we validate that HoroPCA outperforms existing dimensionality reduction methods, significantly reducing error in distance preservation. As a data whitening method, it improves downstream classification by up to 3.9% compared to methods that don’t use whitening. Finally, we show that HoroPCA can be used to visualize hyperbolic data in two dimensions.

----

## [130] Goal-Conditioned Reinforcement Learning with Imagined Subgoals

**Authors**: *Elliot Chane-Sane, Cordelia Schmid, Ivan Laptev*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chane-sane21a.html](http://proceedings.mlr.press/v139/chane-sane21a.html)

**Abstract**:

Goal-conditioned reinforcement learning endows an agent with a large variety of skills, but it often struggles to solve tasks that require more temporally extended reasoning. In this work, we propose to incorporate imagined subgoals into policy learning to facilitate learning of complex tasks. Imagined subgoals are predicted by a separate high-level policy, which is trained simultaneously with the policy and its critic. This high-level policy predicts intermediate states halfway to the goal using the value function as a reachability metric. We don’t require the policy to reach these subgoals explicitly. Instead, we use them to define a prior policy, and incorporate this prior into a KL-constrained policy iteration scheme to speed up and regularize learning. Imagined subgoals are used during policy learning, but not during test time, where we only apply the learned policy. We evaluate our approach on complex robotic navigation and manipulation tasks and show that it outperforms existing methods by a large margin.

----

## [131] Locally Private k-Means in One Round

**Authors**: *Alisa Chang, Badih Ghazi, Ravi Kumar, Pasin Manurangsi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chang21a.html](http://proceedings.mlr.press/v139/chang21a.html)

**Abstract**:

We provide an approximation algorithm for k-means clustering in the \emph{one-round} (aka \emph{non-interactive}) local model of differential privacy (DP). Our algorithm achieves an approximation ratio arbitrarily close to the best \emph{non private} approximation algorithm, improving upon previously known algorithms that only guarantee large (constant) approximation ratios. Furthermore, ours is the first constant-factor approximation algorithm for k-means that requires only \emph{one} round of communication in the local DP model, positively resolving an open question of Stemmer (SODA 2020). Our algorithmic framework is quite flexible; we demonstrate this by showing that it also yields a similar near-optimal approximation algorithm in the (one-round) shuffle DP model.

----

## [132] Modularity in Reinforcement Learning via Algorithmic Independence in Credit Assignment

**Authors**: *Michael Chang, Sidhant Kaushik, Sergey Levine, Tom Griffiths*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chang21b.html](http://proceedings.mlr.press/v139/chang21b.html)

**Abstract**:

Many transfer problems require re-using previously optimal decisions for solving new tasks, which suggests the need for learning algorithms that can modify the mechanisms for choosing certain actions independently of those for choosing others. However, there is currently no formalism nor theory for how to achieve this kind of modular credit assignment. To answer this question, we define modular credit assignment as a constraint on minimizing the algorithmic mutual information among feedback signals for different decisions. We introduce what we call the modularity criterion for testing whether a learning algorithm satisfies this constraint by performing causal analysis on the algorithm itself. We generalize the recently proposed societal decision-making framework as a more granular formalism than the Markov decision process to prove that for decision sequences that do not contain cycles, certain single-step temporal difference action-value methods meet this criterion while all policy-gradient methods do not. Empirical evidence suggests that such action-value methods are more sample efficient than policy-gradient methods on transfer problems that require only sparse changes to a sequence of previously optimal decisions.

----

## [133] Image-Level or Object-Level? A Tale of Two Resampling Strategies for Long-Tailed Detection

**Authors**: *Nadine Chang, Zhiding Yu, Yu-Xiong Wang, Animashree Anandkumar, Sanja Fidler, José M. Álvarez*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chang21c.html](http://proceedings.mlr.press/v139/chang21c.html)

**Abstract**:

Training on datasets with long-tailed distributions has been challenging for major recognition tasks such as classification and detection. To deal with this challenge, image resampling is typically introduced as a simple but effective approach. However, we observe that long-tailed detection differs from classification since multiple classes may be present in one image. As a result, image resampling alone is not enough to yield a sufficiently balanced distribution at the object-level. We address object-level resampling by introducing an object-centric sampling strategy based on a dynamic, episodic memory bank. Our proposed strategy has two benefits: 1) convenient object-level resampling without significant extra computation, and 2) implicit feature-level augmentation from model updates. We show that image-level and object-level resamplings are both important, and thus unify them with a joint resampling strategy. Our method achieves state-of-the-art performance on the rare categories of LVIS, with 1.89% and 3.13% relative improvements over Forest R-CNN on detection and instance segmentation.

----

## [134] DeepWalking Backwards: From Embeddings Back to Graphs

**Authors**: *Sudhanshu Chanpuriya, Cameron Musco, Konstantinos Sotiropoulos, Charalampos E. Tsourakakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chanpuriya21a.html](http://proceedings.mlr.press/v139/chanpuriya21a.html)

**Abstract**:

Low-dimensional node embeddings play a key role in analyzing graph datasets. However, little work studies exactly what information is encoded by popular embedding methods, and how this information correlates with performance in downstream learning tasks. We tackle this question by studying whether embeddings can be inverted to (approximately) recover the graph used to generate them. Focusing on a variant of the popular DeepWalk method \cite{PerozziAl-RfouSkiena:2014, QiuDongMa:2018}, we present algorithms for accurate embedding inversion – i.e., from the low-dimensional embedding of a graph $G$, we can find a graph $\tilde G$ with a very similar embedding. We perform numerous experiments on real-world networks, observing that significant information about $G$, such as specific edges and bulk properties like triangle density, is often lost in $\tilde G$. However, community structure is often preserved or even enhanced. Our findings are a step towards a more rigorous understanding of exactly what information embeddings encode about the input graph, and why this information is useful for learning tasks.

----

## [135] Differentiable Spatial Planning using Transformers

**Authors**: *Devendra Singh Chaplot, Deepak Pathak, Jitendra Malik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chaplot21a.html](http://proceedings.mlr.press/v139/chaplot21a.html)

**Abstract**:

We consider the problem of spatial path planning. In contrast to the classical solutions which optimize a new plan from scratch and assume access to the full map with ground truth obstacle locations, we learn a planner from the data in a differentiable manner that allows us to leverage statistical regularities from past data. We propose Spatial Planning Transformers (SPT), which given an obstacle map learns to generate actions by planning over long-range spatial dependencies, unlike prior data-driven planners that propagate information locally via convolutional structure in an iterative manner. In the setting where the ground truth map is not known to the agent, we leverage pre-trained SPTs in an end-to-end framework that has the structure of mapper and planner built into it which allows seamless generalization to out-of-distribution maps and goals. SPTs outperform prior state-of-the-art differentiable planners across all the setups for both manipulation and navigation tasks, leading to an absolute improvement of 7-19%.

----

## [136] Solving Challenging Dexterous Manipulation Tasks With Trajectory Optimisation and Reinforcement Learning

**Authors**: *Henry Charlesworth, Giovanni Montana*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/charlesworth21a.html](http://proceedings.mlr.press/v139/charlesworth21a.html)

**Abstract**:

Training agents to autonomously control anthropomorphic robotic hands has the potential to lead to systems capable of performing a multitude of complex manipulation tasks in unstructured and uncertain environments. In this work, we first introduce a suite of challenging simulated manipulation tasks where current reinforcement learning and trajectory optimisation techniques perform poorly. These include environments where two simulated hands have to pass or throw objects between each other, as well as an environment where the agent must learn to spin a long pen between its fingers. We then introduce a simple trajectory optimisation algorithm that performs significantly better than existing methods on these environments. Finally, on the most challenging “PenSpin" task, we combine sub-optimal demonstrations generated through trajectory optimisation with off-policy reinforcement learning, obtaining performance that far exceeds either of these approaches individually. Videos of all of our results are available at: https://dexterous-manipulation.github.io

----

## [137] Classification with Rejection Based on Cost-sensitive Classification

**Authors**: *Nontawat Charoenphakdee, Zhenghang Cui, Yivan Zhang, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/charoenphakdee21a.html](http://proceedings.mlr.press/v139/charoenphakdee21a.html)

**Abstract**:

The goal of classification with rejection is to avoid risky misclassification in error-critical applications such as medical diagnosis and product inspection. In this paper, based on the relationship between classification with rejection and cost-sensitive classification, we propose a novel method of classification with rejection by learning an ensemble of cost-sensitive classifiers, which satisfies all the following properties: (i) it can avoid estimating class-posterior probabilities, resulting in improved classification accuracy. (ii) it allows a flexible choice of losses including non-convex ones, (iii) it does not require complicated modifications when using different losses, (iv) it is applicable to both binary and multiclass cases, and (v) it is theoretically justifiable for any classification-calibrated loss. Experimental results demonstrate the usefulness of our proposed approach in clean-labeled, noisy-labeled, and positive-unlabeled classification.

----

## [138] Actionable Models: Unsupervised Offline Reinforcement Learning of Robotic Skills

**Authors**: *Yevgen Chebotar, Karol Hausman, Yao Lu, Ted Xiao, Dmitry Kalashnikov, Jacob Varley, Alex Irpan, Benjamin Eysenbach, Ryan Julian, Chelsea Finn, Sergey Levine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chebotar21a.html](http://proceedings.mlr.press/v139/chebotar21a.html)

**Abstract**:

We consider the problem of learning useful robotic skills from previously collected offline data without access to manually specified rewards or additional online exploration, a setting that is becoming increasingly important for scaling robot learning by reusing past robotic data. In particular, we propose the objective of learning a functional understanding of the environment by learning to reach any goal state in a given dataset. We employ goal-conditioned Q-learning with hindsight relabeling and develop several techniques that enable training in a particularly challenging offline setting. We find that our method can operate on high-dimensional camera images and learn a variety of skills on real robots that generalize to previously unseen scenes and objects. We also show that our method can learn to reach long-horizon goals across multiple episodes through goal chaining, and learn rich representations that can help with downstream tasks through pre-training or auxiliary objectives.

----

## [139] Unified Robust Semi-Supervised Variational Autoencoder

**Authors**: *Xu Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21a.html](http://proceedings.mlr.press/v139/chen21a.html)

**Abstract**:

In this paper, we propose a novel noise-robust semi-supervised deep generative model by jointly tackling noisy labels and outliers simultaneously in a unified robust semi-supervised variational autoencoder (URSVAE). Typically, the uncertainty of of input data is characterized by placing uncertainty prior on the parameters of the probability density distributions in order to ensure the robustness of the variational encoder towards outliers. Subsequently, a noise transition model is integrated naturally into our model to alleviate the detrimental effects of noisy labels. Moreover, a robust divergence measure is employed to further enhance the robustness, where a novel variational lower bound is derived and optimized to infer the network parameters. By proving the influence function on the proposed evidence lower bound is bounded, the enormous potential of the proposed model in the classification in the presence of the compound noise is demonstrated. The experimental results highlight the superiority of the proposed framework by the evaluating on image classification tasks and comparing with the state-of-the-art approaches.

----

## [140] Unsupervised Learning of Visual 3D Keypoints for Control

**Authors**: *Boyuan Chen, Pieter Abbeel, Deepak Pathak*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21b.html](http://proceedings.mlr.press/v139/chen21b.html)

**Abstract**:

Learning sensorimotor control policies from high-dimensional images crucially relies on the quality of the underlying visual representations. Prior works show that structured latent space such as visual keypoints often outperforms unstructured representations for robotic control. However, most of these representations, whether structured or unstructured are learned in a 2D space even though the control tasks are usually performed in a 3D environment. In this work, we propose a framework to learn such a 3D geometric structure directly from images in an end-to-end unsupervised manner. The input images are embedded into latent 3D keypoints via a differentiable encoder which is trained to optimize both a multi-view consistency loss and downstream task objective. These discovered 3D keypoints tend to meaningfully capture robot joints as well as object movements in a consistent manner across both time and 3D space. The proposed approach outperforms prior state-of-art methods across a variety of reinforcement learning benchmarks. Code and videos at https://buoyancy99.github.io/unsup-3d-keypoints/.

----

## [141] Integer Programming for Causal Structure Learning in the Presence of Latent Variables

**Authors**: *Rui Chen, Sanjeeb Dash, Tian Gao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21c.html](http://proceedings.mlr.press/v139/chen21c.html)

**Abstract**:

The problem of finding an ancestral acyclic directed mixed graph (ADMG) that represents the causal relationships between a set of variables is an important area of research on causal inference. Most existing score-based structure learning methods focus on learning directed acyclic graph (DAG) models without latent variables. A number of score-based methods have recently been proposed for the ADMG learning, yet they are heuristic in nature and do not guarantee an optimal solution. We propose a novel exact score-based method that solves an integer programming (IP) formulation and returns a score-maximizing ancestral ADMG for a set of continuous variables that follow a multivariate Gaussian distribution. We generalize the state-of-the-art IP model for DAG learning problems and derive new classes of valid inequalities to formulate an IP model for ADMG learning. Empirically, our model can be solved efficiently for medium-sized problems and achieves better accuracy than state-of-the-art score-based methods as well as benchmark constraint-based methods.

----

## [142] Improved Corruption Robust Algorithms for Episodic Reinforcement Learning

**Authors**: *Yifang Chen, Simon S. Du, Kevin Jamieson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21d.html](http://proceedings.mlr.press/v139/chen21d.html)

**Abstract**:

We study episodic reinforcement learning under unknown adversarial corruptions in both the rewards and the transition probabilities of the underlying system. We propose new algorithms which, compared to the existing results in \cite{lykouris2020corruption}, achieve strictly better regret bounds in terms of total corruptions for the tabular setting. To be specific, firstly, our regret bounds depend on more precise numerical values of total rewards corruptions and transition corruptions, instead of only on the total number of corrupted episodes. Secondly, our regret bounds are the first of their kind in the reinforcement learning setting to have the number of corruptions show up additively with respect to $\min\{ \sqrt{T},\text{PolicyGapComplexity} \}$ rather than multiplicatively. Our results follow from a general algorithmic framework that combines corruption-robust policy elimination meta-algorithms, and plug-in reward-free exploration sub-algorithms. Replacing the meta-algorithm or sub-algorithm may extend the framework to address other corrupted settings with potentially more structure.

----

## [143] Scalable Computations of Wasserstein Barycenter via Input Convex Neural Networks

**Authors**: *Yongxin Chen, Jiaojiao Fan, Amirhossein Taghvaei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21e.html](http://proceedings.mlr.press/v139/chen21e.html)

**Abstract**:

Wasserstein Barycenter is a principled approach to represent the weighted mean of a given set of probability distributions, utilizing the geometry induced by optimal transport. In this work, we present a novel scalable algorithm to approximate the Wasserstein Barycenters aiming at high-dimensional applications in machine learning. Our proposed algorithm is based on the Kantorovich dual formulation of the Wasserstein-2 distance as well as a recent neural network architecture, input convex neural network, that is known to parametrize convex functions. The distinguishing features of our method are: i) it only requires samples from the marginal distributions; ii) unlike the existing approaches, it represents the Barycenter with a generative model and can thus generate infinite samples from the barycenter without querying the marginal distributions; iii) it works similar to Generative Adversarial Model in one marginal case. We demonstrate the efficacy of our algorithm by comparing it with the state-of-art methods in multiple experiments.

----

## [144] Neural Feature Matching in Implicit 3D Representations

**Authors**: *Yunlu Chen, Basura Fernando, Hakan Bilen, Thomas Mensink, Efstratios Gavves*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21f.html](http://proceedings.mlr.press/v139/chen21f.html)

**Abstract**:

Recently, neural implicit functions have achieved impressive results for encoding 3D shapes. Conditioning on low-dimensional latent codes generalises a single implicit function to learn shared representation space for a variety of shapes, with the advantage of smooth interpolation. While the benefits from the global latent space do not correspond to explicit points at local level, we propose to track the continuous point trajectory by matching implicit features with the latent code interpolating between shapes, from which we corroborate the hierarchical functionality of the deep implicit functions, where early layers map the latent code to fitting the coarse shape structure, and deeper layers further refine the shape details. Furthermore, the structured representation space of implicit functions enables to apply feature matching for shape deformation, with the benefits to handle topology and semantics inconsistency, such as from an armchair to a chair with no arms, without explicit flow functions or manual annotations.

----

## [145] Decentralized Riemannian Gradient Descent on the Stiefel Manifold

**Authors**: *Shixiang Chen, Alfredo Garcia, Mingyi Hong, Shahin Shahrampour*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21g.html](http://proceedings.mlr.press/v139/chen21g.html)

**Abstract**:

We consider a distributed non-convex optimization where a network of agents aims at minimizing a global function over the Stiefel manifold. The global function is represented as a finite sum of smooth local functions, where each local function is associated with one agent and agents communicate with each other over an undirected connected graph. The problem is non-convex as local functions are possibly non-convex (but smooth) and the Steifel manifold is a non-convex set. We present a decentralized Riemannian stochastic gradient method (DRSGD) with the convergence rate of $\mathcal{O}(1/\sqrt{K})$ to a stationary point. To have exact convergence with constant stepsize, we also propose a decentralized Riemannian gradient tracking algorithm (DRGTA) with the convergence rate of $\mathcal{O}(1/K)$ to a stationary point. We use multi-step consensus to preserve the iteration in the local (consensus) region. DRGTA is the first decentralized algorithm with exact convergence for distributed optimization on Stiefel manifold.

----

## [146] Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation

**Authors**: *Chao Chen, Haoyu Geng, Nianzu Yang, Junchi Yan, Daiyue Xue, Jianping Yu, Xiaokang Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21h.html](http://proceedings.mlr.press/v139/chen21h.html)

**Abstract**:

User interests are usually dynamic in the real world, which poses both theoretical and practical challenges for learning accurate preferences from rich behavior data. Among existing user behavior modeling solutions, attention networks are widely adopted for its effectiveness and relative simplicity. Despite being extensively studied, existing attentions still suffer from two limitations: i) conventional attentions mainly take into account the spatial correlation between user behaviors, regardless the distance between those behaviors in the continuous time space; and ii) these attentions mostly provide a dense and undistinguished distribution over all past behaviors then attentively encode them into the output latent representations. This is however not suitable in practical scenarios where a user’s future actions are relevant to a small subset of her/his historical behaviors. In this paper, we propose a novel attention network, named \textit{self-modulating attention}, that models the complex and non-linearly evolving dynamic user preferences. We empirically demonstrate the effectiveness of our method on top-N sequential recommendation tasks, and the results on three large-scale real-world datasets show that our model can achieve state-of-the-art performance.

----

## [147] Mandoline: Model Evaluation under Distribution Shift

**Authors**: *Mayee F. Chen, Karan Goel, Nimit Sharad Sohoni, Fait Poms, Kayvon Fatahalian, Christopher Ré*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21i.html](http://proceedings.mlr.press/v139/chen21i.html)

**Abstract**:

Machine learning models are often deployed in different settings than they were trained and validated on, posing a challenge to practitioners who wish to predict how well the deployed model will perform on a target distribution. If an unlabeled sample from the target distribution is available, along with a labeled sample from a possibly different source distribution, standard approaches such as importance weighting can be applied to estimate performance on the target. However, importance weighting struggles when the source and target distributions have non-overlapping support or are high-dimensional. Taking inspiration from fields such as epidemiology and polling, we develop Mandoline, a new evaluation framework that mitigates these issues. Our key insight is that practitioners may have prior knowledge about the ways in which the distribution shifts, which we can use to better guide the importance weighting procedure. Specifically, users write simple "slicing functions" {–} noisy, potentially correlated binary functions intended to capture possible axes of distribution shift {–} to compute reweighted performance estimates. We further describe a density ratio estimation framework for the slices and show how its estimation error scales with slice quality and dataset size. Empirical validation on NLP and vision tasks shows that Mandoline can estimate performance on the target distribution up to 3x more accurately compared to standard baselines.

----

## [148] Order Matters: Probabilistic Modeling of Node Sequence for Graph Generation

**Authors**: *Xiaohui Chen, Xu Han, Jiajing Hu, Francisco J. R. Ruiz, Li-Ping Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21j.html](http://proceedings.mlr.press/v139/chen21j.html)

**Abstract**:

A graph generative model defines a distribution over graphs. Typically, the model consists of a sequential process that creates and adds nodes and edges. Such sequential process defines an ordering of the nodes in the graph. The computation of the model’s likelihood requires to marginalize the node orderings; this makes maximum likelihood estimation (MLE) challenging due to the (factorial) number of possible permutations. In this work, we provide an expression for the likelihood of a graph generative model and show that its calculation is closely related to the problem of graph automorphism. In addition, we derive a variational inference (VI) algorithm for fitting a graph generative model that is based on the maximization of a variational bound of the log-likelihood. This allows the model to be trained with node orderings from the approximate posterior instead of ad-hoc orderings. Our experiments show that our log-likelihood bound is significantly tighter than the bound of previous schemes. The models fitted with the VI algorithm are able to generate high-quality graphs that match the structures of target graphs not seen during training.

----

## [149] CARTL: Cooperative Adversarially-Robust Transfer Learning

**Authors**: *Dian Chen, Hongxin Hu, Qian Wang, Yinli Li, Cong Wang, Chao Shen, Qi Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21k.html](http://proceedings.mlr.press/v139/chen21k.html)

**Abstract**:

Transfer learning eases the burden of training a well-performed model from scratch, especially when training data is scarce and computation power is limited. In deep learning, a typical strategy for transfer learning is to freeze the early layers of a pre-trained model and fine-tune the rest of its layers on the target domain. Previous work focuses on the accuracy of the transferred model but neglects the transfer of adversarial robustness. In this work, we first show that transfer learning improves the accuracy on the target domain but degrades the inherited robustness of the target model. To address such a problem, we propose a novel cooperative adversarially-robust transfer learning (CARTL) by pre-training the model via feature distance minimization and fine-tuning the pre-trained model with non-expansive fine-tuning for target domain tasks. Empirical results show that CARTL improves the inherited robustness by about 28% at most compared with the baseline with the same degree of accuracy. Furthermore, we study the relationship between the batch normalization (BN) layers and the robustness in the context of transfer learning, and we reveal that freezing BN layers can further boost the robustness transfer.

----

## [150] Finding the Stochastic Shortest Path with Low Regret: the Adversarial Cost and Unknown Transition Case

**Authors**: *Liyu Chen, Haipeng Luo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21l.html](http://proceedings.mlr.press/v139/chen21l.html)

**Abstract**:

We make significant progress toward the stochastic shortest path problem with adversarial costs and unknown transition. Specifically, we develop algorithms that achieve $O(\sqrt{S^2ADT_\star K})$ regret for the full-information setting and $O(\sqrt{S^3A^2DT_\star K})$ regret for the bandit feedback setting, where $D$ is the diameter, $T_\star$ is the expected hitting time of the optimal policy, $S$ is the number of states, $A$ is the number of actions, and $K$ is the number of episodes. Our work strictly improves (Rosenberg and Mansour, 2020) in the full information setting, extends (Chen et al., 2020) from known transition to unknown transition, and is also the first to consider the most challenging combination: bandit feedback with adversarial costs and unknown transition. To remedy the gap between our upper bounds and the current best lower bounds constructed via a stochastically oblivious adversary, we also propose algorithms with near-optimal regret for this special case.

----

## [151] SpreadsheetCoder: Formula Prediction from Semi-structured Context

**Authors**: *Xinyun Chen, Petros Maniatis, Rishabh Singh, Charles Sutton, Hanjun Dai, Max Lin, Denny Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21m.html](http://proceedings.mlr.press/v139/chen21m.html)

**Abstract**:

Spreadsheet formula prediction has been an important program synthesis problem with many real-world applications. Previous works typically utilize input-output examples as the specification for spreadsheet formula synthesis, where each input-output pair simulates a separate row in the spreadsheet. However, this formulation does not fully capture the rich context in real-world spreadsheets. First, spreadsheet data entries are organized as tables, thus rows and columns are not necessarily independent from each other. In addition, many spreadsheet tables include headers, which provide high-level descriptions of the cell data. However, previous synthesis approaches do not consider headers as part of the specification. In this work, we present the first approach for synthesizing spreadsheet formulas from tabular context, which includes both headers and semi-structured tabular data. In particular, we propose SpreadsheetCoder, a BERT-based model architecture to represent the tabular context in both row-based and column-based formats. We train our model on a large dataset of spreadsheets, and demonstrate that SpreadsheetCoder achieves top-1 prediction accuracy of 42.51%, which is a considerable improvement over baselines that do not employ rich tabular context. Compared to the rule-based system, SpreadsheetCoder assists 82% more users in composing formulas on Google Sheets.

----

## [152] Large-Margin Contrastive Learning with Distance Polarization Regularizer

**Authors**: *Shuo Chen, Gang Niu, Chen Gong, Jun Li, Jian Yang, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21n.html](http://proceedings.mlr.press/v139/chen21n.html)

**Abstract**:

\emph{Contrastive learning} (CL) pretrains models in a pairwise manner, where given a data point, other data points are all regarded as dissimilar, including some that are \emph{semantically} similar. The issue has been addressed by properly weighting similar and dissimilar pairs as in \emph{positive-unlabeled learning}, so that the objective of CL is \emph{unbiased} and CL is \emph{consistent}. However, in this paper, we argue that this great solution is still not enough: its weighted objective \emph{hides} the issue where the semantically similar pairs are still pushed away; as CL is pretraining, this phenomenon is not our desideratum and might affect downstream tasks. To this end, we propose \emph{large-margin contrastive learning} (LMCL) with \emph{distance polarization regularizer}, motivated by the distribution characteristic of pairwise distances in \emph{metric learning}. In LMCL, we can distinguish between \emph{intra-cluster} and \emph{inter-cluster} pairs, and then only push away inter-cluster pairs, which \emph{solves} the above issue explicitly. Theoretically, we prove a tighter error bound for LMCL; empirically, the superiority of LMCL is demonstrated across multiple domains, \emph{i.e.}, image classification, sentence representation, and reinforcement learning.

----

## [153] Z-GCNETs: Time Zigzags at Graph Convolutional Networks for Time Series Forecasting

**Authors**: *Yuzhou Chen, Ignacio Segovia-Dominguez, Yulia R. Gel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21o.html](http://proceedings.mlr.press/v139/chen21o.html)

**Abstract**:

There recently has been a surge of interest in developing a new class of deep learning (DL) architectures that integrate an explicit time dimension as a fundamental building block of learning and representation mechanisms. In turn, many recent results show that topological descriptors of the observed data, encoding information on the shape of the dataset in a topological space at different scales, that is, persistent homology of the data, may contain important complementary information, improving both performance and robustness of DL. As convergence of these two emerging ideas, we propose to enhance DL architectures with the most salient time-conditioned topological information of the data and introduce the concept of zigzag persistence into time-aware graph convolutional networks (GCNs). Zigzag persistence provides a systematic and mathematically rigorous framework to track the most important topological features of the observed data that tend to manifest themselves over time. To integrate the extracted time-conditioned topological descriptors into DL, we develop a new topological summary, zigzag persistence image, and derive its theoretical stability guarantees. We validate the new GCNs with a time-aware zigzag topological layer (Z-GCNETs), in application to traffic forecasting and Ethereum blockchain price prediction. Our results indicate that Z-GCNET outperforms 13 state-of-the-art methods on 4 time series datasets.

----

## [154] A Unified Lottery Ticket Hypothesis for Graph Neural Networks

**Authors**: *Tianlong Chen, Yongduo Sui, Xuxi Chen, Aston Zhang, Zhangyang Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21p.html](http://proceedings.mlr.press/v139/chen21p.html)

**Abstract**:

With graphs rapidly growing in size and deeper graph neural networks (GNNs) emerging, the training and inference of GNNs become increasingly expensive. Existing network weight pruning algorithms cannot address the main space and computational bottleneck in GNNs, caused by the size and connectivity of the graph. To this end, this paper first presents a unified GNN sparsification (UGS) framework that simultaneously prunes the graph adjacency matrix and the model weights, for effectively accelerating GNN inference on large-scale graphs. Leveraging this new tool, we further generalize the recently popular lottery ticket hypothesis to GNNs for the first time, by defining a graph lottery ticket (GLT) as a pair of core sub-dataset and sparse sub-network, which can be jointly identified from the original GNN and the full dense graph by iteratively applying UGS. Like its counterpart in convolutional neural networks, GLT can be trained in isolation to match the performance of training with the full model and graph, and can be drawn from both randomly initialized and self-supervised pre-trained GNNs. Our proposal has been experimentally verified across various GNN architectures and diverse tasks, on both small-scale graph datasets (Cora, Citeseer and PubMed), and large-scale datasets from the challenging Open Graph Benchmark (OGB). Specifically, for node classification, our found GLTs achieve the same accuracies with 20% 98% MACs saving on small graphs and 25% 85% MACs saving on large ones. For link prediction, GLTs lead to 48% 97% and 70% MACs saving on small and large graph datasets, respectively, without compromising predictive performance. Codes are at https://github.com/VITA-Group/Unified-LTH-GNN.

----

## [155] Network Inference and Influence Maximization from Samples

**Authors**: *Wei Chen, Xiaoming Sun, Jialin Zhang, Zhijie Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21q.html](http://proceedings.mlr.press/v139/chen21q.html)

**Abstract**:

Influence maximization is the task of selecting a small number of seed nodes in a social network to maximize the spread of the influence from these seeds, and it has been widely investigated in the past two decades. In the canonical setting, the whole social network as well as its diffusion parameters is given as input. In this paper, we consider the more realistic sampling setting where the network is unknown and we only have a set of passively observed cascades that record the set of activated nodes at each diffusion step. We study the task of influence maximization from these cascade samples (IMS), and present constant approximation algorithms for this task under mild conditions on the seed set distribution. To achieve the optimization goal, we also provide a novel solution to the network inference problem, that is, learning diffusion parameters and the network structure from the cascade data. Comparing with prior solutions, our network inference algorithm requires weaker assumptions and does not rely on maximum-likelihood estimation and convex programming. Our IMS algorithms enhance the learning-and-then-optimization approach by allowing a constant approximation ratio even when the diffusion parameters are hard to learn, and we do not need any assumption related to the network structure or diffusion parameters.

----

## [156] Data-driven Prediction of General Hamiltonian Dynamics via Learning Exactly-Symplectic Maps

**Authors**: *Renyi Chen, Molei Tao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21r.html](http://proceedings.mlr.press/v139/chen21r.html)

**Abstract**:

We consider the learning and prediction of nonlinear time series generated by a latent symplectic map. A special case is (not necessarily separable) Hamiltonian systems, whose solution flows give such symplectic maps. For this special case, both generic approaches based on learning the vector field of the latent ODE and specialized approaches based on learning the Hamiltonian that generates the vector field exist. Our method, however, is different as it does not rely on the vector field nor assume its existence; instead, it directly learns the symplectic evolution map in discrete time. Moreover, we do so by representing the symplectic map via a generating function, which we approximate by a neural network (hence the name GFNN). This way, our approximation of the evolution map is always \emph{exactly} symplectic. This additional geometric structure allows the local prediction error at each step to accumulate in a controlled fashion, and we will prove, under reasonable assumptions, that the global prediction error grows at most \emph{linearly} with long prediction time, which significantly improves an otherwise exponential growth. In addition, as a map-based and thus purely data-driven method, GFNN avoids two additional sources of inaccuracies common in vector-field based approaches, namely the error in approximating the vector field by finite difference of the data, and the error in numerical integration of the vector field for making predictions. Numerical experiments further demonstrate our claims.

----

## [157] Analysis of stochastic Lanczos quadrature for spectrum approximation

**Authors**: *Tyler Chen, Thomas Trogdon, Shashanka Ubaru*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21s.html](http://proceedings.mlr.press/v139/chen21s.html)

**Abstract**:

The cumulative empirical spectral measure (CESM) $\Phi[\mathbf{A}] : \mathbb{R} \to [0,1]$ of a $n\times n$ symmetric matrix $\mathbf{A}$ is defined as the fraction of eigenvalues of $\mathbf{A}$ less than a given threshold, i.e., $\Phi[\mathbf{A}](x) := \sum_{i=1}^{n} \frac{1}{n} {\large\unicode{x1D7D9}}[ \lambda_i[\mathbf{A}]\leq x]$. Spectral sums $\operatorname{tr}(f[\mathbf{A}])$ can be computed as the Riemann–Stieltjes integral of $f$ against $\Phi[\mathbf{A}]$, so the task of estimating CESM arises frequently in a number of applications, including machine learning. We present an error analysis for stochastic Lanczos quadrature (SLQ). We show that SLQ obtains an approximation to the CESM within a Wasserstein distance of $t \: | \lambda_{\text{max}}[\mathbf{A}] - \lambda_{\text{min}}[\mathbf{A}] |$ with probability at least $1-\eta$, by applying the Lanczos algorithm for $\lceil 12 t^{-1} + \frac{1}{2} \rceil$ iterations to $\lceil 4 ( n+2 )^{-1}t^{-2} \ln(2n\eta^{-1}) \rceil$ vectors sampled independently and uniformly from the unit sphere. We additionally provide (matrix-dependent) a posteriori error bounds for the Wasserstein and Kolmogorov–Smirnov distances between the output of this algorithm and the true CESM. The quality of our bounds is demonstrated using numerical experiments.

----

## [158] Large-Scale Multi-Agent Deep FBSDEs

**Authors**: *Tianrong Chen, Ziyi Wang, Ioannis Exarchos, Evangelos A. Theodorou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21t.html](http://proceedings.mlr.press/v139/chen21t.html)

**Abstract**:

In this paper we present a scalable deep learning framework for finding Markovian Nash Equilibria in multi-agent stochastic games using fictitious play. The motivation is inspired by theoretical analysis of Forward Backward Stochastic Differential Equations and their implementation in a deep learning setting, which is the source of our algorithm’s sample efficiency improvement. By taking advantage of the permutation-invariant property of agents in symmetric games, the scalability and performance is further enhanced significantly. We showcase superior performance of our framework over the state-of-the-art deep fictitious play algorithm on an inter-bank lending/borrowing problem in terms of multiple metrics. More importantly, our approach scales up to 3000 agents in simulation, a scale which, to the best of our knowledge, represents a new state-of-the-art. We also demonstrate the applicability of our framework in robotics on a belief space autonomous racing problem.

----

## [159] Representation Subspace Distance for Domain Adaptation Regression

**Authors**: *Xinyang Chen, Sinan Wang, Jianmin Wang, Mingsheng Long*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21u.html](http://proceedings.mlr.press/v139/chen21u.html)

**Abstract**:

Regression, as a counterpart to classification, is a major paradigm with a wide range of applications. Domain adaptation regression extends it by generalizing a regressor from a labeled source domain to an unlabeled target domain. Existing domain adaptation regression methods have achieved positive results limited only to the shallow regime. A question arises: Why learning invariant representations in the deep regime less pronounced? A key finding of this paper is that classification is robust to feature scaling but regression is not, and aligning the distributions of deep representations will alter feature scale and impede domain adaptation regression. Based on this finding, we propose to close the domain gap through orthogonal bases of the representation spaces, which are free from feature scaling. Inspired by Riemannian geometry of Grassmann manifold, we define a geometrical distance over representation subspaces and learn deep transferable representations by minimizing it. To avoid breaking the geometrical properties of deep representations, we further introduce the bases mismatch penalization to match the ordering of orthogonal bases across representation subspaces. Our method is evaluated on three domain adaptation regression benchmarks, two of which are introduced in this paper. Our method outperforms the state-of-the-art methods significantly, forming early positive results in the deep regime.

----

## [160] Overcoming Catastrophic Forgetting by Bayesian Generative Regularization

**Authors**: *Pei-Hung Chen, Wei Wei, Cho-Jui Hsieh, Bo Dai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21v.html](http://proceedings.mlr.press/v139/chen21v.html)

**Abstract**:

In this paper, we propose a new method to over-come catastrophic forgetting by adding generative regularization to Bayesian inference frame-work. Bayesian method provides a general frame-work for continual learning. We could further construct a generative regularization term for all given classification models by leveraging energy-based models and Langevin dynamic sampling to enrich the features learned in each task. By combining discriminative and generative loss together, we empirically show that the proposed method outperforms state-of-the-art methods on a variety of tasks, avoiding catastrophic forgetting in continual learning. In particular, the proposed method outperforms baseline methods over 15%on the Fashion-MNIST dataset and 10%on the CUB dataset.

----

## [161] Cyclically Equivariant Neural Decoders for Cyclic Codes

**Authors**: *Xiangyu Chen, Min Ye*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21w.html](http://proceedings.mlr.press/v139/chen21w.html)

**Abstract**:

Neural decoders were introduced as a generalization of the classic Belief Propagation (BP) decoding algorithms, where the Trellis graph in the BP algorithm is viewed as a neural network, and the weights in the Trellis graph are optimized by training the neural network. In this work, we propose a novel neural decoder for cyclic codes by exploiting their cyclically invariant property. More precisely, we impose a shift invariant structure on the weights of our neural decoder so that any cyclic shift of inputs results in the same cyclic shift of outputs. Extensive simulations with BCH codes and punctured Reed-Muller (RM) codes show that our new decoder consistently outperforms previous neural decoders when decoding cyclic codes. Finally, we propose a list decoding procedure that can significantly reduce the decoding error probability for BCH codes and punctured RM codes. For certain high-rate codes, the gap between our list decoder and the Maximum Likelihood decoder is less than $0.1$dB. Code available at github.com/cyclicallyneuraldecoder

----

## [162] A Receptor Skeleton for Capsule Neural Networks

**Authors**: *Jintai Chen, Hongyun Yu, Chengde Qian, Danny Z. Chen, Jian Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21x.html](http://proceedings.mlr.press/v139/chen21x.html)

**Abstract**:

In previous Capsule Neural Networks (CapsNets), routing algorithms often performed clustering processes to assemble the child capsules’ representations into parent capsules. Such routing algorithms were typically implemented with iterative processes and incurred high computing complexity. This paper presents a new capsule structure, which contains a set of optimizable receptors and a transmitter is devised on the capsule’s representation. Specifically, child capsules’ representations are sent to the parent capsules whose receptors match well the transmitters of the child capsules’ representations, avoiding applying computationally complex routing algorithms. To ensure the receptors in a CapsNet work cooperatively, we build a skeleton to organize the receptors in different capsule layers in a CapsNet. The receptor skeleton assigns a share-out objective for each receptor, making the CapsNet perform as a hierarchical agglomerative clustering process. Comprehensive experiments verify that our approach facilitates efficient clustering processes, and CapsNets with our approach significantly outperform CapsNets with previous routing algorithms on image classification, affine transformation generalization, overlapped object recognition, and representation semantic decoupling.

----

## [163] Accelerating Gossip SGD with Periodic Global Averaging

**Authors**: *Yiming Chen, Kun Yuan, Yingya Zhang, Pan Pan, Yinghui Xu, Wotao Yin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21y.html](http://proceedings.mlr.press/v139/chen21y.html)

**Abstract**:

Communication overhead hinders the scalability of large-scale distributed training. Gossip SGD, where each node averages only with its neighbors, is more communication-efficient than the prevalent parallel SGD. However, its convergence rate is reversely proportional to quantity $1-\beta$ which measures the network connectivity. On large and sparse networks where $1-\beta \to 0$, Gossip SGD requires more iterations to converge, which offsets against its communication benefit. This paper introduces Gossip-PGA, which adds Periodic Global Averaging to accelerate Gossip SGD. Its transient stage, i.e., the iterations required to reach asymptotic linear speedup stage, improves from $\Omega(\beta^4 n^3/(1-\beta)^4)$ to $\Omega(\beta^4 n^3 H^4)$ for non-convex problems. The influence of network topology in Gossip-PGA can be controlled by the averaging period $H$. Its transient-stage complexity is also superior to local SGD which has order $\Omega(n^3 H^4)$. Empirical results of large-scale training on image classification (ResNet50) and language modeling (BERT) validate our theoretical findings.

----

## [164] ActNN: Reducing Training Memory Footprint via 2-Bit Activation Compressed Training

**Authors**: *Jianfei Chen, Lianmin Zheng, Zhewei Yao, Dequan Wang, Ion Stoica, Michael W. Mahoney, Joseph Gonzalez*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chen21z.html](http://proceedings.mlr.press/v139/chen21z.html)

**Abstract**:

The increasing size of neural network models has been critical for improvements in their accuracy, but device memory is not growing at the same rate. This creates fundamental challenges for training neural networks within limited memory environments. In this work, we propose ActNN, a memory-efficient training framework that stores randomly quantized activations for back propagation. We prove the convergence of ActNN for general network architectures, and we characterize the impact of quantization on the convergence via an exact expression for the gradient variance. Using our theory, we propose novel mixed-precision quantization strategies that exploit the activation’s heterogeneity across feature dimensions, samples, and layers. These techniques can be readily applied to existing dynamic graph frameworks, such as PyTorch, simply by substituting the layers. We evaluate ActNN on mainstream computer vision models for classification, detection, and segmentation tasks. On all these tasks, ActNN compresses the activation to 2 bits on average, with negligible accuracy loss. ActNN reduces the memory footprint of the activation by 12x, and it enables training with a 6.6x to 14x larger batch size.

----

## [165] SPADE: A Spectral Method for Black-Box Adversarial Robustness Evaluation

**Authors**: *Wuxinlin Cheng, Chenhui Deng, Zhiqiang Zhao, Yaohui Cai, Zhiru Zhang, Zhuo Feng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cheng21a.html](http://proceedings.mlr.press/v139/cheng21a.html)

**Abstract**:

A black-box spectral method is introduced for evaluating the adversarial robustness of a given machine learning (ML) model. Our approach, named SPADE, exploits bijective distance mapping between the input/output graphs constructed for approximating the manifolds corresponding to the input/output data. By leveraging the generalized Courant-Fischer theorem, we propose a SPADE score for evaluating the adversarial robustness of a given model, which is proved to be an upper bound of the best Lipschitz constant under the manifold setting. To reveal the most non-robust data samples highly vulnerable to adversarial attacks, we develop a spectral graph embedding procedure leveraging dominant generalized eigenvectors. This embedding step allows assigning each data point a robustness score that can be further harnessed for more effective adversarial training of ML models. Our experiments show promising empirical results for neural networks trained with the MNIST and CIFAR-10 data sets.

----

## [166] Self-supervised and Supervised Joint Training for Resource-rich Machine Translation

**Authors**: *Yong Cheng, Wei Wang, Lu Jiang, Wolfgang Macherey*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cheng21b.html](http://proceedings.mlr.press/v139/cheng21b.html)

**Abstract**:

Self-supervised pre-training of text representations has been successfully applied to low-resource Neural Machine Translation (NMT). However, it usually fails to achieve notable gains on resource-rich NMT. In this paper, we propose a joint training approach, F2-XEnDec, to combine self-supervised and supervised learning to optimize NMT models. To exploit complementary self-supervised signals for supervised learning, NMT models are trained on examples that are interbred from monolingual and parallel sentences through a new process called crossover encoder-decoder. Experiments on two resource-rich translation benchmarks, WMT’14 English-German and WMT’14 English-French, demonstrate that our approach achieves substantial improvements over several strong baseline methods and obtains a new state of the art of 46.19 BLEU on English-French when incorporating back translation. Results also show that our approach is capable of improving model robustness to input perturbations such as code-switching noise which frequently appears on the social media.

----

## [167] Exact Optimization of Conformal Predictors via Incremental and Decremental Learning

**Authors**: *Giovanni Cherubin, Konstantinos Chatzikokolakis, Martin Jaggi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cherubin21a.html](http://proceedings.mlr.press/v139/cherubin21a.html)

**Abstract**:

Conformal Predictors (CP) are wrappers around ML models, providing error guarantees under weak assumptions on the data distribution. They are suitable for a wide range of problems, from classification and regression to anomaly detection. Unfortunately, their very high computational complexity limits their applicability to large datasets. In this work, we show that it is possible to speed up a CP classifier considerably, by studying it in conjunction with the underlying ML method, and by exploiting incremental&decremental learning. For methods such as k-NN, KDE, and kernel LS-SVM, our approach reduces the running time by one order of magnitude, whilst producing exact solutions. With similar ideas, we also achieve a linear speed up for the harder case of bootstrapping. Finally, we extend these techniques to improve upon an optimization of k-NN CP for regression. We evaluate our findings empirically, and discuss when methods are suitable for CP optimization.

----

## [168] Problem Dependent View on Structured Thresholding Bandit Problems

**Authors**: *James Cheshire, Pierre Ménard, Alexandra Carpentier*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cheshire21a.html](http://proceedings.mlr.press/v139/cheshire21a.html)

**Abstract**:

We investigate the \textit{problem dependent regime} in the stochastic \emph{Thresholding Bandit problem} (\tbp) under several \emph{shape constraints}. In the \tbp the objective of the learner is to output, after interacting with the environment, the set of arms whose means are above a given threshold. The vanilla, unstructured, case is already well studied in the literature. Taking $K$ as the number of arms, we consider the case where (i) the sequence of arm’s means $(\mu_k){k=1}^K$ is monotonically increasing (\textit{MTBP}) and (ii) the case where $(\mu_k){k=1}^K$ is concave (\textit{CTBP}). We consider both cases in the \emph{problem dependent} regime and study the probability of error - i.e. the probability to mis-classify at least one arm. In the fixed budget setting, we provide nearly matching upper and lower bounds for the probability of error in both the concave and monotone settings, as well as associated algorithms. Of interest, is that for both the monotone and concave cases, optimal bounds on probability of error are of the same order as those for the two armed bandit problem.

----

## [169] Online Optimization in Games via Control Theory: Connecting Regret, Passivity and Poincaré Recurrence

**Authors**: *Yun Kuen Cheung, Georgios Piliouras*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cheung21a.html](http://proceedings.mlr.press/v139/cheung21a.html)

**Abstract**:

We present a novel control-theoretic understanding of online optimization and learning in games, via the notion of passivity. Passivity is a fundamental concept in control theory, which abstracts energy conservation and dissipation in physical systems. It has become a standard tool in analysis of general feedback systems, to which game dynamics belong. Our starting point is to show that all continuous-time Follow-the-Regularized-Leader (FTRL) dynamics, which include the well-known Replicator Dynamic, are lossless, i.e. it is passive with no energy dissipation. Interestingly, we prove that passivity implies bounded regret, connecting two fundamental primitives of control theory and online optimization. The observation of energy conservation in FTRL inspires us to present a family of lossless learning dynamics, each of which has an underlying energy function with a simple gradient structure. This family is closed under convex combination; as an immediate corollary, any convex combination of FTRL dynamics is lossless and thus has bounded regret. This allows us to extend the framework of Fox & Shamma [Games 2013] to prove not just global asymptotic stability results for game dynamics, but Poincar{é} recurrence results as well. Intuitively, when a lossless game (e.g. graphical constant-sum game) is coupled with lossless learning dynamic, their interconnection is also lossless, which results in a pendulum-like energy-preserving recurrent behavior, generalizing Piliouras & Shamma [SODA 2014] and Mertikopoulos et al. [SODA 2018].

----

## [170] Understanding and Mitigating Accuracy Disparity in Regression

**Authors**: *Jianfeng Chi, Yuan Tian, Geoffrey J. Gordon, Han Zhao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chi21a.html](http://proceedings.mlr.press/v139/chi21a.html)

**Abstract**:

With the widespread deployment of large-scale prediction systems in high-stakes domains, e.g., face recognition, criminal justice, etc., disparity on prediction accuracy between different demographic subgroups has called for fundamental understanding on the source of such disparity and algorithmic intervention to mitigate it. In this paper, we study the accuracy disparity problem in regression. To begin with, we first propose an error decomposition theorem, which decomposes the accuracy disparity into the distance between marginal label distributions and the distance between conditional representations, to help explain why such accuracy disparity appears in practice. Motivated by this error decomposition and the general idea of distribution alignment with statistical distances, we then propose an algorithm to reduce this disparity, and analyze its game-theoretic optima of the proposed objective functions. To corroborate our theoretical findings, we also conduct experiments on five benchmark datasets. The experimental results suggest that our proposed algorithms can effectively mitigate accuracy disparity while maintaining the predictive power of the regression models.

----

## [171] Private Alternating Least Squares: Practical Private Matrix Completion with Tighter Rates

**Authors**: *Steve Chien, Prateek Jain, Walid Krichene, Steffen Rendle, Shuang Song, Abhradeep Thakurta, Li Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chien21a.html](http://proceedings.mlr.press/v139/chien21a.html)

**Abstract**:

We study the problem of differentially private (DP) matrix completion under user-level privacy. We design a joint differentially private variant of the popular Alternating-Least-Squares (ALS) method that achieves: i) (nearly) optimal sample complexity for matrix completion (in terms of number of items, users), and ii) the best known privacy/utility trade-off both theoretically, as well as on benchmark data sets. In particular, we provide the first global convergence analysis of ALS with noise introduced to ensure DP, and show that, in comparison to the best known alternative (the Private Frank-Wolfe algorithm by Jain et al. (2018)), our error bounds scale significantly better with respect to the number of items and users, which is critical in practical problems. Extensive validation on standard benchmarks demonstrate that the algorithm, in combination with carefully designed sampling procedures, is significantly more accurate than existing techniques, thus promising to be the first practical DP embedding model.

----

## [172] Light RUMs

**Authors**: *Flavio Chierichetti, Ravi Kumar, Andrew Tomkins*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chierichetti21a.html](http://proceedings.mlr.press/v139/chierichetti21a.html)

**Abstract**:

A Random Utility Model (RUM) is a distribution on permutations over a universe of items. For each subset of the universe, a RUM induces a natural distribution of the winner in the subset: choose a permutation according to the RUM distribution and pick the maximum item in the subset according to the chosen permutation. RUMs are widely used in the theory of discrete choice. In this paper we consider the question of the (lossy) compressibility of RUMs on a universe of size $n$, i.e., the minimum number of bits required to approximate the winning probabilities of each slate. Our main result is that RUMs can be approximated using $\tilde{O}(n^2)$ bits, an exponential improvement over the standard representation; furthermore, we show that this bound is optimal. En route, we sharpen the classical existential result of McFadden and Train (2000) by showing that the minimum size of a mixture of multinomial logits required to can approximate a general RUM is $\tilde{\Theta}(n)$.

----

## [173] Parallelizing Legendre Memory Unit Training

**Authors**: *Narsimha Reddy Chilkuri, Chris Eliasmith*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chilkuri21a.html](http://proceedings.mlr.press/v139/chilkuri21a.html)

**Abstract**:

Recently, a new recurrent neural network (RNN) named the Legendre Memory Unit (LMU) was proposed and shown to achieve state-of-the-art performance on several benchmark datasets. Here we leverage the linear time-invariant (LTI) memory component of the LMU to construct a simplified variant that can be parallelized during training (and yet executed as an RNN during inference), resulting in up to 200 times faster training. We note that our efficient parallelizing scheme is general and is applicable to any deep network whose recurrent components are linear dynamical systems. We demonstrate the improved accuracy of our new architecture compared to the original LMU and a variety of published LSTM and transformer networks across seven benchmarks. For instance, our LMU sets a new state-of-the-art result on psMNIST, and uses half the parameters while outperforming DistilBERT and LSTM models on IMDB sentiment analysis.

----

## [174] Quantifying and Reducing Bias in Maximum Likelihood Estimation of Structured Anomalies

**Authors**: *Uthsav Chitra, Kimberly Ding, Jasper C. H. Lee, Benjamin J. Raphael*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chitra21a.html](http://proceedings.mlr.press/v139/chitra21a.html)

**Abstract**:

Anomaly estimation, or the problem of finding a subset of a dataset that differs from the rest of the dataset, is a classic problem in machine learning and data mining. In both theoretical work and in applications, the anomaly is assumed to have a specific structure defined by membership in an anomaly family. For example, in temporal data the anomaly family may be time intervals, while in network data the anomaly family may be connected subgraphs. The most prominent approach for anomaly estimation is to compute the Maximum Likelihood Estimator (MLE) of the anomaly; however, it was recently observed that for normally distributed data, the MLE is a biased estimator for some anomaly families. In this work, we demonstrate that in the normal means setting, the bias of the MLE depends on the size of the anomaly family. We prove that if the number of sets in the anomaly family that contain the anomaly is sub-exponential, then the MLE is asymptotically unbiased. We also provide empirical evidence that the converse is true: if the number of such sets is exponential, then the MLE is asymptotically biased. Our analysis unifies a number of earlier results on the bias of the MLE for specific anomaly families. Next, we derive a new anomaly estimator using a mixture model, and we prove that our anomaly estimator is asymptotically unbiased regardless of the size of the anomaly family. We illustrate the advantages of our estimator versus the MLE on disease outbreak data and highway traffic data.

----

## [175] Robust Learning-Augmented Caching: An Experimental Study

**Authors**: *Jakub Chledowski, Adam Polak, Bartosz Szabucki, Konrad Tomasz Zolna*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chledowski21a.html](http://proceedings.mlr.press/v139/chledowski21a.html)

**Abstract**:

Effective caching is crucial for performance of modern-day computing systems. A key optimization problem arising in caching – which item to evict to make room for a new item – cannot be optimally solved without knowing the future. There are many classical approximation algorithms for this problem, but more recently researchers started to successfully apply machine learning to decide what to evict by discovering implicit input patterns and predicting the future. While machine learning typically does not provide any worst-case guarantees, the new field of learning-augmented algorithms proposes solutions which leverage classical online caching algorithms to make the machine-learned predictors robust. We are the first to comprehensively evaluate these learning-augmented algorithms on real-world caching datasets and state-of-the-art machine-learned predictors. We show that a straightforward method – blindly following either a predictor or a classical robust algorithm, and switching whenever one becomes worse than the other – has only a low overhead over a well-performing predictor, while competing with classical methods when the coupled predictor fails, thus providing a cheap worst-case insurance.

----

## [176] Unifying Vision-and-Language Tasks via Text Generation

**Authors**: *Jaemin Cho, Jie Lei, Hao Tan, Mohit Bansal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cho21a.html](http://proceedings.mlr.press/v139/cho21a.html)

**Abstract**:

Existing methods for vision-and-language learning typically require designing task-specific architectures and objectives for each task. For example, a multi-label answer classifier for visual question answering, a region scorer for referring expression comprehension, and a language decoder for image captioning, etc. To alleviate these hassles, in this work, we propose a unified framework that learns different tasks in a single architecture with the same language modeling objective, i.e., multimodal conditional text generation, where our models learn to generate labels in text based on the visual and textual inputs. On 7 popular vision-and-language benchmarks, including visual question answering, referring expression comprehension, visual commonsense reasoning, most of which have been previously modeled as discriminative tasks, our generative approach (with a single unified architecture) reaches comparable performance to recent task-specific state-of-the-art vision-and-language models. Moreover, our generative approach shows better generalization ability on questions that have rare answers. Also, we show that our framework allows multi-task learning in a single architecture with a single set of parameters, achieving similar performance to separately optimized single-task models. Our code is publicly available at: https://github.com/j-min/VL-T5

----

## [177] Learning from Nested Data with Ornstein Auto-Encoders

**Authors**: *Youngwon Choi, Sungdong Lee, Joong-Ho Won*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/choi21a.html](http://proceedings.mlr.press/v139/choi21a.html)

**Abstract**:

Many of real-world data, e.g., the VGGFace2 dataset, which is a collection of multiple portraits of individuals, come with nested structures due to grouped observation. The Ornstein auto-encoder (OAE) is an emerging framework for representation learning from nested data, based on an optimal transport distance between random processes. An attractive feature of OAE is its ability to generate new variations nested within an observational unit, whether or not the unit is known to the model. A previously proposed algorithm for OAE, termed the random-intercept OAE (RIOAE), showed an impressive performance in learning nested representations, yet lacks theoretical justification. In this work, we show that RIOAE minimizes a loose upper bound of the employed optimal transport distance. After identifying several issues with RIOAE, we present the product-space OAE (PSOAE) that minimizes a tighter upper bound of the distance and achieves orthogonality in the representation space. PSOAE alleviates the instability of RIOAE and provides more flexible representation of nested data. We demonstrate the high performance of PSOAE in the three key tasks of generative models: exemplar generation, style transfer, and new concept generation.

----

## [178] Variational Empowerment as Representation Learning for Goal-Conditioned Reinforcement Learning

**Authors**: *Jongwook Choi, Archit Sharma, Honglak Lee, Sergey Levine, Shixiang Shane Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/choi21b.html](http://proceedings.mlr.press/v139/choi21b.html)

**Abstract**:

Learning to reach goal states and learning diverse skills through mutual information maximization have been proposed as principled frameworks for unsupervised reinforcement learning, allowing agents to acquire broadly applicable multi-task policies with minimal reward engineering. In this paper, we discuss how these two approaches {—} goal-conditioned RL (GCRL) and MI-based RL {—} can be generalized into a single family of methods, interpreting mutual information maximization and variational empowerment as representation learning methods that acquire function-ally aware state representations for goal reaching.Starting from a simple observation that the standard GCRL is encapsulated by the optimization objective of variational empowerment, we can derive novel variants of GCRL and variational empowerment under a single, unified optimization objective, such as adaptive-variance GCRL and linear-mapping GCRL, and study the characteristics of representation learning each variant provides. Furthermore, through the lens of GCRL, we show that adapting powerful techniques fromGCRL such as goal relabeling into the variationalMI context as well as proper regularization on the variational posterior provides substantial gains in algorithm performance, and propose a novel evaluation metric named latent goal reaching (LGR)as an objective measure for evaluating empowerment algorithms akin to goal-based RL. Through principled mathematical derivations and careful experimental validations, our work lays a novel foundation from which representation learning can be evaluated and analyzed in goal-based RL

----

## [179] Label-Only Membership Inference Attacks

**Authors**: *Christopher A. Choquette-Choo, Florian Tramèr, Nicholas Carlini, Nicolas Papernot*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/choquette-choo21a.html](http://proceedings.mlr.press/v139/choquette-choo21a.html)

**Abstract**:

Membership inference is one of the simplest privacy threats faced by machine learning models that are trained on private sensitive data. In this attack, an adversary infers whether a particular point was used to train the model, or not, by observing the model’s predictions. Whereas current attack methods all require access to the model’s predicted confidence score, we introduce a label-only attack that instead evaluates the robustness of the model’s predicted (hard) labels under perturbations of the input, to infer membership. Our label-only attack is not only as-effective as attacks requiring access to confidence scores, it also demonstrates that a class of defenses against membership inference, which we call “confidence masking” because they obfuscate the confidence scores to thwart attacks, are insufficient to prevent the leakage of private information. Our experiments show that training with differential privacy or strong L2 regularization are the only current defenses that meaningfully decrease leakage of private information, even for points that are outliers of the training distribution.

----

## [180] Modeling Hierarchical Structures with Continuous Recursive Neural Networks

**Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chowdhury21a.html](http://proceedings.mlr.press/v139/chowdhury21a.html)

**Abstract**:

Recursive Neural Networks (RvNNs), which compose sequences according to their underlying hierarchical syntactic structure, have performed well in several natural language processing tasks compared to similar models without structural biases. However, traditional RvNNs are incapable of inducing the latent structure in a plain text sequence on their own. Several extensions have been proposed to overcome this limitation. Nevertheless, these extensions tend to rely on surrogate gradients or reinforcement learning at the cost of higher bias or variance. In this work, we propose Continuous Recursive Neural Network (CRvNN) as a backpropagation-friendly alternative to address the aforementioned limitations. This is done by incorporating a continuous relaxation to the induced structure. We demonstrate that CRvNN achieves strong performance in challenging synthetic tasks such as logical inference (Bowman et al., 2015b) and ListOps (Nangia & Bowman, 2018). We also show that CRvNN performs comparably or better than prior latent structure models on real-world tasks such as sentiment analysis and natural language inference.

----

## [181] Scaling Multi-Agent Reinforcement Learning with Selective Parameter Sharing

**Authors**: *Filippos Christianos, Georgios Papoudakis, Arrasy Rahman, Stefano V. Albrecht*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/christianos21a.html](http://proceedings.mlr.press/v139/christianos21a.html)

**Abstract**:

Sharing parameters in multi-agent deep reinforcement learning has played an essential role in allowing algorithms to scale to a large number of agents. Parameter sharing between agents significantly decreases the number of trainable parameters, shortening training times to tractable levels, and has been linked to more efficient learning. However, having all agents share the same parameters can also have a detrimental effect on learning. We demonstrate the impact of parameter sharing methods on training speed and converged returns, establishing that when applied indiscriminately, their effectiveness is highly dependent on the environment. We propose a novel method to automatically identify agents which may benefit from sharing parameters by partitioning them based on their abilities and goals. Our approach combines the increased sample efficiency of parameter sharing with the representational capacity of multiple independent networks to reduce training time and increase final returns.

----

## [182] Beyond Variance Reduction: Understanding the True Impact of Baselines on Policy Optimization

**Authors**: *Wesley Chung, Valentin Thomas, Marlos C. Machado, Nicolas Le Roux*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/chung21a.html](http://proceedings.mlr.press/v139/chung21a.html)

**Abstract**:

Bandit and reinforcement learning (RL) problems can often be framed as optimization problems where the goal is to maximize average performance while having access only to stochastic estimates of the true gradient. Traditionally, stochastic optimization theory predicts that learning dynamics are governed by the curvature of the loss function and the noise of the gradient estimates. In this paper we demonstrate that the standard view is too limited for bandit and RL problems. To allow our analysis to be interpreted in light of multi-step MDPs, we focus on techniques derived from stochastic optimization principles (e.g., natural policy gradient and EXP3) and we show that some standard assumptions from optimization theory are violated in these problems. We present theoretical results showing that, at least for bandit problems, curvature and noise are not sufficient to explain the learning dynamics and that seemingly innocuous choices like the baseline can determine whether an algorithm converges. These theoretical findings match our empirical evaluation, which we extend to multi-state MDPs.

----

## [183] First-Order Methods for Wasserstein Distributionally Robust MDP

**Authors**: *Julien Grand-Clément, Christian Kroer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/clement21a.html](http://proceedings.mlr.press/v139/clement21a.html)

**Abstract**:

Markov decision processes (MDPs) are known to be sensitive to parameter specification. Distributionally robust MDPs alleviate this issue by allowing for \textit{ambiguity sets} which give a set of possible distributions over parameter sets. The goal is to find an optimal policy with respect to the worst-case parameter distribution. We propose a framework for solving Distributionally robust MDPs via first-order methods, and instantiate it for several types of Wasserstein ambiguity sets. By developing efficient proximal updates, our algorithms achieve a convergence rate of $O\left(NA^{2.5}S^{3.5}\log(S)\log(\epsilon^{-1})\epsilon^{-1.5} \right)$ for the number of kernels $N$ in the support of the nominal distribution, states $S$, and actions $A$; this rate varies slightly based on the Wasserstein setup. Our dependence on $N,A$ and $S$ is significantly better than existing methods, which have a complexity of $O\left(N^{3.5}A^{3.5}S^{4.5}\log^{2}(\epsilon^{-1}) \right)$. Numerical experiments show that our algorithm is significantly more scalable than state-of-the-art approaches across several domains.

----

## [184] Phasic Policy Gradient

**Authors**: *Karl Cobbe, Jacob Hilton, Oleg Klimov, John Schulman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cobbe21a.html](http://proceedings.mlr.press/v139/cobbe21a.html)

**Abstract**:

We introduce Phasic Policy Gradient (PPG), a reinforcement learning framework which modifies traditional on-policy actor-critic methods by separating policy and value function training into distinct phases. In prior methods, one must choose between using a shared network or separate networks to represent the policy and value function. Using separate networks avoids interference between objectives, while using a shared network allows useful features to be shared. PPG is able to achieve the best of both worlds by splitting optimization into two phases, one that advances training and one that distills features. PPG also enables the value function to be more aggressively optimized with a higher level of sample reuse. Compared to PPO, we find that PPG significantly improves sample efficiency on the challenging Procgen Benchmark.

----

## [185] Riemannian Convex Potential Maps

**Authors**: *Samuel Cohen, Brandon Amos, Yaron Lipman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cohen21a.html](http://proceedings.mlr.press/v139/cohen21a.html)

**Abstract**:

Modeling distributions on Riemannian manifolds is a crucial component in understanding non-Euclidean data that arises, e.g., in physics and geology. The budding approaches in this space are limited by representational and computational tradeoffs. We propose and study a class of flows that uses convex potentials from Riemannian optimal transport. These are universal and can model distributions on any compact Riemannian manifold without requiring domain knowledge of the manifold to be integrated into the architecture. We demonstrate that these flows can model standard distributions on spheres, and tori, on synthetic and geological data.

----

## [186] Scaling Properties of Deep Residual Networks

**Authors**: *Alain-Sam Cohen, Rama Cont, Alain Rossier, Renyuan Xu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cohen21b.html](http://proceedings.mlr.press/v139/cohen21b.html)

**Abstract**:

Residual networks (ResNets) have displayed impressive results in pattern recognition and, recently, have garnered considerable theoretical interest due to a perceived link with neural ordinary differential equations (neural ODEs). This link relies on the convergence of network weights to a smooth function as the number of layers increases. We investigate the properties of weights trained by stochastic gradient descent and their scaling with network depth through detailed numerical experiments. We observe the existence of scaling regimes markedly different from those assumed in neural ODE literature. Depending on certain features of the network architecture, such as the smoothness of the activation function, one may obtain an alternative ODE limit, a stochastic differential equation or neither of these. These findings cast doubts on the validity of the neural ODE model as an adequate asymptotic description of deep ResNets and point to an alternative class of differential equations as a better description of the deep network limit.

----

## [187] Differentially-Private Clustering of Easy Instances

**Authors**: *Edith Cohen, Haim Kaplan, Yishay Mansour, Uri Stemmer, Eliad Tsfadia*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cohen21c.html](http://proceedings.mlr.press/v139/cohen21c.html)

**Abstract**:

Clustering is a fundamental problem in data analysis. In differentially private clustering, the goal is to identify k cluster centers without disclosing information on individual data points. Despite significant research progress, the problem had so far resisted practical solutions. In this work we aim at providing simple implementable differentrially private clustering algorithms when the the data is "easy," e.g., when there exists a significant separation between the clusters. For the easy instances we consider, we have a simple implementation based on utilizing non-private clustering algorithms, and combining them privately. We are able to get improved sample complexity bounds in some cases of Gaussian mixtures and k-means. We complement our theoretical algorithms with experiments of simulated data.

----

## [188] Improving Ultrametrics Embeddings Through Coresets

**Authors**: *Vincent Cohen-Addad, Rémi de Joannis de Verclos, Guillaume Lagarde*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cohen-addad21a.html](http://proceedings.mlr.press/v139/cohen-addad21a.html)

**Abstract**:

To tackle the curse of dimensionality in data analysis and unsupervised learning, it is critical to be able to efficiently compute “simple” faithful representations of the data that helps extract information, improves understanding and visualization of the structure. When the dataset consists of $d$-dimensional vectors, simple representations of the data may consist in trees or ultrametrics, and the goal is to best preserve the distances (i.e.: dissimilarity values) between data elements. To circumvent the quadratic running times of the most popular methods for fitting ultrametrics, such as average, single, or complete linkage, \citet{CKL20} recently presented a new algorithm that for any $c \ge 1$, outputs in time $n^{1+O(1/c^2)}$ an ultrametric $\Delta$ such that for any two points $u, v$, $\Delta(u, v)$ is within a multiplicative factor of $5c$ to the distance between $u$ and $v$ in the “best” ultrametric representation. We improve the above result and show how to improve the above guarantee from $5c$ to $\sqrt{2}c + \varepsilon$ while achieving the same asymptotic running time. To complement the improved theoretical bound, we additionally show that the performances of our algorithm are significantly better for various real-world datasets.

----

## [189] Correlation Clustering in Constant Many Parallel Rounds

**Authors**: *Vincent Cohen-Addad, Silvio Lattanzi, Slobodan Mitrovic, Ashkan Norouzi-Fard, Nikos Parotsidis, Jakub Tarnawski*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cohen-addad21b.html](http://proceedings.mlr.press/v139/cohen-addad21b.html)

**Abstract**:

Correlation clustering is a central topic in unsupervised learning, with many applications in ML and data mining. In correlation clustering, one receives as input a signed graph and the goal is to partition it to minimize the number of disagreements. In this work we propose a massively parallel computation (MPC) algorithm for this problem that is considerably faster than prior work. In particular, our algorithm uses machines with memory sublinear in the number of nodes in the graph and returns a constant approximation while running only for a constant number of rounds. To the best of our knowledge, our algorithm is the first that can provably approximate a clustering problem using only a constant number of MPC rounds in the sublinear memory regime. We complement our analysis with an experimental scalability evaluation of our techniques.

----

## [190] Concentric mixtures of Mallows models for top-k rankings: sampling and identifiability

**Authors**: *Fabien Collas, Ekhine Irurozki*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/collas21a.html](http://proceedings.mlr.press/v139/collas21a.html)

**Abstract**:

In this paper, we study mixtures of two Mallows models for top-$k$ rankings with equal location parameters but with different scale parameters (a mixture of concentric Mallows models). These models arise when we have a heterogeneous population of voters formed by two populations, one of which is a subpopulation of expert voters. We show the identifiability of both components and the learnability of their respective parameters. These results are based upon, first, bounding the sample complexity for the Borda algorithm with top-$k$ rankings. Second, we characterize the distances between rankings, showing that an off-the-shelf clustering algorithm separates the rankings by components with high probability -provided the scales are well-separated.As a by-product, we include an efficient sampling algorithm for Mallows top-$k$ rankings. Finally, since the rank aggregation will suffer from a large amount of noise introduced by the non-expert voters, we adapt the Borda algorithm to be able to recover the ground truth consensus ranking which is especially consistent with the expert rankings.

----

## [191] Exploiting Shared Representations for Personalized Federated Learning

**Authors**: *Liam Collins, Hamed Hassani, Aryan Mokhtari, Sanjay Shakkottai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/collins21a.html](http://proceedings.mlr.press/v139/collins21a.html)

**Abstract**:

Deep neural networks have shown the ability to extract universal feature representations from data such as images and text that have been useful for a variety of learning tasks. However, the fruits of representation learning have yet to be fully-realized in federated settings. Although data in federated settings is often non-i.i.d. across clients, the success of centralized deep learning suggests that data often shares a global {\em feature representation}, while the statistical heterogeneity across clients or tasks is concentrated in the {\em labels}. Based on this intuition, we propose a novel federated learning framework and algorithm for learning a shared data representation across clients and unique local heads for each client. Our algorithm harnesses the distributed computational power across clients to perform many local-updates with respect to the low-dimensional local parameters for every update of the representation. We prove that this method obtains linear convergence to the ground-truth representation with near-optimal sample complexity in a linear setting, demonstrating that it can efficiently reduce the problem dimension for each client. Further, we provide extensive experimental results demonstrating the improvement of our method over alternative personalized federated learning approaches in heterogeneous settings.

----

## [192] Differentiable Particle Filtering via Entropy-Regularized Optimal Transport

**Authors**: *Adrien Corenflos, James Thornton, George Deligiannidis, Arnaud Doucet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/corenflos21a.html](http://proceedings.mlr.press/v139/corenflos21a.html)

**Abstract**:

Particle Filtering (PF) methods are an established class of procedures for performing inference in non-linear state-space models. Resampling is a key ingredient of PF necessary to obtain low variance likelihood and states estimates. However, traditional resampling methods result in PF-based loss functions being non-differentiable with respect to model and PF parameters. In a variational inference context, resampling also yields high variance gradient estimates of the PF-based evidence lower bound. By leveraging optimal transport ideas, we introduce a principled differentiable particle filter and provide convergence results. We demonstrate this novel method on a variety of applications.

----

## [193] Fairness and Bias in Online Selection

**Authors**: *José Correa, Andrés Cristi, Paul Duetting, Ashkan Norouzi-Fard*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/correa21a.html](http://proceedings.mlr.press/v139/correa21a.html)

**Abstract**:

There is growing awareness and concern about fairness in machine learning and algorithm design. This is particularly true in online selection problems where decisions are often biased, for example, when assessing credit risks or hiring staff. We address the issues of fairness and bias in online selection by introducing multi-color versions of the classic secretary and prophet problem. Interestingly, existing algorithms for these problems are either very unfair or very inefficient, so we develop optimal fair algorithms for these new problems and provide tight bounds on their competitiveness. We validate our theoretical findings on real-world data.

----

## [194] Relative Deviation Margin Bounds

**Authors**: *Corinna Cortes, Mehryar Mohri, Ananda Theertha Suresh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cortes21a.html](http://proceedings.mlr.press/v139/cortes21a.html)

**Abstract**:

We present a series of new and more favorable margin-based learning guarantees that depend on the empirical margin loss of a predictor. e give two types of learning bounds, in terms of either the Rademacher complexity or the empirical $\ell_\infty$-covering number of the hypothesis set used, both distribution-dependent and valid for general families. Furthermore, using our relative deviation margin bounds, we derive distribution-dependent generalization bounds for unbounded loss functions under the assumption of a finite moment. We also briefly highlight several applications of these bounds and discuss their connection with existing results.

----

## [195] A Discriminative Technique for Multiple-Source Adaptation

**Authors**: *Corinna Cortes, Mehryar Mohri, Ananda Theertha Suresh, Ningshan Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cortes21b.html](http://proceedings.mlr.press/v139/cortes21b.html)

**Abstract**:

We present a new discriminative technique for the multiple-source adaptation (MSA) problem. Unlike previous work, which relies on density estimation for each source domain, our solution only requires conditional probabilities that can be straightforwardly accurately estimated from unlabeled data from the source domains. We give a detailed analysis of our new technique, including general guarantees based on Rényi divergences, and learning bounds when conditional Maxent is used for estimating conditional probabilities for a point to belong to a source domain. We show that these guarantees compare favorably to those that can be derived for the generative solution, using kernel density estimation. Our experiments with real-world applications further demonstrate that our new discriminative MSA algorithm outperforms the previous generative solution as well as other domain adaptation baselines.

----

## [196] Characterizing Fairness Over the Set of Good Models Under Selective Labels

**Authors**: *Amanda Coston, Ashesh Rambachan, Alexandra Chouldechova*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/coston21a.html](http://proceedings.mlr.press/v139/coston21a.html)

**Abstract**:

Algorithmic risk assessments are used to inform decisions in a wide variety of high-stakes settings. Often multiple predictive models deliver similar overall performance but differ markedly in their predictions for individual cases, an empirical phenomenon known as the “Rashomon Effect.” These models may have different properties over various groups, and therefore have different predictive fairness properties. We develop a framework for characterizing predictive fairness properties over the set of models that deliver similar overall performance, or “the set of good models.” Our framework addresses the empirically relevant challenge of selectively labelled data in the setting where the selection decision and outcome are unconfounded given the observed data features. Our framework can be used to 1) audit for predictive bias; or 2) replace an existing model with one that has better fairness properties. We illustrate these use cases on a recidivism prediction task and a real-world credit-scoring task.

----

## [197] Two-way kernel matrix puncturing: towards resource-efficient PCA and spectral clustering

**Authors**: *Romain Couillet, Florent Chatelain, Nicolas Le Bihan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/couillet21a.html](http://proceedings.mlr.press/v139/couillet21a.html)

**Abstract**:

The article introduces an elementary cost and storage reduction method for spectral clustering and principal component analysis. The method consists in randomly “puncturing” both the data matrix $X\in\mathbb{C}^{p\times n}$ (or $\mathbb{R}^{p\times n}$) and its corresponding kernel (Gram) matrix $K$ through Bernoulli masks: $S\in\{0,1\}^{p\times n}$ for $X$ and $B\in\{0,1\}^{n\times n}$ for $K$. The resulting “two-way punctured” kernel is thus given by $K=\frac1p[(X\odot S)^\H (X\odot S)]\odot B$. We demonstrate that, for $X$ composed of independent columns drawn from a Gaussian mixture model, as $n,p\to\infty$ with $p/n\to c_0\in(0,\infty)$, the spectral behavior of $K$ – its limiting eigenvalue distribution, as well as its isolated eigenvalues and eigenvectors – is fully tractable and exhibits a series of counter-intuitive phenomena. We notably prove, and empirically confirm on various image databases, that it is possible to drastically puncture the data, thereby providing possibly huge computational and storage gains, for a virtually constant (clustering or PCA) performance. This preliminary study opens as such the path towards rethinking, from a large dimensional standpoint, computational and storage costs in elementary machine learning models.

----

## [198] Explaining Time Series Predictions with Dynamic Masks

**Authors**: *Jonathan Crabbé, Mihaela van der Schaar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/crabbe21a.html](http://proceedings.mlr.press/v139/crabbe21a.html)

**Abstract**:

How can we explain the predictions of a machine learning model? When the data is structured as a multivariate time series, this question induces additional difficulties such as the necessity for the explanation to embody the time dependency and the large number of inputs. To address these challenges, we propose dynamic masks (Dynamask). This method produces instance-wise importance scores for each feature at each time step by fitting a perturbation mask to the input sequence. In order to incorporate the time dependency of the data, Dynamask studies the effects of dynamic perturbation operators. In order to tackle the large number of inputs, we propose a scheme to make the feature selection parsimonious (to select no more feature than necessary) and legible (a notion that we detail by making a parallel with information theory). With synthetic and real-world data, we demonstrate that the dynamic underpinning of Dynamask, together with its parsimony, offer a neat improvement in the identification of feature importance over time. The modularity of Dynamask makes it ideal as a plug-in to increase the transparency of a wide range of machine learning models in areas such as medicine and finance, where time series are abundant.

----

## [199] Generalised Lipschitz Regularisation Equals Distributional Robustness

**Authors**: *Zac Cranko, Zhan Shi, Xinhua Zhang, Richard Nock, Simon Kornblith*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/cranko21a.html](http://proceedings.mlr.press/v139/cranko21a.html)

**Abstract**:

The problem of adversarial examples has highlighted the need for a theory of regularisation that is general enough to apply to exotic function classes, such as universal approximators. In response, we have been able to significantly sharpen existing results regarding the relationship between distributional robustness and regularisation, when defined with a transportation cost uncertainty set. The theory allows us to characterise the conditions under which the distributional robustness equals a Lipschitz-regularised model, and to tightly quantify, for the first time, the slackness under very mild assumptions. As a theoretical application we show a new result explicating the connection between adversarial learning and distributional robustness. We then give new results for how to achieve Lipschitz regularisation of kernel classifiers, which are demonstrated experimentally.

----



[Go to the next page](ICML-2021-list02.md)

[Go to the catalog section](README.md)