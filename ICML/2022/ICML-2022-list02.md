## [200] Understanding Robust Generalization in Learning Regular Languages

        **Authors**: *Soham Dan, Osbert Bastani, Dan Roth*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dan22a.html](https://proceedings.mlr.press/v162/dan22a.html)

        **Abstract**:

        A key feature of human intelligence is the ability to generalize beyond the training distribution, for instance, parsing longer sentences than seen in the past. Currently, deep neural networks struggle to generalize robustly to such shifts in the data distribution. We study robust generalization in the context of using recurrent neural networks (RNNs) to learn regular languages. We hypothesize that standard end-to-end modeling strategies cannot generalize well to systematic distribution shifts and propose a compositional strategy to address this. We compare an end-to-end strategy that maps strings to labels with a compositional strategy that predicts the structure of the deterministic finite state automaton (DFA) that accepts the regular language. We theoretically prove that the compositional strategy generalizes significantly better than the end-to-end strategy. In our experiments, we implement the compositional strategy via an auxiliary task where the goal is to predict the intermediate states visited by the DFA when parsing a string. Our empirical results support our hypothesis, showing that auxiliary tasks can enable robust generalization. Interestingly, the end-to-end RNN generalizes significantly better than the theoretical lower bound, suggesting that it is able to achieve atleast some degree of robust generalization.

        ----

        ## [201] Unsupervised Image Representation Learning with Deep Latent Particles

        **Authors**: *Tal Daniel, Aviv Tamar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/daniel22a.html](https://proceedings.mlr.press/v162/daniel22a.html)

        **Abstract**:

        We propose a new representation of visual data that disentangles object position from appearance. Our method, termed Deep Latent Particles (DLP), decomposes the visual input into low-dimensional latent “particles”, where each particle is described by its spatial location and features of its surrounding region. To drive learning of such representations, we follow a VAE-based based approach and introduce a prior for particle positions based on a spatial-Softmax architecture, and a modification of the evidence lower bound loss inspired by the Chamfer distance between particles. We demonstrate that our DLP representations are useful for downstream tasks such as unsupervised keypoint (KP) detection, image manipulation, and video prediction for scenes composed of multiple dynamic objects. In addition, we show that our probabilistic interpretation of the problem naturally provides uncertainty estimates for particle locations, which can be used for model selection, among other tasks.

        ----

        ## [202] Guarantees for Epsilon-Greedy Reinforcement Learning with Function Approximation

        **Authors**: *Christoph Dann, Yishay Mansour, Mehryar Mohri, Ayush Sekhari, Karthik Sridharan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dann22a.html](https://proceedings.mlr.press/v162/dann22a.html)

        **Abstract**:

        Myopic exploration policies such as epsilon-greedy, softmax, or Gaussian noise fail to explore efficiently in some reinforcement learning tasks and yet, they perform well in many others. In fact, in practice, they are often selected as the top choices, due to their simplicity. But, for what tasks do such policies succeed? Can we give theoretical guarantees for their favorable performance? These crucial questions have been scarcely investigated, despite the prominent practical importance of these policies. This paper presents a theoretical analysis of such policies and provides the first regret and sample-complexity bounds for reinforcement learning with myopic exploration. Our results apply to value-function-based algorithms in episodic MDPs with bounded Bellman Eluder dimension. We propose a new complexity measure called myopic exploration gap, denoted by alpha, that captures a structural property of the MDP, the exploration policy and the given value function class. We show that the sample-complexity of myopic exploration scales quadratically with the inverse of this quantity, 1 / alpha^2. We further demonstrate through concrete examples that myopic exploration gap is indeed favorable in several tasks where myopic exploration succeeds, due to the corresponding dynamics and reward structure.

        ----

        ## [203] Monarch: Expressive Structured Matrices for Efficient and Accurate Training

        **Authors**: *Tri Dao, Beidi Chen, Nimit Sharad Sohoni, Arjun D. Desai, Michael Poli, Jessica Grogan, Alexander Liu, Aniruddh Rao, Atri Rudra, Christopher Ré*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dao22a.html](https://proceedings.mlr.press/v162/dao22a.html)

        **Abstract**:

        Large neural networks excel in many domains, but they are expensive to train and fine-tune. A popular approach to reduce their compute or memory requirements is to replace dense weight matrices with structured ones (e.g., sparse, low-rank, Fourier transform). These methods have not seen widespread adoption (1) in end-to-end training due to unfavorable efficiency–quality tradeoffs, and (2) in dense-to-sparse fine-tuning due to lack of tractable algorithms to approximate a given dense weight matrix. To address these issues, we propose a class of matrices (Monarch) that is hardware-efficient (they are parameterized as products of two block-diagonal matrices for better hardware utilization) and expressive (they can represent many commonly used transforms). Surprisingly, the problem of approximating a dense weight matrix with a Monarch matrix, though nonconvex, has an analytical optimal solution. These properties of Monarch matrices unlock new ways to train and fine-tune sparse and dense models. We empirically validate that Monarch can achieve favorable accuracy-efficiency tradeoffs in several end-to-end sparse training applications: speeding up ViT and GPT-2 training on ImageNet classification and Wikitext-103 language modeling by 2x with comparable model quality, and reducing the error on PDE solving and MRI reconstruction tasks by 40%. In sparse-to-dense training, with a simple technique called "reverse sparsification," Monarch matrices serve as a useful intermediate representation to speed up GPT-2 pretraining on OpenWebText by 2x without quality drop. The same technique brings 23% faster BERT pretraining than even the very optimized implementation from Nvidia that set the MLPerf 1.1 record. In dense-to-sparse fine-tuning, as a proof-of-concept, our Monarch approximation algorithm speeds up BERT fine-tuning on GLUE by 1.7x with comparable accuracy.

        ----

        ## [204] Score-Guided Intermediate Level Optimization: Fast Langevin Mixing for Inverse Problems

        **Authors**: *Giannis Daras, Yuval Dagan, Alex Dimakis, Constantinos Daskalakis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/daras22a.html](https://proceedings.mlr.press/v162/daras22a.html)

        **Abstract**:

        We prove fast mixing and characterize the stationary distribution of the Langevin Algorithm for inverting random weighted DNN generators. This result extends the work of Hand and Voroninski from efficient inversion to efficient posterior sampling. In practice, to allow for increased expressivity, we propose to do posterior sampling in the latent space of a pre-trained generative model. To achieve that, we train a score-based model in the latent space of a StyleGAN-2 and we use it to solve inverse problems. Our framework, Score-Guided Intermediate Layer Optimization (SGILO), extends prior work by replacing the sparsity regularization with a generative prior in the intermediate layer. Experimentally, we obtain significant improvements over the previous state-of-the-art, especially in the low measurement regime.

        ----

        ## [205] Test-Time Training Can Close the Natural Distribution Shift Performance Gap in Deep Learning Based Compressed Sensing

        **Authors**: *Mohammad Zalbagi Darestani, Jiayu Liu, Reinhard Heckel*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/darestani22a.html](https://proceedings.mlr.press/v162/darestani22a.html)

        **Abstract**:

        Deep learning based image reconstruction methods outperform traditional methods. However, neural networks suffer from a performance drop when applied to images from a different distribution than the training images. For example, a model trained for reconstructing knees in accelerated magnetic resonance imaging (MRI) does not reconstruct brains well, even though the same network trained on brains reconstructs brains perfectly well. Thus there is a distribution shift performance gap for a given neural network, defined as the difference in performance when training on a distribution $P$ and training on another distribution $Q$, and evaluating both models on $Q$. In this work, we propose a domain adaptation method for deep learning based compressive sensing that relies on self-supervision during training paired with test-time training at inference. We show that for four natural distribution shifts, this method essentially closes the distribution shift performance gap for state-of-the-art architectures for accelerated MRI.

        ----

        ## [206] Knowledge Base Question Answering by Case-based Reasoning over Subgraphs

        **Authors**: *Rajarshi Das, Ameya Godbole, Ankita Naik, Elliot Tower, Manzil Zaheer, Hannaneh Hajishirzi, Robin Jia, Andrew McCallum*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/das22a.html](https://proceedings.mlr.press/v162/das22a.html)

        **Abstract**:

        Question answering (QA) over knowledge bases (KBs) is challenging because of the diverse, essentially unbounded, types of reasoning patterns needed. However, we hypothesize in a large KB, reasoning patterns required to answer a query type reoccur for various entities in their respective subgraph neighborhoods. Leveraging this structural similarity between local neighborhoods of different subgraphs, we introduce a semiparametric model (CBR-SUBG) with (i) a nonparametric component that for each query, dynamically retrieves other similar $k$-nearest neighbor (KNN) training queries along with query-specific subgraphs and (ii) a parametric component that is trained to identify the (latent) reasoning patterns from the subgraphs of KNN queries and then apply them to the subgraph of the target query. We also propose an adaptive subgraph collection strategy to select a query-specific compact subgraph, allowing us to scale to full Freebase KB containing billions of facts. We show that CBR-SUBG can answer queries requiring subgraph reasoning patterns and performs competitively with the best models on several KBQA benchmarks. Our subgraph collection strategy also produces more compact subgraphs (e.g. 55% reduction in size for WebQSP while increasing answer recall by 4.85%)\footnote{Code, model, and subgraphs are available at \url{https://github.com/rajarshd/CBR-SUBG}}.

        ----

        ## [207] Framework for Evaluating Faithfulness of Local Explanations

        **Authors**: *Sanjoy Dasgupta, Nave Frost, Michal Moshkovitz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dasgupta22a.html](https://proceedings.mlr.press/v162/dasgupta22a.html)

        **Abstract**:

        We study the faithfulness of an explanation system to the underlying prediction model. We show that this can be captured by two properties, consistency and sufficiency, and introduce quantitative measures of the extent to which these hold. Interestingly, these measures depend on the test-time data distribution. For a variety of existing explanation systems, such as anchors, we analytically study these quantities. We also provide estimators and sample complexity bounds for empirically determining the faithfulness of black-box explanation systems. Finally, we experimentally validate the new properties and estimators.

        ----

        ## [208] Distinguishing rule and exemplar-based generalization in learning systems

        **Authors**: *Ishita Dasgupta, Erin Grant, Tom Griffiths*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dasgupta22b.html](https://proceedings.mlr.press/v162/dasgupta22b.html)

        **Abstract**:

        Machine learning systems often do not share the same inductive biases as humans and, as a result, extrapolate or generalize in ways that are inconsistent with our expectations. The trade-off between exemplar- and rule-based generalization has been studied extensively in cognitive psychology; in this work, we present a protocol inspired by these experimental approaches to probe the inductive biases that control this trade-off in category-learning systems such as artificial neural networks. We isolate two such inductive biases: feature-level bias (differences in which features are more readily learned) and exemplar-vs-rule bias (differences in how these learned features are used for generalization of category labels). We find that standard neural network models are feature-biased and have a propensity towards exemplar-based extrapolation; we discuss the implications of these findings for machine-learning research on data augmentation, fairness, and systematic generalization.

        ----

        ## [209] Robust Multi-Objective Bayesian Optimization Under Input Noise

        **Authors**: *Samuel Daulton, Sait Cakmak, Maximilian Balandat, Michael A. Osborne, Enlu Zhou, Eytan Bakshy*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/daulton22a.html](https://proceedings.mlr.press/v162/daulton22a.html)

        **Abstract**:

        Bayesian optimization (BO) is a sample-efficient approach for tuning design parameters to optimize expensive-to-evaluate, black-box performance metrics. In many manufacturing processes, the design parameters are subject to random input noise, resulting in a product that is often less performant than expected. Although BO methods have been proposed for optimizing a single objective under input noise, no existing method addresses the practical scenario where there are multiple objectives that are sensitive to input perturbations. In this work, we propose the first multi-objective BO method that is robust to input noise. We formalize our goal as optimizing the multivariate value-at-risk (MVaR), a risk measure of the uncertain objectives. Since directly optimizing MVaR is computationally infeasible in many settings, we propose a scalable, theoretically-grounded approach for optimizing MVaR using random scalarizations. Empirically, we find that our approach significantly outperforms alternative methods and efficiently identifies optimal robust designs that will satisfy specifications across multiple metrics with high probability.

        ----

        ## [210] Attentional Meta-learners for Few-shot Polythetic Classification

        **Authors**: *Ben J. Day, Ramón Viñas Torné, Nikola Simidjievski, Pietro Lió*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/day22a.html](https://proceedings.mlr.press/v162/day22a.html)

        **Abstract**:

        Polythetic classifications, based on shared patterns of features that need neither be universal nor constant among members of a class, are common in the natural world and greatly outnumber monothetic classifications over a set of features. We show that threshold meta-learners, such as Prototypical Networks, require an embedding dimension that is exponential in the number of task-relevant features to emulate these functions. In contrast, attentional classifiers, such as Matching Networks, are polythetic by default and able to solve these problems with a linear embedding dimension. However, we find that in the presence of task-irrelevant features, inherent to meta-learning problems, attentional models are susceptible to misclassification. To address this challenge, we propose a self-attention feature-selection mechanism that adaptively dilutes non-discriminative features. We demonstrate the effectiveness of our approach in meta-learning Boolean functions, and synthetic and real-world few-shot learning tasks.

        ----

        ## [211] Adversarial Vulnerability of Randomized Ensembles

        **Authors**: *Hassan Dbouk, Naresh R. Shanbhag*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dbouk22a.html](https://proceedings.mlr.press/v162/dbouk22a.html)

        **Abstract**:

        Despite the tremendous success of deep neural networks across various tasks, their vulnerability to imperceptible adversarial perturbations has hindered their deployment in the real world. Recently, works on randomized ensembles have empirically demonstrated significant improvements in adversarial robustness over standard adversarially trained (AT) models with minimal computational overhead, making them a promising solution for safety-critical resource-constrained applications. However, this impressive performance raises the question: Are these robustness gains provided by randomized ensembles real? In this work we address this question both theoretically and empirically. We first establish theoretically that commonly employed robustness evaluation methods such as adaptive PGD provide a false sense of security in this setting. Subsequently, we propose a theoretically-sound and efficient adversarial attack algorithm (ARC) capable of compromising random ensembles even in cases where adaptive PGD fails to do so. We conduct comprehensive experiments across a variety of network architectures, training schemes, datasets, and norms to support our claims, and empirically establish that randomized ensembles are in fact more vulnerable to $\ell_p$-bounded adversarial perturbations than even standard AT models. Our code can be found at https://github.com/hsndbk4/ARC.

        ----

        ## [212] Born-Infeld (BI) for AI: Energy-Conserving Descent (ECD) for Optimization

        **Authors**: *Giuseppe Bruno De Luca, Eva Silverstein*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/de-luca22a.html](https://proceedings.mlr.press/v162/de-luca22a.html)

        **Abstract**:

        We introduce a novel framework for optimization based on energy-conserving Hamiltonian dynamics in a strongly mixing (chaotic) regime and establish its key properties analytically and numerically. The prototype is a discretization of Born-Infeld dynamics, with a squared relativistic speed limit depending on the objective function. This class of frictionless, energy-conserving optimizers proceeds unobstructed until slowing naturally near the minimal loss, which dominates the phase space volume of the system. Building from studies of chaotic systems such as dynamical billiards, we formulate a specific algorithm with good performance on machine learning and PDE-solving tasks, including generalization. It cannot stop at a high local minimum, an advantage in non-convex loss functions, and proceeds faster than GD+momentum in shallow valleys.

        ----

        ## [213] Error-driven Input Modulation: Solving the Credit Assignment Problem without a Backward Pass

        **Authors**: *Giorgia Dellaferrera, Gabriel Kreiman*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dellaferrera22a.html](https://proceedings.mlr.press/v162/dellaferrera22a.html)

        **Abstract**:

        Supervised learning in artificial neural networks typically relies on backpropagation, where the weights are updated based on the error-function gradients and sequentially propagated from the output layer to the input layer. Although this approach has proven effective in a wide domain of applications, it lacks biological plausibility in many regards, including the weight symmetry problem, the dependence of learning on non-local signals, the freezing of neural activity during error propagation, and the update locking problem. Alternative training schemes have been introduced, including sign symmetry, feedback alignment, and direct feedback alignment, but they invariably rely on a backward pass that hinders the possibility of solving all the issues simultaneously. Here, we propose to replace the backward pass with a second forward pass in which the input signal is modulated based on the error of the network. We show that this novel learning rule comprehensively addresses all the above-mentioned issues and can be applied to both fully connected and convolutional models. We test this learning rule on MNIST, CIFAR-10, and CIFAR-100. These results help incorporate biological principles into machine learning.

        ----

        ## [214] DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations

        **Authors**: *Fei Deng, Ingook Jang, Sungjin Ahn*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/deng22a.html](https://proceedings.mlr.press/v162/deng22a.html)

        **Abstract**:

        Reconstruction-based Model-Based Reinforcement Learning (MBRL) agents, such as Dreamer, often fail to discard task-irrelevant visual distractions that are prevalent in natural scenes. In this paper, we propose a reconstruction-free MBRL agent, called DreamerPro, that can enhance robustness to distractions. Motivated by the recent success of prototypical representations, a non-contrastive self-supervised learning approach in computer vision, DreamerPro combines Dreamer with prototypes. In order for the prototypes to benefit temporal dynamics learning in MBRL, we propose to additionally learn the prototypes from the recurrent states of the world model, thereby distilling temporal structures from past observations and actions into the prototypes. Experiments on the DeepMind Control suite show that DreamerPro achieves better overall performance than state-of-the-art contrastive MBRL agents when there are complex background distractions, and maintains similar performance as Dreamer in standard tasks where contrastive MBRL agents can perform much worse.

        ----

        ## [215] NeuralEF: Deconstructing Kernels by Deep Neural Networks

        **Authors**: *Zhijie Deng, Jiaxin Shi, Jun Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/deng22b.html](https://proceedings.mlr.press/v162/deng22b.html)

        **Abstract**:

        Learning the principal eigenfunctions of an integral operator defined by a kernel and a data distribution is at the core of many machine learning problems. Traditional nonparametric solutions based on the Nystrom formula suffer from scalability issues. Recent work has resorted to a parametric approach, i.e., training neural networks to approximate the eigenfunctions. However, the existing method relies on an expensive orthogonalization step and is difficult to implement. We show that these problems can be fixed by using a new series of objective functions that generalizes the EigenGame to function space. We test our method on a variety of supervised and unsupervised learning problems and show it provides accurate approximations to the eigenfunctions of polynomial, radial basis, neural network Gaussian process, and neural tangent kernels. Finally, we demonstrate our method can scale up linearised Laplace approximation of deep neural networks to modern image classification datasets through approximating the Gauss-Newton matrix. Code is available at https://github.com/thudzj/neuraleigenfunction.

        ----

        ## [216] Deep Causal Metric Learning

        **Authors**: *Xiang Deng, Zhongfei Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/deng22c.html](https://proceedings.mlr.press/v162/deng22c.html)

        **Abstract**:

        Deep metric learning aims to learn distance metrics that measure similarities and dissimilarities between samples. The existing approaches typically focus on designing different hard sample mining or distance margin strategies and then minimize a pair/triplet-based or proxy-based loss over the training data. However, this can lead the model to recklessly learn all the correlated distances found in training data including the spurious distance (e.g., background differences) that is not the distance of interest and can harm the generalization of the learned metric. To address this issue, we study metric learning from a causality perspective and accordingly propose deep causal metric learning (DCML) that pursues the true causality of the distance between samples. DCML is achieved through explicitly learning environment-invariant attention and task-invariant embedding based on causal inference. Extensive experiments on several benchmark datasets demonstrate the superiority of DCML over the existing methods.

        ----

        ## [217] On the Convergence of Inexact Predictor-Corrector Methods for Linear Programming

        **Authors**: *Gregory Dexter, Agniva Chowdhury, Haim Avron, Petros Drineas*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dexter22a.html](https://proceedings.mlr.press/v162/dexter22a.html)

        **Abstract**:

        Interior point methods (IPMs) are a common approach for solving linear programs (LPs) with strong theoretical guarantees and solid empirical performance. The time complexity of these methods is dominated by the cost of solving a linear system of equations at each iteration. In common applications of linear programming, particularly in machine learning and scientific computing, the size of this linear system can become prohibitively large, requiring the use of iterative solvers, which provide an approximate solution to the linear system. However, approximately solving the linear system at each iteration of an IPM invalidates the theoretical guarantees of common IPM analyses. To remedy this, we theoretically and empirically analyze (slightly modified) predictor-corrector IPMs when using approximate linear solvers: our approach guarantees that, when certain conditions are satisfied, the number of IPM iterations does not increase and that the final solution remains feasible. We also provide practical instantiations of approximate linear solvers that satisfy these conditions for special classes of constraint matrices using randomized linear algebra.

        ----

        ## [218] Analysis of Stochastic Processes through Replay Buffers

        **Authors**: *Shirli Di-Castro Shashua, Shie Mannor, Dotan Di Castro*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/di-castro22a.html](https://proceedings.mlr.press/v162/di-castro22a.html)

        **Abstract**:

        Replay buffers are a key component in many reinforcement learning schemes. Yet, their theoretical properties are not fully understood. In this paper we analyze a system where a stochastic process X is pushed into a replay buffer and then randomly sampled to generate a stochastic process Y from the replay buffer. We provide an analysis of the properties of the sampled process such as stationarity, Markovity and autocorrelation in terms of the properties of the original process. Our theoretical analysis sheds light on why replay buffer may be a good de-correlator. Our analysis provides theoretical tools for proving the convergence of replay buffer based algorithms which are prevalent in reinforcement learning schemes.

        ----

        ## [219] Streaming Algorithms for High-Dimensional Robust Statistics

        **Authors**: *Ilias Diakonikolas, Daniel M. Kane, Ankit Pensia, Thanasis Pittas*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/diakonikolas22a.html](https://proceedings.mlr.press/v162/diakonikolas22a.html)

        **Abstract**:

        We study high-dimensional robust statistics tasks in the streaming model. A recent line of work obtained computationally efficient algorithms for a range of high-dimensional robust statistics tasks. Unfortunately, all previous algorithms require storing the entire dataset, incurring memory at least quadratic in the dimension. In this work, we develop the first efficient streaming algorithms for high-dimensional robust statistics with near-optimal memory requirements (up to logarithmic factors). Our main result is for the task of high-dimensional robust mean estimation in (a strengthening of) Huber’s contamination model. We give an efficient single-pass streaming algorithm for this task with near-optimal error guarantees and space complexity nearly-linear in the dimension. As a corollary, we obtain streaming algorithms with near-optimal space complexity for several more complex tasks, including robust covariance estimation, robust regression, and more generally robust stochastic optimization.

        ----

        ## [220] Learning General Halfspaces with Adversarial Label Noise via Online Gradient Descent

        **Authors**: *Ilias Diakonikolas, Vasilis Kontonis, Christos Tzamos, Nikos Zarifis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/diakonikolas22b.html](https://proceedings.mlr.press/v162/diakonikolas22b.html)

        **Abstract**:

        We study the problem of learning general {—} i.e., not necessarily homogeneous {—} halfspaces with adversarial label noise under the Gaussian distribution. Prior work has provided a sophisticated polynomial-time algorithm for this problem. In this work, we show that the problem can be solved directly via online gradient descent applied to a sequence of natural non-convex surrogates. This approach yields a simple iterative learning algorithm for general halfspaces with near-optimal sample complexity, runtime, and error guarantee. At the conceptual level, our work establishes an intriguing connection between learning halfspaces with adversarial noise and online optimization that may find other applications.

        ----

        ## [221] Variational Feature Pyramid Networks

        **Authors**: *Panagiotis Dimitrakopoulos, Giorgos Sfikas, Christophoros Nikou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dimitrakopoulos22a.html](https://proceedings.mlr.press/v162/dimitrakopoulos22a.html)

        **Abstract**:

        Recent architectures for object detection adopt a Feature Pyramid Network as a backbone for deep feature extraction. Many works focus on the design of pyramid networks which produce richer feature representations. In this work, we opt to learn a dataset-specific architecture for Feature Pyramid Networks. With the proposed method, the network fuses features at multiple scales, it is efficient in terms of parameters and operations, and yields better results across a variety of tasks and datasets. Starting by a complex network, we adopt Variational Inference to prune redundant connections. Our model, integrated with standard detectors, outperforms the state-of-the-art feature fusion networks.

        ----

        ## [222] Understanding Doubly Stochastic Clustering

        **Authors**: *Tianjiao Ding, Derek Lim, René Vidal, Benjamin D. Haeffele*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ding22a.html](https://proceedings.mlr.press/v162/ding22a.html)

        **Abstract**:

        The problem of projecting a matrix onto the space of doubly stochastic matrices finds several applications in machine learning. For example, in spectral clustering, it has been shown that forming the normalized Laplacian matrix from a data affinity matrix has close connections to projecting it onto the set of doubly stochastic matrices. However, the analysis of why this projection improves clustering has been limited. In this paper we present theoretical conditions on the given affinity matrix under which its doubly stochastic projection is an ideal affinity matrix (i.e., it has no false connections between clusters, and is well-connected within each cluster). In particular, we show that a necessary and sufficient condition for a projected affinity matrix to be ideal reduces to a set of conditions on the input affinity that decompose along each cluster. Further, in the subspace clustering problem, where each cluster is defined by a linear subspace, we provide geometric conditions on the underlying subspaces which guarantee correct clustering via a continuous version of the problem. This allows us to explain theoretically the remarkable performance of a recently proposed doubly stochastic subspace clustering method.

        ----

        ## [223] Independent Policy Gradient for Large-Scale Markov Potential Games: Sharper Rates, Function Approximation, and Game-Agnostic Convergence

        **Authors**: *Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, Mihailo R. Jovanovic*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ding22b.html](https://proceedings.mlr.press/v162/ding22b.html)

        **Abstract**:

        We examine global non-asymptotic convergence properties of policy gradient methods for multi-agent reinforcement learning (RL) problems in Markov potential games (MPGs). To learn a Nash equilibrium of an MPG in which the size of state space and/or the number of players can be very large, we propose new independent policy gradient algorithms that are run by all players in tandem. When there is no uncertainty in the gradient evaluation, we show that our algorithm finds an $\epsilon$-Nash equilibrium with $O(1/\epsilon^2)$ iteration complexity which does not explicitly depend on the state space size. When the exact gradient is not available, we establish $O(1/\epsilon^5)$ sample complexity bound in a potentially infinitely large state space for a sample-based algorithm that utilizes function approximation. Moreover, we identify a class of independent policy gradient algorithms that enjoy convergence for both zero-sum Markov games and Markov cooperative games with the players that are oblivious to the types of games being played. Finally, we provide computational experiments to corroborate the merits and the effectiveness of our theoretical developments.

        ----

        ## [224] Generalization and Robustness Implications in Object-Centric Learning

        **Authors**: *Andrea Dittadi, Samuele S. Papa, Michele De Vita, Bernhard Schölkopf, Ole Winther, Francesco Locatello*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dittadi22a.html](https://proceedings.mlr.press/v162/dittadi22a.html)

        **Abstract**:

        The idea behind object-centric representation learning is that natural scenes can better be modeled as compositions of objects and their relations as opposed to distributed representations. This inductive bias can be injected into neural networks to potentially improve systematic generalization and performance of downstream tasks in scenes with multiple objects. In this paper, we train state-of-the-art unsupervised models on five common multi-object datasets and evaluate segmentation metrics and downstream object property prediction. In addition, we study generalization and robustness by investigating the settings where either a single object is out of distribution – e.g., having an unseen color, texture, or shape – or global properties of the scene are altered – e.g., by occlusions, cropping, or increasing the number of objects. From our experimental study, we find object-centric representations to be useful for downstream tasks and generally robust to most distribution shifts affecting objects. However, when the distribution shift affects the input in a less structured manner, robustness in terms of segmentation and downstream task performance may vary significantly across models and distribution shifts.

        ----

        ## [225] Fair Generalized Linear Models with a Convex Penalty

        **Authors**: *Hyungrok Do, Preston Putzel, Axel S. Martin, Padhraic Smyth, Judy Zhong*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/do22a.html](https://proceedings.mlr.press/v162/do22a.html)

        **Abstract**:

        Despite recent advances in algorithmic fairness, methodologies for achieving fairness with generalized linear models (GLMs) have yet to be explored in general, despite GLMs being widely used in practice. In this paper we introduce two fairness criteria for GLMs based on equalizing expected outcomes or log-likelihoods. We prove that for GLMs both criteria can be achieved via a convex penalty term based solely on the linear components of the GLM, thus permitting efficient optimization. We also derive theoretical properties for the resulting fair GLM estimator. To empirically demonstrate the efficacy of the proposed fair GLM, we compare it with other well-known fair prediction methods on an extensive set of benchmark datasets for binary classification and regression. In addition, we demonstrate that the fair GLM can generate fair predictions for a range of response variables, other than binary and continuous outcomes.

        ----

        ## [226] Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense

        **Authors**: *Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/doan22a.html](https://proceedings.mlr.press/v162/doan22a.html)

        **Abstract**:

        We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the learning approach for approximating the multi-modal posterior distribution of an adversarially trained Bayesian model can lead to mode collapse; consequently, the model’s achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step towards a basis for a principled method of adversarially training BNNs. Our extensive experimental results demonstrate significantly improved robustness up to 20% compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 dataset.

        ----

        ## [227] On the Adversarial Robustness of Causal Algorithmic Recourse

        **Authors**: *Ricardo Dominguez-Olmedo, Amir-Hossein Karimi, Bernhard Schölkopf*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dominguez-olmedo22a.html](https://proceedings.mlr.press/v162/dominguez-olmedo22a.html)

        **Abstract**:

        Algorithmic recourse seeks to provide actionable recommendations for individuals to overcome unfavorable classification outcomes from automated decision-making systems. Recourse recommendations should ideally be robust to reasonably small uncertainty in the features of the individual seeking recourse. In this work, we formulate the adversarially robust recourse problem and show that recourse methods that offer minimally costly recourse fail to be robust. We then present methods for generating adversarially robust recourse for linear and for differentiable classifiers. Finally, we show that regularizing the decision-making classifier to behave locally linearly and to rely more strongly on actionable features facilitates the existence of adversarially robust recourse.

        ----

        ## [228] Finding the Task-Optimal Low-Bit Sub-Distribution in Deep Neural Networks

        **Authors**: *Runpei Dong, Zhanhong Tan, Mengdi Wu, Linfeng Zhang, Kaisheng Ma*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dong22a.html](https://proceedings.mlr.press/v162/dong22a.html)

        **Abstract**:

        Quantized neural networks typically require smaller memory footprints and lower computation complexity, which is crucial for efficient deployment. However, quantization inevitably leads to a distribution divergence from the original network, which generally degrades the performance. To tackle this issue, massive efforts have been made, but most existing approaches lack statistical considerations and depend on several manual configurations. In this paper, we present an adaptive-mapping quantization method to learn an optimal latent sub-distribution that is inherent within models and smoothly approximated with a concrete Gaussian Mixture (GM). In particular, the network weights are projected in compliance with the GM-approximated sub-distribution. This sub-distribution evolves along with the weight update in a co-tuning schema guided by the direct task-objective optimization. Sufficient experiments on image classification and object detection over various modern architectures demonstrate the effectiveness, generalization property, and transferability of the proposed method. Besides, an efficient deployment flow for the mobile CPU is developed, achieving up to 7.46$\times$ inference acceleration on an octa-core ARM CPU. Our codes have been publicly released at https://github.com/RunpeiDong/DGMS.

        ----

        ## [229] PACE: A Parallelizable Computation Encoder for Directed Acyclic Graphs

        **Authors**: *Zehao Dong, Muhan Zhang, Fuhai Li, Yixin Chen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dong22b.html](https://proceedings.mlr.press/v162/dong22b.html)

        **Abstract**:

        Optimization of directed acyclic graph (DAG) structures has many applications, such as neural architecture search (NAS) and probabilistic graphical model learning. Encoding DAGs into real vectors is a dominant component in most neural-network-based DAG optimization frameworks. Currently, most popular DAG encoders use an asynchronous message passing scheme which sequentially processes nodes according to the dependency between nodes in a DAG. That is, a node must not be processed until all its predecessors are processed. As a result, they are inherently not parallelizable. In this work, we propose a Parallelizable Attention-based Computation structure Encoder (PACE) that processes nodes simultaneously and encodes DAGs in parallel. We demonstrate the superiority of PACE through encoder-dependent optimization subroutines that search the optimal DAG structure based on the learned DAG embeddings. Experiments show that PACE not only improves the effectiveness over previous sequential DAG encoders with a significantly boosted training and inference speed, but also generates smooth latent (DAG encoding) spaces that are beneficial to downstream optimization subroutines.

        ----

        ## [230] Privacy for Free: How does Dataset Condensation Help Privacy?

        **Authors**: *Tian Dong, Bo Zhao, Lingjuan Lyu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dong22c.html](https://proceedings.mlr.press/v162/dong22c.html)

        **Abstract**:

        To prevent unintentional data leakage, research community has resorted to data generators that can produce differentially private data for model training. However, for the sake of the data privacy, existing solutions suffer from either expensive training cost or poor generalization performance. Therefore, we raise the question whether training efficiency and privacy can be achieved simultaneously. In this work, we for the first time identify that dataset condensation (DC) which is originally designed for improving training efficiency is also a better solution to replace the traditional data generators for private data generation, thus providing privacy for free. To demonstrate the privacy benefit of DC, we build a connection between DC and differential privacy, and theoretically prove on linear feature extractors (and then extended to non-linear feature extractors) that the existence of one sample has limited impact ($O(m/n)$) on the parameter distribution of networks trained on $m$ samples synthesized from $n (n \gg m)$ raw samples by DC. We also empirically validate the visual privacy and membership privacy of DC-synthesized data by launching both the loss-based and the state-of-the-art likelihood-based membership inference attacks. We envision this work as a milestone for data-efficient and privacy-preserving machine learning.

        ----

        ## [231] Fast rates for noisy interpolation require rethinking the effect of inductive bias

        **Authors**: *Konstantin Donhauser, Nicolò Ruggeri, Stefan Stojanovic, Fanny Yang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/donhauser22a.html](https://proceedings.mlr.press/v162/donhauser22a.html)

        **Abstract**:

        Good generalization performance on high-dimensional data crucially hinges on a simple structure of the ground truth and a corresponding strong inductive bias of the estimator. Even though this intuition is valid for regularized models, in this paper we caution against a strong inductive bias for interpolation in the presence of noise: While a stronger inductive bias encourages a simpler structure that is more aligned with the ground truth, it also increases the detrimental effect of noise. Specifically, for both linear regression and classification with a sparse ground truth, we prove that minimum $\ell_p$-norm and maximum $\ell_p$-margin interpolators achieve fast polynomial rates close to order $1/n$ for $p > 1$ compared to a logarithmic rate for $p = 1$. Finally, we provide preliminary experimental evidence that this trade-off may also play a crucial role in understanding non-linear interpolating models used in practice.

        ----

        ## [232] Adapting to Mixing Time in Stochastic Optimization with Markovian Data

        **Authors**: *Ron Dorfman, Kfir Yehuda Levy*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dorfman22a.html](https://proceedings.mlr.press/v162/dorfman22a.html)

        **Abstract**:

        We consider stochastic optimization problems where data is drawn from a Markov chain. Existing methods for this setting crucially rely on knowing the mixing time of the chain, which in real-world applications is usually unknown. We propose the first optimization method that does not require the knowledge of the mixing time, yet obtains the optimal asymptotic convergence rate when applied to convex problems. We further show that our approach can be extended to: (i) finding stationary points in non-convex optimization with Markovian data, and (ii) obtaining better dependence on the mixing time in temporal difference (TD) learning; in both cases, our method is completely oblivious to the mixing time. Our method relies on a novel combination of multi-level Monte Carlo (MLMC) gradient estimation together with an adaptive learning method.

        ----

        ## [233] TACTiS: Transformer-Attentional Copulas for Time Series

        **Authors**: *Alexandre Drouin, Étienne Marcotte, Nicolas Chapados*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/drouin22a.html](https://proceedings.mlr.press/v162/drouin22a.html)

        **Abstract**:

        The estimation of time-varying quantities is a fundamental component of decision making in fields such as healthcare and finance. However, the practical utility of such estimates is limited by how accurately they quantify predictive uncertainty. In this work, we address the problem of estimating the joint predictive distribution of high-dimensional multivariate time series. We propose a versatile method, based on the transformer architecture, that estimates joint distributions using an attention-based decoder that provably learns to mimic the properties of non-parametric copulas. The resulting model has several desirable properties: it can scale to hundreds of time series, supports both forecasting and interpolation, can handle unaligned and non-uniformly sampled data, and can seamlessly adapt to missing data during training. We demonstrate these properties empirically and show that our model produces state-of-the-art predictions on multiple real-world datasets.

        ----

        ## [234] Branching Reinforcement Learning

        **Authors**: *Yihan Du, Wei Chen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/du22a.html](https://proceedings.mlr.press/v162/du22a.html)

        **Abstract**:

        In this paper, we propose a novel Branching Reinforcement Learning (Branching RL) model, and investigate both Regret Minimization (RM) and Reward-Free Exploration (RFE) metrics for this model. Unlike standard RL where the trajectory of each episode is a single $H$-step path, branching RL allows an agent to take multiple base actions in a state such that transitions branch out to multiple successor states correspondingly, and thus it generates a tree-structured trajectory. This model finds important applications in hierarchical recommendation systems and online advertising. For branching RL, we establish new Bellman equations and key lemmas, i.e., branching value difference lemma and branching law of total variance, and also bound the total variance by only $O(H^2)$ under an exponentially-large trajectory. For RM and RFE metrics, we propose computationally efficient algorithms BranchVI and BranchRFE, respectively, and derive nearly matching upper and lower bounds. Our regret and sample complexity results are polynomial in all problem parameters despite exponentially-large trajectories.

        ----

        ## [235] Bayesian Imitation Learning for End-to-End Mobile Manipulation

        **Authors**: *Yuqing Du, Daniel Ho, Alex Alemi, Eric Jang, Mohi Khansari*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/du22b.html](https://proceedings.mlr.press/v162/du22b.html)

        **Abstract**:

        In this work we investigate and demonstrate benefits of a Bayesian approach to imitation learning from multiple sensor inputs, as applied to the task of opening office doors with a mobile manipulator. Augmenting policies with additional sensor inputs{—}such as RGB + depth cameras{—}is a straightforward approach to improving robot perception capabilities, especially for tasks that may favor different sensors in different situations. As we scale multi-sensor robotic learning to unstructured real-world settings (e.g. offices, homes) and more complex robot behaviors, we also increase reliance on simulators for cost, efficiency, and safety. Consequently, the sim-to-real gap across multiple sensor modalities also increases, making simulated validation more difficult. We show that using the Variational Information Bottleneck (Alemi et al., 2016) to regularize convolutional neural networks improves generalization to heldout domains and reduces the sim-to-real gap in a sensor-agnostic manner. As a side effect, the learned embeddings also provide useful estimates of model uncertainty for each sensor. We demonstrate that our method is able to help close the sim-to-real gap and successfully fuse RGB and depth modalities based on understanding of the situational uncertainty of each sensor. In a real-world office environment, we achieve 96% task success, improving upon the baseline by +16%.

        ----

        ## [236] GLaM: Efficient Scaling of Language Models with Mixture-of-Experts

        **Authors**: *Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten P. Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen S. Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, Claire Cui*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/du22c.html](https://proceedings.mlr.press/v162/du22c.html)

        **Abstract**:

        Scaling language models with more data, compute and parameters has driven significant progress in natural language processing. For example, thanks to scaling, GPT-3 was able to achieve strong results on in-context learning tasks. However, training these large dense models requires significant amounts of computing resources. In this paper, we propose and develop a family of language models named \glam (\textbf{G}eneralist \textbf{La}nguage \textbf{M}odel), which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants. The largest \glam has 1.2 trillion parameters, which is approximately 7x larger than GPT-3. It consumes only 1/3 of the energy used to train GPT-3 and requires half of the computation flops for inference, while still achieving better overall fewshot performance across 29 NLP tasks.

        ----

        ## [237] Learning Iterative Reasoning through Energy Minimization

        **Authors**: *Yilun Du, Shuang Li, Joshua B. Tenenbaum, Igor Mordatch*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/du22d.html](https://proceedings.mlr.press/v162/du22d.html)

        **Abstract**:

        Deep learning has excelled on complex pattern recognition tasks such as image classification and object recognition. However, it struggles with tasks requiring nontrivial reasoning, such as algorithmic computation. Humans are able to solve such tasks through iterative reasoning – spending more time to think about harder tasks. Most existing neural networks, however, exhibit a fixed computational budget controlled by the neural network architecture, preventing additional computational processing on harder tasks. In this work, we present a new framework for iterative reasoning with neural networks. We train a neural network to parameterize an energy landscape over all outputs, and implement each step of the iterative reasoning as an energy minimization step to find a minimal energy solution. By formulating reasoning as an energy minimization problem, for harder problems that lead to more complex energy landscapes, we may then adjust our underlying computational budget by running a more complex optimization procedure. We empirically illustrate that our iterative reasoning approach can solve more accurate and generalizable algorithmic reasoning tasks in both graph and continuous domains. Finally, we illustrate that our approach can recursively solve algorithmic problems requiring nested reasoning.

        ----

        ## [238] SE(3) Equivariant Graph Neural Networks with Complete Local Frames

        **Authors**: *Weitao Du, He Zhang, Yuanqi Du, Qi Meng, Wei Chen, Nanning Zheng, Bin Shao, Tie-Yan Liu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/du22e.html](https://proceedings.mlr.press/v162/du22e.html)

        **Abstract**:

        Group equivariance (e.g. SE(3) equivariance) is a critical physical symmetry in science, from classical and quantum physics to computational biology. It enables robust and accurate prediction under arbitrary reference transformations. In light of this, great efforts have been put on encoding this symmetry into deep neural networks, which has been shown to improve the generalization performance and data efficiency for downstream tasks. Constructing an equivariant neural network generally brings high computational costs to ensure expressiveness. Therefore, how to better trade-off the expressiveness and computational efficiency plays a core role in the design of the equivariant deep learning models. In this paper, we propose a framework to construct SE(3) equivariant graph neural networks that can approximate the geometric quantities efficiently. Inspired by differential geometry and physics, we introduce equivariant local complete frames to graph neural networks, such that tensor information at given orders can be projected onto the frames. The local frame is constructed to form an orthonormal basis that avoids direction degeneration and ensure completeness. Since the frames are built only by cross product operations, our method is computationally efficient. We evaluate our method on two tasks: Newton mechanics modeling and equilibrium molecule conformation generation. Extensive experimental results demonstrate that our model achieves the best or competitive performance in two types of datasets.

        ----

        ## [239] A Context-Integrated Transformer-Based Neural Network for Auction Design

        **Authors**: *Zhijian Duan, Jingwu Tang, Yutong Yin, Zhe Feng, Xiang Yan, Manzil Zaheer, Xiaotie Deng*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/duan22a.html](https://proceedings.mlr.press/v162/duan22a.html)

        **Abstract**:

        One of the central problems in auction design is developing an incentive-compatible mechanism that maximizes the auctioneer’s expected revenue. While theoretical approaches have encountered bottlenecks in multi-item auctions, recently, there has been much progress on finding the optimal mechanism through deep learning. However, these works either focus on a fixed set of bidders and items, or restrict the auction to be symmetric. In this work, we overcome such limitations by factoring public contextual information of bidders and items into the auction learning framework. We propose $\mathtt{CITransNet}$, a context-integrated transformer-based neural network for optimal auction design, which maintains permutation-equivariance over bids and contexts while being able to find asymmetric solutions. We show by extensive experiments that $\mathtt{CITransNet}$ can recover the known optimal solutions in single-item settings, outperform strong baselines in multi-item auctions, and generalize well to cases other than those in training.

        ----

        ## [240] Augment with Care: Contrastive Learning for Combinatorial Problems

        **Authors**: *Haonan Duan, Pashootan Vaezipoor, Max B. Paulus, Yangjun Ruan, Chris J. Maddison*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/duan22b.html](https://proceedings.mlr.press/v162/duan22b.html)

        **Abstract**:

        Supervised learning can improve the design of state-of-the-art solvers for combinatorial problems, but labelling large numbers of combinatorial instances is often impractical due to exponential worst-case complexity. Inspired by the recent success of contrastive pre-training for images, we conduct a scientific study of the effect of augmentation design on contrastive pre-training for the Boolean satisfiability problem. While typical graph contrastive pre-training uses label-agnostic augmentations, our key insight is that many combinatorial problems have well-studied invariances, which allow for the design of label-preserving augmentations. We find that label-preserving augmentations are critical for the success of contrastive pre-training. We show that our representations are able to achieve comparable test accuracy to fully-supervised learning while using only 1% of the labels. We also demonstrate that our representations are more transferable to larger problems from unseen domains. Our code is available at https://github.com/h4duan/contrastive-sat.

        ----

        ## [241] Parametric Visual Program Induction with Function Modularization

        **Authors**: *Xuguang Duan, Xin Wang, Ziwei Zhang, Wenwu Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/duan22c.html](https://proceedings.mlr.press/v162/duan22c.html)

        **Abstract**:

        Generating programs to describe visual observations has gained much research attention recently. However, most of the existing approaches are based on non-parametric primitive functions, making them unable to handle complex visual scenes involving many attributes and details. In this paper, we propose the concept of parametric visual program induction. Learning to generate parametric programs for visual scenes is challenging due to the huge number of function variants and the complex function correlations. To solve these challenges, we propose the method of function modularization, capable of dealing with numerous function variants and complex correlations. Specifically, we model each parametric function as a multi-head self-contained neural module to cover different function variants. Moreover, to eliminate the complex correlations between functions, we propose the hierarchical heterogeneous Monto-Carlo tree search (H2MCTS) algorithm which can provide high-quality uncorrelated supervision during training, and serve as an efficient searching technique during testing. We demonstrate the superiority of the proposed method on three visual program induction datasets involving parametric primitive functions. Experimental results show that our proposed model is able to significantly outperform the state-of-the-art baseline methods in terms of generating accurate programs.

        ----

        ## [242] Bayesian Deep Embedding Topic Meta-Learner

        **Authors**: *Zhibin Duan, Yishi Xu, Jianqiao Sun, Bo Chen, Wenchao Chen, Chaojie Wang, Mingyuan Zhou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/duan22d.html](https://proceedings.mlr.press/v162/duan22d.html)

        **Abstract**:

        Existing deep topic models are effective in capturing the latent semantic structures in textual data but usually rely on a plethora of documents. This is less than satisfactory in practical applications when only a limited amount of data is available. In this paper, we propose a novel framework that efficiently solves the problem of topic modeling under the small data regime. Specifically, the framework involves two innovations: a bi-level generative model that aims to exploit the task information to guide the document generation, and a topic meta-learner that strives to learn a group of global topic embeddings so that fast adaptation to the task-specific topic embeddings can be achieved with a few examples. We apply the proposed framework to a hierarchical embedded topic model and achieve better performance than various baseline models on diverse experiments, including few-shot topic discovery and few-shot document classification.

        ----

        ## [243] Deletion Robust Submodular Maximization over Matroids

        **Authors**: *Paul Duetting, Federico Fusco, Silvio Lattanzi, Ashkan Norouzi-Fard, Morteza Zadimoghaddam*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/duetting22a.html](https://proceedings.mlr.press/v162/duetting22a.html)

        **Abstract**:

        Maximizing a monotone submodular function is a fundamental task in machine learning. In this paper we study the deletion robust version of the problem under the classic matroids constraint. Here the goal is to extract a small size summary of the dataset that contains a high value independent set even after an adversary deleted some elements. We present constant-factor approximation algorithms, whose space complexity depends on the rank $k$ of the matroid and the number $d$ of deleted elements. In the centralized setting we present a $(3.582+O(\varepsilon))$-approximation algorithm with summary size $O(k + \frac{d}{\eps^2}\log \frac{k}{\eps})$. In the streaming setting we provide a $(5.582+O(\varepsilon))$-approximation algorithm with summary size and memory $O(k + \frac{d}{\eps^2}\log \frac{k}{\eps})$. We complement our theoretical results with an in-depth experimental analysis showing the effectiveness of our algorithms on real-world datasets.

        ----

        ## [244] From data to functa: Your data point is a function and you can treat it like one

        **Authors**: *Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo Jimenez Rezende, Dan Rosenbaum*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dupont22a.html](https://proceedings.mlr.press/v162/dupont22a.html)

        **Abstract**:

        It is common practice in deep learning to represent a measurement of the world on a discrete grid, e.g. a 2D grid of pixels. However, the underlying signal represented by these measurements is often continuous, e.g. the scene depicted in an image. A powerful continuous alternative is then to represent these measurements using an implicit neural representation, a neural function trained to output the appropriate measurement value for any input spatial location. In this paper, we take this idea to its next level: what would it take to perform deep learning on these functions instead, treating them as data? In this context we refer to the data as functa, and propose a framework for deep learning on functa. This view presents a number of challenges around efficient conversion from data to functa, compact representation of functa, and effectively solving downstream tasks on functa. We outline a recipe to overcome these challenges and apply it to a wide range of data modalities including images, 3D shapes, neural radiance fields (NeRF) and data on manifolds. We demonstrate that this approach has various compelling properties across data modalities, in particular on the canonical tasks of generative modeling, data imputation, novel view synthesis and classification.

        ----

        ## [245] Efficient Low Rank Convex Bounds for Pairwise Discrete Graphical Models

        **Authors**: *Valentin Durante, George Katsirelos, Thomas Schiex*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/durante22a.html](https://proceedings.mlr.press/v162/durante22a.html)

        **Abstract**:

        In this paper, we extend a Burer-Monteiro style method to compute low rank Semi-Definite Programming (SDP) bounds for the MAP problem on discrete graphical models with an arbitrary number of states and arbitrary pairwise potentials. We consider both a penalized constraint approach and a dedicated Block Coordinate Descent (BCD) approach which avoids large penalty coefficients in the cost matrix. We show our algorithm is decreasing. Experiments show that the BCD approach compares favorably to the penalized approach and to usual linear bounds relying on convergent message passing approaches.

        ----

        ## [246] Robust Counterfactual Explanations for Tree-Based Ensembles

        **Authors**: *Sanghamitra Dutta, Jason Long, Saumitra Mishra, Cecilia Tilli, Daniele Magazzeni*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dutta22a.html](https://proceedings.mlr.press/v162/dutta22a.html)

        **Abstract**:

        Counterfactual explanations inform ways to achieve a desired outcome from a machine learning model. However, such explanations are not robust to certain real-world changes in the underlying model (e.g., retraining the model, changing hyperparameters, etc.), questioning their reliability in several applications, e.g., credit lending. In this work, we propose a novel strategy - that we call RobX - to generate robust counterfactuals for tree-based ensembles, e.g., XGBoost. Tree-based ensembles pose additional challenges in robust counterfactual generation, e.g., they have a non-smooth and non-differentiable objective function, and they can change a lot in the parameter space under retraining on very similar data. We first introduce a novel metric - that we call Counterfactual Stability - that attempts to quantify how robust a counterfactual is going to be to model changes under retraining, and comes with desirable theoretical properties. Our proposed strategy RobX works with any counterfactual generation method (base method) and searches for robust counterfactuals by iteratively refining the counterfactual generated by the base method using our metric Counterfactual Stability. We compare the performance of RobX with popular counterfactual generation methods (for tree-based ensembles) across benchmark datasets. The results demonstrate that our strategy generates counterfactuals that are significantly more robust (nearly 100% validity after actual model changes) and also realistic (in terms of local outlier factor) over existing state-of-the-art methods.

        ----

        ## [247] On the Difficulty of Defending Self-Supervised Learning against Model Extraction

        **Authors**: *Adam Dziedzic, Nikita Dhawan, Muhammad Ahmad Kaleem, Jonas Guan, Nicolas Papernot*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/dziedzic22a.html](https://proceedings.mlr.press/v162/dziedzic22a.html)

        **Abstract**:

        Self-Supervised Learning (SSL) is an increasingly popular ML paradigm that trains models to transform complex inputs into representations without relying on explicit labels. These representations encode similarity structures that enable efficient learning of multiple downstream tasks. Recently, ML-as-a-Service providers have commenced offering trained SSL models over inference APIs, which transform user inputs into useful representations for a fee. However, the high cost involved to train these models and their exposure over APIs both make black-box extraction a realistic security threat. We thus explore model stealing attacks against SSL. Unlike traditional model extraction on classifiers that output labels, the victim models here output representations; these representations are of significantly higher dimensionality compared to the low-dimensional prediction scores output by classifiers. We construct several novel attacks and find that approaches that train directly on a victim’s stolen representations are query efficient and enable high accuracy for downstream models. We then show that existing defenses against model extraction are inadequate and not easily retrofitted to the specificities of SSL.

        ----

        ## [248] LIMO: Latent Inceptionism for Targeted Molecule Generation

        **Authors**: *Peter Eckmann, Kunyang Sun, Bo Zhao, Mudong Feng, Michael K. Gilson, Rose Yu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/eckmann22a.html](https://proceedings.mlr.press/v162/eckmann22a.html)

        **Abstract**:

        Generation of drug-like molecules with high binding affinity to target proteins remains a difficult and resource-intensive task in drug discovery. Existing approaches primarily employ reinforcement learning, Markov sampling, or deep generative models guided by Gaussian processes, which can be prohibitively slow when generating molecules with high binding affinity calculated by computationally-expensive physics-based methods. We present Latent Inceptionism on Molecules (LIMO), which significantly accelerates molecule generation with an inceptionism-like technique. LIMO employs a variational autoencoder-generated latent space and property prediction by two neural networks in sequence to enable faster gradient-based reverse-optimization of molecular properties. Comprehensive experiments show that LIMO performs competitively on benchmark tasks and markedly outperforms state-of-the-art techniques on the novel task of generating drug-like compounds with high binding affinity, reaching nanomolar range against two protein targets. We corroborate these docking-based results with more accurate molecular dynamics-based calculations of absolute binding free energy and show that one of our generated drug-like compounds has a predicted $K_D$ (a measure of binding affinity) of $6 \cdot 10^{-14}$ M against the human estrogen receptor, well beyond the affinities of typical early-stage drug candidates and most FDA-approved drugs to their respective targets. Code is available at https://github.com/Rose-STL-Lab/LIMO.

        ----

        ## [249] Inductive Biases and Variable Creation in Self-Attention Mechanisms

        **Authors**: *Benjamin L. Edelman, Surbhi Goel, Sham M. Kakade, Cyril Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/edelman22a.html](https://proceedings.mlr.press/v162/edelman22a.html)

        **Abstract**:

        Self-attention, an architectural motif designed to model long-range interactions in sequential data, has driven numerous recent breakthroughs in natural language processing and beyond. This work provides a theoretical analysis of the inductive biases of self-attention modules. Our focus is to rigorously establish which functions and long-range dependencies self-attention blocks prefer to represent. Our main result shows that bounded-norm Transformer networks "create sparse variables": a single self-attention head can represent a sparse function of the input sequence, with sample complexity scaling only logarithmically with the context length. To support our analysis, we present synthetic experiments to probe the sample complexity of learning sparse Boolean functions with Transformers.

        ----

        ## [250] Provable Reinforcement Learning with a Short-Term Memory

        **Authors**: *Yonathan Efroni, Chi Jin, Akshay Krishnamurthy, Sobhan Miryoosefi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/efroni22a.html](https://proceedings.mlr.press/v162/efroni22a.html)

        **Abstract**:

        Real-world sequential decision making problems commonly involve partial observability, which requires the agent to maintain a memory of history in order to infer the latent states, plan and make good decisions. Coping with partial observability in general is extremely challenging, as a number of worst-case statistical and computational barriers are known in learning Partially Observable Markov Decision Processes (POMDPs). Motivated by the problem structure in several physical applications, as well as a commonly used technique known as "frame stacking", this paper proposes to study a new subclass of POMDPs, whose latent states can be decoded by the most recent history of a short length m. We establish a set of upper and lower bounds on the sample complexity for learning near-optimal policies for this class of problems in both tabular and rich-observation settings (where the number of observations is enormous). In particular, in the rich-observation setting, we develop new algorithms using a novel "moment matching" approach with a sample complexity that scales exponentially with the short length m rather than the problem horizon, and is independent of the number of observations. Our results show that a short-term memory suffices for reinforcement learning in these environments.

        ----

        ## [251] Sparsity in Partially Controllable Linear Systems

        **Authors**: *Yonathan Efroni, Sham M. Kakade, Akshay Krishnamurthy, Cyril Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/efroni22b.html](https://proceedings.mlr.press/v162/efroni22b.html)

        **Abstract**:

        A fundamental concept in control theory is that of controllability, where any system state can be reached through an appropriate choice of control inputs. Indeed, a large body of classical and modern approaches are designed for controllable linear dynamical systems. However, in practice, we often encounter systems in which a large set of state variables evolve exogenously and independently of the control inputs; such systems are only partially controllable. The focus of this work is on a large class of partially controllable linear dynamical systems, specified by an underlying sparsity pattern. Our main results establish structural conditions and finite-sample guarantees for learning to control such systems. In particular, our structural results characterize those state variables which are irrelevant for optimal control, an analysis which departs from classical control techniques. Our algorithmic results adapt techniques from high-dimensional statistics{—}specifically soft-thresholding and semiparametric least-squares{—}to exploit the underlying sparsity pattern in order to obtain finite-sample guarantees that significantly improve over those based on certainty-equivalence. We also corroborate these theoretical improvements over certainty-equivalent control through a simulation study.

        ----

        ## [252] FedNew: A Communication-Efficient and Privacy-Preserving Newton-Type Method for Federated Learning

        **Authors**: *Anis Elgabli, Chaouki Ben Issaid, Amrit Singh Bedi, Ketan Rajawat, Mehdi Bennis, Vaneet Aggarwal*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/elgabli22a.html](https://proceedings.mlr.press/v162/elgabli22a.html)

        **Abstract**:

        Newton-type methods are popular in federated learning due to their fast convergence. Still, they suffer from two main issues, namely: low communication efficiency and low privacy due to the requirement of sending Hessian information from clients to parameter server (PS). In this work, we introduced a novel framework called FedNew in which there is no need to transmit Hessian information from clients to PS, hence resolving the bottleneck to improve communication efficiency. In addition, FedNew hides the gradient information and results in a privacy-preserving approach compared to the existing state-of-the-art. The core novel idea in FedNew is to introduce a two level framework, and alternate between updating the inverse Hessian-gradient product using only one alternating direction method of multipliers (ADMM) step and then performing the global model update using Newton’s method. Though only one ADMM pass is used to approximate the inverse Hessian-gradient product at each iteration, we develop a novel theoretical approach to show the converging behavior of FedNew for convex problems. Additionally, a significant reduction in communication overhead is achieved by utilizing stochastic quantization. Numerical results using real datasets show the superiority of FedNew compared to existing methods in terms of communication costs.

        ----

        ## [253] pathGCN: Learning General Graph Spatial Operators from Paths

        **Authors**: *Moshe Eliasof, Eldad Haber, Eran Treister*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/eliasof22a.html](https://proceedings.mlr.press/v162/eliasof22a.html)

        **Abstract**:

        Graph Convolutional Networks (GCNs), similarly to Convolutional Neural Networks (CNNs), are typically based on two main operations - spatial and point-wise convolutions. In the context of GCNs, differently from CNNs, a pre-determined spatial operator based on the graph Laplacian is often chosen, allowing only the point-wise operations to be learnt. However, learning a meaningful spatial operator is critical for developing more expressive GCNs for improved performance. In this paper we propose pathGCN, a novel approach to learn the spatial operator from random paths on the graph. We analyze the convergence of our method and its difference from existing GCNs. Furthermore, we discuss several options of combining our learnt spatial operator with point-wise convolutions. Our extensive experiments on numerous datasets suggest that by properly learning both the spatial and point-wise convolutions, phenomena like over-smoothing can be inherently avoided, and new state-of-the-art performance is achieved.

        ----

        ## [254] Discrete Tree Flows via Tree-Structured Permutations

        **Authors**: *Mai Elkady, Hyung Zin Lim, David I. Inouye*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/elkady22a.html](https://proceedings.mlr.press/v162/elkady22a.html)

        **Abstract**:

        While normalizing flows for continuous data have been extensively researched, flows for discrete data have only recently been explored. These prior models, however, suffer from limitations that are distinct from those of continuous flows. Most notably, discrete flow-based models cannot be straightforwardly optimized with conventional deep learning methods because gradients of discrete functions are undefined or zero. Previous works approximate pseudo-gradients of the discrete functions but do not solve the problem on a fundamental level. In addition to that, backpropagation can be computationally burdensome compared to alternative discrete algorithms such as decision tree algorithms. Our approach seeks to reduce computational burden and remove the need for pseudo-gradients by developing a discrete flow based on decision trees—building upon the success of efficient tree-based methods for classification and regression for discrete data. We first define a tree-structured permutation (TSP) that compactly encodes a permutation of discrete data where the inverse is easy to compute; thus, we can efficiently compute the density value and sample new data. We then propose a decision tree algorithm to build TSPs that learns the tree structure and permutations at each node via novel criteria. We empirically demonstrate the feasibility of our method on multiple datasets.

        ----

        ## [255] For Learning in Symmetric Teams, Local Optima are Global Nash Equilibria

        **Authors**: *Scott Emmons, Caspar Oesterheld, Andrew Critch, Vincent Conitzer, Stuart Russell*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/emmons22a.html](https://proceedings.mlr.press/v162/emmons22a.html)

        **Abstract**:

        Although it has been known since the 1970s that a globally optimal strategy profile in a common-payoff game is a Nash equilibrium, global optimality is a strict requirement that limits the result’s applicability. In this work, we show that any locally optimal symmetric strategy profile is also a (global) Nash equilibrium. Furthermore, we show that this result is robust to perturbations to the common payoff and to the local optimum. Applied to machine learning, our result provides a global guarantee for any gradient method that finds a local optimum in symmetric strategy space. While this result indicates stability to unilateral deviation, we nevertheless identify broad classes of games where mixed local optima are unstable under joint, asymmetric deviations. We analyze the prevalence of instability by running learning algorithms in a suite of symmetric games, and we conclude by discussing the applicability of our results to multi-agent RL, cooperative inverse RL, and decentralized POMDPs.

        ----

        ## [256] Streaming Algorithm for Monotone k-Submodular Maximization with Cardinality Constraints

        **Authors**: *Alina Ene, Huy L. Nguyen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ene22a.html](https://proceedings.mlr.press/v162/ene22a.html)

        **Abstract**:

        Maximizing a monotone k-submodular function subject to cardinality constraints is a general model for several applications ranging from influence maximization with multiple products to sensor placement with multiple sensor types and online ad allocation. Due to the large problem scale in many applications and the online nature of ad allocation, a need arises for algorithms that process elements in a streaming fashion and possibly make online decisions. In this work, we develop a new streaming algorithm for maximizing a monotone k-submodular function subject to a per-coordinate cardinality constraint attaining an approximation guarantee close to the state of the art guarantee in the offline setting. Though not typical for streaming algorithms, our streaming algorithm also readily applies to the online setting with free disposal. Our algorithm is combinatorial and enjoys fast running time and small number of function evaluations. Furthermore, its guarantee improves as the cardinality constraints get larger, which is especially suited for the large scale applications. For the special case of maximizing a submodular function with large budgets, our combinatorial algorithm matches the guarantee of the state-of-the-art continuous algorithm, which requires significantly more time and function evaluations.

        ----

        ## [257] Towards Scaling Difference Target Propagation by Learning Backprop Targets

        **Authors**: *Maxence Ernoult, Fabrice Normandin, Abhinav Moudgil, Sean Spinney, Eugene Belilovsky, Irina Rish, Blake A. Richards, Yoshua Bengio*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ernoult22a.html](https://proceedings.mlr.press/v162/ernoult22a.html)

        **Abstract**:

        The development of biologically-plausible learning algorithms is important for understanding learning in the brain, but most of them fail to scale-up to real-world tasks, limiting their potential as explanations for learning by real brains. As such, it is important to explore learning algorithms that come with strong theoretical guarantees and can match the performance of backpropagation (BP) on complex tasks. One such algorithm is Difference Target Propagation (DTP), a biologically-plausible learning algorithm whose close relation with Gauss-Newton (GN) optimization has been recently established. However, the conditions under which this connection rigorously holds preclude layer-wise training of the feedback pathway synaptic weights (which is more biologically plausible). Moreover, good alignment between DTP weight updates and loss gradients is only loosely guaranteed and under very specific conditions for the architecture being trained. In this paper, we propose a novel feedback weight training scheme that ensures both that DTP approximates BP and that layer-wise feedback weight training can be restored without sacrificing any theoretical guarantees. Our theory is corroborated by experimental results and we report the best performance ever achieved by DTP on CIFAR-10 and ImageNet 32x32.

        ----

        ## [258] Understanding Dataset Difficulty with V-Usable Information

        **Authors**: *Kawin Ethayarajh, Yejin Choi, Swabha Swayamdipta*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ethayarajh22a.html](https://proceedings.mlr.press/v162/ethayarajh22a.html)

        **Abstract**:

        Estimating the difficulty of a dataset typically involves comparing state-of-the-art models to humans; the bigger the performance gap, the harder the dataset is said to be. However, this comparison provides little understanding of how difficult each instance in a given distribution is, or what attributes make the dataset difficult for a given model. To address these questions, we frame dataset difficulty—w.r.t. a model $\mathcal{V}$—as the lack of $\mathcal{V}$-usable information (Xu et al., 2019), where a lower value indicates a more difficult dataset for $\mathcal{V}$. We further introduce pointwise $\mathcal{V}$-information (PVI) for measuring the difficulty of individual instances w.r.t. a given distribution. While standard evaluation metrics typically only compare different models for the same dataset, $\mathcal{V}$-usable information and PVI also permit the converse: for a given model $\mathcal{V}$, we can compare different datasets, as well as different instances/slices of the same dataset. Furthermore, our framework allows for the interpretability of different input attributes via transformations of the input, which we use to discover annotation artefacts in widely-used NLP benchmarks.

        ----

        ## [259] Head2Toe: Utilizing Intermediate Representations for Better Transfer Learning

        **Authors**: *Utku Evci, Vincent Dumoulin, Hugo Larochelle, Michael C. Mozer*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/evci22a.html](https://proceedings.mlr.press/v162/evci22a.html)

        **Abstract**:

        Transfer-learning methods aim to improve performance in a data-scarce target domain using a model pretrained on a data-rich source domain. A cost-efficient strategy, linear probing, involves freezing the source model and training a new classification head for the target domain. This strategy is outperformed by a more costly but state-of-the-art method – fine-tuning all parameters of the source model to the target domain – possibly because fine-tuning allows the model to leverage useful information from intermediate layers which is otherwise discarded by the later previously trained layers. We explore the hypothesis that these intermediate layers might be directly exploited. We propose a method, Head-to-Toe probing (Head2Toe), that selects features from all layers of the source model to train a classification head for the target-domain. In evaluations on the Visual Task Adaptation Benchmark-1k, Head2Toe matches performance obtained with fine-tuning on average while reducing training and storage cost hundred folds or more, but critically, for out-of-distribution transfer, Head2Toe outperforms fine-tuning. Code used in our experiments can be found in supplementary materials.

        ----

        ## [260] Variational Sparse Coding with Learned Thresholding

        **Authors**: *Kion Fallah, Christopher J. Rozell*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fallah22a.html](https://proceedings.mlr.press/v162/fallah22a.html)

        **Abstract**:

        Sparse coding strategies have been lauded for their parsimonious representations of data that leverage low dimensional structure. However, inference of these codes typically relies on an optimization procedure with poor computational scaling in high-dimensional problems. For example, sparse inference in the representations learned in the high-dimensional intermediary layers of deep neural networks (DNNs) requires an iterative minimization to be performed at each training step. As such, recent, quick methods in variational inference have been proposed to infer sparse codes by learning a distribution over the codes with a DNN. In this work, we propose a new approach to variational sparse coding that allows us to learn sparse distributions by thresholding samples, avoiding the use of problematic relaxations. We first evaluate and analyze our method by training a linear generator, showing that it has superior performance, statistical efficiency, and gradient estimation compared to other sparse distributions. We then compare to a standard variational autoencoder using a DNN generator on the CelebA dataset.

        ----

        ## [261] Training Discrete Deep Generative Models via Gapped Straight-Through Estimator

        **Authors**: *Ting-Han Fan, Ta-Chung Chi, Alexander I. Rudnicky, Peter J. Ramadge*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fan22a.html](https://proceedings.mlr.press/v162/fan22a.html)

        **Abstract**:

        While deep generative models have succeeded in image processing, natural language processing, and reinforcement learning, training that involves discrete random variables remains challenging due to the high variance of its gradient estimation process. Monte Carlo is a common solution used in most variance reduction approaches. However, this involves time-consuming resampling and multiple function evaluations. We propose a Gapped Straight-Through (GST) estimator to reduce the variance without incurring resampling overhead. This estimator is inspired by the essential properties of Straight-Through Gumbel-Softmax. We determine these properties and show via an ablation study that they are essential. Experiments demonstrate that the proposed GST estimator enjoys better performance compared to strong baselines on two discrete deep generative modeling tasks, MNIST-VAE and ListOps.

        ----

        ## [262] DRIBO: Robust Deep Reinforcement Learning via Multi-View Information Bottleneck

        **Authors**: *Jiameng Fan, Wenchao Li*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fan22b.html](https://proceedings.mlr.press/v162/fan22b.html)

        **Abstract**:

        Deep reinforcement learning (DRL) agents are often sensitive to visual changes that were unseen in their training environments. To address this problem, we leverage the sequential nature of RL to learn robust representations that encode only task-relevant information from observations based on the unsupervised multi-view setting. Specif- ically, we introduce a novel contrastive version of the Multi-View Information Bottleneck (MIB) objective for temporal data. We train RL agents from pixels with this auxiliary objective to learn robust representations that can compress away task-irrelevant information and are predictive of task-relevant dynamics. This approach enables us to train high-performance policies that are robust to visual distractions and can generalize well to unseen environments. We demonstrate that our approach can achieve SOTA performance on a di- verse set of visual control tasks in the DeepMind Control Suite when the background is replaced with natural videos. In addition, we show that our approach outperforms well-established base- lines for generalization to unseen environments on the Procgen benchmark. Our code is open- sourced and available at https://github. com/BU-DEPEND-Lab/DRIBO.

        ----

        ## [263] Generalized Data Distribution Iteration

        **Authors**: *Jiajun Fan, Changnan Xiao*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fan22c.html](https://proceedings.mlr.press/v162/fan22c.html)

        **Abstract**:

        To obtain higher sample efficiency and superior final performance simultaneously has been one of the major challenges for deep reinforcement learning (DRL). Previous work could handle one of these challenges but typically failed to address them concurrently. In this paper, we try to tackle these two challenges simultaneously. To achieve this, we firstly decouple these challenges into two classic RL problems: data richness and exploration-exploitation trade-off. Then, we cast these two problems into the training data distribution optimization problem, namely to obtain desired training data within limited interactions, and address them concurrently via i) explicit modeling and control of the capacity and diversity of behavior policy and ii) more fine-grained and adaptive control of selective/sampling distribution of the behavior policy using a monotonic data distribution optimization. Finally, we integrate this process into Generalized Policy Iteration (GPI) and obtain a more general framework called Generalized Data Distribution Iteration (GDI). We use the GDI framework to introduce operator-based versions of well-known RL methods from DQN to Agent57. Theoretical guarantee of the superiority of GDI compared with GPI is concluded. We also demonstrate our state-of-the-art (SOTA) performance on Arcade Learning Environment (ALE), wherein our algorithm has achieved 9620.98% mean human normalized score (HNS), 1146.39% median HNS, and surpassed 22 human world records using only 200M training frames. Our performance is comparable to Agent57’s while we consume 500 times less data. We argue that there is still a long way to go before obtaining real superhuman agents in ALE.

        ----

        ## [264] Variational Wasserstein gradient flow

        **Authors**: *Jiaojiao Fan, Qinsheng Zhang, Amirhossein Taghvaei, Yongxin Chen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fan22d.html](https://proceedings.mlr.press/v162/fan22d.html)

        **Abstract**:

        Wasserstein gradient flow has emerged as a promising approach to solve optimization problems over the space of probability distributions. A recent trend is to use the well-known JKO scheme in combination with input convex neural networks to numerically implement the proximal step. The most challenging step, in this setup, is to evaluate functions involving density explicitly, such as entropy, in terms of samples. This paper builds on the recent works with a slight but crucial difference: we propose to utilize a variational formulation of the objective function formulated as maximization over a parametric class of functions. Theoretically, the proposed variational formulation allows the construction of gradient flows directly for empirical distributions with a well-defined and meaningful objective function. Computationally, this approach replaces the computationally expensive step in existing methods, to handle objective functions involving density, with inner loop updates that only require a small batch of samples and scale well with the dimension. The performance and scalability of the proposed method are illustrated with the aid of several numerical experiments involving high-dimensional synthetic and real datasets.

        ----

        ## [265] Data Determines Distributional Robustness in Contrastive Language Image Pre-training (CLIP)

        **Authors**: *Alex Fang, Gabriel Ilharco, Mitchell Wortsman, Yuhao Wan, Vaishaal Shankar, Achal Dave, Ludwig Schmidt*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fang22a.html](https://proceedings.mlr.press/v162/fang22a.html)

        **Abstract**:

        Contrastively trained language-image models such as CLIP, ALIGN, and BASIC have demonstrated unprecedented robustness to multiple challenging natural distribution shifts. Since these language-image models differ from previous training approaches in several ways, an important question is what causes the large robustness gains. We answer this question via a systematic experimental investigation. Concretely, we study five different possible causes for the robustness gains: (i) the training set size, (ii) the training distribution, (iii) language supervision at training time, (iv) language supervision at test time, and (v) the contrastive loss function. Our experiments show that the more diverse training distribution is the main cause for the robustness gains, with the other factors contributing little to no robustness. Beyond our experimental results, we also introduce ImageNet-Captions, a version of ImageNet with original text annotations from Flickr, to enable further controlled experiments of language-image training.

        ----

        ## [266] Bayesian Continuous-Time Tucker Decomposition

        **Authors**: *Shikai Fang, Akil Narayan, Robert M. Kirby, Shandian Zhe*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fang22b.html](https://proceedings.mlr.press/v162/fang22b.html)

        **Abstract**:

        Tensor decomposition is a dominant framework for multiway data analysis and prediction. Although practical data often contains timestamps for the observed entries, existing tensor decomposition approaches overlook or under-use this valuable time information. They either drop the timestamps or bin them into crude steps and hence ignore the temporal dynamics within each step or use simple parametric time coefficients. To overcome these limitations, we propose Bayesian Continuous-Time Tucker Decomposition. We model the tensor-core of the classical Tucker decomposition as a time-varying function, and place a Gaussian process prior to flexibly estimate all kinds of temporal dynamics. In this way, our model maintains the interpretability while is flexible enough to capture various complex temporal relationships between the tensor nodes. For efficient and high-quality posterior inference, we use the stochastic differential equation (SDE) representation of temporal GPs to build an equivalent state-space prior, which avoids huge kernel matrix computation and sparse/low-rank approximations. We then use Kalman filtering, RTS smoothing, and conditional moment matching to develop a scalable message passing inference algorithm. We show the advantage of our method in simulation and several real-world applications.

        ----

        ## [267] Byzantine Machine Learning Made Easy By Resilient Averaging of Momentums

        **Authors**: *Sadegh Farhadkhani, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, John Stephan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/farhadkhani22a.html](https://proceedings.mlr.press/v162/farhadkhani22a.html)

        **Abstract**:

        Byzantine resilience emerged as a prominent topic within the distributed machine learning community. Essentially, the goal is to enhance distributed optimization algorithms, such as distributed SGD, in a way that guarantees convergence despite the presence of some misbehaving (a.k.a., Byzantine) workers. Although a myriad of techniques addressing the problem have been proposed, the field arguably rests on fragile foundations. These techniques are hard to prove correct and rely on assumptions that are (a) quite unrealistic, i.e., often violated in practice, and (b) heterogeneous, i.e., making it difficult to compare approaches. We present RESAM (RESilient Averaging of Momentums), a unified framework that makes it simple to establish optimal Byzantine resilience, relying only on standard machine learning assumptions. Our framework is mainly composed of two operators: resilient averaging at the server and distributed momentum at the workers. We prove a general theorem stating the convergence of distributed SGD under RESAM. Interestingly, demonstrating and comparing the convergence of many existing techniques become direct corollaries of our theorem, without resorting to stringent assumptions. We also present an empirical evaluation of the practical relevance of RESAM.

        ----

        ## [268] An Equivalence Between Data Poisoning and Byzantine Gradient Attacks

        **Authors**: *Sadegh Farhadkhani, Rachid Guerraoui, Lê Nguyên Hoang, Oscar Villemaud*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/farhadkhani22b.html](https://proceedings.mlr.press/v162/farhadkhani22b.html)

        **Abstract**:

        To study the resilience of distributed learning, the “Byzantine" literature considers a strong threat model where workers can report arbitrary gradients to the parameter server. Whereas this model helped obtain several fundamental results, it has sometimes been considered unrealistic, when the workers are mostly trustworthy machines. In this paper, we show a surprising equivalence between this model and data poisoning, a threat considered much more realistic. More specifically, we prove that every gradient attack can be reduced to data poisoning, in any personalized federated learning system with PAC guarantees (which we show are both desirable and realistic). This equivalence makes it possible to obtain new impossibility results on the resilience of any “robust” learning algorithm to data poisoning in highly heterogeneous applications, as corollaries of existing impossibility theorems on Byzantine machine learning. Moreover, using our equivalence, we derive a practical attack that we show (theoretically and empirically) can be very effective against classical personalized federated learning models.

        ----

        ## [269] Investigating Generalization by Controlling Normalized Margin

        **Authors**: *Alexander R. Farhang, Jeremy D. Bernstein, Kushal Tirumala, Yang Liu, Yisong Yue*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/farhang22a.html](https://proceedings.mlr.press/v162/farhang22a.html)

        **Abstract**:

        Weight norm $\|w\|$ and margin $\gamma$ participate in learning theory via the normalized margin $\gamma/\|w\|$. Since standard neural net optimizers do not control normalized margin, it is hard to test whether this quantity causally relates to generalization. This paper designs a series of experimental studies that explicitly control normalized margin and thereby tackle two central questions. First: does normalized margin always have a causal effect on generalization? The paper finds that no—networks can be produced where normalized margin has seemingly no relationship with generalization, counter to the theory of Bartlett et al. (2017). Second: does normalized margin ever have a causal effect on generalization? The paper finds that yes—in a standard training setup, test performance closely tracks normalized margin. The paper suggests a Gaussian process model as a promising explanation for this behavior.

        ----

        ## [270] Kernelized Multiplicative Weights for 0/1-Polyhedral Games: Bridging the Gap Between Learning in Extensive-Form and Normal-Form Games

        **Authors**: *Gabriele Farina, Chung-Wei Lee, Haipeng Luo, Christian Kroer*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/farina22a.html](https://proceedings.mlr.press/v162/farina22a.html)

        **Abstract**:

        While extensive-form games (EFGs) can be converted into normal-form games (NFGs), doing so comes at the cost of an exponential blowup of the strategy space. So, progress on NFGs and EFGs has historically followed separate tracks, with the EFG community often having to catch up with advances (\eg last-iterate convergence and predictive regret bounds) from the larger NFG community. In this paper we show that the Optimistic Multiplicative Weights Update (OMWU) algorithm—the premier learning algorithm for NFGs—can be simulated on the normal-form equivalent of an EFG in linear time per iteration in the game tree size using a kernel trick. The resulting algorithm, Kernelized OMWU (KOMWU), applies more broadly to all convex games whose strategy space is a polytope with 0/1 integral vertices, as long as the kernel can be evaluated efficiently. In the particular case of EFGs, KOMWU closes several standing gaps between NFG and EFG learning, by enabling direct, black-box transfer to EFGs of desirable properties of learning dynamics that were so far known to be achievable only in NFGs. Specifically, KOMWU gives the first algorithm that guarantees at the same time last-iterate convergence, lower dependence on the size of the game tree than all prior algorithms, and $\tilde{\bigOh}(1)$ regret when followed by all players.

        ----

        ## [271] Local Linear Convergence of Douglas-Rachford for Linear Programming: a Probabilistic Analysis

        **Authors**: *Oisin Faust, Hamza Fawzi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/faust22a.html](https://proceedings.mlr.press/v162/faust22a.html)

        **Abstract**:

        Douglas-Rachford splitting/ADMM (henceforth DRS) is a very popular algorithm for solving convex optimisation problems to low or moderate accuracy, and in particular for solving large-scale linear programs. Despite recent progress, obtaining highly accurate solutions to linear programs with DRS remains elusive. In this paper we analyze the local linear convergence rate $r$ of the DRS method for random linear programs, and give explicit and tight bounds on $r$. We show that $1-r^2$ is typically of the order of $m^{-1}(n-m)^{-1}$, where $n$ is the number of variables and $m$ is the number of constraints. This provides a quantitative explanation for the very slow convergence of DRS/ADMM on random LPs. The proof of our result relies on an established characterisation of the linear rate of convergence as the cosine of the Friedrichs angle between two subspaces associated to the problem. We also show that the cosecant of this angle can be interpreted as a condition number for the LP. The proof of our result relies on a characterization of the linear rate of convergence as the cosine of the Friedrichs angle between two subspaces associated to the problem. We also show that the cosecant of this angle can be interpreted as a condition number for the LP.

        ----

        ## [272] Matching Structure for Dual Learning

        **Authors**: *Hao Fei, Shengqiong Wu, Yafeng Ren, Meishan Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fei22a.html](https://proceedings.mlr.press/v162/fei22a.html)

        **Abstract**:

        Many natural language processing (NLP) tasks appear in dual forms, which are generally solved by dual learning technique that models the dualities between the coupled tasks. In this work, we propose to further enhance dual learning with structure matching that explicitly builds structural connections in between. Starting with the dual text$\leftrightarrow$text generation, we perform dually-syntactic structure co-echoing of the region of interest (RoI) between the task pair, together with a syntax cross-reconstruction at the decoding side. We next extend the idea to a text$\leftrightarrow$non-text setup, making alignment between the syntactic-semantic structure. Over 2*14 tasks covering 5 dual learning scenarios, the proposed structure matching method shows its significant effectiveness in enhancing existing dual learning. Our method can retrieve the key RoIs that are highly crucial to the task performance. Besides NLP tasks, it is also revealed that our approach has great potential in facilitating more non-text$\leftrightarrow$non-text scenarios.

        ----

        ## [273] Cascaded Gaps: Towards Logarithmic Regret for Risk-Sensitive Reinforcement Learning

        **Authors**: *Yingjie Fei, Ruitu Xu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fei22b.html](https://proceedings.mlr.press/v162/fei22b.html)

        **Abstract**:

        In this paper, we study gap-dependent regret guarantees for risk-sensitive reinforcement learning based on the entropic risk measure. We propose a novel definition of sub-optimality gaps, which we call cascaded gaps, and we discuss their key components that adapt to underlying structures of the problem. Based on the cascaded gaps, we derive non-asymptotic and logarithmic regret bounds for two model-free algorithms under episodic Markov decision processes. We show that, in appropriate settings, these bounds feature exponential improvement over existing ones that are independent of gaps. We also prove gap-dependent lower bounds, which certify the near optimality of the upper bounds.

        ----

        ## [274] Private frequency estimation via projective geometry

        **Authors**: *Vitaly Feldman, Jelani Nelson, Huy L. Nguyen, Kunal Talwar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/feldman22a.html](https://proceedings.mlr.press/v162/feldman22a.html)

        **Abstract**:

        In this work, we propose a new algorithm ProjectiveGeometryResponse (PGR) for locally differentially private (LDP) frequency estimation. For universe size of k and with n users, our eps-LDP algorithm has communication cost ceil(log_2 k) and computation cost O(n + k\exp(eps) log k) for the server to approximately reconstruct the frequency histogram, while achieve optimal privacy-utility tradeoff. In many practical settings this is a significant improvement over the O (n+k^2) computation cost that is achieved by the recent PI-RAPPOR algorithm (Feldman and Talwar; 2021). Our empirical evaluation shows a speedup of over 50x over PI-RAPPOR while using approximately 75x less memory. In addition, the running time of our algorithm is comparable to that of HadamardResponse (Acharya, Sun, and Zhang; 2019) and RecursiveHadamardResponse (Chen, Kairouz, and Ozgur; 2020) which have significantly worse reconstruction error. The error of our algorithm essentially matches that of the communication- and time-inefficient but utility-optimal SubsetSelection (SS) algorithm (Ye and Barg; 2017). Our new algorithm is based on using Projective Planes over a finite field to define a small collection of sets that are close to being pairwise independent and a dynamic programming algorithm for approximate histogram reconstruction for the server.

        ----

        ## [275] An Intriguing Property of Geophysics Inversion

        **Authors**: *Yinan Feng, Yinpeng Chen, Shihang Feng, Peng Jin, Zicheng Liu, Youzuo Lin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/feng22a.html](https://proceedings.mlr.press/v162/feng22a.html)

        **Abstract**:

        Inversion techniques are widely used to reconstruct subsurface physical properties (e.g., velocity, conductivity) from surface-based geophysical measurements (e.g., seismic, electric/magnetic (EM) data). The problems are governed by partial differential equations (PDEs) like the wave or Maxwell’s equations. Solving geophysical inversion problems is challenging due to the ill-posedness and high computational cost. To alleviate those issues, recent studies leverage deep neural networks to learn the inversion mappings from measurements to the property directly. In this paper, we show that such a mapping can be well modeled by a very shallow (but not wide) network with only five layers. This is achieved based on our new finding of an intriguing property: a near-linear relationship between the input and output, after applying integral transform in high dimensional space. In particular, when dealing with the inversion from seismic data to subsurface velocity governed by a wave equation, the integral results of velocity with Gaussian kernels are linearly correlated to the integral of seismic data with sine kernels. Furthermore, this property can be easily turned into a light-weight encoder-decoder network for inversion. The encoder contains the integration of seismic data and the linear transformation without need for fine-tuning. The decoder only consists of a single transformer block to reverse the integral of velocity. Experiments show that this interesting property holds for two geophysics inversion problems over four different datasets. Compared to much deeper InversionNet, our method achieves comparable accuracy, but consumes significantly fewer parameters

        ----

        ## [276] Principled Knowledge Extrapolation with GANs

        **Authors**: *Ruili Feng, Jie Xiao, Kecheng Zheng, Deli Zhao, Jingren Zhou, Qibin Sun, Zheng-Jun Zha*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/feng22b.html](https://proceedings.mlr.press/v162/feng22b.html)

        **Abstract**:

        Human can extrapolate well, generalize daily knowledge into unseen scenarios, raise and answer counterfactual questions. To imitate this ability via generative models, previous works have extensively studied explicitly encoding Structural Causal Models (SCMs) into architectures of generator networks. This methodology, however, limits the flexibility of the generator as they must be carefully crafted to follow the causal graph, and demands a ground truth SCM with strong ignorability assumption as prior, which is a nontrivial assumption in many real scenarios. Thus, many current causal GAN methods fail to generate high fidelity counterfactual results as they cannot easily leverage state-of-the-art generative models. In this paper, we propose to study counterfactual synthesis from a new perspective of knowledge extrapolation, where a given knowledge dimension of the data distribution is extrapolated, but the remaining knowledge is kept indistinguishable from the original distribution. We show that an adversarial game with a closed-form discriminator can be used to address the knowledge extrapolation problem, and a novel principal knowledge descent method can efficiently estimate the extrapolated distribution through the adversarial game. Our method enjoys both elegant theoretical guarantees and superior performance in many scenarios.

        ----

        ## [277] A Resilient Distributed Boosting Algorithm

        **Authors**: *Yuval Filmus, Idan Mehalel, Shay Moran*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/filmus22a.html](https://proceedings.mlr.press/v162/filmus22a.html)

        **Abstract**:

        Given a learning task where the data is distributed among several parties, communication is one of the fundamental resources which the parties would like to minimize. We present a distributed boosting algorithm which is resilient to a limited amount of noise. Our algorithm is similar to classical boosting algorithms, although it is equipped with a new component, inspired by Impagliazzo’s hard-core lemma (Impagliazzo, 1995), adding a robustness quality to the algorithm. We also complement this result by showing that resilience to any asymptotically larger noise is not achievable by a communication-efficient algorithm.

        ----

        ## [278] Model-Value Inconsistency as a Signal for Epistemic Uncertainty

        **Authors**: *Angelos Filos, Eszter Vértes, Zita Marinho, Gregory Farquhar, Diana Borsa, Abram L. Friesen, Feryal M. P. Behbahani, Tom Schaul, André Barreto, Simon Osindero*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/filos22a.html](https://proceedings.mlr.press/v162/filos22a.html)

        **Abstract**:

        Using a model of the environment and a value function, an agent can construct many estimates of a state’s value, by unrolling the model for different lengths and bootstrapping with its value function. Our key insight is that one can treat this set of value estimates as a type of ensemble, which we call an implicit value ensemble (IVE). Consequently, the discrepancy between these estimates can be used as a proxy for the agent’s epistemic uncertainty; we term this signal model-value inconsistency or self-inconsistency for short. Unlike prior work which estimates uncertainty by training an ensemble of many models and/or value functions, this approach requires only the single model and value function which are already being learned in most model-based reinforcement learning algorithms. We provide empirical evidence in both tabular and function approximation settings from pixels that self-inconsistency is useful (i) as a signal for exploration, (ii) for acting safely under distribution shifts, and (iii) for robustifying value-based planning with a learned model.

        ----

        ## [279] Coordinated Double Machine Learning

        **Authors**: *Nitai Fingerhut, Matteo Sesia, Yaniv Romano*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fingerhut22a.html](https://proceedings.mlr.press/v162/fingerhut22a.html)

        **Abstract**:

        Double machine learning is a statistical method for leveraging complex black-box models to construct approximately unbiased treatment effect estimates given observational data with high-dimensional covariates, under the assumption of a partially linear model. The idea is to first fit on a subset of the samples two non-linear predictive models, one for the continuous outcome of interest and one for the observed treatment, and then to estimate a linear coefficient for the treatment using the remaining samples through a simple orthogonalized regression. While this methodology is flexible and can accommodate arbitrary predictive models, typically trained independently of one another, this paper argues that a carefully coordinated learning algorithm for deep neural networks may reduce the estimation bias. The improved empirical performance of the proposed method is demonstrated through numerical experiments on both simulated and real data.

        ----

        ## [280] Conformal Prediction Sets with Limited False Positives

        **Authors**: *Adam Fisch, Tal Schuster, Tommi S. Jaakkola, Regina Barzilay*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fisch22a.html](https://proceedings.mlr.press/v162/fisch22a.html)

        **Abstract**:

        We develop a new approach to multi-label conformal prediction in which we aim to output a precise set of promising prediction candidates with a bounded number of incorrect answers. Standard conformal prediction provides the ability to adapt to model uncertainty by constructing a calibrated candidate set in place of a single prediction, with guarantees that the set contains the correct answer with high probability. In order to obey this coverage property, however, conformal sets can become inundated with noisy candidates—which can render them unhelpful in practice. This is particularly relevant to practical applications where there is a limited budget, and the cost (monetary or otherwise) associated with false positives is non-negligible. We propose to trade coverage for a notion of precision by enforcing that the presence of incorrect candidates in the predicted conformal sets (i.e., the total number of false positives) is bounded according to a user-specified tolerance. Subject to this constraint, our algorithm then optimizes for a generalized notion of set coverage (i.e., the true positive rate) that allows for any number of true answers for a given query (including zero). We demonstrate the effectiveness of this approach across a number of classification tasks in natural language processing, computer vision, and computational chemistry.

        ----

        ## [281] Fast Population-Based Reinforcement Learning on a Single Machine

        **Authors**: *Arthur Flajolet, Claire Bizon Monroc, Karim Beguir, Thomas Pierrot*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/flajolet22a.html](https://proceedings.mlr.press/v162/flajolet22a.html)

        **Abstract**:

        Training populations of agents has demonstrated great promise in Reinforcement Learning for stabilizing training, improving exploration and asymptotic performance, and generating a diverse set of solutions. However, population-based training is often not considered by practitioners as it is perceived to be either prohibitively slow (when implemented sequentially), or computationally expensive (if agents are trained in parallel on independent accelerators). In this work, we compare implementations and revisit previous studies to show that the judicious use of compilation and vectorization allows population-based training to be performed on a single machine with one accelerator with minimal overhead compared to training a single agent. We also show that, when provided with a few accelerators, our protocols extend to large population sizes for applications such as hyperparameter tuning. We hope that this work and the public release of our code will encourage practitioners to use population-based learning techniques more frequently for their research and applications.

        ----

        ## [282] Fast Relative Entropy Coding with A* coding

        **Authors**: *Gergely Flamich, Stratis Markou, José Miguel Hernández-Lobato*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/flamich22a.html](https://proceedings.mlr.press/v162/flamich22a.html)

        **Abstract**:

        Relative entropy coding (REC) algorithms encode a sample from a target distribution Q using a proposal distribution P, such that the expected codelength is O(KL[Q || P]). REC can be seamlessly integrated with existing learned compression models since, unlike entropy coding, it does not assume discrete Q or P, and does not require quantisation. However, general REC algorithms require an intractable $\Omega$(exp(KL[Q || P])) runtime. We introduce AS* and AD* coding, two REC algorithms based on A* sampling. We prove that, for continuous distributions over the reals, if the density ratio is unimodal, AS* has O(D$\infty$[Q || P]) expected runtime, where D$\infty$[Q || P] is the Renyi $\infty$-divergence. We provide experimental evidence that AD* also has O(D$\infty$[Q || P]) expected runtime. We prove that AS* and AD* achieve an expected codelength of O(KL[Q || P]). Further, we introduce DAD*, an approximate algorithm based on AD* which retains its favourable runtime and has bias similar to that of alternative methods. Focusing on VAEs, we propose the IsoKL VAE (IKVAE), which can be used with DAD* to further improve compression efficiency. We evaluate A* coding with (IK)VAEs on MNIST, showing that it can losslessly compress images near the theoretically optimal limit.

        ----

        ## [283] Contrastive Mixture of Posteriors for Counterfactual Inference, Data Integration and Fairness

        **Authors**: *Adam Foster, Árpi Vezér, Craig A. Glastonbury, Páidí Creed, Samer Abujudeh, Aaron Sim*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/foster22a.html](https://proceedings.mlr.press/v162/foster22a.html)

        **Abstract**:

        Learning meaningful representations of data that can address challenges such as batch effect correction and counterfactual inference is a central problem in many domains including computational biology. Adopting a Conditional VAE framework, we show that marginal independence between the representation and a condition variable plays a key role in both of these challenges. We propose the Contrastive Mixture of Posteriors (CoMP) method that uses a novel misalignment penalty defined in terms of mixtures of the variational posteriors to enforce this independence in latent space. We show that CoMP has attractive theoretical properties compared to previous approaches, and we prove counterfactual identifiability of CoMP under additional assumptions. We demonstrate state-of-the-art performance on a set of challenging tasks including aligning human tumour samples with cancer cell-lines, predicting transcriptome-level perturbation responses, and batch correction on single-cell RNA sequencing data. We also find parallels to fair representation learning and demonstrate that CoMP is competitive on a common task in the field.

        ----

        ## [284] Label Ranking through Nonparametric Regression

        **Authors**: *Dimitris Fotakis, Alkis Kalavasis, Eleni Psaroudaki*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fotakis22a.html](https://proceedings.mlr.press/v162/fotakis22a.html)

        **Abstract**:

        Label Ranking (LR) corresponds to the problem of learning a hypothesis that maps features to rankings over a finite set of labels. We adopt a nonparametric regression approach to LR and obtain theoretical performance guarantees for this fundamental practical problem. We introduce a generative model for Label Ranking, in noiseless and noisy nonparametric regression settings, and provide sample complexity bounds for learning algorithms in both cases. In the noiseless setting, we study the LR problem with full rankings and provide computationally efficient algorithms using decision trees and random forests in the high-dimensional regime. In the noisy setting, we consider the more general cases of LR with incomplete and partial rankings from a statistical viewpoint and obtain sample complexity bounds using the One-Versus-One approach of multiclass classification. Finally, we complement our theoretical contributions with experiments, aiming to understand how the input regression noise affects the observed output.

        ----

        ## [285] A Neural Tangent Kernel Perspective of GANs

        **Authors**: *Jean-Yves Franceschi, Emmanuel de Bézenac, Ibrahim Ayed, Mickaël Chen, Sylvain Lamprier, Patrick Gallinari*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/franceschi22a.html](https://proceedings.mlr.press/v162/franceschi22a.html)

        **Abstract**:

        We propose a novel theoretical framework of analysis for Generative Adversarial Networks (GANs). We reveal a fundamental flaw of previous analyses which, by incorrectly modeling GANs’ training scheme, are subject to ill-defined discriminator gradients. We overcome this issue which impedes a principled study of GAN training, solving it within our framework by taking into account the discriminator’s architecture. To this end, we leverage the theory of infinite-width neural networks for the discriminator via its Neural Tangent Kernel. We characterize the trained discriminator for a wide range of losses and establish general differentiability properties of the network. From this, we derive new insights about the convergence of the generated distribution, advancing our understanding of GANs’ training dynamics. We empirically corroborate these results via an analysis toolkit based on our framework, unveiling intuitions that are consistent with GAN practice.

        ----

        ## [286] Extracting Latent State Representations with Linear Dynamics from Rich Observations

        **Authors**: *Abraham Frandsen, Rong Ge, Holden Lee*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/frandsen22a.html](https://proceedings.mlr.press/v162/frandsen22a.html)

        **Abstract**:

        Recently, many reinforcement learning techniques have been shown to have provable guarantees in the simple case of linear dynamics, especially in problems like linear quadratic regulators. However, in practice many tasks require learning a policy from rich, high-dimensional features such as images, which are unlikely to be linear. We consider a setting where there is a hidden linear subspace of the high-dimensional feature space in which the dynamics are linear. We design natural objectives based on forward and inverse dynamics models. We prove that these objectives can be efficiently optimized and their local optimizers extract the hidden linear subspace. We empirically verify our theoretical results with synthetic data and explore the effectiveness of our approach (generalized to nonlinear settings) in simple control tasks with rich observations.

        ----

        ## [287] SPDY: Accurate Pruning with Speedup Guarantees

        **Authors**: *Elias Frantar, Dan Alistarh*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/frantar22a.html](https://proceedings.mlr.press/v162/frantar22a.html)

        **Abstract**:

        The recent focus on the efficiency of deep neural networks (DNNs) has led to significant work on model compression approaches, of which weight pruning is one of the most popular. At the same time, there is rapidly-growing computational support for efficiently executing the unstructured-sparse models obtained via pruning. Yet, most existing pruning methods minimize just the number of remaining weights, i.e. the size of the model, rather than optimizing for inference time. We address this gap by introducing SPDY, a new compression method which automatically determines layer-wise sparsity targets achieving a desired inference speedup on a given system, while minimizing accuracy loss. SPDY is the composition of two new techniques. The first is an efficient and general dynamic programming algorithm for solving constrained layer-wise compression problems, given a set of layer-wise error scores. The second technique is a local search procedure for automatically determining such scores in an accurate and robust manner. Experiments across popular vision and language models show that SPDY guarantees speedups while recovering higher accuracy relative to existing strategies, both for one-shot and gradual pruning scenarios, and is compatible with most existing pruning approaches. We also extend our approach to the recently-proposed task of pruning with very little data, where we achieve the best known accuracy recovery when pruning to the GPU-supported 2:4 sparsity pattern.

        ----

        ## [288] Revisiting the Effects of Stochasticity for Hamiltonian Samplers

        **Authors**: *Giulio Franzese, Dimitrios Milios, Maurizio Filippone, Pietro Michiardi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/franzese22a.html](https://proceedings.mlr.press/v162/franzese22a.html)

        **Abstract**:

        We revisit the theoretical properties of Hamiltonian stochastic differential equations (SDES) for Bayesian posterior sampling, and we study the two types of errors that arise from numerical SDE simulation: the discretization error and the error due to noisy gradient estimates in the context of data subsampling. Our main result is a novel analysis for the effect of mini-batches through the lens of differential operator splitting, revising previous literature results. The stochastic component of a Hamiltonian SDE is decoupled from the gradient noise, for which we make no normality assumptions. This leads to the identification of a convergence bottleneck: when considering mini-batches, the best achievable error rate is $\mathcal{O}(\eta^2)$, with $\eta$ being the integrator step size. Our theoretical results are supported by an empirical study on a variety of regression and classification tasks for Bayesian neural networks.

        ----

        ## [289] Bregman Neural Networks

        **Authors**: *Jordan Frécon, Gilles Gasso, Massimiliano Pontil, Saverio Salzo*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/frecon22a.html](https://proceedings.mlr.press/v162/frecon22a.html)

        **Abstract**:

        We present a framework based on bilevel optimization for learning multilayer, deep data representations. On the one hand, the lower-level problem finds a representation by successively minimizing layer-wise objectives made of the sum of a prescribed regularizer as well as a fidelity term and some linear function both depending on the representation found at the previous layer. On the other hand, the upper-level problem optimizes over the linear functions to yield a linearly separable final representation. We show that, by choosing the fidelity term as the quadratic distance between two successive layer-wise representations, the bilevel problem reduces to the training of a feed-forward neural network. Instead, by elaborating on Bregman distances, we devise a novel neural network architecture additionally involving the inverse of the activation function reminiscent of the skip connection used in ResNets. Numerical experiments suggest that the proposed Bregman variant benefits from better learning properties and more robust prediction performance.

        ----

        ## [290] (Non-)Convergence Results for Predictive Coding Networks

        **Authors**: *Simon Frieder, Thomas Lukasiewicz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/frieder22a.html](https://proceedings.mlr.press/v162/frieder22a.html)

        **Abstract**:

        Predictive coding networks (PCNs) are (un)supervised learning models, coming from neuroscience, that approximate how the brain works. One major open problem around PCNs is their convergence behavior. In this paper, we use dynamical systems theory to formally investigate the convergence of PCNs as they are used in machine learning. Doing so, we put their theory on a firm, rigorous basis, by developing a precise mathematical framework for PCN and show that for sufficiently small weights and initializations, PCNs converge for any input. Thereby, we provide the theoretical assurance that previous implementations, whose convergence was assessed solely by numerical experiments, can indeed capture the correct behavior of PCNs. Outside of the identified regime of small weights and small initializations, we show via a counterexample that PCNs can diverge, countering common beliefs held in the community. This is achieved by identifying a Neimark-Sacker bifurcation in a PCN of small size, which gives rise to an unstable fixed point and an invariant curve around it.

        ----

        ## [291] Scaling Structured Inference with Randomization

        **Authors**: *Yao Fu, John P. Cunningham, Mirella Lapata*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fu22a.html](https://proceedings.mlr.press/v162/fu22a.html)

        **Abstract**:

        Deep discrete structured models have seen considerable progress recently, but traditional inference using dynamic programming (DP) typically works with a small number of states (less than hundreds), which severely limits model capacity. At the same time, across machine learning, there is a recent trend of using randomized truncation techniques to accelerate computations involving large sums. Here, we propose a family of randomized dynamic programming (RDP) algorithms for scaling structured models to tens of thousands of latent states. Our method is widely applicable to classical DP-based inference (partition, marginal, reparameterization, entropy) and different graph structures (chains, trees, and more general hypergraphs). It is also compatible with automatic differentiation: it can be integrated with neural networks seamlessly and learned with gradient-based optimizers. Our core technique approximates the sum-product by restricting and reweighting DP on a small subset of nodes, which reduces computation by orders of magnitude. We further achieve low bias and variance via Rao-Blackwellization and importance sampling. Experiments over different graphs demonstrate the accuracy and efficiency of our approach. Furthermore, when using RDP for training a structured variational autoencoder with a scaled inference network, we achieve better test likelihood than baselines and successfully prevent posterior collapse.

        ----

        ## [292] Greedy when Sure and Conservative when Uncertain about the Opponents

        **Authors**: *Haobo Fu, Ye Tian, Hongxiang Yu, Weiming Liu, Shuang Wu, Jiechao Xiong, Ying Wen, Kai Li, Junliang Xing, Qiang Fu, Wei Yang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fu22b.html](https://proceedings.mlr.press/v162/fu22b.html)

        **Abstract**:

        We develop a new approach, named Greedy when Sure and Conservative when Uncertain (GSCU), to competing online against unknown and nonstationary opponents. GSCU improves in four aspects: 1) introduces a novel way of learning opponent policy embeddings offline; 2) trains offline a single best response (conditional additionally on our opponent policy embedding) instead of a finite set of separate best responses against any opponent; 3) computes online a posterior of the current opponent policy embedding, without making the discrete and ineffective decision which type the current opponent belongs to; and 4) selects online between a real-time greedy policy and a fixed conservative policy via an adversarial bandit algorithm, gaining a theoretically better regret than adhering to either. Experimental studies on popular benchmarks demonstrate GSCU’s superiority over the state-of-the-art methods. The code is available online at \url{https://github.com/YeTianJHU/GSCU}.

        ----

        ## [293] DepthShrinker: A New Compression Paradigm Towards Boosting Real-Hardware Efficiency of Compact Neural Networks

        **Authors**: *Yonggan Fu, Haichuan Yang, Jiayi Yuan, Meng Li, Cheng Wan, Raghuraman Krishnamoorthi, Vikas Chandra, Yingyan Lin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fu22c.html](https://proceedings.mlr.press/v162/fu22c.html)

        **Abstract**:

        Efficient deep neural network (DNN) models equipped with compact operators (e.g., depthwise convolutions) have shown great potential in reducing DNNs’ theoretical complexity (e.g., the total number of weights/operations) while maintaining a decent model accuracy. However, existing efficient DNNs are still limited in fulfilling their promise in boosting real-hardware efficiency, due to their commonly adopted compact operators’ low hardware utilization. In this work, we open up a new compression paradigm for developing real-hardware efficient DNNs, leading to boosted hardware efficiency while maintaining model accuracy. Interestingly, we observe that while some DNN layers’ activation functions help DNNs’ training optimization and achievable accuracy, they can be properly removed after training without compromising the model accuracy. Inspired by this observation, we propose a framework dubbed DepthShrinker, which develops hardware-friendly compact networks via shrinking the basic building blocks of existing efficient DNNs that feature irregular computation patterns into dense ones with much improved hardware utilization and thus real-hardware efficiency. Excitingly, our DepthShrinker framework delivers hardware-friendly compact networks that outperform both state-of-the-art efficient DNNs and compression techniques, e.g., a 3.06% higher accuracy and 1.53x throughput on Tesla V100 over SOTA channel-wise pruning method MetaPruning. Our codes are available at: https://github.com/facebookresearch/DepthShrinker.

        ----

        ## [294] Revisiting Some Common Practices in Cooperative Multi-Agent Reinforcement Learning

        **Authors**: *Wei Fu, Chao Yu, Zelai Xu, Jiaqi Yang, Yi Wu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fu22d.html](https://proceedings.mlr.press/v162/fu22d.html)

        **Abstract**:

        Many advances in cooperative multi-agent reinforcement learning (MARL) are based on two common design principles: value decomposition and parameter sharing. A typical MARL algorithm of this fashion decomposes a centralized Q-function into local Q-networks with parameters shared across agents. Such an algorithmic paradigm enables centralized training and decentralized execution (CTDE) and leads to efficient learning in practice. Despite all the advantages, we revisit these two principles and show that in certain scenarios, e.g., environments with a highly multi-modal reward landscape, value decomposition, and parameter sharing can be problematic and lead to undesired outcomes. In contrast, policy gradient (PG) methods with individual policies provably converge to an optimal solution in these cases, which partially supports some recent empirical observations that PG can be effective in many MARL testbeds. Inspired by our theoretical analysis, we present practical suggestions on implementing multi-agent PG algorithms for either high rewards or diverse emergent behaviors and empirically validate our findings on a variety of domains, ranging from the simplified matrix and grid-world games to complex benchmarks such as StarCraft Multi-Agent Challenge and Google Research Football. We hope our insights could benefit the community towards developing more general and more powerful MARL algorithms.

        ----

        ## [295] p-Laplacian Based Graph Neural Networks

        **Authors**: *Guoji Fu, Peilin Zhao, Yatao Bian*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fu22e.html](https://proceedings.mlr.press/v162/fu22e.html)

        **Abstract**:

        Graph neural networks (GNNs) have demonstrated superior performance for semi-supervised node classification on graphs, as a result of their ability to exploit node features and topological information simultaneously. However, most GNNs implicitly assume that the labels of nodes and their neighbors in a graph are the same or consistent, which does not hold in heterophilic graphs, where the labels of linked nodes are likely to differ. Moreover, when the topology is non-informative for label prediction, ordinary GNNs may work significantly worse than simply applying multi-layer perceptrons (MLPs) on each node. To tackle the above problem, we propose a new $p$-Laplacian based GNN model, termed as $^p$GNN, whose message passing mechanism is derived from a discrete regularization framework and could be theoretically explained as an approximation of a polynomial graph filter defined on the spectral domain of $p$-Laplacians. The spectral analysis shows that the new message passing mechanism works as low-high-pass filters, thus making $^p$GNNs are effective on both homophilic and heterophilic graphs. Empirical studies on real-world and synthetic datasets validate our findings and demonstrate that $^p$GNNs significantly outperform several state-of-the-art GNN architectures on heterophilic benchmarks while achieving competitive performance on homophilic benchmarks. Moreover, $^p$GNNs can adaptively learn aggregation weights and are robust to noisy edges.

        ----

        ## [296] Why Should I Trust You, Bellman? The Bellman Error is a Poor Replacement for Value Error

        **Authors**: *Scott Fujimoto, David Meger, Doina Precup, Ofir Nachum, Shixiang Shane Gu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/fujimoto22a.html](https://proceedings.mlr.press/v162/fujimoto22a.html)

        **Abstract**:

        In this work, we study the use of the Bellman equation as a surrogate objective for value prediction accuracy. While the Bellman equation is uniquely solved by the true value function over all state-action pairs, we find that the Bellman error (the difference between both sides of the equation) is a poor proxy for the accuracy of the value function. In particular, we show that (1) due to cancellations from both sides of the Bellman equation, the magnitude of the Bellman error is only weakly related to the distance to the true value function, even when considering all state-action pairs, and (2) in the finite data regime, the Bellman equation can be satisfied exactly by infinitely many suboptimal solutions. This means that the Bellman error can be minimized without improving the accuracy of the value function. We demonstrate these phenomena through a series of propositions, illustrative toy examples, and empirical analysis in standard benchmark domains.

        ----

        ## [297] Robin Hood and Matthew Effects: Differential Privacy Has Disparate Impact on Synthetic Data

        **Authors**: *Georgi Ganev, Bristena Oprisanu, Emiliano De Cristofaro*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ganev22a.html](https://proceedings.mlr.press/v162/ganev22a.html)

        **Abstract**:

        Generative models trained with Differential Privacy (DP) can be used to generate synthetic data while minimizing privacy risks. We analyze the impact of DP on these models vis-a-vis underrepresented classes/subgroups of data, specifically, studying: 1) the size of classes/subgroups in the synthetic data, and 2) the accuracy of classification tasks run on them. We also evaluate the effect of various levels of imbalance and privacy budgets. Our analysis uses three state-of-the-art DP models (PrivBayes, DP-WGAN, and PATE-GAN) and shows that DP yields opposite size distributions in the generated synthetic data. It affects the gap between the majority and minority classes/subgroups; in some cases by reducing it (a "Robin Hood" effect) and, in others, by increasing it (a "Matthew" effect). Either way, this leads to (similar) disparate impacts on the accuracy of classification tasks on the synthetic data, affecting disproportionately more the underrepresented subparts of the data. Consequently, when training models on synthetic data, one might incur the risk of treating different subpopulations unevenly, leading to unreliable or unfair conclusions.

        ----

        ## [298] The Complexity of k-Means Clustering when Little is Known

        **Authors**: *Robert Ganian, Thekla Hamm, Viktoriia Korchemna, Karolina Okrasa, Kirill Simonov*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ganian22a.html](https://proceedings.mlr.press/v162/ganian22a.html)

        **Abstract**:

        In the area of data analysis and arguably even in machine learning as a whole, few approaches have been as impactful as the classical k-means clustering. Here, we study the complexity of k-means clustering in settings where most of the data is not known or simply irrelevant. To obtain a more fine-grained understanding of the tractability of this clustering problem, we apply the parameterized complexity paradigm and obtain three new algorithms for k-means clustering of incomplete data: one for the clustering of bounded-domain (i.e., integer) data, and two incomparable algorithms that target real-valued data. Our approach is based on exploiting structural properties of a graphical encoding of the missing entries, and we show that tractability can be achieved using significantly less restrictive parameterizations than in the complementary case of few missing entries.

        ----

        ## [299] IDYNO: Learning Nonparametric DAGs from Interventional Dynamic Data

        **Authors**: *Tian Gao, Debarun Bhattacharjya, Elliot Nelson, Miao Liu, Yue Yu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22a.html](https://proceedings.mlr.press/v162/gao22a.html)

        **Abstract**:

        Causal discovery in the form of a directed acyclic graph (DAG) for time series data has been widely studied in various domains. The resulting DAG typically represents a dynamic Bayesian network (DBN), capturing both the instantaneous and time-delayed relationships among variables of interest. We propose a new algorithm, IDYNO, to learn the DAG structure from potentially nonlinear times series data by using a continuous optimization framework that includes a recent formulation for continuous acyclicity constraint. The proposed algorithm is designed to handle both observational and interventional time series data. We demonstrate the promising performance of our method on synthetic benchmark datasets against state-of-the-art baselines. In addition, we show that the proposed method can more accurately learn the underlying structure of a sequential decision model, such as a Markov decision process, with a fixed policy in typical continuous control tasks.

        ----

        ## [300] Loss Function Learning for Domain Generalization by Implicit Gradient

        **Authors**: *Boyan Gao, Henry Gouk, Yongxin Yang, Timothy M. Hospedales*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22b.html](https://proceedings.mlr.press/v162/gao22b.html)

        **Abstract**:

        Generalising robustly to distribution shift is a major challenge that is pervasive across most real-world applications of machine learning. A recent study highlighted that many advanced algorithms proposed to tackle such domain generalisation (DG) fail to outperform a properly tuned empirical risk minimisation (ERM) baseline. We take a different approach, and explore the impact of the ERM loss function on out-of-domain generalisation. In particular, we introduce a novel meta-learning approach to loss function search based on implicit gradient. This enables us to discover a general purpose parametric loss function that provides a drop-in replacement for cross-entropy. Our loss can be used in standard training pipelines to efficiently train robust models using any neural architecture on new datasets. The results show that it clearly surpasses cross-entropy, enables simple ERM to outperform some more complicated prior DG methods, and provides state-of-the-art performance across a variety of DG benchmarks. Furthermore, unlike most existing DG approaches, our setup applies to the most practical setting of single-source domain generalisation, on which we show significant improvement.

        ----

        ## [301] On the Convergence of Local Stochastic Compositional Gradient Descent with Momentum

        **Authors**: *Hongchang Gao, Junyi Li, Heng Huang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22c.html](https://proceedings.mlr.press/v162/gao22c.html)

        **Abstract**:

        Federated Learning has been actively studied due to its efficiency in numerous real-world applications in the past few years. However, the federated stochastic compositional optimization problem is still underexplored, even though it has widespread applications in machine learning. In this paper, we developed a novel local stochastic compositional gradient descent with momentum method, which facilitates Federated Learning for the stochastic compositional problem. Importantly, we investigated the convergence rate of our proposed method and proved that it can achieve the $O(1/\epsilon^4)$ sample complexity, which is better than existing methods. Meanwhile, our communication complexity $O(1/\epsilon^3)$ can match existing methods. To the best of our knowledge, this is the first work achieving such favorable sample and communication complexities. Additionally, extensive experimental results demonstrate the superior empirical performance over existing methods, confirming the efficacy of our method.

        ----

        ## [302] Deep Reference Priors: What is the best way to pretrain a model?

        **Authors**: *Yansong Gao, Rahul Ramesh, Pratik Chaudhari*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22d.html](https://proceedings.mlr.press/v162/gao22d.html)

        **Abstract**:

        What is the best way to exploit extra data – be it unlabeled data from the same task, or labeled data from a related task – to learn a given task? This paper formalizes the question using the theory of reference priors. Reference priors are objective, uninformative Bayesian priors that maximize the mutual information between the task and the weights of the model. Such priors enable the task to maximally affect the Bayesian posterior, e.g., reference priors depend upon the number of samples available for learning the task and for very small sample sizes, the prior puts more probability mass on low-complexity models in the hypothesis space. This paper presents the first demonstration of reference priors for medium-scale deep networks and image-based data. We develop generalizations of reference priors and demonstrate applications to two problems. First, by using unlabeled data to compute the reference prior, we develop new Bayesian semi-supervised learning methods that remain effective even with very few samples per class. Second, by using labeled data from the source task to compute the reference prior, we develop a new pretraining method for transfer learning that allows data from the target task to maximally affect the Bayesian posterior. Empirical validation of these methods is conducted on image classification datasets. Code is available at https://github.com/grasp-lyrl/deep_reference_priors

        ----

        ## [303] On the Equivalence Between Temporal and Static Equivariant Graph Representations

        **Authors**: *Jianfei Gao, Bruno Ribeiro*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22e.html](https://proceedings.mlr.press/v162/gao22e.html)

        **Abstract**:

        This work formalizes the associational task of predicting node attribute evolution in temporal graphs from the perspective of learning equivariant representations. We show that node representations in temporal graphs can be cast into two distinct frameworks: (a) The most popular approach, which we denote as time-and-graph, where equivariant graph (e.g., GNN) and sequence (e.g., RNN) representations are intertwined to represent the temporal evolution of node attributes in the graph; and (b) an approach that we denote as time-then-graph, where the sequences describing the node and edge dynamics are represented first, then fed as node and edge attributes into a static equivariant graph representation that comes after. Interestingly, we show that time-then-graph representations have an expressivity advantage over time-and-graph representations when both use component GNNs that are not most-expressive (e.g., 1-Weisfeiler-Lehman GNNs). Moreover, while our goal is not necessarily to obtain state-of-the-art results, our experiments show that time-then-graph methods are capable of achieving better performance and efficiency than state-of-the-art time-and-graph methods in some real-world tasks, thereby showcasing that the time-then-graph framework is a worthy addition to the graph ML toolbox.

        ----

        ## [304] Generalizing Gaussian Smoothing for Random Search

        **Authors**: *Katelyn Gao, Ozan Sener*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22f.html](https://proceedings.mlr.press/v162/gao22f.html)

        **Abstract**:

        Gaussian smoothing (GS) is a derivative-free optimization (DFO) algorithm that estimates the gradient of an objective using perturbations of the current parameters sampled from a standard normal distribution. We generalize it to sampling perturbations from a larger family of distributions. Based on an analysis of DFO for non-convex functions, we propose to choose a distribution for perturbations that minimizes the mean squared error (MSE) of the gradient estimate. We derive three such distributions with provably smaller MSE than Gaussian smoothing. We conduct evaluations of the three sampling distributions on linear regression, reinforcement learning, and DFO benchmarks in order to validate our claims. Our proposal improves on GS with the same computational complexity, and are competitive with and usually outperform Guided ES and Orthogonal ES, two computationally more expensive algorithms that adapt the covariance matrix of normally distributed perturbations.

        ----

        ## [305] Rethinking Image-Scaling Attacks: The Interplay Between Vulnerabilities in Machine Learning Systems

        **Authors**: *Yue Gao, Ilia Shumailov, Kassem Fawaz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22g.html](https://proceedings.mlr.press/v162/gao22g.html)

        **Abstract**:

        As real-world images come in varying sizes, the machine learning model is part of a larger system that includes an upstream image scaling algorithm. In this paper, we investigate the interplay between vulnerabilities of the image scaling procedure and machine learning models in the decision-based black-box setting. We propose a novel sampling strategy to make a black-box attack exploit vulnerabilities in scaling algorithms, scaling defenses, and the final machine learning model in an end-to-end manner. Based on this scaling-aware attack, we reveal that most existing scaling defenses are ineffective under threat from downstream models. Moreover, we empirically observe that standard black-box attacks can significantly improve their performance by exploiting the vulnerable scaling procedure. We further demonstrate this problem on a commercial Image Analysis API with decision-based black-box attacks.

        ----

        ## [306] Lazy Estimation of Variable Importance for Large Neural Networks

        **Authors**: *Yue Gao, Abby Stevens, Garvesh Raskutti, Rebecca Willett*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22h.html](https://proceedings.mlr.press/v162/gao22h.html)

        **Abstract**:

        As opaque predictive models increasingly impact many areas of modern life, interest in quantifying the importance of a given input variable for making a specific prediction has grown. Recently, there has been a proliferation of model-agnostic methods to measure variable importance (VI) that analyze the difference in predictive power between a full model trained on all variables and a reduced model that excludes the variable(s) of interest. A bottleneck common to these methods is the estimation of the reduced model for each variable (or subset of variables), which is an expensive process that often does not come with theoretical guarantees. In this work, we propose a fast and flexible method for approximating the reduced model with important inferential guarantees. We replace the need for fully retraining a wide neural network by a linearization initialized at the full model parameters. By adding a ridge-like penalty to make the problem convex, we prove that when the ridge penalty parameter is sufficiently large, our method estimates the variable importance measure with an error rate of O(1/n) where n is the number of training samples. We also show that our estimator is asymptotically normal, enabling us to provide confidence bounds for the VI estimates. We demonstrate through simulations that our method is fast and accurate under several data-generating regimes, and we demonstrate its real-world applicability on a seasonal climate forecasting example.

        ----

        ## [307] Fast and Reliable Evaluation of Adversarial Robustness with Minimum-Margin Attack

        **Authors**: *Ruize Gao, Jiongxiao Wang, Kaiwen Zhou, Feng Liu, Binghui Xie, Gang Niu, Bo Han, James Cheng*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22i.html](https://proceedings.mlr.press/v162/gao22i.html)

        **Abstract**:

        The AutoAttack (AA) has been the most reliable method to evaluate adversarial robustness when considerable computational resources are available. However, the high computational cost (e.g., 100 times more than that of the project gradient descent attack) makes AA infeasible for practitioners with limited computational resources, and also hinders applications of AA in the adversarial training (AT). In this paper, we propose a novel method, minimum-margin (MM) attack, to fast and reliably evaluate adversarial robustness. Compared with AA, our method achieves comparable performance but only costs 3% of the computational time in extensive experiments. The reliability of our method lies in that we evaluate the quality of adversarial examples using the margin between two targets that can precisely identify the most adversarial example. The computational efficiency of our method lies in an effective Sequential TArget Ranking Selection (STARS) method, ensuring that the cost of the MM attack is independent of the number of classes. The MM attack opens a new way for evaluating adversarial robustness and provides a feasible and reliable way to generate high-quality adversarial examples in AT.

        ----

        ## [308] Value Function based Difference-of-Convex Algorithm for Bilevel Hyperparameter Selection Problems

        **Authors**: *Lucy L. Gao, Jane J. Ye, Haian Yin, Shangzhi Zeng, Jin Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22j.html](https://proceedings.mlr.press/v162/gao22j.html)

        **Abstract**:

        Existing gradient-based optimization methods for hyperparameter tuning can only guarantee theoretical convergence to stationary solutions when the bilevel program satisfies the condition that for fixed upper-level variables, the lower-level is strongly convex (LLSC) and smooth (LLS). This condition is not satisfied for bilevel programs arising from tuning hyperparameters in many machine learning algorithms. In this work, we develop a sequentially convergent Value Function based Difference-of-Convex Algorithm with inexactness (VF-iDCA). We then ask: can this algorithm achieve stationary solutions without LLSC and LLS assumptions? We provide a positive answer to this question for bilevel programs from a broad class of hyperparameter tuning applications. Extensive experiments justify our theoretical results and demonstrate the superiority of the proposed VF-iDCA when applied to tune hyperparameters.

        ----

        ## [309] Learning to Incorporate Texture Saliency Adaptive Attention to Image Cartoonization

        **Authors**: *Xiang Gao, Yuqi Zhang, Yingjie Tian*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gao22k.html](https://proceedings.mlr.press/v162/gao22k.html)

        **Abstract**:

        Image cartoonization is recently dominated by generative adversarial networks (GANs) from the perspective of unsupervised image-to-image translation, in which an inherent challenge is to precisely capture and sufficiently transfer characteristic cartoon styles (e.g., clear edges, smooth color shading, vivid colors, etc.). Existing advanced models try to enhance cartoonization effect by learning to promote edges adversarially, introducing style transfer loss, or learning to align style from multiple representation space. This paper demonstrates that more distinct and vivid cartoonization effect could be easily achieved with only basic adversarial loss. Observing that cartoon style is more evident in cartoon-texture-salient local image regions, we build a region-level adversarial learning branch in parallel with the normal image-level one, which constrains adversarial learning on cartoon-texture-salient local patches for better perceiving and transferring cartoon texture features. To this end, a novel cartoon-texture-saliency-sampler (CTSS) module is proposed to adaptively sample cartoon-texture-salient patches from training data. We present that such texture saliency adaptive attention is of significant importance in facilitating and enhancing cartoon stylization, which is a key missing ingredient of related methods. The superiority of our model in promoting cartoonization effect, especially for high-resolution input images, are fully demonstrated with extensive experiments.

        ----

        ## [310] Stochastic smoothing of the top-K calibrated hinge loss for deep imbalanced classification

        **Authors**: *Camille Garcin, Maximilien Servajean, Alexis Joly, Joseph Salmon*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/garcin22a.html](https://proceedings.mlr.press/v162/garcin22a.html)

        **Abstract**:

        In modern classification tasks, the number of labels is getting larger and larger, as is the size of the datasets encountered in practice. As the number of classes increases, class ambiguity and class imbalance become more and more problematic to achieve high top-1 accuracy. Meanwhile, Top-K metrics (metrics allowing K guesses) have become popular, especially for performance reporting. Yet, proposing top-K losses tailored for deep learning remains a challenge, both theoretically and practically. In this paper we introduce a stochastic top-K hinge loss inspired by recent developments on top-K calibrated losses. Our proposal is based on the smoothing of the top-K operator building on the flexible "perturbed optimizer" framework. We show that our loss function performs very well in the case of balanced datasets, while benefiting from a significantly lower computational time than the state-of-the-art top-K loss function. In addition, we propose a simple variant of our loss for the imbalanced case. Experiments on a heavy-tailed dataset show that our loss function significantly outperforms other baseline loss functions.

        ----

        ## [311] PAGE-PG: A Simple and Loopless Variance-Reduced Policy Gradient Method with Probabilistic Gradient Estimation

        **Authors**: *Matilde Gargiani, Andrea Zanelli, Andrea Martinelli, Tyler H. Summers, John Lygeros*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gargiani22a.html](https://proceedings.mlr.press/v162/gargiani22a.html)

        **Abstract**:

        Despite their success, policy gradient methods suffer from high variance of the gradient estimator, which can result in unsatisfactory sample complexity. Recently, numerous variance-reduced extensions of policy gradient methods with provably better sample complexity and competitive numerical performance have been proposed. After a compact survey on some of the main variance-reduced REINFORCE-type methods, we propose ProbAbilistic Gradient Estimation for Policy Gradient (PAGE-PG), a novel loopless variance-reduced policy gradient method based on a probabilistic switch between two types of update. Our method is inspired by the PAGE estimator for supervised learning and leverages importance sampling to obtain an unbiased gradient estimator. We show that PAGE-PG enjoys a $\mathcal{O}\left( \epsilon^{-3} \right)$ average sample complexity to reach an $\epsilon$-stationary solution, which matches the sample complexity of its most competitive counterparts under the same setting. A numerical evaluation confirms the competitive performance of our method on classical control tasks.

        ----

        ## [312] The power of first-order smooth optimization for black-box non-smooth problems

        **Authors**: *Alexander V. Gasnikov, Anton Novitskii, Vasilii Novitskii, Farshed Abdukhakimov, Dmitry Kamzolov, Aleksandr Beznosikov, Martin Takác, Pavel E. Dvurechensky, Bin Gu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gasnikov22a.html](https://proceedings.mlr.press/v162/gasnikov22a.html)

        **Abstract**:

        Gradient-free/zeroth-order methods for black-box convex optimization have been extensively studied in the last decade with the main focus on oracle calls complexity. In this paper, besides the oracle complexity, we focus also on iteration complexity, and propose a generic approach that, based on optimal first-order methods, allows to obtain in a black-box fashion new zeroth-order algorithms for non-smooth convex optimization problems. Our approach not only leads to optimal oracle complexity, but also allows to obtain iteration complexity similar to first-order methods, which, in turn, allows to exploit parallel computations to accelerate the convergence of our algorithms. We also elaborate on extensions for stochastic optimization problems, saddle-point problems, and distributed optimization.

        ----

        ## [313] A Functional Information Perspective on Model Interpretation

        **Authors**: *Itai Gat, Nitay Calderon, Roi Reichart, Tamir Hazan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gat22a.html](https://proceedings.mlr.press/v162/gat22a.html)

        **Abstract**:

        Contemporary predictive models are hard to interpret as their deep nets exploit numerous complex relations between input elements. This work suggests a theoretical framework for model interpretability by measuring the contribution of relevant features to the functional entropy of the network with respect to the input. We rely on the log-Sobolev inequality that bounds the functional entropy by the functional Fisher information with respect to the covariance of the data. This provides a principled way to measure the amount of information contribution of a subset of features to the decision function. Through extensive experiments, we show that our method surpasses existing interpretability sampling-based methods on various data signals such as image, text, and audio.

        ----

        ## [314] UniRank: Unimodal Bandit Algorithms for Online Ranking

        **Authors**: *Camille-Sovanneary Gauthier, Romaric Gaudel, Élisa Fromont*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gauthier22a.html](https://proceedings.mlr.press/v162/gauthier22a.html)

        **Abstract**:

        We tackle, in the multiple-play bandit setting, the online ranking problem of assigning L items to K predefined positions on a web page in order to maximize the number of user clicks. We propose a generic algorithm, UniRank, that tackles state-of-the-art click models. The regret bound of this algorithm is a direct consequence of the pseudo-unimodality property of the bandit setting with respect to a graph where nodes are ordered sets of indistinguishable items. The main contribution of UniRank is its O(L/$\Delta$ logT) regret for T consecutive assignments, where $\Delta$ relates to the reward-gap between two items. This regret bound is based on the usually implicit condition that two items may not have the same attractiveness. Experiments against state-of-the-art learning algorithms specialized or not for different click models, show that our method has better regret performance than other generic algorithms on real life and synthetic datasets.

        ----

        ## [315] Variational Inference with Locally Enhanced Bounds for Hierarchical Models

        **Authors**: *Tomas Geffner, Justin Domke*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/geffner22a.html](https://proceedings.mlr.press/v162/geffner22a.html)

        **Abstract**:

        Hierarchical models represent a challenging setting for inference algorithms. MCMC methods struggle to scale to large models with many local variables and observations, and variational inference (VI) may fail to provide accurate approximations due to the use of simple variational families. Some variational methods (e.g. importance weighted VI) integrate Monte Carlo methods to give better accuracy, but these tend to be unsuitable for hierarchical models, as they do not allow for subsampling and their performance tends to degrade for high dimensional models. We propose a new family of variational bounds for hierarchical models, based on the application of tightening methods (e.g. importance weighting) separately for each group of local random variables. We show that our approach naturally allows the use of subsampling to get unbiased gradients, and that it fully leverages the power of methods that build tighter lower bounds by applying them independently in lower dimensional spaces, leading to better results and more accurate posterior approximations than relevant baselines.

        ----

        ## [316] Inducing Causal Structure for Interpretable Neural Networks

        **Authors**: *Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah D. Goodman, Christopher Potts*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/geiger22a.html](https://proceedings.mlr.press/v162/geiger22a.html)

        **Abstract**:

        In many areas, we have well-founded insights about causal structure that would be useful to bring into our trained models while still allowing them to learn in a data-driven fashion. To achieve this, we present the new method of interchange intervention training (IIT). In IIT, we (1) align variables in a causal model (e.g., a deterministic program or Bayesian network) with representations in a neural model and (2) train the neural model to match the counterfactual behavior of the causal model on a base input when aligned representations in both models are set to be the value they would be for a source input. IIT is fully differentiable, flexibly combines with other objectives, and guarantees that the target causal model is a causal abstraction of the neural model when its loss is zero. We evaluate IIT on a structural vision task (MNIST-PVR), a navigational language task (ReaSCAN), and a natural language inference task (MQNLI). We compare IIT against multi-task training objectives and data augmentation. In all our experiments, IIT achieves the best results and produces neural models that are more interpretable in the sense that they more successfully realize the target causal model.

        ----

        ## [317] Achieving Minimax Rates in Pool-Based Batch Active Learning

        **Authors**: *Claudio Gentile, Zhilei Wang, Tong Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gentile22a.html](https://proceedings.mlr.press/v162/gentile22a.html)

        **Abstract**:

        We consider a batch active learning scenario where the learner adaptively issues batches of points to a labeling oracle. Sampling labels in batches is highly desirable in practice due to the smaller number of interactive rounds with the labeling oracle (often human beings). However, batch active learning typically pays the price of a reduced adaptivity, leading to suboptimal results. In this paper we propose a solution which requires a careful trade off between the informativeness of the queried points and their diversity. We theoretically investigate batch active learning in the practically relevant scenario where the unlabeled pool of data is available beforehand (pool-based active learning). We analyze a novel stage-wise greedy algorithm and show that, as a function of the label complexity, the excess risk of this algorithm %operating in the realizable setting for which we prove matches the known minimax rates in standard statistical learning settings. Our results also exhibit a mild dependence on the batch size. These are the first theoretical results that employ careful trade offs between informativeness and diversity to rigorously quantify the statistical performance of batch active learning in the pool-based scenario.

        ----

        ## [318] Near-Exact Recovery for Tomographic Inverse Problems via Deep Learning

        **Authors**: *Martin Genzel, Ingo Gühring, Jan MacDonald, Maximilian März*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/genzel22a.html](https://proceedings.mlr.press/v162/genzel22a.html)

        **Abstract**:

        This work is concerned with the following fundamental question in scientific machine learning: Can deep-learning-based methods solve noise-free inverse problems to near-perfect accuracy? Positive evidence is provided for the first time, focusing on a prototypical computed tomography (CT) setup. We demonstrate that an iterative end-to-end network scheme enables reconstructions close to numerical precision, comparable to classical compressed sensing strategies. Our results build on our winning submission to the recent AAPM DL-Sparse-View CT Challenge. Its goal was to identify the state-of-the-art in solving the sparse-view CT inverse problem with data-driven techniques. A specific difficulty of the challenge setup was that the precise forward model remained unknown to the participants. Therefore, a key feature of our approach was to initially estimate the unknown fanbeam geometry in a data-driven calibration step. Apart from an in-depth analysis of our methodology, we also demonstrate its state-of-the-art performance on the open-access real-world dataset LoDoPaB CT.

        ----

        ## [319] Online Learning for Min Sum Set Cover and Pandora's Box

        **Authors**: *Evangelia Gergatsouli, Christos Tzamos*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gergatsouli22a.html](https://proceedings.mlr.press/v162/gergatsouli22a.html)

        **Abstract**:

        Two central problems in Stochastic Optimization are Min-Sum Set Cover and Pandora’s Box. In Pandora’s Box, we are presented with n boxes, each containing an unknown value and the goal is to open the boxes in some order to minimize the sum of the search cost and the smallest value found. Given a distribution of value vectors, we are asked to identify a near-optimal search order. Min-Sum Set Cover corresponds to the case where values are either 0 or infinity. In this work, we study the case where the value vectors are not drawn from a distribution but are presented to a learner in an online fashion. We present a computationally efficient algorithm that is constant-competitive against the cost of the optimal search order. We extend our results to a bandit setting where only the values of the boxes opened are revealed to the learner after every round. We also generalize our results to other commonly studied variants of Pandora’s Box and Min-Sum Set Cover that involve selecting more than a single value subject to a matroid constraint.

        ----

        ## [320] Equivariance versus Augmentation for Spherical Images

        **Authors**: *Jan E. Gerken, Oscar Carlsson, Hampus Linander, Fredrik Ohlsson, Christoffer Petersson, Daniel Persson*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gerken22a.html](https://proceedings.mlr.press/v162/gerken22a.html)

        **Abstract**:

        We analyze the role of rotational equivariance in convolutional neural networks (CNNs) applied to spherical images. We compare the performance of the group equivariant networks known as S2CNNs and standard non-equivariant CNNs trained with an increasing amount of data augmentation. The chosen architectures can be considered baseline references for the respective design paradigms. Our models are trained and evaluated on single or multiple items from the MNIST- or FashionMNIST dataset projected onto the sphere. For the task of image classification, which is inherently rotationally invariant, we find that by considerably increasing the amount of data augmentation and the size of the networks, it is possible for the standard CNNs to reach at least the same performance as the equivariant network. In contrast, for the inherently equivariant task of semantic segmentation, the non-equivariant networks are consistently outperformed by the equivariant networks with significantly fewer parameters. We also analyze and compare the inference latency and training times of the different networks, enabling detailed tradeoff considerations between equivariant architectures and data augmentation for practical problems.

        ----

        ## [321] A Regret Minimization Approach to Multi-Agent Control

        **Authors**: *Udaya Ghai, Udari Madhushani, Naomi Ehrich Leonard, Elad Hazan*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ghai22a.html](https://proceedings.mlr.press/v162/ghai22a.html)

        **Abstract**:

        We study the problem of multi-agent control of a dynamical system with known dynamics and adversarial disturbances. Our study focuses on optimal control without centralized precomputed policies, but rather with adaptive control policies for the different agents that are only equipped with a stabilizing controller. We give a reduction from any (standard) regret minimizing control method to a distributed algorithm. The reduction guarantees that the resulting distributed algorithm has low regret relative to the optimal precomputed joint policy. Our methodology involves generalizing online convex optimization to a multi-agent setting and applying recent tools from nonstochastic control derived for a single agent. We empirically evaluate our method on a model of an overactuated aircraft. We show that the distributed method is robust to failure and to adversarial perturbations in the dynamics.

        ----

        ## [322] Blocks Assemble! Learning to Assemble with Large-Scale Structured Reinforcement Learning

        **Authors**: *Seyed Kamyar Seyed Ghasemipour, Satoshi Kataoka, Byron David, Daniel Freeman, Shixiang Shane Gu, Igor Mordatch*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ghasemipour22a.html](https://proceedings.mlr.press/v162/ghasemipour22a.html)

        **Abstract**:

        Assembly of multi-part physical structures is both a valuable end product for autonomous robotics, as well as a valuable diagnostic task for open-ended training of embodied intelligent agents. We introduce a naturalistic physics-based environment with a set of connectable magnet blocks inspired by children’s toy kits. The objective is to assemble blocks into a succession of target blueprints. Despite the simplicity of this objective, the compositional nature of building diverse blueprints from a set of blocks leads to an explosion of complexity in structures that agents encounter. Furthermore, assembly stresses agents’ multi-step planning, physical reasoning, and bimanual coordination. We find that the combination of large-scale reinforcement learning and graph-based policies – surprisingly without any additional complexity – is an effective recipe for training agents that not only generalize to complex unseen blueprints in a zero-shot manner, but even operate in a reset-free setting without being trained to do so. Through extensive experiments, we highlight the importance of large-scale training, structured representations, contributions of multi-task vs. single-task learning, as well as the effects of curriculums, and discuss qualitative behaviors of trained agents. Our accompanying project webpage can be found at: https://sites.google.com/view/learning-direct-assembly/home

        ----

        ## [323] Faster Privacy Accounting via Evolving Discretization

        **Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ghazi22a.html](https://proceedings.mlr.press/v162/ghazi22a.html)

        **Abstract**:

        We introduce a new algorithm for numerical composition of privacy random variables, useful for computing the accurate differential privacy parameters for compositions of mechanisms. Our algorithm achieves a running time and memory usage of $polylog(k)$ for the task of self-composing a mechanism, from a broad class of mechanisms, $k$ times; this class, e.g., includes the sub-sampled Gaussian mechanism, that appears in the analysis of differentially private stochastic gradient descent (DP-SGD). By comparison, recent work by Gopi et al. (NeurIPS 2021) has obtained a running time of $\widetilde{O}(\sqrt{k})$ for the same task. Our approach extends to the case of composing $k$ different mechanisms in the same class, improving upon the running time and memory usage in their work from $\widetilde{O}(k^{1.5})$ to $\wtilde{O}(k)$.

        ----

        ## [324] Plug-In Inversion: Model-Agnostic Inversion for Vision with Data Augmentations

        **Authors**: *Amin Ghiasi, Hamid Kazemi, Steven Reich, Chen Zhu, Micah Goldblum, Tom Goldstein*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ghiasi22a.html](https://proceedings.mlr.press/v162/ghiasi22a.html)

        **Abstract**:

        Existing techniques for model inversion typically rely on hard-to-tune regularizers, such as total variation or feature regularization, which must be individually calibrated for each network in order to produce adequate images. In this work, we introduce Plug-In Inversion, which relies on a simple set of augmentations and does not require excessive hyper-parameter tuning. Under our proposed augmentation-based scheme, the same set of augmentation hyper-parameters can be used for inverting a wide range of image classification models, regardless of input dimensions or the architecture. We illustrate the practicality of our approach by inverting Vision Transformers (ViTs) and Multi-Layer Perceptrons (MLPs) trained on the ImageNet dataset, tasks which to the best of our knowledge have not been successfully accomplished by any previous works.

        ----

        ## [325] Offline RL Policies Should Be Trained to be Adaptive

        **Authors**: *Dibya Ghosh, Anurag Ajay, Pulkit Agrawal, Sergey Levine*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ghosh22a.html](https://proceedings.mlr.press/v162/ghosh22a.html)

        **Abstract**:

        Offline RL algorithms must account for the fact that the dataset they are provided may leave many facets of the environment unknown. The most common way to approach this challenge is to employ pessimistic or conservative methods, which avoid behaviors that are too dissimilar from those in the training dataset. However, relying exclusively on conservatism has drawbacks: performance is sensitive to the exact degree of conservatism, and conservative objectives can recover highly suboptimal policies. In this work, we propose that offline RL methods should instead be adaptive in the presence of uncertainty. We show that acting optimally in offline RL in a Bayesian sense involves solving an implicit POMDP. As a result, optimal policies for offline RL must be adaptive, depending not just on the current state but rather all the transitions seen so far during evaluation. We present a model-free algorithm for approximating this optimal adaptive policy, and demonstrate the efficacy of learning such adaptive policies in offline RL benchmarks.

        ----

        ## [326] Breaking the $\sqrt{T}$ Barrier: Instance-Independent Logarithmic Regret in Stochastic Contextual Linear Bandits

        **Authors**: *Avishek Ghosh, Abishek Sankararaman*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/ghosh22b.html](https://proceedings.mlr.press/v162/ghosh22b.html)

        **Abstract**:

        We prove an instance independent (poly) logarithmic regret for stochastic contextual bandits with linear payoff. Previously, in \cite{chu2011contextual}, a lower bound of $\mathcal{O}(\sqrt{T})$ is shown for the contextual linear bandit problem with arbitrary (adversarily chosen) contexts. In this paper, we show that stochastic contexts indeed help to reduce the regret from $\sqrt{T}$ to $\polylog(T)$. We propose Low Regret Stochastic Contextual Bandits (\texttt{LR-SCB}), which takes advantage of the stochastic contexts and performs parameter estimation (in $\ell_2$ norm) and regret minimization simultaneously. \texttt{LR-SCB} works in epochs, where the parameter estimation of the previous epoch is used to reduce the regret of the current epoch. The (poly) logarithmic regret of \texttt{LR-SCB} stems from two crucial facts: (a) the application of a norm adaptive algorithm to exploit the parameter estimation and (b) an analysis of the shifted linear contextual bandit algorithm, showing that shifting results in increasing regret. We have also shown experimentally that stochastic contexts indeed incurs a regret that scales with $\polylog(T)$.

        ----

        ## [327] SCHA-VAE: Hierarchical Context Aggregation for Few-Shot Generation

        **Authors**: *Giorgio Giannone, Ole Winther*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/giannone22a.html](https://proceedings.mlr.press/v162/giannone22a.html)

        **Abstract**:

        A few-shot generative model should be able to generate data from a novel distribution by only observing a limited set of examples. In few-shot learning the model is trained on data from many sets from distributions sharing some underlying properties such as sets of characters from different alphabets or objects from different categories. We extend current latent variable models for sets to a fully hierarchical approach with an attention-based point to set-level aggregation and call our method SCHA-VAE for Set-Context-Hierarchical-Aggregation Variational Autoencoder. We explore likelihood-based model comparison, iterative data sampling, and adaptation-free out-of-distribution generalization. Our results show that the hierarchical formulation better captures the intrinsic variability within the sets in the small data regime. This work generalizes deep latent variable approaches to few-shot learning, taking a step toward large-scale few-shot generation with a formulation that readily works with current state-of-the-art deep generative models.

        ----

        ## [328] A Joint Exponential Mechanism For Differentially Private Top-k

        **Authors**: *Jennifer Gillenwater, Matthew Joseph, Andres Muñoz Medina, Mónica Ribero Diaz*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gillenwater22a.html](https://proceedings.mlr.press/v162/gillenwater22a.html)

        **Abstract**:

        We present a differentially private algorithm for releasing the sequence of $k$ elements with the highest counts from a data domain of $d$ elements. The algorithm is a "joint" instance of the exponential mechanism, and its output space consists of all $O(d^k)$ length-$k$ sequences. Our main contribution is a method to sample this exponential mechanism in time $O(dk\log(k) + d\log(d))$ and space $O(dk)$. Experiments show that this approach outperforms existing pure differential privacy methods and improves upon even approximate differential privacy methods for moderate $k$.

        ----

        ## [329] Neuro-Symbolic Hierarchical Rule Induction

        **Authors**: *Claire Glanois, Zhaohui Jiang, Xuening Feng, Paul Weng, Matthieu Zimmer, Dong Li, Wulong Liu, Jianye Hao*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/glanois22a.html](https://proceedings.mlr.press/v162/glanois22a.html)

        **Abstract**:

        We propose Neuro-Symbolic Hierarchical Rule Induction, an efficient interpretable neuro-symbolic model, to solve Inductive Logic Programming (ILP) problems. In this model, which is built from a pre-defined set of meta-rules organized in a hierarchical structure, first-order rules are invented by learning embeddings to match facts and body predicates of a meta-rule. To instantiate, we specifically design an expressive set of generic meta-rules, and demonstrate they generate a consequent fragment of Horn clauses. As a differentiable model, HRI can be trained both via supervised learning and reinforcement learning. To converge to interpretable rules, we inject a controlled noise to avoid local optima and employ an interpretability-regularization term. We empirically validate our model on various tasks (ILP, visual genome, reinforcement learning) against relevant state-of-the-art methods, including traditional ILP methods and neuro-symbolic models.

        ----

        ## [330] It's Raw! Audio Generation with State-Space Models

        **Authors**: *Karan Goel, Albert Gu, Chris Donahue, Christopher Ré*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/goel22a.html](https://proceedings.mlr.press/v162/goel22a.html)

        **Abstract**:

        Developing architectures suitable for modeling raw audio is a challenging problem due to the high sampling rates of audio waveforms. Standard sequence modeling approaches like RNNs and CNNs have previously been tailored to fit the demands of audio, but the resultant architectures make undesirable computational tradeoffs and struggle to model waveforms effectively. We propose SaShiMi, a new multi-scale architecture for waveform modeling built around the recently introduced S4 model for long sequence modeling. We identify that S4 can be unstable during autoregressive generation, and provide a simple improvement to its parameterization by drawing connections to Hurwitz matrices. SaShiMi yields state-of-the-art performance for unconditional waveform generation in the autoregressive setting. Additionally, SaShiMi improves non-autoregressive generation performance when used as the backbone architecture for a diffusion model. Compared to prior architectures in the autoregressive generation setting, SaShiMi generates piano and speech waveforms which humans find more musical and coherent respectively, e.g. 2{\texttimes} better mean opinion scores than WaveNet on an unconditional speech generation task. On a music generation task, SaShiMi outperforms WaveNet on density estimation and speed at both training and inference even when using 3{\texttimes} fewer parameters

        ----

        ## [331] RankSim: Ranking Similarity Regularization for Deep Imbalanced Regression

        **Authors**: *Yu Gong, Greg Mori, Frederick Tung*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gong22a.html](https://proceedings.mlr.press/v162/gong22a.html)

        **Abstract**:

        Data imbalance, in which a plurality of the data samples come from a small proportion of labels, poses a challenge in training deep neural networks. Unlike classification, in regression the labels are continuous, potentially boundless, and form a natural ordering. These distinct features of regression call for new techniques that leverage the additional information encoded in label-space relationships. This paper presents the RankSim (ranking similarity) regularizer for deep imbalanced regression, which encodes an inductive bias that samples that are closer in label space should also be closer in feature space. In contrast to recent distribution smoothing based approaches, RankSim captures both nearby and distant relationships: for a given data sample, RankSim encourages the sorted list of its neighbors in label space to match the sorted list of its neighbors in feature space. RankSim is complementary to conventional imbalanced learning techniques, including re-weighting, two-stage training, and distribution smoothing, and lifts the state-of-the-art performance on three imbalanced regression benchmarks: IMDB-WIKI-DIR, AgeDB-DIR, and STS-B-DIR.

        ----

        ## [332] How to Fill the Optimum Set? Population Gradient Descent with Harmless Diversity

        **Authors**: *Chengyue Gong, Lemeng Wu, Qiang Liu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gong22b.html](https://proceedings.mlr.press/v162/gong22b.html)

        **Abstract**:

        Although traditional optimization methods focus on finding a single optimal solution, most objective functions in modern machine learning problems, especially those in deep learning, often have multiple or infinite number of optimal points. Therefore, it is useful to consider the problem of finding a set of diverse points in the optimum set of an objective function. In this work, we frame this problem as a bi-level optimization problem of maximizing a diversity score inside the optimum set of the main loss function, and solve it with a simple population gradient descent framework that iteratively updates the points to maximize the diversity score in a fashion that does not hurt the optimization of the main loss. We demonstrate that our method can efficiently generate diverse solutions on multiple applications, e.g. text-to-image generation, text-to-mesh generation, molecular conformation generation and ensemble neural network training.

        ----

        ## [333] Partial Label Learning via Label Influence Function

        **Authors**: *Xiuwen Gong, Dong Yuan, Wei Bao*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gong22c.html](https://proceedings.mlr.press/v162/gong22c.html)

        **Abstract**:

        To deal with ambiguities in partial label learning (PLL), state-of-the-art strategies implement disambiguations by identifying the ground-truth label directly from the candidate label set. However, these approaches usually take the label that incurs a minimal loss as the ground-truth label or use the weight to represent which label has a high likelihood to be the ground-truth label. Little work has been done to investigate from the perspective of how a candidate label changing a predictive model. In this paper, inspired by influence function, we develop a novel PLL framework called Partial Label Learning via Label Influence Function (PLL-IF). Moreover, we implement the framework with two specific representative models, an SVM model and a neural network model, which are called PLL-IF+SVM and PLL-IF+NN method respectively. Extensive experiments conducted on various datasets demonstrate the superiorities of the proposed methods in terms of prediction accuracy, which in turn validates the effectiveness of the proposed PLL-IF framework.

        ----

        ## [334] Secure Distributed Training at Scale

        **Authors**: *Eduard Gorbunov, Alexander Borzunov, Michael Diskin, Max Ryabinin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gorbunov22a.html](https://proceedings.mlr.press/v162/gorbunov22a.html)

        **Abstract**:

        Many areas of deep learning benefit from using increasingly larger neural networks trained on public data, as is the case for pre-trained models for NLP and computer vision. Training such models requires a lot of computational resources (e.g., HPC clusters) that are not available to small research groups and independent researchers. One way to address it is for several smaller groups to pool their computational resources together and train a model that benefits all participants. Unfortunately, in this case, any participant can jeopardize the entire training run by sending incorrect updates, deliberately or by mistake. Training in presence of such peers requires specialized distributed training algorithms with Byzantine tolerance. These algorithms often sacrifice efficiency by introducing redundant communication or passing all updates through a trusted server, making it infeasible to apply them to large-scale deep learning, where models can have billions of parameters. In this work, we propose a novel protocol for secure (Byzantine-tolerant) decentralized training that emphasizes communication efficiency.

        ----

        ## [335] Retrieval-Augmented Reinforcement Learning

        **Authors**: *Anirudh Goyal, Abram L. Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke, Adrià Puigdomènech Badia, Arthur Guez, Mehdi Mirza, Peter C. Humphreys, Ksenia Konyushkova, Michal Valko, Simon Osindero, Timothy P. Lillicrap, Nicolas Heess, Charles Blundell*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/goyal22a.html](https://proceedings.mlr.press/v162/goyal22a.html)

        **Abstract**:

        Most deep reinforcement learning (RL) algorithms distill experience into parametric behavior policies or value functions via gradient updates. While effective, this approach has several disadvantages: (1) it is computationally expensive, (2) it can take many updates to integrate experiences into the parametric model, (3) experiences that are not fully integrated do not appropriately influence the agent’s behavior, and (4) behavior is limited by the capacity of the model. In this paper we explore an alternative paradigm in which we train a network to map a dataset of past experiences to optimal behavior. Specifically, we augment an RL agent with a retrieval process (parameterized as a neural network) that has direct access to a dataset of experiences. This dataset can come from the agent’s past experiences, expert demonstrations, or any other relevant source. The retrieval process is trained to retrieve information from the dataset that may be useful in the current context, to help the agent achieve its goal faster and more efficiently. The proposed method facilitates learning agents that at test time can condition their behavior on the entire dataset and not only the current state, or current trajectory. We integrate our method into two different RL agents: an offline DQN agent and an online R2D2 agent. In offline multi-task problems, we show that the retrieval-augmented DQN agent avoids task interference and learns faster than the baseline DQN agent. On Atari, we show that retrieval-augmented R2D2 learns significantly faster than the baseline R2D2 agent and achieves higher scores. We run extensive ablations to measure the contributions of the components of our proposed method.

        ----

        ## [336] The State of Sparse Training in Deep Reinforcement Learning

        **Authors**: *Laura Graesser, Utku Evci, Erich Elsen, Pablo Samuel Castro*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/graesser22a.html](https://proceedings.mlr.press/v162/graesser22a.html)

        **Abstract**:

        The use of sparse neural networks has seen rapid growth in recent years, particularly in computer vision. Their appeal stems largely from the reduced number of parameters required to train and store, as well as in an increase in learning efficiency. Somewhat surprisingly, there have been very few efforts exploring their use in Deep Reinforcement Learning (DRL). In this work we perform a systematic investigation into applying a number of existing sparse training techniques on a variety of DRL agents and environments. Our results corroborate the findings from sparse training in the computer vision domain {–}sparse networks perform better than dense networks for the same parameter count{–} in the DRL domain. We provide detailed analyses on how the various components in DRL are affected by the use of sparse networks and conclude by suggesting promising avenues for improving the effectiveness of sparse training methods, as well as for advancing their use in DRL.

        ----

        ## [337] Causal Inference Through the Structural Causal Marginal Problem

        **Authors**: *Luigi Gresele, Julius von Kügelgen, Jonas M. Kübler, Elke Kirschbaum, Bernhard Schölkopf, Dominik Janzing*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gresele22a.html](https://proceedings.mlr.press/v162/gresele22a.html)

        **Abstract**:

        We introduce an approach to counterfactual inference based on merging information from multiple datasets. We consider a causal reformulation of the statistical marginal problem: given a collection of marginal structural causal models (SCMs) over distinct but overlapping sets of variables, determine the set of joint SCMs that are counterfactually consistent with the marginal ones. We formalise this approach for categorical SCMs using the response function formulation and show that it reduces the space of allowed marginal and joint SCMs. Our work thus highlights a new mode of falsifiability through additional variables, in contrast to the statistical one via additional data.

        ----

        ## [338] Mirror Learning: A Unifying Framework of Policy Optimisation

        **Authors**: *Jakub Grudzien Kuba, Christian A. Schröder de Witt, Jakob N. Foerster*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/grudzien22a.html](https://proceedings.mlr.press/v162/grudzien22a.html)

        **Abstract**:

        Modern deep reinforcement learning (RL) algorithms are motivated by either the general policy improvement (GPI) or trust-region learning (TRL) frameworks. However, algorithms that strictly respect these theoretical frameworks have proven unscalable. Surprisingly, the only known scalable algorithms violate the GPI/TRL assumptions, e.g. due to required regularisation or other heuristics. The current explanation of their empirical success is essentially “by analogy”: they are deemed approximate adaptations of theoretically sound methods. Unfortunately, studies have shown that in practice these algorithms differ greatly from their conceptual ancestors. In contrast, in this paper, we introduce a novel theoretical framework, named Mirror Learning, which provides theoretical guarantees to a large class of algorithms, including TRPO and PPO. While the latter two exploit the flexibility of our framework, GPI and TRL fit in merely as pathologically restrictive corner cases thereof. This suggests that the empirical performance of state-of-the-art methods is a direct consequence of their theoretical properties, rather than of aforementioned approximate analogies. Mirror learning sets us free to boldly explore novel, theoretically sound RL algorithms, a thus far uncharted wonderland.

        ----

        ## [339] Adapting k-means Algorithms for Outliers

        **Authors**: *Christoph Grunau, Václav Rozhon*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/grunau22a.html](https://proceedings.mlr.press/v162/grunau22a.html)

        **Abstract**:

        This paper shows how to adapt several simple and classical sampling-based algorithms for the k-means problem to the setting with outliers. Recently, Bhaskara et al. (NeurIPS 2019) showed how to adapt the classical k-means++ algorithm to the setting with outliers. However, their algorithm needs to output O(log(k)$\cdot$z) outliers, where z is the number of true outliers, to match the O(log k)-approximation guarantee of k-means++. In this paper, we build on their ideas and show how to adapt several sequential and distributed k-means algorithms to the setting with outliers, but with substantially stronger theoretical guarantees: our algorithms output (1 + $\epsilon$)z outliers while achieving an O(1/$\epsilon$)-approximation to the objective function. In the sequential world, we achieve this by adapting a recent algorithm of Lattanzi and Sohler (ICML 2019). In the distributed setting, we adapt a simple algorithm of Guha et al. (IEEE Trans. Know. and Data Engineering 2003) and the popular k-means\|{of} Bahmani et al. (PVLDB2012). A theoretical application of our techniques is an algorithm with running time O(nk^2/z) that achieves an O(1)-approximation to the objective function while outputting O(z) outliers, assuming k << z << n. This is complemented with a matching lower bound of $\Omega$(nk^2/z) for this problem in the oracle model.

        ----

        ## [340] Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics

        **Authors**: *Yichen Gu, David T. Blaauw, Joshua D. Welch*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gu22a.html](https://proceedings.mlr.press/v162/gu22a.html)

        **Abstract**:

        A key problem in computational biology is discovering the gene expression changes that regulate cell fate transitions, in which one cell type turns into another. However, each individual cell cannot be tracked longitudinally, and cells at the same point in real time may be at different stages of the transition process. This can be viewed as a problem of learning the behavior of a dynamical system from observations whose times are unknown. Additionally, a single progenitor cell type often bifurcates into multiple child cell types, further complicating the problem of modeling the dynamics. To address this problem, we developed an approach called variational mixtures of ordinary differential equations. By using a simple family of ODEs informed by the biochemistry of gene expression to constrain the likelihood of a deep generative model, we can simultaneously infer the latent time and latent state of each cell and predict its future gene expression state. The model can be interpreted as a mixture of ODEs whose parameters vary continuously across a latent space of cell states. Our approach dramatically improves data fit, latent time inference, and future cell state estimation of single-cell gene expression data compared to previous approaches.

        ----

        ## [341] Learning Pseudometric-based Action Representations for Offline Reinforcement Learning

        **Authors**: *Pengjie Gu, Mengchen Zhao, Chen Chen, Dong Li, Jianye Hao, Bo An*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gu22b.html](https://proceedings.mlr.press/v162/gu22b.html)

        **Abstract**:

        Offline reinforcement learning is a promising approach for practical applications since it does not require interactions with real-world environments. However, existing offline RL methods only work well in environments with continuous or small discrete action spaces. In environments with large and discrete action spaces, such as recommender systems and dialogue systems, the performance of existing methods decreases drastically because they suffer from inaccurate value estimation for a large proportion of out-of-distribution (o.o.d.) actions. While recent works have demonstrated that online RL benefits from incorporating semantic information in action representations, unfortunately, they fail to learn reasonable relative distances between action representations, which is key to offline RL to reduce the influence of o.o.d. actions. This paper proposes an action representation learning framework for offline RL based on a pseudometric, which measures both the behavioral relation and the data-distributional relation between actions. We provide theoretical analysis on the continuity of the expected Q-values and the offline policy improvement using the learned action representations. Experimental results show that our methods significantly improve the performance of two typical offline RL methods in environments with large and discrete action spaces.

        ----

        ## [342] NeuroFluid: Fluid Dynamics Grounding with Particle-Driven Neural Radiance Fields

        **Authors**: *Shanyan Guan, Huayu Deng, Yunbo Wang, Xiaokang Yang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guan22a.html](https://proceedings.mlr.press/v162/guan22a.html)

        **Abstract**:

        Deep learning has shown great potential for modeling the physical dynamics of complex particle systems such as fluids. Existing approaches, however, require the supervision of consecutive particle properties, including positions and velocities. In this paper, we consider a partially observable scenario known as fluid dynamics grounding, that is, inferring the state transitions and interactions within the fluid particle systems from sequential visual observations of the fluid surface. We propose a differentiable two-stage network named NeuroFluid. Our approach consists of (i) a particle-driven neural renderer, which involves fluid physical properties into the volume rendering function, and (ii) a particle transition model optimized to reduce the differences between the rendered and the observed images. NeuroFluid provides the first solution to unsupervised learning of particle-based fluid dynamics by training these two models jointly. It is shown to reasonably estimate the underlying physics of fluids with different initial shapes, viscosity, and densities.

        ----

        ## [343] Fast-Rate PAC-Bayesian Generalization Bounds for Meta-Learning

        **Authors**: *Jiechao Guan, Zhiwu Lu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guan22b.html](https://proceedings.mlr.press/v162/guan22b.html)

        **Abstract**:

        PAC-Bayesian error bounds provide a theoretical guarantee on the generalization abilities of meta-learning from training tasks to unseen tasks. However, it is still unclear how tight PAC-Bayesian bounds we can achieve for meta-learning. In this work, we propose a general PAC-Bayesian framework to cope with single-task learning and meta-learning uniformly. With this framework, we generalize the two tightest PAC-Bayesian bounds (i.e., kl-bound and Catoni-bound) from single-task learning to standard meta-learning, resulting in fast convergence rates for PAC-Bayesian meta-learners. By minimizing the derived two bounds, we develop two meta-learning algorithms for classification problems with deep neural networks. For regression problems, by setting Gibbs optimal posterior for each training task, we obtain the closed-form formula of the minimizer of our Catoni-bound, leading to an efficient Gibbs meta-learning algorithm. Although minimizing our kl-bound can not yield a closed-form solution, we show that it can be extended for analyzing the more challenging meta-learning setting where samples from different training tasks exhibit interdependencies. Experiments empirically show that our proposed meta-learning algorithms achieve competitive results with respect to latest works.

        ----

        ## [344] Leveraging Approximate Symbolic Models for Reinforcement Learning via Skill Diversity

        **Authors**: *Lin Guan, Sarath Sreedharan, Subbarao Kambhampati*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guan22c.html](https://proceedings.mlr.press/v162/guan22c.html)

        **Abstract**:

        Creating reinforcement learning (RL) agents that are capable of accepting and leveraging task-specific knowledge from humans has been long identified as a possible strategy for developing scalable approaches for solving long-horizon problems. While previous works have looked at the possibility of using symbolic models along with RL approaches, they tend to assume that the high-level action models are executable at low level and the fluents can exclusively characterize all desirable MDP states. Symbolic models of real world tasks are however often incomplete. To this end, we introduce Approximate Symbolic-Model Guided Reinforcement Learning, wherein we will formalize the relationship between the symbolic model and the underlying MDP that will allow us to characterize the incompleteness of the symbolic model. We will use these models to extract high-level landmarks that will be used to decompose the task. At the low level, we learn a set of diverse policies for each possible task subgoal identified by the landmark, which are then stitched together. We evaluate our system by testing on three different benchmark domains and show how even with incomplete symbolic model information, our approach is able to discover the task structure and efficiently guide the RL agent towards the goal.

        ----

        ## [345] Large-Scale Graph Neural Architecture Search

        **Authors**: *Chaoyu Guan, Xin Wang, Hong Chen, Ziwei Zhang, Wenwu Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guan22d.html](https://proceedings.mlr.press/v162/guan22d.html)

        **Abstract**:

        Graph Neural Architecture Search (GNAS) has become a powerful method in automatically discovering suitable Graph Neural Network (GNN) architectures for different tasks. However, existing approaches fail to handle large-scale graphs because current performance estimation strategies in GNAS are computationally expensive for large-scale graphs and suffer from consistency collapse issues. To tackle these problems, we propose the Graph ArchitectUre Search at Scale (GAUSS) method that can handle large-scale graphs by designing an efficient light-weight supernet and the joint architecture-graph sampling. In particular, a graph sampling-based single-path one-shot supernet is proposed to reduce the computation burden. To address the consistency collapse issues, we further explicitly consider the joint architecture-graph sampling through a novel architecture peer learning mechanism on the sampled sub-graphs and an architecture importance sampling algorithm. Our proposed framework is able to smooth the highly non-convex optimization objective and stabilize the architecture sampling process. We provide theoretical analyses on GAUSS and empirically evaluate it on five datasets whose vertex sizes range from 10^4 to 10^8. The experimental results demonstrate substantial improvements of GAUSS over other GNAS baselines on all datasets. To the best of our knowledge, the proposed GAUSS method is the first graph neural architecture search framework that can handle graphs with billions of edges within 1 GPU day.

        ----

        ## [346] Identifiability Conditions for Domain Adaptation

        **Authors**: *Ishaan Gulrajani, Tatsunori Hashimoto*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gulrajani22a.html](https://proceedings.mlr.press/v162/gulrajani22a.html)

        **Abstract**:

        Domain adaptation algorithms and theory have relied upon an assumption that the observed data uniquely specify the correct correspondence between the domains. Unfortunately, it is unclear under what conditions this identifiability assumption holds, even when restricting ourselves to the case where a correct bijective map between domains exists. We study this bijective domain mapping problem and provide several new sufficient conditions for the identifiability of linear domain maps. As a consequence of our analysis, we show that weak constraints on the third moment tensor suffice for identifiability, prove identifiability for common latent variable models such as topic models, and give a computationally tractable method for generating certificates for the identifiability of linear maps. Inspired by our certification method, we derive a new objective function for domain mapping that explicitly accounts for uncertainty over maps arising from unidentifiability. We demonstrate that our objective leads to improvements in uncertainty quantification and model performance estimation.

        ----

        ## [347] A Parametric Class of Approximate Gradient Updates for Policy Optimization

        **Authors**: *Ramki Gummadi, Saurabh Kumar, Junfeng Wen, Dale Schuurmans*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gummadi22a.html](https://proceedings.mlr.press/v162/gummadi22a.html)

        **Abstract**:

        Approaches to policy optimization have been motivated from diverse principles, based on how the parametric model is interpreted (e.g. value versus policy representation) or how the learning objective is formulated, yet they share a common goal of maximizing expected return. To better capture the commonalities and identify key differences between policy optimization methods, we develop a unified perspective that re-expresses the underlying updates in terms of a limited choice of gradient form and scaling function. In particular, we identify a parameterized space of approximate gradient updates for policy optimization that is highly structured, yet covers both classical and recent examples, including PPO. 	As a result, we obtain novel yet well motivated updates that generalize existing algorithms in a way that can deliver benefits both in terms of convergence speed and final result quality. An experimental investigation demonstrates that the additional degrees of freedom provided in the parameterized family of updates can be leveraged to obtain non-trivial improvements both in synthetic domains and on popular deep RL benchmarks.

        ----

        ## [348] Provably Efficient Offline Reinforcement Learning for Partially Observable Markov Decision Processes

        **Authors**: *Hongyi Guo, Qi Cai, Yufeng Zhang, Zhuoran Yang, Zhaoran Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22a.html](https://proceedings.mlr.press/v162/guo22a.html)

        **Abstract**:

        We study offline reinforcement learning (RL) for partially observable Markov decision processes (POMDPs) with possibly infinite state and observation spaces. Under the undercompleteness assumption, the optimal policy in such POMDPs are characterized by a class of finite-memory Bellman operators. In the offline setting, estimating these operators directly is challenging due to (i) the large observation space and (ii) insufficient coverage of the offline dataset. To tackle these challenges, we propose a novel algorithm that constructs confidence regions for these Bellman operators via offline estimation of their RKHS embeddings, and returns the final policy via pessimistic planning within the confidence regions. We prove that the proposed algorithm attains an \(\epsilon\)-optimal policy using an offline dataset containing \(\tilde\cO(1 / \epsilon^2)\){episodes}, provided that the behavior policy has good coverage over the optimal trajectory. To our best knowledge, our algorithm is the first provably sample efficient offline algorithm for POMDPs without uniform coverage assumptions.

        ----

        ## [349] No-Regret Learning in Partially-Informed Auctions

        **Authors**: *Wenshuo Guo, Michael I. Jordan, Ellen Vitercik*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22b.html](https://proceedings.mlr.press/v162/guo22b.html)

        **Abstract**:

        Auctions with partially-revealed information about items are broadly employed in real-world applications, but the underlying mechanisms have limited theoretical support. In this work, we study a machine learning formulation of these types of mechanisms, presenting algorithms that are no-regret from the buyer’s perspective. Specifically, a buyer who wishes to maximize his utility interacts repeatedly with a platform over a series of $T$ rounds. In each round, a new item is drawn from an unknown distribution and the platform publishes a price together with incomplete, “masked” information about the item. The buyer then decides whether to purchase the item. We formalize this problem as an online learning task where the goal is to have low regret with respect to a myopic oracle that has perfect knowledge of the distribution over items and the seller’s masking function. When the distribution over items is known to the buyer and the mask is a SimHash function mapping $\R^d$ to $\{0,1\}^{\ell}$, our algorithm has regret $\tilde \cO((Td\ell)^{\nicefrac{1}{2}})$. In a fully agnostic setting when the mask is an arbitrary function mapping to a set of size $n$ and the prices are stochastic, our algorithm has regret $\tilde \cO((Tn)^{\nicefrac{1}{2}})$.

        ----

        ## [350] Bounding Training Data Reconstruction in Private (Deep) Learning

        **Authors**: *Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22c.html](https://proceedings.mlr.press/v162/guo22c.html)

        **Abstract**:

        Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary’s capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods—Renyi differential privacy and Fisher information leakage—both offer strong semantic protection against data reconstruction attacks.

        ----

        ## [351] Adversarially trained neural representations are already as robust as biological neural representations

        **Authors**: *Chong Guo, Michael J. Lee, Guillaume Leclerc, Joel Dapello, Yug Rao, Aleksander Madry, James J. DiCarlo*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22d.html](https://proceedings.mlr.press/v162/guo22d.html)

        **Abstract**:

        Visual systems of primates are the gold standard of robust perception. There is thus a general belief that mimicking the neural representations that underlie those systems will yield artificial visual systems that are adversarially robust. In this work, we develop a method for performing adversarial visual attacks directly on primate brain activity. We then leverage this method to demonstrate that the above-mentioned belief might not be well-founded. Specifically, we report that the biological neurons that make up visual systems of primates exhibit susceptibility to adversarial perturbations that is comparable in magnitude to existing (robustly trained) artificial neural networks.

        ----

        ## [352] Class-Imbalanced Semi-Supervised Learning with Adaptive Thresholding

        **Authors**: *Lan-Zhe Guo, Yu-Feng Li*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22e.html](https://proceedings.mlr.press/v162/guo22e.html)

        **Abstract**:

        Semi-supervised learning (SSL) has proven to be successful in overcoming labeling difficulties by leveraging unlabeled data. Previous SSL algorithms typically assume a balanced class distribution. However, real-world datasets are usually class-imbalanced, causing the performance of existing SSL algorithms to be seriously decreased. One essential reason is that pseudo-labels for unlabeled data are selected based on a fixed confidence threshold, resulting in low performance on minority classes. In this paper, we develop a simple yet effective framework, which only involves adaptive thresholding for different classes in SSL algorithms, and achieves remarkable performance improvement on more than twenty imbalance ratios. Specifically, we explicitly optimize the number of pseudo-labels for each class in the SSL objective, so as to simultaneously obtain adaptive thresholds and minimize empirical risk. Moreover, the determination of the adaptive threshold can be efficiently obtained by a closed-form solution. Extensive experimental results demonstrate the effectiveness of our proposed algorithms.

        ----

        ## [353] Deep Squared Euclidean Approximation to the Levenshtein Distance for DNA Storage

        **Authors**: *Alan J. X. Guo, Cong Liang, Qing-Hu Hou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22f.html](https://proceedings.mlr.press/v162/guo22f.html)

        **Abstract**:

        Storing information in DNA molecules is of great interest because of its advantages in longevity, high storage density, and low maintenance cost. A key step in the DNA storage pipeline is to efficiently cluster the retrieved DNA sequences according to their similarities. Levenshtein distance is the most suitable metric on the similarity between two DNA sequences, but it is inferior in terms of computational complexity and less compatible with mature clustering algorithms. In this work, we propose a novel deep squared Euclidean embedding for DNA sequences using Siamese neural network, squared Euclidean embedding, and chi-squared regression. The Levenshtein distance is approximated by the squared Euclidean distance between the embedding vectors, which is fast calculated and clustering algorithm friendly. The proposed approach is analyzed theoretically and experimentally. The results show that the proposed embedding is efficient and robust.

        ----

        ## [354] Online Continual Learning through Mutual Information Maximization

        **Authors**: *Yiduo Guo, Bing Liu, Dongyan Zhao*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22g.html](https://proceedings.mlr.press/v162/guo22g.html)

        **Abstract**:

        This paper proposed a new online continual learning approach called OCM based on mutual information (MI) maximization. It achieves two objectives that are critical in dealing with catastrophic forgetting (CF). (1) It reduces feature bias caused by cross entropy (CE) as CE learns only discriminative features for each task, but these features may not be discriminative for another task. To learn a new task well, the network parameters learned before have to be modified, which causes CF. The new approach encourages the learning of each task to make use of the full features of the task training data. (2) It encourages preservation of the previously learned knowledge when training a new batch of incrementally arriving data. Empirical evaluation shows that OCM substantially outperforms the latest online CL baselines. For example, for CIFAR10, OCM improves the accuracy of the best baseline by 13.1% from 64.1% (baseline) to 77.2% (OCM).The code is publicly available at https://github.com/gydpku/OCM.

        ----

        ## [355] Fast Provably Robust Decision Trees and Boosting

        **Authors**: *Jun-Qi Guo, Ming-Zhuo Teng, Wei Gao, Zhi-Hua Zhou*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22h.html](https://proceedings.mlr.press/v162/guo22h.html)

        **Abstract**:

        Learning with adversarial robustness has been a challenge in contemporary machine learning, and recent years have witnessed increasing attention on robust decision trees and ensembles, mostly working with high computational complexity or without guarantees of provable robustness. This work proposes the Fast Provably Robust Decision Tree (FPRDT) with the smallest computational complexity O(n log n), a tradeoff between global and local optimizations over the adversarial 0/1 loss. We further develop the Provably Robust AdaBoost (PRAdaBoost) according to our robust decision trees, and present convergence analysis for training adversarial 0/1 loss. We conduct extensive experiments to support our approaches; in particular, our approaches are superior to those unprovably robust methods, and achieve better or comparable performance to those provably robust methods yet with the smallest running time.

        ----

        ## [356] Understanding and Improving Knowledge Graph Embedding for Entity Alignment

        **Authors**: *Lingbing Guo, Qiang Zhang, Zequn Sun, Mingyang Chen, Wei Hu, Huajun Chen*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/guo22i.html](https://proceedings.mlr.press/v162/guo22i.html)

        **Abstract**:

        Embedding-based entity alignment (EEA) has recently received great attention. Despite significant performance improvement, few efforts have been paid to facilitate understanding of EEA methods. Most existing studies rest on the assumption that a small number of pre-aligned entities can serve as anchors connecting the embedding spaces of two KGs. Nevertheless, no one has investigated the rationality of such an assumption. To fill the research gap, we define a typical paradigm abstracted from existing EEA methods and analyze how the embedding discrepancy between two potentially aligned entities is implicitly bounded by a predefined margin in the score function. Further, we find that such a bound cannot guarantee to be tight enough for alignment learning. We mitigate this problem by proposing a new approach, named NeoEA, to explicitly learn KG-invariant and principled entity embeddings. In this sense, an EEA model not only pursues the closeness of aligned entities based on geometric distance, but also aligns the neural ontologies of two KGs by eliminating the discrepancy in embedding distribution and underlying ontology knowledge. Our experiments demonstrate consistent and significant performance improvement against the best-performing EEA methods.

        ----

        ## [357] NISPA: Neuro-Inspired Stability-Plasticity Adaptation for Continual Learning in Sparse Networks

        **Authors**: *Mustafa Burak Gurbuz, Constantine Dovrolis*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/gurbuz22a.html](https://proceedings.mlr.press/v162/gurbuz22a.html)

        **Abstract**:

        The goal of continual learning (CL) is to learn different tasks over time. The main desiderata associated with CL are to maintain performance on older tasks, leverage the latter to improve learning of future tasks, and to introduce minimal overhead in the training process (for instance, to not require a growing model or retraining). We propose the Neuro-Inspired Stability-Plasticity Adaptation (NISPA) architecture that addresses these desiderata through a sparse neural network with fixed density. NISPA forms stable paths to preserve learned knowledge from older tasks. Also, NISPA uses connection rewiring to create new plastic paths that reuse existing knowledge on novel tasks. Our extensive evaluation on EMNIST, FashionMNIST, CIFAR10, and CIFAR100 datasets shows that NISPA significantly outperforms representative state-of-the-art continual learning baselines, and it uses up to ten times fewer learnable parameters compared to baselines. We also make the case that sparsity is an essential ingredient for continual learning. The NISPA code is available at https://github.com/BurakGurbuz97/NISPA.

        ----

        ## [358] Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets

        **Authors**: *Guy Hacohen, Avihu Dekel, Daphna Weinshall*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hacohen22a.html](https://proceedings.mlr.press/v162/hacohen22a.html)

        **Abstract**:

        Investigating active learning, we focus on the relation between the number of labeled examples (budget size), and suitable querying strategies. Our theoretical analysis shows a behavior reminiscent of phase transition: typical examples are best queried when the budget is low, while unrepresentative examples are best queried when the budget is large. Combined evidence shows that a similar phenomenon occurs in common classification models. Accordingly, we propose TypiClust – a deep active learning strategy suited for low budgets. In a comparative empirical investigation of supervised learning, using a variety of architectures and image datasets, TypiClust outperforms all other active learning strategies in the low-budget regime. Using TypiClust in the semi-supervised framework, performance gets an even more significant boost. In particular, state-of-the-art semi-supervised methods trained on CIFAR-10 with 10 labeled examples selected by TypiClust, reach 93.2% accuracy – an improvement of 39.4% over random selection. Code is available at https://github.com/avihu111/TypiClust.

        ----

        ## [359] You Only Cut Once: Boosting Data Augmentation with a Single Cut

        **Authors**: *Junlin Han, Pengfei Fang, Weihao Li, Jie Hong, Mohammad Ali Armin, Ian D. Reid, Lars Petersson, Hongdong Li*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22a.html](https://proceedings.mlr.press/v162/han22a.html)

        **Abstract**:

        We present You Only Cut Once (YOCO) for performing data augmentations. YOCO cuts one image into two pieces and performs data augmentations individually within each piece. Applying YOCO improves the diversity of the augmentation per sample and encourages neural networks to recognize objects from partial information. YOCO enjoys the properties of parameter-free, easy usage, and boosting almost all augmentations for free. Thorough experiments are conducted to evaluate its effectiveness. We first demonstrate that YOCO can be seamlessly applied to varying data augmentations, neural network architectures, and brings performance gains on CIFAR and ImageNet classification tasks, sometimes surpassing conventional image-level augmentation by large margins. Moreover, we show YOCO benefits contrastive pre-training toward a more powerful representation that can be better transferred to multiple downstream tasks. Finally, we study a number of variants of YOCO and empirically analyze the performance for respective settings.

        ----

        ## [360] Scalable MCMC Sampling for Nonsymmetric Determinantal Point Processes

        **Authors**: *Insu Han, Mike Gartrell, Elvis Dohmatob, Amin Karbasi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22b.html](https://proceedings.mlr.press/v162/han22b.html)

        **Abstract**:

        A determinantal point process (DPP) is an elegant model that assigns a probability to every subset of a collection of $n$ items. While conventionally a DPP is parameterized by a symmetric kernel matrix, removing this symmetry constraint, resulting in nonsymmetric DPPs (NDPPs), leads to significant improvements in modeling power and predictive performance. Recent work has studied an approximate Markov chain Monte Carlo (MCMC) sampling algorithm for NDPPs restricted to size-$k$ subsets (called $k$-NDPPs). However, the runtime of this approach is quadratic in $n$, making it infeasible for large-scale settings. In this work, we develop a scalable MCMC sampling algorithm for $k$-NDPPs with low-rank kernels, thus enabling runtime that is sublinear in $n$. Our method is based on a state-of-the-art NDPP rejection sampling algorithm, which we enhance with a novel approach for efficiently constructing the proposal distribution. Furthermore, we extend our scalable $k$-NDPP sampling algorithm to NDPPs without size constraints. Our resulting sampling method has polynomial time complexity in the rank of the kernel, while the existing approach has runtime that is exponential in the rank. With both a theoretical analysis and experiments on real-world datasets, we verify that our scalable approximate sampling algorithms are orders of magnitude faster than existing sampling approaches for $k$-NDPPs and NDPPs.

        ----

        ## [361] G-Mixup: Graph Data Augmentation for Graph Classification

        **Authors**: *Xiaotian Han, Zhimeng Jiang, Ninghao Liu, Xia Hu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22c.html](https://proceedings.mlr.press/v162/han22c.html)

        **Abstract**:

        This work develops mixup for graph data. Mixup has shown superiority in improving the generalization and robustness of neural networks by interpolating features and labels between two random samples. Traditionally, Mixup can work on regular, grid-like, and Euclidean data such as image or tabular data. However, it is challenging to directly adopt Mixup to augment graph data because different graphs typically: 1) have different numbers of nodes; 2) are not readily aligned; and 3) have unique typologies in non-Euclidean space. To this end, we propose G-Mixup to augment graphs for graph classification by interpolating the generator (i.e., graphon) of different classes of graphs. Specifically, we first use graphs within the same class to estimate a graphon. Then, instead of directly manipulating graphs, we interpolate graphons of different classes in the Euclidean space to get mixed graphons, where the synthetic graphs are generated through sampling based on the mixed graphons. Extensive experiments show that G-Mixup substantially improves the generalization and robustness of GNNs.

        ----

        ## [362] Private Streaming SCO in ℓp geometry with Applications in High Dimensional Online Decision Making

        **Authors**: *Yuxuan Han, Zhicong Liang, Zhipeng Liang, Yang Wang, Yuan Yao, Jiheng Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22d.html](https://proceedings.mlr.press/v162/han22d.html)

        **Abstract**:

        Differentially private (DP) stochastic convex optimization (SCO) is ubiquitous in trustworthy machine learning algorithm design. This paper studies the DP-SCO problem with streaming data sampled from a distribution and arrives sequentially. We also consider the continual release model where parameters related to private information are updated and released upon each new data. Numerous algorithms have been developed to achieve optimal excess risks in different $\ell_p$ norm geometries, but none of the existing ones can be adapted to the streaming and continual release setting. We propose a private variant of the Frank-Wolfe algorithm with recursive gradients for variance reduction to update and reveal the parameters upon each data. Combined with the adaptive DP analysis, our algorithm achieves the first optimal excess risk in linear time in the case $1
Cite this Paper



    BibTeX
  



@InProceedings{pmlr-v162-han22d,
  title = 	 {Private Streaming {SCO} in $\ell_p$ geometry with Applications in High Dimensional Online Decision Making},
  author =       {Han, Yuxuan and Liang, Zhicong and Liang, Zhipeng and Wang, Yang and Yao, Yuan and Zhang, Jiheng},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {8249--8279},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/han22d/han22d.pdf},
  url = 	 {https://proceedings.mlr.press/v162/han22d.html},
  abstract = 	 {Differentially private (DP) stochastic convex optimization (SCO) is ubiquitous in trustworthy machine learning algorithm design. This paper studies the DP-SCO problem with streaming data sampled from a distribution and arrives sequentially. We also consider the continual release model where parameters related to private information are updated and released upon each new data. Numerous algorithms have been developed to achieve optimal excess risks in different $\ell_p$ norm geometries, but none of the existing ones can be adapted to the streaming and continual release setting. We propose a private variant of the Frank-Wolfe algorithm with recursive gradients for variance reduction to update and reveal the parameters upon each data. Combined with the adaptive DP analysis, our algorithm achieves the first optimal excess risk in linear time in the case $1

Copy to Clipboard
Download




    Endnote
  


%0 Conference Paper
%T Private Streaming SCO in $\ell_p$ geometry with Applications in High Dimensional Online Decision Making
%A Yuxuan Han
%A Zhicong Liang
%A Zhipeng Liang
%A Yang Wang
%A Yuan Yao
%A Jiheng Zhang
%B Proceedings of the 39th International Conference on Machine Learning
%C Proceedings of Machine Learning Research
%D 2022
%E Kamalika Chaudhuri
%E Stefanie Jegelka
%E Le Song
%E Csaba Szepesvari
%E Gang Niu
%E Sivan Sabato	
%F pmlr-v162-han22d
%I PMLR
%P 8249--8279
%U https://proceedings.mlr.press/v162/han22d.html
%V 162
%X Differentially private (DP) stochastic convex optimization (SCO) is ubiquitous in trustworthy machine learning algorithm design. This paper studies the DP-SCO problem with streaming data sampled from a distribution and arrives sequentially. We also consider the continual release model where parameters related to private information are updated and released upon each new data. Numerous algorithms have been developed to achieve optimal excess risks in different $\ell_p$ norm geometries, but none of the existing ones can be adapted to the streaming and continual release setting. We propose a private variant of the Frank-Wolfe algorithm with recursive gradients for variance reduction to update and reveal the parameters upon each data. Combined with the adaptive DP analysis, our algorithm achieves the first optimal excess risk in linear time in the case $1

Copy to Clipboard
Download




    APA
  



Han, Y., Liang, Z., Liang, Z., Wang, Y., Yao, Y. & Zhang, J.. (2022). Private Streaming SCO in $\ell_p$ geometry with Applications in High Dimensional Online Decision Making. Proceedings of the 39th International Conference on Machine Learning, in Proceedings of Machine Learning Research 162:8249-8279 Available from https://proceedings.mlr.press/v162/han22d.html.



Copy to Clipboard
Download



Related Material


Download PDF

        ----

        ## [363] Off-Policy Reinforcement Learning with Delayed Rewards

        **Authors**: *Beining Han, Zhizhou Ren, Zuofan Wu, Yuan Zhou, Jian Peng*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22e.html](https://proceedings.mlr.press/v162/han22e.html)

        **Abstract**:

        We study deep reinforcement learning (RL) algorithms with delayed rewards. In many real-world tasks, instant rewards are often not readily accessible or even defined immediately after the agent performs actions. In this work, we first formally define the environment with delayed rewards and discuss the challenges raised due to the non-Markovian nature of such environments. Then, we introduce a general off-policy RL framework with a new Q-function formulation that can handle the delayed rewards with theoretical convergence guarantees. For practical tasks with high dimensional state spaces, we further introduce the HC-decomposition rule of the Q-function in our framework which naturally leads to an approximation scheme that helps boost the training efficiency and stability. We finally conduct extensive experiments to demonstrate the superior performance of our algorithms over the existing work and their variants.

        ----

        ## [364] Adversarial Attacks on Gaussian Process Bandits

        **Authors**: *Eric Han, Jonathan Scarlett*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22f.html](https://proceedings.mlr.press/v162/han22f.html)

        **Abstract**:

        Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker’s perspective, proposing various adversarial attack methods with differing assumptions on the attacker’s strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function f to produce another function f  whose optima are in some target region. Based on our theoretical analysis, we devise both white-box attacks (known f) and black-box attacks (unknown f), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards the target region even with a low attack budget, and we test our attacks’ effectiveness on a diverse range of objective functions.

        ----

        ## [365] Random Gegenbauer Features for Scalable Kernel Methods

        **Authors**: *Insu Han, Amir Zandieh, Haim Avron*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/han22g.html](https://proceedings.mlr.press/v162/han22g.html)

        **Abstract**:

        We propose efficient random features for approximating a new and rich class of kernel functions that we refer to as Generalized Zonal Kernels (GZK). Our proposed GZK family, generalizes the zonal kernels (i.e., dot-product kernels on the unit sphere) by introducing radial factors in the Gegenbauer series expansion of these kernel functions. The GZK class of kernels includes a wide range of ubiquitous kernel functions such as the entirety of dot-product kernels as well as the Gaussian and the recently introduced Neural Tangent kernels. Interestingly, by exploiting the reproducing property of the Gegenbauer (Zonal) Harmonics, we can construct efficient random features for the GZK family based on randomly oriented Gegenbauer harmonics. We prove subspace embedding guarantees for our Gegenbauer features which ensures that our features can be used for approximately solving learning problems such as kernel k-means clustering, kernel ridge regression, etc. Empirical results show that our proposed features outperform recent kernel approximation methods.

        ----

        ## [366] Stochastic Reweighted Gradient Descent

        **Authors**: *Ayoub El Hanchi, David A. Stephens, Chris J. Maddison*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hanchi22a.html](https://proceedings.mlr.press/v162/hanchi22a.html)

        **Abstract**:

        Importance sampling is a promising strategy for improving the convergence rate of stochastic gradient methods. It is typically used to precondition the optimization problem, but it can also be used to reduce the variance of the gradient estimator. Unfortunately, this latter point of view has yet to lead to practical methods that provably improve the asymptotic error of stochastic gradient methods. In this work, we propose stochastic reweighted gradient descent (SRG), a stochastic gradient method based solely on importance sampling that can reduce the variance of the gradient estimator and improve on the asymptotic error of stochastic gradient descent (SGD) in the strongly convex and smooth case. We show that SRG can be extended to combine the benefits of both importance-sampling-based preconditioning and variance reduction. When compared to SGD, the resulting algorithm can simultaneously reduce the condition number and the asymptotic error, both by up to a factor equal to the number of component functions. We demonstrate improved convergence in practice on regularized logistic regression problems.

        ----

        ## [367] Dual Perspective of Label-Specific Feature Learning for Multi-Label Classification

        **Authors**: *Jun-Yi Hang, Min-Ling Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hang22a.html](https://proceedings.mlr.press/v162/hang22a.html)

        **Abstract**:

        Label-specific features serve as an effective strategy to facilitate multi-label classification, which account for the distinct discriminative properties of each class label via tailoring its own features. Existing approaches implement this strategy in a quite straightforward way, i.e. finding the most pertinent and discriminative features for each class label and directly inducing classifiers on constructed label-specific features. In this paper, we propose a dual perspective for label-specific feature learning, where label-specific discriminative properties are considered by identifying each label’s own non-informative features and making the discrimination process immutable to variations of these features. To instantiate it, we present a perturbation-based approach DELA to provide classifiers with label-specific immutability on simultaneously identified non-informative features, which is optimized towards a probabilistically-relaxed expected risk minimization problem. Comprehensive experiments on 10 benchmark data sets show that our approach outperforms the state-of-the-art counterparts.

        ----

        ## [368] Temporal Difference Learning for Model Predictive Control

        **Authors**: *Nicklas Hansen, Hao Su, Xiaolong Wang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hansen22a.html](https://proceedings.mlr.press/v162/hansen22a.html)

        **Abstract**:

        Data-driven model predictive control has two key advantages over model-free methods: a potential for improved sample efficiency through model learning, and better performance as computational budget for planning increases. However, it is both costly to plan over long horizons and challenging to obtain an accurate model of the environment. In this work, we combine the strengths of model-free and model-based methods. We use a learned task-oriented latent dynamics model for local trajectory optimization over a short horizon, and use a learned terminal value function to estimate long-term return, both of which are learned jointly by temporal difference learning. Our method, TD-MPC, achieves superior sample efficiency and asymptotic performance over prior work on both state and image-based continuous control tasks from DMControl and Meta-World. Code and videos are available at https://nicklashansen.github.io/td-mpc.

        ----

        ## [369] Bisimulation Makes Analogies in Goal-Conditioned Reinforcement Learning

        **Authors**: *Philippe Hansen-Estruch, Amy Zhang, Ashvin Nair, Patrick Yin, Sergey Levine*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hansen-estruch22a.html](https://proceedings.mlr.press/v162/hansen-estruch22a.html)

        **Abstract**:

        Building generalizable goal-conditioned agents from rich observations is a key to reinforcement learning (RL) solving real world problems. Traditionally in goal-conditioned RL, an agent is provided with the exact goal they intend to reach. However, it is often not realistic to know the configuration of the goal before performing a task. A more scalable framework would allow us to provide the agent with an example of an analogous task, and have the agent then infer what the goal should be for its current state. We propose a new form of state abstraction called goal-conditioned bisimulation that captures functional equivariance, allowing for the reuse of skills to achieve new goals. We learn this representation using a metric form of this abstraction, and show its ability to generalize to new goals in real world manipulation tasks. Further, we prove that this learned representation is sufficient not only for goal-conditioned tasks, but is amenable to any downstream task described by a state-only reward function.

        ----

        ## [370] TURF: Two-Factor, Universal, Robust, Fast Distribution Learning Algorithm

        **Authors**: *Yi Hao, Ayush Jain, Alon Orlitsky, Vaishakh Ravindrakumar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hao22a.html](https://proceedings.mlr.press/v162/hao22a.html)

        **Abstract**:

        Approximating distributions from their samples is a canonical statistical-learning problem. One of its most powerful and successful modalities approximates every distribution to an $\ell_1$ distance essentially at most a constant times larger than its closest $t$-piece degree-$d$ polynomial, where $t\ge1$ and $d\ge0$. Letting $c_{t,d}$ denote the smallest such factor, clearly $c_{1,0}=1$, and it can be shown that $c_{t,d}\ge 2$ for all other $t$ and $d$. Yet current computationally efficient algorithms show only $c_{t,1}\le 2.25$ and the bound rises quickly to $c_{t,d}\le 3$ for $d\ge 9$. We derive a near-linear-time and essentially sample-optimal estimator that establishes $c_{t,d}=2$ for all $(t,d)\ne(1,0)$. Additionally, for many practical distributions, the lowest approximation distance is achieved by polynomials with vastly varying number of pieces. We provide a method that estimates this number near-optimally, hence helps approach the best possible approximation. Experiments combining the two techniques confirm improved performance over existing methodologies.

        ----

        ## [371] Contextual Information-Directed Sampling

        **Authors**: *Botao Hao, Tor Lattimore, Chao Qin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hao22b.html](https://proceedings.mlr.press/v162/hao22b.html)

        **Abstract**:

        Information-directed sampling (IDS) has recently demonstrated its potential as a data-efficient reinforcement learning algorithm. However, it is still unclear what is the right form of information ratio to optimize when contextual information is available. We investigate the IDS design through two contextual bandit problems: contextual bandits with graph feedback and sparse linear contextual bandits. We provably demonstrate the advantage of contextual IDS over conditional IDS and emphasize the importance of considering the context distribution. The main message is that an intelligent agent should invest more on the actions that are beneficial for the future unseen contexts while the conditional IDS can be myopic. We further propose a computationally-efficient version of contextual IDS based on Actor-Critic and evaluate it empirically on a neural network contextual bandit.

        ----

        ## [372] GSmooth: Certified Robustness against Semantic Transformations via Generalized Randomized Smoothing

        **Authors**: *Zhongkai Hao, Chengyang Ying, Yinpeng Dong, Hang Su, Jian Song, Jun Zhu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hao22c.html](https://proceedings.mlr.press/v162/hao22c.html)

        **Abstract**:

        Certified defenses such as randomized smoothing have shown promise towards building reliable machine learning systems against $\ell_p$ norm bounded attacks. However, existing methods are insufficient or unable to provably defend against semantic transformations, especially those without closed-form expressions (such as defocus blur and pixelate), which are more common in practice and often unrestricted. To fill up this gap, we propose generalized randomized smoothing (GSmooth), a unified theoretical framework for certifying robustness against general semantic transformations via a novel dimension augmentation strategy. Under the GSmooth framework, we present a scalable algorithm that uses a surrogate image-to-image network to approximate the complex transformation. The surrogate model provides a powerful tool for studying the properties of semantic transformations and certifying robustness. Experimental results on several datasets demonstrate the effectiveness of our approach for robustness certification against multiple kinds of semantic transformations and corruptions, which is not achievable by the alternative baselines.

        ----

        ## [373] Implicit Regularization with Polynomial Growth in Deep Tensor Factorization

        **Authors**: *Kais Hariz, Hachem Kadri, Stéphane Ayache, Maher Moakher, Thierry Artières*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hariz22a.html](https://proceedings.mlr.press/v162/hariz22a.html)

        **Abstract**:

        We study the implicit regularization effects of deep learning in tensor factorization. While implicit regularization in deep matrix and ’shallow’ tensor factorization via linear and certain type of non-linear neural networks promotes low-rank solutions with at most quadratic growth, we show that its effect in deep tensor factorization grows polynomially with the depth of the network. This provides a remarkably faithful description of the observed experimental behaviour. Using numerical experiments, we demonstrate the benefits of this implicit regularization in yielding a more accurate estimation and better convergence properties.

        ----

        ## [374] Strategic Instrumental Variable Regression: Recovering Causal Relationships From Strategic Responses

        **Authors**: *Keegan Harris, Dung Daniel T. Ngo, Logan Stapleton, Hoda Heidari, Steven Wu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/harris22a.html](https://proceedings.mlr.press/v162/harris22a.html)

        **Abstract**:

        In settings where Machine Learning (ML) algorithms automate or inform consequential decisions about people, individual decision subjects are often incentivized to strategically modify their observable attributes to receive more favorable predictions. As a result, the distribution the assessment rule is trained on may differ from the one it operates on in deployment. While such distribution shifts, in general, can hinder accurate predictions, our work identifies a unique opportunity associated with shifts due to strategic responses: We show that we can use strategic responses effectively to recover causal relationships between the observable features and outcomes we wish to predict, even under the presence of unobserved confounding variables. Specifically, our work establishes a novel connection between strategic responses to ML models and instrumental variable (IV) regression by observing that the sequence of deployed models can be viewed as an instrument that affects agents’ observable features but does not directly influence their outcomes. We show that our causal recovery method can be utilized to improve decision-making across several important criteria: individual fairness, agent outcomes, and predictive risk. In particular, we show that if decision subjects differ in their ability to modify non-causal attributes, any decision rule deviating from the causal coefficients can lead to (potentially unbounded) individual-level unfairness. .

        ----

        ## [375] C*-algebra Net: A New Approach Generalizing Neural Network Parameters to C*-algebra

        **Authors**: *Yuka Hashimoto, Zhao Wang, Tomoko Matsui*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hashimoto22a.html](https://proceedings.mlr.press/v162/hashimoto22a.html)

        **Abstract**:

        We propose a new framework that generalizes the parameters of neural network models to $C^*$-algebra-valued ones. $C^*$-algebra is a generalization of the space of complex numbers. A typical example is the space of continuous functions on a compact space. This generalization enables us to combine multiple models continuously and use tools for functions such as regression and integration. Consequently, we can learn features of data efficiently and adapt the models to problems continuously. We apply our framework to practical problems such as density estimation and few-shot learning and show that our framework enables us to learn features of data even with a limited number of samples. Our new framework highlights the potential possibility of applying the theory of $C^*$-algebra to general neural network models.

        ----

        ## [376] General-purpose, long-context autoregressive modeling with Perceiver AR

        **Authors**: *Curtis Hawthorne, Andrew Jaegle, Catalina Cangea, Sebastian Borgeaud, Charlie Nash, Mateusz Malinowski, Sander Dieleman, Oriol Vinyals, Matthew M. Botvinick, Ian Simon, Hannah Sheahan, Neil Zeghidour, Jean-Baptiste Alayrac, João Carreira, Jesse H. Engel*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hawthorne22a.html](https://proceedings.mlr.press/v162/hawthorne22a.html)

        **Abstract**:

        Real-world data is high-dimensional: a book, image, or musical performance can easily contain hundreds of thousands of elements even after compression. However, the most commonly used autoregressive models, Transformers, are prohibitively expensive to scale to the number of inputs and layers needed to capture this long-range structure. We develop Perceiver AR, an autoregressive, modality-agnostic architecture which uses cross-attention to map long-range inputs to a small number of latents while also maintaining end-to-end causal masking. Perceiver AR can directly attend to over a hundred thousand tokens, enabling practical long-context density estimation without the need for hand-crafted sparsity patterns or memory mechanisms. When trained on images or music, Perceiver AR generates outputs with clear long-term coherence and structure. Our architecture also obtains state-of-the-art likelihood on long-sequence benchmarks, including 64x64 ImageNet images and PG-19 books.

        ----

        ## [377] On Distribution Shift in Learning-based Bug Detectors

        **Authors**: *Jingxuan He, Luca Beurer-Kellner, Martin T. Vechev*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/he22a.html](https://proceedings.mlr.press/v162/he22a.html)

        **Abstract**:

        Deep learning has recently achieved initial success in program analysis tasks such as bug detection. Lacking real bugs, most existing works construct training and test data by injecting synthetic bugs into correct programs. Despite achieving high test accuracy (e.g., >90%), the resulting bug detectors are found to be surprisingly unusable in practice, i.e., <10% precision when used to scan real software repositories. In this work, we argue that this massive performance difference is caused by a distribution shift, i.e., a fundamental mismatch between the real bug distribution and the synthetic bug distribution used to train and evaluate the detectors. To address this key challenge, we propose to train a bug detector in two phases, first on a synthetic bug distribution to adapt the model to the bug detection domain, and then on a real bug distribution to drive the model towards the real distribution. During these two phases, we leverage a multi-task hierarchy, focal loss, and contrastive learning to further boost performance. We evaluate our approach extensively on three widely studied bug types, for which we construct new datasets carefully designed to capture the real bug distribution. The results demonstrate that our approach is practically effective and successfully mitigates the distribution shift: our learned detectors are highly performant on both our test set and the latest version of open source repositories. Our code, datasets, and models are publicly available at https://github.com/eth-sri/learning-real-bug-detector.

        ----

        ## [378] GNNRank: Learning Global Rankings from Pairwise Comparisons via Directed Graph Neural Networks

        **Authors**: *Yixuan He, Quan Gan, David Wipf, Gesine D. Reinert, Junchi Yan, Mihai Cucuringu*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/he22b.html](https://proceedings.mlr.press/v162/he22b.html)

        **Abstract**:

        Recovering global rankings from pairwise comparisons has wide applications from time synchronization to sports team ranking. Pairwise comparisons corresponding to matches in a competition can be construed as edges in a directed graph (digraph), whose nodes represent e.g. competitors with an unknown rank. In this paper, we introduce neural networks into the ranking recovery problem by proposing the so-called GNNRank, a trainable GNN-based framework with digraph embedding. Moreover, new objectives are devised to encode ranking upsets/violations. The framework involves a ranking score estimation approach, and adds an inductive bias by unfolding the Fiedler vector computation of the graph constructed from a learnable similarity matrix. Experimental results on extensive data sets show that our methods attain competitive and often superior performance against baselines, as well as showing promising transfer ability. Codes and preprocessed data are at: \url{https://github.com/SherylHYX/GNNRank}.

        ----

        ## [379] Exploring the Gap between Collapsed & Whitened Features in Self-Supervised Learning

        **Authors**: *Bobby He, Mete Ozay*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/he22c.html](https://proceedings.mlr.press/v162/he22c.html)

        **Abstract**:

        Avoiding feature collapse, when a Neural Network (NN) encoder maps all inputs to a constant vector, is a shared implicit desideratum of various methodological advances in self-supervised learning (SSL). To that end, whitened features have been proposed as an explicit objective to ensure uncollapsed features \cite{zbontar2021barlow,ermolov2021whitening,hua2021feature,bardes2022vicreg}. We identify power law behaviour in eigenvalue decay, parameterised by exponent $\beta{\geq}0$, as a spectrum that bridges between the collapsed & whitened feature extremes. We provide theoretical & empirical evidence highlighting the factors in SSL, like projection layers & regularisation strength, that influence eigenvalue decay rate, & demonstrate that the degree of feature whitening affects generalisation, particularly in label scarce regimes. We use our insights to motivate a novel method, PMP (PostMan-Pat), which efficiently post-processes a pretrained encoder to enforce eigenvalue decay rate with power law exponent $\beta$, & find that PostMan-Pat delivers improved label efficiency and transferability across a range of SSL methods and encoder architectures.

        ----

        ## [380] Sparse Double Descent: Where Network Pruning Aggravates Overfitting

        **Authors**: *Zheng He, Zeke Xie, Quanzhi Zhu, Zengchang Qin*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/he22d.html](https://proceedings.mlr.press/v162/he22d.html)

        **Abstract**:

        People usually believe that network pruning not only reduces the computational cost of deep networks, but also prevents overfitting by decreasing model capacity. However, our work surprisingly discovers that network pruning sometimes even aggravates overfitting. We report an unexpected sparse double descent phenomenon that, as we increase model sparsity via network pruning, test performance first gets worse (due to overfitting), then gets better (due to relieved overfitting), and gets worse at last (due to forgetting useful information). While recent studies focused on the deep double descent with respect to model overparameterization, they failed to recognize that sparsity may also cause double descent. In this paper, we have three main contributions. First, we report the novel sparse double descent phenomenon through extensive experiments. Second, for this phenomenon, we propose a novel learning distance interpretation that the curve of l2 learning distance of sparse models (from initialized parameters to final parameters) may correlate with the sparse double descent curve well and reflect generalization better than minima flatness. Third, in the context of sparse double descent, a winning ticket in the lottery ticket hypothesis surprisingly may not always win.

        ----

        ## [381] A Reduction from Linear Contextual Bandit Lower Bounds to Estimation Lower Bounds

        **Authors**: *Jiahao He, Jiheng Zhang, Rachel Q. Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/he22e.html](https://proceedings.mlr.press/v162/he22e.html)

        **Abstract**:

        Linear contextual bandits and their variants are usually solved using algorithms guided by parameter estimation. Cauchy-Schwartz inequality established that estimation errors dominate algorithm regrets, and thus, accurate estimators suffice to guarantee algorithms with low regrets. In this paper, we complete the reverse direction by establishing the necessity. In particular, we provide a generic transformation from algorithms for linear contextual bandits to estimators for linear models, and show that algorithm regrets dominate estimation errors of their induced estimators, i.e., low-regret algorithms must imply accurate estimators. Moreover, our analysis reduces the regret lower bound to an estimation error, bridging the lower bound analysis in linear contextual bandit problems and linear regression.

        ----

        ## [382] HyperPrompt: Prompt-based Task-Conditioning of Transformers

        **Authors**: *Yun He, Huaixiu Steven Zheng, Yi Tay, Jai Prakash Gupta, Yu Du, Vamsi Aribandi, Zhe Zhao, YaGuang Li, Zhao Chen, Donald Metzler, Heng-Tze Cheng, Ed H. Chi*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/he22f.html](https://proceedings.mlr.press/v162/he22f.html)

        **Abstract**:

        Prompt-Tuning is a new paradigm for finetuning pre-trained language models in a parameter efficient way. Here, we explore the use of HyperNetworks to generate hyper-prompts: we propose HyperPrompt, a novel architecture for prompt-based task-conditioning of self-attention in Transformers. The hyper-prompts are end-to-end learnable via generation by a HyperNetwork. HyperPrompt allows the network to learn task-specific feature maps where the hyper-prompts serve as task global memories for the queries to attend to, at the same time enabling flexible information sharing among tasks. We show that HyperPrompt is competitive against strong multi-task learning baselines with as few as 0.14% of additional task-conditioning parameters, achieving great parameter and computational efficiency. Through extensive empirical experiments, we demonstrate that HyperPrompt can achieve superior performances over strong T5 multi-task learning baselines and parameter-efficient adapter variants including Prompt-Tuning and HyperFormer++ on Natural Language Understanding benchmarks of GLUE and SuperGLUE across many model sizes.

        ----

        ## [383] Label-Descriptive Patterns and Their Application to Characterizing Classification Errors

        **Authors**: *Michael A. Hedderich, Jonas Fischer, Dietrich Klakow, Jilles Vreeken*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hedderich22a.html](https://proceedings.mlr.press/v162/hedderich22a.html)

        **Abstract**:

        State-of-the-art deep learning methods achieve human-like performance on many tasks, but make errors nevertheless. Characterizing these errors in easily interpretable terms gives insight into whether a classifier is prone to making systematic errors, but also gives a way to act and improve the classifier. We propose to discover those feature-value combinations (i.e., patterns) that strongly correlate with correct resp. erroneous predictions to obtain a global and interpretable description for arbitrary classifiers. We show this is an instance of the more general label description problem, which we formulate in terms of the Minimum Description Length principle. To discover a good pattern set, we develop the efficient Premise algorithm. Through an extensive set of experiments we show it performs very well in practice on both synthetic and real-world data. Unlike existing solutions, it ably recovers ground truth patterns, even on highly imbalanced data over many features. Through two case studies on Visual Question Answering and Named Entity Recognition, we confirm that Premise gives clear and actionable insight into the systematic errors made by modern NLP classifiers.

        ----

        ## [384] NOMU: Neural Optimization-based Model Uncertainty

        **Authors**: *Jakob Heiss, Jakob Weissteiner, Hanna S. Wutte, Sven Seuken, Josef Teichmann*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/heiss22a.html](https://proceedings.mlr.press/v162/heiss22a.html)

        **Abstract**:

        We study methods for estimating model uncertainty for neural networks (NNs) in regression. To isolate the effect of model uncertainty, we focus on a noiseless setting with scarce training data. We introduce five important desiderata regarding model uncertainty that any method should satisfy. However, we find that established benchmarks often fail to reliably capture some of these desiderata, even those that are required by Bayesian theory. To address this, we introduce a new approach for capturing model uncertainty for NNs, which we call Neural Optimization-based Model Uncertainty (NOMU). The main idea of NOMU is to design a network architecture consisting of two connected sub-NNs, one for model prediction and one for model uncertainty, and to train it using a carefully-designed loss function. Importantly, our design enforces that NOMU satisfies our five desiderata. Due to its modular architecture, NOMU can provide model uncertainty for any given (previously trained) NN if given access to its training data. We evaluate NOMU in various regressions tasks and noiseless Bayesian optimization (BO) with costly evaluations. In regression, NOMU performs at least as well as state-of-the-art methods. In BO, NOMU even outperforms all considered benchmarks.

        ----

        ## [385] Scaling Out-of-Distribution Detection for Real-World Settings

        **Authors**: *Dan Hendrycks, Steven Basart, Mantas Mazeika, Andy Zou, Joseph Kwon, Mohammadreza Mostajabi, Jacob Steinhardt, Dawn Song*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hendrycks22a.html](https://proceedings.mlr.press/v162/hendrycks22a.html)

        **Abstract**:

        Detecting out-of-distribution examples is important for safety-critical machine learning applications such as detecting novel biological phenomena and self-driving cars. However, existing research mainly focuses on simple small-scale settings. To set the stage for more realistic out-of-distribution detection, we depart from small-scale settings and explore large-scale multiclass and multi-label settings with high-resolution images and thousands of classes. To make future work in real-world settings possible, we create new benchmarks for three large-scale settings. To test ImageNet multiclass anomaly detectors, we introduce the Species dataset containing over 700,000 images and over a thousand anomalous species. We leverage ImageNet-21K to evaluate PASCAL VOC and COCO multilabel anomaly detectors. Third, we introduce a new benchmark for anomaly segmentation by introducing a segmentation benchmark with road anomalies. We conduct extensive experiments in these more realistic settings for out-of-distribution detection and find that a surprisingly simple detector based on the maximum logit outperforms prior methods in all the large-scale multi-class, multi-label, and segmentation tasks, establishing a simple new baseline for future work.

        ----

        ## [386] Generalization Bounds using Lower Tail Exponents in Stochastic Optimizers

        **Authors**: *Liam Hodgkinson, Umut Simsekli, Rajiv Khanna, Michael W. Mahoney*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hodgkinson22a.html](https://proceedings.mlr.press/v162/hodgkinson22a.html)

        **Abstract**:

        Despite the ubiquitous use of stochastic optimization algorithms in machine learning, the precise impact of these algorithms and their dynamics on generalization performance in realistic non-convex settings is still poorly understood. While recent work has revealed connections between generalization and heavy-tailed behavior in stochastic optimization, they mainly relied on continuous-time approximations; and a rigorous treatment for the original discrete-time iterations is yet to be performed. To bridge this gap, we present novel bounds linking generalization to the lower tail exponent of the transition kernel associated with the optimizer around a local minimum, in both discrete- and continuous-time settings. To achieve this, we first prove a data- and algorithm-dependent generalization bound in terms of the celebrated Fernique-Talagrand functional applied to the trajectory of the optimizer. Then, we specialize this result by exploiting the Markovian structure of stochastic optimizers, and derive bounds in terms of their (data-dependent) transition kernels. We support our theory with empirical results from a variety of neural networks, showing correlations between generalization error and lower tail exponents.

        ----

        ## [387] Unsupervised Detection of Contextualized Embedding Bias with Application to Ideology

        **Authors**: *Valentin Hofmann, Janet B. Pierrehumbert, Hinrich Schütze*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hofmann22a.html](https://proceedings.mlr.press/v162/hofmann22a.html)

        **Abstract**:

        We propose a fully unsupervised method to detect bias in contextualized embeddings. The method leverages the assortative information latently encoded by social networks and combines orthogonality regularization, structured sparsity learning, and graph neural networks to find the embedding subspace capturing this information. As a concrete example, we focus on the phenomenon of ideological bias: we introduce the concept of an ideological subspace, show how it can be found by applying our method to online discussion forums, and present techniques to probe it. Our experiments suggest that the ideological subspace encodes abstract evaluative semantics and reflects changes in the political left-right spectrum during the presidency of Donald Trump.

        ----

        ## [388] Neural Laplace: Learning diverse classes of differential equations in the Laplace domain

        **Authors**: *Samuel Holt, Zhaozhi Qian, Mihaela van der Schaar*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/holt22a.html](https://proceedings.mlr.press/v162/holt22a.html)

        **Abstract**:

        Neural Ordinary Differential Equations model dynamical systems with ODEs learned by neural networks. However, ODEs are fundamentally inadequate to model systems with long-range dependencies or discontinuities, which are common in engineering and biological systems. Broader classes of differential equations (DE) have been proposed as remedies, including delay differential equations and integro-differential equations. Furthermore, Neural ODE suffers from numerical instability when modelling stiff ODEs and ODEs with piecewise forcing functions. In this work, we propose Neural Laplace, a unifying framework for learning diverse classes of DEs including all the aforementioned ones. Instead of modelling the dynamics in the time domain, we model it in the Laplace domain, where the history-dependencies and discontinuities in time can be represented as summations of complex exponentials. To make learning more efficient, we use the geometrical stereographic map of a Riemann sphere to induce more smoothness in the Laplace domain. In the experiments, Neural Laplace shows superior performance in modelling and extrapolating the trajectories of diverse classes of DEs, including the ones with complex history dependency and abrupt changes.

        ----

        ## [389] Deep Hierarchy in Bandits

        **Authors**: *Joey Hong, Branislav Kveton, Sumeet Katariya, Manzil Zaheer, Mohammad Ghavamzadeh*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hong22a.html](https://proceedings.mlr.press/v162/hong22a.html)

        **Abstract**:

        Mean rewards of actions are often correlated. The form of these correlations may be complex and unknown a priori, such as the preferences of users for recommended products and their categories. To maximize statistical efficiency, it is important to leverage these correlations when learning. We formulate a bandit variant of this problem where the correlations of mean action rewards are represented by a hierarchical Bayesian model with latent variables. Since the hierarchy can have multiple layers, we call it deep. We propose a hierarchical Thompson sampling algorithm (HierTS) for this problem and show how to implement it efficiently for Gaussian hierarchies. The efficient implementation is possible due to a novel exact hierarchical representation of the posterior, which itself is of independent interest. We use this exact posterior to analyze the Bayes regret of HierTS. Our regret bounds reflect the structure of the problem, that the regret decreases with more informative priors, and can be recast to highlight reduced dependence on the number of actions. We confirm these theoretical findings empirically, in both synthetic and real-world experiments.

        ----

        ## [390] DAdaQuant: Doubly-adaptive quantization for communication-efficient Federated Learning

        **Authors**: *Robert Hönig, Yiren Zhao, Robert D. Mullins*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/honig22a.html](https://proceedings.mlr.press/v162/honig22a.html)

        **Abstract**:

        Federated Learning (FL) is a powerful technique to train a model on a server with data from several clients in a privacy-preserving manner. FL incurs significant communication costs because it repeatedly transmits the model between the server and clients. Recently proposed algorithms quantize the model parameters to efficiently compress FL communication. We find that dynamic adaptations of the quantization level can boost compression without sacrificing model quality. We introduce DAdaQuant as a doubly-adaptive quantization algorithm that dynamically changes the quantization level across time and different clients. Our experiments show that DAdaQuant consistently improves client$\rightarrow$server compression, outperforming the strongest non-adaptive baselines by up to $2.8\times$.

        ----

        ## [391] Equivariant Diffusion for Molecule Generation in 3D

        **Authors**: *Emiel Hoogeboom, Victor Garcia Satorras, Clément Vignac, Max Welling*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hoogeboom22a.html](https://proceedings.mlr.press/v162/hoogeboom22a.html)

        **Abstract**:

        This work introduces a diffusion model for molecule generation in 3D that is equivariant to Euclidean transformations. Our E(3) Equivariant Diffusion Model (EDM) learns to denoise a diffusion process with an equivariant network that jointly operates on both continuous (atom coordinates) and categorical features (atom types). In addition, we provide a probabilistic analysis which admits likelihood computation of molecules using our model. Experimentally, the proposed method significantly outperforms previous 3D molecular generative methods regarding the quality of generated samples and the efficiency at training time.

        ----

        ## [392] Conditional GANs with Auxiliary Discriminative Classifier

        **Authors**: *Liang Hou, Qi Cao, Huawei Shen, Siyuan Pan, Xiaoshuang Li, Xueqi Cheng*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hou22a.html](https://proceedings.mlr.press/v162/hou22a.html)

        **Abstract**:

        Conditional generative models aim to learn the underlying joint distribution of data and labels to achieve conditional data generation. Among them, the auxiliary classifier generative adversarial network (AC-GAN) has been widely used, but suffers from the problem of low intra-class diversity of the generated samples. The fundamental reason pointed out in this paper is that the classifier of AC-GAN is generator-agnostic, which therefore cannot provide informative guidance for the generator to approach the joint distribution, resulting in a minimization of the conditional entropy that decreases the intra-class diversity. Motivated by this understanding, we propose a novel conditional GAN with an auxiliary discriminative classifier (ADC-GAN) to resolve the above problem. Specifically, the proposed auxiliary discriminative classifier becomes generator-aware by recognizing the class-labels of the real data and the generated data discriminatively. Our theoretical analysis reveals that the generator can faithfully learn the joint distribution even without the original discriminator, making the proposed ADC-GAN robust to the value of the coefficient hyperparameter and the selection of the GAN loss, and stable during training. Extensive experimental results on synthetic and real-world datasets demonstrate the superiority of ADC-GAN in conditional generative modeling compared to state-of-the-art classifier-based and projection-based conditional GANs.

        ----

        ## [393] AdAUC: End-to-end Adversarial AUC Optimization Against Long-tail Problems

        **Authors**: *Wenzheng Hou, Qianqian Xu, Zhiyong Yang, Shilong Bao, Yuan He, Qingming Huang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hou22b.html](https://proceedings.mlr.press/v162/hou22b.html)

        **Abstract**:

        It is well-known that deep learning models are vulnerable to adversarial examples. Existing studies of adversarial training have made great progress against this challenge. As a typical trait, they often assume that the class distribution is overall balanced. However, long-tail datasets are ubiquitous in a wide spectrum of applications, where the amount of head class instances is significantly larger than the tail classes. Under such a scenario, AUC is a much more reasonable metric than accuracy since it is insensitive toward class distribution. Motivated by this, we present an early trial to explore adversarial training methods to optimize AUC. The main challenge lies in that the positive and negative examples are tightly coupled in the objective function. As a direct result, one cannot generate adversarial examples without a full scan of the dataset. To address this issue, based on a concavity regularization scheme, we reformulate the AUC optimization problem as a saddle point problem, where the objective becomes an instance-wise function. This leads to an end-to-end training protocol. Furthermore, we provide a convergence guarantee of the proposed training algorithm. Our analysis differs from the existing studies since the algorithm is asked to generate adversarial examples by calculating the gradient of a min-max problem. Finally, the extensive experimental results show the performance and robustness of our algorithm in three long-tail datasets.

        ----

        ## [394] Wide Bayesian neural networks have a simple weight posterior: theory and accelerated sampling

        **Authors**: *Jiri Hron, Roman Novak, Jeffrey Pennington, Jascha Sohl-Dickstein*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hron22a.html](https://proceedings.mlr.press/v162/hron22a.html)

        **Abstract**:

        We introduce repriorisation, a data-dependent reparameterisation which transforms a Bayesian neural network (BNN) posterior to a distribution whose KL divergence to the BNN prior vanishes as layer widths grow. The repriorisation map acts directly on parameters, and its analytic simplicity complements the known neural network Gaussian process (NNGP) behaviour of wide BNNs in function space. Exploiting the repriorisation, we develop a Markov chain Monte Carlo (MCMC) posterior sampling algorithm which mixes faster the wider the BNN. This contrasts with the typically poor performance of MCMC in high dimensions. We observe up to 50x higher effective sample size relative to no reparametrisation for both fully-connected and residual networks. Improvements are achieved at all widths, with the margin between reparametrised and standard BNNs growing with layer width.

        ----

        ## [395] Learning inverse folding from millions of predicted structures

        **Authors**: *Chloe Hsu, Robert Verkuil, Jason Liu, Zeming Lin, Brian Hie, Tom Sercu, Adam Lerer, Alexander Rives*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hsu22a.html](https://proceedings.mlr.press/v162/hsu22a.html)

        **Abstract**:

        We consider the problem of predicting a protein sequence from its backbone atom coordinates. Machine learning approaches to this problem to date have been limited by the number of available experimentally determined protein structures. We augment training data by nearly three orders of magnitude by predicting structures for 12M protein sequences using AlphaFold2. Trained with this additional data, a sequence-to-sequence transformer with invariant geometric input processing layers achieves 51% native sequence recovery on structurally held-out backbones with 72% recovery for buried residues, an overall improvement of almost 10 percentage points over existing methods. The model generalizes to a variety of more complex tasks including design of protein complexes, partially masked structures, binding interfaces, and multiple states.

        ----

        ## [396] Nearly Minimax Optimal Reinforcement Learning with Linear Function Approximation

        **Authors**: *Pihe Hu, Yu Chen, Longbo Huang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hu22a.html](https://proceedings.mlr.press/v162/hu22a.html)

        **Abstract**:

        We study reinforcement learning with linear function approximation where the transition probability and reward functions are linear with respect to a feature mapping $\boldsymbol{\phi}(s,a)$. Specifically, we consider the episodic inhomogeneous linear Markov Decision Process (MDP), and propose a novel computation-efficient algorithm, LSVI-UCB$^+$, which achieves an $\widetilde{O}(Hd\sqrt{T})$ regret bound where $H$ is the episode length, $d$ is the feature dimension, and $T$ is the number of steps. LSVI-UCB$^+$ builds on weighted ridge regression and upper confidence value iteration with a Bernstein-type exploration bonus. Our statistical results are obtained with novel analytical tools, including a new Bernstein self-normalized bound with conservatism on elliptical potentials, and refined analysis of the correction term. To the best of our knowledge, this is the first minimax optimal algorithm for linear MDPs up to logarithmic factors, which closes the $\sqrt{Hd}$ gap between the best known upper bound of $\widetilde{O}(\sqrt{H^3d^3T})$ in \cite{jin2020provably} and lower bound of $\Omega(Hd\sqrt{T})$ for linear MDPs.

        ----

        ## [397] Neuron Dependency Graphs: A Causal Abstraction of Neural Networks

        **Authors**: *Yaojie Hu, Jin Tian*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hu22b.html](https://proceedings.mlr.press/v162/hu22b.html)

        **Abstract**:

        We discover that neural networks exhibit approximate logical dependencies among neurons, and we introduce Neuron Dependency Graphs (NDG) that extract and present them as directed graphs. In an NDG, each node corresponds to the boolean activation value of a neuron, and each edge models an approximate logical implication from one node to another. We show that the logical dependencies extracted from the training dataset generalize well to the test set. In addition to providing symbolic explanations to the neural network’s internal structure, NDGs can represent a Structural Causal Model. We empirically show that an NDG is a causal abstraction of the corresponding neural network that "unfolds" the same way under causal interventions using the theory by Geiger et al. (2021). Code is available at https://github.com/phimachine/ndg.

        ----

        ## [398] Policy Diagnosis via Measuring Role Diversity in Cooperative Multi-agent RL

        **Authors**: *Siyi Hu, Chuanlong Xie, Xiaodan Liang, Xiaojun Chang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hu22c.html](https://proceedings.mlr.press/v162/hu22c.html)

        **Abstract**:

        Cooperative multi-agent reinforcement learning (MARL) is making rapid progress for solving tasks in a grid world and real-world scenarios, in which agents are given different attributes and goals, resulting in different behavior through the whole multi-agent task. In this study, we quantify the agent’s behavior difference and build its relationship with the policy performance via {\bf Role Diversity}, a metric to measure the characteristics of MARL tasks. We define role diversity from three perspectives: action-based, trajectory-based, and contribution-based to fully measure a multi-agent task. Through theoretical analysis, we find that the error bound in MARL can be decomposed into three parts that have a strong relation to the role diversity. The decomposed factors can significantly impact policy optimization in three popular directions including parameter sharing, communication mechanism, and credit assignment. The main experimental platforms are based on {\bf Multiagent Particle Environment (MPE) }and {\bf The StarCraft Multi-Agent Challenge (SMAC)}. Extensive experiments clearly show that role diversity can serve as a robust measurement for the characteristics of a multi-agent cooperation task and help diagnose whether the policy fits the current multi-agent system for better policy performance.

        ----

        ## [399] On the Role of Discount Factor in Offline Reinforcement Learning

        **Authors**: *Hao Hu, Yiqin Yang, Qianchuan Zhao, Chongjie Zhang*

        **Conference**: *icml 2022*

        **URL**: [https://proceedings.mlr.press/v162/hu22d.html](https://proceedings.mlr.press/v162/hu22d.html)

        **Abstract**:

        Offline reinforcement learning (RL) enables effective learning from previously collected data without exploration, which shows great promise in real-world applications when exploration is expensive or even infeasible. The discount factor, $\gamma$, plays a vital role in improving online RL sample efficiency and estimation accuracy, but the role of the discount factor in offline RL is not well explored. This paper examines two distinct effects of $\gamma$ in offline RL with theoretical analysis, namely the regularization effect and the pessimism effect. On the one hand, $\gamma$ is a regulator to trade-off optimality with sample efficiency upon existing offline techniques. On the other hand, lower guidance $\gamma$ can also be seen as a way of pessimism where we optimize the policy’s performance in the worst possible models. We empirically verify the above theoretical observation with tabular MDPs and standard D4RL tasks. The results show that the discount factor plays an essential role in the performance of offline RL algorithms, both under small data regimes upon existing offline methods and in large data regimes without other conservative methods.

        ----

        

[Go to the previous page](ICML-2022-list01.md)

[Go to the next page](ICML-2022-list03.md)

[Go to the catalog section](README.md)