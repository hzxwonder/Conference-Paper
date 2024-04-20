## [400] Latent Hierarchical Causal Structure Discovery with Rank Constraints

        **Authors**: *Biwei Huang, Charles Jia Han Low, Feng Xie, Clark Glymour, Kun Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/24d2dd6dc9b79116f8ebc852ddb9dc94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/24d2dd6dc9b79116f8ebc852ddb9dc94-Abstract-Conference.html)

        **Abstract**:

        Most causal discovery procedures assume that there are no latent confounders in the system, which is often violated in real-world problems. In this paper, we consider a challenging scenario for causal structure identification, where some variables are latent and they may form a hierarchical graph structure to generate the measured variables; the children of latent variables may still be latent and only leaf nodes are measured, and moreover, there can be multiple paths between every pair of variables (i.e., it is beyond tree structure). We propose an estimation procedure that can efficiently locate latent variables, determine their cardinalities, and identify the latent hierarchical structure, by leveraging rank deficiency constraints over the measured variables. We show that the proposed algorithm can find the correct Markov equivalence class of the whole graph asymptotically under proper restrictions on the graph structure and with linear causal relations.

        ----

        ## [401] Beyond Real-world Benchmark Datasets: An Empirical Study of Node Classification with GNNs

        **Authors**: *Seiji Maekawa, Koki Noda, Yuya Sasaki, Makoto Onizuka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/24d6d158531508115e628188e2697f76-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/24d6d158531508115e628188e2697f76-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have achieved great success on a node classification task. Despite the broad interest in developing and evaluating GNNs, they have been assessed with limited benchmark datasets. As a result, the existing evaluation of GNNs lacks fine-grained analysis from various characteristics of graphs. Motivated by this, we conduct extensive experiments with a synthetic graph generator that can generate graphs having controlled characteristics for fine-grained analysis. Our empirical studies clarify the strengths and weaknesses of GNNs from four major characteristics of real-world graphs with class labels of nodes, i.e., 1) class size distributions (balanced vs. imbalanced), 2) edge connection proportions between classes (homophilic vs. heterophilic), 3) attribute values (biased vs. random), and 4) graph sizes (small vs. large). In addition, to foster future research on GNNs, we publicly release our codebase that allows users to evaluate various GNNs with various graphs. We hope this work offers interesting insights for future research.

        ----

        ## [402] Doubly-Asynchronous Value Iteration: Making Value Iteration Asynchronous in Actions

        **Authors**: *Tian Tian, Kenny Young, Richard S. Sutton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/24e4e3234178a836b70e0aa48827e0ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/24e4e3234178a836b70e0aa48827e0ff-Abstract-Conference.html)

        **Abstract**:

        Value iteration (VI) is a foundational dynamic programming method, important for learning and planning in optimal control and reinforcement learning.  VI proceeds in batches, where the update to the value of each state must be completed before the next batch of updates can begin.  Completing a single batch is prohibitively expensive if the state space is large, rendering VI impractical for many applications.  Asynchronous VI helps to address the large state space problem by updating one state at a time, in-place and in an arbitrary order.  However, Asynchronous VI still requires a maximization over the entire action space, making it impractical for domains with large action space.  To address this issue, we propose doubly-asynchronous value iteration (DAVI), a new algorithm that generalizes the idea of asynchrony from states to states and actions.  More concretely, DAVI maximizes over a sampled subset of actions that can be of any user-defined size.  This simple approach of using sampling to reduce computation maintains similarly appealing theoretical properties to VI without the need to wait for a full sweep through the entire action space in each update.  In this paper, we show DAVI converges to the optimal value function with probability one, converges at a near-geometric rate with probability $1-\delta$, and returns a near-optimal policy in computation time that nearly matches a previously established bound for VI.  We also empirically demonstrate DAVI's effectiveness in several experiments.

        ----

        ## [403] Turbocharging Solution Concepts: Solving NEs, CEs and CCEs with Neural Equilibrium Solvers

        **Authors**: *Luke Marris, Ian Gemp, Thomas Anthony, Andrea Tacchetti, Siqi Liu, Karl Tuyls*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/24f420aa4c99642dbb9aae18b166bbbc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/24f420aa4c99642dbb9aae18b166bbbc-Abstract-Conference.html)

        **Abstract**:

        Solution concepts such as Nash Equilibria, Correlated Equilibria, and Coarse Correlated Equilibria are useful components for many multiagent machine learning algorithms. Unfortunately, solving a normal-form game could take prohibitive or non-deterministic time to converge, and could fail. We introduce the Neural Equilibrium Solver which utilizes a special equivariant neural network architecture to approximately solve the space of all games of fixed shape, buying speed and determinism. We define a flexible equilibrium selection framework, that is capable of uniquely selecting an equilibrium that minimizes relative entropy, or maximizes welfare. The network is trained without needing to generate any supervised training data. We show remarkable zero-shot generalization to larger games. We argue that such a network is a powerful component for many possible multiagent algorithms.

        ----

        ## [404] NOMAD: Nonlinear Manifold Decoders for Operator Learning

        **Authors**: *Jacob H. Seidman, Georgios Kissas, Paris Perdikaris, George J. Pappas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/24f49b2ad9fbe65eefbfd99d6f6c3fd2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/24f49b2ad9fbe65eefbfd99d6f6c3fd2-Abstract-Conference.html)

        **Abstract**:

        Supervised learning in function spaces is an emerging area of machine learning research with applications to the prediction of complex physical systems such as fluid flows, solid mechanics, and climate modeling.  By directly learning maps (operators) between infinite dimensional function spaces, these models are able to learn discretization invariant representations of target functions.  A common approach is to represent such target functions as linear combinations of basis elements learned from data. However, there are simple scenarios where, even though the target functions form a low dimensional submanifold, a very large number of basis elements is needed for an accurate linear representation. Here we present NOMAD, a novel operator learning framework with a nonlinear decoder map capable of learning finite dimensional representations of nonlinear submanifolds in function spaces.  We show this method is able to accurately learn low dimensional representations of solution manifolds to partial differential equations while outperforming linear models of larger size.  Additionally, we compare to state-of-the-art operator learning methods on a complex fluid dynamics benchmark and achieve competitive performance with a significantly smaller model size and training cost.

        ----

        ## [405] Semi-Supervised Video Salient Object Detection Based on Uncertainty-Guided Pseudo Labels

        **Authors**: *Yongri Piao, Chenyang Lu, Miao Zhang, Huchuan Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/24f7b98aef14fcd68acf3c941af1b59e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/24f7b98aef14fcd68acf3c941af1b59e-Abstract-Conference.html)

        **Abstract**:

        Semi-Supervised Video Salient Object Detection (SS-VSOD) is challenging because of the lack of temporal information in video sequences caused by sparse annotations. Most works address this problem by generating pseudo labels for unlabeled data. However, error-prone pseudo labels negatively affect the VOSD model. Therefore, a deeper insight into pseudo labels should be developed. In this work, we aim to explore 1) how to utilize the incorrect predictions in pseudo labels to guide the network to generate more robust pseudo labels and 2) how to further screen out the noise that still exists in the improved pseudo labels. To this end, we propose an Uncertainty-Guided Pseudo Label Generator (UGPLG), which makes full use of inter-frame information to ensure the temporal consistency of the pseudo labels and improves the robustness of the pseudo labels by strengthening the learning of difficult scenarios. Furthermore, we also introduce the adversarial learning to address the noise problems in pseudo labels, guaranteeing the positive guidance of pseudo labels during model training. Experimental results demonstrate that our methods outperform existing semi-supervised method and partial fully-supervised methods across five public benchmarks of DAVIS, FBMS, MCL, ViSal and SegTrack-V2.

        ----

        ## [406] Debiased Causal Tree: Heterogeneous Treatment Effects Estimation with Unmeasured Confounding

        **Authors**: *Caizhi Tang, Huiyuan Wang, Xinyu Li, Qing Cui, Ya-Lin Zhang, Feng Zhu, Longfei Li, Jun Zhou, Linbo Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2526d439030a3af95fc647dd20e9d049-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2526d439030a3af95fc647dd20e9d049-Abstract-Conference.html)

        **Abstract**:

        Unmeasured confounding poses a significant threat to the validity of causal inference. Despite that various ad hoc methods are developed to remove confounding effects, they are subject to certain fairly strong assumptions. In this work, we consider the estimation of conditional causal effects in the presence of unmeasured confounding using observational data and historical controls. Under an interpretable transportability condition, we prove the partial identifiability of conditional average treatment effect on the treated group (CATT). For tree-based models, a new notion, \emph{confounding entropy}, is proposed to measure the discrepancy introduced by unobserved confounders between the conditional outcome distribution of the treated and control groups. The confounding entropy generalizes conventional confounding bias, and can be estimated effectively using historical controls. We develop a new method, debiased causal tree, whose splitting rule is to minimize the empirical risk regularized by the confounding entropy. Notably, our method integrates current observational data (for empirical risk) and their historical controls (for confounding entropy) harmoniously.  We highlight that, debiased causal tree can not only estimate CATT well in the presence of unmeasured confounding, but also is a robust estimator of conditional average treatment effect (CATE) against the imbalance of the treated and control populations when all confounders are observed. An extension of combining multiple debiased causal trees to further reduce biases by gradient boosting is considered. The computational feasibility and statistical power of our method are evidenced by simulations and a study of a credit card balance dataset.

        ----

        ## [407] Asymptotic Properties for Bayesian Neural Network in Besov Space

        **Authors**: *Kyeongwon Lee, Jaeyong Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/257a7d7934bad9b3a1774e8994e2a0dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/257a7d7934bad9b3a1774e8994e2a0dc-Abstract-Conference.html)

        **Abstract**:

        Neural networks have shown great predictive power when applied to unstructured data such as images and natural languages. The Bayesian neural network captures the uncertainty of prediction by computing the posterior distribution of the model parameters. In this paper, we show that the Bayesian neural network with spikeand-slab prior has posterior consistency with a near minimax optimal convergence rate when the true regression function belongs to the Besov space. The spikeand-slab prior is adaptive to the smoothness of the regression function and the posterior convergence rate does not change even when the smoothness of the regression function is unknown. We also consider the shrinkage prior, which is computationally more feasible than the spike-and-slab prior, and show that it has the same posterior convergence rate as the spike-and-slab prior.

        ----

        ## [408] A PAC-Bayesian Generalization Bound for Equivariant Networks

        **Authors**: *Arash Behboodi, Gabriele Cesa, Taco S. Cohen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/257b9a6a0e3856735d0e624e38fb6803-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/257b9a6a0e3856735d0e624e38fb6803-Abstract-Conference.html)

        **Abstract**:

        Equivariant networks capture the inductive bias about the symmetry of the learning task by building those symmetries into the model. In this paper, we study how equivariance relates to generalization error utilizing PAC Bayesian analysis for equivariant networks, where the transformation laws of feature spaces are deter- mined by group representations. By using perturbation analysis of equivariant networks in Fourier domain for each layer, we derive norm-based PAC-Bayesian generalization bounds. The bound characterizes the impact of group size, and multiplicity and degree of irreducible representations on the generalization error and thereby provide a guideline for selecting them. In general, the bound indicates that using larger group size in the model improves the generalization error substantiated by extensive numerical experiments.

        ----

        ## [409] Neural Network Architecture Beyond Width and Depth

        **Authors**: *Shijun Zhang, Zuowei Shen, Haizhao Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/257be12f31dfa7cc158dda99822c6fd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/257be12f31dfa7cc158dda99822c6fd1-Abstract-Conference.html)

        **Abstract**:

        This paper proposes a new neural network architecture by introducing an additional dimension called height beyond width and depth. Neural network architectures with height, width, and depth as hyper-parameters are called three-dimensional architectures. It is shown that neural networks with three-dimensional architectures are significantly more expressive than the ones with two-dimensional architectures (those with only width and depth as hyper-parameters), e.g., standard fully connected networks. The new network architecture is constructed recursively via a nested structure, and hence we call a network with the new architecture nested network (NestNet). A NestNet of height $s$ is built with each hidden neuron activated by a NestNet of height $\le s-1$. When $s=1$, a NestNet degenerates to a standard network with a two-dimensional architecture. It is proved by construction that height-$s$ ReLU NestNets with $\mathcal{O}(n)$ parameters can approximate $1$-Lipschitz continuous functions on $[0,1]^d$ with an error $\mathcal{O}(n^{-(s+1)/d})$, while the optimal approximation error of standard ReLU networks with $\mathcal{O}(n)$ parameters is $\mathcal{O}(n^{-2/d})$. Furthermore, such a result is extended to generic continuous functions on $[0,1]^d$ with the approximation error characterized by the modulus of continuity. Finally, we use numerical experimentation to show the advantages of the super-approximation power of ReLU NestNets.

        ----

        ## [410] S-Prompts Learning with Pre-trained Transformers: An Occam's Razor for Domain Incremental Learning

        **Authors**: *Yabin Wang, Zhiwu Huang, Xiaopeng Hong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/25886d7a7cf4e33fd44072a0cd81bf30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/25886d7a7cf4e33fd44072a0cd81bf30-Abstract-Conference.html)

        **Abstract**:

        State-of-the-art deep neural networks are still struggling to address the catastrophic forgetting problem in continual learning. In this paper, we propose one simple paradigm (named as S-Prompting) and two concrete approaches to highly reduce the forgetting degree in one of the most typical continual learning scenarios, i.e., domain increment learning (DIL). The key idea of the paradigm is to learn prompts independently across domains with pre-trained transformers, avoiding the use of exemplars that commonly appear in conventional methods. This results in a win-win game where the prompting can achieve the best for each domain. The independent prompting across domains only requests one single cross-entropy loss for training and one simple K-NN operation as a domain identifier for inference. The learning paradigm derives an image prompt learning approach and a novel language-image prompt learning approach. Owning an excellent scalability (0.03% parameter increase per domain), the best of our approaches achieves a remarkable relative improvement (an average of about 30%) over the best of the state-of-the-art exemplar-free methods for three standard DIL tasks, and even surpasses the best of them relatively by about 6% in average when they use exemplars. Source code is available at https://github.com/iamwangyabin/S-Prompts.

        ----

        ## [411] OmniVL: One Foundation Model for Image-Language and Video-Language Tasks

        **Authors**: *Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Luowei Zhou, Yucheng Zhao, Yujia Xie, Ce Liu, Yu-Gang Jiang, Lu Yuan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/259a5df46308d60f8454bd4adcc3b462-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/259a5df46308d60f8454bd4adcc3b462-Abstract-Conference.html)

        **Abstract**:

        This paper presents OmniVL, a new foundation model to support both image-language and video-language tasks using one universal architecture. It adopts a unified transformer-based visual encoder for both image and video inputs, and thus can perform joint image-language and video-language pretraining. We demonstrate, for the first time, such a paradigm benefits both image and video tasks, as opposed to the conventional one-directional transfer (e.g., use image-language to help video-language). To this end, we propose a \emph{decoupled} joint pretraining of image-language and video-language to effectively decompose the vision-language modeling into spatial and temporal dimensions and obtain performance boost on both image and video tasks. Moreover, we introduce a novel unified vision-language contrastive (UniVLC) loss to leverage image-text, video-text, image-label (e.g., image classification), video-label (e.g., video action recognition) data together, so that both supervised and noisily supervised pretraining data are utilized as much as possible. Without incurring extra task-specific adaptors, OmniVL can simultaneously support visual only tasks (e.g., image classification, video action recognition), cross-modal alignment tasks (e.g., image/video-text retrieval), and multi-modal understanding and generation tasks (e.g., image/video question answering, captioning). We evaluate OmniVL on a wide range of downstream tasks and achieve state-of-the-art or competitive results with similar model size and data scale.

        ----

        ## [412] Self-Organized Group for Cooperative Multi-agent Reinforcement Learning

        **Authors**: *Jianzhun Shao, Zhiqiang Lou, Hongchang Zhang, Yuhang Jiang, Shuncheng He, Xiangyang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/25b040c97a75021e57100648a20b1e10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/25b040c97a75021e57100648a20b1e10-Abstract-Conference.html)

        **Abstract**:

        Centralized training with decentralized execution (CTDE) has achieved great success in cooperative multi-agent reinforcement learning (MARL) in practical applications. However, CTDE-based methods typically suffer from poor zero-shot generalization ability with dynamic team composition and varying partial observability. To tackle these issues, we propose a spontaneously grouping mechanism, termed Self-Organized Group (SOG), which is featured with conductor election (CE) and message summary (MS). In CE, a certain number of conductors are elected every $T$ time-steps to temporally construct groups, each with conductor-follower consensus where the followers are constrained to only communicate with their conductor. In MS, each conductor summarize and distribute the received messages to all affiliate group members to hold a unified scheduling. SOG provides zero-shot generalization ability to the dynamic number of agents and the varying partial observability. Sufficient experiments on mainstream multi-agent benchmarks exhibit superiority of SOG.

        ----

        ## [413] (Optimal) Online Bipartite Matching with Degree Information

        **Authors**: *Anders Aamand, Justin Y. Chen, Piotr Indyk*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/25c5133ad2ab138f448b71b3c7345ec3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/25c5133ad2ab138f448b71b3c7345ec3-Abstract-Conference.html)

        **Abstract**:

        We propose a model for online graph problems where algorithms are given access to an oracle that predicts (e.g., based on modeling assumptions or past data) the degrees of nodes in the graph. Within this model, we study the classic problem of online bipartite matching, and a natural greedy matching algorithm called MinPredictedDegree, which uses predictions of the degrees of offline nodes. For the bipartite version of a stochastic graph model due to Chung, Lu, and Vu where the expected values of the offline degrees are known and used as predictions, we show that MinPredictedDegree stochastically dominates any other online algorithm, i.e., it is optimal for graphs drawn from this model. Since the "symmetric" version of the model, where all online nodes are identical, is a special case of the well-studied "known i.i.d. model", it follows that the competitive ratio of MinPredictedDegree on such inputs is at least 0.7299. For the special case of graphs with power law degree distributions, we show that MinPredictedDegree frequently produces matchings almost as large as the true maximum matching on such graphs. We complement these results with an extensive empirical evaluation showing that MinPredictedDegree compares favorably to state-of-the-art online algorithms for online matching.

        ----

        ## [414] Fairness in Federated Learning via Core-Stability

        **Authors**: *Bhaskar Ray Chaudhury, Linyi Li, Mintong Kang, Bo Li, Ruta Mehta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/25e92e33ac8c35fd49f394c37f21b6da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/25e92e33ac8c35fd49f394c37f21b6da-Abstract-Conference.html)

        **Abstract**:

        Federated learning provides an effective paradigm to jointly optimize a model benefited from rich distributed data while protecting data privacy. Nonetheless, the heterogeneity nature of distributed data, especially in the non-IID setting, makes it challenging to define and ensure fairness among local agents. For instance, it is intuitively ``unfair" for agents with data of high quality to sacrifice their performance due to other agents with low quality data. Currently popular egalitarian and weighted equity-based fairness measures suffer from the aforementioned pitfall. In this work, we aim to formally represent this problem and address these fairness issues using concepts from co-operative game theory and social choice theory. We model the task of learning a shared predictor in the federated setting as a fair public decision making problem, and then define the notion of core-stable fairness: Given $N$ agents, there is no subset of agents $S$ that can benefit significantly by forming a coalition among themselves based on their utilities $U_N$ and $U_S$ (i.e., $ (|S|/ N) U_S \geq U_N$). Core-stable predictors are robust to low quality local data from some agents, and additionally they satisfy Proportionality (each agent gets at least $1/n$ fraction of the best utility that she can get from any predictor) and Pareto-optimality (there exists no model that can increase the utility of an agent without decreasing the utility of another), two well sought-after fairness and efficiency notions within social choice. We then propose an efficient federated learning protocol CoreFed to optimize a core stable predictor. CoreFed determines a core-stable predictor when the loss functions of the agents are convex. CoreFed also determines approximate core-stable predictors when the loss functions are not convex, like smooth neural networks. We further show the existence of core-stable predictors in more general settings using Kakutani's fixed point theorem. Finally, we empirically validate our analysis on two real-world datasets, and we show that CoreFed achieves higher core-stability fairness than FedAvg while maintaining similar accuracy.

        ----

        ## [415] Log-Polar Space Convolution Layers

        **Authors**: *Bing Su, Ji-Rong Wen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/25eb42c46526071479f871b8bc9ad331-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/25eb42c46526071479f871b8bc9ad331-Abstract-Conference.html)

        **Abstract**:

        Convolutional neural networks use regular quadrilateral convolution kernels to extract features. Since the number of parameters increases quadratically with the size of the convolution kernel, many popular models use small convolution kernels, resulting in small local receptive fields in lower layers. This paper proposes a novel log-polar space convolution (LPSC) layer, where the convolution kernel is elliptical and adaptively divides its local receptive field into different regions according to the relative directions and logarithmic distances. The local receptive field grows exponentially with the number of distance levels. Therefore, the proposed LPSC not only naturally encodes local spatial structures, but also greatly increases the single-layer receptive field while maintaining the number of parameters. We show that LPSC can be implemented with conventional convolution via log-polar space pooling and can be applied in any network architecture to replace conventional convolutions. Experiments on different tasks and datasets demonstrate the effectiveness of the proposed LPSC.

        ----

        ## [416] CoNSoLe: Convex Neural Symbolic Learning

        **Authors**: *Haoran Li, Yang Weng, Hanghang Tong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2608470747b9b55b89de1c0d70418cab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2608470747b9b55b89de1c0d70418cab-Abstract-Conference.html)

        **Abstract**:

        Learning the underlying equation from data is a fundamental problem in many disciplines. Recent advances rely on Neural Networks (NNs) but do not provide theoretical guarantees in obtaining the exact equations owing to the non-convexity of NNs. In this paper, we propose Convex Neural Symbolic Learning (CoNSoLe) to seek convexity under mild conditions. The main idea is to decompose the recovering process into two steps and convexify each step. In the first step of searching for right symbols, we convexify the deep Q-learning. The key is to maintain double convexity for both the negative Q-function and the negative reward function in each iteration, leading to provable convexity of the negative optimal Q  function to learn the true symbol connections. Conditioned on the exact searching result, we construct a Locally Convex equation Learning (LoCaL) neural network to convexify the estimation of symbol coefficients. With such a design, we quantify a large region with strict convexity in the loss surface of LoCaL for commonly used physical functions. Finally, we demonstrate the superior performance of the CoNSoLe framework over the state-of-the-art on a diverse set of datasets.

        ----

        ## [417] DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps

        **Authors**: *Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, Jun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html)

        **Abstract**:

        Diffusion probabilistic models (DPMs) are emerging powerful generative models. Despite their high-quality generation performance, DPMs still suffer from their slow sampling as they generally need hundreds or thousands of sequential function evaluations (steps) of large neural networks to draw a sample. Sampling from DPMs can be viewed alternatively as solving the corresponding diffusion ordinary differential equations (ODEs). In this work, we propose an exact formulation of the solution of diffusion ODEs. The formulation analytically computes the linear part of the solution, rather than leaving all terms to black-box ODE solvers as adopted in previous works. By applying change-of-variable, the solution can be equivalently simplified to an exponentially weighted integral of the neural network. Based on our formulation, we propose DPM-Solver, a fast dedicated high-order solver for diffusion ODEs with the convergence order guarantee. DPM-Solver is suitable for both discrete-time and continuous-time DPMs without any further training. Experimental results show that DPM-Solver can generate high-quality samples in only 10 to 20 function evaluations on various datasets. We achieve 4.70 FID in 10 function evaluations and 2.87 FID in 20 function evaluations on the CIFAR10 dataset, and a 4~16x speedup compared with previous state-of-the-art training-free samplers on various datasets.

        ----

        ## [418] Hierarchical Channel-spatial Encoding for Communication-efficient Collaborative Learning

        **Authors**: *Qihua Zhou, Song Guo, Yi Liu, Jie Zhang, Jiewei Zhang, Tao Guo, Zhenda Xu, Xun Liu, Zhihao Qu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2616697705f72f16a8eac9c295d37d94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2616697705f72f16a8eac9c295d37d94-Abstract-Conference.html)

        **Abstract**:

        It witnesses that the collaborative learning (CL) systems often face the performance bottleneck of limited bandwidth, where multiple low-end devices continuously generate data and transmit intermediate features to the cloud for incremental training. To this end, improving the communication efficiency by reducing traffic size is one of the most crucial issues for realistic deployment. Existing systems mostly compress features at pixel level and ignore the characteristics of feature structure, which could be further exploited for more efficient compression. In this paper, we take new insights into implementing scalable CL systems through a hierarchical compression on features, termed Stripe-wise Group Quantization (SGQ). Different from previous unstructured quantization methods, SGQ captures both channel and spatial similarity in pixels, and simultaneously encodes features in these two levels to gain a much higher compression ratio. In particular, we refactor feature structure based on inter-channel similarity and bound the gradient deviation caused by quantization, in forward and backward passes, respectively. Such a double-stage pipeline makes SGQ hold a sublinear convergence order as the vanilla SGD-based optimization. Extensive experiments show that SGQ achieves a higher traffic reduction ratio by up to 15.97 times and provides 9.22 times image processing speedup over the uniform quantized training, while preserving adequate model accuracy as FP32 does, even using 4-bit quantization. This verifies that SGQ can be applied to a wide spectrum of edge intelligence applications.

        ----

        ## [419] Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation

        **Authors**: *Shiqi Yang, Yaxing Wang, Kai Wang, Shangling Jui, Joost van de Weijer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/26300457961c3e056ea61c9d3ebec2a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/26300457961c3e056ea61c9d3ebec2a4-Abstract-Conference.html)

        **Abstract**:

        We propose a simple but effective source-free domain adaptation (SFDA) method. Treating SFDA as an unsupervised clustering problem and following the intuition that local neighbors in feature space should have more similar predictions than other features, we propose to optimize an objective of prediction consistency. This objective encourages local neighborhood features in feature space to have similar predictions while features farther away in feature space have dissimilar predictions, leading to efficient feature clustering and cluster assignment simultaneously. For efficient training, we seek to optimize an upper-bound of the objective resulting in two simple terms. Furthermore, we relate popular existing methods in domain adaptation, source-free domain adaptation and contrastive learning via the perspective of discriminability and diversity. The experimental results prove the superiority of our method, and our method can be adopted as a simple but strong baseline for future research in SFDA. Our method can be also adapted to source-free open-set and partial-set DA which further shows the generalization ability of our method. Code is available in https://github.com/Albert0147/AaD_SFDA.

        ----

        ## [420] SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction

        **Authors**: *Minhao Liu, Ailing Zeng, Muxi Chen, Zhijian Xu, Qiuxia Lai, Lingna Ma, Qiang Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/266983d0949aed78a16fa4782237dea7-Abstract-Conference.html)

        **Abstract**:

        One unique property of time series is that the temporal relations are largely preserved after downsampling into two sub-sequences. By taking advantage of this property, we propose a novel neural network architecture that conducts sample convolution and interaction for temporal modeling and forecasting, named SCINet. Specifically, SCINet is a recursive downsample-convolve-interact architecture. In each layer, we use multiple convolutional filters to extract distinct yet valuable temporal features from the downsampled sub-sequences or features. By combining these rich features aggregated from multiple resolutions, SCINet effectively models time series with complex temporal dynamics. Experimental results show that SCINet achieves significant forecasting accuracy improvements over both existing convolutional models and Transformer-based solutions across various real-world time series forecasting datasets. Our codes and data are available at https://github.com/cure-lab/SCINet.

        ----

        ## [421] Exploration-Guided Reward Shaping for Reinforcement Learning under Sparse Rewards

        **Authors**: *Rati Devidze, Parameswaran Kamalaruban, Adish Singla*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/266c0f191b04cbbbe529016d0edc847e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/266c0f191b04cbbbe529016d0edc847e-Abstract-Conference.html)

        **Abstract**:

        We study the problem of reward shaping to accelerate the training process of a reinforcement learning agent. Existing works have considered a number of different reward shaping formulations; however, they either require external domain knowledge or fail in environments with extremely sparse rewards. In this paper, we propose a novel framework, Exploration-Guided Reward Shaping (ExploRS), that operates in a fully self-supervised manner and can accelerate an agent's learning even in sparse-reward environments. The key idea of ExploRS is to learn an intrinsic reward function in combination with exploration-based bonuses to maximize the agent's utility w.r.t. extrinsic rewards. We theoretically showcase the usefulness of our reward shaping framework in a special family of MDPs. Experimental results on several environments with sparse/noisy reward signals demonstrate the effectiveness of ExploRS.

        ----

        ## [422] Active Exploration for Inverse Reinforcement Learning

        **Authors**: *David Lindner, Andreas Krause, Giorgia Ramponi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/26d01e5ed42d8dcedd6aa0e3e99cffc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/26d01e5ed42d8dcedd6aa0e3e99cffc4-Abstract-Conference.html)

        **Abstract**:

        Inverse Reinforcement Learning (IRL) is a powerful paradigm for inferring a reward function from expert demonstrations. Many IRL algorithms require a known transition model and sometimes even a known expert policy, or they at least require access to a generative model. However, these assumptions are too strong for many real-world applications, where the environment can be accessed only through sequential interaction. We propose a novel IRL algorithm: Active exploration for Inverse Reinforcement Learning (AceIRL), which actively explores an unknown environment and expert policy to quickly learn the expertâ€™s reward function and identify a good policy. AceIRL uses previous observations to construct confidence intervals that capture plausible reward functions and find exploration policies that focus on the most informative regions of the environment. AceIRL is the first approach to active IRL with sample-complexity bounds that does not require a generative model of the environment. AceIRL matches the sample complexity of active IRL with a generative model in the worst case. Additionally, we establish a problem-dependent bound that relates the sample complexity of AceIRL to the suboptimality gap of a given IRL problem. We empirically evaluate AceIRL in simulations and find that it significantly outperforms more naive exploration strategies.

        ----

        ## [423] Subspace Recovery from Heterogeneous Data with Non-isotropic Noise

        **Authors**: *John C. Duchi, Vitaly Feldman, Lunjia Hu, Kunal Talwar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/26ded5c8ee8ec1bc4caced4e1c9b1584-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/26ded5c8ee8ec1bc4caced4e1c9b1584-Abstract-Conference.html)

        **Abstract**:

        Recovering linear subspaces from data is a fundamental and important task in statistics and machine learning. Motivated by heterogeneity in Federated Learning settings, we study a basic formulation of this problem: the principal component analysis (PCA),  with a focus on dealing with irregular noise. Our data come from $n$ users with user $i$ contributing data samples from a $d$-dimensional distribution with mean $\mu_i$. Our goal is to recover the linear subspace shared by $\mu_1,\ldots,\mu_n$ using the data points from all users, where every data point from user $i$ is formed by adding an independent mean-zero noise vector to $\mu_i$. If we only have one data point from every user, subspace recovery is information-theoretically impossible when the covariance matrices of the noise vectors can be non-spherical, necessitating additional restrictive assumptions in previous work. We avoid these assumptions by leveraging at least two data points from each user, which allows us to design an efficiently-computable estimator under non-spherical and user-dependent noise. We prove an upper bound for the estimation error of our estimator in general scenarios where the number of data points and amount of noise can vary across users, and prove an information-theoretic error lower bound that not only matches the upper bound up to a constant factor, but also holds even for spherical Gaussian noise. This implies that our estimator does not introduce additional estimation error (up to a constant factor) due to irregularity in the noise. We show additional results for a linear regression problem in a similar setup.

        ----

        ## [424] Lifelong Neural Predictive Coding: Learning Cumulatively Online without Forgetting

        **Authors**: *Alexander Ororbia, Ankur Arjun Mali, C. Lee Giles, Daniel Kifer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/26f5a4e26c13d1e0a47f46790c999361-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/26f5a4e26c13d1e0a47f46790c999361-Abstract-Conference.html)

        **Abstract**:

        In lifelong learning systems based on artificial neural networks, one of the biggest obstacles is the inability to retain old knowledge as new information is encountered. This phenomenon is known as catastrophic forgetting. In this paper, we propose a new kind of connectionist architecture, the Sequential Neural Coding Network, that is robust to forgetting when learning from streams of data points and, unlike networks of today, does not learn via the popular back-propagation of errors. Grounded in the neurocognitive theory of predictive coding, our model adapts its synapses in a biologically-plausible fashion while another neural system learns to direct and control this cortex-like structure, mimicking some of the task-executive control functionality of the basal ganglia. In our experiments, we demonstrate that our self-organizing system experiences significantly less forgetting compared to standard neural models, outperforming a swath of previously proposed methods, including rehearsal/data buffer-based methods, on both standard (SplitMNIST, Split Fashion MNIST, etc.) and custom benchmarks even though it is trained in a stream-like fashion. Our work offers evidence that emulating mechanisms in real neuronal systems, e.g., local learning, lateral competition, can yield new directions and possibilities for tackling the grand challenge of lifelong machine learning.

        ----

        ## [425] Align then Fusion: Generalized Large-scale Multi-view Clustering with Anchor Matching Correspondences

        **Authors**: *Siwei Wang, Xinwang Liu, Suyuan Liu, Jiaqi Jin, Wenxuan Tu, Xinzhong Zhu, En Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/270339c997293ca2988c62f4308e389f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/270339c997293ca2988c62f4308e389f-Abstract-Conference.html)

        **Abstract**:

        Multi-view anchor graph clustering selects representative anchors to avoid full pair-wise similarities and therefore reduce the complexity of graph methods. Although widely applied in large-scale applications, existing approaches do not pay sufficient attention to establishing correct correspondences between the anchor sets across views. To be specific, anchor graphs obtained from different views are not aligned column-wisely. Such an Anchor-Unaligned Problem (AUP) would cause inaccurate graph fusion and degrade the clustering performance. Under multi-view scenarios, generating correct correspondences could be extremely difficult since anchors are not consistent in feature dimensions. To solve this challenging issue, we propose the first study of the generalized and flexible anchor graph fusion framework termed Fast Multi-View Anchor-Correspondence Clustering (FMVACC). Specifically, we show how to find anchor correspondence with both feature and structure information, after which anchor graph fusion is performed column-wisely. Moreover, we theoretically show the connection between FMVACC and existing multi-view late fusion and partial view-aligned clustering, which further demonstrates our generality. Extensive experiments on seven benchmark datasets demonstrate the effectiveness and efficiency of our proposed method. Moreover, the proposed alignment module also shows significant performance improvement applying to existing multi-view anchor graph competitors indicating the importance of anchor alignment. Our code is available at \url{https://github.com/wangsiwei2010/NeurIPS22-FMVACC}.

        ----

        ## [426] Geodesic Graph Neural Network for Efficient Graph Representation Learning

        **Authors**: *Lecheng Kong, Yixin Chen, Muhan Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2708a06584ffc33acf092fe9d029dbeb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2708a06584ffc33acf092fe9d029dbeb-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have recently been applied to graph learning tasks and achieved state-of-the-art (SOTA) results. However, many competitive methods run GNNs multiple times with subgraph extraction and customized labeling to capture information that is hard for normal GNNs to learn. Such operations are time-consuming and do not scale to large graphs. In this paper, we propose an efficient GNN framework called Geodesic GNN (GDGNN) that requires only one GNN run and injects conditional relationships between nodes into the model without labeling. This strategy effectively reduces the runtime of subgraph methods. Specifically, we view the shortest paths between two nodes as the spatial graph context of the neighborhood around them. The GNN embeddings of nodes on the shortest paths are used to generate geodesic representations. Conditioned on the geodesic representations, GDGNN can generate node, link, and graph representations that carry much richer structural information than plain GNNs. We theoretically prove that GDGNN is more powerful than plain GNNs. We present experimental results to show that GDGNN achieves highly competitive performance with SOTA GNN models on various graph learning tasks while taking significantly less time.

        ----

        ## [427] Improved Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive Streams

        **Authors**: *Sergey Denisov, H. Brendan McMahan, John Rush, Adam D. Smith, Abhradeep Guha Thakurta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/271ec4d1a9ff5e6b81a6e21d38b1ba96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/271ec4d1a9ff5e6b81a6e21d38b1ba96-Abstract-Conference.html)

        **Abstract**:

        Motivated by recent applications requiring differential privacy in  the setting of adaptive streams, we investigate the question of optimal instantiations of the matrix mechanism in this setting. We prove fundamental theoretical results on the applicability of matrix factorizations to the adaptive streaming setting, and provide a new parameter-free fixed-point algorithm for computing optimal factorizations. We instantiate this framework with respect to concrete matrices which arise naturally in the machine learning setting, and train user-level differentially private models with the resulting optimal mechanisms, yielding significant improvements on a notable problem in federated learning with user-level differential privacy.

        ----

        ## [428] On Privacy and Personalization in Cross-Silo Federated Learning

        **Authors**: *Ken Ziyu Liu, Shengyuan Hu, Steven Wu, Virginia Smith*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2788b4cdf421e03650868cc4184bfed8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2788b4cdf421e03650868cc4184bfed8-Abstract-Conference.html)

        **Abstract**:

        While the application of differential privacy (DP) has been well-studied in cross-device federated learning (FL), there is a lack of work considering DP and its implications for cross-silo FL, a setting characterized by a limited number of clients each containing many data subjects. In cross-silo FL, usual notions of client-level DP are less suitable as real-world privacy regulations typically concern the in-silo data subjects rather than the silos themselves. In this work, we instead consider an alternative notion of silo-specific sample-level DP, where silos set their own privacy targets for their local examples. Under this setting, we reconsider the roles of personalization in federated learning. In particular, we show that mean-regularized multi-task learning (MR-MTL), a simple personalization framework, is a strong baseline for cross-silo FL: under stronger privacy requirements, silos are incentivized to federate more with each other to mitigate DP noise, resulting in consistent improvements relative to standard baseline methods. We provide an empirical study of competing methods as well as a theoretical characterization of MR-MTL for mean estimation, highlighting the interplay between privacy and cross-silo data heterogeneity. Our work serves to establish baselines for private cross-silo FL as well as identify key directions of future work in this area.

        ----

        ## [429] SHAQ: Incorporating Shapley Value Theory into Multi-Agent Q-Learning

        **Authors**: *Jianhong Wang, Yuan Zhang, Yunjie Gu, Tae-Kyun Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/27985d21f0b751b933d675930aa25022-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/27985d21f0b751b933d675930aa25022-Abstract-Conference.html)

        **Abstract**:

        Value factorisation is a useful technique for multi-agent reinforcement learning (MARL) in global reward game, however, its underlying mechanism is not yet fully understood. This paper studies a theoretical framework for value factorisation with interpretability via Shapley value theory. We generalise Shapley value to Markov convex game called Markov Shapley value (MSV) and apply it as a value factorisation method in global reward game, which is obtained by the equivalence between the two games. Based on the properties of MSV, we derive Shapley-Bellman optimality equation (SBOE) to evaluate the optimal MSV, which corresponds to an optimal joint deterministic policy. Furthermore, we propose Shapley-Bellman operator (SBO) that is proved to solve SBOE. With a stochastic approximation and some transformations, a new MARL algorithm called Shapley Q-learning (SHAQ) is established, the implementation of which is guided by the theoretical results of SBO and MSV. We also discuss the relationship between SHAQ and relevant value factorisation methods. In the experiments, SHAQ exhibits not only superior performances on all tasks but also the interpretability that agrees with the theoretical analysis. The implementation of this paper is placed on https://github.com/hsvgbkhgbv/shapley-q-learning.

        ----

        ## [430] Trajectory balance: Improved credit assignment in GFlowNets

        **Authors**: *Nikolay Malkin, Moksh Jain, Emmanuel Bengio, Chen Sun, Yoshua Bengio*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/27b51baca8377a0cf109f6ecc15a0f70-Abstract-Conference.html)

        **Abstract**:

        Generative flow networks (GFlowNets) are a method for learning a stochastic policy for generating compositional objects, such as graphs or strings, from a given unnormalized density by sequences of actions, where many possible action sequences may lead to the same object. We find previously proposed learning objectives for GFlowNets, flow matching and detailed balance, which are analogous to temporal difference learning, to be prone to inefficient credit propagation across long action sequences. We thus propose a new learning objective for GFlowNets, trajectory balance, as a more efficient alternative to previously used objectives. We prove that any global minimizer of the trajectory balance objective can define a policy that samples exactly from the target distribution. In experiments on four distinct domains, we empirically demonstrate the benefits of the trajectory balance objective for GFlowNet convergence, diversity of generated samples, and robustness to long action sequences and large action spaces.

        ----

        ## [431] Instance-Dependent Near-Optimal Policy Identification in Linear MDPs via Online Experiment Design

        **Authors**: *Andrew Wagenmaker, Kevin G. Jamieson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/27bf08fe91a31495099a0b9febcc9592-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/27bf08fe91a31495099a0b9febcc9592-Abstract-Conference.html)

        **Abstract**:

        While much progress has been made in understanding the minimax sample complexity of reinforcement learning (RL)---the complexity of learning on the worst-case'' instance---such measures of complexity often do not capture the true difficulty of learning. In practice, on aneasy'' instance, we might hope to achieve a complexity far better than that achievable on the worst-case instance. In this work we seek to understand this instance-dependent'' complexity of learning in the setting of RL with linear function approximation. We propose an algorithm, PEDEL, which achieves a fine-grained instance-dependent measure of complexity, the first of its kind in the RL with function approximation setting, thereby capturing the difficulty of learning on each particular problem instance. Through an explicit example, we show that PEDEL yields provable gains over low-regret, minimax-optimal algorithms and that such algorithms are unable to hit the instance-optimal rate. Our approach relies on a novel online experiment design-based procedure which focuses the exploration budget on thedirections'' most relevant to learning a near-optimal policy, and may be of independent interest.

        ----

        ## [432] 🏘️ ProcTHOR: Large-Scale Embodied AI Using Procedural Generation

        **Authors**: *Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Kiana Ehsani, Jordi Salvador, Winson Han, Eric Kolve, Aniruddha Kembhavi, Roozbeh Mottaghi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html)

        **Abstract**:

        Massive datasets and high-capacity models have driven many recent advancements in computer vision and natural language understanding. This work presents a platform to enable similar success stories in Embodied AI. We propose ProcTHOR, a framework for procedural generation of Embodied AI environments. ProcTHOR enables us to sample arbitrarily large datasets of diverse, interactive, customizable, and performant virtual environments to train and evaluate embodied agents across navigation, interaction, and manipulation tasks. We demonstrate the power and potential of ProcTHOR via a sample of 10,000 generated houses and a simple neural model. Models trained using only RGB images on ProcTHOR, with no explicit mapping and no human task supervision produce state-of-the-art results across 6 embodied AI benchmarks for navigation, rearrangement, and arm manipulation, including the presently running Habitat 2022, AI2-THOR Rearrangement 2022, and RoboTHOR challenges. We also demonstrate strong 0-shot results on these benchmarks, via pre-training on ProcTHOR with no fine-tuning on the downstream benchmark, often beating previous state-of-the-art systems that access the downstream training data.

        ----

        ## [433] Efficient Sampling on Riemannian Manifolds via Langevin MCMC

        **Authors**: *Xiang Cheng, Jingzhao Zhang, Suvrit Sra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/27c852e9d6c76890ca633f111c556a4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/27c852e9d6c76890ca633f111c556a4f-Abstract-Conference.html)

        **Abstract**:

        We study the task of efficiently sampling from a Gibbs distribution $d \pi^* = e^{-h} d {\text{vol}}_g$ over a Riemannian manifold $M$ via (geometric) Langevin MCMC; this algorithm involves computing exponential maps in random Gaussian directions and is efficiently implementable in practice. The key to our analysis of Langevin MCMC is a bound on the discretization error of the geometric Euler-Murayama scheme, assuming $\nabla h$ is Lipschitz and $M$ has bounded sectional curvature.  Our error bound matches the error of Euclidean Euler-Murayama in terms of its stepsize dependence.  Combined with a contraction guarantee for the geometric Langevin Diffusion under Kendall-Cranston coupling, we prove that the Langevin MCMC iterates lie within $\epsilon$-Wasserstein distance of $\pi^*$  after $\tilde{O}(\epsilon^{-2})$ steps, which matches the iteration complexity for Euclidean Langevin MCMC. Our results apply in general settings where $h$ can be nonconvex and $M$ can have negative Ricci curvature. Under additional assumptions that the Riemannian curvature tensor has bounded derivatives, and that $\pi^*$ satisfies a $CD(\cdot,\infty)$ condition, we analyze the stochastic gradient version of Langevin MCMC, and bound its iteration complexity by $\tilde{O}(\epsilon^{-2})$ as well.

        ----

        ## [434] OpenFWI: Large-scale Multi-structural Benchmark Datasets for Full Waveform Inversion

        **Authors**: *Chengyuan Deng, Shihang Feng, Hanchen Wang, Xitong Zhang, Peng Jin, Yinan Feng, Qili Zeng, Yinpeng Chen, Youzuo Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/27d3ef263c7cb8d542c4f9815a49b69b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/27d3ef263c7cb8d542c4f9815a49b69b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Full waveform inversion (FWI) is widely used in geophysics to reconstruct high-resolution velocity maps from seismic data. The recent success of data-driven FWI methods results in a rapidly increasing demand for open datasets to serve the geophysics community. We present OpenFWI, a collection of large-scale multi-structural benchmark datasets, to facilitate diversified, rigorous, and reproducible research on FWI. In particular, OpenFWI consists of $12$ datasets ($2.1$TB in total) synthesized from multiple sources. It encompasses diverse domains in geophysics (interface, fault, CO$_2$ reservoir, etc.), covers different geological subsurface structures (flat, curve, etc.), and contain various amounts of data samples (2K - 67K). It also includes a dataset for 3D FWI. Moreover, we use OpenFWI to perform benchmarking over four deep learning methods, covering both supervised and unsupervised learning regimes. Along with the benchmarks, we implement additional experiments, including physics-driven methods, complexity analysis, generalization study, uncertainty quantification, and so on, to sharpen our understanding of datasets and methods. The studies either provide valuable insights into the datasets and the performance, or uncover their current limitations. We hope OpenFWI supports prospective research on FWI and inspires future open-source efforts on AI for science. All datasets and related information can be accessed through our website at https://openfwi-lanl.github.io/

        ----

        ## [435] High-Order Pooling for Graph Neural Networks with Tensor Decomposition

        **Authors**: *Chenqing Hua, Guillaume Rabusseau, Jian Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/282967f8abaae52a452a97ee961410f3-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) are attracting growing attention due to their effectiveness and flexibility in modeling a variety of graph-structured data. Exiting GNN architectures usually adopt simple pooling operations~(\eg{} sum, average, max) when aggregating messages from a local neighborhood for updating node representation or pooling node representations from the entire graph to compute the graph representation. Though simple and effective, these linear operations do not model high-order non-linear interactions among nodes. We propose the Tensorized Graph Neural Network (tGNN), a highly expressive GNN architecture relying on tensor decomposition to model high-order non-linear node interactions. tGNN leverages the symmetric CP decomposition to efficiently parameterize permutation-invariant multilinear maps for modeling node interactions. Theoretical and empirical analysis on both node and graph classification tasks show the superiority of tGNN over competitive baselines. In particular, tGNN achieves the most solid results on two OGB node classification datasets and one OGB graph classification dataset.

        ----

        ## [436] Natural image synthesis for the retina with variational information bottleneck representation

        **Authors**: *Babak Rahmani, Demetri Psaltis, Christophe Moser*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/283f1354f1de1c53a14afe0a6740e889-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/283f1354f1de1c53a14afe0a6740e889-Abstract-Conference.html)

        **Abstract**:

        In the early visual system, high dimensional natural stimuli are encoded into the trains of neuronal spikes that transmit the information to the brain to produce perception. However, is all the visual scene information required to explain the neuronal responses? In this work, we search for answers to this question by developing a joint model of the natural visual input and neuronal responses using the Information Bottleneck (IB) framework that can represent features of the input data into a few latent variables that play a role in the prediction of the outputs. The correlations between data samples acquired from published experiments on ex-vivo retinas are accounted for in the model by a Gaussian Process (GP) prior. The proposed IB-GP model performs competitively to the state-of-the-art feedforward convolutional networks in predicting spike responses to natural stimuli. Finally, the IB-GP model is used in a closed-loop iterative process to obtain reduced-complexity inputs that elicit responses as elicited by the original stimuli. We found three properties of the retina's IB-GP model. First, the reconstructed stimuli from the latent variables show robustness in spike prediction across models. Second, surprisingly the dynamics of the high-dimensional stimuli and RGCs' responses are very well represented in the embeddings of the IB-GP model. Third, the minimum stimuli consist of different patterns: Gabor-type locally high-frequency filters, on- and off-center Gaussians, or a mixture of both. Overall, this work demonstrates that the IB-GP model provides a principled approach for joint learning of the stimuli and retina codes, capturing dynamics of the stimuli-RGCs in the latent space which could help better understand the computation of the early visual system.

        ----

        ## [437] Imitating Past Successes can be Very Suboptimal

        **Authors**: *Benjamin Eysenbach, Soumith Udatha, Russ Salakhutdinov, Sergey Levine*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/284afdc2309f9667d2d4fb9290235b0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/284afdc2309f9667d2d4fb9290235b0c-Abstract-Conference.html)

        **Abstract**:

        Prior work has proposed a simple strategy for reinforcement learning (RL): label experience with the outcomes achieved in that experience, and then imitate the relabeled experience. These outcome-conditioned imitation learning methods are appealing because of their simplicity, strong performance, and close ties with supervised learning. However, it remains unclear how these methods relate to the standard RL objective, reward maximization. In this paper, we prove that existing outcome-conditioned imitation learning methods do not necessarily improve the policy. However, we show that a simple modification results in a method that does guarantee policy improvement. Our aim is not to develop an entirely new method, but rather to explain how a variant of outcome-conditioned imitation learning can be used to maximize rewards

        ----

        ## [438] A Communication-efficient Algorithm with Linear Convergence for Federated Minimax Learning

        **Authors**: *Zhenyu Sun, Ermin Wei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28553688c204ddbb06a51e00684f8bb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28553688c204ddbb06a51e00684f8bb7-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study a large-scale multi-agent minimax optimization problem, which models many interesting applications in statistical learning and game theory, including Generative Adversarial Networks (GANs). The overall objective is a sum of agents' private local objective functions. We focus on the federated setting, where agents can perform local computation and communicate with a central server. Most existing federated minimax algorithms either require communication per iteration or lack performance guarantees with the exception of Local Stochastic Gradient Descent Ascent (SGDA), a multiple-local-update descent ascent algorithm which guarantees convergence under a diminishing stepsize. By analyzing Local SGDA under the ideal condition of no gradient noise, we show that generally it cannot guarantee exact convergence with constant stepsizes and thus suffers from slow rates of convergence. To tackle this issue, we propose FedGDA-GT, an improved Federated (Fed) Gradient Descent Ascent (GDA) method based on Gradient Tracking (GT). When local objectives are Lipschitz smooth and strongly-convex-strongly-concave, we prove that FedGDA-GT converges linearly with a constant stepsize to global $\epsilon$-approximation solution with $\mathcal{O}(\log (1/\epsilon))$ rounds of communication, which matches the time complexity of centralized GDA method. Then, we analyze the general distributed minimax problem from a statistical aspect, where the overall objective approximates a true population minimax risk by empirical samples. We provide generalization bounds for learning with this objective through Rademacher complexity analysis. Finally, we numerically show that FedGDA-GT outperforms Local SGDA.

        ----

        ## [439] Dynamic Graph Neural Networks Under Spatio-Temporal Distribution Shift

        **Authors**: *Zeyang Zhang, Xin Wang, Ziwei Zhang, Haoyang Li, Zhou Qin, Wenwu Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2857242c9e97de339ce642e75b15ff24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2857242c9e97de339ce642e75b15ff24-Abstract-Conference.html)

        **Abstract**:

        Dynamic graph neural networks (DyGNNs) have demonstrated powerful predictive abilities by exploiting graph structural and temporal dynamics. However, the existing DyGNNs fail to handle distribution shifts, which naturally exist in dynamic graphs, mainly because the patterns exploited by DyGNNs may be variant with respect to labels under distribution shifts. In this paper, we propose to handle spatio-temporal distribution shifts in dynamic graphs by discovering and utilizing {\it invariant patterns}, i.e., structures and features whose predictive abilities are stable across distribution shifts, which faces two key challenges: 1) How to discover the complex variant and invariant spatio-temporal patterns in dynamic graphs, which involve both time-varying graph structures and node features. 2) How to handle spatio-temporal distribution shifts with the discovered variant and invariant patterns. To tackle these challenges, we propose the Disentangled Intervention-based Dynamic graph Attention networks (DIDA). Our proposed method can effectively handle spatio-temporal distribution shifts in dynamic graphs by discovering and fully utilizing invariant spatio-temporal patterns. Specifically, we first propose a disentangled spatio-temporal attention network to capture the variant and invariant patterns.  Then, we design a spatio-temporal intervention mechanism to create multiple interventional distributions by sampling and reassembling variant patterns across neighborhoods and time stamps to eliminate the spurious impacts of variant patterns.  Lastly, we propose an invariance regularization term to minimize the variance of predictions in intervened distributions so that our model can make predictions based on invariant patterns with stable predictive abilities and therefore handle distribution shifts. Experiments on three real-world datasets and one synthetic dataset demonstrate the superiority of our method over state-of-the-art baselines under distribution shifts. Our work is the first study of spatio-temporal distribution shifts in dynamic graphs, to the best of our knowledge.

        ----

        ## [440] Biologically Inspired Dynamic Thresholds for Spiking Neural Networks

        **Authors**: *Jianchuan Ding, Bo Dong, Felix Heide, Yufei Ding, Yunduo Zhou, Baocai Yin, Xin Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2858f8c8683aaa8c12d487354cf328dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2858f8c8683aaa8c12d487354cf328dc-Abstract-Conference.html)

        **Abstract**:

        The dynamic membrane potential threshold, as one of the essential properties of a biological neuron, is a spontaneous regulation mechanism that maintains neuronal homeostasis, i.e., the constant overall spiking firing rate of a neuron. As such, the neuron firing rate is regulated by a dynamic spiking threshold, which has been extensively studied in biology. Existing work in the machine learning community does not employ bioinspired spiking threshold schemes. This work aims at bridging this gap by introducing a novel bioinspired dynamic energy-temporal threshold (BDETT) scheme for spiking neural networks (SNNs). The proposed BDETT scheme mirrors two bioplausible observations: a dynamic threshold has 1) a positive correlation with the average membrane potential and 2) a negative correlation with the preceding rate of depolarization. We validate the effectiveness of the proposed BDETT on robot obstacle avoidance and continuous control tasks under both normal conditions and various degraded conditions, including noisy observations, weights, and dynamic environments. We find that the BDETT outperforms existing static and heuristic threshold approaches by significant margins in all tested conditions, and we confirm that the proposed bioinspired dynamic threshold scheme offers homeostasis to SNNs in complex real-world tasks.

        ----

        ## [441] GraphQNTK: Quantum Neural Tangent Kernel for Graph Data

        **Authors**: *Yehui Tang, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/285b06e0dd856f20591b0a5beb954151-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/285b06e0dd856f20591b0a5beb954151-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) and Graph Kernels (GKs) are two fundamental tools used to analyze graph-structured data.  Efforts have been recently made in developing a composite graph learning architecture combining the expressive power of GNNs and the transparent trainability of GKs. However, learning efficiency on these models should be carefully considered as the huge computation overhead. Besides, their convolutional methods are often straightforward and introduce severe loss of graph structure information. In this paper, we design a novel quantum graph learning model to characterize the structural information while using quantum parallelism to improve computing efficiency. Specifically, a quantum algorithm is proposed to approximately estimate the neural tangent kernel of the underlying graph neural network where a multi-head quantum attention mechanism is introduced to properly incorporate semantic similarity information of nodes into the model. We empirically show that our method achieves competitive performance on several graph classification benchmarks, and theoretical analysis is provided to demonstrate the superiority of our quantum algorithm. Source code is available at \url{https://github.com/abel1231/graphQNTK}.

        ----

        ## [442] Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline

        **Authors**: *Penghao Wu, Xiaosong Jia, Li Chen, Junchi Yan, Hongyang Li, Yu Qiao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/286a371d8a0a559281f682f8fbf89834-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/286a371d8a0a559281f682f8fbf89834-Abstract-Conference.html)

        **Abstract**:

        Current end-to-end autonomous driving methods either run a controller based on a planned trajectory or perform control prediction directly, which have spanned two separately studied lines of research. Seeing their potential mutual benefits to each other, this paper takes the initiative to explore the combination of these two well-developed worlds. Specifically, our integrated approach has two branches for trajectory planning and direct control, respectively. The trajectory branch predicts the future trajectory, while the control branch involves a novel multi-step prediction scheme such that the relationship between current actions and future states can be reasoned. The two branches are connected so that the control branch receives corresponding guidance from the trajectory branch at each time step. The outputs from two branches are then fused to achieve complementary advantages. Our results are evaluated in the closed-loop urban driving setting with challenging scenarios using the CARLA simulator. Even with a monocular camera input, the proposed approach ranks first on the official CARLA Leaderboard, outperforming other complex candidates with multiple sensors or fusion mechanisms by a large margin. The sourcecode is publicly available at https://github.com/OpenPerceptionX/TCP

        ----

        ## [443] 360-MLC: Multi-view Layout Consistency for Self-training and Hyper-parameter Tuning

        **Authors**: *Bolivar Solarte, Chin-Hsuan Wu, Yueh-Cheng Liu, Yi-Hsuan Tsai, Min Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/286e7ab0ce6a68282394c92361c27b57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/286e7ab0ce6a68282394c92361c27b57-Abstract-Conference.html)

        **Abstract**:

        We present 360-MLC, a self-training method based on multi-view layout consistency for finetuning monocular room-layout models using unlabeled 360-images only. This can be valuable in practical scenarios where a pre-trained model needs to be adapted to a new data domain without using any ground truth annotations. Our simple yet effective assumption is that multiple layout estimations in the same scene must define a consistent geometry regardless of their camera positions. Based on this idea, we leverage a pre-trained model to project estimated layout boundaries from several camera views into the 3D world coordinate. Then, we re-project them back to the spherical coordinate and build a probability function, from which we sample the pseudo-labels for self-training. To handle unconfident pseudo-labels, we evaluate the variance in the re-projected boundaries as an uncertainty value to weight each pseudo-label in our loss function during training. In addition, since ground truth annotations are not available during training nor in testing, we leverage the entropy information in multiple layout estimations as a quantitative metric to measure the geometry consistency of the scene, allowing us to evaluate any layout estimator for hyper-parameter tuning, including model selection without ground truth annotations. Experimental results show that our solution achieves favorable performance against state-of-the-art methods when self-training from three publicly available source datasets to a unique, newly labeled dataset consisting of multi-view images of the same scenes.

        ----

        ## [444] Distributed Optimization for Overparameterized Problems: Achieving Optimal Dimension Independent Communication Complexity

        **Authors**: *Bingqing Song, Ioannis C. Tsaknakis, Chung-Yiu Yau, Hoi-To Wai, Mingyi Hong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28795419a644f41ede3fa058b13fc622-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28795419a644f41ede3fa058b13fc622-Abstract-Conference.html)

        **Abstract**:

        Decentralized optimization are playing an important role in applications such as training large machine learning models, among others. Despite its superior practical performance, there has been some lack of fundamental understanding about its theoretical properties. In this work, we address the following open research question: To train an overparameterized model over a set of distributed nodes, what is the {\it minimum} communication overhead (in terms of the bits got exchanged) that the system needs to sustain, while still achieving (near) zero training loss? We show that for a class of overparameterized models where the number of parameters $D$ is much larger than the total data samples $N$, the best possible communication complexity is ${\Omega}(N)$, which is independent of the problem dimension $D$. Further, for a few specific overparameterized models (i.e., the linear regression, and certain multi-layer neural network with one wide layer), we develop a set of algorithms which uses certain linear compression followed by adaptive quantization, and show that they achieve dimension independent, and sometimes near optimal, communication complexity. To our knowledge, this is the first time that dimension independent communication complexity has been shown for distributed optimization.

        ----

        ## [445] Falsification before Extrapolation in Causal Effect Estimation

        **Authors**: *Zeshan M. Hussain, Michael Oberst, Ming-Chieh Shih, David A. Sontag*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28b5dfc51e5ae12d84fb7c6172a00df4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28b5dfc51e5ae12d84fb7c6172a00df4-Abstract-Conference.html)

        **Abstract**:

        Randomized Controlled Trials (RCTs) represent a gold standard when developing policy guidelines. However, RCTs are often narrow, and lack data on broader populations of interest.  Causal effects in these populations are often estimated using observational datasets, which may suffer from unobserved confounding and selection bias.  Given a set of observational estimates (e.g., from multiple studies), we propose a meta-algorithm that attempts to reject observational estimates that are biased. We do so using validation effects, causal effects that can be inferred from both RCT and observational data. After rejecting estimators that do not pass this test, we generate conservative confidence intervals on the extrapolated causal effects for subgroups not observed in the RCT. Under the assumption that at least one observational estimator is asymptotically normal and consistent for both the validation and extrapolated effects, we provide guarantees on the coverage probability of the intervals output by our algorithm. To facilitate hypothesis testing in settings where causal effect transportation across datasets is necessary, we give conditions under which a doubly-robust estimator of group average treatment effects is asymptotically normal, even when flexible machine learning methods are used for estimation of nuisance parameters. We illustrate the properties of our approach on semi-synthetic experiments based on the IHDP dataset, and show that it compares favorably to standard meta-analysis techniques.

        ----

        ## [446] Surprise Minimizing Multi-Agent Learning with Energy-based Models

        **Authors**: *Karush Suri, Xiao Qi Shi, Konstantinos N. Plataniotis, Yuri A. Lawryshyn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28dad4a70f748a2980998d3ed0f1b8d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28dad4a70f748a2980998d3ed0f1b8d2-Abstract-Conference.html)

        **Abstract**:

        Multi-Agent Reinforcement Learning (MARL) has demonstrated significant suc2 cess by virtue of collaboration across agents. Recent work, on the other hand, introduces surprise which quantifies the degree of change in an agentâ€™s environ4 ment. Surprise-based learning has received significant attention in the case of single-agent entropic settings but remains an open problem for fast-paced dynamics in multi-agent scenarios. A potential alternative to address surprise may be realized through the lens of free-energy minimization. We explore surprise minimization in multi-agent learning by utilizing the free energy across all agents in a multi-agent system. A temporal Energy-Based Model (EBM) represents an estimate of surprise which is minimized over the joint agent distribution. Our formulation of the EBM is theoretically akin to the minimum conjugate entropy objective and highlights suitable convergence towards minimum surprising states. We further validate our theoretical claims in an empirical study of multi-agent tasks demanding collabora14 tion in the presence of fast-paced dynamics. Our implementation and agent videos are available at the Project Webpage.

        ----

        ## [447] Geodesic Self-Attention for 3D Point Clouds

        **Authors**: *Zhengyu Li, Xuan Tang, Zihao Xu, Xihao Wang, Hui Yu, Mingsong Chen, Xian Wei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28e4ee96c94e31b2d040b4521d2b299e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28e4ee96c94e31b2d040b4521d2b299e-Abstract-Conference.html)

        **Abstract**:

        Due to the outstanding competence in capturing long-range relationships, self-attention mechanism has achieved remarkable progress in point cloud tasks. Nevertheless, point cloud object often has complex non-Euclidean spatial structures, with the behavior changing dynamically and unpredictably. Most current self-attention modules highly rely on the dot product multiplication in Euclidean space, which cannot capture internal non-Euclidean structures of point cloud objects, especially the long-range relationships along the curve of the implicit manifold surface represented by point cloud objects. To address this problem, in this paper, we introduce a novel metric on the Riemannian manifold to capture the long-range geometrical dependencies of point cloud objects to replace traditional self-attention modules, namely, the Geodesic Self-Attention (GSA) module. Our approach achieves state-of-the-art performance compared to point cloud Transformers on object classification, few-shot classification and part segmentation benchmarks.

        ----

        ## [448] Test Time Adaptation via Conjugate Pseudo-labels

        **Authors**: *Sachin Goyal, Mingjie Sun, Aditi Raghunathan, J. Zico Kolter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28e9eff897f98372409b40ae1ed3ea4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28e9eff897f98372409b40ae1ed3ea4c-Abstract-Conference.html)

        **Abstract**:

        Test-time adaptation (TTA) refers to adapting neural networks to distribution shifts, specifically with just access to unlabeled test samples from the new domain at test-time. Prior TTA methods optimize over unsupervised objectives such as the entropy of model predictions in TENT (Wang et al., 2021), but it is unclear what exactly makes a good TTA loss. In this paper, we start by presenting a surprising phenomenon: if we attempt to $\textit{meta-learn}$ the ``best'' possible TTA loss over a wide class of functions, then we recover a function that is $\textit{remarkably}$ similar to (a temperature-scaled version of) the softmax-entropy employed by TENT. This only holds, however, if the classifier we are adapting is trained via cross-entropy loss; if the classifier is trained via squared loss, a different ``best'' TTA loss emerges.To explain this phenomenon, we analyze test-time adaptation through the lens of the training losses's $\textit{convex conjugate}$.  We show that under natural conditions, this (unsupervised) conjugate function can be viewed as a good local approximation to the original supervised loss and indeed, it recovers the ``best'' losses found by meta-learning. This leads to a generic recipe than be used to find  a good TTA loss for $\textit{any}$ given supervised training loss function of a general class. Empirically, our approach dominates other TTA alternatives over a wide range of domain adaptation benchmarks. Our approach is particularly of interest when applied to classifiers trained with $\textit{novel}$ loss functions, e.g., the recently-proposed PolyLoss (Leng et al., 2022) function, where it differs substantially from (and outperforms) an entropy-based loss. Further, we show that our conjugate based approach can also be interpreted as a kind of self-training using a very specific soft label, which we refer to as the $\textit{conjugate pseudo-label}$. Overall, therefore, our method provides a broad framework for better understanding and improving test-time adaptation. Code is available at https://github.com/locuslab/tta_conjugate.

        ----

        ## [449] SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG

        **Authors**: *Reinmar J. Kobler, Jun-ichiro Hirayama, Qibin Zhao, Motoaki Kawanabe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28ef7ee7cd3e03093acc39e1272411b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28ef7ee7cd3e03093acc39e1272411b7-Abstract-Conference.html)

        **Abstract**:

        Electroencephalography (EEG) provides access to neuronal dynamics non-invasively with millisecond resolution, rendering it a viable method in neuroscience and healthcare. However, its utility is limited as current EEG technology does not generalize well across domains (i.e., sessions and subjects) without expensive supervised re-calibration. Contemporary methods cast this transfer learning (TL) problem as a multi-source/-target unsupervised domain adaptation (UDA) problem and address it with deep learning or shallow, Riemannian geometry aware alignment methods. Both directions have, so far, failed to consistently close the performance gap to state-of-the-art domain-specific methods based on tangent space mapping (TSM) on the symmetric, positive definite (SPD) manifold.Here, we propose a machine learning framework that enables, for the first time, learning domain-invariant TSM models in an end-to-end fashion. To achieve this, we propose a new building block for geometric deep learning, which we denote  SPD domain-specific momentum batch normalization (SPDDSMBN). A SPDDSMBN layer can transform domain-specific SPD inputs into domain-invariant SPD outputs, and can be readily applied to multi-source/-target and online UDA scenarios. In extensive experiments with 6 diverse EEG brain-computer interface (BCI) datasets, we obtain state-of-the-art performance in inter-session and -subject TL with a simple, intrinsically interpretable network architecture, which we denote TSMNet. Code: https://github.com/rkobler/TSMNet

        ----

        ## [450] AVLEN: Audio-Visual-Language Embodied Navigation in 3D Environments

        **Authors**: *Sudipta Paul, Amit Roy-Chowdhury, Anoop Cherian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/28f699175783a2c828ae74d53dd3da20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/28f699175783a2c828ae74d53dd3da20-Abstract-Conference.html)

        **Abstract**:

        Recent years have seen embodied visual navigation advance in two distinct directions: (i) in equipping the AI agent to follow natural language instructions, and (ii) in making the navigable world multimodal, e.g., audio-visual navigation. However, the real world is not only multimodal, but also often complex, and thus in spite of these advances, agents still need to understand the uncertainty in their actions and seek instructions to navigate. To this end, we present AVLEN -- an interactive agent for Audio-Visual-Language Embodied Navigation. Similar to audio-visual navigation tasks, the goal of our embodied agent is to localize an audio event via navigating the 3D visual world; however, the agent may also seek help from a human (oracle), where the assistance is provided in free-form natural language. To realize these abilities, AVLEN uses a multimodal hierarchical reinforcement learning backbone that learns: (a) high-level policies to choose either audio-cues for navigation or to query the oracle, and (b) lower-level policies to select navigation actions based on its audio-visual and language inputs. The policies are trained via rewarding for the success on the navigation task while minimizing the number of queries to the oracle. To empirically evaluate AVLEN, we present experiments on the SoundSpaces framework for semantic audio-visual navigation tasks. Our results show that equipping the agent to ask for help leads to a clear improvement in performances, especially in challenging cases, e.g., when the sound is unheard during training or in the presence of distractor sounds.

        ----

        ## [451] Semantic uncertainty intervals for disentangled latent spaces

        **Authors**: *Swami Sankaranarayanan, Anastasios Angelopoulos, Stephen Bates, Yaniv Romano, Phillip Isola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/290141d6bfd7ea4d3f4483d126609bf6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/290141d6bfd7ea4d3f4483d126609bf6-Abstract-Conference.html)

        **Abstract**:

        Meaningful uncertainty quantification in computer vision requires reasoning about semantic information---say, the hair color of the person in a photo or the location of a car on the street. To this end, recent breakthroughs in generative modeling allow us to represent semantic information in disentangled latent spaces, but providing uncertainties on the semantic latent variables has remained challenging. In this work, we provide principled uncertainty intervals that are guaranteed to contain the true semantic factors for any underlying generative model. The method does the following: (1) it uses quantile regression to output a heuristic uncertainty interval for each element in the latent space (2) calibrates these uncertainties such that they contain the true value of the latent for a new, unseen input. The endpoints of these calibrated intervals can then be propagated through the generator to produce interpretable uncertainty visualizations for each semantic factor. This technique reliably communicates semantically meaningful, principled, and instance-adaptive uncertainty in inverse problems like image super-resolution and image completion. Project page: https://swamiviv.github.io/semanticuncertaintyintervals/

        ----

        ## [452] A Fast Scale-Invariant Algorithm for Non-negative Least Squares with Non-negative Data

        **Authors**: *Jelena Diakonikolas, Chenghui Li, Swati Padmanabhan, Chaobing Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29021b06afa4c648ee438584f7ef3e7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29021b06afa4c648ee438584f7ef3e7e-Abstract-Conference.html)

        **Abstract**:

        Nonnegative (linear) least square problems are a fundamental class of problems that is well-studied in statistical learning and for which solvers have been implemented in many of the standard programming languages used within the machine learning community. The existing off-the-shelf solvers view the non-negativity constraint in these problems as an obstacle and, compared to unconstrained least squares, perform additional effort to address it. However, in many of the typical applications, the data itself is nonnegative as well, and we show that the nonnegativity in this case makes the problem easier. In particular, while the worst-case dimension-independent oracle complexity of unconstrained least squares problems necessarily scales with one of the data matrix constants (typically the spectral norm) and these problems are solved to additive error, we show that nonnegative least squares problems with nonnegative data are solvable to  multiplicative error and with complexity that is independent of any matrix constants. The algorithm we introduce is accelerated and based on a primal-dual perspective. We further show how to provably obtain linear convergence using adaptive restart coupled with our method and demonstrate its effectiveness on large-scale data via numerical experiments.

        ----

        ## [453] Self-supervised Amodal Video Object Segmentation

        **Authors**: *Jian Yao, Yuxin Hong, Chiyu Wang, Tianjun Xiao, Tong He, Francesco Locatello, David P. Wipf, Yanwei Fu, Zheng Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29171e32e652ac40244e96bb8529cb44-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29171e32e652ac40244e96bb8529cb44-Abstract-Conference.html)

        **Abstract**:

        Amodal perception requires inferring the full shape of an object that is partially occluded. This task is particularly challenging on two levels: (1) it requires more information than what is contained in the instant retina or imaging sensor, (2) it is difficult to obtain enough well-annotated amodal labels for supervision. To this end, this paper develops a new framework of Self-supervised amodal Video object segmentation (SaVos). Our method efficiently leverages the visual information of video temporal sequences to infer the amodal mask of objects. The key intuition is that the occluded part of an object can be explained away if that part is visible in other frames, possibly deformed as long as the deformation can be reasonably learned. Accordingly, we derive a novel self-supervised learning paradigm that efficiently utilizes the visible object parts as the supervision to guide the training on videos. In addition to learning type prior to complete masks for known types, SaVos also learns the spatiotemporal prior, which is also useful for the amodal task and could generalize to unseen types. The proposed framework achieves the state-of-the-art performance on the synthetic amodal segmentation benchmark FISHBOWL and the real world benchmark KINS-Video-Car. Further, it lends itself well to being transferred to novel distributions using test-time adaptation, outperforming existing models even after the transfer to a new distribution.

        ----

        ## [454] Beyond the Return: Off-policy Function Estimation under User-specified Error-measuring Distributions

        **Authors**: *Audrey Huang, Nan Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29241c7caf8230e47cefca6ad3a91f21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29241c7caf8230e47cefca6ad3a91f21-Abstract-Conference.html)

        **Abstract**:

        Off-policy evaluation often refers to two related tasks: estimating the expected return of a policy and estimating its value function (or other functions of interest, such as density ratios). While recent works on marginalized importance sampling (MIS) show that the former can enjoy provable guarantees under realizable function approximation, the latter is only known to be feasible under much stronger assumptions such as prohibitively expressive discriminators. In this work, we provide guarantees for off-policy function estimation under only realizability, by imposing proper regularization on the MIS objectives. Compared to commonly used regularization in MIS, our regularizer is much more flexible and can account for an arbitrary user-specified distribution, under which the learned function will be close to the groundtruth. We provide exact characterization of the optimal dual solution that needs to be realized by the discriminator class, which determines the data-coverage assumption in the case of value-function learning. As another surprising observation, the regularizer can be altered to relax the data-coverage requirement, and completely eliminate it in the ideal case with strong side information.

        ----

        ## [455] Disentangling Transfer in Continual Reinforcement Learning

        **Authors**: *Maciej Wolczyk, Michal Zajac, Razvan Pascanu, Lukasz Kucinski, Piotr Milos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2938ad0434a6506b125d8adaff084a4a-Abstract-Conference.html)

        **Abstract**:

        The ability of continual learning systems to transfer knowledge from previously seen tasks in order to maximize performance on new tasks is a significant challenge for the field, limiting the applicability of continual learning solutions to realistic scenarios. Consequently, this study aims to broaden our understanding of transfer and its driving forces in the specific case of continual reinforcement learning. We adopt SAC as the underlying RL algorithm and Continual World as a suite of continuous control tasks. We systematically study how different components of SAC (the actor and the critic, exploration, and data) affect transfer efficacy, and we provide recommendations regarding various modeling options. The best set of choices, dubbed ClonEx-SAC, is evaluated on the recent Continual World benchmark. ClonEx-SAC achieves 87% final success rate compared to 80% of PackNet, the best method in the benchmark. Moreover, the transfer grows from 0.18 to 0.54 according to the metric provided by Continual World.

        ----

        ## [456] Meta-Learning with Self-Improving Momentum Target

        **Authors**: *Jihoon Tack, Jongjin Park, Hankook Lee, Jaeho Lee, Jinwoo Shin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29440165fee0471389ba3f80a7b3f95f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29440165fee0471389ba3f80a7b3f95f-Abstract-Conference.html)

        **Abstract**:

        The idea of using a separately trained target model (or teacher) to improve the performance of the student model has been increasingly popular in various machine learning domains, and meta-learning is no exception; a recent discovery shows that utilizing task-wise target models can significantly boost the generalization performance. However, obtaining a target model for each task can be highly expensive, especially when the number of tasks for meta-learning is large. To tackle this issue, we propose a simple yet effective method, coined Self-improving Momentum Target (SiMT). SiMT generates the target model by adapting from the temporal ensemble of the meta-learner, i.e., the momentum network. This momentum network and its task-specific adaptations enjoy a favorable generalization performance, enabling self-improving of the meta-learner through knowledge distillation. Moreover, we found that perturbing parameters of the meta-learner, e.g., dropout, further stabilize this self-improving process by preventing fast convergence of the distillation loss during meta-training. Our experimental results demonstrate that SiMT brings a significant performance gain when combined with a wide range of meta-learning methods under various applications, including few-shot regression, few-shot classification, and meta-reinforcement learning. Code is available at https://github.com/jihoontack/SiMT.

        ----

        ## [457] Online Allocation and Learning in the Presence of Strategic Agents

        **Authors**: *Steven Yin, Shipra Agrawal, Assaf Zeevi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29455889cb812d661573f6fbc800bd9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29455889cb812d661573f6fbc800bd9e-Abstract-Conference.html)

        **Abstract**:

        We study the problem of allocating $T$ sequentially arriving items among $n$ homogenous agents under the constraint that each agent must receive a prespecified fraction of all items, with the objective of maximizing the agents' total valuation of items allocated to them. The agents' valuations for the item in each round are assumed to be i.i.d. but their distribution is apriori unknown to the central planner.vTherefore, the central planner needs to implicitly learn these distributions from the observed values in order to pick a good allocation policy. However, an added challenge here is that the agents are strategic with incentives to misreport their valuations in order to receive better allocations. This sets our work apart both from the online auction mechanism design settings which typically assume known valuation distributions and/or involve payments, and from the online learning settings that do not consider strategic agents. To that end, our main contribution is an online learning based allocation mechanism that is approximately Bayesian incentive compatible, and when all agents are truthful, guarantees a sublinear regret for individual agents' utility compared to that under the optimal offline allocation policy.

        ----

        ## [458] Learning Contrastive Embedding in Low-Dimensional Space

        **Authors**: *Shuo Chen, Chen Gong, Jun Li, Jian Yang, Gang Niu, Masashi Sugiyama*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/297f7c6c56af81239f7c47d21558b75a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/297f7c6c56af81239f7c47d21558b75a-Abstract-Conference.html)

        **Abstract**:

        Contrastive learning (CL) pretrains feature embeddings to scatter instances in the feature space so that the training data can be well discriminated. Most existing CL techniques usually encourage learning such feature embeddings in the highdimensional space to maximize the instance discrimination. However, this practice may lead to undesired results where the scattering instances are sparsely distributed in the high-dimensional feature space, making it difficult to capture the underlying similarity between pairwise instances. To this end, we propose a novel framework called contrastive learning with low-dimensional reconstruction (CLLR), which adopts a regularized projection layer to reduce the dimensionality of the feature embedding. In CLLR, we build the sparse / low-rank regularizer to adaptively reconstruct a low-dimensional projection space while preserving the basic objective for instance discrimination, and thus successfully learning contrastive embeddings that alleviate the above issue. Theoretically, we prove a tighter error bound for CLLR; empirically, the superiority of CLLR is demonstrated across multiple domains. Both theoretical and experimental results emphasize the significance of learning low-dimensional contrastive embeddings.

        ----

        ## [459] Near-Optimal Randomized Exploration for Tabular Markov Decision Processes

        **Authors**: *Zhihan Xiong, Ruoqi Shen, Qiwen Cui, Maryam Fazel, Simon S. Du*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/298c3e32d7d402189444be2ff5d19979-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/298c3e32d7d402189444be2ff5d19979-Abstract-Conference.html)

        **Abstract**:

        We study algorithms using randomized value functions for exploration in reinforcement learning. This type of algorithms enjoys appealing empirical performance. We show that when we use 1) a single random seed in each episode, and 2) a Bernstein-type magnitude of noise, we obtain a worst-case $\widetilde{O}\left(H\sqrt{SAT}\right)$ regret bound for episodic time-inhomogeneous Markov Decision Process where $S$ is the size of state space, $A$ is the size of action space, $H$ is the planning horizon and $T$ is the number of interactions. This bound polynomially improves all existing bounds for algorithms based on randomized value functions, and for the first time, matches the $\Omega\left(H\sqrt{SAT}\right)$ lower bound up to logarithmic factors. Our result highlights that randomized exploration can be near-optimal, which was previously achieved only by optimistic algorithms. To achieve the desired result, we develop 1) a new clipping operation to ensure both the probability of being optimistic and the probability of being pessimistic are lower bounded by a constant, and 2) a new recursive  formula for the absolute value of estimation errors to analyze the regret.

        ----

        ## [460] Defending Against Adversarial Attacks via Neural Dynamic System

        **Authors**: *Xiyuan Li, Zou Xin, Weiwei Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html)

        **Abstract**:

        Although deep neural networks (DNN) have achieved great success, their applications in safety-critical areas are hindered due to their vulnerability to adversarial attacks. Some recent works have accordingly proposed to enhance the robustness of DNN from a dynamic system perspective. Following this line of inquiry, and inspired by the asymptotic stability of the general nonautonomous dynamical system, we propose to make each clean instance be the asymptotically stable equilibrium points of a slowly time-varying system in order to defend against adversarial attacks. We present a theoretical guarantee that if a clean instance is an asymptotically stable equilibrium point and the adversarial instance is in the neighborhood of this point, the asymptotic stability will reduce the adversarial noise to bring the adversarial instance close to the clean instance. Motivated by our theoretical results, we go on to propose a nonautonomous neural ordinary differential equation (ASODE) and place constraints on its corresponding linear time-variant system to make all clean instances act as its asymptotically stable equilibrium points. Our analysis suggests that the constraints can be converted to regularizers in implementation. The experimental results show that ASODE improves robustness against adversarial attacks and outperforms state-of-the-art methods.

        ----

        ## [461] On the Robustness of Graph Neural Diffusion to Topology Perturbations

        **Authors**: *Yang Song, Qiyu Kang, Sijie Wang, Kai Zhao, Wee Peng Tay*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29a0ea49a103a233b17c0705cdeccb66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29a0ea49a103a233b17c0705cdeccb66-Abstract-Conference.html)

        **Abstract**:

        Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.

        ----

        ## [462] Few-shot Relational Reasoning via Connection Subgraph Pretraining

        **Authors**: *Qian Huang, Hongyu Ren, Jure Leskovec*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29de5722d7d2a6946c08d0c0162a1c71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29de5722d7d2a6946c08d0c0162a1c71-Abstract-Conference.html)

        **Abstract**:

        Few-shot knowledge graph (KG) completion task aims to perform inductive reasoning over the KG: given only a few support triplets of a new relation $\bowtie$ (e.g., (chop,$\bowtie$,kitchen), (read,$\bowtie$,library), the goal is to predict the query triplets of the same unseen relation $\bowtie$, e.g., (sleep,$\bowtie$,?). Current approaches cast the problem in a meta-learning framework, where the model needs to be first jointly trained over many  training few-shot tasks, each being defined by its own relation, so that learning/prediction on the  target few-shot task can be effective. However, in real-world KGs, curating many training tasks is a challenging  ad hoc process. Here we propose Connection Subgraph Reasoner (CSR), which can make predictions for the target few-shot task directly without the need for pre-training on the human curated set of training tasks. The key to CSR is that we explicitly model a shared connection subgraph between support and query triplets, as inspired by the principle of eliminative induction. To adapt to specific KG, we design a corresponding self-supervised pretraining scheme with the objective of reconstructing automatically sampled connection subgraphs. Our pretrained model can then be directly applied to target few-shot tasks on without the need for training few-shot tasks. Extensive experiments on real KGs, including NELL, FB15K-237, and ConceptNet, demonstrate the effectiveness of our framework: we show that even a learning-free implementation of CSR can already perform competitively to existing methods on target few-shot tasks; with pretraining, CSR can achieve significant gains of up to 52% on the more challenging inductive few-shot tasks where the entities are also unseen during (pre)training.

        ----

        ## [463] Equivariant Networks for Zero-Shot Coordination

        **Authors**: *Darius Muglich, Christian Schröder de Witt, Elise van der Pol, Shimon Whiteson, Jakob N. Foerster*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29e4b51d45dc8f534260adc45b587363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29e4b51d45dc8f534260adc45b587363-Abstract-Conference.html)

        **Abstract**:

        Successful coordination in Dec-POMDPs requires agents to adopt robust strategies and interpretable styles of play for their partner. A common failure mode is symmetry breaking, when agents arbitrarily converge on one out of many equivalent but mutually incompatible policies. Commonly these examples include partial observability, e.g. waving your right hand vs. left hand to convey a covert message. In this paper, we present a novel equivariant network architecture for use in Dec-POMDPs that prevents the agent from learning policies which break symmetries, doing so more effectively than prior methods. Our method also acts as a "coordination-improvement operator" for generic, pre-trained policies, and thus may be applied at test-time in conjunction with any self-play algorithm. We provide theoretical guarantees of our work and test on the AI benchmark task of Hanabi, where we demonstrate our methods outperforming other symmetry-aware baselines in zero-shot coordination, as well as able to improve the coordination ability of a variety of pre-trained policies. In particular, we show our method can be used to improve on the state of the art for zero-shot coordination on the Hanabi benchmark.

        ----

        ## [464] Randomized Sketches for Clustering: Fast and Optimal Kernel $k$-Means

        **Authors**: *Rong Yin, Yong Liu, Weiping Wang, Dan Meng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/29ef811e72b2b97cf18dd5d866b0f472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/29ef811e72b2b97cf18dd5d866b0f472-Abstract-Conference.html)

        **Abstract**:

        Kernel $k$-means is arguably one of the most common approaches to clustering. In this paper, we investigate the efficiency of kernel $k$-means combined with randomized sketches in terms of both statistical analysis and computational requirements. More precisely, we propose a unified randomized sketches framework to kernel $k$-means and investigate its excess risk bounds, obtaining the state-of-the-art risk bound with only a fraction of computations. Indeed, we prove that it suffices to choose the sketch dimension $\Omega(\sqrt{n})$ to obtain the same accuracy of exact kernel $k$-means with greatly reducing the computational costs, for sub-Gaussian sketches, the randomized orthogonal system (ROS) sketches, and Nystr\"{o}m kernel $k$-means, where $n$ is the number of samples. To the best of our knowledge, this is the first result of this kind for unsupervised learning. Finally, the numerical experiments on simulated data and real-world datasets validate our theoretical analysis.

        ----

        ## [465] Quantile Constrained Reinforcement Learning: A Reinforcement Learning Framework Constraining Outage Probability

        **Authors**: *Whiyoung Jung, Myungsik Cho, Jongeui Park, Youngchul Sung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2a07348a6a7b2c208ab5cb1ee0e78ab5-Abstract-Conference.html)

        **Abstract**:

        Constrained reinforcement learning (RL) is an area of RL whose objective is to find an optimal policy that maximizes expected cumulative return while satisfying a given constraint. Most of the previous constrained RL works consider expected cumulative sum cost as the constraint. However, optimization with this constraint cannot guarantee a target probability of outage event that the cumulative sum cost exceeds a given threshold. This paper proposes a framework, named Quantile Constrained RL (QCRL), to constrain the quantile of the distribution of the cumulative sum cost that is a necessary and sufficient condition to satisfy the outage constraint. This is the first work that tackles the issue of applying the policy gradient theorem to the quantile and provides theoretical results for approximating the gradient of the quantile. Based on the derived theoretical results and the technique of the Lagrange multiplier, we construct a constrained RL algorithm named Quantile Constrained Policy Optimization (QCPO). We use distributional RL with the Large Deviation Principle (LDP) to estimate quantiles and tail probability of the cumulative sum cost for the implementation of QCPO. The implemented algorithm satisfies the outage probability constraint after the training period.

        ----

        ## [466] Procedural Image Programs for Representation Learning

        **Authors**: *Manel Baradad, Chun-Fu Richard Chen, Jonas Wulff, Tongzhou Wang, Rogério Feris, Antonio Torralba, Phillip Isola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2a25d9d873e9ae6d242c62e36f89ee3a-Abstract-Conference.html)

        **Abstract**:

        Learning image representations using synthetic data allows training neural networks without some of the concerns associated with real images, such as privacy and bias. Existing work focuses on a handful of curated generative processes which require expert knowledge to design, making it hard to scale up. To overcome this, we propose training with a large dataset of twenty-one thousand programs, each one generating a diverse set of synthetic images. These programs are short code snippets, which are easy to modify and fast to execute using OpenGL. The proposed dataset can be used for both supervised and unsupervised representation learning, and reduces the gap between pre-training with real and procedurally generated images by 38%.

        ----

        ## [467] RényiCL: Contrastive Representation Learning with Skew Rényi Divergence

        **Authors**: *Kyungmin Lee, Jinwoo Shin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2a7157c84dcf263f77b37d6c11d7d149-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2a7157c84dcf263f77b37d6c11d7d149-Abstract-Conference.html)

        **Abstract**:

        Contrastive representation learning seeks to acquire useful representations by estimating the shared information between multiple views of data. Here, the choice of data augmentation is sensitive to the quality of learned representations: as harder the data augmentations are applied, the views share more task-relevant information, but also task-irrelevant one that can hinder the generalization capability of representation. Motivated by this, we present a new robust contrastive learning scheme, coined RényiCL, which can effectively manage harder augmentations by utilizing Rényi divergence. Our method is built upon the variational lower bound of a Rényi divergence, but a naive usage of a variational method exhibits unstable training due to the large variance. To tackle this challenge, we propose a novel contrastive objective that conducts variational estimation of a skew Renyi divergence and provides a theoretical guarantee on how variational estimation of skew divergence leads to stable training.  We show that Rényi contrastive learning objectives perform innate hard negative sampling and easy positive sampling simultaneously so that it can selectively learn useful features and ignore nuisance features. Through experiments on ImageNet, we show that Rényi contrastive learning with stronger augmentations outperforms other self-supervised methods without extra regularization or computational overhead. Also, we validate our method on various domains such as graph and tabular datasets, showing empirical gain over original contrastive methods.

        ----

        ## [468] On the Efficient Implementation of High Accuracy Optimality of Profile Maximum Likelihood

        **Authors**: *Moses Charikar, Zhihao Jiang, Kirankumar Shiragur, Aaron Sidford*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2a8ce71baac4c89bf9ff479d8240c7d9-Abstract-Conference.html)

        **Abstract**:

        We provide an efficient unified plug-in approach for estimating symmetric properties of distributions given $n$ independent samples. Our estimator is based on profile-maximum-likelihood (PML) and is sample optimal for estimating various symmetric properties when the estimation error $\epsilon \gg n^{-1/3}$. This result improves upon the previous best accuracy threshold of $\epsilon \gg n^{-1/4}$ achievable by polynomial time computable PML-based universal estimators \cite{ACSS20, ACSS20b}. Our estimator reaches a theoretical limit for universal symmetric property estimation as \cite{Han20} shows that a broad class of universal estimators (containing many well known approaches including ours) cannot be sample optimal for every $1$-Lipschitz property when $\epsilon \ll n^{-1/3}$.

        ----

        ## [469] Paraphrasing Is All You Need for Novel Object Captioning

        **Authors**: *Cheng-Fu Yang, Yao-Hung Hubert Tsai, Wan-Cyuan Fan, Russ Salakhutdinov, Louis-Philippe Morency, Frank Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2a8e6c09a1fd747e43a74710c79efdd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2a8e6c09a1fd747e43a74710c79efdd5-Abstract-Conference.html)

        **Abstract**:

        Novel object captioning (NOC) aims to describe images containing objects without observing their ground truth captions during training. Due to the absence of caption annotation, captioning models cannot be directly optimized via sequence-to-sequence training or CIDEr optimization. As a result, we present Paraphrasing-to-Captioning (P2C), a two-stage learning framework for NOC, which would heuristically optimize the output captions via paraphrasing. With P2C, the captioning model first learns paraphrasing from a language model pre-trained on text-only corpus, allowing expansion of the word bank for improving linguistic fluency. To further enforce the output caption sufficiently describing the visual content of the input image, we perform self-paraphrasing for the captioning model with fidelity and adequacy objectives introduced. Since no ground truth captions are available for novel object images during training, our P2C leverages cross-modality (image-text) association modules to ensure the above caption characteristics can be properly preserved. In the experiments, we not only show that our P2C achieves state-of-the-art performances on nocaps and COCO Caption datasets, we also verify the effectiveness and flexibility of our learning framework by replacing language and cross-modality association models for NOC. Implementation details and code are available in the supplementary materials.

        ----

        ## [470] Adversarial Robustness is at Odds with Lazy Training

        **Authors**: *Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2aab664e0d1656e8b56c74f868e1ea69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2aab664e0d1656e8b56c74f868e1ea69-Abstract-Conference.html)

        **Abstract**:

        Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to ``lazy training'' of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.

        ----

        ## [471] Uncertainty Estimation for Multi-view Data: The Power of Seeing the Whole Picture

        **Authors**: *Myong Chol Jung, He Zhao, Joanna Dipnall, Belinda Gabbe, Lan Du*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2ab3163ee384cd46baa7f1abb2b1bf19-Abstract-Conference.html)

        **Abstract**:

        Uncertainty estimation is essential to make neural networks trustworthy in real-world applications. Extensive research efforts have been made to quantify and reduce predictive uncertainty. However, most existing works are designed for unimodal data, whereas multi-view uncertainty estimation has not been sufficiently investigated. Therefore, we propose a new multi-view classification framework for better uncertainty estimation and out-of-domain sample detection, where we associate each view with an uncertainty-aware classifier and combine the predictions of all the views in a principled way. The experimental results with real-world datasets demonstrate that our proposed approach is an accurate, reliable, and well-calibrated classifier, which predominantly outperforms the multi-view baselines tested in terms of expected calibration error, robustness to noise, and accuracy for the in-domain sample classification and the out-of-domain sample detection tasks

        ----

        ## [472] Motion Transformer with Global Intention Localization and Local Movement Refinement

        **Authors**: *Shaoshuai Shi, Li Jiang, Dengxin Dai, Bernt Schiele*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2ab47c960bfee4f86dfc362f26ad066a-Abstract-Conference.html)

        **Abstract**:

        Predicting multimodal future behavior of traffic participants is essential for robotic vehicles to make safe decisions. Existing works explore to directly predict future trajectories based on latent features or utilize dense goal candidates to identify agent's destinations, where the former strategy converges slowly since all motion modes are derived from the same feature while the latter strategy has efficiency issue since its performance highly relies on the density of goal candidates. In this paper, we propose the Motion TRansformer (MTR) framework that models motion prediction as the joint optimization of global intention localization and local movement refinement. Instead of using goal candidates, MTR incorporates spatial intention priors by adopting a small set of learnable motion query pairs. Each motion query pair takes charge of trajectory prediction and refinement for a specific motion mode, which stabilizes the training process and facilitates better multimodal predictions. Experiments show that MTR achieves state-of-the-art performance on both the marginal and joint motion prediction challenges, ranking 1st on the leaderbaords of Waymo Open Motion Dataset. Code will be available at https://github.com/sshaoshuai/MTR.

        ----

        ## [473] No-regret learning in games with noisy feedback: Faster rates and adaptivity via learning rate separation

        **Authors**: *Yu-Guan Hsieh, Kimon Antonakopoulos, Volkan Cevher, Panayotis Mertikopoulos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2abad9fd438b40604ddaabe75e6c51dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2abad9fd438b40604ddaabe75e6c51dd-Abstract-Conference.html)

        **Abstract**:

        We examine the problem of regret minimization when the learner is involved in a continuous game with other optimizing agents: in this case, if all players follow a no-regret algorithm, it is possible to achieve significantly lower regret relative to fully adversarial environments. We study this problem in the context of variationally stable games (a class of continuous games which includes all convex-concave and monotone games), and when the players only have access to noisy estimates of their individual payoff gradients. If the noise is additive, the game-theoretic and purely adversarial settings enjoy similar regret guarantees; however, if the noise is \emph{multiplicative}, we show that the learners can, in fact, achieve \emph{constant} regret. We achieve this faster rate via an optimistic gradient scheme with \emph{learning rate separation} \textendash\ that is, the method's extrapolation and update steps are tuned to different schedules, depending on the noise profile. Subsequently, to eliminate the need for delicate hyperparameter tuning, we propose a fully adaptive method that smoothly interpolates between worst- and best-case regret guarantees.

        ----

        ## [474] First-Order Algorithms for Min-Max Optimization in Geodesic Metric Spaces

        **Authors**: *Michael I. Jordan, Tianyi Lin, Emmanouil V. Vlatakis-Gkaragkounis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2ad9a1a6ffac3dd72cc1df96019eca01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2ad9a1a6ffac3dd72cc1df96019eca01-Abstract-Conference.html)

        **Abstract**:

        From optimal transport to robust dimensionality reduction, many machine learning applicationscan be cast into the min-max optimization problems over Riemannian manifolds. Though manymin-max algorithms have been analyzed in the Euclidean setting, it has been elusive how theseresults translate to the Riemannian case. Zhang et al. (2022) have recently identified that geodesic convexconcave Riemannian problems admit always Sionâ€™s saddle point solutions. Immediately, an importantquestion that arises is if a performance gap between the Riemannian and the optimal Euclidean spaceconvex concave algorithms is necessary. Our work is the first to answer the question in the negative:We prove that the Riemannian corrected extragradient (RCEG) method achieves last-iterate at alinear convergence rate at the geodesically strongly convex concave case, matching the euclidean one.Our results also extend to the stochastic or non-smooth case where RCEG & Riemanian gradientascent descent (RGDA) achieve respectively near-optimal convergence rates up to factors dependingon curvature of the manifold. Finally, we empirically demonstrate the effectiveness of RCEG insolving robust PCA.

        ----

        ## [475] Feature-Proxy Transformer for Few-Shot Segmentation

        **Authors**: *Jian-Wei Zhang, Yifan Sun, Yi Yang, Wei Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2ae33575c3374050654ae7802326c81d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2ae33575c3374050654ae7802326c81d-Abstract-Conference.html)

        **Abstract**:

        Few-shot segmentation~(FSS) aims at performing semantic segmentation on novel classes given a few annotated support samples. With a rethink of recent advances, we find that the current FSS framework has deviated far from the supervised segmentation framework: Given the deep features, FSS methods typically use an intricate decoder to perform sophisticated pixel-wise matching, while the supervised segmentation methods use a simple linear classification head. Due to the intricacy of the decoder and its matching pipeline, it is not easy to follow such an FSS framework. This paper revives the straightforward framework of ``feature extractor $+$ linear classification head'' and proposes a novel Feature-Proxy Transformer (FPTrans) method, in which the ``proxy'' is the vector representing a semantic class in the linear classification head. FPTrans has two keypoints for learning discriminative features and representative proxies: 1) To better utilize the limited support samples, the feature extractor makes the query interact with the support features from bottom to top layers using a novel prompting strategy. 2) FPTrans uses multiple local background proxies (instead of a single one) because the background is not homogeneous and may contain some novel foreground regions. These two keypoints are easily integrated into the vision transformer backbone with the prompting mechanism in the transformer. Given the learned features and proxies, FPTrans directly compares their cosine similarity for segmentation. Although the framework is straightforward, we show that FPTrans achieves competitive FSS accuracy on par with state-of-the-art decoder-based methods.

        ----

        ## [476] Conformal Frequency Estimation with Sketched Data

        **Authors**: *Matteo Sesia, Stefano Favaro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2b2011a7d5396faf5899863d896a3c24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2b2011a7d5396faf5899863d896a3c24-Abstract-Conference.html)

        **Abstract**:

        A flexible conformal inference method is developed to construct confidence intervals for the frequencies of queried objects in very large data sets, based on a much smaller sketch of those data. The approach is data-adaptive and requires no knowledge of the data distribution or of the details of the sketching algorithm; instead, it constructs provably valid frequentist confidence intervals under the sole assumption of data exchangeability. Although our solution is broadly applicable, this paper focuses on applications involving the count-min sketch algorithm and a non-linear variation thereof. The performance is compared to that of frequentist and Bayesian alternatives through simulations and experiments with data sets of SARS-CoV-2 DNA sequences and classic English literature.

        ----

        ## [477] Revisiting Active Sets for Gaussian Process Decoders

        **Authors**: *Pablo Moreno-Muñoz, Cilie W. Feldager, Søren Hauberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2b2bf329be5da02422a1d15ce4a81fdb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2b2bf329be5da02422a1d15ce4a81fdb-Abstract-Conference.html)

        **Abstract**:

        Decoders built on Gaussian processes (GPs) are enticing due to the marginalisation over the non-linear function space. Such models (also known as GP-LVMs) are often expensive and notoriously difficult to train in practice, but can be scaled using variational inference and inducing points. In this paper, we revisit active set approximations. We develop a new stochastic estimate of the log-marginal likelihood based on recently discovered links to cross-validation, and we propose a computationally efficient approximation thereof. We demonstrate that the resulting stochastic active sets (SAS) approximation significantly improves the robustness of GP decoder training, while reducing computational cost. The SAS-GP obtains more structure in the latent space, scales to many datapoints, and learns better representations than variational autoencoders, which is rarely the case for GP decoders.

        ----

        ## [478] Exact learning dynamics of deep linear networks with prior knowledge

        **Authors**: *Lukas Braun, Clémentine Dominé, James Fitzgerald, Andrew Saxe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2b3bb2c95195130977a51b3bb251c40a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2b3bb2c95195130977a51b3bb251c40a-Abstract-Conference.html)

        **Abstract**:

        Learning in deep neural networks is known to depend critically on the knowledge embedded in the initial network weights. However, few theoretical results have precisely linked prior knowledge to learning dynamics. Here we derive exact solutions to the dynamics of learning with rich prior knowledge in deep linear networks by generalising Fukumizu's matrix Riccati solution \citep{fukumizu1998effect}. We obtain explicit expressions for the evolving network function, hidden representational similarity, and neural tangent kernel over training for a broad class of initialisations and tasks. The expressions reveal a class of task-independent initialisations that radically alter learning dynamics from slow non-linear dynamics to fast exponential trajectories while converging to a global optimum with identical representational similarity, dissociating learning trajectories from the structure of initial internal representations. We characterise how network weights dynamically align with task structure, rigorously justifying why previous solutions successfully described learning from small initial weights without incorporating their fine-scale structure. Finally, we discuss the implications of these findings for continual learning, reversal learning and learning of structured knowledge. Taken together, our results provide a mathematical toolkit for understanding the impact of prior knowledge on deep learning.

        ----

        ## [479] AMP: Automatically Finding Model Parallel Strategies with Heterogeneity Awareness

        **Authors**: *Dacheng Li, Hongyi Wang, Eric P. Xing, Hao Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2b4bfa1cebe78d125fefd7ea6ffcfc6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2b4bfa1cebe78d125fefd7ea6ffcfc6d-Abstract-Conference.html)

        **Abstract**:

        Scaling up model sizes can lead to fundamentally new capabilities in many machine learning (ML) tasks. However, training big models requires strong distributed system expertise to carefully design model-parallel execution strategies that suit the model architectures and cluster setups. In this paper, we develop AMP, a framework that automatically derives such strategies. AMP identifies a valid space of model parallelism strategies and efficiently searches the space for high-performed strategies, by leveraging a cost model designed to capture the heterogeneity of the model and cluster specifications. Unlike existing methods, AMP is specifically tailored to support complex models composed of uneven layers and cluster setups with more heterogeneous accelerators and bandwidth. We evaluate AMP on popular modelsand cluster setups from public clouds and show that AMP returns parallel strategies that match the expert-tuned strategies on typical cluster setups. On heterogeneous clusters or models with heterogeneous architectures, AMP finds strategies with 1.54$\times$ and 1.77$\times$ higher throughput than state-of-the-art model-parallel systems, respectively.

        ----

        ## [480] A consistently adaptive trust-region method

        **Authors**: *Fadi Hamad, Oliver Hinder*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2c19666cbb2c14d45d39e2dcf6ab0b99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2c19666cbb2c14d45d39e2dcf6ab0b99-Abstract-Conference.html)

        **Abstract**:

        Adaptive trust-region methods attempt to maintain strong convergence guarantees without depending on conservative estimates of problem properties such as Lipschitz constants. However, on close inspection, one can show existing adaptive trust-region methods have theoretical guarantees with severely suboptimal dependence on problem properties such as the Lipschitz constant of the Hessian. For example, TRACE developed by Curtis et al. obtains a $O(\Delta_f L^{3/2} \epsilon^{-3/2}) + \tilde{O}(1)$ iteration bound where $L$ is the Lipschitz constant of the Hessian. Compared with the optimal $O(\Delta_f L^{1/2} \epsilon^{-3/2})$ bound this is suboptimal with respect to $L$. We present the first adaptive trust-region method which circumvents this issue and requires at most $O( \Delta_f L^{1/2}  \epsilon^{-3/2}) + \tilde{O}(1)$ iterations to find an $\epsilon$-approximate stationary point, matching the optimal iteration bound up to an additive logarithmic term. Our method is a simple variant of a classic trust-region method and in our experiments performs competitively with both ARC and a classical trust-region method.

        ----

        ## [481] Online Neural Sequence Detection with Hierarchical Dirichlet Point Process

        **Authors**: *Weihan Li, Yu Qi, Gang Pan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2c3b636b64ca1dfdae3e096e4deeaa42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2c3b636b64ca1dfdae3e096e4deeaa42-Abstract-Conference.html)

        **Abstract**:

        Neural sequence detection plays a vital role in neuroscience research. Recent impressive works utilize convolutive nonnegative matrix factorization and Neyman-Scott process to solve this problem. However, they still face two limitations. Firstly, they accommodate the entire dataset into memory and perform iterative updates of multiple passes, which can be inefficient when the dataset is large or grows frequently. Secondly, they rely on the prior knowledge of the number of sequence types, which can be impractical with data when the future situation is unknown. To tackle these limitations, we propose a hierarchical Dirichlet point process model for efficient neural sequence detection. Instead of computing the entire data, our model can sequentially detect sequences in an online unsupervised manner with Particle filters. Besides, the Dirichlet prior enables our model to automatically introduce new sequence types on the fly as needed, thus avoiding specifying the number of types in advance. We manifest these advantages on synthetic data and neural recordings from songbird higher vocal center and rodent hippocampus.

        ----

        ## [482] Leveraging Inter-Layer Dependency for Post -Training Quantization

        **Authors**: *Changbao Wang, Dandan Zheng, Yuanliu Liu, Liang Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2c570b0f9938c7a58a612e5b00af9cc0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2c570b0f9938c7a58a612e5b00af9cc0-Abstract-Conference.html)

        **Abstract**:

        Prior works on Post-training Quantization (PTQ) typically separate a neural network into sub-nets and quantize them sequentially. This process pays little attention to the dependency across the sub-nets, hence is less optimal. In this paper, we propose a novel Network-Wise Quantization (NWQ) approach to fully leveraging inter-layer dependency. NWQ faces a larger scale combinatorial optimization problem of discrete variables than in previous  works, which raises two major challenges: over-fitting and discrete optimization problem. NWQ alleviates over-fitting via a Activation Regularization (AR) technique, which better controls the activation distribution. To optimize discrete variables, NWQ introduces Annealing Softmax (ASoftmax) and Annealing Mixup (AMixup) to progressively transition quantized weights and activations from continuity to discretization, respectively. Extensive experiments demonstrate that NWQ outperforms previous state-of-the-art by a large margin: 20.24\% for the challenging configuration of MobileNetV2 with 2 bits on ImageNet, pushing extremely low-bit PTQ from feasibility to usability. In addition, NWQ is able to achieve competitive results with only 10\% computation cost of previous works.

        ----

        ## [483] Emergence of Hierarchical Layers in a Single Sheet of Self-Organizing Spiking Neurons

        **Authors**: *Paul Bertens, Seong-Whan Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2c625366ae28066fcb1827b44517d674-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2c625366ae28066fcb1827b44517d674-Abstract-Conference.html)

        **Abstract**:

        Traditionally convolutional neural network architectures have been designed by stacking layers on top of each other to form deeper hierarchical networks. The cortex in the brain however does not just stack layers as done in standard convolution neural networks, instead different regions are organized next to each other in a large single sheet of neurons. Biological neurons self organize to form topographic maps, where neurons encoding similar stimuli group together to form logical clusters. Here we propose new self-organization principles that allow for the formation of hierarchical cortical regions (i.e. layers) in a completely unsupervised manner without requiring any predefined architecture. Synaptic connections are dynamically grown and pruned, which allows us to actively constrain the number of incoming and outgoing connections. This way we can minimize the wiring cost by taking into account both the synaptic strength and the connection length. The proposed method uses purely local learning rules in the form of spike-timing-dependent plasticity (STDP) with lateral excitation and inhibition. We show experimentally that these self-organization rules are sufficient for topographic maps and hierarchical layers to emerge. Our proposed Self-Organizing Neural Sheet (SONS) model can thus form traditional neural network layers in a completely unsupervised manner from just a single large pool of unstructured spiking neurons.

        ----

        ## [484] A gradient sampling method with complexity guarantees for Lipschitz functions in high and low dimensions

        **Authors**: *Damek Davis, Dmitriy Drusvyatskiy, Yin Tat Lee, Swati Padmanabhan, Guanghao Ye*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2c8d9636f74d0207ff4f65956010f450-Abstract-Conference.html)

        **Abstract**:

        Zhang et al. (ICML 2020) introduced a novel modification of Goldstein's classical subgradient method, with an efficiency guarantee of $O(\varepsilon^{-4})$ for minimizing Lipschitz functions. Their work, however, makes use of an oracle that is not efficiently implementable. In this paper, we obtain the same efficiency guarantee with a standard subgradient oracle, thus making our algorithm efficiently implementable. Our resulting method works on any Lipschitz function whose value and gradient can be evaluated at points of differentiability. We additionally present a new cutting plane algorithm that achieves an efficiency of  $O(d\varepsilon^{-2}\log S)$ for the class of $S$-smooth (and possibly non-convex) functions in low dimensions. Strikingly, this $\epsilon$-dependence matches the lower bounds for the convex setting.

        ----

        ## [485] CyCLIP: Cyclic Contrastive Language-Image Pretraining

        **Authors**: *Shashank Goel, Hritik Bansal, Sumit Bhatia, Ryan A. Rossi, Vishwa Vinay, Aditya Grover*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2cd36d327f33d47b372d4711edd08de0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2cd36d327f33d47b372d4711edd08de0-Abstract-Conference.html)

        **Abstract**:

        Recent advances in contrastive representation learning over paired image-text data have led to models such as CLIP that achieve state-of-the-art performance for zero-shot classification and distributional robustness. Such models typically require joint reasoning in the image and text representation spaces for downstream inference tasks. Contrary to prior beliefs, we demonstrate that the image and text representations learned via a standard contrastive objective are not interchangeable and can lead to inconsistent downstream predictions. To mitigate this issue, we formalize consistency and propose CyCLIP, a framework for contrastive representation learning that explicitly optimizes for the learned representations to be geometrically consistent in the image and text space. In particular, we show that consistent representations can be learned by explicitly symmetrizing (a) the similarity between the two mismatched image-text pairs (cross-modal consistency); and (b) the similarity between the image-image pair and the text-text pair (in-modal consistency). Empirically, we show that the improved consistency in CyCLIP translates to significant gains over CLIP, with gains ranging from 10%-24% for zero-shot classification on standard benchmarks (CIFAR-10, CIFAR-100, ImageNet1K) and 10%-27% for robustness to various natural distribution shifts.

        ----

        ## [486] When does dough become a bagel? Analyzing the remaining mistakes on ImageNet

        **Authors**: *Vijay Vasudevan, Benjamin Caine, Raphael Gontijo Lopes, Sara Fridovich-Keil, Rebecca Roelofs*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2cd5737c59645f7ef23b2842b705edf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2cd5737c59645f7ef23b2842b705edf2-Abstract-Conference.html)

        **Abstract**:

        Image classification accuracy on the ImageNet dataset has been a barometer for progress in computer vision over the last decade. Several recent papers have questioned the degree to which the benchmark remains useful to the community, yet innovations continue to contribute gains to performance, with today's largest models achieving 90%+ top-1 accuracy. To help contextualize progress on ImageNet and provide a more meaningful evaluation for today's state-of-the-art models, we manually review and categorize every remaining mistake that a few top models make in order to provide insight into the long-tail of errors on one of the most benchmarked datasets in computer vision. We focus on the multi-label subset evaluation of ImageNet, where today's best models achieve upwards of 97% top-1 accuracy. Our analysis reveals that nearly half of the supposed mistakes are not mistakes at all, and we uncover new valid multi-labels, demonstrating that, without careful review, we are significantly underestimating the performance of these models. On the other hand, we also find that today's best models still make a significant number of mistakes (40%) that are obviously wrong to human reviewers. To calibrate future progress on ImageNet, we provide an updated multi-label evaluation set, and we curate ImageNet-Major: a 68-example "major error" slice of the obvious mistakes made by today's top models -- a slice where models should achieve near perfection, but today are far from doing so.

        ----

        ## [487] Spatial Pruned Sparse Convolution for Efficient 3D Object Detection

        **Authors**: *Jianhui Liu, Yukang Chen, Xiaoqing Ye, Zhuotao Tian, Xiao Tan, Xiaojuan Qi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2ce10f144bb93449767f355c01f24cc1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2ce10f144bb93449767f355c01f24cc1-Abstract-Conference.html)

        **Abstract**:

        3D scenes are dominated by a large number of background points, which is redundant for the detection task that mainly needs to focus on foreground objects. In this paper, we analyze major components of existing sparse 3D CNNs and find that 3D CNNs ignores the redundancy of data and further amplifies it in the down-sampling process, which brings a huge amount of extra and unnecessary computational overhead. Inspired by this, we propose a new convolution operator named spatial pruned sparse convolution (SPS-Conv), which includes two variants, spatial pruned submanifold sparse convolution (SPSS-Conv) and spatial pruned regular sparse convolution (SPRS-Conv), both of which are based on the idea of dynamically determine crucial areas for performing computations to reduce redundancy. We empirically find that magnitude of features can serve as an important cues to determine crucial areas which get rid of the heavy computations of learning-based methods. The proposed modules can easily be incorporated into existing sparse  3D CNNs without extra architectural modifications. Extensive experiments on the KITTI and nuScenes datasets demonstrate that our method can achieve more than 50% reduction in GFLOPs without compromising the performance.

        ----

        ## [488] Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization

        **Authors**: *Aniket Das, Bernhard Schölkopf, Michael Muehlebach*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2ce4f0b8e24c45318352068603153590-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2ce4f0b8e24c45318352068603153590-Abstract-Conference.html)

        **Abstract**:

        We analyze the convergence rates of stochastic gradient algorithms for smooth finite-sum minimax optimization and show that, for many such algorithms, sampling the data points \emph{without replacement} leads to faster convergence compared to sampling with replacement. For the smooth and strongly convex-strongly concave setting, we consider gradient descent ascent and the proximal point method, and present a unified analysis of two popular without-replacement sampling strategies, namely \emph{Random Reshuffling} (RR), which shuffles the data every epoch, and \emph{Single Shuffling} or \emph{Shuffle Once} (SO), which shuffles only at the beginning. We obtain tight convergence rates for RR and SO and demonstrate that these strategies lead to faster convergence than uniform sampling. Moving beyond convexity, we obtain similar results for smooth nonconvex-nonconcave objectives satisfying a two-sided Polyak-\L{}ojasiewicz inequality. Finally, we demonstrate that our techniques are general enough to analyze the effect of \emph{data-ordering attacks}, where an adversary manipulates the order in which data points are supplied to the optimizer. Our analysis also recovers tight rates for the \emph{incremental gradient} method, where the data points are not shuffled at all.

        ----

        ## [489] Feature Learning in $L_2$-regularized DNNs: Attraction/Repulsion and Sparsity

        **Authors**: *Arthur Jacot, Eugene Golikov, Clément Hongler, Franck Gabriel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d2f85c0f93e69cf71f58eebaebb5e8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d2f85c0f93e69cf71f58eebaebb5e8d-Abstract-Conference.html)

        **Abstract**:

        We study the loss surface of DNNs with $L_{2}$ regularization. Weshow that the loss in terms of the parameters can be reformulatedinto a loss in terms of the layerwise activations $Z_{\ell}$ of thetraining set. This reformulation reveals the dynamics behind featurelearning: each hidden representations $Z_{\ell}$ are optimal w.r.t.to an attraction/repulsion problem and interpolate between the inputand output representations, keeping as little information from theinput as necessary to construct the activation of the next layer.For positively homogeneous non-linearities, the loss can be furtherreformulated in terms of the covariances of the hidden representations,which takes the form of a partially convex optimization over a convexcone.This second reformulation allows us to prove a sparsity result forhomogeneous DNNs: any local minimum of the $L_{2}$-regularized losscan be achieved with at most $N(N+1)$ neurons in each hidden layer(where $N$ is the size of the training set). We show that this boundis tight by giving an example of a local minimum that requires $N^{2}/4$hidden neurons. But we also observe numerically that in more traditionalsettings much less than $N^{2}$ neurons are required to reach theminima.

        ----

        ## [490] Set-based Meta-Interpolation for Few-Task Meta-Learning

        **Authors**: *Seanie Lee, Bruno Andreis, Kenji Kawaguchi, Juho Lee, Sung Ju Hwang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d43f7a61b57f83619f82c971e4bddc0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d43f7a61b57f83619f82c971e4bddc0-Abstract-Conference.html)

        **Abstract**:

        Meta-learning approaches enable machine learning systems to adapt to new tasks given few examples by leveraging knowledge from related tasks.  However, a large number of meta-training tasks are still required for generalization to unseen tasks during meta-testing, which introduces a critical bottleneck for real-world problems that come with only few tasks, due to various reasons including the difficulty and cost of constructing tasks. Recently, several task augmentation methods have been proposed to tackle this issue using domain-specific knowledge to design augmentation techniques to densify the meta-training task distribution. However, such reliance on domain-specific knowledge renders these methods inapplicable to other domains. While Manifold Mixup based task augmentation methods are domain-agnostic, we empirically find them ineffective on non-image domains. To tackle these limitations, we propose a novel domain-agnostic task augmentation method, Meta-Interpolation, which utilizes expressive neural set functions to densify the meta-training task distribution using bilevel optimization. We empirically validate the efficacy of Meta-Interpolation on eight datasets spanning across various domains such as image classification, molecule property prediction, text classification and speech recognition. Experimentally, we show that Meta-Interpolation consistently outperforms all the relevant baselines. Theoretically, we prove that task interpolation with the set function regularizes the meta-learner to improve generalization. We provide our source code in the supplementary material.

        ----

        ## [491] Non-deep Networks

        **Authors**: *Ankit Goyal, Alexey Bochkovskiy, Jia Deng, Vladlen Koltun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d52879ef2ba487445ca2e143b104c3b-Abstract-Conference.html)

        **Abstract**:

        Latency is of utmost importance in safety-critical systems. In neural networks, lowest theoretical latency is dependent on the depth of the network. This begs the question -- is it possible to build high-performing ``non-deep" neural networks? We show that it is. To do so, we use parallel subnetworks instead of stacking one layer after another. This helps effectively reduce depth while maintaining high performance. By utilizing parallel substructures, we show, for the first time, that a network with a depth of just 12 can achieve top-1 accuracy over 80% on ImageNet, 96% on CIFAR10, and 81% on CIFAR100. We also show that a network with a low-depth (12) backbone can achieve an AP of 48% on MS-COCO. We analyze the scaling rules for our design and show how to increase performance without changing the network's depth. Finally, we provide a proof of concept for how non-deep networks could be used to build low-latency recognition systems. Code is available at https://github.com/imankgoyal/NonDeepNetworks.

        ----

        ## [492] Low-rank Optimal Transport: Approximation, Statistics and Debiasing

        **Authors**: *Meyer Scetbon, Marco Cuturi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d69e771d9f274f7c624198ea74f5b98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d69e771d9f274f7c624198ea74f5b98-Abstract-Conference.html)

        **Abstract**:

        The matching principles behind optimal transport (OT) play an increasingly important role in machine learning, a trend which can be observed when OT is used to disambiguate datasets in applications (e.g. single-cell genomics) or used to improve more complex methods (e.g. balanced attention in transformers or self-supervised learning). To scale to more challenging problems, there is a growing consensus that OT requires solvers that can operate on millions, not thousands, of points. The low-rank optimal transport (LOT) approach advocated in \cite{scetbon2021lowrank} holds several promises in that regard, and was shown to complement more established entropic regularization approaches, being able to insert itself in more complex pipelines, such as quadratic OT. LOT restricts the search for low-cost couplings to those that have a low-nonnegative rank, yielding linear time algorithms in cases of interest. However, these promises can only be fulfilled if the LOT approach is seen as a legitimate contender to entropic regularization when compared on properties of interest, where the scorecard typically includes theoretical properties (statistical complexity and relation to other methods) or practical aspects (debiasing, hyperparameter tuning, initialization). We target each of these areas in this paper in order to cement the impact of low-rank approaches in computational OT.

        ----

        ## [493] Embodied Scene-aware Human Pose Estimation

        **Authors**: *Zhengyi Luo, Shun Iwase, Ye Yuan, Kris Kitani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d6e0f68dd49e5518fc4aeef58d759e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d6e0f68dd49e5518fc4aeef58d759e2-Abstract-Conference.html)

        **Abstract**:

        We propose embodied scene-aware human pose estimation where we estimate 3D poses based on a simulated agent's proprioception and scene awareness, along with external third-person observations. Unlike prior methods that often resort to multistage optimization, non-causal inference, and complex contact modeling to estimate human pose and human scene interactions, our method is one-stage, causal, and recovers global 3D human poses in a simulated environment. Since 2D third-person observations are coupled with the camera pose, we propose to disentangle the camera pose and use a multi-step projection gradient defined in the global coordinate frame as the movement cue for our embodied agent. Leveraging a physics simulation and prescanned scenes (e.g., 3D mesh), we simulate our agent in everyday environments (library, office, bedroom, etc.) and equip our agent with environmental sensors to intelligently navigate and interact with the geometries of the scene. Our method also relies only on 2D keypoints and can be trained on synthetic datasets derived from popular human motion databases. To evaluate, we use the popular H36M and PROX datasets and achieve high quality pose estimation on the challenging PROX dataset without ever using PROX motion sequences for training. Code and videos are available on the project page.

        ----

        ## [494] Privacy Induces Robustness: Information-Computation Gaps and Sparse Mean Estimation

        **Authors**: *Kristian Georgiev, Samuel B. Hopkins*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d76b6a9f96181ab717c1a15ab9302e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d76b6a9f96181ab717c1a15ab9302e1-Abstract-Conference.html)

        **Abstract**:

        We establish a simple connection between robust and differentially-private algorithms: private mechanisms which perform well with very high probability are automatically robust in the sense that they retain accuracy even if a constant fraction of the samples they receive are adversarially corrupted. Since optimal mechanisms typically achieve these high success probabilities, our results imply that optimal private mechanisms for many basic statistics problems are robust. We investigate the consequences of this observation for both algorithms and computational complexity across different statistical problems. Assuming the Brennan-Bresler secret-leakage planted clique conjecture, we demonstrate a fundamental tradeoff between computational efficiency, privacy leakage, and success probability for sparse mean estimation. Private algorithms which match this tradeoff are not yet known -- we achieve that (up to polylogarithmic factors) in a polynomially-large range of parameters via theSum-of-Squares method.To establish an information-computation gap for sparse mean estimation, we also design new (exponential-time) mechanisms using fewer samples than efficient algorithms must use. Finally, we give evidence for privacy-induced information-computation gaps for several other statistics and learning problems, including PAC learning parity functions and estimation of the mean of a multivariate Gaussian.

        ----

        ## [495] Batch Bayesian Optimization on Permutations using the Acquisition Weighted Kernel

        **Authors**: *ChangYong Oh, Roberto Bondesan, Efstratios Gavves, Max Welling*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d779258dd899505b56f237de66ae470-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d779258dd899505b56f237de66ae470-Abstract-Conference.html)

        **Abstract**:

        In this work we propose a batch Bayesian optimization method for combinatorial problems on permutations, which is well suited for expensive-to-evaluate objectives. We first introduce LAW, an efficient batch acquisition method based on determinantal point processes using the acquisition weighted kernel. Relying on multiple parallel evaluations, LAW enables accelerated search on combinatorial spaces. We then apply the framework to permutation problems, which have so far received little attention in the Bayesian Optimization literature, despite their practical importance. We call this method LAW2ORDER. On the theoretical front, we prove that LAW2ORDER has vanishing simple regret by showing that the batch cumulative regret is sublinear. Empirically, we assess the method on several standard combinatorial problems involving permutations such as quadratic assignment, flowshop scheduling and the traveling salesman, as well as on a structure learning task.

        ----

        ## [496] Supervised Training of Conditional Monge Maps

        **Authors**: *Charlotte Bunne, Andreas Krause, Marco Cuturi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d880acd7b31e25d45097455c8e8257f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d880acd7b31e25d45097455c8e8257f-Abstract-Conference.html)

        **Abstract**:

        Optimal transport (OT) theory describes general principles to define and select, among many possible choices, the most efficient way to map a probability measure onto another. That theory has been mostly used to estimate, given a pair of source and target probability measures $(\mu,\nu)$, a parameterized map $T_\theta$ that can efficiently map $\mu$ onto $\nu$. In many applications, such as predicting cell responses to treatments, pairs of input/output data measures $(\mu,\nu)$ that define optimal transport problems do not arise in isolation but are associated with a context $c$, as for instance a treatment when comparing populations of untreated and treated cells. To account for that context in OT estimation, we introduce CondOT, a multi-task approach to estimate a family of OT maps conditioned on a context variable, using several pairs of measures $(\mu_i, \nu_i)$ tagged with a context label $c_i$. CondOT learns a global map $\mathcal{T}_{\theta}$ conditioned on context that is not only expected to fit all labeled pairs in the dataset $\{(c_i, (\mu_i, \nu_i))\}$, i.e., $\mathcal{T}_{\theta}(c_i) \sharp\mu_i \approx \nu_i$, but should also generalize to produce meaningful maps $\mathcal{T}_{\theta}(c_{\text{new}})$ when conditioned on unseen contexts $c_{\text{new}}$. Our approach harnesses and provides a novel usage for partially input convex neural networks, for which we introduce a robust and efficient initialization strategy inspired by Gaussian approximations. We demonstrate the ability of CondOT to infer the effect of an arbitrary combination of genetic or therapeutic perturbations on single cells, using only observations of the effects of said perturbations separately.

        ----

        ## [497] Formalizing Consistency and Coherence of Representation Learning

        **Authors**: *Harald Strömfelt, Luke Dickens, Artur S. d'Avila Garcez, Alessandra Russo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2d95270d763751439626d91f57e9a750-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2d95270d763751439626d91f57e9a750-Abstract-Conference.html)

        **Abstract**:

        In the study of reasoning in neural networks, recent efforts have sought to improve consistency and coherence of sequence models, leading to important developments in the area of neuro-symbolic AI. In symbolic AI, the concepts of consistency and coherence can be defined and verified formally, but for neural networks these definitions are lacking. The provision of such formal definitions is crucial to offer a common basis for the quantitative evaluation and systematic comparison of connectionist, neuro-symbolic and transfer learning approaches. In this paper, we introduce formal definitions of consistency and coherence for neural systems. To illustrate the usefulness of our definitions, we propose a new dynamic relation-decoder model built around the principles of consistency and coherence. We compare our results with several existing relation-decoders using a partial transfer learning task based on a novel data set introduced in this paper. Our experiments show that relation-decoders that maintain consistency over unobserved regions of representation space retaincoherence across domains, whilst achieving better transfer learning performance.

        ----

        ## [498] Positively Weighted Kernel Quadrature via Subsampling

        **Authors**: *Satoshi Hayakawa, Harald Oberhauser, Terry J. Lyons*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2dae7d1ccf1edf76f8ce7c282bdf4730-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2dae7d1ccf1edf76f8ce7c282bdf4730-Abstract-Conference.html)

        **Abstract**:

        We study kernel quadrature rules with convex weights. Our approach combines the spectral properties of the kernel with recombination results about point measures. This results in effective algorithms that construct convex quadrature rules using only access to i.i.d. samples from the underlying measure and evaluation of the kernel and that result in a small worst-case error. In addition to our theoretical results and the benefits resulting from convex weights, our experiments indicate that this construction can compete with the optimal bounds in well-known examples.

        ----

        ## [499] Guaranteed Conservation of Momentum for Learning Particle-based Fluid Dynamics

        **Authors**: *Lukas Prantl, Benjamin Ummenhofer, Vladlen Koltun, Nils Thuerey*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2dd7f33ffbb59b4ff987be5442a13016-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2dd7f33ffbb59b4ff987be5442a13016-Abstract-Conference.html)

        **Abstract**:

        We present a novel method for guaranteeing linear momentum in learned physics simulations. Unlike existing methods, we enforce conservation of momentum with a hard constraint, which we realize via antisymmetrical continuous convolutional layers. We combine these strict constraints with a hierarchical network architecture, a carefully constructed resampling scheme, and a training approach for temporal coherence. In combination, the proposed method allows us to increase the physical accuracy of the learned simulator substantially. In addition, the induced physical bias leads to significantly better generalization performance and makes our method more reliable in unseen test cases. We evaluate our method on a range of different, challenging fluid scenarios. Among others, we demonstrate that our approach generalizes to new scenarios with up to one million particles. Our results show that the proposed algorithm can learn complex dynamics while outperforming existing approaches in generalization and training performance. An implementation of our approach is available at https://github.com/tum-pbs/DMCF.

        ----

        ## [500] M4Singer: A Multi-Style, Multi-Singer and Musical Score Provided Mandarin Singing Corpus

        **Authors**: *Lichao Zhang, Ruiqi Li, Shoutong Wang, Liqun Deng, Jinglin Liu, Yi Ren, Jinzheng He, Rongjie Huang, Jieming Zhu, Xiao Chen, Zhou Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2de60892dd329683ec21877a4e7c3091-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/2de60892dd329683ec21877a4e7c3091-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The lack of publicly available high-quality and accurately labeled datasets has long been a major bottleneck for singing voice synthesis (SVS). To tackle this problem, we present M4Singer, a free-to-use Multi-style, Multi-singer Mandarin singing collection with elaborately annotated Musical scores as well as its benchmarks. Specifically, 1) we construct and release a large high-quality Chinese singing voice corpus, which is recorded by 20 professional singers, covering 700 Chinese pop songs as well as all the four SATB types (i.e.,  soprano, alto, tenor, and bass); 2) we take extensive efforts to manually compose the musical scores for each recorded song, which are necessary to the study of the prosody modeling for SVS. 3) To facilitate the use and demonstrate the quality of M4Singer, we conduct four different benchmark experiments: score-based SVS, controllable singing voice (CSV), singing voice conversion (SVC) and automatic music transcription (AMT).

        ----

        ## [501] Bayesian Spline Learning for Equation Discovery of Nonlinear Dynamics with Quantified Uncertainty

        **Authors**: *Luning Sun, Daniel Huang, Hao Sun, Jian-Xun Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2df278b7fbbea06c3892d2f4388640b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2df278b7fbbea06c3892d2f4388640b6-Abstract-Conference.html)

        **Abstract**:

        Nonlinear dynamics are ubiquitous in science and engineering applications, but the physics of most complex systems is far from being fully understood. Discovering interpretable governing equations from measurement data can help us understand and predict the behavior of complex dynamic systems. Although extensive work has recently been done in this field, robustly distilling explicit model forms from very sparse data with considerable noise remains intractable. Moreover, quantifying and propagating the uncertainty of the identified system from noisy data is challenging, and relevant literature is still limited. To bridge this gap, we develop a novel Bayesian spline learning framework to identify parsimonious governing equations of nonlinear (spatio)temporal dynamics from sparse, noisy data with quantified uncertainty. The proposed method utilizes spline basis to handle the data scarcity and measurement noise, upon which a group of derivatives can be accurately computed to form a library of candidate model terms. The equation residuals are used to inform the spline learning in a Bayesian manner, where approximate Bayesian uncertainty calibration techniques are employed to approximate posterior distributions of the trainable parameters. To promote the sparsity, an iterative sequential-threshold Bayesian learning approach is developed, using the alternative direction optimization strategy to systematically approximate L0 sparsity constraints. The proposed algorithm is evaluated on multiple nonlinear dynamical systems governed by canonical ordinary and partial differential equations, and the merit/superiority of the proposed method is demonstrated by comparison with state-of-the-art methods.

        ----

        ## [502] Convexity Certificates from Hessians

        **Authors**: *Julien Klaus, Niklas Merk, Konstantin Wiedom, Sören Laue, Joachim Giesen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e0802e2898522a0ab8858ca8831a206-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e0802e2898522a0ab8858ca8831a206-Abstract-Conference.html)

        **Abstract**:

        The Hessian of a differentiable convex function is positive semidefinite. Therefore, checking the Hessian of a given function is a natural approach to certify convexity. However, implementing this approach is not straightforward, since it requires a representation of the Hessian that allows its analysis. Here, we implement this approach for a class of functions that is rich enough to support classical machine learning. For this class of functions, it was recently shown how to compute computational graphs of their Hessians. We show how to check these graphs for positive-semidefiniteness. We compare our implementation of the Hessian approach with the well-established disciplined convex programming (DCP) approach and prove that the Hessian approach is at least as powerful as the DCP approach for differentiable functions. Furthermore, we show for a state-of-the-art implementation of the DCP approach that the Hessian approach is  actually more powerful, that is, it can certify the convexity of a larger class of differentiable functions.

        ----

        ## [503] FR: Folded Rationalization with a Unified Encoder

        **Authors**: *Wei Liu, Haozhao Wang, Jun Wang, Ruixuan Li, Chao Yue, Yuankai Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e0bd92a1d3600d4288df51ac5e6be5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e0bd92a1d3600d4288df51ac5e6be5f-Abstract-Conference.html)

        **Abstract**:

        Rationalization aims to strengthen the interpretability of NLP models by extracting a subset of human-intelligible pieces of their inputting texts. Conventional works generally employ a two-phase model in which a generator selects the most important pieces, followed by a predictor that makes predictions based on the selected pieces. However, such a two-phase model may incur the degeneration problem where the predictor overfits to the noise generated by a not yet well-trained generator and in turn, leads the generator to converge to a suboptimal model that tends to select senseless pieces. To tackle this challenge, we propose Folded Rationalization (FR) that folds the two phases of the rationale model into one from the perspective of text semantic extraction. The key idea of FR is to employ a unified encoder between the generator and predictor, based on which FR can facilitate a better predictor by access to valuable information blocked by the generator in the traditional two-phase model and thus bring a better generator. Empirically, we show that FR improves the F1 score by up to 10.3% as compared to state-of-the-art methods.

        ----

        ## [504] Max-Min Off-Policy Actor-Critic Method Focusing on Worst-Case Robustness to Model Misspecification

        **Authors**: *Takumi Tanabe, Rei Sato, Kazuto Fukuchi, Jun Sakuma, Youhei Akimoto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e0f5561c1553a97cee5fa64575358c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e0f5561c1553a97cee5fa64575358c9-Abstract-Conference.html)

        **Abstract**:

        In the field of reinforcement learning, because of the high cost and risk of policy training in the real world, policies are trained in a simulation environment and transferred to the corresponding real-world environment.However, the simulation environment does not perfectly mimic the real-world environment, lead to model misspecification. Multiple studies report significant deterioration of policy performance in a real-world environment.In this study, we focus on scenarios involving a simulation environment with uncertainty parameters and the set of their possible values, called the uncertainty parameter set. The aim is to optimize the worst-case performance on the uncertainty parameter set to guarantee the performance in the corresponding real-world environment.To obtain a policy for the optimization, we propose an off-policy actor-critic approach called the Max-Min Twin Delayed Deep Deterministic Policy Gradient algorithm (M2TD3), which solves a max-min optimization problem using a simultaneous gradient ascent descent approach.Experiments in multi-joint dynamics with contact (MuJoCo) environments show that the proposed method exhibited a worst-case performance superior to several baseline approaches.

        ----

        ## [505] One-Inlier is First: Towards Efficient Position Encoding for Point Cloud Registration

        **Authors**: *Fan Yang, Lin Guo, Zhi Chen, Wenbing Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e163450c1ae3167832971e6da29f38d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e163450c1ae3167832971e6da29f38d-Abstract-Conference.html)

        **Abstract**:

        Transformer architecture has shown great potential for many visual tasks, including point cloud registration. As an order-aware module, position encoding plays an important role in Transformer architecture applied to point cloud registration task. In this paper, we propose OIF-PCR, a one-inlier based position encoding method for point cloud registration network. Specifically, we first find one correspondence by a differentiable optimal transport layer, and use it to normalize each point for position encoding. It can eliminate the challenges brought by the different reference frames of two point clouds, and mitigate the feature ambiguity by learning the spatial consistency. Then, we propose a joint approach for establishing correspondence and position encoding, presenting an iterative optimization process. Finally, we design a progressive way for point cloud alignment and feature learning to gradually optimize the rigid transformation. The proposed position encoding is very efficient, requiring only a small addition of memory and computing overhead. Extensive experiments demonstrate the proposed method can achieve competitive performance with the state-of-the-art methods in both indoor and outdoor scenes.

        ----

        ## [506] Adaptive Distribution Calibration for Few-Shot Learning with Hierarchical Optimal Transport

        **Authors**: *Dandan Guo, Long Tian, He Zhao, Mingyuan Zhou, Hongyuan Zha*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html)

        **Abstract**:

        Few-shot classification aims to learn a classifier to recognize unseen classes during training, where the learned model can easily become over-fitted based on the biased distribution formed by only a few training examples. A recent solution to this problem is calibrating the distribution of these few sample classes by transferring statistics from the base classes with sufficient examples, where how to decide the transfer weights from base classes to novel classes is the key. However, principled approaches for learning the transfer weights have not been carefully studied. To this end, we propose a novel distribution calibration method by learning the adaptive weight matrix between novel samples and base classes, which is built upon a hierarchical Optimal Transport (H-OT) framework. By minimizing the high-level OT distance between novel samples and base classes, we can view the learned transport plan as the adaptive weight information for transferring the statistics of base classes. The learning of the cost function between a base class and novel class in the high-level OT leads to the introduction of the low-level OT, which considers the weights of all the data samples in the base class. Experimental results on standard benchmarks demonstrate that our proposed plug-and-play model outperforms competing approaches and owns desired cross-domain generalization ability, indicating the effectiveness of the learned adaptive weights.

        ----

        ## [507] InsNet: An Efficient, Flexible, and Performant Insertion-based Text Generation Model

        **Authors**: *Sidi Lu, Tao Meng, Nanyun Peng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e32d3a10985fc94c7e11ee6ea165cca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e32d3a10985fc94c7e11ee6ea165cca-Abstract-Conference.html)

        **Abstract**:

        We propose InsNet, an expressive insertion-based text generator with efficient training and flexible decoding (parallel or sequential). Unlike most existing insertion-based text generation works that require re-encoding of the (decoding) context after each insertion operation and thus are inefficient to train, InsNet only requires one pass of context encoding for the entire insertion sequence during training by using a novel insertion-oriented position encoding to enable computation sharing. Furthermore, InsNet provides a controllable switch between parallel and sequential decoding, making it flexible to handle more parallelizable tasks such as machine translation to support efficient decoding, or less parallelizable tasks such as lexically constrained text generation to guarantee high-quality outputs. Experiments on two unsupervised lexically constrained text generation datasets and three machine translation datasets demonstrate InsNetâ€™s advantages over previous insertion-based methods in terms of training speed, inference efficiency, and generation quality.

        ----

        ## [508] Weighted Distillation with Unlabeled Examples

        **Authors**: *Fotis Iliopoulos, Vasilis Kontonis, Cenk Baykal, Gaurav Menghani, Khoa Trinh, Erik Vee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e3435554b430bd8fe92a60c509929a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e3435554b430bd8fe92a60c509929a0-Abstract-Conference.html)

        **Abstract**:

        Distillation with unlabeled examples is a popular and powerful method for training deep neural networks in settings where the amount of labeled data is limited: A large “teacher” neural network is trained on the labeled data available, and then it is used to generate labels on an unlabeled dataset (typically much larger in size). These labels are then utilized to train the smaller “student” model which will actually be deployed. Naturally, the success of the approach depends on the quality of the teacher’s labels, since the student could be confused if trained on inaccurate data. This paper proposes a principled approach for addressing this issue based on a “debiasing" reweighting of the student’s loss function tailored to the distillation training paradigm. Our method is hyper-parameter free, data-agnostic, and simple to implement. We demonstrate significant improvements on popular academic datasets and we accompany our results with a theoretical analysis which rigorously justifies the performance of our method in certain settings.

        ----

        ## [509] When Does Group Invariant Learning Survive Spurious Correlations?

        **Authors**: *Yimeng Chen, Ruibin Xiong, Zhi-Ming Ma, Yanyan Lan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e351740d4ec4200df6160f34cd181c3-Abstract-Conference.html)

        **Abstract**:

        By inferring latent groups in the training data, recent works introduce invariant learning to the case where environment annotations are unavailable. Typically, learning group invariance under a majority/minority split is empirically shown to be effective in improving out-of-distribution generalization on many datasets. However, theoretical guarantee for these methods on learning invariant mechanisms is lacking. In this paper, we reveal the insufficiency of existing group invariant learning methods in preventing classifiers from depending on spurious correlations in the training set. Specifically, we propose two criteria on judging such sufficiency. Theoretically and empirically, we show that existing methods can violate both criteria and thus fail in generalizing to spurious correlation shifts. Motivated by this, we design a new group invariant learning method, which constructs groups with statistical independence tests, and reweights samples by group label proportion to meet the criteria. Experiments on both synthetic and real data demonstrate that the new method significantly outperforms existing group invariant learning methods in generalizing to spurious correlation shifts.

        ----

        ## [510] SNAKE: Shape-aware Neural 3D Keypoint Field

        **Authors**: *Chengliang Zhong, Peixing You, Xiaoxue Chen, Hao Zhao, Fuchun Sun, Guyue Zhou, Xiaodong Mu, Chuang Gan, Wenbing Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e3eccb54649186564ad6627ed80848c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e3eccb54649186564ad6627ed80848c-Abstract-Conference.html)

        **Abstract**:

        Detecting 3D keypoints from point clouds is important for shape reconstruction, while this work investigates the dual question: can shape reconstruction benefit 3D keypoint detection? Existing methods either seek salient features according to statistics of different orders or learn to predict keypoints that are invariant to transformation. Nevertheless, the idea of incorporating shape reconstruction into 3D keypoint detection is under-explored. We argue that this is restricted by former problem formulations. To this end, a novel unsupervised paradigm named SNAKE is proposed, which is short for shape-aware neural 3D keypoint field. Similar to recent coordinate-based radiance or distance field, our network takes 3D coordinates as inputs and predicts implicit shape indicators and keypoint saliency simultaneously, thus naturally entangling 3D keypoint detection and shape reconstruction. We achieve superior performance on various public benchmarks, including standalone object datasets ModelNet40, KeypointNet, SMPL meshes and scene-level datasets 3DMatch and Redwood. Intrinsic shape awareness brings several advantages as follows. (1) SNAKE generates 3D keypoints consistent with human semantic annotation, even without such supervision. (2) SNAKE outperforms counterparts in terms of repeatability, especially when the input point clouds are down-sampled. (3) the generated keypoints allow accurate geometric registration, notably in a zero-shot setting. Codes and models are available at https://github.com/zhongcl-thu/SNAKE.

        ----

        ## [511] Single Loop Gaussian Homotopy Method for Non-convex Optimization

        **Authors**: *Hidenori Iwakiri, Yuhang Wang, Shinji Ito, Akiko Takeda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e622ac74f66df03b686a12e2e0e4424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e622ac74f66df03b686a12e2e0e4424-Abstract-Conference.html)

        **Abstract**:

        The Gaussian homotopy (GH) method is a popular approach to finding better stationary points for non-convex optimization problems by gradually reducing a parameter value $t$, which changes the problem to be solved from an almost convex one to the original target one. Existing GH-based methods repeatedly call an iterative optimization solver to find a stationary point every time $t$ is updated, which incurs high computational costs. We propose a novel single loop framework for GH methods (SLGH) that updates the parameter $t$ and the optimization decision variables at the same. Computational complexity analysis is performed on the SLGH algorithm under various situations: either a gradient or gradient-free oracle of a GH function can be obtained for both deterministic and stochastic settings. The convergence rate of SLGH with a tuned hyperparameter becomes consistent with the convergence rate of gradient descent, even though the problem to be solved is gradually changed due to $t$. In numerical experiments, our SLGH algorithms show faster convergence than an existing double loop GH method while outperforming gradient descent-based methods in terms of finding a better solution.

        ----

        ## [512] Minimax Optimal Online Imitation Learning via Replay Estimation

        **Authors**: *Gokul Swamy, Nived Rajaraman, Matthew Peng, Sanjiban Choudhury, J. Andrew Bagnell, Steven Wu, Jiantao Jiao, Kannan Ramchandran*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2e809adc337594e0fee330a64acbb982-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2e809adc337594e0fee330a64acbb982-Abstract-Conference.html)

        **Abstract**:

        Online imitation learning is the problem of how best to mimic expert demonstrations, given access to the environment or an accurate simulator. Prior work has shown that in the \textit{infinite} sample regime, exact moment matching achieves value equivalence to the expert policy. However, in the \textit{finite} sample regime, even if one has no optimization error, empirical variance can lead to a performance gap that scales with $H^2 / N_{\text{exp}}$ for behavioral cloning and $H / N_{\text{exp}}$ for online moment matching, where $H$ is the horizon and $N_{\text{exp}}$ is the size of the expert dataset. We introduce the technique of ``replay estimation'' to reduce this empirical variance: by repeatedly executing cached expert actions in a stochastic simulator, we compute a smoother expert visitation distribution estimate to match. In the presence of general function approximation, we prove a meta theorem reducing the performance gap of our approach to the \textit{parameter estimation error} for offline classification (i.e. learning the expert policy). In the tabular setting or with linear function approximation, our meta theorem shows that the performance gap incurred by our approach achieves the optimal $\widetilde{O} \left( \min( H^{3/2} / N_{\text{exp}}, H / \sqrt{N_{\text{exp}}} \right)$ dependency, under significantly weaker assumptions compared to prior work. We implement multiple instantiations of our approach on several continuous control tasks and find that we are able to significantly improve policy performance across a variety of dataset sizes.

        ----

        ## [513] Multi-layer State Evolution Under Random Convolutional Design

        **Authors**: *Max Daniels, Cédric Gerbelot, Florent Krzakala, Lenka Zdeborová*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2eb74636e69ca26ab6cba8b724b0776e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2eb74636e69ca26ab6cba8b724b0776e-Abstract-Conference.html)

        **Abstract**:

        Signal recovery under generative neural network priors has emerged as a promising direction in statistical inference and computational imaging. Theoretical analysis of reconstruction algorithms under generative priors is, however, challenging. For generative priors with fully connected layers and Gaussian i.i.d. weights, this was achieved by the multi-layer approximate message (ML-AMP) algorithm via a rigorous state evolution. However, practical generative priors are typically convolutional, allowing for computational benefits and inductive biases, and so the Gaussian i.i.d. weight assumption is very limiting. In this paper, we overcome this limitation and establish the state evolution of ML-AMP for random convolutional layers. We prove in particular that random convolutional layers belong to the same universality class as Gaussian matrices. Our proof technique is of an independent interest as it establishes a mapping between convolutional matrices and spatially coupled sensing matrices used in coding theory.

        ----

        ## [514] Mixture-of-Experts with Expert Choice Routing

        **Authors**: *Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Y. Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le, James Laudon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)

        **Abstract**:

        Sparsely-activated Mixture-of-experts (MoE) models allow the number of parameters to greatly increase while keeping the amount of computation for a given token or a given sample unchanged. However, a poor expert routing strategy (e.g. one resulting in load imbalance) can cause certain experts to be under-trained, leading to an expert being under or over-specialized. Prior work allocates a fixed number of experts to each token using a top-k function regardless of the relative importance of different tokens. To address this, we propose a heterogeneous mixture-of-experts employing an expert choice method. Instead of letting tokens select the top-k experts, we have experts selecting the top-k tokens. As a result, each token can be routed to a variable number of experts and each expert can have a fixed bucket size. We systematically study pre-training speedups using the same computational resources of the Switch Transformer top-1 and GShard top-2 gating of prior work and find that our method improves training convergence time by more than 2Ã—. For the same computational cost, our method demonstrates higher performance in fine-tuning 11 selected tasks in the GLUE and SuperGLUE benchmarks. For a smaller activation cost, our method outperforms the T5 dense model in 7 out of the 11 tasks.

        ----

        ## [515] GULP: a prediction-based metric between representations

        **Authors**: *Enric Boix-Adserà, Hannah Lawrence, George Stepaniants, Philippe Rigollet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f0435cffef91068ced08d7c7d8e643e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f0435cffef91068ced08d7c7d8e643e-Abstract-Conference.html)

        **Abstract**:

        Comparing the representations learned by different neural networks has recently emerged as a key tool to understand various architectures and ultimately optimize them. In this work, we introduce GULP, a family of distance measures between representations that is explicitly motivated by  downstream predictive tasks. By construction, GULP provides uniform control over the difference in prediction performance between two representations, with respect to regularized linear prediction tasks. Moreover, it satisfies several desirable structural properties, such as the triangle inequality and invariance under orthogonal transformations, and thus lends itself to data embedding and visualization. We extensively evaluate GULP relative to other methods, and demonstrate that it correctly differentiates between architecture families, converges over the course of training, and captures generalization performance on downstream linear tasks.

        ----

        ## [516] On the convergence of policy gradient methods to Nash equilibria in general stochastic games

        **Authors**: *Angeliki Giannou, Kyriakos Lotidis, Panayotis Mertikopoulos, Emmanouil V. Vlatakis-Gkaragkounis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f060912eacace9ce61ef339205ec54c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f060912eacace9ce61ef339205ec54c-Abstract-Conference.html)

        **Abstract**:

        Learning in stochastic games is a notoriously difficult problem because, in addition to each other's strategic decisions, the players must also contend with the fact that the game itself evolves over time, possibly in a very complicated manner. Because of this, the convergence properties of popular learning algorithms — like policy gradient and its variants — are poorly understood, except in specific classes of games (such as potential or two-player, zero-sum games). In view of this, we examine the long-run behavior of policy gradient methods with respect to Nash equilibrium policies that are second-order stationary (SOS) in a sense similar to the type of sufficiency conditions used in optimization. Our first result is that SOS policies are locally attracting with high probability, and we show that policy gradient trajectories with gradient estimates provided by the REINFORCE algorithm achieve an $\mathcal{O}(1/\sqrt{n})$ distance-squared convergence rate if the method's step-size is chosen appropriately. Subsequently, specializing to the class of deterministic Nash policies, we show that this rate can be improved dramatically and, in fact, policy gradient methods converge within a finite number of iterations in that case.

        ----

        ## [517] SAGDA: Achieving $\mathcal{O}(\epsilon^{-2})$ Communication Complexity in Federated Min-Max Learning

        **Authors**: *Haibo Yang, Zhuqing Liu, Xin Zhang, Jia Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f13806d6580db60d9d7d6f89ba529ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f13806d6580db60d9d7d6f89ba529ca-Abstract-Conference.html)

        **Abstract**:

        Federated min-max learning has received increasing attention in recent years thanks to its wide range of applications in various learning paradigms. Similar to the conventional federated learning for empirical risk minimization problems, communication complexity also emerges as one of the most critical concerns that affects the future prospect of federated min-max learning. To lower the communication complexity of federated min-max learning, a natural approach is to utilize the idea of infrequent communications (through multiple local updates) same as in conventional federated learning. However, due to the more complicated inter-outer problem structure in federated min-max learning, theoretical understandings of communication complexity for federated min-max learning with infrequent communications remain very limited in the literature. This is particularly true for settings with non-i.i.d. datasets and partial client participation. To address this challenge, in this paper, we propose a new algorithmic framework called \ul{s}tochastic \ul{s}ampling \ul{a}veraging \ul{g}radient \ul{d}escent \ul{a}scent ($\mathsf{SAGDA}$), which i) assembles stochastic gradient estimators from randomly sampled clients as control variates  and ii) leverages two learning rates on both server and client sides. We show that $\mathsf{SAGDA}$ achieves a linear speedup in terms of both the number of clients and local update steps, which yields an $\mathcal{O}(\epsilon^{-2})$ communication complexity that is orders of magnitude lower than the state of the art. Interestingly, by noting that the standard federated stochastic gradient descent ascent (FSGDA) is in fact a control-variate-free special version of $\mathsf{SAGDA}$, we immediately arrive at an $\mathcal{O}(\epsilon^{-2})$ communication complexity result for FSGDA. Therefore, through the lens of $\mathsf{SAGDA}$, we also advance the current understanding on communication complexity of the standard FSGDA method for federated min-max learning.

        ----

        ## [518] ALMA: Hierarchical Learning for Composite Multi-Agent Tasks

        **Authors**: *Shariq Iqbal, Robby Costales, Fei Sha*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html)

        **Abstract**:

        Despite significant progress on multi-agent reinforcement learning (MARL) in recent years, coordination in complex domains remains a challenge. Work in MARL often focuses on solving tasks where agents interact with all other agents and entities in the environment; however, we observe that real-world tasks are often composed of several isolated instances of local agent interactions (subtasks), and each agent can meaningfully focus on one subtask to the exclusion of all else in the environment. In these composite tasks, successful policies can often be decomposed into two levels of decision-making: agents are allocated to specific subtasks and each agent acts productively towards their assigned subtask alone. This decomposed decision making provides a strong structural inductive bias, significantly reduces agent observation spaces, and encourages subtask-specific policies to be reused and composed during training, as opposed to treating each new composition of subtasks as unique. We introduce ALMA, a general learning method for taking advantage of these structured tasks. ALMA simultaneously learns a high-level subtask allocation policy and low-level agent policies. We demonstrate that ALMA learns sophisticated coordination behavior in a number of challenging environments, outperforming strong baselines. ALMA's modularity also enables it to better generalize to new environment configurations. Finally, we find that while ALMA can integrate separately trained allocation and action policies, the best performance is obtained only by training all components jointly. Our code is available at https://github.com/shariqiqbal2810/ALMA

        ----

        ## [519] Improved Bounds on Neural Complexity for Representing Piecewise Linear Functions

        **Authors**: *Kuan-Lin Chen, Harinath Garudadri, Bhaskar D. Rao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f4b6febe0b70805c3be75e5d6a66918-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f4b6febe0b70805c3be75e5d6a66918-Abstract-Conference.html)

        **Abstract**:

        A deep neural network using rectified linear units represents a continuous piecewise linear (CPWL) function and vice versa. Recent results in the literature estimated that the number of neurons needed to exactly represent any CPWL function grows exponentially with the number of pieces or exponentially in terms of the factorial of the number of distinct linear components. Moreover, such growth is amplified linearly with the input dimension. These existing results seem to indicate that the cost of representing a CPWL function is expensive. In this paper, we propose much tighter bounds and establish a polynomial time algorithm to find a network satisfying these bounds for any given CPWL function. We prove that the number of hidden neurons required to exactly represent any CPWL function is at most a quadratic function of the number of pieces. In contrast to all previous results, this upper bound is invariant to the input dimension. Besides the number of pieces, we also study the number of distinct linear components in CPWL functions. When such a number is also given, we prove that the quadratic complexity turns into bilinear, which implies a lower neural complexity because the number of distinct linear components is always not greater than the minimum number of pieces in a CPWL function. When the number of pieces is unknown, we prove that, in terms of the number of distinct linear components, the neural complexities of any CPWL function are at most polynomial growth for low-dimensional inputs and factorial growth for the worst-case scenario, which are significantly better than existing results in the literature.

        ----

        ## [520] Assaying Out-Of-Distribution Generalization in Transfer Learning

        **Authors**: *Florian Wenzel, Andrea Dittadi, Peter V. Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f5acc925919209370a3af4eac5cad4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f5acc925919209370a3af4eac5cad4a-Abstract-Conference.html)

        **Abstract**:

        Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.

        ----

        ## [521] M2N: Mesh Movement Networks for PDE Solvers

        **Authors**: *Wenbin Song, Mingrui Zhang, Joseph G. Wallwork, Junpeng Gao, Zheng Tian, Fanglei Sun, Matthew D. Piggott, Junqing Chen, Zuoqiang Shi, Xiang Chen, Jun Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f88d8061f12abae9d14d376fd69c933-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f88d8061f12abae9d14d376fd69c933-Abstract-Conference.html)

        **Abstract**:

        Numerical Partial Differential Equation (PDE) solvers often require discretizing the physical domain by using a mesh. Mesh movement methods provide the capability to improve the accuracy of the numerical solution without introducing extra computational burden to the PDE solver, by increasing mesh resolution where the solution is not well-resolved, whilst reducing unnecessary resolution elsewhere. However, sophisticated mesh movement methods, such as the Monge-Ampère method, generally require the solution of auxiliary equations. These solutions can be extremely expensive to compute when the mesh needs to be adapted frequently. In this paper, we propose to the best of our knowledge the first learning-based end-to-end mesh movement framework for PDE solvers. Key requirements of learning-based mesh movement methods are: alleviating mesh tangling, boundary consistency, and generalization to mesh with different resolutions. To achieve these goals, we introduce the neural spline model and the graph attention network (GAT) into our models respectively. While the Neural-Spline based model provides more flexibility for large mesh deformation, the GAT based model can handle domains with more complicated shapes and is better at performing delicate local deformation. We validate our methods on stationary and time-dependent, linear and non-linear equations, as well as regularly and irregularly shaped domains. Compared to the traditional Monge-Ampère method, our approach can greatly accelerate the mesh adaptation process by three to four orders of magnitude, whilst achieving comparable numerical error reduction.

        ----

        ## [522] Physics-Informed Implicit Representations of Equilibrium Network Flows

        **Authors**: *Kevin D. Smith, Francesco Seccamonte, Ananthram Swami, Francesco Bullo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f891d026c7ba978168621842bc6fe73-Abstract-Conference.html)

        **Abstract**:

        Flow networks are ubiquitous in natural and engineered systems, and in order to understand and manage these networks, one must quantify the flow of commodities across their edges. This paper considers the estimation problem of predicting unlabeled edge flows from nodal supply and demand. We propose an implicit neural network layer that incorporates two fundamental physical laws: conservation of mass, and the existence of a constitutive relationship between edge flows and nodal states (e.g., Ohm's law). Computing the edge flows from these two laws is a nonlinear inverse problem, which our layer solves efficiently with a specialized contraction mapping. Using implicit differentiation to compute the solution's gradients, our model is able to learn the constitutive relationship within a semi-supervised framework. We demonstrate that our approach can accurately predict edge flows in several experiments on AC power networks and water distribution systems.

        ----

        ## [523] Learning Interface Conditions in Domain Decomposition Solvers

        **Authors**: *Ali Taghibakhshi, Nicolas Nytko, Tareq Uz Zaman, Scott P. MacLachlan, Luke N. Olson, Matthew West*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f8928efe957139e9c0efc98f173f4be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f8928efe957139e9c0efc98f173f4be-Abstract-Conference.html)

        **Abstract**:

        Domain decomposition methods are widely used and effective in the approximation of solutions to partial differential equations.  Yet the \textit{optimal} construction of these methods requires tedious analysis and is often available only in simplified, structured-grid settings, limiting their use for more complex problems. In this work, we generalize optimized Schwarz domain decomposition methods to unstructured-grid problems, using Graph Convolutional Neural Networks (GCNNs) and unsupervised learning to learn optimal modifications at subdomain interfaces. A key ingredient in our approach is an improved loss function, enabling effective training on relatively small problems, but robust performance on arbitrarily large problems, with computational cost linear in problem size. The performance of the learned linear solvers is compared with both classical and optimized domain decomposition algorithms, for both structured- and unstructured-grid problems.

        ----

        ## [524] TANKBind: Trigonometry-Aware Neural NetworKs for Drug-Protein Binding Structure Prediction

        **Authors**: *Wei Lu, Qifeng Wu, Jixian Zhang, Jiahua Rao, Chengtao Li, Shuangjia Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f89a23a19d1617e7fb16d4f7a049ce2-Abstract-Conference.html)

        **Abstract**:

        Illuminating interactions between proteins and small drug molecules is a long-standing challenge in the field of drug discovery. Despite the importance of understanding these interactions, most previous works are limited by hand-designed scoring functions and insufficient conformation sampling. The recently-proposed graph neural network-based methods provides alternatives to predict protein-ligand complex conformation in a one-shot manner. However, these methods neglect the geometric constraints of the complex structure and weaken the role of local functional regions. As a result, they might produce unreasonable conformations for challenging targets and generalize poorly to novel proteins. In this paper, we propose Trigonometry-Aware Neural networKs for binding structure prediction, TANKBind, that builds trigonometry constraint as a vigorous inductive bias into the model and explicitly attends to all possible binding sites for each protein by segmenting the whole protein into functional blocks. We construct novel contrastive losses with local region negative sampling to jointly optimize the binding interaction and affinity. Extensive experiments show substantial performance gains in comparison to state-of-the-art physics-based and deep learning-based methods on commonly-used benchmark datasets for both binding structure and affinity predictions with variant settings.

        ----

        ## [525] Hamiltonian Latent Operators for content and motion disentanglement in image sequences

        **Authors**: *Asif Khan, Amos J. Storkey*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f8ee6a3d766b426d2618e555b5aeb39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f8ee6a3d766b426d2618e555b5aeb39-Abstract-Conference.html)

        **Abstract**:

        We introduce \textit{HALO} -- a deep generative model utilising HAmiltonian Latent Operators to reliably disentangle content and motion information in image sequences. The \textit{content} represents summary statistics of a sequence, and \textit{motion} is a dynamic process that determines how information is expressed in any part of the sequence. By modelling the dynamics as a Hamiltonian motion, important desiderata are ensured: (1) the motion is reversible, (2) the symplectic, volume-preserving structure in phase space means paths are continuous and are not divergent in the latent space. Consequently, the nearness of sequence frames is realised by the nearness of their coordinates in the phase space, which proves valuable for disentanglement and long-term sequence generation. The sequence space is generally comprised of different types of dynamical motions. To ensure long-term separability and allow controlled generation, we associate every motion with a unique Hamiltonian that acts in its respective subspace. We demonstrate the utility of \textit{HALO} by swapping the motion of a pair of sequences, controlled generation, and image rotations.

        ----

        ## [526] Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited

        **Authors**: *Mingguo He, Zhewei Wei, Ji-Rong Wen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2f9b3ee2bcea04b327c09d7e3145bd1e-Abstract-Conference.html)

        **Abstract**:

        Designing spectral convolutional networks is a challenging problem in graph learning. ChebNet, one of the early attempts, approximates the spectral graph convolutions using Chebyshev polynomials. GCN simplifies ChebNet by utilizing only the first two Chebyshev polynomials while still outperforming it on real-world datasets. GPR-GNN and BernNet demonstrate that the Monomial and Bernstein bases also outperform the Chebyshev basis in terms of learning the spectral graph convolutions. Such conclusions are counter-intuitive in the field of approximation theory, where it is established that the Chebyshev polynomial achieves the optimum convergent rate for approximating a function. In this paper, we revisit the problem of approximating the spectral graph convolutions with Chebyshev polynomials. We show that ChebNet's inferior performance is primarily due to illegal coefficients learnt by ChebNet approximating analytic filter functions, which leads to over-fitting. We then propose ChebNetII, a new GNN model based on Chebyshev interpolation, which enhances the original Chebyshev polynomial approximation while reducing the Runge phenomenon. We conducted an extensive experimental study to demonstrate that ChebNetII can learn arbitrary graph convolutions and achieve superior performance in both full- and semi-supervised node classification tasks. Most notably, we scale ChebNetII to a billion graph ogbn-papers100M, showing that spectral-based GNNs have superior performance. Our code is available at https://github.com/ivam-he/ChebNetII.

        ----

        ## [527] A Kernelised Stein Statistic for Assessing Implicit Generative Models

        **Authors**: *Wenkai Xu, Gesine D. Reinert*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2fb462e23667ad5e6471a4e9af8e4774-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2fb462e23667ad5e6471a4e9af8e4774-Abstract-Conference.html)

        **Abstract**:

        Synthetic data generation has become a key ingredient for training machine learning procedures,  addressing tasks such as data augmentation, analysing privacy-sensitive data, or visualising representative samples. Assessing the quality of such synthetic data generators hence has to be addressed. As (deep) generative models for synthetic data often do not admit explicit probability distributions, classical statistical procedures for assessing model goodness-of-fit may not be applicable. In this paper, we propose a principled procedure to assess the quality of a synthetic data generator. The procedure is a Kernelised Stein Discrepancy-type test which is based on a non-parametric Stein operator for the synthetic data generator of interest. This operator is estimated from samples which are obtained from the synthetic data generator and hence can be applied even when the model is only implicit. In contrast to classical testing, the sample size from the synthetic data generator can be as large as desired, while the size of the observed data that the generator aims to emulate is fixed. Experimental results on synthetic distributions and trained generative models on synthetic and real datasets illustrate that the method shows improved power performance compared to existing approaches.

        ----

        ## [528] Fine-Grained Semantically Aligned Vision-Language Pre-Training

        **Authors**: *Juncheng Li, Xin He, Longhui Wei, Long Qian, Linchao Zhu, Lingxi Xie, Yueting Zhuang, Qi Tian, Siliang Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2fb4be70fc9668e9ec2c71b34fb127d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2fb4be70fc9668e9ec2c71b34fb127d4-Abstract-Conference.html)

        **Abstract**:

        Large-scale vision-language pre-training has shown impressive advances in a wide range of downstream tasks. Existing methods mainly model the cross-modal alignment by the similarity of the global representations of images and text, or advanced cross-modal attention upon image and text features. However, they fail to explicitly learn the fine-grained semantic alignment between visual regions and textual phrases, as only global image-text alignment information is available. In this paper, we introduce LOUPE, a fine-grained semantically aLigned visiOn-langUage PrE-training framework, which learns fine-grained semantic alignment from the novel perspective of game-theoretic interactions. To efficiently estimate the game-theoretic interactions, we further propose an uncertainty-aware neural Shapley interaction learning module. Experiments show that LOUPE achieves state-of-the-art performance on a variety of  vision-language tasks. Without any object-level human annotations and fine-tuning, LOUPE achieves competitive performance on object detection and visual grounding. More importantly, LOUPE opens a new promising direction of learning fine-grained semantics from large-scale raw image-text pairs.

        ----

        ## [529] Fast Distance Oracles for Any Symmetric Norm

        **Authors**: *Yichuan Deng, Zhao Song, Omri Weinstein, Ruizhe Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2fc6b8a3fc23108f184daa4759024c25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2fc6b8a3fc23108f184daa4759024c25-Abstract-Conference.html)

        **Abstract**:

        In the \emph{Distance Oracle} problem, the goal is to preprocess $n$ vectors $x_1, x_2, \cdots, x_n$ in a $d$-dimensional normed space $(\mathbb{X}^d, \| \cdot \|_l)$ into a cheap data structure, so that given a query vector $q \in \mathbb{X}^d$, all distances $\| q - x_i \|_l$ to the data points $\{x_i\}_{i\in [n]}$ can be quickly approximated (faster than the trivial $\sim nd$ query time). This primitive is a basic subroutine in machine learning, data mining and similarity search applications. In the case of $\ell_p$ norms, the problem is well understood, and optimal data structures are known for most values of $p$.  Our main contribution is a fast $(1\pm \varepsilon)$ distance oracle for  \emph{any symmetric} norm $\|\cdot\|_l$. This class includes $\ell_p$ norms and Orlicz norms as special cases, as well as other norms used in practice, e.g. top-$k$ norms, max-mixture and sum-mixture of $\ell_p$ norms, small-support norms and the box-norm. We propose a novel data structure with  $\tilde{O}(n (d + \mathrm{mmc}(l)^2 ) )$ preprocessing time and space, and $t_q = \tilde{O}(d + n \cdot \mathrm{mmc}(l)^2)$ query time, where $\mathrm{mmc}(l)$ is a complexity-measure (modulus) of the symmetric norm under consideration. When $l = \ell_{p}$ , this runtime matches the aforementioned state-of-art oracles.

        ----

        ## [530] Outlier-Robust Sparse Estimation via Non-Convex Optimization

        **Authors**: *Yu Cheng, Ilias Diakonikolas, Rong Ge, Shivam Gupta, Daniel Kane, Mahdi Soltanolkotabi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/2fefbb34af8008e81fb3f457fa5a2fc2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/2fefbb34af8008e81fb3f457fa5a2fc2-Abstract-Conference.html)

        **Abstract**:

        We explore the connection between outlier-robust high-dimensional statistics and non-convex optimization in the presence of sparsity constraints, with a focus on the fundamental tasks of robust sparse mean estimation and robust sparse PCA. We develop novel and simple optimization formulations for these problems such that any approximate stationary point of the associated optimization problem yields a near-optimal solution for the underlying robust estimation task. As a corollary, we obtain that any first-order method that efficiently converges to stationarity yields an efficient algorithm for these tasks. The obtained algorithms are simple, practical, and succeed under broader distributional assumptions compared to prior work.

        ----

        ## [531] Recurrent Convolutional Neural Networks Learn Succinct Learning Algorithms

        **Authors**: *Surbhi Goel, Sham M. Kakade, Adam Kalai, Cyril Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/300900a706e788163b88ed3c08cbe23c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/300900a706e788163b88ed3c08cbe23c-Abstract-Conference.html)

        **Abstract**:

        Neural networks (NNs) struggle to efficiently solve certain problems, such as learning parities, even when there are simple learning algorithms for those problems. Can NNs discover learning algorithms on their own? We exhibit a NN architecture that, in polynomial time, learns as well as any efficient learning algorithm describable by a constant-sized program. For example, on parity problems, the NN learns as well as Gaussian elimination, an efficient algorithm that can be succinctly described. Our architecture combines both recurrent weight sharing between layers and convolutional weight sharing to reduce the number of parameters down to a constant, even though the network itself may have trillions of nodes. While in practice the constants in our analysis are too large to be directly meaningful, our work suggests that the synergy of Recurrent and Convolutional NNs (RCNNs) may be more natural and powerful than either alone, particularly for concisely parameterizing discrete algorithms.

        ----

        ## [532] Navigating Memory Construction by Global Pseudo-Task Simulation for Continual Learning

        **Authors**: *Yejia Liu, Wang Zhu, Shaolei Ren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3013680bf2d072b5f3851aec70b39a59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3013680bf2d072b5f3851aec70b39a59-Abstract-Conference.html)

        **Abstract**:

        Continual learning faces a crucial challenge of catastrophic forgetting. To address this challenge, experience replay (ER) that maintains a tiny subset of samples from previous tasks has been commonly used. Existing ER works usually focus on refining the learning objective for each task with a static memory construction policy. In this paper, we formulate the dynamic memory construction in ER as a combinatorial optimization problem, which aims at directly minimizing the global loss across all experienced tasks. We first apply three tactics to solve the problem in the offline setting as a starting point. To provide an approximate solution to this problem under the online continual learning setting, we further propose the Global Pseudo-task Simulation (GPS), which mimics future catastrophic forgetting of the current task by permutation. Our empirical results and analyses suggest that the GPS consistently improves accuracy across four commonly used vision benchmarks. We have also shown that our GPS can serve as the unified framework for integrating various memory construction policies in existing ER works.

        ----

        ## [533] Contact-aware Human Motion Forecasting

        **Authors**: *Wei Mao, Miaomiao Liu, Richard I. Hartley, Mathieu Salzmann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3018804d037cc101b73624f381bed0cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3018804d037cc101b73624f381bed0cb-Abstract-Conference.html)

        **Abstract**:

        In this paper, we tackle the task of scene-aware 3D human motion forecasting, which consists of predicting future human poses given a 3D scene and a past human motion. A key challenge of this task is to ensure consistency between the human and the scene, accounting for human-scene interactions. Previous attempts to do so model such interactions only implicitly, and thus tend to produce artifacts such as ``ghost motion" because of the lack of explicit constraints between the local poses and the global motion. Here, by contrast, we propose to explicitly model the human-scene contacts. To this end, we introduce distance-based contact maps that capture the contact relationships between every joint and every 3D scene point at each time instant. We then develop a two-stage pipeline that first predicts the future contact maps from the past ones and the scene point cloud, and then forecasts the future human poses by conditioning them on the predicted contact maps. During training, we explicitly encourage consistency between the global motion and the local poses via a prior defined using the contact maps and future poses. Our approach outperforms the state-of-the-art human motion forecasting and human synthesis methods on both synthetic and real datasets. Our code is available at https://github.com/wei-mao-2019/ContAwareMotionPred.

        ----

        ## [534] Spectral Bias in Practice: The Role of Function Frequency in Generalization

        **Authors**: *Sara Fridovich-Keil, Raphael Gontijo Lopes, Rebecca Roelofs*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/306264db5698839230be3642aafc849c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/306264db5698839230be3642aafc849c-Abstract-Conference.html)

        **Abstract**:

        Despite their ability to represent highly expressive functions, deep learning models seem to find simple solutions that generalize surprisingly well. Spectral bias -- the tendency of neural networks to prioritize learning low frequency functions -- is one possible explanation for this phenomenon, but so far spectral bias has primarily been observed in theoretical models and simplified experiments. In this work, we propose methodologies for measuring spectral bias in modern image classification networks on CIFAR-10 and ImageNet. We find that these networks indeed exhibit spectral bias, and that interventions that improve test accuracy on CIFAR-10 tend to produce learned functions that have higher frequencies overall but lower frequencies in the vicinity of examples from each class. This trend holds across variation in training time, model architecture, number of training examples, data augmentation, and self-distillation. We also explore the connections between function frequency and image frequency and find that spectral bias is sensitive to the low frequencies prevalent in natural images. On ImageNet, we find that learned function frequency also varies with internal class diversity, with higher frequencies on more diverse classes. Our work enables measuring and ultimately influencing the spectral behavior of neural networks used for image classification, and is a step towards understanding why deep models generalize well.

        ----

        ## [535] Out-of-Distribution Detection with An Adaptive Likelihood Ratio on Informative Hierarchical VAE

        **Authors**: *Yewen Li, Chaojie Wang, Xiaobo Xia, Tongliang Liu, Xin Miao, Bo An*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3066f60a91d652f4dc690637ac3a2f8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3066f60a91d652f4dc690637ac3a2f8c-Abstract-Conference.html)

        **Abstract**:

        Unsupervised out-of-distribution (OOD) detection is essential for the reliability of machine learning. In the literature, existing work has shown that higher-level semantics captured by hierarchical VAEs can be used to detect OOD instances.However, we empirically show that, the inherent issue of hierarchical VAEs, i.e., `posterior collapse'', would seriously limit their capacity for OOD detection.Based on a thorough analysis forposterior collapse'', we propose a novel informative hierarchical VAE to alleviate this issue through enhancing the connections between the data sample and its multi-layer stochastic latent representations during training.Furthermore, we propose a novel score function for unsupervised OOD detection, referred to as Adaptive Likelihood Ratio. With this score function, one can selectively aggregate the semantic information on multiple hidden layers of hierarchical VAEs, leading to a strong separability between in-distribution and OOD samples. Experimental results demonstrate that our method can significantly outperform existing state-of-the-art unsupervised OOD detection approaches.

        ----

        ## [536] FOF: Learning Fourier Occupancy Field for Monocular Real-time Human Reconstruction

        **Authors**: *Qiao Feng, Yebin Liu, Yu-Kun Lai, Jingyu Yang, Kun Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/30d046e94d7b8037d6ef27c4357a8dd4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/30d046e94d7b8037d6ef27c4357a8dd4-Abstract-Conference.html)

        **Abstract**:

        The advent of deep learning has led to significant progress in monocular human reconstruction. However, existing representations, such as parametric models, voxel grids, meshes and implicit neural representations, have difficulties achieving high-quality results and real-time speed at the same time. In this paper, we propose Fourier Occupancy Field (FOF), a novel, powerful, efficient and flexible 3D geometry representation, for monocular real-time and accurate human reconstruction. A FOF represents a 3D object with a 2D field orthogonal to the view direction where at each 2D position the occupancy field of the object along the view direction is compactly represented with the first few terms of Fourier series, which retains the topology and neighborhood relation in the 2D domain. A FOF can be stored as a multi-channel image, which is compatible with 2D convolutional neural networks and can bridge the gap between 3D geometries and 2D images. A FOF is very flexible and extensible, \eg, parametric models can be easily integrated into a FOF as a prior to generate more robust results. Meshes and our FOF can be easily inter-converted. Based on FOF, we design the first 30+FPS high-fidelity real-time monocular human reconstruction framework. We demonstrate the potential of FOF on both public datasets and real captured data. The code is available for research purposes at http://cic.tju.edu.cn/faculty/likun/projects/FOF.

        ----

        ## [537] Rate-Optimal Online Convex Optimization in Adaptive Linear Control

        **Authors**: *Asaf B. Cassel, Alon Peled-Cohen, Tomer Koren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/30dfe47a3ccbee68cffa0c19ccb1bc00-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of controlling an unknown linear dynamical system under adversarially-changing convex costs and full feedback of both the state and cost function. We present the first computationally-efficient algorithm that attains an optimal $\sqrt{T}$-regret rate compared to the best stabilizing linear controller in hindsight, while avoiding stringent assumptions on the costs such as strong convexity. Our approach is based on a careful design of non-convex lower confidence bounds for the online costs, and uses a novel technique for computationally-efficient regret minimization of these bounds that leverages their particular non-convex structure.

        ----

        ## [538] RTFormer: Efficient Design for Real-Time Semantic Segmentation with Transformer

        **Authors**: *Jian Wang, Chenhui Gou, Qiman Wu, Haocheng Feng, Junyu Han, Errui Ding, Jingdong Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/30e10e671c5e43edb67eb257abb6c3ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/30e10e671c5e43edb67eb257abb6c3ea-Abstract-Conference.html)

        **Abstract**:

        Recently, transformer-based networks have shown impressive results in semantic segmentation. Yet for real-time semantic segmentation, pure CNN-based approaches still dominate in this field, due to the time-consuming computation mechanism of transformer. We propose RTFormer, an efficient dual-resolution transformer for real-time semantic segmenation, which achieves better trade-off between performance and efficiency than CNN-based models. To achieve high inference efficiency on GPU-like devices, our RTFormer leverages GPU-Friendly Attention with linear complexity and discards the multi-head mechanism. Besides, we find that cross-resolution attention is more efficient to gather global context information for high-resolution branch by spreading the high level knowledge learned from low-resolution branch. Extensive experiments on mainstream benchmarks demonstrate the effectiveness of our proposed RTFormer, it achieves state-of-the-art on Cityscapes, CamVid and COCOStuff, and shows promising results on ADE20K.

        ----

        ## [539] Staggered Rollout Designs Enable Causal Inference Under Interference Without Network Knowledge

        **Authors**: *Mayleen Cortez, Matthew Eichhorn, Christina Lee Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3103b25853719847502559bf67eb4037-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3103b25853719847502559bf67eb4037-Abstract-Conference.html)

        **Abstract**:

        Randomized experiments are widely used to estimate causal effects across many domains. However, classical causal inference approaches rely on independence assumptions that are violated by network interference, when the treatment of one individual influences the outcomes of others. All existing approaches require at least approximate knowledge of the network, which may be unavailable or costly to collect. We consider the task of estimating the total treatment effect (TTE), the average difference between the outcomes when the whole population is treated versus when the whole population is untreated. By leveraging a staggered rollout design, in which treatment is incrementally given to random subsets of individuals, we derive unbiased estimators for TTE that do not rely on any prior structural knowledge of the network, as long as the network interference effects are constrained to low-degree interactions among neighbors of an individual. We derive bounds on the variance of the estimators, and we show in experiments that our estimator performs well against baselines on simulated data. Central to our theoretical contribution is a connection between staggered rollout observations and polynomial extrapolation.

        ----

        ## [540] Censored Quantile Regression Neural Networks for Distribution-Free Survival Analysis

        **Authors**: *Tim Pearce, Jong-Hyeon Jeong, Yichen Jia, Jun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/312c92e0cb862422eaa49452cdf55caf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/312c92e0cb862422eaa49452cdf55caf-Abstract-Conference.html)

        **Abstract**:

        This paper considers doing quantile regression on censored data using neural networks (NNs). This adds to the survival analysis toolkit by allowing direct prediction of the target variable, along with a distribution-free characterisation of uncertainty, using a flexible function approximator. We begin by showing how an algorithm popular in linear models can be applied to NNs. However, the resulting procedure is inefficient, requiring sequential optimisation of an individual NN at each desired quantile. Our major contribution is a novel algorithm that simultaneously optimises a grid of quantiles output by a single NN. To offer theoretical insight into our algorithm, we show firstly that it can be interpreted as a form of expectation-maximisation, and secondly that it exhibits a desirable `self-correcting' property. Experimentally, the algorithm produces quantiles that are better calibrated than existing methods on 10 out of 12 real datasets.

        ----

        ## [541] Learning Graph-embedded Key-event Back-tracing for Object Tracking in Event Clouds

        **Authors**: *Zhiyu Zhu, Junhui Hou, Xianqiang Lyu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31421b112e5f7faf4fc577b74e45dab2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31421b112e5f7faf4fc577b74e45dab2-Abstract-Conference.html)

        **Abstract**:

        Event data-based object tracking is attracting attention increasingly. Unfortunately, the unusual data structure caused by the unique sensing mechanism poses great challenges in designing downstream algorithms. To tackle such challenges,  existing methods usually re-organize raw event data (or event clouds) with the event frame/image representation to adapt to mature RGB data-based tracking paradigms, which compromises the high temporal resolution and sparse characteristics. By contrast, we advocate developing new designs/techniques tailored to the special data structure to realize object tracking. To this end, we make the first attempt to construct a new end-to-end learning-based paradigm that directly consumes event clouds. Specifically, to process a non-uniformly distributed large-scale event cloud efficiently, we propose a simple yet effective density-insensitive downsampling strategy to sample a subset called key-events. Then, we employ a graph-based network to embed the irregular spatio-temporal information of key-events into a high-dimensional feature space, and the resulting embeddings are utilized to predict their target likelihoods via semantic-driven Siamese-matching. Besides, we also propose motion-aware target likelihood prediction, which learns the motion flow to back-trace the potential initial positions of key-events and measures them with the previous proposal. Finally, we obtain the bounding box by adaptively fusing the two intermediate ones separately regressed from the weighted embeddings of key-events by the two types of predicted target likelihoods. Extensive experiments on both synthetic and real event datasets demonstrate the superiority of the proposed framework over state-of-the-art methods in terms of both the tracking accuracy and speed. The code is publicly available at https://github.com/ZHU-Zhiyu/Event-tracking.

        ----

        ## [542] Efficient and Near-Optimal Smoothed Online Learning for Generalized Linear Functions

        **Authors**: *Adam Block, Max Simchowitz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31455446488c433ef29495ab44d4f53c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31455446488c433ef29495ab44d4f53c-Abstract-Conference.html)

        **Abstract**:

        Due to the drastic gap in complexity between sequential and batch statistical learning, recent work has studied a smoothed  sequential learning setting, where Nature is constrained to select contexts with density bounded by $1/\sigma$ with respect to a known measure $\mu$. Unfortunately, for some function classes, there is an exponential gap between the statistically optimal regret and that which can be achieved efficiently.  In this paper, we give a computationally efficient algorithm that is the first to enjoy the statistically optimal $\log(T/\sigma)$ regret for realizable $K$-wise linear classification. We extend our results to settings where the true classifier is linear in an over-parameterized polynomial featurization of the contexts, as well as to a realizable piecewise-regression setting assuming access to an appropriate ERM oracle.  Somewhat surprisingly, standard disagreement-based analyses are insufficient to achieve regret logarithmic in $1/\sigma$.  Instead,  we develop a novel characterization of the geometry of the disagreement region induced by generalized linear classifiers. Along the way, we develop numerous technical tools of independent interest, including a general anti-concentration bound for the determinant of certain  matrix averages.

        ----

        ## [543] Systematic improvement of neural network quantum states using Lanczos

        **Authors**: *Hongwei Chen, Douglas Hendry, Phillip Weinberg, Adrian E. Feiguin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3173c427cb4ed2d5eaab029c17f221ae-Abstract-Conference.html)

        **Abstract**:

        The quantum many-body problem lies at the center of the most important open challenges in condensed matter, quantum chemistry, atomic, nuclear, and high-energy physics. While quantum Monte Carlo, when applicable, remains the most powerful numerical technique capable of treating dozens or hundreds of degrees of freedom with high accuracy, it is restricted to models that are not afflicted by the infamous sign problem. A powerful alternative that has emerged in recent years is the use of neural networks as variational estimators for quantum states. In this work, we propose a symmetry-projected variational solution in the form of linear combinations of simple restricted Boltzmann machines. This construction allows one to explore states outside of the original variational manifold and increase the representation power with moderate computational effort. Besides allowing one to restore spatial symmetries, an expansion in terms of Krylov states using a Lanczos recursion offers a solution that can further improve the quantum state accuracy. We illustrate these ideas with an application to the Heisenberg $J_1-J_2$ model on the square lattice, a paradigmatic problem under debate in condensed matter physics, and achieve state-of-the-art accuracy in the representation of the ground state.

        ----

        ## [544] Grounded Reinforcement Learning: Learning to Win the Game under Human Commands

        **Authors**: *Shusheng Xu, Huaijie Wang, Yi Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/318f3ae8be3c97cb7555e1c932f472a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/318f3ae8be3c97cb7555e1c932f472a1-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of building a reinforcement learning (RL) agent that can both accomplish non-trivial tasks, like winning a real-time strategy game, and strictly follow high-level language commands from humans, like “attack”, even if a command is sub-optimal. We call this novel yet important problem, Grounded Reinforcement Learning (GRL). Compared with other language grounding tasks, GRL is particularly non-trivial and cannot be simply solved by pure RL or behavior cloning (BC). From the RL perspective, it is extremely challenging to derive a precise reward function for human preferences since the commands are abstract and the valid behaviors are highly complicated and multi-modal. From the BC perspective, it is impossible to obtain perfect demonstrations since human strategies in complex games are typically sub-optimal. We tackle GRL via a simple, tractable, and practical constrained RL objective and develop an iterative RL algorithm, REinforced demonstration Distillation (RED), to obtain a strong GRL policy. We evaluate the policies derived by RED, BC and pure RL methods on a simplified real-time strategy game, MiniRTS. Experiment results and human studies show that the RED policy is able to consistently follow human commands and achieve a higher win rate than the baselines. We release our code and present more examples at https://sites.google.com/view/grounded-rl.

        ----

        ## [545] Enhance the Visual Representation via Discrete Adversarial Training

        **Authors**: *Xiaofeng Mao, Yuefeng Chen, Ranjie Duan, Yao Zhu, Gege Qi, Shaokai Ye, Xiaodan Li, Rong Zhang, Hui Xue*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31928aa24124da335bec23f5e1f91a46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31928aa24124da335bec23f5e1f91a46-Abstract-Conference.html)

        **Abstract**:

        Adversarial Training (AT), which is commonly accepted as one of the most effective approaches defending against adversarial examples, can largely harm the standard performance, thus has limited usefulness on industrial-scale production and applications. Surprisingly, this phenomenon is totally opposite in Natural Language Processing (NLP) task, where AT can even benefit for generalization. We notice the merit of AT in NLP tasks could derive from the discrete and symbolic input space. For borrowing the advantage from NLP-style AT, we propose Discrete Adversarial Training (DAT). DAT leverages VQGAN to reform the image data to discrete text-like inputs, i.e. visual words. Then it minimizes the maximal risk on such discrete images with symbolic adversarial perturbations. We further give an explanation from the perspective of distribution to demonstrate the effectiveness of DAT. As a plug-and-play technique for enhancing the visual representation, DAT achieves significant improvement on multiple tasks including image classification, object detection and self-supervised learning. Especially, the model pre-trained with Masked Auto-Encoding (MAE) and fine-tuned by our DAT without extra data can get 31.40 mCE on ImageNet-C and 32.77% top-1 accuracy on Stylized-ImageNet, building the new state-of-the-art. The code will be available at https://github.com/alibaba/easyrobust.

        ----

        ## [546] Trading Off Resource Budgets For Improved Regret Bounds

        **Authors**: *Thomas Orton, Damon Falck*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31a57804448363bcab777f818f75f5b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31a57804448363bcab777f818f75f5b4-Abstract-Conference.html)

        **Abstract**:

        In this work we consider a variant of adversarial online learning where in each round one picks $B$ out of $N$ arms and incurs cost equal to the $\textit{minimum}$ of the costs of each arm chosen. We propose an algorithm called Follow the Perturbed Multiple Leaders (FPML) for this problem, which we show (by adapting the techniques of Kalai and Vempala [2005]) achieves expected regret $\mathcal{O}(T^{\frac{1}{B+1}}\ln(N)^{\frac{B}{B+1}})$ over time horizon $T$ relative to the $\textit{single}$ best arm in hindsight. This introduces a trade-off between the budget $B$ and the single-best-arm regret, and we proceed to investigate several applications of this trade-off. First, we observe that algorithms which use standard regret minimizers as subroutines can sometimes be adapted by replacing these subroutines with FPML, and we use this to generalize existing algorithms for Online Submodular Function Maximization [Streeter and Golovin, 2008] in both the full feedback and semi-bandit feedback settings. Next, we empirically evaluate our new algorithms on an online black-box hyperparameter optimization problem. Finally, we show how FPML can lead to new algorithms for Linear Programming which require stronger oracles at the benefit of fewer oracle calls.

        ----

        ## [547] Natural Color Fool: Towards Boosting Black-box Unrestricted Attacks

        **Authors**: *Shengming Yuan, Qilong Zhang, Lianli Gao, Yaya Cheng, Jingkuan Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31d0d59fe946684bb228e9c8e887e176-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31d0d59fe946684bb228e9c8e887e176-Abstract-Conference.html)

        **Abstract**:

        Unrestricted color attacks, which manipulate semantically meaningful color of an image, have shown their stealthiness and success in fooling both human eyes and deep neural networks. However, current works usually sacrifice the flexibility of the uncontrolled setting to ensure the naturalness of adversarial examples. As a result, the black-box attack performance of these methods is limited. To boost transferability of adversarial examples without damaging image quality, we propose a novel Natural Color Fool (NCF) which is guided by realistic color distributions sampled from a publicly available dataset and optimized by our neighborhood search and initialization reset. By conducting extensive experiments and visualizations, we convincingly demonstrate the effectiveness of our proposed method. Notably, on average, results show that our NCF can outperform state-of-the-art approaches by 15.0%$\sim$32.9% for fooling normally trained models and 10.0%$\sim$25.3% for evading defense methods. Our code is available at https://github.com/VL-Group/Natural-Color-Fool.

        ----

        ## [548] Old can be Gold: Better Gradient Flow can Make Vanilla-GCNs Great Again

        **Authors**: *Ajay Jaiswal, Peihao Wang, Tianlong Chen, Justin F. Rousseau, Ying Ding, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31df5479712197232485d4c2387f6033-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31df5479712197232485d4c2387f6033-Abstract-Conference.html)

        **Abstract**:

        Despite the enormous success of Graph Convolutional Networks (GCNs) in modeling graph-structured data, most of the current GCNs are shallow due to the notoriously challenging problems of over-smoothening and information squashing along with conventional difficulty caused by vanishing gradients and over-fitting. Previous works have been primarily focused on the study of over-smoothening and over-squashing phenomena in training deep GCNs. Surprisingly, in comparison with CNNs/RNNs, very limited attention has been given to understanding how healthy gradient flow can benefit the trainability of deep GCNs. In this paper, firstly, we provide a new perspective of gradient flow to understand the substandard performance of deep GCNs and hypothesize that by facilitating healthy gradient flow, we can significantly improve their trainability, as well as achieve state-of-the-art (SOTA) level performance from vanilla-GCNs. Next, we argue that blindly adopting the Glorot initialization for GCNs is not optimal, and derive a topology-aware isometric initialization scheme for vanilla-GCNs based on the principles of isometry. Additionally, contrary to ad-hoc addition of skip-connections, we propose to use gradient-guided dynamic rewiring of vanilla-GCNs with skip connections. Our dynamic rewiring method uses the gradient flow within each layer during training to introduce on-demand skip-connections adaptively. We provide extensive empirical evidence across multiple datasets that our methods improve gradient flow in deep vanilla-GCNs and significantly boost their performance to comfortably compete and outperform many fancy state-of-the-art methods. Codes are available at:  https://github.com/VITA-Group/GradientGCN.

        ----

        ## [549] Egocentric Video-Language Pretraining

        **Authors**: *Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Zhongcong Xu, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, Chengfei Cai, Hongfa Wang, Dima Damen, Bernard Ghanem, Wei Liu, Mike Zheng Shou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/31fb284a0aaaad837d2930a610cd5e50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/31fb284a0aaaad837d2930a610cd5e50-Abstract-Conference.html)

        **Abstract**:

        Video-Language Pretraining (VLP), which aims to learn transferable representation to advance a wide range of video-text downstream tasks, has recently received increasing attention. Best performing works rely on large-scale, 3rd-person video-text datasets, such as HowTo100M. In this work, we exploit the recently released Ego4D dataset to pioneer Egocentric VLP along three directions. (i) We create EgoClip, a 1st-person video-text pretraining dataset comprising 3.8M clip-text pairs well-chosen from Ego4D, covering a large variety of human daily activities. (ii) We propose a novel pretraining objective, dubbed EgoNCE, which adapts video-text contrastive learning to the egocentric domain by mining egocentric-aware positive and negative samples. (iii) We introduce EgoMCQ, a development benchmark that is close to EgoClip and hence can support effective validation and fast exploration of our design decisions in EgoClip and EgoNCE. Furthermore, we demonstrate strong performance on five egocentric downstream tasks across three datasets: video-text retrieval on EPIC-KITCHENS-100; action recognition on Charades-Ego; natural language query, moment query, and object state change classification on Ego4D challenge benchmarks. The dataset and code are available at https://github.com/showlab/EgoVLP.

        ----

        ## [550] Society of Agents: Regret Bounds of Concurrent Thompson Sampling

        **Authors**: *Yan Chen, Perry Dong, Qinxun Bai, Maria Dimakopoulou, Wei Xu, Zhengyuan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32133a6a24d6554263d3584e3ac10faa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32133a6a24d6554263d3584e3ac10faa-Abstract-Conference.html)

        **Abstract**:

        We consider the concurrent reinforcement learning problem where $n$ agents simultaneously learn to make decisions in the same environment by sharing experience with each other. Existing works in this emerging area have empirically demonstrated that Thompson sampling (TS) based algorithms provide a particularly attractive alternative for inducing cooperation, because each agent can independently sample a belief environment (and compute a corresponding optimal policy) from the joint posterior computed by aggregating all agents' data , which induces diversity in exploration among agents while benefiting shared experience from all agents.  However, theoretical guarantees in this area remain under-explored; in particular, no regret bound is known on TS based concurrent RL algorithms.   In this paper, we fill in this gap by considering two settings. In the first, we study the simple finite-horizon episodic RL setting, where TS is naturally adapted into the concurrent setup by having each agent sample from the current joint posterior at the beginning of each episode. We establish a $\tilde{O}(HS\sqrt{\frac{AT}{n}})$ per-agent regret bound, where $H$ is the horizon of the episode, $S$ is the number of states, $A$ is the number of actions, $T$ is the number of episodes and $n$ is the number of agents.   In the second setting, we consider the infinite-horizon RL problem, where a policy is measured by its long-run average reward. Here, despite not having natural episodic breakpoints, we show that by a doubling-horizon schedule, we can adapt TS to the infinite-horizon concurrent learning setting to achieve a regret bound of $\tilde{O}(DS\sqrt{ATn})$, where $D$ is the standard notion of diameter of the underlying MDP and $T$ is the number of timesteps.  Note that in both settings, the per-agent regret decreases at an optimal rate of $\Theta(\frac{1}{\sqrt{n}})$, which manifests the power of cooperation in concurrent RL.

        ----

        ## [551] A Non-asymptotic Analysis of Non-parametric Temporal-Difference Learning

        **Authors**: *Eloïse Berthier, Ziad Kobeissi, Francis R. Bach*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32246544c237164c365c0527b677a79a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32246544c237164c365c0527b677a79a-Abstract-Conference.html)

        **Abstract**:

        Temporal-difference learning is a popular algorithm for policy evaluation. In this paper, we study the convergence of the regularized non-parametric TD(0) algorithm, in both the independent and Markovian observation settings. In particular, when TD is performed in a universal reproducing kernel Hilbert space (RKHS), we prove convergence of the averaged iterates to the optimal value function, even when it does not belong to the RKHS. We provide explicit convergence rates that depend on a source condition relating the regularity of the optimal value function to the RKHS. We illustrate this convergence numerically on a simple continuous-state Markov reward process.

        ----

        ## [552] CEIP: Combining Explicit and Implicit Priors for Reinforcement Learning with Demonstrations

        **Authors**: *Kai Yan, Alexander G. Schwing, Yu-Xiong Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/322e4a595afd9442a89f0bfaa441871e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/322e4a595afd9442a89f0bfaa441871e-Abstract-Conference.html)

        **Abstract**:

        Although reinforcement learning has found widespread use in dense reward settings, training autonomous agents with sparse rewards remains challenging. To address this difficulty, prior work has shown promising results when using not only task-specific demonstrations but also task-agnostic albeit somewhat related demonstrations. In most cases, the available demonstrations are distilled into an implicit prior, commonly represented via a single deep net. Explicit priors in the form of a database that can be queried have also been shown to lead to encouraging results. To better benefit from available demonstrations, we develop a method to Combine Explicit and Implicit Priors (CEIP). CEIP exploits multiple implicit priors in the form of normalizing flows in parallel to form a single complex prior. Moreover, CEIP uses an effective explicit retrieval and push-forward mechanism to condition the implicit priors. In three challenging environments, we find the proposed CEIP method to improve upon sophisticated state-of-the-art techniques.

        ----

        ## [553] Memorization and Optimization in Deep Neural Networks with Minimum Over-parameterization

        **Authors**: *Simone Bombari, Mohammad Hossein Amani, Marco Mondelli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/323746f0ae2fbd8b6f500dc2d5c5f898-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/323746f0ae2fbd8b6f500dc2d5c5f898-Abstract-Conference.html)

        **Abstract**:

        The Neural Tangent Kernel (NTK) has emerged as a powerful tool to provide memorization, optimization and generalization guarantees in deep neural networks. A line of work has studied the NTK spectrum for two-layer and deep networks with at least a layer with $\Omega(N)$ neurons, $N$ being the number of training samples. Furthermore, there is increasing evidence suggesting that deep networks with sub-linear layer widths are powerful memorizers and optimizers, as long as the number of parameters exceeds the number of samples. Thus, a natural open question is whether the NTK is well conditioned in such a challenging sub-linear setup. In this paper, we answer this question in the affirmative. Our key technical contribution is a lower bound on the smallest NTK eigenvalue for deep networks with the minimum possible over-parameterization: up to logarithmic factors, the number of parameters is $\Omega(N)$ and, hence, the number of neurons is as little as $\Omega(\sqrt{N})$. To showcase the applicability of our NTK bounds, we provide two results concerning memorization capacity and optimization guarantees for gradient descent training.

        ----

        ## [554] LAMP: Extracting Text from Gradients with Language Model Priors

        **Authors**: *Mislav Balunovic, Dimitar I. Dimitrov, Nikola Jovanovic, Martin T. Vechev*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32375260090404f907ceae19f3564a7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32375260090404f907ceae19f3564a7e-Abstract-Conference.html)

        **Abstract**:

        Recent work shows that sensitive user data can be reconstructed from gradient updates, breaking the key privacy promise of federated learning. While success was demonstrated primarily on image data, these methods do not directly transfer to other domains such as text. In this work, we propose LAMP, a novel attack tailored to textual data, that successfully reconstructs original text from gradients. Our attack is based on two key insights: (i) modelling prior text probability via an auxiliary language model, guiding the search towards more natural text, and (ii) alternating continuous and discrete optimization which minimizes reconstruction loss on embeddings while avoiding local minima via discrete text transformations. Our experiments demonstrate that LAMP is significantly more effective than prior work: it reconstructs 5x more bigrams and $23\%$ longer subsequences on average. Moreover, we are first to recover inputs from batch sizes larger than 1 for textual models. These findings indicate that gradient updates of models operating on textual data leak more information than previously thought.

        ----

        ## [555] Explainability Via Causal Self-Talk

        **Authors**: *Nicholas A. Roy, Junkyung Kim, Neil C. Rabinowitz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/324bb74b6d557428e21528379eeb7a0c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/324bb74b6d557428e21528379eeb7a0c-Abstract-Conference.html)

        **Abstract**:

        Explaining the behavior of AI systems is an important problem that, in practice, is generally avoided. While the XAI community has been developing an abundance of techniques, most incur a set of costs that the wider deep learning community has been unwilling to pay in most situations. We take a pragmatic view of the issue, and define a set of desiderata that capture both the ambitions of XAI and the practical constraints of deep learning. We describe an effective way to satisfy all the desiderata: train the AI system to build a causal model of itself. We develop an instance of this solution for Deep RL agents: Causal Self-Talk. CST operates by training the agent to communicate with itself across time. We implement this method in a simulated 3D environment, and show how it enables agents to generate faithful and semantically-meaningful explanations of their own behavior. Beyond explanations, we also demonstrate that these learned models provide new ways of building semantic control interfaces to AI systems.

        ----

        ## [556] Continual Learning with Evolving Class Ontologies

        **Authors**: *Zhiqiu Lin, Deepak Pathak, Yu-Xiong Wang, Deva Ramanan, Shu Kong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3255a7554605a88800f4e120b3a929e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3255a7554605a88800f4e120b3a929e1-Abstract-Conference.html)

        **Abstract**:

        Lifelong learners must recognize concept vocabularies that evolve over time. A common yet underexplored scenario is learning with class labels that continually refine/expand old classes. For example, humans learn to recognize ${\tt dog}$ before dog breeds. In practical settings, dataset ${\it versioning}$ often introduces refinement to ontologies, such as autonomous vehicle benchmarks that refine a previous ${\tt vehicle}$ class into ${\tt school-bus}$ as autonomous operations expand to new cities. This paper formalizes a protocol for studying the problem of ${\it Learning with Evolving Class Ontology}$ (LECO). LECO requires learning classifiers in distinct time periods (TPs); each TP introduces a new ontology of "fine" labels that refines old ontologies of  "coarse" labels (e.g., dog breeds that refine the previous ${\tt dog}$). LECO explores such questions as whether to annotate new data or relabel the old, how to exploit coarse labels, and whether to finetune the previous TP's model or train from scratch. To answer these questions, we leverage insights from related problems such as  class-incremental learning. We validate them under the LECO protocol through the lens of image classification (on CIFAR and iNaturalist) and semantic segmentation (on Mapillary). Extensive experiments lead to some surprising conclusions; while the current status quo in the field is to relabel existing datasets with new class ontologies (such as COCO-to-LVIS or Mapillary1.2-to-2.0), LECO demonstrates that a far better strategy is to annotate ${\it new}$ data with the new ontology. However, this produces an aggregate dataset with inconsistent old-vs-new labels, complicating learning. To address this challenge, we adopt methods from semi-supervised and partial-label learning. We demonstrate that such strategies can surprisingly be made near-optimal, in the sense of approaching an "oracle" that learns on the aggregate dataset exhaustively labeled with the newest ontology.

        ----

        ## [557] A gradient estimator via L1-randomization for online zero-order optimization with two point feedback

        **Authors**: *Arya Akhavan, Evgenii Chzhen, Massimiliano Pontil, Alexandre B. Tsybakov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/329ef22fd8cb68223d5df09a037f7dd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/329ef22fd8cb68223d5df09a037f7dd9-Abstract-Conference.html)

        **Abstract**:

        This work studies online zero-order optimization of convex and Lipschitz functions. We present  a novel gradient estimator based on two function evaluations and randomization on the $\ell_1$-sphere. Considering different geometries of feasible sets and Lipschitz assumptions we analyse online dual averaging algorithm with our estimator in place of the usual gradient. We consider two types of  assumptions on the noise of the zero-order oracle: canceling noise and adversarial noise. We provide an anytime and completely data-driven algorithm, which is adaptive to all parameters of the problem. In the case of canceling noise that was previously studied in the literature, our guarantees are either comparable or better than state-of-the-art bounds obtained by~\citet{duchi2015} and \citet{Shamir17} for non-adaptive algorithms. Our analysis is based on deriving a new weighted Poincar√© type inequality for the uniform measure on the $\ell_1$-sphere with explicit constants, which may be of independent interest.

        ----

        ## [558] On the SDEs and Scaling Rules for Adaptive Gradient Algorithms

        **Authors**: *Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, Sanjeev Arora*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32ac710102f0620d0f28d5d05a44fe08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32ac710102f0620d0f28d5d05a44fe08-Abstract-Conference.html)

        **Abstract**:

        Approximating Stochastic Gradient Descent (SGD) as a Stochastic Differential Equation (SDE) has allowed researchers to enjoy the benefits of studying a continuous optimization trajectory while carefully preserving the stochasticity of SGD. Analogous study of adaptive gradient methods, such as RMSprop and Adam, has been challenging because there were no rigorously proven SDE approximations for these methods. This paper derives the SDE approximations for RMSprop and Adam, giving theoretical guarantees of their correctness as well as experimental validation of their applicability to common large-scaling vision and language settings. A key practical result is the derivation of a square root scaling rule to adjust the optimization hyperparameters of RMSprop and Adam when changing batch size, and its empirical validation in deep learning settings.

        ----

        ## [559] Improving Neural Ordinary Differential Equations with Nesterov's Accelerated Gradient Method

        **Authors**: *Ho Huu Nghia Nguyen, Tan Nguyen, Huyen Vo, Stanley J. Osher, Thieu Vo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32cc61322f1e2f56f989d29ccc7cfbb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32cc61322f1e2f56f989d29ccc7cfbb7-Abstract-Conference.html)

        **Abstract**:

        We propose the Nesterov neural ordinary differential equations (NesterovNODEs), whose layers solve the second-order ordinary differential equations (ODEs) limit of Nesterov's accelerated gradient (NAG) method, and a generalization called GNesterovNODEs. Taking the advantage of the convergence rate $\mathcal{O}(1/k^{2})$ of the NAG scheme, GNesterovNODEs speed up training and inference by reducing the number of function evaluations (NFEs) needed to solve the ODEs. We also prove that the adjoint state of a GNesterovNODEs also satisfies a GNesterovNODEs, thus accelerating both forward and backward ODE solvers and allowing the model to be scaled up for large-scale tasks. We empirically corroborate the advantage of GNesterovNODEs on a wide range of practical applications, including point cloud separation, image classification, and sequence modeling. Compared to NODEs, GNesterovNODEs require a significantly smaller number of NFEs while achieving better accuracy across our experiments.

        ----

        ## [560] VER: Scaling On-Policy RL Leads to the Emergence of Navigation in Embodied Rearrangement

        **Authors**: *Erik Wijmans, Irfan Essa, Dhruv Batra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32d2cf79062126dc032ca4235068f52e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32d2cf79062126dc032ca4235068f52e-Abstract-Conference.html)

        **Abstract**:

        We present Variable Experience Rollout (VER), a technique for efficiently scaling batched on-policy reinforcement learning in heterogenous environments (where different environments take vastly different times to generate rollouts) to many GPUs residing on, potentially, many machines. VER combines the strengths of and blurs the line between synchronous and asynchronous on-policy RL methods (SyncOnRL and AsyncOnRL, respectively). Specifically, it learns from on-policy experience (like SyncOnRL) and has no synchronization points (like AsyncOnRL) enabling high throughput.We find that VER leads to significant and consistent speed-ups across a broad range of embodied navigation and mobile manipulation tasks in photorealistic 3D simulation environments. Specifically, for PointGoal navigation and ObjectGoal navigation in Habitat 1.0, VER is 60-100% faster (1.6-2x speedup) than DD-PPO, the current state of art for distributed SyncOnRL, with similar sample efficiency. For mobile manipulation tasks (open fridge/cabinet, pick/place objects) in Habitat 2.0 VER is 150% faster (2.5x speedup) on 1 GPU and 170% faster (2.7x speedup) on 8 GPUs than DD-PPO. Compared to SampleFactory (the current state-of-the-art AsyncOnRL), VER matches its speed on 1 GPU, and is 70% faster (1.7x speedup) on 8 GPUs with better sample efficiency.We leverage these speed-ups to train chained skills for GeometricGoal rearrangement tasks in the Home Assistant Benchmark (HAB). We find a surprising emergence of navigation in skills that do not ostensible require any navigation. Specifically, the Pick skill involves a robot picking an object from a table. During training the robot was always spawned close to the table and never needed to navigate. However, we find that if base movement is part of the action space, the robot learns to navigate then pick an object in new environments with 50% success, demonstrating surprisingly high out-of-distribution generalization.

        ----

        ## [561] Neural Attentive Circuits

        **Authors**: *Martin Weiss, Nasim Rahaman, Francesco Locatello, Chris Pal, Yoshua Bengio, Bernhard Schölkopf, Li Erran Li, Nicolas Ballas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32f227c41a0b4e36f65bebb4aeda94a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/32f227c41a0b4e36f65bebb4aeda94a2-Abstract-Conference.html)

        **Abstract**:

        Recent work has seen the development of general purpose neural architectures that can be trained to perform tasks across diverse data modalities. General purpose models typically make few assumptions about the underlying data-structure and are known to perform well in the large-data regime. At the same time, there has been growing interest in modular neural architectures that represent the data using sparsely interacting modules. These models can be more robust out-of-distribution, computationally efficient, and capable of sample-efficient adaptation to new data. However, they tend to make domain-specific assumptions about the data, and present challenges in how module behavior (i.e., parameterization) and connectivity (i.e., their layout) can be jointly learned. In this work, we introduce a general purpose, yet modular neural architecture called Neural Attentive Circuits (NACs) that jointly learns the parameterization and a sparse connectivity of neural modules without using domain knowledge. NACs are best understood as the combination of two systems that are jointly trained end-to-end: one that determines the module configuration and the other that executes it on an input.  We demonstrate qualitatively that NACs learn diverse and meaningful module configurations on the Natural Language and Visual Reasoning for Real (NLVR2) dataset without additional supervision. Quantitatively, we show that by incorporating modularity in this way, NACs improve upon a strong non-modular baseline in terms of low-shot adaptation on CIFAR and Caltech-UCSD Birds dataset (CUB) by about 10 percent, and OOD robustness on Tiny ImageNet-R by about 2.5 percent. Further, we find that NACs can achieve an 8x speedup at inference time while losing less than 3 percent performance. Finally, we find NACs to yield competitive results on diverse data modalities spanning point-cloud classification, symbolic processing and text-classification from ASCII bytes, thereby confirming its general purpose nature.

        ----

        ## [562] CLEVRER-Humans: Describing Physical and Causal Events the Human Way

        **Authors**: *Jiayuan Mao, Xuelin Yang, Xikun Zhang, Noah D. Goodman, Jiajun Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/32fea358f4811ec6d703e8c17028ce1d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/32fea358f4811ec6d703e8c17028ce1d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Building machines that can reason about physical events and their causal relationships is crucial for flexible interaction with the physical world. However, most existing physical and causal reasoning benchmarks are exclusively based on synthetically generated events and synthetic natural language descriptions of the causal relationships. This design brings up two issues. First, there is a lack of diversity in both event types and natural language descriptions; second, causal relationships based on manually-defined heuristics are different from human judgments. To address both shortcomings, we present the CLEVRER-Humans benchmark, a video reasoning dataset for causal judgment of physical events with human labels. We employ two techniques to improve data collection efficiency: first, a novel iterative event cloze task to elicit a new representation of events in videos, which we term Causal Event Graphs (CEGs); second, a data augmentation technique based on neural language generative models. We convert the collected CEGs into questions and answers to be consistent with prior work. Finally, we study a collection of baseline approaches for CLEVRER-Humans question-answering, highlighting great challenges set forth by our benchmark.

        ----

        ## [563] 3D Concept Grounding on Neural Fields

        **Authors**: *Yining Hong, Yilun Du, Chunru Lin, Josh Tenenbaum, Chuang Gan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/330073c95529dae593936d387edac58c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/330073c95529dae593936d387edac58c-Abstract-Conference.html)

        **Abstract**:

        In this paper, we address the challenging problem of 3D concept grounding (i.e., segmenting and learning visual concepts) by looking at RGBD images and reasoning about paired questions and answers. Existing visual reasoning approaches typically utilize supervised methods to extract 2D segmentation masks on which concepts are grounded. In contrast, humans are capable of grounding concepts on the underlying 3D representation of images. However, traditionally inferred 3D representations (e.g., point clouds, voxelgrids and meshes) cannot capture continuous 3D features flexibly, thus making it challenging to ground concepts to 3D regions based on the language description of the object being referred to. To address both issues, we propose to leverage the continuous, differentiable nature of neural fields to segment and learn concepts. Specifically, each 3D coordinate in a scene is represented as a high dimensional descriptor. Concept grounding can then be performed by computing the similarity between the descriptor vector of a 3D coordinate and the vector embedding of a language concept, which enables segmentations and concept learning to be jointly learned on neural fields in a differentiable fashion.  As a result, both 3D semantic and instance segmentations can emerge directly from question answering supervision using a set of defined neural operators on top of neural fields (e.g., filtering  and counting). Experimental results show that our proposed framework outperforms unsupervised / language-mediated segmentation models on semantic and instance segmentation tasks, as well as outperforms existing models on the challenging 3D aware visual reasoning tasks. Furthermore, our framework can generalize well to unseen shape categories and real scans.

        ----

        ## [564] Evaluating Graph Generative Models with Contrastively Learned Features

        **Authors**: *Hamed Shirzad, Kaveh Hassani, Danica J. Sutherland*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3309b4112c9f04a993f2bbdd0274bba1-Abstract-Conference.html)

        **Abstract**:

        A wide range of models have been proposed for Graph Generative Models, necessitating effective methods to evaluate their quality. So far, most techniques use either traditional metrics based on subgraph counting, or the representations of randomly initialized Graph Neural Networks (GNNs). We propose using representations from constrastively trained GNNs, rather than random GNNs, and show this gives more reliable evaluation metrics. Neither traditional approaches nor GNN-based approaches dominate the other, however: we give examples of graphs that each approach is unable to distinguish. We demonstrate that Graph Substructure Networks (GSNs), which in a way combine both approaches, are better at distinguishing the distances between graph datasets.

        ----

        ## [565] Path Independent Equilibrium Models Can Better Exploit Test-Time Computation

        **Authors**: *Cem Anil, Ashwini Pokle, Kaiqu Liang, Johannes Treutlein, Yuhuai Wu, Shaojie Bai, J. Zico Kolter, Roger B. Grosse*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/331c41353b053683e17f7c88a797701d-Abstract-Conference.html)

        **Abstract**:

        Designing networks capable of attaining better performance with an increased inference budget is important to facilitate generalization to harder problem instances. Recent efforts have shown promising results in this direction by making use of depth-wise recurrent networks. In this work, we reproduce the performance of the prior art using a broader class of architectures called equilibrium models, and find that stronger generalization performance on harder examples (which require more iterations of inference to get correct) strongly correlates with the path independence of the systemâ€”its ability to converge to the same attractor (or limit cycle) regardless of initialization, given enough computation. Experimental interventions made to promote path independence result in improved generalization on harder (and thus more compute-hungry) problem instances, while those that penalize it degrade this ability. Path independence analyses are also useful on a per-example basis: for equilibrium models that have good in-distribution performance, path independence on out-of-distribution samples strongly correlates with accuracy.  Thus, considering equilibrium models and path independence jointly leads to a valuable new viewpoint under which we can study the generalization performance of these networks on hard problem instances.

        ----

        ## [566] Implications of Model Indeterminacy for Explanations of Automated Decisions

        **Authors**: *Marc-Etienne Brunet, Ashton Anderson, Richard S. Zemel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/33201f38001dd381aba2c462051449ba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/33201f38001dd381aba2c462051449ba-Abstract-Conference.html)

        **Abstract**:

        There has been a significant research effort focused on explaining predictive models, for example through post-hoc explainability and recourse methods. Most of the proposed techniques operate upon a single, fixed, predictive model. However, it is well-known that given a dataset and a predictive task, there may be a multiplicity of models that solve the problem (nearly) equally well. In this work, we investigate the implications of this kind of model indeterminacy on the post-hoc explanations of predictive models. We show how it can lead to explanatory multiplicity, and we explore the underlying drivers. We show how predictive multiplicity, and the related concept of epistemic uncertainty, are not reliable indicators of explanatory multiplicity. We further illustrate how a set of models showing very similar aggregate performance on a test dataset may show large variations in their local explanations, i.e., for a specific input. We explore these effects for Shapley value based explanations on three risk assessment datasets. Our results indicate that model indeterminacy may have a substantial impact on explanations in practice, leading to inconsistent and even contradicting explanations.

        ----

        ## [567] Debugging and Explaining Metric Learning Approaches: An Influence Function Based Perspective

        **Authors**: *Ruofan Liu, Yun Lin, Xianglin Yang, Jin Song Dong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3322a9a72a1707de14badd5e552ff466-Abstract-Conference.html)

        **Abstract**:

        Deep metric learning (DML) learns a generalizable embedding space where the representations of semantically similar samples are closer. Despite achieving good performance, the state-of-the-art models still suffer from the generalization errors such as farther similar samples and closer dissimilar samples in the space. In this work, we design an empirical influence function (EIF), a debugging and explaining technique for the generalization errors of state-of-the-art metric learning models. EIF is designed to efficiently identify and quantify how a subset of training samples contributes to the generalization errors. Moreover, given a user-specific error, EIF can be used to relabel a potentially noisy training sample as mitigation. In our quantitative experiment, EIF outperforms the traditional baseline in identifying more relevant training samples with statistical significance and 33.5% less time. In the field study on well-known datasets such as CUB200, CARS196, and InShop, EIF identifies 4.4%, 6.6%, and 17.7% labelling mistakes, indicating the direction of the DML community to further improve the model performance. Our code is available at https://github.com/lindsey98/Influencefunctionmetric_learning.

        ----

        ## [568] Meta-Reinforcement Learning with Self-Modifying Networks

        **Authors**: *Mathieu Chalvidal, Thomas Serre, Rufin VanRullen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/332b4fbe322e11a71fa39d91c664d8fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/332b4fbe322e11a71fa39d91c664d8fa-Abstract-Conference.html)

        **Abstract**:

        Deep Reinforcement Learning has demonstrated the potential of neural networks tuned with gradient descent for solving complex tasks in well-delimited environments. However, these neural systems are slow learners producing specialized agents with no mechanism to continue learning beyond their training curriculum. On the contrary, biological synaptic plasticity is persistent and manifold, and has been hypothesized to play a key role in executive functions such as working memory and cognitive flexibility, potentially supporting more efficient and generic learning abilities. Inspired by this, we propose to build networks with dynamic weights, able to continually perform self-reflexive modification as a function of their current synaptic state and action-reward feedback, rather than a fixed network configuration. The resulting model, MetODS (for Meta-Optimized Dynamical Synapses) is a broadly applicable meta-reinforcement learning system able to learn efficient and powerful control rules in the agent policy space. A single layer with dynamic synapses can perform one-shot learning, generalize navigation principles to unseen environments and demonstrates a strong ability to learn adaptive motor policies, comparing favorably with previous meta-reinforcement learning approaches.

        ----

        ## [569] FairVFL: A Fair Vertical Federated Learning Framework with Contrastive Adversarial Learning

        **Authors**: *Tao Qi, Fangzhao Wu, Chuhan Wu, Lingjuan Lyu, Tong Xu, Hao Liao, Zhongliang Yang, Yongfeng Huang, Xing Xie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/333a7697dbb67f09249337f81c27d749-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/333a7697dbb67f09249337f81c27d749-Abstract-Conference.html)

        **Abstract**:

        Vertical federated learning (VFL) is a privacy-preserving machine learning paradigm that can learn models from features distributed on different platforms in a privacy-preserving way. Since in real-world applications the data may contain bias on fairness-sensitive features (e.g., gender), VFL models may inherit bias from training data and become unfair for some user groups. However, existing fair machine learning methods usually rely on the centralized storage of fairness-sensitive features to achieve model fairness, which are usually inapplicable in federated scenarios. In this paper, we propose a fair vertical federated learning framework (FairVFL), which can improve the fairness of VFL models. The core idea of FairVFL is to learn unified and fair representations of samples based on the decentralized feature fields in a privacy-preserving way. Specifically, each platform with fairness-insensitive features first learns local data representations from local features. Then, these local representations are uploaded to a server and aggregated into a unified representation for the target task. In order to learn a fair unified representation, we send it to each platform storing fairness-sensitive features and apply adversarial learning to remove bias from the unified representation inherited from the biased data. Moreover, for protecting user privacy, we further propose a contrastive adversarial learning method to remove private information from the unified representation in server before sending it to the platforms keeping fairness-sensitive features. Experiments on three real-world datasets validate that our method can effectively improve model fairness with user privacy well-protected.

        ----

        ## [570] Incorporating Bias-aware Margins into Contrastive Loss for Collaborative Filtering

        **Authors**: *An Zhang, Wenchang Ma, Xiang Wang, Tat-Seng Chua*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/334da4cbb76302f37bd2e9d86f558869-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/334da4cbb76302f37bd2e9d86f558869-Abstract-Conference.html)

        **Abstract**:

        Collaborative ﬁltering (CF) models easily suffer from popularity bias, which makes recommendation deviate from users’ actual preferences. However, most current debiasing strategies are prone to playing a trade-off game between head and tail performance, thus inevitably degrading the overall recommendation accuracy. To reduce the negative impact of popularity bias on CF models, we incorporate Bias-aware margins into Contrastive loss and propose a simple yet effective BC Loss, where the margin tailors quantitatively to the bias degree of each user-item interaction. We investigate the geometric interpretation of BC loss, then further visualize and theoretically prove that it simultaneously learns better head and tail representations by encouraging the compactness of similar users/items and enlarging the dispersion of dissimilar users/items. Over six benchmark datasets, we use BC loss to optimize two high-performing CF models. In various evaluation settings (i.e., imbalanced/balanced, temporal split, fully-observed unbiased, tail/head test evaluations), BC loss outperforms the state-of-the-art debiasing and non-debiasing methods with remarkable improvements. Considering the theoretical guarantee and empirical success of BC loss, we advocate using it not just as a debiasing strategy, but also as a standard loss in recommender models. Codes are available at https://github.com/anzhang314/BC-Loss.

        ----

        ## [571] Multiview Human Body Reconstruction from Uncalibrated Cameras

        **Authors**: *Zhixuan Yu, Linguang Zhang, Yuanlu Xu, Chengcheng Tang, Luan Tran, Cem Keskin, Hyun Soo Park*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/33610fba262d7b6fed0810b89f55e147-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/33610fba262d7b6fed0810b89f55e147-Abstract-Conference.html)

        **Abstract**:

        We present a new method to reconstruct 3D human body pose and shape by fusing visual features from multiview images captured by uncalibrated cameras. Existing multiview approaches often use spatial camera calibration (intrinsic and extrinsic parameters) to geometrically align and fuse visual features. Despite remarkable performances, the requirement of camera calibration restricted their applicability to real-world scenarios, e.g., reconstruction from social videos with wide-baseline cameras. We address this challenge by leveraging the commonly observed human body as a semantic calibration target, which eliminates the requirement of camera calibration. Specifically, we map per-pixel image features to a canonical body surface coordinate system agnostic to views and poses using dense keypoints (correspondences). This feature mapping allows us to semantically, instead of geometrically, align and fuse visual features from multiview images. We learn a self-attention mechanism to reason about the confidence of visual features across and within views. With fused visual features, a regressor is learned to predict the parameters of a body model. We demonstrate that our calibration-free multiview fusion method reliably reconstructs 3D body pose and shape, outperforming state-of-the-art single view methods with post-hoc multiview fusion, particularly in the presence of non-trivial occlusion, and showing comparable accuracy to multiview methods that require calibration.

        ----

        ## [572] MACK: Multimodal Aligned Conceptual Knowledge for Unpaired Image-text Matching

        **Authors**: *Yan Huang, Yuming Wang, Yunan Zeng, Liang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3379ce104189b72d5f7baaa03ae81329-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3379ce104189b72d5f7baaa03ae81329-Abstract-Conference.html)

        **Abstract**:

        Recently, the accuracy of image-text matching has been greatly improved by multimodal pretrained models, all of which are trained on millions or billions of paired images and texts. Different from them, this paper studies a new scenario as unpaired image-text matching, in which paired images and texts are assumed to be unavailable during model training. To deal with this, we propose a simple yet effective method namely Multimodal Aligned Conceptual Knowledge (MACK), which is inspired by the knowledge use in human brain. It can be directly used as general knowledge to correlate images and texts even without model training, or further fine-tuned based on unpaired images and texts to better generalize to certain datasets. In addition, we extend it as a re-ranking method, which can be easily combined with existing image-text matching models to substantially improve their performance.

        ----

        ## [573] Learning Optical Flow from Continuous Spike Streams

        **Authors**: *Rui Zhao, Ruiqin Xiong, Jing Zhao, Zhaofei Yu, Xiaopeng Fan, Tiejun Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/33951c28630e48c441cb59db356f2037-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/33951c28630e48c441cb59db356f2037-Abstract-Conference.html)

        **Abstract**:

        Spike camera is an emerging bio-inspired vision sensor with ultra-high temporal resolution. It records scenes by accumulating photons and outputting continuous binary spike streams. Optical flow is a key task for spike cameras and their applications. A previous attempt has been made for spike-based optical flow. However, the previous work only focuses on motion between two moments, and it uses graphics-based data for training, whose generalization is limited. In this paper, we propose a tailored network,  Spike2Flow that extracts information from binary spikes with temporal-spatial representation based on the differential of spike firing time and spatial information aggregation. The network utilizes continuous motion clues through joint correlation decoding. Besides, a new dataset with real-world scenes is proposed for better generalization. Experimental results show that our approach achieves state-of-the-art performance on existing synthetic datasets and real data captured by spike cameras. The source code and dataset are available at \url{https://github.com/ruizhao26/Spike2Flow}.

        ----

        ## [574] Improving 3D-aware Image Synthesis with A Geometry-aware Discriminator

        **Authors**: *Zifan Shi, Yinghao Xu, Yujun Shen, Deli Zhao, Qifeng Chen, Dit-Yan Yeung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/33bb58be3f0e903c75afa73d75b5c67e-Abstract-Conference.html)

        **Abstract**:

        3D-aware image synthesis aims at learning a generative model that can render photo-realistic 2D images while capturing decent underlying 3D shapes. A popular solution is to adopt the generative adversarial network (GAN) and replace the generator with a 3D renderer, where volume rendering with neural radiance field (NeRF) is commonly used. Despite the advancement of synthesis quality, existing methods fail to obtain moderate 3D shapes. We argue that, considering the two-player game in the formulation of GANs, only making the generator 3D-aware is not enough. In other words, displacing the generative mechanism only offers the capability, but not the guarantee, of producing 3D-aware images, because the supervision of the generator primarily comes from the discriminator. To address this issue, we propose GeoD through learning a geometry-aware discriminator to improve 3D-aware GANs. Concretely, besides differentiating real and fake samples from the 2D image space, the discriminator is additionally asked to derive the geometry information from the inputs, which is then applied as the guidance of the generator. Such a simple yet effective design facilitates learning substantially more accurate 3D shapes. Extensive experiments on various generator architectures and training datasets verify the superiority of GeoD over state-of-the-art alternatives. Moreover, our approach is registered as a general framework such that a more capable discriminator (i.e., with a third task of novel view synthesis beyond domain classification and geometry extraction) can further assist the generator with a better multi-view consistency. Project page can be found at https://vivianszf.github.io/geod.

        ----

        ## [575] A Consistent and Differentiable Lp Canonical Calibration Error Estimator

        **Authors**: *Teodora Popordanoska, Raphael Sayer, Matthew B. Blaschko*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/33d6e648ee4fb24acec3a4bbcd4f001e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/33d6e648ee4fb24acec3a4bbcd4f001e-Abstract-Conference.html)

        **Abstract**:

        Calibrated probabilistic classifiers are models whose predicted probabilities can directly be interpreted as uncertainty estimates. It has been shown recently that deep neural networks are poorly calibrated and tend to output overconfident predictions. As a remedy, we propose a low-bias, trainable calibration error estimator based on Dirichlet kernel density estimates, which asymptotically converges to the true $L_p$ calibration error. This novel estimator enables us to tackle the strongest notion of multiclass calibration, called canonical (or distribution) calibration, while other common calibration methods are tractable only for top-label and marginal calibration. The computational complexity of our estimator is $\mathcal{O}(n^2)$, the convergence rate is $\mathcal{O}(n^{-1/2})$, and it is unbiased up to $\mathcal{O}(n^{-2})$, achieved by a geometric series debiasing scheme. In practice, this means that the estimator can be applied to small subsets of data, enabling efficient estimation and mini-batch updates. The proposed method has a natural choice of kernel, and can be used to generate consistent estimates of other quantities based on conditional expectation, such as the sharpness of a probabilistic classifier. Empirical results validate the correctness of our estimator, and demonstrate its utility in canonical calibration error estimation and calibration error regularized risk minimization.

        ----

        ## [576] Transform Once: Efficient Operator Learning in Frequency Domain

        **Authors**: *Michael Poli, Stefano Massaroli, Federico Berto, Jinkyoo Park, Tri Dao, Christopher Ré, Stefano Ermon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/342339109d7d756ef7bb30cf672aec02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/342339109d7d756ef7bb30cf672aec02-Abstract-Conference.html)

        **Abstract**:

        Spectral analysis provides one of the most effective paradigms for information-preserving dimensionality reduction, as simple descriptions of naturally occurring signals are often obtained via few terms of periodic basis functions. In this work, we study deep neural networks designed to harness the structure in frequency domain for efficient learning of long-range correlations in space or time: frequency-domain models (FDMs). Existing FDMs are based on complex-valued transforms i.e. Fourier Transforms (FT), and layers that perform computation on the spectrum and input data separately. This design introduces considerable computational overhead: for each layer, a forward and inverse FT. Instead, this work introduces a blueprint for frequency domain learning through a single transform: transform once (T1). To enable efficient, direct learning in the frequency domain we derive a variance preserving weight initialization scheme and investigate methods for frequency selection in reduced-order FDMs. Our results noticeably streamline the design process of FDMs, pruning redundant transforms, and leading to speedups of 3x to 10x that increase with data resolution and model size. We perform extensive experiments on learning the solution operator of spatio-temporal dynamics, including incompressible Navier-Stokes, turbulent flows around airfoils and high-resolution video of smoke. T1 models improve on the test performance of FDMs while requiring significantly less computation (5 hours instead of 32 for our large-scale experiment), with over 20% reduction in predictive error across tasks.

        ----

        ## [577] Fused Orthogonal Alternating Least Squares for Tensor Clustering

        **Authors**: *Jiacheng Wang, Dan Nicolae*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/34260a400e39a802961470b3d3de99cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/34260a400e39a802961470b3d3de99cc-Abstract-Conference.html)

        **Abstract**:

        We introduce a multi-modes tensor clustering method that implements a fused version of the alternating least squares algorithm (Fused-Orth-ALS) for simultaneous tensor factorization and clustering.  The statistical convergence rates of recovery and clustering are established when the data are a noise contaminated tensor with a latent low rank CP decomposition structure. Furthermore, we show that a modified alternating least squares algorithm can provably recover the true latent low rank factorization structure when the data form an asymmetric tensor with perturbation. Clustering consistency is also established. Finally, we illustrate the accuracy and computational efficient implementation of the Fused-Orth-ALS algorithm by using both simulations and real datasets.

        ----

        ## [578] A Solver-free Framework for Scalable Learning in Neural ILP Architectures

        **Authors**: *Yatin Nandwani, Rishabh Ranjan, Mausam, Parag Singla*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3469b211b829b39d2b0cfd3b880a869c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3469b211b829b39d2b0cfd3b880a869c-Abstract-Conference.html)

        **Abstract**:

        There is a recent focus on designing architectures that have an Integer Linear Programming (ILP) layer within a neural model (referred to as \emph{Neural ILP} in this paper). Neural ILP architectures are suitable for pure reasoning tasks that require data-driven constraint learning or for tasks requiring both perception (neural) and reasoning (ILP). A recent SOTA approach for end-to-end training of Neural ILP explicitly defines gradients through the ILP black box [Paulus et al. [2021]] â€“ this trains extremely slowly, owing to a call to the underlying ILP solver for every training data point in a minibatch. In response, we present an alternative training strategy that is \emph{solver-free}, i.e., does not call the ILP solver at all at training time. Neural ILP has a set of trainable hyperplanes (for cost and constraints in ILP), together representing a polyhedron. Our key idea is that the training loss should impose that the final polyhedron separates the positives (all constraints satisfied) from the negatives (at least one violated constraint or a suboptimal cost value), via a soft-margin formulation.  While positive example(s) are provided as part of the training data, we devise novel techniques for generating negative samples. Our solution is flexible enough to handle equality as well as inequality constraints. Experiments on several problems, both perceptual as well as symbolic, which require learning the constraints of an ILP, show that our approach has superior performance and scales much better compared to purely neural baselines and other state-of-the-art models that require solver-based training. In particular, we are able to obtain excellent performance in 9 x 9 symbolic and visual Sudoku, to which the other Neural ILP solver is not able to scale.

        ----

        ## [579] Distributed Distributionally Robust Optimization with Non-Convex Objectives

        **Authors**: *Yang Jiao, Kai Yang, Dongjin Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/34899013589ef41aea4d7b2f0ef310c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/34899013589ef41aea4d7b2f0ef310c1-Abstract-Conference.html)

        **Abstract**:

        Distributionally Robust Optimization (DRO), which aims to find an optimal decision that minimizes the worst case cost over the ambiguity set of probability distribution, has been applied in diverse applications, e.g., network behavior analysis, risk management, etc. However, existing DRO techniques face three key challenges: 1) how to deal with the asynchronous updating in a distributed environment;  2) how to leverage the prior distribution effectively; 3) how to properly adjust the degree of robustness according to difference scenarios. To this end, we propose an asynchronous distributed algorithm, named Asynchronous Single-looP alternatIve gRadient projEction (ASPIRE) algorithm with the itErative Active SEt method (EASE) to tackle the distributed distributionally robust optimization (DDRO) problem. Furthermore, a new uncertainty set, i.e., constrained $D$-norm uncertainty set, is developed to effectively leverage the prior distribution and flexibly control the degree of robustness. Finally, our theoretical analysis elucidates that the proposed algorithm is guaranteed to converge and the iteration complexity is also analyzed. Extensive empirical studies on real-world datasets demonstrate that the proposed method can not only achieve fast convergence, remain robust against data heterogeneity and malicious attacks, but also tradeoff robustness with performance.

        ----

        ## [580] Parameter-free Regret in High Probability with Heavy Tails

        **Authors**: *Jiujia Zhang, Ashok Cutkosky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/349956dee974cfdcbbb2d06afad5dd4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/349956dee974cfdcbbb2d06afad5dd4a-Abstract-Conference.html)

        **Abstract**:

        We present new algorithms for online convex optimization over unbounded domains that obtain parameter-free regret in high-probability given access only to potentially heavy-tailed subgradient estimates. Previous work in unbounded domains con- siders only in-expectation results for sub-exponential subgradients. Unlike in the bounded domain case, we cannot rely on straight-forward martingale concentration due to exponentially large iterates produced by the algorithm. We develop new regularization techniques to overcome these problems. Overall, with probability at most δ, for all comparators u our algorithm achieves regret O ̃(∥u∥T 1/p log(1/δ)) for subgradients with bounded pth moments for some p ∈ (1, 2].

        ----

        ## [581] Non-Convex Bilevel Games with Critical Point Selection Maps

        **Authors**: *Michael Arbel, Julien Mairal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/349a45f211fb1b3850da1ccd829e869e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/349a45f211fb1b3850da1ccd829e869e-Abstract-Conference.html)

        **Abstract**:

        Bilevel optimization problems involve two nested objectives, where an upper-level objective depends on a solution to a lower-level problem. When the latter is non-convex, multiple critical points may be present, leading to an ambiguous definition of the problem. In this paper, we introduce a key ingredient for resolving this ambiguity through the concept of a selection map which allows one to choose a particular solution to the lower-level problem. Using such maps, we define a class of hierarchical games between two agents that resolve the ambiguity in bilevel problems. This new class of games requires introducing new analytical tools in Morse theory to extend implicit differentiation, a technique used in bilevel optimization resulting from the implicit function theorem. In particular, we establish the validity of such a method even when the latter theorem is inapplicable due to degenerate critical points.Finally, we show that algorithms for solving bilevel problems based on unrolled optimization solve these games up to approximation errors due to finite computational power. A simple correction to these algorithms is then proposed for removing these errors.

        ----

        ## [582] Consistent Sufficient Explanations and Minimal Local Rules for explaining the decision of any classifier or regressor

        **Authors**: *Salim I. Amoukou, Nicolas J.-B. Brunel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/34b13425b5ba8ad05d97b0043df52ed3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/34b13425b5ba8ad05d97b0043df52ed3-Abstract-Conference.html)

        **Abstract**:

        To explain the decision of any regression and classification model, we extend the notion of probabilistic sufficient explanations (P-SE). For each instance, this approach selects the minimal subset of features that is sufficient to yield the same prediction with high probability, while removing other features. The crux of P-SE is to compute the conditional probability of maintaining the same prediction. Therefore, we introduce an accurate and fast estimator of this probability via random Forests for any data $(\boldsymbol{X}, Y)$ and show its efficiency through a theoretical analysis of its consistency. As a consequence, we extend the P-SE to regression problems. In addition, we deal with non-discrete features, without learning the distribution of $\boldsymbol{X}$ nor having the model for making predictions. Finally, we introduce local rule-based explanations for regression/classification based on the P-SE and compare our approaches w.r.t other explainable AI methods. These methods are available as a Python Package.

        ----

        ## [583] High-dimensional Additive Gaussian Processes under Monotonicity Constraints

        **Authors**: *Andrés F. López-Lopera, François Bachoc, Olivier Roustant*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/34b70ece5f8d273fd670a17e2248d034-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/34b70ece5f8d273fd670a17e2248d034-Abstract-Conference.html)

        **Abstract**:

        We introduce an additive Gaussian process (GP) framework accounting for monotonicity constraints and scalable to high dimensions. Our contributions are threefold. First, we show that our framework enables to satisfy the constraints everywhere in the input space. We also show that more general componentwise linear inequality constraints can be handled similarly, such as componentwise convexity. Second, we propose the additive MaxMod algorithm for sequential dimension reduction. By sequentially maximizing a squared-norm criterion, MaxMod identifies the active input dimensions and refines the most important ones. This criterion can be computed explicitly at a linear cost. Finally, we provide open-source codes for our full framework. We demonstrate the performance and scalability of the methodology in several synthetic examples with hundreds of dimensions under monotonicity constraints as well as on a real-world flood application.

        ----

        ## [584] Spherical Channels for Modeling Atomic Interactions

        **Authors**: *Larry Zitnick, Abhishek Das, Adeesh Kolluru, Janice Lan, Muhammed Shuaibi, Anuroop Sriram, Zachary W. Ulissi, Brandon M. Wood*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3501bea1ac61fedbaaff2f88e5fa9447-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3501bea1ac61fedbaaff2f88e5fa9447-Abstract-Conference.html)

        **Abstract**:

        Modeling the energy and forces of atomic systems is a fundamental problem in computational chemistry with the potential to help address many of the worldâ€™s most pressing problems, including those related to energy scarcity and climate change. These calculations are traditionally performed using Density Functional Theory, which is computationally very expensive. Machine learning has the potential to dramatically improve the efficiency of these calculations from days or hours to seconds.We propose the Spherical Channel Network (SCN) to model atomic energies and forces. The SCN is a graph neural network where nodes represent atoms and edges their neighboring atoms. The atom embeddings are a set of spherical functions, called spherical channels, represented using spherical harmonics. We demonstrate, that by rotating the embeddings based on the 3D edge orientation, more information may be utilized while maintaining the rotational equivariance of the messages. While equivariance is a desirable property, we find that by relaxing this constraint in both message passing and aggregation, improved accuracy may be achieved. We demonstrate state-of-the-art results on the large-scale Open Catalyst 2020 dataset in both energy and force prediction for numerous tasks and metrics.

        ----

        ## [585] Handcrafted Backdoors in Deep Neural Networks

        **Authors**: *Sanghyun Hong, Nicholas Carlini, Alexey Kurakin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/3538a22cd3ceb8f009cc62b9e535c29f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/3538a22cd3ceb8f009cc62b9e535c29f-Abstract-Conference.html)

        **Abstract**:

        When machine learning training is outsourced to third parties, $backdoor$ $attacks$ become practical as the third party who trains the model may act maliciously to inject hidden behaviors into the otherwise accurate model. Until now, the mechanism to inject backdoors has been limited to $poisoning$. We argue that a supply-chain attacker has more attack techniques available by introducing a $handcrafted$ attack that directly manipulates a model's weights. This direct modification gives our attacker more degrees of freedom compared to poisoning, and we show it can be used to evade many backdoor detection or removal defenses effectively. Across four datasets and four network architectures our backdoor attacks maintain an attack success rate above 96%. Our results suggest that further research is needed for understanding the complete space of supply-chain backdoor attacks.

        ----

        ## [586] Touch and Go: Learning from Human-Collected Vision and Touch

        **Authors**: *Fengyu Yang, Chenyang Ma, Jiacheng Zhang, Jing Zhu, Wenzhen Yuan, Andrew Owens*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/354892587fe39b17c2b727af02abff4a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/354892587fe39b17c2b727af02abff4a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The ability to associate touch with sight is essential for tasks that require physically interacting with objects in the world. We propose a dataset with paired visual and tactile data called Touch and Go, in which human data collectors probe objects in natural environments using tactile sensors, while simultaneously recording egocentric video. In contrast to previous efforts, which have largely been confined to lab settings or simulated environments, our dataset spans a large number of “in the wild” objects and scenes. We successfully apply our dataset to a variety of multimodal learning tasks: 1) self-supervised visuo-tactile feature learning, 2) tactile-driven image stylization, i.e., making the visual appearance of an object more consistent with a given tactile signal, and 3) predicting future frames of a tactile signal from visuo-tactile inputs.

        ----

        ## [587] SoLar: Sinkhorn Label Refinery for Imbalanced Partial-Label Learning

        **Authors**: *Haobo Wang, Mingxuan Xia, Yixuan Li, Yuren Mao, Lei Feng, Gang Chen, Junbo Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/357a0a771bf65ee07926d6af41b75030-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/357a0a771bf65ee07926d6af41b75030-Abstract-Conference.html)

        **Abstract**:

        Partial-label learning (PLL) is a peculiar weakly-supervised learning task where the training samples are generally associated with a set of candidate labels instead of single ground truth. While a variety of label disambiguation methods have been proposed in this domain, they normally assume a class-balanced scenario that may not hold in many real-world applications. Empirically, we observe degenerated performance of the prior methods when facing the combinatorial challenge from the long-tailed distribution and partial-labeling. In this work, we first identify the major reasons that the prior work failed. We subsequently propose SoLar, a novel Optimal Transport-based framework that allows to refine the disambiguated labels towards matching the marginal class prior distribution. SoLar additionally incorporates a new and systematic mechanism for estimating the long-tailed class prior distribution under the PLL setup. Through extensive experiments, SoLar exhibits substantially superior results on standardized benchmarks compared to the previous state-of-the-art PLL methods. Code and data are available at: https://github.com/hbzju/SoLar.

        ----

        ## [588] Log-Linear-Time Gaussian Processes Using Binary Tree Kernels

        **Authors**: *Michael K. Cohen, Samuel Daulton, Michael A. Osborne*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/359ddb9caccb4c54cc915dceeacf4892-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/359ddb9caccb4c54cc915dceeacf4892-Abstract-Conference.html)

        **Abstract**:

        Gaussian processes (GPs) produce good probabilistic models of functions, but most GP kernels require $O((n+m)n^2)$ time, where $n$ is the number of data points and $m$ the number of predictive locations. We present a new kernel that allows for Gaussian process regression in $O((n+m)\log(n+m))$ time. Our "binary tree" kernel places all data points on the leaves of a binary tree, with the kernel depending only on the depth of the deepest common ancestor. We can store the resulting kernel matrix in $O(n)$ space in $O(n \log n)$ time, as a sum of sparse rank-one matrices, and approximately invert the kernel matrix in $O(n)$ time. Sparse GP methods also offer linear run time, but they predict less well than higher dimensional kernels. On a classic suite of regression tasks, we compare our kernel against Mat\'ern, sparse, and sparse variational kernels. The binary tree GP assigns the highest likelihood to the test data on a plurality of datasets, usually achieves lower mean squared error than the sparse methods, and often ties or beats the Mat\'ern GP. On large datasets, the binary tree GP is fastest, and much faster than a Mat\'ern GP.

        ----

        ## [589] Recovering Private Text in Federated Learning of Language Models

        **Authors**: *Samyak Gupta, Yangsibo Huang, Zexuan Zhong, Tianyu Gao, Kai Li, Danqi Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/35b5c175e139bff5f22a5361270fce87-Abstract-Conference.html)

        **Abstract**:

        Federated learning allows distributed users to collaboratively train a model while keeping each userâ€™s data private. Recently, a growing body of work has demonstrated that an eavesdropping attacker can effectively recover image data from gradients transmitted during federated learning. However, little progress has been made in recovering text data. In this paper, we present a novel attack method FILM for federated learning of language models (LMs). For the first time, we show the feasibility of recovering text from large batch sizes of up to 128 sentences. Unlike image-recovery methods that are optimized to match gradients, we take a distinct approach that first identifies a set of words from gradients and then directly reconstructs sentences based on beam search and a prior-based reordering strategy. We conduct the FILM attack on several large-scale datasets and show that it can successfully reconstruct single sentences with high fidelity for large batch sizes and even multiple sentences if applied iteratively.We evaluate three defense methods: gradient pruning, DPSGD, and a simple approach to freeze word embeddings that we propose.  We show that both gradient pruning and DPSGD lead to a significant drop in utility. However, if we fine-tune a public pre-trained LM on private text without updating word embeddings, it can effectively defend the attack with minimal data utility loss. Together, we hope that our results can encourage the community to rethink the privacy concerns of LM training and its standard practices in the future. Our code is publicly available at https://github.com/Princeton-SysML/FILM .

        ----

        ## [590] INRAS: Implicit Neural Representation for Audio Scenes

        **Authors**: *Kun Su, Mingfei Chen, Eli Shlizerman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/35d5ad984cc0ddd84c6f1c177a2066e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/35d5ad984cc0ddd84c6f1c177a2066e5-Abstract-Conference.html)

        **Abstract**:

        The spatial acoustic information of a scene, i.e., how sounds emitted from a particular location in the scene are perceived in another location, is key for immersive scene modeling. Robust representation of scene's acoustics can be formulated through a continuous field formulation along with impulse responses varied by emitter-listener locations. The impulse responses are then used to render sounds perceived by the listener. While such representation is advantageous, parameterization of impulse responses for generic scenes presents itself as a challenge. Indeed, traditional pre-computation methods have only implemented parameterization at discrete probe points and require large storage, while other existing methods such as geometry-based sound simulations still suffer from inability to simulate all wave-based sound effects. In this work, we introduce a novel neural network for light-weight Implicit Neural Representation for Audio Scenes (INRAS), which can render a high fidelity time-domain impulse responses at any arbitrary emitter-listener positions by learning a continuous implicit function. INRAS disentangles sceneâ€™s geometry features with three modules to generate independent features for the emitter, the geometry of the scene, and the listener respectively. These lead to an efficient reuse of scene-dependent features and support effective multi-condition training for multiple scenes.  Our experimental results show that INRAS outperforms existing approaches for representation and rendering of sounds for varying emitter-listener locations in all aspects, including the impulse response quality, inference speed, and storage requirements.

        ----

        ## [591] Non-Monotonic Latent Alignments for CTC-Based Non-Autoregressive Machine Translation

        **Authors**: *Chenze Shao, Yang Feng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/35f805e65c77652efa731edc10c8e3a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/35f805e65c77652efa731edc10c8e3a6-Abstract-Conference.html)

        **Abstract**:

        Non-autoregressive translation (NAT) models are typically trained with the cross-entropy loss, which forces the model outputs to be aligned verbatim with the target sentence and will highly penalize small shifts in word positions. Latent alignment models relax the explicit alignment by marginalizing out all monotonic latent alignments with the CTC loss. However, they cannot handle non-monotonic alignments, which is non-negligible as there is typically global word reordering in machine translation. In this work, we explore non-monotonic latent alignments for NAT. We extend the alignment space to non-monotonic alignments to allow for the global word reordering and further consider all alignments that overlap with the target sentence. We non-monotonically match the alignments to the target sentence and train the latent alignment model to maximize the F1 score of non-monotonic matching. Extensive experiments on major WMT benchmarks show that our method substantially improves the translation performance of CTC-based models. Our best model achieves 30.06 BLEU on WMT14 En-De with only one-iteration decoding, closing the gap between non-autoregressive and autoregressive models.

        ----

        ## [592] Attention-based Neural Cellular Automata

        **Authors**: *Mattie Tesfaldet, Derek Nowrouzezahrai, Chris Pal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/361e5112d2eca09513bbd266e4b2d2be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/361e5112d2eca09513bbd266e4b2d2be-Abstract-Conference.html)

        **Abstract**:

        Recent extensions of Cellular Automata (CA) have incorporated key ideas from modern deep learning, dramatically extending their capabilities and catalyzing a new family of Neural Cellular Automata (NCA) techniques. Inspired by Transformer-based architectures, our work presents a new class of attention-based NCAs formed using a spatially localized—yet globally organized—self-attention scheme. We introduce an instance of this class named Vision Transformer Cellular Automata (ViTCA). We present quantitative and qualitative results on denoising autoencoding across six benchmark datasets, comparing ViTCA to a U-Net, a U-Net-based CA baseline (UNetCA), and a Vision Transformer (ViT). When comparing across architectures configured to similar parameter complexity, ViTCA architectures yield superior performance across all benchmarks and for nearly every evaluation metric. We present an ablation study on various architectural configurations of ViTCA, an analysis of its effect on cell states, and an investigation on its inductive biases. Finally, we examine its learned representations via linear probes on its converged cell state hidden representations, yielding, on average, superior results when compared to our U-Net, ViT, and UNetCA baselines.

        ----

        ## [593] Learning Deep Input-Output Stable Dynamics

        **Authors**: *Ryosuke Kojima, Yuji Okamoto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/364721b3d5d9da2aaa19adafd7b0f49c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/364721b3d5d9da2aaa19adafd7b0f49c-Abstract-Conference.html)

        **Abstract**:

        Learning stable dynamics from observed time-series data is an essential problem in robotics, physical modeling, and systems biology. Many of these dynamics are represented as an inputs-output system to communicate with the external environment. In this study, we focus on input-output stable systems, exhibiting robustness against unexpected stimuli and noise. We propose a method to learn nonlinear systems guaranteeing the input-output stability. Our proposed method utilizes the differentiable projection onto the space satisfying the Hamilton-Jacobi inequality to realize the input-output stability. The problem of finding this projection can be formulated as a quadratic constraint quadratic programming problem, and we derive the particular solution analytically. Also, we apply our method to a toy bistable model and the task of training a benchmark generated from a glucose-insulin simulator. The results show that the nonlinear system with neural networks by our method achieves the input-output stability, unlike naive neural networks. Our code is available at https://github.com/clinfo/DeepIOStability .

        ----

        ## [594] Grounded Video Situation Recognition

        **Authors**: *Zeeshan Khan, C. V. Jawahar, Makarand Tapaswi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/36474dd150d970be6fcb17035ecc4dce-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/36474dd150d970be6fcb17035ecc4dce-Abstract-Conference.html)

        **Abstract**:

        Dense video understanding requires answering several questions such as who is doing what to whom, with what, how, why, and where. Recently, Video Situation Recognition (VidSitu) is framed as a task for structured prediction of multiple events, their relationships, and actions and various verb-role pairs attached to descriptive entities. This task poses several challenges in identifying, disambiguating, and co-referencing entities across multiple verb-role pairs, but also faces some challenges of evaluation. In this work, we propose the addition of spatio-temporal grounding as an essential component of the structured prediction task in a weakly supervised setting, and present a novel three stage Transformer model, VideoWhisperer, that is empowered to make joint predictions. In stage one, we learn contextualised embeddings for video features in parallel with key objects that appear in the video clips to enable fine-grained spatio-temporal reasoning. The second stage sees verb-role queries attend and pool information from object embeddings, localising answers to questions posed about the action. The final stage generates these answers as captions to describe each verb-role pair present in the video. Our model operates on a group of events (clips) simultaneously and predicts verbs, verb-role pairs, their nouns, and their grounding on-the-fly. When evaluated on a grounding-augmented version of the VidSitu dataset, we observe a large improvement in entity captioning accuracy, as well as the ability to localize verb-roles without grounding annotations at training time.

        ----

        ## [595] Policy Optimization with Advantage Regularization for Long-Term Fairness in Decision Systems

        **Authors**: *Eric Yang Yu, Zhizhen Qin, Min Kyung Lee, Sicun Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/36b76e1f69bbba80d3463f7d6c02bc3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/36b76e1f69bbba80d3463f7d6c02bc3d-Abstract-Conference.html)

        **Abstract**:

        Long-term fairness is an important factor of consideration in designing and deploying learning-based decision systems in high-stake decision-making contexts. Recent work has proposed the use of Markov Decision Processes (MDPs) to formulate decision-making with long-term fairness requirements in dynamically changing environments, and demonstrated major challenges in directly deploying heuristic and rule-based policies that worked well in static environments. We show that policy optimization methods from deep reinforcement learning can be used to find strictly better decision policies that can often achieve both higher overall utility and less violation of the fairness requirements, compared to previously-known strategies. In particular, we propose new methods for imposing fairness requirements in policy optimization by regularizing the advantage evaluation of different actions. Our proposed methods make it easy to impose fairness constraints without reward engineering or sacrificing training efficiency. We perform detailed analyses in three established case studies, including attention allocation in incident monitoring, bank loan approval, and vaccine distribution in population networks.

        ----

        ## [596] Gradient Descent: The Ultimate Optimizer

        **Authors**: *Kartik Chandra, Audrey Xie, Jonathan Ragan-Kelley, Erik Meijer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html)

        **Abstract**:

        Working with any gradient-based machine learning algorithm involves the tedious task of tuning the optimizer's hyperparameters, such as its step size. Recent work has shown how the step size can itself be optimized alongside the model parameters by manually deriving expressions for "hypergradients" ahead of time.We show how to automatically compute hypergradients with a simple and elegant modification to backpropagation. This allows us to easily apply the method to other optimizers and hyperparameters (e.g. momentum coefficients). We can even recursively apply the method to its own hyper-hyperparameters, and so on ad infinitum. As these towers of optimizers grow taller, they become less sensitive to the initial choice of hyperparameters. We present experiments validating this for MLPs, CNNs, and RNNs. Finally, we provide a simple PyTorch implementation of this algorithm (see http://people.csail.mit.edu/kach/gradient-descent-the-ultimate-optimizer).

        ----

        ## [597] DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization

        **Authors**: *Kevin Bello, Bryon Aragam, Pradeep Ravikumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/36e2967f87c3362e37cf988781a887ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/36e2967f87c3362e37cf988781a887ad-Abstract-Conference.html)

        **Abstract**:

        The combinatorial problem of learning directed acyclic graphs (DAGs) from data was recently framed as a purely continuous optimization problem by leveraging a differentiable acyclicity characterization of DAGs based on the trace of a matrix exponential function. Existing acyclicity characterizations are based on the idea that powers of an adjacency matrix contain information about walks and cycles. In this work, we propose a new acyclicity characterization based on the log-determinant (log-det) function, which leverages the nilpotency property of DAGs. To deal with the inherent asymmetries of a DAG, we relate the domain of our log-det characterization to the set of $\textit{M-matrices}$, which is a key difference to the classical log-det function defined over the cone of positive definite matrices.Similar to acyclicity functions previously proposed, our characterization is also exact and differentiable. However, when compared to existing characterizations, our log-det function: (1) Is better at detecting large cycles; (2) Has better-behaved gradients; and (3) Its runtime is in practice about an order of magnitude faster. From the optimization side, we drop the typically used augmented Lagrangian scheme and propose DAGMA ($\textit{Directed Acyclic Graphs via M-matrices for Acyclicity}$), a method that resembles the central path for barrier methods. Each point in the central path of DAGMA is a solution to an unconstrained problem regularized by our log-det function, then we show that at the limit of the central path the solution is guaranteed to be a DAG. Finally, we provide extensive experiments for $\textit{linear}$ and $\textit{nonlinear}$ SEMs and show that our approach can reach large speed-ups and smaller structural Hamming distances against state-of-the-art methods. Code implementing the proposed method is open-source and publicly available at https://github.com/kevinsbello/dagma.

        ----

        ## [598] Forward-Backward Latent State Inference for Hidden Continuous-Time semi-Markov Chains

        **Authors**: *Nicolai Engelmann, Heinz Koeppl*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/36ecc1d1b883afc0e882876cbdd123ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/36ecc1d1b883afc0e882876cbdd123ab-Abstract-Conference.html)

        **Abstract**:

        Hidden semi-Markov Models (HSMM's) - while broadly in use - are restricted to a discrete and uniform time grid. They are thus not well suited to explain often irregularly spaced discrete event data from continuous-time phenomena. We show that non-sampling-based latent state inference used in HSMM's can be generalized to latent Continuous-Time semi-Markov Chains (CTSMC's). We formulate integro-differential forward and backward equations adjusted to the observation likelihood and introduce an exact integral equation for the Bayesian posterior marginals and a scalable Viterbi-type algorithm for posterior path estimates. The presented equations can be efficiently solved using well-known numerical methods. As a practical tool, variable-step HSMM's are introduced. We evaluate our approaches in latent state inference scenarios in comparison to classical HSMM's.

        ----

        ## [599] LobsDICE: Offline Learning from Observation via Stationary Distribution Correction Estimation

        **Authors**: *Geon-Hyeong Kim, Jongmin Lee, Youngsoo Jang, Hongseok Yang, Kee-Eung Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/372593bd318ad8b34b3a8da77e20272b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/372593bd318ad8b34b3a8da77e20272b-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of learning from observation (LfO), in which the agent aims to mimic the expert's behavior from the state-only demonstrations by experts. We additionally assume that the agent cannot interact with the environment but has access to the action-labeled transition data collected by some agents with unknown qualities. This offline setting for LfO is appealing in many real-world scenarios where the ground-truth expert actions are inaccessible and the arbitrary environment interactions are costly or risky. In this paper, we present LobsDICE, an offline LfO algorithm that learns to imitate the expert policy via optimization in the space of stationary distributions. Our algorithm solves a single convex minimization problem, which minimizes the divergence between the two state-transition distributions induced by the expert and the agent policy. Through an extensive set of offline LfO tasks, we show that LobsDICE outperforms strong baseline methods.

        ----

        

[Go to the previous page](NIPS-2022-list02.md)

[Go to the next page](NIPS-2022-list04.md)

[Go to the catalog section](README.md)