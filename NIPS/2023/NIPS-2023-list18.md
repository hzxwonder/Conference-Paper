## [0] Instructing Goal-Conditioned Reinforcement Learning Agents with Temporal Logic Objectives

**Authors**: *Wenjie Qiu, Wensen Mao, He Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b35a69f434b5eb07ed1b1ef16ace52c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b35a69f434b5eb07ed1b1ef16ace52c-Abstract-Conference.html)

**Abstract**:

Goal-conditioned reinforcement learning (RL) is a powerful approach for learning general-purpose skills by reaching diverse goals. However, it has limitations when it comes to task-conditioned policies, where goals are specified by temporally extended instructions written in the Linear Temporal Logic (LTL) formal language. Existing approaches for finding LTL-satisfying policies rely on sampling a large set of LTL instructions during training to adapt to unseen tasks at inference time. However, these approaches do not guarantee generalization to out-of-distribution LTL objectives, which may have increased complexity. In this paper, we propose a novel approach to address this challenge. We show that simple goal-conditioned RL agents can be instructed to follow arbitrary LTL specifications without additional training over the LTL task space. Unlike existing approaches that focus on LTL specifications expressible as regular expressions, our technique is unrestricted and generalizes to $\omega$-regular expressions. Experiment results demonstrate the effectiveness of our approach in adapting goal-conditioned RL agents to satisfy complex temporal logic task specifications zero-shot.

----

## [0] Neural Multi-Objective Combinatorial Optimization with Diversity Enhancement

**Authors**: *Jinbiao Chen, Zizhen Zhang, Zhiguang Cao, Yaoxin Wu, Yining Ma, Te Ye, Jiahai Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b5ae891000049b91b3b62de596b1560-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b5ae891000049b91b3b62de596b1560-Abstract-Conference.html)

**Abstract**:

Most of existing neural methods for multi-objective combinatorial optimization (MOCO) problems solely rely on decomposition, which often leads to repetitive solutions for the respective subproblems, thus a limited Pareto set. Beyond decomposition, we propose a novel neural heuristic with diversity enhancement (NHDE) to produce more Pareto solutions from two perspectives. On the one hand, to hinder duplicated solutions for different subproblems, we propose an indicator-enhanced deep reinforcement learning method to guide the model, and design a heterogeneous graph attention mechanism to capture the relations between the instance graph and the Pareto front graph. On the other hand, to excavate more solutions in the neighborhood of each subproblem, we present a multiple Pareto optima strategy to sample and preserve desirable solutions. Experimental results on classic MOCO problems show that our NHDE is able to generate a Pareto front with higher diversity, thereby achieving superior overall performance. Moreover, our NHDE is generic and can be applied to different neural methods for MOCO.

----

## [0] Multi-Agent First Order Constrained Optimization in Policy Space

**Authors**: *Youpeng Zhao, Yaodong Yang, Zhenbo Lu, Wengang Zhou, Houqiang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b64c47dcb067efd6be5eee854c14835-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b64c47dcb067efd6be5eee854c14835-Abstract-Conference.html)

**Abstract**:

In the realm of multi-agent reinforcement learning (MARL), achieving high performance is crucial for a successful multi-agent system.Meanwhile, the ability to avoid unsafe actions is becoming an urgent and imperative problem to solve for real-life applications. Whereas, it is still challenging to develop a safety-aware method for multi-agent systems in MARL. In this work, we introduce a novel approach called Multi-Agent First Order Constrained Optimization in Policy Space (MAFOCOPS), which effectively addresses the dual objectives of attaining satisfactory performance and enforcing safety constraints. Using data generated from the current policy, MAFOCOPS first finds the optimal update policy by solving a constrained optimization problem in the nonparameterized policy space. Then, the update policy is projected back into the parametric policy space to achieve a feasible policy. Notably, our method is first-order in nature, ensuring the ease of implementation, and exhibits an approximate upper bound on the worst-case constraint violation. Empirical results show that our approach achieves remarkable performance while satisfying safe constraints on several safe MARL benchmarks.

----

## [0] This Looks Like Those: Illuminating Prototypical Concepts Using Multiple Visualizations

**Authors**: *Chiyu Ma, Brandon Zhao, Chaofan Chen, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b76eea0c3683e440c3d362620f578cd-Abstract-Conference.html)

**Abstract**:

We present ProtoConcepts, a method for interpretable image classification combining deep learning and case-based reasoning using prototypical parts. Existing work in prototype-based image classification uses a "this looks like that'' reasoning process, which dissects a test image by finding prototypical parts and combining evidence from these prototypes to make a final classification. However, all of the existing prototypical part-based image classifiers provide only one-to-one comparisons, where a single training image patch serves as a prototype to compare with a part of our test image. With these single-image comparisons, it can often be difficult to identify the underlying concept being compared (e.g., "is it comparing the color or the shape?''). Our proposed method modifies the architecture of prototype-based networks to instead learn prototypical concepts which are visualized using multiple image patches. Having multiple visualizations of the same prototype allows us to more easily identify the concept captured by that prototype (e.g., "the test image and the related training patches are all the same shade of blue''), and allows our model to create richer, more interpretable visual explanations. Our experiments show that our ``this looks like those'' reasoning process can be applied as a modification to a wide range of existing prototypical image classification networks while achieving comparable accuracy on benchmark datasets.

----

## [0] Speculative Decoding with Big Little Decoder

**Authors**: *Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, Kurt Keutzer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html)

**Abstract**:

The recent emergence of Large Language Models based on the Transformer architecture has enabled dramatic advancements in the field of Natural Language Processing. However, these models have long inference latency, which limits their deployment and makes them prohibitively expensive for various real-time applications. The inference latency is further exacerbated by autoregressive generative tasks, as models need to run iteratively to generate tokens sequentially without leveraging token-level parallelization. To address this, we propose Big Little Decoder (BiLD), a framework that can improve inference efficiency and latency for a wide range of text generation applications. The BiLD framework contains two models with different sizes that collaboratively generate text. The small model runs autoregressively to generate text with a low inference cost, and the large model is only invoked occasionally to refine the small modelâ€™s inaccurate predictions in a non-autoregressive manner. To coordinate the small and large models, BiLD introduces two simple yet effective policies: (1) the fallback policy that determines when to hand control over to the large model; and (2) the rollback policy that determines when the large model needs to correct the small model's inaccurate predictions. To evaluate our framework across different tasks and models, we apply BiLD to various text generation scenarios encompassing machine translation on IWSLT 2017 De-En and WMT 2014 De-En, and summarization on XSUM and CNN/DailyMail. On an NVIDIA T4 GPU, our framework achieves a speedup of up to 2.12x speedup with minimal generation quality degradation. Furthermore, our framework is fully plug-and-play and can be applied without any modifications in the training process or model architecture. Our code is open-sourced.

----

## [0] Intrinsic Dimension Estimation for Robust Detection of AI-Generated Texts

**Authors**: *Eduard Tulchinskii, Kristian Kuznetsov, Laida Kushnareva, Daniil Cherniavskii, Sergey I. Nikolenko, Evgeny Burnaev, Serguei Barannikov, Irina Piontkovskaya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7baa48bc166aa2013d78cbdc15010530-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7baa48bc166aa2013d78cbdc15010530-Abstract-Conference.html)

**Abstract**:

Rapidly increasing quality of AI-generated content makes it difficult to distinguish between human and AI-generated texts, which may lead to undesirable consequences for society. Therefore, it becomes increasingly important to study the properties of human texts that are invariant over text domains and various proficiency of human writers, can be easily calculated for any language, and can robustly separate natural and AI-generated texts regardless of the generation model and sampling method. In this work, we propose such an invariant of human texts, namely the intrinsic dimensionality of the manifold underlying the set of embeddings of a given text sample. We show that the average intrinsic dimensionality of fluent texts in natural language is hovering around the value $9$ for several alphabet-based languages and around $7$ for Chinese, while the average intrinsic dimensionality of AI-generated texts for each language is $\approx 1.5$ lower, with a clear statistical separation between human-generated and AI-generated distributions. This property allows us to build a score-based artificial text detector. The proposed detector's accuracy is stable over text domains, generator models, and human writer proficiency levels, outperforming SOTA detectors in model-agnostic and cross-domain scenarios by a significant margin.

----

## [0] Replicable Clustering

**Authors**: *Hossein Esfandiari, Amin Karbasi, Vahab Mirrokni, Grigoris Velegkas, Felix Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bc3fe234454107149fa9d44faacaa64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bc3fe234454107149fa9d44faacaa64-Abstract-Conference.html)

**Abstract**:

We design replicable algorithms in the context of statistical clustering under the recently introduced notion of replicability from Impagliazzo et al. [2022]. According to this definition, a clustering algorithm is replicable if, with high probability, its output induces the exact same partition of the sample space after two executions on different inputs drawn from the same distribution, when its internal randomness is shared across the executions. We propose such algorithms for the statistical $k$-medians, statistical $k$-means, and statistical $k$-centers problems by utilizing approximation routines for their combinatorial counterparts in a black-box manner. In particular, we demonstrate a replicable $O(1)$-approximation algorithm for statistical Euclidean $k$-medians ($k$-means) with $\operatorname{poly}(d)$ sample complexity. We also describe an $O(1)$-approximation algorithm with an additional $O(1)$-additive error for statistical Euclidean $k$-centers, albeit with $\exp(d)$ sample complexity. In addition, we provide experiments on synthetic distributions in 2D using the $k$-means++ implementation from sklearn as a black-box that validate our theoretical results.

----

## [0] Counterfactual Memorization in Neural Language Models

**Authors**: *Chiyuan Zhang, Daphne Ippolito, Katherine Lee, Matthew Jagielski, Florian Tramèr, Nicholas Carlini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bc4f74e35bcfe8cfe43b0a860786d6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bc4f74e35bcfe8cfe43b0a860786d6a-Abstract-Conference.html)

**Abstract**:

Modern neural language models that are widely used in various NLP tasks risk memorizing sensitive information from their training data.Understanding this memorization is important in real world applications and also from a learning-theoretical perspective. An open question in previous studies of language model memorization is how to filter out ``common'' memorization. In fact, most memorization criteria strongly correlate with the number of occurrences in the training set, capturing memorized familiar phrases, public knowledge, templated texts, or other repeated data.We formulate a notion of counterfactual memorization which characterizes how a model's predictions change if a particular document is omitted during training.We identify and study counterfactually-memorized training examples in standard text datasets.We estimate the influence of each memorized training example on the validation set and on generated texts, showing how this can provide direct evidence of the source of memorization at test time.

----

## [0] Learning Generalizable Agents via Saliency-guided Features Decorrelation

**Authors**: *Sili Huang, Yanchao Sun, Jifeng Hu, Siyuan Guo, Hechang Chen, Yi Chang, Lichao Sun, Bo Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bd4a7d0e6773072c2e3c77b11d93065-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bd4a7d0e6773072c2e3c77b11d93065-Abstract-Conference.html)

**Abstract**:

In visual-based Reinforcement Learning (RL), agents often struggle to generalize well to environmental variations in the state space that were not observed during training. The variations can arise in both task-irrelevant features, such as background noise, and task-relevant features, such as robot configurations, that are related to the optimal decisions. To achieve generalization in both situations, agents are required to accurately understand the impact of changed features on the decisions, i.e., establishing the true associations between changed features and decisions in the policy model. However, due to the inherent correlations among features in the state space, the associations between features and decisions become entangled, making it difficult for the policy to distinguish them. To this end, we propose Saliency-Guided Features Decorrelation (SGFD) to eliminate these correlations through sample reweighting. Concretely, SGFD consists of two core techniques: Random Fourier Functions (RFF) and the saliency map. RFF is utilized to estimate the complex non-linear correlations in high-dimensional images, while the saliency map is designed to identify the changed features. Under the guidance of the saliency map, SGFD employs sample reweighting to minimize the estimated correlations related to changed features, thereby achieving decorrelation in visual RL tasks. Our experimental results demonstrate that SGFD can generalize well on a wide range of test environments and significantly outperforms state-of-the-art methods in handling both task-irrelevant variations and task-relevant variations.

----

## [0] You Only Condense Once: Two Rules for Pruning Condensed Datasets

**Authors**: *Yang He, Lingao Xiao, Joey Tianyi Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bdd36a198a8408f444834039b09f518-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bdd36a198a8408f444834039b09f518-Abstract-Conference.html)

**Abstract**:

Dataset condensation is a crucial tool for enhancing training efficiency by reducing the size of the training dataset, particularly in on-device scenarios. However, these scenarios have two significant challenges: 1) the varying computational resources available on the devices require a dataset size different from the pre-defined condensed dataset, and 2) the limited computational resources often preclude the possibility of conducting additional condensation processes. We introduce You Only Condense Once (YOCO) to overcome these limitations. On top of one condensed dataset, YOCO produces smaller condensed datasets with two embarrassingly simple dataset pruning rules: Low LBPE Score and Balanced Construction. YOCO offers two key advantages: 1) it can flexibly resize the dataset to fit varying computational constraints, and 2) it eliminates the need for extra condensation processes, which can be computationally prohibitive. Experiments validate our findings on networks including ConvNet, ResNet and DenseNet, and datasets including CIFAR-10, CIFAR-100 and ImageNet. For example, our YOCO surpassed various dataset condensation and dataset pruning methods on CIFAR-10 with ten Images Per Class (IPC), achieving 6.98-8.89% and 6.31-23.92% accuracy gains, respectively. The code is available at: https://github.com/he-y/you-only-condense-once.

----

## [0] Provably Efficient Offline Reinforcement Learning in Regular Decision Processes

**Authors**: *Roberto Cipollone, Anders Jonsson, Alessandro Ronca, Mohammad Sadegh Talebi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7bf3e93543a612b75b6373178ba1faa4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7bf3e93543a612b75b6373178ba1faa4-Abstract-Conference.html)

**Abstract**:

This paper deals with offline (or batch) Reinforcement Learning (RL) in episodic Regular Decision Processes (RDPs). RDPs are the subclass of Non-Markov Decision Processes where the dependency on the history of past events can be captured by a finite-state automaton. We consider a setting where the automaton that underlies the RDP is unknown, and a learner strives to learn a near-optimal policy using pre-collected data, in the form of non-Markov sequences of observations, without further exploration. We present RegORL, an algorithm that suitably combines automata learning techniques and state-of-the-art algorithms for offline RL in MDPs. RegORL has a modular design allowing one to use any off-the-shelf offline RL algorithm in MDPs. We report a non-asymptotic high-probability sample complexity bound for RegORL to yield an $\varepsilon$-optimal policy, which makes appear a notion of concentrability relevant for RDPs. Furthermore, we present a sample complexity lower bound for offline RL in RDPs. To our best knowledge, this is the first work presenting a provably efficient algorithm for offline learning in RDPs.

----

## [0] CP-SLAM: Collaborative Neural Point-based SLAM System

**Authors**: *Jiarui Hu, Mao Mao, Hujun Bao, Guofeng Zhang, Zhaopeng Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c10e259c7e56fa218ee03d9ae7d728e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c10e259c7e56fa218ee03d9ae7d728e-Abstract-Conference.html)

**Abstract**:

This paper presents a collaborative implicit neural simultaneous localization and mapping (SLAM) system with RGB-D image sequences, which consists of complete front-end and back-end modules including odometry, loop detection, sub-map fusion, and global refinement. In order to enable all these modules in a unified framework, we propose a novel neural point based 3D scene representation in which each point maintains a learnable neural feature for scene encoding and is associated with a certain keyframe. Moreover, a distributed-to-centralized learning strategy is proposed for the collaborative implicit SLAM to improve consistency and cooperation. A novel global optimization framework is also proposed to improve the system accuracy like traditional bundle adjustment. Experiments on various datasets demonstrate the superiority of the proposed method in both camera tracking and mapping.

----

## [0] The Surprising Effectiveness of Diffusion Models for Optical Flow and Monocular Depth Estimation

**Authors**: *Saurabh Saxena, Charles Herrmann, Junhwa Hur, Abhishek Kar, Mohammad Norouzi, Deqing Sun, David J. Fleet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c119415672ae2186e17d492e1d5da2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c119415672ae2186e17d492e1d5da2f-Abstract-Conference.html)

**Abstract**:

Denoising diffusion probabilistic models have transformed image generation with their impressive fidelity and diversity.We show that they also excel in estimating optical flow and monocular depth, surprisingly without task-specific architectures and loss functions that are predominant for these tasks. Compared to the point estimates of conventional regression-based methods, diffusion models also enable Monte Carlo inference, e.g., capturing uncertainty and ambiguity in flow and depth.With self-supervised pre-training, the combined use of synthetic and real data for supervised training, and technical innovations (infilling and step-unrolled denoising diffusion training) to handle noisy-incomplete training data, one can train state-of-the-art diffusion models for depth and optical flow estimation, with additional zero-shot coarse-to-fine refinement for high resolution estimates. Extensive experiments focus on quantitative performance against benchmarks, ablations, and the model's ability to capture uncertainty and multimodality, and impute missing values. Our model obtains a state-of-the-art relative depth error of 0.074 on the indoor NYU benchmark and an Fl-all score of 3.26\% on the KITTI  optical flow benchmark, about 25\% better than the best published method.

----

## [0] Efficient Testable Learning of Halfspaces with Adversarial Label Noise

**Authors**: *Ilias Diakonikolas, Daniel Kane, Vasilis Kontonis, Sihan Liu, Nikos Zarifis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c319b62e2257b34cb0e1040ced2e007-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c319b62e2257b34cb0e1040ced2e007-Abstract-Conference.html)

**Abstract**:

We give the first polynomial-time algorithm for the testable learning of halfspaces in the presence of adversarial label noise under the Gaussian distribution. In the recently introduced testable learning model, one is required to produce a tester-learner such that if the data passes the tester, then one can trust the output of the robust learner on the data. Our tester-learner runs in time $\text{poly}(d/\epsilon)$ and outputs a halfspace with misclassification error $O(\text{opt})+\epsilon$, where $\text{opt}$ is the 0-1 error of the best fitting halfspace. At a technical level, our algorithm employs an iterative soft localization technique enhanced with appropriate testers to ensure that the data distribution is sufficiently similar to a Gaussian. Finally, our algorithm can be readily adapted to yield an efficient and testable active learner requiring only $d ~ \text{polylog}(1/\epsilon)$ labeled examples.

----

## [0] Achieving O(ε-15) Complexity in Hessian/Jacobian-free Stochastic Bilevel Optimization

**Authors**: *Yifan Yang, Peiyao Xiao, Kaiyi Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c3a8d20ceadb7c519e9ac1bb77a15ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c3a8d20ceadb7c519e9ac1bb77a15ff-Abstract-Conference.html)

**Abstract**:

In this paper, we revisit the bilevel optimization problem, in which the upper-level objective function is generally nonconvex and the lower-level objective function is strongly convex. Although this type of problem has been studied extensively, it still remains an open question how to achieve an $\mathcal{O}(\epsilon^{-1.5})$ sample complexity in Hessian/Jacobian-free stochastic bilevel optimization without any second-order derivative computation. To fill this gap, we propose a novel Hessian/Jacobian-free bilevel optimizer named FdeHBO, which features a simple fully single-loop structure, a projection-aided finite-difference Hessian/Jacobian-vector approximation, and momentum-based updates. Theoretically, we show that FdeHBO requires $\mathcal{O}(\epsilon^{-1.5})$ iterations (each using $\mathcal{O}(1)$ samples and only first-order gradient information) to find an $\epsilon$-accurate stationary point. As far as we know, this is the first Hessian/Jacobian-free method with an $\mathcal{O}(\epsilon^{-1.5})$ sample complexity for nonconvex-strongly-convex stochastic bilevel optimization.

----

## [0] Robust and Actively Secure Serverless Collaborative Learning

**Authors**: *Nicholas Franzese, Adam Dziedzic, Christopher A. Choquette-Choo, Mark R. Thomas, Muhammad Ahmad Kaleem, Stephan Rabanser, Congyu Fang, Somesh Jha, Nicolas Papernot, Xiao Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c5a4b7a31dffef8ce296deedb6214a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c5a4b7a31dffef8ce296deedb6214a9-Abstract-Conference.html)

**Abstract**:

Collaborative machine learning (ML) is widely used to enable institutions to learn better models from distributed data. While collaborative approaches to learning intuitively protect user data, they remain vulnerable to either the server, the clients, or both, deviating from the protocol. Indeed, because the protocol is asymmetric, a malicious server can abuse its power to reconstruct client data points. Conversely, malicious clients can corrupt learning with malicious updates. Thus, both clients and servers require a guarantee when the other cannot be trusted to fully cooperate. In this work, we propose a peer-to-peer (P2P) learning scheme that is secure against malicious servers and robust to malicious clients. Our core contribution is a generic framework that transforms any (compatible) algorithm for robust aggregation of model updates to the setting where servers and clients can act maliciously. Finally, we demonstrate the computational efficiency of our approach even with 1-million parameter models trained by 100s of peers on standard datasets.

----

## [0] Birder: Communication-Efficient 1-bit Adaptive Optimizer for Practical Distributed DNN Training

**Authors**: *Hanyang Peng, Shuang Qin, Yue Yu, Jin Wang, Hui Wang, Ge Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c72fcd7b6bffc3864c7152ab5a2dd83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c72fcd7b6bffc3864c7152ab5a2dd83-Abstract-Conference.html)

**Abstract**:

Various gradient compression algorithms have been proposed to alleviate the communication bottleneck in distributed learning, and they have demonstrated effectiveness in terms of  high compression ratios and theoretical low communication complexity. However, when it comes to practically training modern deep neural networks (DNNs),  these algorithms have yet to match the inference performance of uncompressed SGD-momentum  (SGDM) and adaptive optimizers (e.g.,Adam). More importantly,  recent studies suggest that these algorithms actually offer no speed advantages over SGDM/Adam when used with common  distributed DNN training frameworks ( e.g., DistributedDataParallel (DDP)) in the typical settings, due to heavy compression/decompression computation or incompatibility with the efficient All-Reduce or the requirement of uncompressed warmup at the early stage. For these reasons, we propose a novel 1-bit adaptive optimizer, dubbed *Bi*nary *r*andomization a*d*aptive optimiz*er* (**Birder**). The quantization of Birder can be easily and lightly computed, and it does not require warmup with its uncompressed version in the beginning. Also, we devise Hierarchical-1-bit-All-Reduce to further lower the communication volume. We theoretically prove that it promises the same convergence rate as the Adam. Extensive experiments, conducted on 8 to 64 GPUs (1 to 8 nodes) using DDP, demonstrate that Birder achieves comparable inference performance to uncompressed SGDM/Adam, with up to ${2.5 \times}$ speedup for training ResNet-50 and ${6.3\times}$ speedup for training BERT-Base. Code is publicly available at https://openi.pcl.ac.cn/c2net_optim/Birder.

----

## [0] MIMONets: Multiple-Input-Multiple-Output Neural Networks Exploiting Computation in Superposition

**Authors**: *Nicolas Menet, Michael Hersche, Geethan Karunaratne, Luca Benini, Abu Sebastian, Abbas Rahimi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7c7a12559be4501f70d221352514397c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7c7a12559be4501f70d221352514397c-Abstract-Conference.html)

**Abstract**:

With the advent of deep learning, progressively larger neural networks have been designed to solve complex tasks. We take advantage of these capacity-rich models to lower the cost of inference by exploiting computation in superposition. To reduce the computational burden per input, we propose Multiple-Input-Multiple-Output Neural Networks (MIMONets) capable of handling many inputs at once. MIMONets augment various deep neural network architectures with variable binding mechanisms to represent an arbitrary number of inputs in a compositional data structure via fixed-width distributed representations. Accordingly, MIMONets adapt nonlinear neural transformations to process the data structure holistically, leading to a speedup nearly proportional to the number of superposed input items in the data structure. After processing in superposition, an unbinding mechanism recovers each transformed input of interest. MIMONets also provide a dynamic trade-off between accuracy and throughput by an instantaneous on-demand switching between a set of accuracy-throughput operating points, yet within a single set of fixed parameters. We apply the concept of MIMONets to both CNN and Transformer architectures resulting in MIMOConv and MIMOFormer, respectively. Empirical evaluations show that MIMOConv achieves $\approx 2$–$4\times$ speedup at an accuracy delta within [+0.68, -3.18]% compared to WideResNet CNNs on CIFAR10 and CIFAR100.  Similarly, MIMOFormer can handle $2$–$4$ inputs at once while maintaining a high average accuracy within a [-1.07, -3.43]% delta on the long range arena benchmark. Finally, we provide mathematical bounds on the interference between superposition channels in MIMOFormer. Our code is available at https://github.com/IBM/multiple-input-multiple-output-nets.

----

## [0] C-Disentanglement: Discovering Causally-Independent Generative Factors under an Inductive Bias of Confounder

**Authors**: *Xiaoyu Liu, Jiaxin Yuan, Bang An, Yuancheng Xu, Yifan Yang, Furong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ca55c8276acf1f0aa996cd3622d1df4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ca55c8276acf1f0aa996cd3622d1df4-Abstract-Conference.html)

**Abstract**:

Representation learning assumes that real-world data is generated by a few semantically meaningful generative factors (i.e., sources of variation) and aims to discover them in the latent space. These factors are expected to be causally disentangled, meaning that distinct factors are encoded into separate latent variables, and changes in one factor will not affect the values of the others. Compared to statistical independence, causal disentanglement allows more controllable data generation, improved robustness, and better generalization. However, most existing work assumes unconfoundedness in the discovery process, that there are no common causes to the generative factors and thus obtain only statistical independence. In this paper, we recognize the importance of modeling confounders in discovering causal generative factors. Unfortunately, such factors are not identifiable without proper inductive bias. We fill the gap by introducing a framework entitled Confounded-Disentanglement (C-Disentanglement), the first framework that explicitly introduces the inductive bias of confounder via labels from domain expertise. In addition, we accordingly propose an approach to sufficiently identify the causally-disentangled factors under any inductive bias of the confounder.  We conduct extensive experiments on both synthetic and real-world datasets. Our method demonstrates competitive results compared to various SOTA baselines in obtaining causally disentangled features and downstream tasks under domain shifts.

----

## [0] Representation Learning via Consistent Assignment of Views over Random Partitions

**Authors**: *Thalles Santos Silva, Adín Ramírez Rivera*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7caf9d251b546bc78078b35b4a6f3b7e-Abstract-Conference.html)

**Abstract**:

We present Consistent Assignment of Views over Random Partitions (CARP), a self-supervised clustering method for representation learning of visual features. CARP learns prototypes in an end-to-end online fashion using gradient descent without additional non-differentiable modules to solve the cluster assignment problem. CARP optimizes a new pretext task based on random partitions of prototypes that regularizes the model and enforces consistency between views' assignments.  Additionally, our method improves training stability and prevents collapsed solutions in joint-embedding training. Through an extensive evaluation, we demonstrate that CARP's representations are suitable for learning downstream tasks. We evaluate CARP's representations capabilities in 17 datasets across many standard protocols, including linear evaluation, few-shot classification, $k$-NN, $k$-means, image retrieval, and copy detection. We compare CARP performance to 11 existing self-supervised methods. We extensively ablate our method and demonstrate that our proposed random partition pretext task improves the quality of the learned representations by devising multiple random classification tasks.In transfer learning tasks, CARP achieves the best performance on average against many SSL methods trained for a longer time.

----

## [0] Federated Multi-Objective Learning

**Authors**: *Haibo Yang, Zhuqing Liu, Jia Liu, Chaosheng Dong, Michinari Momma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cb2c2a8d35576c00078b6591ec26a7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cb2c2a8d35576c00078b6591ec26a7d-Abstract-Conference.html)

**Abstract**:

In recent years, multi-objective optimization (MOO) emerges as a foundational problem underpinning many multi-agent multi-task learning applications. However, existing algorithms in MOO literature remain limited to centralized learning settings, which do not satisfy the distributed nature and data privacy needs of such multi-agent multi-task learning applications. This motivates us to propose a new federated multi-objective learning (FMOL) framework with multiple clients distributively and collaboratively solving an MOO problem while keeping their training data private. Notably, our FMOL framework allows a different set of objective functions across different clients to support a wide range of applications, which advances and generalizes the MOO formulation to the federated learning paradigm for the first time. For this FMOL framework, we propose two new federated multi-objective optimization (FMOO) algorithms called federated multi-gradient descent averaging (FMGDA) and federated stochastic multi-gradient descent averaging (FSMGDA). Both algorithms allow local updates to significantly reduce communication costs, while achieving the {\em same} convergence rates as those of their algorithmic counterparts in the single-objective federated learning. Our extensive experiments also corroborate the efficacy of our proposed FMOO algorithms.

----

## [0] MARBLE: Music Audio Representation Benchmark for Universal Evaluation

**Authors**: *Ruibin Yuan, Yinghao Ma, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, Le Zhuo, Yiqi Liu, Jiawen Huang, Zeyue Tian, Binyue Deng, Ningzhi Wang, Chenghua Lin, Emmanouil Benetos, Anton Ragni, Norbert Gyenge, Roger B. Dannenberg, Wenhu Chen, Gus Xia, Wei Xue, Si Liu, Shi Wang, Ruibo Liu, Yike Guo, Jie Fu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In the era of extensive intersection between art and Artificial Intelligence (AI), such as image generation and fiction co-creation, AI for music remains relatively nascent, particularly in music understanding. This is evident in the limited work on deep music representations, the scarcity of large-scale datasets, and the absence of a universal and community-driven benchmark. To address this issue, we introduce the Music Audio Representation Benchmark for universaL Evaluation, termed MARBLE. It aims to provide a benchmark for various Music Information Retrieval (MIR) tasks by defining a comprehensive taxonomy with four hierarchy levels, including acoustic, performance, score, and high-level description. We then establish a unified protocol based on 18 tasks on 12 public-available datasets, providing a fair and standard assessment of representations of all open-sourced pre-trained models developed on music recordings as baselines. Besides, MARBLE offers an easy-to-use, extendable, and reproducible suite for the community, with a clear statement on copyright issues on datasets. Results suggest recently proposed large-scale pre-trained musical language models perform the best in most tasks, with room for further improvement. The leaderboard and toolkit repository are published to promote future music AI research.

----

## [0] Language Models can Solve Computer Tasks

**Authors**: *Geunwoo Kim, Pierre Baldi, Stephen McAleer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cc1005ec73cfbaac9fa21192b622507-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cc1005ec73cfbaac9fa21192b622507-Abstract-Conference.html)

**Abstract**:

Agents capable of carrying out general tasks on a computer can improve efficiency and productivity by automating repetitive tasks and assisting in complex problem-solving. Ideally, such agents should be able to solve new computer tasks presented to them through natural language commands. However, previous approaches to this problem require large amounts of expert demonstrations and task-specific reward functions, both of which are impractical for new tasks. In this work, we show that a pre-trained large language model (LLM) agent can execute computer tasks guided by natural language using a simple prompting scheme where the agent \textbf{R}ecursively \textbf{C}riticizes and \textbf{I}mproves its output (RCI). The RCI approach significantly outperforms existing LLM methods for automating computer tasks and surpasses supervised learning (SL) and reinforcement learning (RL) approaches on the MiniWoB++ benchmark. We compare multiple LLMs and find that RCI with the InstructGPT-3+RLHF LLM is state-of-the-art on MiniWoB++, using only a handful of demonstrations per task rather than tens of thousands, and without a task-specific reward function. Furthermore, we demonstrate RCI prompting's effectiveness in enhancing LLMs' reasoning abilities on a suite of natural language reasoning tasks, outperforming chain of thought (CoT) prompting with external feedback. We find that RCI combined with CoT performs better than either separately. Our code can be found here: https://github.com/posgnu/rci-agent.

----

## [0] An NLP Benchmark Dataset for Assessing Corporate Climate Policy Engagement

**Authors**: *Gaku Morio, Christopher D. Manning*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ccaa4f9a89cce6619093226f26b84e6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ccaa4f9a89cce6619093226f26b84e6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

As societal awareness of climate change grows, corporate climate policy engagements are attracting attention.We propose a dataset to estimate corporate climate policy engagement from various PDF-formatted documents.Our dataset comes from LobbyMap (a platform operated by global think tank InfluenceMap) that provides engagement categories and stances on the documents.To convert the LobbyMap data into the structured dataset, we developed a pipeline using text extraction and OCR.Our contributions are: (i) Building an NLP dataset including 10K documents on corporate climate policy engagement. (ii) Analyzing the properties and challenges of the dataset. (iii) Providing experiments for the dataset using pre-trained language models.The results show that while Longformer outperforms baselines and other pre-trained models, there is still room for significant improvement.We hope our work begins to bridge research on NLP and climate change.

----

## [0] Robustness Guarantees for Adversarially Trained Neural Networks

**Authors**: *Poorya Mianjy, Raman Arora*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cde9bd7774c9f5056cb6e5474fbadff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cde9bd7774c9f5056cb6e5474fbadff-Abstract-Conference.html)

**Abstract**:

We study robust adversarial training of two-layer neural networks as a bi-level optimization problem. In particular, for the inner loop that implements the adversarial attack during training using projected gradient descent (PGD), we propose maximizing a \emph{lower bound} on the $0/1$-loss by reflecting a surrogate loss about the origin. This allows us to give a convergence guarantee for the inner-loop PGD attack. Furthermore, assuming the data is linearly separable, we provide precise iteration complexity results for end-to-end adversarial training, which holds for any width and initialization. We provide empirical evidence to support our theoretical results.

----

## [0] Pre-training Contextualized World Models with In-the-wild Videos for Reinforcement Learning

**Authors**: *Jialong Wu, Haoyu Ma, Chaoyi Deng, Mingsheng Long*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ce1cbededb4b0d6202847ac1b484ee8-Abstract-Conference.html)

**Abstract**:

Unsupervised pre-training methods utilizing large and diverse datasets have achieved tremendous success across a range of domains. Recent work has investigated such unsupervised pre-training methods for model-based reinforcement learning (MBRL) but is limited to domain-specific or simulated data. In this paper, we study the problem of pre-training world models with abundant in-the-wild videos for efficient learning of downstream visual control tasks. However, in-the-wild videos are complicated with various contextual factors, such as intricate backgrounds and textured appearance, which precludes a world model from extracting shared world knowledge to generalize better. To tackle this issue, we introduce Contextualized World Models (ContextWM) that explicitly separate context and dynamics modeling to overcome the complexity and diversity of in-the-wild videos and facilitate knowledge transfer between distinct scenes. Specifically, a contextualized extension of the latent dynamics model is elaborately realized by incorporating a context encoder to retain contextual information and empower the image decoder, which encourages the latent dynamics model to concentrate on essential temporal variations. Our experiments show that in-the-wild video pre-training equipped with ContextWM can significantly improve the sample efficiency of MBRL in various domains, including robotic manipulation, locomotion, and autonomous driving. Code is available at this repository: https://github.com/thuml/ContextWM.

----

## [0] Strategyproof Voting under Correlated Beliefs

**Authors**: *Daniel Halpern, Rachel Li, Ariel D. Procaccia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7cefded8659ccc899196860af674b596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7cefded8659ccc899196860af674b596-Abstract-Conference.html)

**Abstract**:

In voting theory, when voters have ranked preferences over candidates, the celebrated Gibbard-Satterthwaite Theorem essentially rules out the existence of reasonable strategyproof methods for picking a winner. What if we weaken strategyproofness to only hold for Bayesian voters with beliefs over others' preferences? When voters believe other participants' rankings are drawn independently from a fixed distribution, the impossibility persists. However, it is quite reasonable for a voter to believe that other votes are correlated, either to each other or to their own ranking. We consider such beliefs induced by classic probabilistic models in social choice such as the Mallows, Placket-Luce, and Thurstone-Mosteller models. We single out the plurality rule (choosing the candidate ranked first most often) as a particularly promising choice as it is strategyproof for a large class of beliefs containing the specific ones we introduce. Further, we show that plurality is unique among positional scoring rules in having this property: no other scoring rule is strategyproof for beliefs induced by the Mallows model when there are a sufficient number of voters. Finally, we give examples of prominent non-scoring voting rules failing to be strategyproof on beliefs in this class, further bolstering the case for plurality.

----

## [0] PCF-GAN: generating sequential data via the characteristic function of measures on the path space

**Authors**: *Hang Lou, Siran Li, Hao Ni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d0e867582cdc156fd280d5a6aa1be08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d0e867582cdc156fd280d5a6aa1be08-Abstract-Conference.html)

**Abstract**:

Generating high-fidelity time series data using generative adversarial networks (GANs) remains a challenging task, as it is difficult to capture the temporal dependence of joint probability distributions induced by time-series data. Towards this goal, a key step is the development of an effective discriminator to distinguish between time series distributions. We propose the so-called PCF-GAN, a novel GAN that incorporates the path characteristic function (PCF) as the principled representation of time series distribution into the discriminator to enhance its generative performance.  On the one hand, we establish theoretical foundations of the PCF distance by proving its characteristicity, boundedness, differentiability with respect to generator parameters, and weak continuity, which ensure the stability and feasibility of training the PCF-GAN. On the other hand, we design efficient initialisation and optimisation schemes for PCFs to strengthen the discriminative power and accelerate training efficiency. To further boost the capabilities of complex time series generation, we integrate the auto-encoder structure via sequential embedding into the PCF-GAN, which provides additional reconstruction functionality. Extensive numerical experiments on various datasets demonstrate the consistently superior performance of PCF-GAN over state-of-the-art baselines, in both generation and reconstruction quality.

----

## [0] A Rigorous Link between Deep Ensembles and (Variational) Bayesian Methods

**Authors**: *Veit David Wild, Sahra Ghalebikesabi, Dino Sejdinovic, Jeremias Knoblauch*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d25b1db211d99d5750ec45d65fd6e4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d25b1db211d99d5750ec45d65fd6e4e-Abstract-Conference.html)

**Abstract**:

We establish the first mathematically rigorous link between Bayesian, variational Bayesian, and ensemble methods. A key step towards this it to reformulate the non-convex optimisation problem typically encountered in deep learning as a convex optimisation in the space of probability measures. On a technical level, our contribution amounts to studying generalised variational inference through the lense of Wasserstein gradient flows. The result is a unified theory of various seemingly disconnected approaches that are commonly used for uncertainty quantification in deep learning---including deep ensembles and (variational) Bayesian methods. This offers a fresh perspective on the reasons behind the success of deep ensembles over procedures based on parameterised variational inference, and allows the derivation of new ensembling schemes with convergence guarantees. We showcase this by proposing a family of interacting deep ensembles with direct parallels to the interactions of particle systems in thermodynamics, and use our theory to prove the convergence of these algorithms to a well-defined global minimiser on the space of probability measures.

----

## [0] Markovian Sliced Wasserstein Distances: Beyond Independent Projections

**Authors**: *Khai Nguyen, Tongzheng Ren, Nhat Ho*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d2b770c3ccd35b41c9453ef6f8765a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d2b770c3ccd35b41c9453ef6f8765a3-Abstract-Conference.html)

**Abstract**:

Sliced Wasserstein (SW) distance suffers from redundant projections due to independent uniform random projecting directions. To partially overcome the issue, max K sliced Wasserstein  (Max-K-SW) distance ($K\geq 1$), seeks the best discriminative orthogonal projecting directions. Despite being able to reduce the number of projections, the metricity of the Max-K-SW cannot be guaranteed in practice due to the non-optimality of the optimization. Moreover,  the orthogonality constraint is also computationally expensive and might not be effective. To address the problem, we introduce a new family of SW distances, named Markovian sliced Wasserstein (MSW) distance, which imposes a first-order Markov structure on projecting directions. We discuss various members of the MSW by specifying the Markov structure including the prior distribution, the transition distribution, and the burning and thinning technique. Moreover, we investigate the theoretical properties of  MSW including topological properties (metricity, weak convergence, and connection to other distances), statistical properties (sample complexity, and Monte Carlo estimation error), and computational properties (computational complexity and memory complexity). Finally, we compare MSW distances with previous SW variants in various applications such as gradient flows, color transfer, and deep generative modeling to demonstrate the favorable performance of the MSW.

----

## [0] Generative Modelling of Stochastic Actions with Arbitrary Constraints in Reinforcement Learning

**Authors**: *Changyu Chen, Ramesha Karunasena, Thanh Hong Nguyen, Arunesh Sinha, Pradeep Varakantham*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d4c0094ae32530494c71468558ab5b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d4c0094ae32530494c71468558ab5b1-Abstract-Conference.html)

**Abstract**:

Many problems in Reinforcement Learning (RL) seek an optimal policy with large discrete multidimensional yet unordered action spaces; these include problems in randomized allocation of resources such as placements of multiple security resources and emergency response units, etc. A challenge in this setting is that the underlying action space is categorical (discrete and unordered) and large, for which existing RL methods do not perform well. Moreover, these problems require validity of the realized action (allocation); this validity constraint is often difficult to express compactly in a closed mathematical form. The allocation nature of the problem also prefers stochastic optimal policies, if one exists. In this work, we address these challenges by (1) applying a (state) conditional normalizing flow to compactly represent the stochastic policy â€” the compactness arises due to the network only producing one sampled action and the corresponding log probability of the action, which is then used by an actor-critic method; and (2) employing an invalid action rejection method (via a valid action oracle) to update the base policy. The action rejection is enabled by a modified policy gradient that we derive. Finally, we conduct extensive experiments to show the scalability of our approach compared to prior methods and the ability to enforce arbitrary state-conditional constraints on the support of the distribution of actions in any state.

----

## [0] On the impact of activation and normalization in obtaining isometric embeddings at initialization

**Authors**: *Amir Joudaki, Hadi Daneshmand, Francis R. Bach*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d535a224c8ae54ba75bac0457b6b279-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d535a224c8ae54ba75bac0457b6b279-Abstract-Conference.html)

**Abstract**:

In this paper, we explore the structure of the penultimate Gram matrix in deep neural networks, which contains the pairwise inner products of outputs corresponding to a batch of inputs. In several architectures it has been observed that this Gram matrix becomes degenerate with depth at initialization, which dramatically slows training. Normalization layers, such as batch or layer normalization, play a pivotal role in preventing the rank collapse issue. Despite promising advances, the existing theoretical results do not extend to layer normalization, which is widely used in transformers, and can not quantitatively characterize the role of non-linear activations.  To bridge this gap, we prove that layer normalization, in conjunction with activation layers, biases the Gram matrix of a multilayer perceptron towards the identity matrix at an exponential rate with depth at initialization. We quantify this rate using the Hermite expansion of the activation function.

----

## [0] Uni3DETR: Unified 3D Detection Transformer

**Authors**: *Zhenyu Wang, Ya-Li Li, Xi Chen, Hengshuang Zhao, Shengjin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d60bfd8458b67acbbaf18b892338d00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d60bfd8458b67acbbaf18b892338d00-Abstract-Conference.html)

**Abstract**:

Existing point cloud based 3D detectors are designed for the particular scene, either indoor or outdoor ones. Because of the substantial differences in object distribution and point density within point clouds collected from various environments, coupled with the intricate nature of 3D metrics, there is still a lack of a unified network architecture that can accommodate diverse scenes. In this paper, we propose Uni3DETR, a unified 3D detector that addresses indoor and outdoor 3D detection within the same framework. Specifically, we employ the detection transformer with point-voxel interaction for object prediction, which leverages voxel features and points for cross-attention and behaves resistant to the discrepancies from data. We then propose the mixture of query points, which sufficiently exploits global information for dense small-range indoor scenes and local information for large-range sparse outdoor ones. Furthermore, our proposed decoupled IoU provides an easy-to-optimize training target for localization by disentangling the $xy$ and $z$ space. Extensive experiments validate that Uni3DETR exhibits excellent performance consistently on both indoor and outdoor 3D detection. In contrast to previous specialized detectors, which may perform well on some particular datasets but suffer a substantial degradation  on different scenes, Uni3DETR demonstrates the strong generalization ability under heterogeneous conditions (Fig. 1).

----

## [0] SceneScape: Text-Driven Consistent Scene Generation

**Authors**: *Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html)

**Abstract**:

We present a method for text-driven perpetual view generation -- synthesizing long-term videos of various scenes solely, given an input text prompt describing the scene and camera poses. We introduce a novel framework that generates such videos in an online fashion by combining the generative power of a pre-trained text-to-image model with the geometric priors learned by a pre-trained monocular depth prediction model. To tackle the pivotal challenge of achieving 3D consistency, i.e., synthesizing videos that depict geometrically-plausible scenes, we deploy an online test-time training to encourage the predicted depth map of the current frame to be geometrically consistent with the synthesized scene. The depth maps are used to construct a \emph{unified} mesh representation of the scene, which is progressively constructed along the video generation process. In contrast to previous works, which are applicable only to limited domains, our method generates diverse scenes, such as walkthroughs in spaceships, caves, or ice castles.

----

## [0] RDumb: A simple approach that questions our progress in continual test-time adaptation

**Authors**: *Ori Press, Steffen Schneider, Matthias Kümmerer, Matthias Bethge*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d640f377893fc5f22b5610e175ef7c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d640f377893fc5f22b5610e175ef7c3-Abstract-Conference.html)

**Abstract**:

Test-Time Adaptation (TTA) allows to update pre-trained models to changing data distributions at deployment time. While early work tested these algorithms for individual fixed distribution shifts, recent work proposed and applied methods for continual adaptation over long timescales. To examine the reported progress in the field, we propose the Continually Changing Corruptions (CCC) benchmark to measure asymptotic performance of TTA techniques. We find that eventually all but one state-of-the-art methods collapse and perform worse than a non-adapting model, including models specifically proposed to be robust to performance collapse. In addition, we introduce a simple baseline, "RDumb", that periodically resets the model to its pretrained state. RDumb performs better or on par with the previously proposed state-of-the-art in all considered benchmarks.Our results show that previous TTA approaches are neither effective at regularizing adaptation to avoid collapse nor able to outperform a simplistic resetting strategy.

----

## [0] Swap Agnostic Learning, or Characterizing Omniprediction via Multicalibration

**Authors**: *Parikshit Gopalan, Michael P. Kim, Omer Reingold*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d693203215325902ff9dbdd067a50ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d693203215325902ff9dbdd067a50ac-Abstract-Conference.html)

**Abstract**:

We introduce and study the notion of Swap Agnostic Learning.The problem can be phrased as a game between a *predictor* and an *adversary*:  first, the predictor selects a hypothesis $h$; then, the adversary plays in response, and for each level set of the predictor, selects a loss-minimizing hypothesis $c_v \in \mathcal{C}$; the predictor wins if $h$ competes with the adaptive adversary's loss.Despite the strength of the adversary, our main result demonstrates the feasibility Swap Agnostic Learning for any convex loss.Somewhat surprisingly, the result follows by proving an *equivalence* between Swap Agnostic Learning and swap variants of the recent notions Omniprediction (ITCS'22) and Multicalibration (ICML'18).Beyond this equivalence, we establish further connections to the literature on Outcome Indistinguishability (STOC'20, ITCS'23), revealing a unified notion of OI that captures all existing notions of omniprediction and multicalibration.

----

## [0] AR-Diffusion: Auto-Regressive Diffusion Model for Text Generation

**Authors**: *Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, Yelong Shen, Jian Jiao, Juntao Li, Zhongyu Wei, Jian Guo, Nan Duan, Weizhu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7d866abba506e5a56335e4644ebe18f9-Abstract-Conference.html)

**Abstract**:

Diffusion models have gained significant attention in the realm of image generation due to their exceptional performance. Their success has been recently expanded to text generation via generating all tokens within a sequence concurrently. However, natural language exhibits a far more pronounced sequential dependency in comparison to images, and the majority of existing language models are trained with a left-to-right auto-regressive approach.To account for the inherent sequential characteristic of natural language, we introduce  Auto-Regressive Diffusion (AR-Diffusion). AR-Diffusion ensures that the generation of tokens on the right depends on the generated ones on the left, a mechanism achieved through employing a dynamic number of denoising steps that vary based on token position. This results in tokens on the left undergoing fewer denoising steps than those on the right, thereby enabling them to generate earlier and subsequently influence the generation of tokens on the right.In a series of experiments on various text generation tasks, including text summarization, machine translation, and common sense generation, AR-Diffusion clearly demonstrated its superiority over existing diffusion language models and that it can be $100\times\sim600\times$ faster when achieving comparable results. Our code is available at https://github.com/microsoft/ProphetNet/tree/master/AR-diffusion.

----

## [0] Adaptive Uncertainty Estimation via High-Dimensional Testing on Latent Representations

**Authors**: *Tsai Hor Chan, Kin Wai Lau, Jiajun Shen, Guosheng Yin, Lequan Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7da558c6bd476ba77f5ba712626bba1a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7da558c6bd476ba77f5ba712626bba1a-Abstract-Conference.html)

**Abstract**:

Uncertainty estimation aims to evaluate the confidence of a trained deep neural network. However, existing uncertainty estimation approaches rely on low-dimensional distributional assumptions and thus suffer from the high dimensionality of latent features. Existing approaches tend to focus on uncertainty on discrete classification probabilities, which leads to poor generalizability to uncertainty estimation for other tasks. Moreover, most of the literature requires seeing the out-of-distribution (OOD) data in the training for better estimation of uncertainty, which limits the uncertainty estimation performance in practice because the OOD data are typically unseen. To overcome these  limitations, we propose a new framework using data-adaptive high-dimensional hypothesis testing for uncertainty estimation, which leverages the statistical properties of the feature representations. Our method directly operates on latent representations and thus does not require retraining the feature encoder under a modified objective. The test statistic relaxes the feature distribution assumptions to high dimensionality, and it is more discriminative to uncertainties in the latent representations. We demonstrate that encoding features with Bayesian neural networks can enhance testing performance and lead to more accurate uncertainty estimation. We further introduce a family-wise testing procedure to determine the optimal threshold of OOD detection, which minimizes the false discovery rate (FDR). Extensive experiments validate the satisfactory performance of our framework on uncertainty estimation and task-specific prediction over a variety of competitors. The experiments on the OOD detection task also show satisfactory performance of our method when the OOD data are unseen in the training. Codes are available at https://github.com/HKU-MedAI/bnn_uncertainty.

----

## [0] Algorithmic Regularization in Tensor Optimization: Towards a Lifted Approach in Matrix Sensing

**Authors**: *Ziye Ma, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7db2348b5bfeca620aa7327df815adcc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7db2348b5bfeca620aa7327df815adcc-Abstract-Conference.html)

**Abstract**:

Gradient descent (GD) is crucial for generalization in machine learning models, as it induces implicit regularization, promoting compact representations. In this work, we examine the role of GD in inducing implicit regularization for tensor optimization, particularly within the context of the lifted matrix sensing framework. This framework has been recently proposed to address the non-convex matrix sensing problem by transforming spurious solutions into strict saddles when optimizing over symmetric, rank-1 tensors. We show that, with sufficiently small initialization scale, GD applied to this lifted problem results in approximate rank-1 tensors and critical points with escape directions. Our findings underscore the significance of the tensor parametrization of matrix sensing, in combination with first-order methods, in achieving global optimality in such problems.

----

## [0] A General Theory of Correct, Incorrect, and Extrinsic Equivariance

**Authors**: *Dian Wang, Xupeng Zhu, Jung Yeon Park, Mingxi Jia, Guanang Su, Robert Platt, Robin Walters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7dc7793c89b93887e126a86f22ef63c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7dc7793c89b93887e126a86f22ef63c6-Abstract-Conference.html)

**Abstract**:

Although equivariant machine learning has proven effective at many tasks, success depends heavily on the assumption that the ground truth function is symmetric over the entire domain matching the symmetry in an equivariant neural network. A missing piece in the equivariant learning literature is the analysis of equivariant networks when symmetry exists only partially in the domain. In this work, we present a general theory for such a situation. We propose pointwise definitions of correct, incorrect, and extrinsic equivariance, which allow us to quantify continuously the degree of each type of equivariance a function displays. We then study the impact of various degrees of incorrect or extrinsic symmetry on model error. We prove error lower bounds for invariant or equivariant networks in classification or regression settings with partially incorrect symmetry. We also analyze the potentially harmful effects of extrinsic equivariance. Experiments validate these results in three different environments.

----

## [0] Analyzing Vision Transformers for Image Classification in Class Embedding Space

**Authors**: *Martina G. Vilas, Timothy Schaumlöffel, Gemma Roig*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7dd309df03d37643b96f5048b44da798-Abstract-Conference.html)

**Abstract**:

Despite the growing use of transformer models in computer vision, a mechanistic understanding of these networks is still needed. This work introduces a method to reverse-engineer Vision Transformers trained to solve image classification tasks. Inspired by previous research in NLP, we demonstrate how the inner representations at any level of the hierarchy can be projected onto the learned class embedding space to uncover how these networks build categorical representations for their predictions. We use our framework to show how image tokens develop class-specific representations that depend on attention mechanisms and contextual information, and give insights on how self-attention and MLP layers differentially contribute to this categorical composition. We additionally demonstrate that this method (1) can be used to determine the parts of an image that would be important for detecting the class of interest, and (2) exhibits significant advantages over traditional linear probing approaches. Taken together, our results position our proposed framework as a powerful tool for mechanistic interpretability and explainability research.

----

## [0] Toward Re-Identifying Any Animal

**Authors**: *Bingliang Jiao, Lingqiao Liu, Liying Gao, Ruiqi Wu, Guosheng Lin, Peng Wang, Yanning Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7df69dbf39705c7a39b40f2d70e806c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7df69dbf39705c7a39b40f2d70e806c1-Abstract-Conference.html)

**Abstract**:

The current state of re-identification (ReID) models poses limitations to their applicability in the open world, as they are primarily designed and trained for specific categories like person or vehicle. In light of the importance of ReID technology for tracking wildlife populations and migration patterns, we propose a new task called ``Re-identify Any Animal in the Wild'' (ReID-AW). This task aims to develop a ReID model capable of handling any unseen wildlife category it encounters. To address this challenge, we have created a comprehensive dataset called Wildlife-71, which includes ReID data from 71 different wildlife categories. This dataset is the first of its kind to encompass multiple object categories in the realm of ReID. Furthermore, we have developed a universal re-identification model named UniReID specifically for the ReID-AW task. To enhance the model's adaptability to the target category, we employ a dynamic prompting mechanism using category-specific visual prompts. These prompts are generated based on knowledge gained from a set of pre-selected images within the target category. Additionally, we leverage explicit semantic knowledge derived from the large-scale pre-trained language model, GPT-4. This allows UniReID to focus on regions that are particularly useful for distinguishing individuals within the target category. Extensive experiments have demonstrated the remarkable generalization capability of our UniReID model. It showcases promising performance in handling arbitrary wildlife categories, offering significant advancements in the field of ReID for wildlife conservation and research purposes.

----

## [0] Critical Initialization of Wide and Deep Neural Networks using Partial Jacobians: General Theory and Applications

**Authors**: *Darshil Doshi, Tianyu He, Andrey Gromov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e02f2910ea7911a37c4691f4201c878-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e02f2910ea7911a37c4691f4201c878-Abstract-Conference.html)

**Abstract**:

Deep neural networks are notorious for defying theoretical treatment. However, when the number of parameters in each layer tends to infinity, the network function is a Gaussian process (GP) and quantitatively predictive description is possible. Gaussian approximation allows one to formulate criteria for selecting hyperparameters, such as variances of weights and biases, as well as the learning rate. These criteria rely on the notion of criticality defined for deep neural networks. In this work we describe a new practical way to diagnose criticality. We introduce *partial Jacobians* of a network, defined as derivatives of preactivations in layer $l$ with respect to preactivations in layer $l_0\leq l$. We derive recurrence relations for the norms of partial Jacobians and utilize these relations to analyze criticality of deep fully connected neural networks with LayerNorm and/or residual connections. We derive and implement a simple and cheap numerical test that allows one to select optimal initialization for a broad class of deep neural networks; containing fully connected, convolutional and normalization layers. Using these tools we show quantitatively that proper stacking of the LayerNorm (applied to preactivations) and residual connections leads to an architecture that is critical for any initialization. Finally, we apply our methods to analyze ResNet and MLP-Mixer architectures; demonstrating the everywhere-critical regime.

----

## [0] Trading-off price for data quality to achieve fair online allocation

**Authors**: *Mathieu Molina, Nicolas Gast, Patrick Loiseau, Vianney Perchet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e0af0d1bc0ec2a90fc294be2e00447e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e0af0d1bc0ec2a90fc294be2e00447e-Abstract-Conference.html)

**Abstract**:

We consider the problem of online allocation subject to a long-term fairness penalty. Contrary to existing works, however, we do not assume that the decision-maker observes the protected attributes---which is often unrealistic in practice. Instead they can purchase data that help estimate them from sources of different quality; and hence reduce the fairness penalty at some cost. We model this problem as a multi-armed bandit problem where each arm corresponds to the choice of a data source, coupled with the fair online allocation problem. We propose an algorithm that jointly solves both problems and show that it has a regret bounded by $\mathcal{O}(\sqrt{T})$. A key difficulty is that the rewards received by selecting a source are correlated by the fairness penalty, which leads to a need for randomization (despite a stochastic setting). Our algorithm takes into account contextual information available before the source selection, and can adapt to many different fairness notions.

----

## [0] Asymptotics of Bayesian Uncertainty Estimation in Random Features Regression

**Authors**: *Youngsoo Baek, Samuel Berchuck, Sayan Mukherjee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e16384b94a1c7e4462a70bb8fb93ca9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e16384b94a1c7e4462a70bb8fb93ca9-Abstract-Conference.html)

**Abstract**:

In this paper we compare and contrast the behavior of the posterior predictive distribution to the risk of the the maximum a posteriori estimator for the random features regression model in the overparameterized regime.  We will focus on the variance of the  posterior predictive distribution (Bayesian model average) and compare its asymptotics to that of the risk of the MAP estimator. In the regime where the model dimensions grow faster than any constant multiple of the number of samples, asymptotic agreement between these two quantities is governed by the phase transition in the signal-to-noise ratio. They also asymptotically agree with each other when the number of samples grow faster than any constant multiple of model dimensions. Numerical simulations illustrate finer distributional properties of the two quantities for finite dimensions. We conjecture they have Gaussian fluctuations and exhibit similar properties as found by previous authors in a Gaussian sequence model, this is of independent theoretical interest.

----

## [0] Decentralized Matrix Sensing: Statistical Guarantees and Fast Convergence

**Authors**: *Marie Maros, Gesualdo Scutari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e6f445a74cdb71931aac64f1e3f49c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e6f445a74cdb71931aac64f1e3f49c9-Abstract-Conference.html)

**Abstract**:

We explore the matrix sensing problem from near-isotropic linear measurements, distributed across a network of agents modeled as an undirected graph, with no centralized node. We provide the first study of statistical, computational/communication guarantees for a decentralized gradient algorithm that solves the (nonconvex) Burer-Monteiro type decomposition associated to the low-rank matrix estimation. With small random initialization, the algorithm displays an approximate two-phase convergence: (i) a spectral phase that aligns the iterates' column space with the underlying low-rank matrix, mimicking centralized spectral initialization (not directly implementable over networks); and (ii) a local refinement phase that diverts the iterates from certain degenerate saddle points, while ensuring swift convergence to the underlying low-rank matrix. Central to our analysis is a novel "in-network" Restricted Isometry Property which accommodates for the decentralized nature of the optimization, revealing an intriguing interplay between sample complexity and network connectivity, topology, and communication complexity.

----

## [0] Dynamics Generalisation in Reinforcement Learning via Adaptive Context-Aware Policies

**Authors**: *Michael Beukman, Devon Jarvis, Richard Klein, Steven James, Benjamin Rosman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e7b768198d24d883d69704eee57efb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e7b768198d24d883d69704eee57efb0-Abstract-Conference.html)

**Abstract**:

While reinforcement learning has achieved remarkable successes in several domains, its real-world application is limited due to many methods failing to generalise to unfamiliar conditions. In this work, we consider the problem of generalising to new transition dynamics, corresponding to cases in which the environment's response to the agent's actions differs. For example, the gravitational force exerted on a robot depends on its mass and changes the robot's mobility. Consequently, in such cases, it is necessary to condition an agent's actions on extrinsic state information and pertinent contextual information reflecting how the environment responds. While the need for context-sensitive policies has been established, the manner in which context is incorporated architecturally has received less attention. Thus, in this work, we present an investigation into how context information should be incorporated into behaviour learning to improve generalisation.  To this end, we introduce a neural network architecture, the Decision Adapter, which generates the weights of an adapter module and conditions the behaviour of an agent on the context information. We show that the Decision Adapter is a useful generalisation of a previously proposed architecture and empirically demonstrate that it results in superior generalisation performance compared to previous approaches in several environments. Beyond this, the Decision Adapter is more robust to irrelevant distractor variables than several alternative methods.

----

## [0] Goal Driven Discovery of Distributional Differences via Language Descriptions

**Authors**: *Ruiqi Zhong, Peter Zhang, Steve Li, Jinwoo Ahn, Dan Klein, Jacob Steinhardt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e810b2c75d69be186cadd2fe3febeab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e810b2c75d69be186cadd2fe3febeab-Abstract-Conference.html)

**Abstract**:

Exploring large corpora can generate useful discoveries but is time-consuming for humans.    We formulate a new task, D5, that automatically discovers differences between two large corpora in a goal-driven way.     The task input is a problem comprising a user-specified research goal (“comparing the side effects of drug A and drug”) and a corpus pair (two large collections of patients' self-reported reactions after taking each drug).     The output is a goal-related description (discovery) of how these corpora differ (patients taking drug A “mention feelings of paranoia” more often).    We build a D5 system, and to quantitatively evaluate its performance, we 1) build a diagnostic benchmark, SynD5, to test whether it can recover known differences between two synthetic corpora, and 2) contribute a meta-dataset, OpenD5, aggregating 675 open-ended problems ranging across business, social sciences, humanities, machine learning, and health.    With both synthetic and real datasets, we confirm that language models can leverage the user-specified goals to propose more relevant candidate discoveries, and they sometimes produce discoveries previously unknown to the authors, including demographic differences in discussion topics, political stances in speech, insights in commercial reviews, and error patterns in NLP models.    Finally, we discuss the limitations of the current D5 system, which discovers correlation rather than causation and has the potential to reinforce societal biases when misused; therefore, practitioners should treat the outputs of our system with caution.

----

## [0] Convex and Non-convex Optimization Under Generalized Smoothness

**Authors**: *Haochuan Li, Jian Qian, Yi Tian, Alexander Rakhlin, Ali Jadbabaie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e8bb8d17bb1cb24dfe972a2f8ff2500-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e8bb8d17bb1cb24dfe972a2f8ff2500-Abstract-Conference.html)

**Abstract**:

Classical analysis of convex and non-convex optimization methods often requires the Lipschitz continuity of the gradient, which limits the analysis to functions bounded by quadratics. Recent work relaxed this requirement to a non-uniform smoothness condition with the Hessian norm  bounded by an affine function of the gradient norm, and proved convergence in the non-convex setting via gradient clipping, assuming bounded noise. In this paper, we further generalize this non-uniform smoothness condition and develop a simple, yet powerful analysis technique that bounds the gradients along the trajectory, thereby leading to  stronger results for both convex and non-convex optimization problems. In particular, we obtain the classical convergence rates for (stochastic) gradient descent and Nesterov's accelerated gradient method in the convex and/or non-convex setting under this general smoothness condition. The new analysis approach does not require gradient clipping and allows heavy-tailed noise with bounded variance in the stochastic setting.

----

## [0] DOSE: Diffusion Dropout with Adaptive Prior for Speech Enhancement

**Authors**: *Wenxin Tai, Yue Lei, Fan Zhou, Goce Trajcevski, Ting Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e966a12c2d6307adb8809aaa9acf057-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e966a12c2d6307adb8809aaa9acf057-Abstract-Conference.html)

**Abstract**:

Speech enhancement (SE) aims to improve the intelligibility and quality of speech in the presence of non-stationary additive noise. Deterministic deep learning models have traditionally been used for SE, but recent studies have shown that generative approaches, such as denoising diffusion probabilistic models (DDPMs), can also be effective. However, incorporating condition information into DDPMs for SE remains a challenge. We propose a model-agnostic method called DOSE that employs two efficient condition-augmentation techniques to address this challenge, based on two key insights: (1) We force the model to prioritize the condition factor when generating samples by training it with dropout operation; (2) We inject the condition information into the sampling process by providing an informative adaptive prior. Experiments demonstrate that our approach yields substantial improvements in high-quality and stable speech generation, consistency with the condition factor, and inference efficiency. Codes are publicly available at https://github.com/ICDM-UESTC/DOSE.

----

## [0] ISP: Multi-Layered Garment Draping with Implicit Sewing Patterns

**Authors**: *Ren Li, Benoît Guillard, Pascal Fua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e976afe805026f7d378a583af5ea9a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e976afe805026f7d378a583af5ea9a2-Abstract-Conference.html)

**Abstract**:

Many approaches to draping individual garments on human body models are realistic, fast, and yield outputs that are differentiable with respect to the body shape on which they are draped. However, they are either unable to handle multi-layered clothing, which is prevalent in everyday dress, or restricted to bodies in T-pose. In this paper, we introduce a parametric garment representation model that addresses these limitations. As in models used by clothing designers, each garment consists of individual 2D panels. Their 2D shape is defined by a Signed Distance Function and 3D shape by a 2D to 3D mapping. The 2D parameterization enables easy detection of potential collisions and the 3D parameterization handles complex shapes effectively. We show that this combination is faster and yields higher quality reconstructions than purely implicit surface representations, and makes the recovery of layered garments from images possible thanks to its differentiability. Furthermore, it supports rapid editing of garment shapes and texture by modifying individual 2D panels.

----

## [0] Optimality of Message-Passing Architectures for Sparse Graphs

**Authors**: *Aseem Baranwal, Kimon Fountoulakis, Aukosh Jagannath*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7e991aa4cd2fdf0014fba2f000f542d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7e991aa4cd2fdf0014fba2f000f542d0-Abstract-Conference.html)

**Abstract**:

We study the node classification problem on feature-decorated graphs in the sparse setting, i.e., when the expected degree of a node is $O(1)$ in the number of nodes, in the fixed-dimensional asymptotic regime, i.e., the dimension of the feature data is fixed while the number of nodes is large. Such graphs are typically known to be locally tree-like. We introduce a notion of Bayes optimality for node classification tasks, called asymptotic local Bayes optimality, and compute the optimal classifier according to this criterion for a fairly general statistical data model with arbitrary distributions of the node features and edge connectivity. The optimal classifier is implementable using a message-passing graph neural network architecture. We then compute the generalization error of this classifier and compare its performance against existing learning methods theoretically on a well-studied statistical model with naturally identifiable signal-to-noise ratios (SNRs) in the data. We find that the optimal message-passing architecture interpolates between a standard MLP in the regime of low graph signal and a typical convolution in the regime of high graph signal. Furthermore, we prove a corresponding non-asymptotic result.

----

## [0] Distribution-Free Statistical Dispersion Control for Societal Applications

**Authors**: *Zhun Deng, Thomas P. Zollo, Jake Snell, Toniann Pitassi, Richard S. Zemel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ea46207ec9bda974b140fe11d8dd727-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ea46207ec9bda974b140fe11d8dd727-Abstract-Conference.html)

**Abstract**:

Explicit finite-sample statistical guarantees on model performance are an important ingredient in responsible machine learning.  Previous work has focused mainly on bounding either the expected loss of a predictor or the probability that an individual prediction will incur a loss value in a specified range.  However, for many high-stakes applications it is crucial to understand and control the \textit{dispersion} of a loss distribution, or the extent to which different members of a population experience unequal effects of algorithmic decisions. We initiate the study of distribution-free control of statistical dispersion measures with societal implications and propose a simple yet flexible framework that allows us to handle a much richer class of statistical functionals beyond previous work. Our methods are verified through experiments in toxic comment detection, medical imaging, and film recommendation.

----

## [0] Switching Temporary Teachers for Semi-Supervised Semantic Segmentation

**Authors**: *Jaemin Na, Jung-Woo Ha, Hyung Jin Chang, Dongyoon Han, Wonjun Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7eeb42802d3750ca59e8a0523068e9e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7eeb42802d3750ca59e8a0523068e9e6-Abstract-Conference.html)

**Abstract**:

The teacher-student framework, prevalent in semi-supervised semantic segmentation, mainly employs the exponential moving average (EMA) to update a single teacher's weights based on the student's. However, EMA updates raise a problem in that the weights of the teacher and student are getting coupled, causing a potential performance bottleneck. Furthermore, this problem may become more severe when training with more complicated labels such as segmentation masks but with few annotated data. This paper introduces Dual Teacher, a simple yet effective approach that employs dual temporary teachers aiming to alleviate the coupling problem for the student. The temporary teachers work in shifts and are progressively improved, so consistently prevent the teacher and student from becoming excessively close. Specifically, the temporary teachers periodically take turns generating pseudo-labels to train a student model and maintain the distinct characteristics of the student model for each epoch. Consequently, Dual Teacher achieves competitive performance on the PASCAL VOC, Cityscapes, and ADE20K benchmarks with remarkably shorter training times than state-of-the-art methods. Moreover, we demonstrate that our approach is model-agnostic and compatible with both CNN- and Transformer-based models. Code is available at https://github.com/naver-ai/dual-teacher.

----

## [0] Extremal Domain Translation with Neural Optimal Transport

**Authors**: *Milena Gazdieva, Alexander Korotin, Daniil Selikhanovych, Evgeny Burnaev*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7eed2822411dc37b3768ae04561caafa-Abstract-Conference.html)

**Abstract**:

In many unpaired image domain translation problems, e.g., style transfer or super-resolution, it is important to keep the translated image similar to its respective input image. We propose the extremal transport (ET) which is a mathematical formalization of the theoretically best possible unpaired translation between a pair of domains w.r.t. the given similarity function. Inspired by the recent advances in neural optimal transport (OT), we propose a scalable algorithm to approximate ET maps as a limit of partial OT maps. We test our algorithm on toy examples and on the unpaired image-to-image translation task. The code is publicly available at https://github.com/milenagazdieva/ExtremalNeuralOptimalTransport

----

## [0] Recaptured Raw Screen Image and Video Demoiréing via Channel and Spatial Modulations

**Authors**: *Yijia Cheng, Xin Liu, Jingyu Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f05193e5487287a890df7fbc3554427-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f05193e5487287a890df7fbc3554427-Abstract-Conference.html)

**Abstract**:

Capturing screen contents by smartphone cameras has become a common way for information sharing. However, these images and videos are often degraded by moiré patterns, which are caused by frequency aliasing between the camera filter array and digital display grids. We observe that the moiré patterns in raw domain is simpler than those in sRGB domain, and the moiré patterns in raw color channels have different properties. Therefore, we propose an image and video demoiréing network tailored for raw inputs. We introduce a color-separated feature branch, and it is fused with the traditional feature-mixed branch via channel and spatial modulations. Specifically, the channel modulation utilizes modulated color-separated features to enhance the color-mixed features. The spatial modulation utilizes the feature with large receptive field to modulate the feature with small receptive field. In addition, we build the first well-aligned raw video demoiréing (RawVDemoiré) dataset and propose an efficient temporal alignment method by inserting alternating patterns. Experiments demonstrate that our method achieves state-of-the-art performance for both image and video demoiréing. Our dataset and code will be released after the acceptance of this work.

----

## [0] On Imitation in Mean-field Games

**Authors**: *Giorgia Ramponi, Pavel Kolev, Olivier Pietquin, Niao He, Mathieu Laurière, Matthieu Geist*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f2223201858b6ff4cc1832d8856459b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f2223201858b6ff4cc1832d8856459b-Abstract-Conference.html)

**Abstract**:

We explore the problem of imitation learning (IL) in the context of mean-field games (MFGs), where the goal is to imitate the behavior of a population of agents following a Nash equilibrium policy according to some unknown payoff function. IL in MFGs presents new challenges compared to single-agent IL, particularly when both the reward function and the transition kernel depend on the population distribution. In this paper, departing from the existing literature on IL for MFGs, we introduce a new solution concept called the Nash imitation gap. Then we show that when only the reward depends on the population distribution, IL in MFGs can be reduced to single-agent IL with similar guarantees. However, when the dynamics is population-dependent, we provide a novel upper-bound that suggests IL is harder in this setting. To address this issue, we propose a new adversarial formulation where the reinforcement learning problem is replaced by a mean-field control (MFC) problem, suggesting progress in IL within MFGs may have to build upon MFC.

----

## [0] CluB: Cluster Meets BEV for LiDAR-Based 3D Object Detection

**Authors**: *Yingjie Wang, Jiajun Deng, Yuenan Hou, Yao Li, Yu Zhang, Jianmin Ji, Wanli Ouyang, Yanyong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f2fc4053a66edfa430bcdf9a6ff3b17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f2fc4053a66edfa430bcdf9a6ff3b17-Abstract-Conference.html)

**Abstract**:

Currently, LiDAR-based 3D detectors are broadly categorized into two groups, namely, BEV-based detectors and cluster-based detectors.BEV-based detectors capture the contextual information from the Bird's Eye View (BEV) and fill their center voxels via feature diffusion with a stack of convolution layers, which, however, weakens the capability of presenting an object with the center point.On the other hand, cluster-based detectors exploit the voting mechanism and aggregate the foreground points into object-centric clusters for further prediction.In this paper, we explore how to effectively combine these two complementary representations into a unified framework.Specifically, we propose a new 3D object detection framework, referred to as CluB, which incorporates an auxiliary cluster-based branch into the BEV-based detector by enriching the object representation at both feature and query levels.Technically, CluB is comprised of two steps.First, we construct a cluster feature diffusion module to establish the association between cluster features and BEV features in a subtle and adaptive fashion. Based on that, an imitation loss is introduced to distill object-centric knowledge from the cluster features to the BEV features.Second, we design a cluster query generation module to leverage the voting centers directly from the cluster branch, thus enriching the diversity of object queries.Meanwhile, a direction loss is employed to encourage a more accurate voting center for each cluster.Extensive experiments are conducted on Waymo and nuScenes datasets, and our CluB achieves state-of-the-art performance on both benchmarks.

----

## [0] Probabilistic Exponential Integrators

**Authors**: *Nathanael Bosch, Philipp Hennig, Filip Tronarp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f64034009f4a5fa417a57e1a987c5cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f64034009f4a5fa417a57e1a987c5cd-Abstract-Conference.html)

**Abstract**:

Probabilistic solvers provide a flexible and efficient framework for simulation, uncertainty quantification, and inference in dynamical systems. However, like standard solvers, they suffer performance penalties for certain stiff systems, where small steps are required not for reasons of numerical accuracy but for the sake of stability. This issue is greatly alleviated in semi-linear problems by the probabilistic exponential integrators developed in this paper. By including the fast, linear dynamics in the prior, we arrive at a class of probabilistic integrators with favorable properties. Namely, they are proven to be L-stable, and in a certain case reduce to a classic exponential integrator---with the added benefit of providing a probabilistic account of the numerical error. The method is also generalized to arbitrary non-linear systems by imposing piece-wise semi-linearity on the prior via Jacobians of the vector field at the previous estimates, resulting in probabilistic exponential Rosenbrock methods. We evaluate the proposed methods on multiple stiff differential equations and demonstrate their improved stability and efficiency over established probabilistic solvers. The present contribution thus expands the range of problems that can be effectively tackled within probabilistic numerics.

----

## [0] Understanding Neural Network Binarization with Forward and Backward Proximal Quantizers

**Authors**: *Yiwei Lu, Yaoliang Yu, Xinlin Li, Vahid Partovi Nia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html)

**Abstract**:

In neural network binarization, BinaryConnect (BC) and its variants are considered the standard. These methods apply the sign function in their forward pass and their respective gradients are backpropagated to update the weights. However, the derivative of the sign function is zero whenever defined, which consequently freezes training. Therefore, implementations of BC (e.g., BNN) usually replace the derivative of sign in the backward computation with identity or other approximate gradient alternatives. Although such practice works well empirically, it is largely a heuristic or ``training trick.'' We aim at shedding some light on these training tricks from the optimization perspective. Building from existing theory on ProxConnect (PC, a generalization of BC), we (1) equip PC with different forward-backward quantizers and obtain ProxConnect++ (PC++) that includes existing binarization techniques as special cases; (2) derive a principled way to synthesize forward-backward quantizers with automatic theoretical guarantees; (3) illustrate our theory by proposing an enhanced binarization algorithm BNN++; (4) conduct image classification experiments on CNNs and vision transformers, and empirically verify that BNN++ generally achieves competitive results on binarizing these models.

----

## [0] QH9: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules

**Authors**: *Haiyang Yu, Meng Liu, Youzhi Luo, Alex Strasser, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f755e271717450020fda40f020922dd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f755e271717450020fda40f020922dd-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Supervised machine learning approaches have been increasingly used in accelerating electronic structure prediction as surrogates of first-principle computational methods, such as density functional theory (DFT). While numerous quantum chemistry datasets focus on chemical properties and atomic forces, the ability to achieve accurate and efficient prediction of the Hamiltonian matrix is highly desired, as it is the most important and fundamental physical quantity that determines the quantum states of physical systems and chemical properties. In this work, we generate a new Quantum Hamiltonian dataset, named as QH9, to provide precise Hamiltonian matrices for 2,399 molecular dynamics trajectories and 130,831  stable molecular geometries, based on the QM9 dataset. By designing benchmark tasks with various molecules, we show that current machine learning models have the capacity to predict Hamiltonian matrices for arbitrary molecules. Both the QH9 dataset and the baseline models are provided to the community through an open-source benchmark, which can be highly valuable for developing machine learning methods and accelerating molecular and materials design for scientific and technological applications. Our benchmark is publicly available at \url{https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench}.

----

## [0] CorresNeRF: Image Correspondence Priors for Neural Radiance Fields

**Authors**: *Yixing Lao, Xiaogang Xu, Zhipeng Cai, Xihui Liu, Hengshuang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f77492bb8070a5c825a87c0c5181da2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f77492bb8070a5c825a87c0c5181da2-Abstract-Conference.html)

**Abstract**:

Neural Radiance Fields (NeRFs) have achieved impressive results in novel view synthesis and surface reconstruction tasks. However, their performance suffers under challenging scenarios with sparse input views. We present CorresNeRF, a novel method that leverages image correspondence priors computed by off-the-shelf methods to supervise NeRF training. We design adaptive processes for augmentation and filtering to generate dense and high-quality correspondences. The correspondences are then used to regularize NeRF training via the correspondence pixel reprojection and depth loss terms. We evaluate our methods on novel view synthesis and surface reconstruction tasks with density-based and SDF-based NeRF models on different datasets. Our method outperforms previous methods in both photometric and geometric metrics. We show that this simple yet effective technique of using correspondence priors can be applied as a plug-and-play module across different NeRF variants. The project page is at https://yxlao.github.io/corres-nerf/.

----

## [0] Score-based Data Assimilation

**Authors**: *François Rozet, Gilles Louppe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f7fa581cc8a1970a4332920cdf87395-Abstract-Conference.html)

**Abstract**:

Data assimilation, in its most comprehensive form, addresses the Bayesian inverse problem of identifying plausible state trajectories that explain noisy or incomplete observations of stochastic dynamical systems. Various approaches have been proposed to solve this problem, including particle-based and variational methods. However, most algorithms depend on the transition dynamics for inference, which becomes intractable for long time horizons or for high-dimensional systems with complex dynamics, such as oceans or atmospheres. In this work, we introduce score-based data assimilation for trajectory inference. We learn a score-based generative model of state trajectories based on the key insight that the score of an arbitrarily long trajectory can be decomposed into a series of scores over short segments. After training, inference is carried out using the score model, in a non-autoregressive manner by generating all states simultaneously. Quite distinctively, we decouple the observation model from the training procedure and use it only at inference to guide the generative process, which enables a wide range of zero-shot observation scenarios. We present theoretical and empirical evidence supporting the effectiveness of our method.

----

## [0] Mr HiSum: A Large-scale Dataset for Video Highlight Detection and Summarization

**Authors**: *Jinhwan Sul, Jihoon Han, Joonseok Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f880e3a325b06e3601af1384a653038-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f880e3a325b06e3601af1384a653038-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Video highlight detection is a task to automatically select the most engaging moments from a long video. This problem is highly challenging since it aims to learn a general way of finding highlights from a variety of videos in the real world.The task has an innate subjectivity because the definition of a highlight differs across individuals. Therefore, to detect consistent and meaningful highlights, prior benchmark datasets have been labeled by multiple (5-20) raters. Due to the high cost of manual labeling, most existing public benchmarks are in extremely small scale, containing only a few tens or hundreds of videos. This insufficient benchmark scale causes multiple issues such as unstable evaluation or high sensitivity in traintest splits. We present Mr. HiSum, a large-scale dataset for video highlight detection and summarization, containing 31,892 videos and reliable labels aggregated over 50,000+ users per video. We empirically prove reliability of the labels as frame importance by cross-dataset transfer and user study.

----

## [0] Sharp Bounds for Generalized Causal Sensitivity Analysis

**Authors**: *Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7f8b8bc8ebac661c442c4dafd5d98c08-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7f8b8bc8ebac661c442c4dafd5d98c08-Abstract-Conference.html)

**Abstract**:

Causal inference from observational data is crucial for many disciplines such as medicine and economics. However, sharp bounds for causal effects under relaxations of the unconfoundedness assumption (causal sensitivity analysis) are subject to ongoing research. So far, works with sharp bounds are restricted to fairly simple settings (e.g., a single binary treatment). In this paper, we propose a unified framework for causal sensitivity analysis under unobserved confounding in various settings. For this, we propose a flexible generalization of the marginal sensitivity model (MSM) and then derive sharp bounds for a large class of causal effects. This includes (conditional) average treatment effects, effects for mediation analysis and path analysis, and distributional effects. Furthermore, our sensitivity model is applicable to discrete, continuous, and time-varying treatments. It allows us to interpret the partial identification problem under unobserved confounding as a distribution shift in the latent confounders while evaluating the causal effect of interest. In the special case of a single binary treatment, our bounds for (conditional) average treatment effects coincide with recent optimality results for causal sensitivity analysis. Finally, we propose a scalable algorithm to estimate our sharp bounds from observational data.

----

## [0] Supported Value Regularization for Offline Reinforcement Learning

**Authors**: *Yixiu Mao, Hongchang Zhang, Chen Chen, Yi Xu, Xiangyang Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fa46657df480226112d5be3faf096c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fa46657df480226112d5be3faf096c4-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning suffers from the extrapolation error and value overestimation caused by out-of-distribution (OOD) actions. To mitigate this issue, value regularization approaches aim to penalize the learned value functions to assign lower values to OOD actions. However, existing value regularization methods lack a proper distinction between the regularization effects on in-distribution (ID) and OOD actions, and fail to guarantee optimal convergence results of the policy. To this end, we propose Supported Value Regularization (SVR), which penalizes the Q-values for all OOD actions while maintaining standard Bellman updates for ID ones. Specifically, we utilize the bias of importance sampling to compute the summation of Q-values over the entire OOD region, which serves as the penalty for policy evaluation. This design automatically separates the regularization for ID and OOD actions without manually distinguishing between them. In tabular MDP, we show that the policy evaluation operator of SVR is a contraction, whose fixed point outputs unbiased Q-values for ID actions and underestimated Q-values for OOD actions. Furthermore, the policy iteration with SVR guarantees strict policy improvement until convergence to the optimal support-constrained policy in the dataset. Empirically, we validate the theoretical properties of SVR in a tabular maze environment and demonstrate its state-of-the-art performance on a range of continuous control tasks in the D4RL benchmark.

----

## [0] Revisit Weakly-Supervised Audio-Visual Video Parsing from the Language Perspective

**Authors**: *Yingying Fan, Yu Wu, Bo Du, Yutian Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fbae0a0885d3d688840bd34e4a8a698-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fbae0a0885d3d688840bd34e4a8a698-Abstract-Conference.html)

**Abstract**:

We focus on the weakly-supervised audio-visual video parsing task (AVVP), which aims to identify and locate all the events in audio/visual modalities. Previous works only concentrate on video-level overall label denoising across modalities, but overlook the segment-level label noise, where adjacent video segments (i.e., 1-second video clips) may contain different events. However, recognizing events on the segment is challenging because its label could be any combination of events that occur in the video. To address this issue, we consider tackling AVVP from the language perspective, since language could freely describe how various events appear in each segment beyond fixed labels. Specifically, we design language prompts to describe all cases of event appearance for each video. Then, the similarity between language prompts and segments is calculated, where the event of the most similar prompt is regarded as the segment-level label. In addition, to deal with the mislabeled segments, we propose to perform dynamic re-weighting on the unreliable segments to adjust their labels. Experiments show that our simple yet effective approach outperforms state-of-the-art methods by a large margin.

----

## [0] Digital Typhoon: Long-term Satellite Image Dataset for the Spatio-Temporal Modeling of Tropical Cyclones

**Authors**: *Asanobu Kitamoto, Jared Hwang, Bastien Vuillod, Lucas Gautier, Yingtao Tian, Tarin Clanuwat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fc36bce5de315751001981baaf4751a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fc36bce5de315751001981baaf4751a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper presents the official release of the Digital Typhoon dataset, the longest typhoon satellite image dataset for 40+ years aimed at benchmarking machine learning models for long-term spatio-temporal data. To build the dataset, we developed a workflow to create an infrared typhoon-centered image for cropping using Lambert azimuthal equal-area projection referring to the best track data. We also address data quality issues such as inter-satellite calibration to create a homogeneous dataset. To take advantage of the dataset, we organized machine learning tasks by the types and targets of inference, with other tasks for meteorological analysis, societal impact, and climate change. The benchmarking results on the analysis, forecasting, and reanalysis for the intensity suggest that the dataset is challenging for recent deep learning models, due to many choices that affect the performance of various models. This dataset reduces the barrier for machine learning researchers to meet large-scale real-world events called tropical cyclones and develop machine learning models that may contribute to advancing scientific knowledge on tropical cyclones as well as solving societal and sustainability issues such as disaster reduction and climate change. The dataset is publicly available at http://agora.ex.nii.ac.jp/digital-typhoon/dataset/ and https://github.com/kitamoto-lab/digital-typhoon/.

----

## [0] Maximum Independent Set: Self-Training through Dynamic Programming

**Authors**: *Lorenzo Brusca, Lars C. P. M. Quaedvlieg, Stratis Skoulakis, Grigorios Chrysos, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7fe3170d88a8310ca86df2843f54236c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7fe3170d88a8310ca86df2843f54236c-Abstract-Conference.html)

**Abstract**:

This work presents a graph neural network (GNN) framework for solving the maximum independent set (MIS) problem, inspired by dynamic programming (DP). Specifically, given a graph, we propose a DP-like recursive algorithm based on GNNs that firstly constructs two smaller sub-graphs, predicts the one with the larger MIS, and then uses it in the next recursive call. To train our algorithm, we require annotated comparisons of different graphs concerning their MIS size. Annotating the comparisons with the output of our algorithm leads to a self-training process that results in more accurate self-annotation of the comparisons and vice versa. We provide numerical evidence showing the superiority of our method vs prior methods in multiple synthetic and real-world datasets.

----

## [0] Reference-Based POMDPs

**Authors**: *Edward Kim, Yohan Karunanayake, Hanna Kurniawati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ffb2b550ff6a75c536b279348a93fb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ffb2b550ff6a75c536b279348a93fb0-Abstract-Conference.html)

**Abstract**:

Making good decisions in partially observable and non-deterministic scenarios is a crucial capability for robots. A Partially Observable Markov Decision Process (POMDP) is a general framework for the above problem. Despite advances in POMDP solving, problems with long planning horizons and evolving environments remain difficult to solve even by the best approximate solvers today. To alleviate this difficulty, we propose a slightly modified POMDP problem, called a Reference-Based POMDP, where the objective is to balance between maximizing the expected total reward and being close to a given reference (stochastic) policy. The optimal policy of a Reference-Based POMDP can be computed via iterative expectations using the given reference policy, thereby avoiding exhaustive enumeration of actions at each belief node of the search tree. We demonstrate theoretically that the standard POMDP under stochastic policies is related to the Reference-Based POMDP. To demonstrate the feasibility of exploiting the formulation, we present a basic algorithm RefSolver. Results from experiments on long-horizon navigation problems indicate that this basic algorithm substantially outperforms POMCP.

----

## [0] Siamese Masked Autoencoders

**Authors**: *Agrim Gupta, Jiajun Wu, Jia Deng, Fei-Fei Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/7ffb9f1b57628932518505b532301603-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/7ffb9f1b57628932518505b532301603-Abstract-Conference.html)

**Abstract**:

Establishing correspondence between images or scenes is a significant challenge in computer vision, especially given occlusions, viewpoint changes, and varying object appearances. In this paper, we present Siamese Masked Autoencoders (SiamMAE), a simple extension of Masked Autoencoders (MAE) for learning visual correspondence from videos. SiamMAE operates on pairs of randomly sampled video frames and asymmetrically masks them. These frames are processed independently by an encoder network, and a decoder composed of a sequence of cross-attention layers is tasked with predicting the missing patches in the future frame. By masking a large fraction (95%) of patches in the future frame while leaving the past frame unchanged, SiamMAE encourages the network to focus on object motion and learn object-centric representations. Despite its conceptual simplicity, features learned via SiamMAE outperform state-of-the-art self-supervised methods on video object segmentation, pose keypoint propagation, and semantic part propagation tasks. SiamMAE achieves competitive results without relying on data augmentation, handcrafted tracking-based pretext tasks, or other techniques to prevent representational collapse.

----

## [0] Score-based Generative Models with Lévy Processes

**Authors**: *Eun-Bi Yoon, Keehun Park, Sungwoong Kim, Sungbin Lim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8011b23e1dc3f57e1b6211ccad498919-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8011b23e1dc3f57e1b6211ccad498919-Abstract-Conference.html)

**Abstract**:

Investigating the optimal stochastic process beyond Gaussian for noise injection in a score-based generative model remains an open question. Brownian motion is a light-tailed process with continuous paths, which leads to a slow convergence rate for the Number of Function Evaluation (NFE). Recent studies have shown that diffusion models suffer from mode-collapse issues on imbalanced data.In order to overcome the limitations of Brownian motion, we introduce a novel score-based generative model referred to as Lévy-Itō Model (LIM). This model utilizes isotropic $\alpha$-stable Lévy processes. We first derive an exact reverse-time stochastic differential equation driven by the Lévy process and develop the corresponding fractional denoising score matching. The proposed generative model takes advantage of the heavy-tailed properties of the Lévy process. Our experimental results show LIM allows for faster and more diverse sampling while maintaining high fidelity compared to existing diffusion models across various image datasets such as CIFAR10, CelebA, and imbalanced dataset CIFAR10LT. Comparing our results to those of DDPM with 3.21 Fréchet Inception Distance (FID) and 0.6437 Recall on the CelebA dataset, we achieve 1.58 FID and 0.7006 Recall using the same architecture. LIM shows the best performance in NFE 500 with $2\times$ faster total wall-clock time than the baseline.

----

## [0] 3D Indoor Instance Segmentation in an Open-World

**Authors**: *Mohamed El Amine Boudjoghra, Salwa K. Al Khatib, Jean Lahoud, Hisham Cholakkal, Rao Muhammad Anwer, Salman H. Khan, Fahad Shahbaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/801750bc49fdc3d498e9ee63479f315e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/801750bc49fdc3d498e9ee63479f315e-Abstract-Conference.html)

**Abstract**:

Existing 3D instance segmentation methods typically assume that all semantic classes to be segmented would be available during training and only seen categories are segmented at inference. We argue that such a closed-world assumption is restrictive and explore for the first time 3D indoor instance segmentation in an open-world setting, where the model is allowed to distinguish a set of known classes as well as identify an unknown object as unknown and then later incrementally learning the semantic category of the unknown when the corresponding category labels are available. To this end, we introduce an open-world 3D indoor instance segmentation method, where an auto-labeling scheme is employed to produce pseudo-labels during training and induce separation to separate known and unknown category labels. We further improve the pseudo-labels quality at inference by adjusting the unknown class probability based on the objectness score distribution. We also introduce carefully curated open-world splits leveraging realistic scenarios based on inherent object distribution, region-based indoor scene exploration and randomness aspect of open-world classes. Extensive experiments reveal the efficacy of the proposed contributions leading to promising open-world 3D instance segmentation performance. Code and splits are available at: https://github.com/aminebdj/3D-OWIS.

----

## [0] AbDiffuser: full-atom generation of in-vitro functioning antibodies

**Authors**: *Karolis Martinkus, Jan Ludwiczak, Wei-Ching Liang, Julien Lafrance-Vanasse, Isidro Hötzel, Arvind Rajpal, Yan Wu, Kyunghyun Cho, Richard Bonneau, Vladimir Gligorijevic, Andreas Loukas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/801ec05b0aae9fcd2ef35c168bd538e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/801ec05b0aae9fcd2ef35c168bd538e0-Abstract-Conference.html)

**Abstract**:

We introduce AbDiffuser, an equivariant and physics-informed diffusion model for the joint generation of antibody 3D structures and sequences. AbDiffuser is built on top of a new representation of protein structure, relies on a novel architecture for aligned proteins, and utilizes strong diffusion priors to improve the denoising process. Our approach improves protein diffusion by taking advantage of domain knowledge and physics-based constraints; handles sequence-length changes; and reduces memory complexity by an order of magnitude, enabling backbone and side chain generation. We validate AbDiffuser in silico and in vitro. Numerical experiments showcase the ability of AbDiffuser to generate antibodies that closely track the sequence and structural properties of a reference set. Laboratory experiments confirm that all 16 HER2 antibodies discovered were expressed at high levels and that 57.1% of the selected designs were tight binders.

----

## [0] Structure Learning with Adaptive Random Neighborhood Informed MCMC

**Authors**: *Xitong Liang, Alberto Caron, Samuel Livingstone, Jim E. Griffin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8027ace571384361920665f1d1b69758-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8027ace571384361920665f1d1b69758-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce a novel MCMC sampler, PARNI-DAG, for a fully-Bayesian approach to the problem of structure learning under observational data. Under the assumption of causal sufficiency, the algorithm allows for approximate sampling directly from the posterior distribution on Directed Acyclic Graphs (DAGs). PARNI-DAG performs efficient sampling of DAGs via locally informed, adaptive random neighborhood proposal that results in better mixing properties. In addition, to ensure better scalability with the number of nodes, we couple PARNI-DAG with a pre-tuning procedure of the sampler's parameters that exploits a skeleton graph derived through some constraint-based or scoring-based algorithms. Thanks to these novel features, PARNI-DAG quickly converges to high-probability regions and is less likely to get stuck in local modes in the presence of high correlation between nodes in high-dimensional settings. After introducing the technical novelties in PARNI-DAG, we empirically demonstrate its mixing efficiency and accuracy in learning DAG structures on a variety of experiments.

----

## [0] Reining Generalization in Offline Reinforcement Learning via Representation Distinction

**Authors**: *Yi Ma, Hongyao Tang, Dong Li, Zhaopeng Meng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/802a4350ca4fced76b13b8b320af1543-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/802a4350ca4fced76b13b8b320af1543-Abstract-Conference.html)

**Abstract**:

Offline Reinforcement Learning (RL) aims to address the challenge of distribution shift between the dataset and the learned policy, where the value of out-of-distribution (OOD) data may be erroneously estimated due to overgeneralization. It has been observed that a considerable portion of the benefits derived from the conservative terms designed by existing offline RL approaches originates from their impact on the learned representation. This observation prompts us to scrutinize the learning dynamics of offline RL, formalize the process of generalization, and delve into the prevalent overgeneralization issue in offline RL. We then investigate the potential to rein the generalization from the representation perspective to enhance offline RL. Finally, we present  Representation Distinction (RD), an innovative plug-in method for improving offline RL algorithm performance by explicitly differentiating between the representations of in-sample and OOD state-action pairs generated by the learning policy. Considering scenarios in which the learning policy mirrors the behavioral policy and similar samples may be erroneously distinguished, we suggest a dynamic adjustment mechanism for RD based on an OOD data generator to prevent data representation collapse and further enhance policy performance. We demonstrate the efficacy of our approach by applying RD to specially-designed backbone algorithms and widely-used offline RL algorithms. The proposed RD method significantly improves their performance across various continuous control tasks on D4RL datasets, surpassing several state-of-the-art offline RL algorithms.

----

## [0] BIRD: Generalizable Backdoor Detection and Removal for Deep Reinforcement Learning

**Authors**: *Xuan Chen, Wenbo Guo, Guanhong Tao, Xiangyu Zhang, Dawn Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/802e90325f4c8546e13e5763b2ecab88-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/802e90325f4c8546e13e5763b2ecab88-Abstract-Conference.html)

**Abstract**:

Backdoor attacks pose a severe threat to the supply chain management of deep reinforcement learning (DRL) policies. Despite initial defenses proposed in recent studies, these methods have very limited generalizability and scalability. To address this issue, we propose BIRD, a technique to detect and remove backdoors from a pretrained DRL policy in a clean environment without requiring any knowledge about the attack specifications and accessing its training process. By analyzing the unique properties and behaviors of backdoor attacks, we formulate trigger restoration as an optimization problem and design a novel metric to detect backdoored policies. We also design a finetuning method to remove the backdoor, while maintaining the agent's performance in the clean environment. We evaluate BIRD against three backdoor attacks in ten different single-agent or multi-agent environments. Our results verify the effectiveness, efficiency, and generalizability of BIRD, as well as its robustness to different attack variations and adaptions.

----

## [0] Cluster-aware Semi-supervised Learning: Relational Knowledge Distillation Provably Learns Clustering

**Authors**: *Yijun Dong, Kevin Miller, Qi Lei, Rachel Ward*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8037f47a6254eb60899a644bd90b4f6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8037f47a6254eb60899a644bd90b4f6a-Abstract-Conference.html)

**Abstract**:

Despite the empirical success and practical significance of (relational) knowledge distillation that matches (the relations of) features between teacher and student models, the corresponding theoretical interpretations remain limited for various knowledge distillation paradigms. In this work, we take an initial step toward a theoretical understanding of relational knowledge distillation (RKD), with a focus on semi-supervised classification problems. We start by casting RKD as spectral clustering on a population-induced graph unveiled by a teacher model. Via a notion of clustering error that quantifies the discrepancy between the predicted and ground truth clusterings, we illustrate that RKD over the population provably leads to low clustering error. Moreover, we provide a sample complexity bound for RKD with limited unlabeled samples. For semi-supervised learning, we further demonstrate the label efficiency of RKD through a general framework of cluster-aware semi-supervised learning that assumes low clustering errors. Finally, by unifying data augmentation consistency regularization into this cluster-aware framework, we show that despite the common effect of learning accurate clusterings, RKD facilitates a "global" perspective through spectral clustering, whereas consistency regularization focuses on a "local" perspective via expansion.

----

## [0] Bicriteria Multidimensional Mechanism Design with Side Information

**Authors**: *Siddharth Prasad, Maria-Florina Balcan, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8039ca1e9860daab3a79e45d010d5398-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8039ca1e9860daab3a79e45d010d5398-Abstract-Conference.html)

**Abstract**:

We develop a versatile new methodology for multidimensional mechanism design that incorporates side information about agent types to generate high social welfare and high revenue simultaneously. Prominent sources of side information in practice include predictions from a machine-learning model trained on historical agent data, advice from domain experts, and even the mechanism designer's own gut instinct. In this paper we adopt a prior-free perspective that makes no assumptions on the correctness, accuracy, or source of the side information. First, we design a meta-mechanism that integrates input side information with an improvement of the classical VCG mechanism. The welfare, revenue, and incentive properties of our meta-mechanism are characterized by novel constructions we introduce based on the notion of a weakest competitor, which is an agent that has the smallest impact on welfare. We show that our meta-mechanism, when carefully instantiated, simultaneously achieves strong welfare and revenue guarantees parameterized by errors in the side information. When the side information is highly informative and accurate, our mechanism achieves welfare and revenue competitive with the total social surplus, and its performance decays continuously and gradually as the quality of the side information decreases. Finally, we apply our meta-mechanism to a setting where each agent's type is determined by a constant number of parameters. Specifically, agent types lie on constant-dimensional subspaces (of the potentially high-dimensional ambient type space) that are known to the mechanism designer. We use our meta-mechanism to obtain the first known welfare and revenue guarantees in this setting.

----

## [0] Are These the Same Apple? Comparing Images Based on Object Intrinsics

**Authors**: *Klemen Kotar, Stephen Tian, Hong-Xing Yu, Dan Yamins, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/803c6ab3d62346e004ef70211d2d15b8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/803c6ab3d62346e004ef70211d2d15b8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The human visual system can effortlessly recognize an object under different extrinsic factors such as lighting, object poses, and background, yet current computer vision systems often struggle with these variations. An important step to understanding and improving artificial vision systems is to measure image similarity purely based on intrinsic object properties that define object identity. This problem has been studied in the computer vision literature as re-identification, though mostly restricted to specific object categories such as people and cars. We propose to extend it to general object categories, exploring an image similarity metric based on object intrinsics. To benchmark such measurements, we collect the Common paired objects Under differenT Extrinsics (CUTE) dataset of 18, 000 images of 180 objects under different extrinsic factors such as lighting, poses, and imaging conditions. While existing methods such as LPIPS and CLIP scores do not measure object intrinsics well, we find that combining deep features learned from contrastive self-supervised learning with foreground filtering is a simple yet effective approach to approximating the similarity. We conduct an extensive survey of pre-trained features and foreground extraction methods to arrive at a strong baseline that best measures intrinsic object-centric image similarity among current methods. Finally, we demonstrate that our approach can aid in downstream applications such as acting as an analog for human subjects and improving generalizable re-identification. Please see our project website at https://s-tian.github.io/projects/cute/ for visualizations of the data and demos of our metric.

----

## [0] The ToMCAT Dataset

**Authors**: *Adarsh Pyarelal, Eric Duong, Caleb Shibu, Paulo Soares, Savannah Boyd, Payal Khosla, Valeria A. Pfeifer, Diheng Zhang, Eric Andrews, Rick Champlin, Vincent Raymond, Meghavarshini Krishnaswamy, Clayton T. Morrison, Emily Butler, Kobus Barnard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/803d8d4b4a549d0d062fc704f8659ce3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/803d8d4b4a549d0d062fc704f8659ce3-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present a rich, multimodal dataset consisting of data from 40 teams of three humans conducting simulated urban search-and-rescue (SAR) missions in a Minecraft-based testbed, collected for the Theory of Mind-based Cognitive Architecture for Teams (ToMCAT) project. Modalities include two kinds of brain scan data---functional near-infrared spectroscopy (fNIRS) and electroencephalography (EEG), as well as skin conductance, heart rate, eye tracking, face images, spoken dialog audio data with automatic speech recognition (ASR) transcriptions, game screenshots, gameplay data, game performance data, demographic data, and self-report questionnaires. Each team undergoes up to six consecutive phases: three behavioral tasks, one mission training session, and two collaborative SAR missions. As time-synchronized multimodal data collected under a variety of circumstances, this dataset will support studying a large variety of research questions on topics including teamwork, coordination, plan recognition, affective computing, physiological linkage, entrainment, and dialog understanding.  We provide an initial public release of the de-identified data, along with analyses illustrating the utility of this dataset to both computer scientists and social scientists.

----

## [0] Exploring Diverse In-Context Configurations for Image Captioning

**Authors**: *Xu Yang, Yongliang Wu, Mingzhuo Yang, Haokun Chen, Xin Geng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/804b5e300c9ed4e3ea3b073f186f4adc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/804b5e300c9ed4e3ea3b073f186f4adc-Abstract-Conference.html)

**Abstract**:

After discovering that Language Models (LMs) can be good in-context few-shot learners, numerous strategies have been proposed to optimize in-context sequence configurations. Recently, researchers in Vision-Language (VL) domains also develop their few-shot learners, while they only use the simplest way, \ie, randomly sampling, to configure in-context image-text pairs. In order to explore the effects of varying configurations on VL in-context learning, we devised four strategies for image selection and four for caption assignment to configure in-context image-text pairs for image captioning. Here Image Captioning is used as the case study since it can be seen as the visually-conditioned LM. Our comprehensive experiments yield two counter-intuitive but valuable insights, highlighting the distinct characteristics of VL in-context learning due to multi-modal synergy, as compared to the NLP case. Furthermore, in our exploration of optimal combination strategies, we observed an average performance enhancement of 20.9 in CIDEr scores compared to the baseline. The code is given in https://github.com/yongliang-wu/ExploreCfg.

----

## [0] DELIFFAS: Deformable Light Fields for Fast Avatar Synthesis

**Authors**: *Youngjoong Kwon, Lingjie Liu, Henry Fuchs, Marc Habermann, Christian Theobalt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/805c06617d2b643278936daadfde4280-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/805c06617d2b643278936daadfde4280-Abstract-Conference.html)

**Abstract**:

Generating controllable and photorealistic digital human avatars is a long-standing and important problem in Vision and Graphics. Recent methods have shown great progress in terms of either photorealism or inference speed while the combination of the two desired properties still remains unsolved. To this end, we propose a novel method, called DELIFFAS, which parameterizes the appearance of the human as a surface light field that is attached to a controllable and deforming human mesh model. At the core, we represent the light field around the human with a deformable two-surface parameterization, which enables fast and accurate inference of the human appearance. This allows perceptual supervision on the full image compared to previous approaches that could only supervise individual pixels or small patches due to their slow runtime. Our carefully designed human representation and supervision strategy leads to state-of-the-art synthesis results and inference time. The video results and code are available at https://vcai.mpi-inf.mpg.de/projects/DELIFFAS.

----

## [0] Zero-Shot Anomaly Detection via Batch Normalization

**Authors**: *Aodong Li, Chen Qiu, Marius Kloft, Padhraic Smyth, Maja Rudolph, Stephan Mandt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8078e8c3055303a884ffae2d3ea00338-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8078e8c3055303a884ffae2d3ea00338-Abstract-Conference.html)

**Abstract**:

Anomaly detection (AD) plays a crucial role in many safety-critical application domains. The challenge of adapting an anomaly detector to drift in the normal data distribution, especially when no training data is available for the "new normal," has led to the development of zero-shot AD techniques. In this paper, we propose a simple yet effective method called Adaptive Centered Representations (ACR) for zero-shot batch-level AD. Our approach trains off-the-shelf deep anomaly detectors (such as deep SVDD) to adapt to a set of inter-related training data distributions in combination with batch normalization, enabling automatic zero-shot generalization for unseen AD tasks. This simple recipe, batch normalization plus meta-training, is a highly effective and versatile tool. Our results demonstrate the first zero-shot AD results for tabular data and outperform existing methods in zero-shot anomaly detection and segmentation on image data from specialized domains.

----

## [0] What Makes Data Suitable for a Locally Connected Neural Network? A Necessary and Sufficient Condition Based on Quantum Entanglement

**Authors**: *Yotam Alexander, Nimrod De La Vega, Noam Razin, Nadav Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/808a79d149c9dd8338d789881c9dab4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/808a79d149c9dd8338d789881c9dab4c-Abstract-Conference.html)

**Abstract**:

The question of what makes a data distribution suitable for deep learning is a fundamental open problem. Focusing on locally connected neural networks (a prevalent family of architectures that includes convolutional and recurrent neural networks as well as local self-attention models), we address this problem by adopting theoretical tools from quantum physics. Our main theoretical result states that a certain locally connected neural network is capable of accurate prediction over a data distribution if and only if the data distribution admits low quantum entanglement under certain canonical partitions of features. As a practical application of this result, we derive a preprocessing method for enhancing the suitability of a data distribution to locally connected neural networks. Experiments with widespread models over various datasets demonstrate our findings. We hope that our use of quantum entanglement will encourage further adoption of tools from physics for formally reasoning about the relation between deep learning and real-world data.

----

## [0] Parameter and Computation Efficient Transfer Learning for Vision-Language Pre-trained Models

**Authors**: *Qiong Wu, Wei Yu, Yiyi Zhou, Shubin Huang, Xiaoshuai Sun, Rongrong Ji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80e354fdac2c7fbf439a51f4853edbac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80e354fdac2c7fbf439a51f4853edbac-Abstract-Conference.html)

**Abstract**:

With ever increasing parameters and computation, vision-language pre-trained (VLP) models exhibit prohibitive expenditure in downstream task adaption. Recent endeavors mainly focus on parameter efficient transfer learning (PETL) for VLP models by only updating a small number of parameters. However, excessive computational overhead still plagues the application of VLPs. In this paper, we aim at parameter and computation efficient transfer learning (PCETL) for VLP models. In particular, PCETL not only needs to limit the number of trainable parameters in VLP models, but also to reduce the computational redundancy during inference, thus enabling a more efficient transfer. To approach this target, we propose a novel dynamic architecture skipping (DAS) approach towards effective PCETL. Instead of directly optimizing the intrinsic architectures of VLP models, DAS first observes the significances of their modules to downstream tasks via a reinforcement learning (RL) based process, and then skips the redundant ones with lightweight networks, i.e. adapters, according to the obtained rewards. In this case, the VLP model can well maintain the scale of trainable parameters while speeding up its inference on downstream tasks. To validate DAS, we apply it to two representative VLP models, namely ViLT and METER, and conduct extensive experiments on a bunch of VL tasks. The experimental results not only show the great advantages of DAS in reducing computational complexity, e.g. -11.97%  FLOPs of METER on VQA2.0, but also confirm its competitiveness against existing PETL methods in terms of parameter scale and performance. Our source code is given in our appendix.

----

## [0] A Dynamical System View of Langevin-Based Non-Convex Sampling

**Authors**: *Mohammad Reza Karimi Jaghargh, Ya-Ping Hsieh, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80f253dcb51cd2af7ce54e9379fb3521-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80f253dcb51cd2af7ce54e9379fb3521-Abstract-Conference.html)

**Abstract**:

Non-convex sampling is a key challenge in machine learning, central to non-convex optimization in deep learning as well as to approximate probabilistic inference. Despite its significance, theoretically there remain some important challenges: Existing guarantees suffer from the drawback of lacking guarantees for the last-iterates, and little is known beyond the elementary schemes of stochastic gradient Langevin dynamics. To address these issues, we develop a novel framework that lifts the above issues by harnessing several tools from the theory of dynamical systems. Our key result is that, for a large class of state-of-the-art sampling schemes, their last-iterate convergence in Wasserstein distances can be reduced to the study of their continuous-time counterparts, which is much better understood. Coupled with standard assumptions of MCMC sampling, our theory immediately yields the last-iterate Wasserstein convergence of many advanced sampling schemes such as mirror Langevin, proximal, randomized mid-point, and Runge-Kutta methods.

----

## [0] OKRidge: Scalable Optimal k-Sparse Ridge Regression

**Authors**: *Jiachang Liu, Sam Rosen, Chudi Zhong, Cynthia Rudin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80f48ffa8022773973a4a5cec7cce19c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80f48ffa8022773973a4a5cec7cce19c-Abstract-Conference.html)

**Abstract**:

We consider an important problem in scientific discovery, namely identifying sparse governing equations for nonlinear dynamical systems. This involves solving sparse ridge regression problems to provable optimality in order to determine which terms drive the underlying dynamics. We propose a fast algorithm, OKRidge, for sparse ridge regression, using a novel lower bound calculation involving, first, a saddle point formulation, and from there, either solving (i) a linear system or (ii) using an ADMM-based approach, where the proximal operators can be efficiently evaluated by solving another linear system and an isotonic regression problem. We also propose a method to warm-start our solver, which leverages a beam search. Experimentally, our methods attain provable optimality with run times that are orders of magnitude faster than those of the existing MIP formulations solved by the commercial solver Gurobi.

----

## [0] Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise

**Authors**: *Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/80fe51a7d8d0c73ff7439c2a2554ed53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/80fe51a7d8d0c73ff7439c2a2554ed53-Abstract-Conference.html)

**Abstract**:

Standard diffusion models involve an image transform  -- adding Gaussian noise -- and an image restoration operator that inverts this degradation.  We observe that the generative behavior of diffusion models is not strongly dependent on the choice of image degradation, and in fact, an entire family of generative models can be constructed by varying this choice. Even when using completely deterministic degradations (e.g., blur, masking, and more), the training and test-time update rules that underlie diffusion models can be easily generalized to create generative models. The success of these fully deterministic models calls into question the community's understanding of diffusion models, which relies on noise in either gradient Langevin dynamics or variational inference and paves the way for generalized diffusion models that invert arbitrary processes.

----

## [0] Towards the Difficulty for a Deep Neural Network to Learn Concepts of Different Complexities

**Authors**: *Dongrui Liu, Huiqi Deng, Xu Cheng, Qihan Ren, Kangrui Wang, Quanshi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8143b8c73073a9a23b9c18e400066471-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8143b8c73073a9a23b9c18e400066471-Abstract-Conference.html)

**Abstract**:

This paper theoretically explains the intuition that simple concepts are more likely to be learned by deep neural networks (DNNs) than complex concepts. In fact, recent studies have observed [24, 15] and proved [26] the emergence of interactive concepts in a DNN, i.e., it is proven that a DNN usually only encodes a small number of interactive concepts, and can be considered to use their interaction effects to compute inference scores. Each interactive concept is encoded by the DNN to represent the collaboration between a set of input variables. Therefore, in this study, we aim to theoretically explain that interactive concepts involving more input variables (i.e., more complex concepts) are more difficult to learn. Our finding clarifies the exact conceptual complexity that boosts the learning difficulty.

----

## [0] Limits, approximation and size transferability for GNNs on sparse graphs via graphops

**Authors**: *Thien Le, Stefanie Jegelka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8154c89c8d3612d39fd1ed6a20f4bab1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8154c89c8d3612d39fd1ed6a20f4bab1-Abstract-Conference.html)

**Abstract**:

Can graph neural networks generalize to graphs that are different from the graphs they were trained on, e.g., in size? In this work, we study this question from a theoretical perspective. While recent work established such transferability and approximation results via graph limits, e.g., via graphons, these only apply nontrivially to dense graphs. To include frequently encountered sparse graphs such as bounded-degree or power law graphs, we take a perspective of taking limits of operators derived from graphs, such as the aggregation operation that makes up GNNs. This leads to the recently introduced limit notion of graphops (Backhausz and Szegedy, 2022). We demonstrate how the operator perspective allows us to develop quantitative bounds on the distance between a finite GNN and its limit on an infinite graph, as well as the distance between the GNN on graphs of different sizes that share structural properties, under a regularity assumption verified for various graph sequences. Our results hold for dense and sparse graphs, and various notions of graph limits.

----

## [0] The Adversarial Consistency of Surrogate Risks for Binary Classification

**Authors**: *Natalie Frank, Jonathan Niles-Weed*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81858558b55a8c63763cfe088090242a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81858558b55a8c63763cfe088090242a-Abstract-Conference.html)

**Abstract**:

We study the consistency of surrogate risks for robust binary classification.It is common to learn robust classifiers by adversarial training, which seeks to minimize the expected $0$-$1$ loss when each example can be maliciously corrupted within a small ball.We give a simple and complete characterization of the set of surrogate loss functions that are \emph{consistent}, i.e., that can replace the $0$-$1$ loss without affecting the minimizing sequences of the original adversarial risk, for any data distribution.We also prove a quantitative version of adversarial consistency for the $\rho$-margin loss.Our results reveal that the class of adversarially consistent surrogates is substantially smaller than in the standard setting, where many common surrogates are known to be consistent.

----

## [0] The Cambridge Law Corpus: A Corpus for Legal AI Research

**Authors**: *Andreas Östling, Holli Sargeant, Huiyuan Xie, Ludwig Bull, Alexander Terenin, Leif Jonsson, Måns Magnusson, Felix Steffek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/819b8452be7d6af1351d4c4f9cbdbd9b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/819b8452be7d6af1351d4c4f9cbdbd9b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce the Cambridge Law Corpus (CLC), a corpus for legal AI research. It consists of over 250 000 court cases from the UK. Most cases are from the 21st century, but the corpus includes cases as old as the 16th century. This paper presents the first release of the corpus, containing the raw text and meta-data. Together with the corpus, we provide annotations on case outcomes for 638 cases, done by legal experts. Using our annotated data, we have trained and evaluated case outcome extraction with GPT-3, GPT-4 and RoBERTa models to provide benchmarks. We include an extensive legal and ethical discussion to address the potentially sensitive nature of this material. As a consequence, the corpus will only be released for research purposes under certain restrictions.

----

## [0] Large Language Models of Code Fail at Completing Code with Potential Bugs

**Authors**: *Tuan Dinh, Jinman Zhao, Samson Tan, Renato Negrinho, Leonard Lausen, Sheng Zha, George Karypis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/819cebb05f993840e8a52d7564c5c282-Abstract-Conference.html)

**Abstract**:

Large language models of code (Code-LLMs) have recently brought tremendous advances to code completion, a fundamental feature of programming assistance and code intelligence. However, most existing works ignore the possible presence of bugs in the code context for generation, which are inevitable in software development. Therefore, we introduce and study the buggy-code completion problem, inspired by the realistic scenario of real-time code suggestion where the code context contains potential bugs â€“ anti-patterns that can become bugs in the completed program. To systematically study the task, we introduce two datasets: one with synthetic bugs derived from semantics-altering operator changes (buggy-HumanEval) and one with realistic bugs derived from user submissions to coding problems (buggy-FixEval). We find that the presence of potential bugs significantly degrades the generation performance of the high-performing Code-LLMs. For instance, the passing rates of CODEGEN-2B-MONO on test cases of buggy-HumanEval drop more than 50% given a single potential bug in the context. Finally, we investigate several post-hoc methods for mitigating the adverse effect of potential bugs and find that there remains a large gap in post-mitigation performance.

----

## [0] Doubly-Robust Self-Training

**Authors**: *Banghua Zhu, Mingyu Ding, Philip L. Jacobson, Ming Wu, Wei Zhan, Michael I. Jordan, Jiantao Jiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/819f426947c27eb5067bb6fdbdde93dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/819f426947c27eb5067bb6fdbdde93dd-Abstract-Conference.html)

**Abstract**:

Self-training is a well-established technique in semi-supervised learning, which leverages unlabeled data by generating pseudo-labels and incorporating them with a limited labeled dataset for training. The effectiveness of self-training heavily relies on the accuracy of these pseudo-labels. In this paper, we introduce doubly-robust self-training, an innovative semi-supervised algorithm that provably balances between two extremes. When pseudo-labels are entirely incorrect, our method reduces to a training process solely using labeled data. Conversely, when pseudo-labels are completely accurate, our method transforms into a training process utilizing all pseudo-labeled data and labeled data, thus increasing the effective sample size. Through empirical evaluations on both the ImageNet dataset for image classification and the nuScenes autonomous driving dataset for 3D object detection, we demonstrate the superiority of the doubly-robust loss over the self-training baseline.

----

## [0] FairLISA: Fair User Modeling with Limited Sensitive Attributes Information

**Authors**: *Zheng Zhang, Qi Liu, Hao Jiang, Fei Wang, Yan Zhuang, Le Wu, Weibo Gao, Enhong Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81a12aed87eb9c75dfdf91ed99d5519d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81a12aed87eb9c75dfdf91ed99d5519d-Abstract-Conference.html)

**Abstract**:

User modeling techniques profile users' latent characteristics (e.g., preference) from their observed behaviors, and play a crucial role in decision-making. Unfortunately, traditional user models may unconsciously capture biases related to sensitive attributes (e.g., gender) from behavior data, even when this sensitive information is not explicitly provided. This can lead to unfair issues and discrimination against certain groups based on these sensitive attributes.  Recent studies have been proposed to improve fairness by explicitly decorrelating user modeling results and sensitive attributes. However, most existing approaches assume that fully sensitive attribute labels are available in the training set, which is unrealistic due to collection limitations like privacy concerns, and hence bear the limitation of performance. In this paper, we focus on a practical situation with limited sensitive data and propose a novel FairLISA framework, which can efficiently utilize data with known and unknown sensitive attributes to facilitate fair model training. We first propose a novel theoretical perspective to build the relationship between data with both known and unknown sensitive attributes with the fairness objective.  Then, based on this, we provide a general adversarial framework to effectively leverage the whole user data for fair user modeling. We conduct experiments on representative user modeling tasks including recommender system and cognitive diagnosis. The results demonstrate that our FairLISA can effectively improve fairness while retaining high accuracy in scenarios with different ratios of missing sensitive attributes.

----

## [0] Inference-Time Intervention: Eliciting Truthful Answers from a Language Model

**Authors**: *Kenneth Li, Oam Patel, Fernanda B. Viégas, Hanspeter Pfister, Martin Wattenberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81b8390039b7302c909cb769f8b6cd93-Abstract-Conference.html)

**Abstract**:

We introduce Inference-Time Intervention (ITI), a technique designed to enhance the "truthfulness" of large language models (LLMs). ITI operates by shifting model activations during inference, following a learned set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from $32.5\%$ to $65.1\%$. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples. Our findings suggest that LLMs may have an internal representation of the likelihood of something being true, even as they produce falsehoods on the surface.

----

## [0] Composable Coresets for Determinant Maximization: Greedy is Almost Optimal

**Authors**: *Siddharth Gollapudi, Sepideh Mahabadi, Varun Sivashankar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81c565e605161fcf25d08aa230431eba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81c565e605161fcf25d08aa230431eba-Abstract-Conference.html)

**Abstract**:

Given a set of $n$ vectors in $\mathbb{R}^d$, the goal of the \emph{determinant maximization} problem is to pick $k$ vectors with the maximum volume. Determinant maximization is the MAP-inference task for determinantal point processes (DPP) and has recently received considerable attention for modeling diversity. As most applications for the problem use large amounts of data, this problem has been studied in the relevant \textit{composable coreset} setting.In particular, [Indyk-Mahabadi-OveisGharan-Rezaei--SODA'20, ICML'19] showed that one can get composable coresets with optimal approximation factor of $\tilde O(k)^k$ for the problem, and that a local search algorithm achieves an almost optimal approximation guarantee of $O(k)^{2k}$.In this work, we show that the widely-used Greedy algorithm also provides composable coresets with an almost optimal approximation factor of $O(k)^{3k}$, which improves over the previously known guarantee of $C^{k^2}$, and supports the prior experimental results showing the practicality of the greedy algorithm as a coreset.Our main result follows by showing a local optimality property for Greedy:swapping a single point from the greedy solution with a vector that was not picked by the greedy algorithm can increase the volume by a factor of at most $(1+\sqrt{k})$. This is tight up to the additive constant $1$. Finally, our experiments show that the local optimality of the greedy algorithm is even lower than the theoretical bound on real data sets.

----

## [0] ProBio: A Protocol-guided Multimodal Dataset for Molecular Biology Lab

**Authors**: *Jieming Cui, Ziren Gong, Baoxiong Jia, Siyuan Huang, Zilong Zheng, Jianzhu Ma, Yixin Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81c7202dbd3cd3006b35a58a076195c0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/81c7202dbd3cd3006b35a58a076195c0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The challenge of replicating research results has posed a significant impediment to the field of molecular biology. The advent of modern intelligent systems has led to notable progress in various domains. Consequently, we embarked on an investigation of intelligent monitoring systems as a means of tackling the issue of the reproducibility crisis. Specifically, we first curate a comprehensive multimodal dataset, named ProBio, as an initial step towards this objective. This dataset comprises fine-grained hierarchical annotations intended for the purpose of studying activity understanding in BioLab. Next, we devise two challenging benchmarks, transparent solution tracking and multimodal action recognition, to emphasize the unique characteristics and difficulties associated with activity understanding in BioLab settings. Finally, we provide a thorough experimental evaluation of contemporary video understanding models and highlight their limitations in this specialized domain to identify potential avenues for future research. We hope \dataset with associated benchmarks may garner increased focus on modern AI techniques in the realm of molecular biology.

----

## [0] Spuriosity Rankings: Sorting Data to Measure and Mitigate Biases

**Authors**: *Mazda Moayeri, Wenxiao Wang, Sahil Singla, Soheil Feizi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/81cca94f16f20d5548c76c3344b27dea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/81cca94f16f20d5548c76c3344b27dea-Abstract-Conference.html)

**Abstract**:

We present a simple but effective method to measure and mitigate model biases caused by reliance on spurious cues. Instead of requiring costly changes to one's data or model training, our method better utilizes the data one already has by sorting them. Specifically, we rank images within their classes based on spuriosity (the degree to which common spurious cues are present), proxied via deep neural features of an interpretable network. With spuriosity rankings, it is easy to identify minority subpopulations (i.e. low spuriosity images) and assess model bias as the gap in accuracy between high and low spuriosity images. One can even efficiently remove a model's bias at little cost to accuracy by finetuning its classification head on low spuriosity images, resulting in fairer treatment of samples regardless of spuriosity. We demonstrate our method on ImageNet, annotating $5000$ class-feature dependencies ($630$ of which we find to be spurious) and generating a dataset of $325k$ soft segmentations for these features along the way. Having computed spuriosity rankings via the identified spurious neural features, we assess biases for $89$ diverse models and find that class-wise biases are highly correlated across models. Our results suggest that model bias due to spurious feature reliance is influenced far more by what the model is trained on than how it is trained.

----



[Go to the previous page](NIPS-2023-list1700.md)

[Go to the next page](NIPS-2023-list1702.md)

[Go to the catalog section](README.md)