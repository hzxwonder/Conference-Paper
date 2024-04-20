## [0] Efficient Online Clustering with Moving Costs

**Authors**: *Dimitrios Christou, Stratis Skoulakis, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4ff08be7b0105049ff3e0ce3d70658c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4ff08be7b0105049ff3e0ce3d70658c5-Abstract-Conference.html)

**Abstract**:

In this work we consider an online learning problem, called Online $k$-Clustering with Moving Costs, at which a learner maintains a set of $k$ facilities over $T$ rounds so as to minimize the connection cost of an adversarially selected sequence of clients. The learner is informed on the positions of the clients at each round $t$ only after its facility-selection and can use this information to update its decision in the next round. However, updating the facility positions comes with an additional moving cost based on the moving distance of the facilities. We present the first $\mathcal{O}(\log n)$-regret polynomial-time online learning algorithm guaranteeing that the overall cost (connection $+$ moving) is at most $\mathcal{O}(\log n)$ times the time-averaged connection cost of the best fixed solution. Our work improves on the recent result of (Fotakis et al., 2021) establishing $\mathcal{O}(k)$-regret guarantees only on the connection cost.

----

## [0] SEGA: Instructing Text-to-Image Models using Semantic Guidance

**Authors**: *Manuel Brack, Felix Friedrich, Dominik Hintersdorf, Lukas Struppek, Patrick Schramowski, Kristian Kersting*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4ff83037e8d97b2171b2d3e96cb8e677-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4ff83037e8d97b2171b2d3e96cb8e677-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models have recently received a lot of interest for their astonishing ability to produce high-fidelity images from text only. However, achieving one-shot generation that aligns with the user’s intent is nearly impossible, yet small changes to the input prompt often result in very different images. This leaves the user with little semantic control. To put the user in control, we show how to interact with the diffusion process to flexibly steer it along semantic directions. This semantic guidance (SEGA) generalizes to any generative architecture using classifier-free guidance. More importantly, it allows for subtle and extensive edits, composition and style changes, and optimizing the overall artistic conception. We demonstrate SEGA’s effectiveness on both latent and pixel-based diffusion models such as Stable Diffusion, Paella, and DeepFloyd-IF using a variety of tasks, thus providing strong evidence for its versatility and flexibility.

----

## [0] Learning Sample Difficulty from Pre-trained Models for Reliable Prediction

**Authors**: *Peng Cui, Dan Zhang, Zhijie Deng, Yinpeng Dong, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/50251f54848a433f3e47ae3b7cbded53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/50251f54848a433f3e47ae3b7cbded53-Abstract-Conference.html)

**Abstract**:

Large-scale pre-trained models have achieved remarkable success in many applications, but how to leverage them to improve the prediction reliability of downstream models is undesirably under-explored. Moreover, modern neural networks have been found to be poorly calibrated and make overconfident predictions regardless of inherent sample difficulty and data uncertainty. To address this issue, we propose to utilize large-scale pre-trained models to guide downstream model training with sample difficulty-aware entropy regularization. Pre-trained models that have been exposed to large-scale datasets and do not overfit the downstream training classes enable us to measure each training sample’s difficulty via feature-space Gaussian modeling and relative Mahalanobis distance computation. Importantly, by adaptively penalizing overconfident prediction based on the sample difficulty, we simultaneously improve accuracy and uncertainty calibration across challenging benchmarks (e.g., +0.55% ACC and −3.7% ECE on ImageNet1k using ResNet34), consistently surpassing competitive baselines for reliable prediction. The improved uncertainty estimate further improves selective classification (abstaining from erroneous predictions) and out-of-distribution detection.

----

## [0] Asynchronous Proportional Response Dynamics: Convergence in Markets with Adversarial Scheduling

**Authors**: *Yoav Kolumbus, Menahem Levy, Noam Nisan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5049acb0d5d976130388f3e8edcae183-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5049acb0d5d976130388f3e8edcae183-Abstract-Conference.html)

**Abstract**:

We study Proportional Response Dynamics (PRD) in linear Fisher markets, where participants act asynchronously. We model this scenario as a sequential process in which at each step, an adversary selects a subset of the players to update their bids, subject to liveness constraints. We show that if every bidder individually applies the PRD update rule whenever they are included in the group of bidders selected by the adversary, then, in the generic case, the entire dynamic converges to a competitive equilibrium of the market. Our proof technique reveals additional properties of linear Fisher markets, such as the uniqueness of the market equilibrium for generic parameters and the convergence of associated no swap regret dynamics and best response dynamics under certain conditions.

----

## [0] Seeing is not always believing: Benchmarking Human and Model Perception of AI-Generated Images

**Authors**: *Zeyu Lu, Di Huang, Lei Bai, Jingjing Qu, Chengyue Wu, Xihui Liu, Wanli Ouyang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/505df5ea30f630661074145149274af0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/505df5ea30f630661074145149274af0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Photos serve as a way for humans to record what they experience in their daily lives, and they are often regarded as trustworthy sources of information. However, there is a growing concern that the advancement of artificial intelligence (AI) technology may produce fake photos, which can create confusion and diminish trust in photographs. This study aims to comprehensively evaluate agents for distinguishing state-of-the-art AI-generated visual content. Our study benchmarks both human capability and cutting-edge fake image detection AI algorithms, using a newly collected large-scale fake image dataset Fake2M. In our human perception evaluation, titled HPBench, we discovered that humans struggle significantly to distinguish real photos from AI-generated ones, with a misclassification rate of 38.7\%. Along with this, we conduct the model capability of AI-Generated images detection evaluation MPBench and the top-performing model from MPBench achieves a 13\% failure rate under the same setting used in the human evaluation.We hope that our study can raise awareness of the potential risks of AI-generated images and facilitate further research to prevent the spread of false information. More information can refer to https://github.com/Inf-imagine/Sentry.

----

## [0] FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding

**Authors**: *Pengxiang Wu, Siman Wang, Kevin Dela Rosa, Derek Hao Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/506630e4a43bb9d64a49f98b9ba934e9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/506630e4a43bb9d64a49f98b9ba934e9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Image retrieval is a fundamental task in computer vision. Despite recent advances in this field, many techniques have been evaluated on a limited number of domains, with a small number of instance categories. Notably, most existing works only consider domains like 3D landmarks, making it difficult to generalize the conclusions made by these works to other domains, e.g., logo and other 2D flat objects. To bridge this gap, we introduce a new dataset for benchmarking visual search methods on flat images with diverse patterns. Our flat object retrieval benchmark (FORB) supplements the commonly adopted 3D object domain, and more importantly, it serves as a testbed for assessing the image embedding quality on out-of-distribution domains. In this benchmark we investigate the retrieval accuracy of representative methods in terms of candidate ranks, as well as matching score margin, a viewpoint which is largely ignored by many works. Our experiments not only highlight the challenges and rich heterogeneity of FORB, but also reveal the hidden properties of different retrieval strategies. The proposed benchmark is a growing project and we expect to expand in both quantity and variety of objects. The dataset and supporting codes are available at https://github.com/pxiangwu/FORB/.

----

## [0] Intra-Modal Proxy Learning for Zero-Shot Visual Categorization with CLIP

**Authors**: *Qi Qian, Yuanhong Xu, Juhua Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/50a057e9fe79ffa3f4120fb6fb88071a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/50a057e9fe79ffa3f4120fb6fb88071a-Abstract-Conference.html)

**Abstract**:

Vision-language pre-training methods, e.g., CLIP, demonstrate an impressive zero-shot performance on visual categorizations with the class proxy from the text embedding of the class name. However, the modality gap between the text and vision space can result in a sub-optimal performance. We theoretically show that the gap cannot be reduced sufficiently by minimizing the contrastive loss in CLIP and the optimal proxy for vision tasks may reside only in the vision space. Therefore, given unlabeled target vision data, we propose to learn the vision proxy directly with the help from the text proxy for zero-shot transfer. Moreover, according to our theoretical analysis, strategies are developed to further refine the pseudo label obtained by the text proxy to facilitate the intra-modal proxy learning (InMaP) for vision. Experiments on extensive downstream tasks confirm the effectiveness and efficiency of our proposal. Concretely, InMaP can obtain the vision proxy within one minute on a single GPU while improving the zero-shot accuracy from $77.02\%$ to $80.21\%$ on ImageNet with ViT-L/14@336 pre-trained by CLIP.

----

## [0] Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation

**Authors**: *Yilin Lyu, Liyuan Wang, Xingxing Zhang, Zicheng Sun, Hang Su, Jun Zhu, Liping Jing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/50ca96a1a9ebe0b5e5688a504feb6107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/50ca96a1a9ebe0b5e5688a504feb6107-Abstract-Conference.html)

**Abstract**:

Continual learning entails learning a sequence of tasks and balancing their knowledge appropriately. With limited access to old training samples, much of the current work in deep neural networks has focused on overcoming catastrophic forgetting of old tasks in gradient-based optimization. However, the normalization layers provide an exception, as they are updated interdependently by the gradient and statistics of currently observed training samples, which require specialized strategies to mitigate recency bias. In this work, we focus on the most popular Batch Normalization (BN) and provide an in-depth theoretical analysis of its sub-optimality in continual learning. Our analysis demonstrates the dilemma between balance and adaptation of BN statistics for incremental tasks, which potentially affects training stability and generalization. Targeting on these particular challenges, we propose Adaptive Balance of BN (AdaB$^2$N), which incorporates appropriately a Bayesian-based strategy to adapt task-wise contributions and a modified momentum to balance BN statistics, corresponding to the training and testing stages. By implementing BN in a continual learning fashion, our approach achieves significant performance gains across a wide range of benchmarks, particularly for the challenging yet realistic online scenarios (e.g., up to 7.68\%, 6.86\% and 4.26\% on Split CIFAR-10, Split CIFAR-100 and Split Mini-ImageNet, respectively). Our code is available at https://github.com/lvyilin/AdaB2N.

----

## [0] The Simplicity Bias in Multi-Task RNNs: Shared Attractors, Reuse of Dynamics, and Geometric Representation

**Authors**: *Elia Turner, Omri Barak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/50d6dbc809b0dc96f7f1090810537acc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/50d6dbc809b0dc96f7f1090810537acc-Abstract-Conference.html)

**Abstract**:

How does a single interconnected neural population perform multiple tasks, each with its own dynamical requirements? The relation between task requirements and neural dynamics in Recurrent Neural Networks (RNNs) has been investigated for single tasks. The forces shaping joint dynamics of multiple tasks, however, are largely unexplored. In this work, we first construct a systematic framework to study multiple tasks in RNNs, minimizing interference from input and output correlations with the hidden representation. This allows us to reveal how RNNs tend to share attractors and reuse dynamics, a tendency we define as the "simplicity bias".We find that RNNs develop attractors sequentially during training, preferentially reusing existing dynamics and opting for simple solutions when possible. This sequenced emergence and preferential reuse encapsulate the simplicity bias. Through concrete examples, we demonstrate that new attractors primarily emerge due to task demands or architectural constraints, illustrating a balance between simplicity bias and external factors.We examine the geometry of joint representations within a single attractor, by constructing a family of tasks from a set of functions. We show that the steepness of the associated functions controls their alignment within the attractor. This arrangement again highlights the simplicity bias, as points with similar input spacings undergo comparable transformations to reach the shared attractor.Our findings propose compelling applications. The geometry of shared attractors might allow us to infer the nature of unknown tasks. Furthermore, the simplicity bias implies that without specific incentives, modularity in RNNs may not spontaneously emerge, providing insights into the conditions required for network specialization.

----

## [0] Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal

**Authors**: *Leah Chrestien, Stefan Edelkamp, Antonín Komenda, Tomás Pevný*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/50ea4dbd1cff6bd3daef939eff10c092-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/50ea4dbd1cff6bd3daef939eff10c092-Abstract-Conference.html)

**Abstract**:

In imitation learning for planning, parameters of heuristic functions are optimized against a set of solved problem instances. This work revisits the necessary and sufficient conditions of strictly optimally efficient heuristics for forward search algorithms, mainly A* and greedy best-first search, which expand only states on the returned optimal path. It then proposes a family of loss functions based on ranking tailored for a given variant of the forward search algorithm. Furthermore, from a learning theory point of view, it discusses why optimizing cost-to-goal h* is unnecessarily difficult. The experimental comparison on a diverse set of problems unequivocally supports the derived theory.

----

## [0] Goal-Conditioned Predictive Coding for Offline Reinforcement Learning

**Authors**: *Zilai Zeng, Ce Zhang, Shijie Wang, Chen Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51053d7b8473df7d5a2165b2a8ee9629-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51053d7b8473df7d5a2165b2a8ee9629-Abstract-Conference.html)

**Abstract**:

Recent work has demonstrated the effectiveness of formulating decision making as supervised learning on offline-collected trajectories. Powerful sequence models, such as GPT or BERT, are often employed to encode the trajectories. However, the benefits of performing sequence modeling on trajectory data remain unclear. In this work, we investigate whether sequence modeling has the ability to condense trajectories into useful representations that enhance policy learning. We adopt a two-stage framework that first leverages sequence models to encode trajectory-level representations, and then learns a goal-conditioned policy employing the encoded representations as its input. This formulation allows us to consider many existing supervised offline RL methods as specific instances of our framework. Within this framework, we introduce Goal-Conditioned Predictive Coding (GCPC), a sequence modeling objective that yields powerful trajectory representations and leads to performant policies. Through extensive empirical evaluations on AntMaze, FrankaKitchen and Locomotion environments, we observe that sequence modeling can have a significant impact on challenging decision making tasks. Furthermore, we demonstrate that GCPC learns a goal-conditioned latent representation encoding the future trajectory, which enables competitive performance on all three benchmarks.

----

## [0] Exposing Attention Glitches with Flip-Flop Language Modeling

**Authors**: *Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, Cyril Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/510ad3018bbdc5b6e3b10646e2e35771-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/510ad3018bbdc5b6e3b10646e2e35771-Abstract-Conference.html)

**Abstract**:

Why do large language models sometimes output factual inaccuracies and exhibit erroneous reasoning? The brittleness of these models, particularly when executing long chains of reasoning, currently seems to be an inevitable price to pay for their advanced capabilities of coherently synthesizing knowledge, pragmatics, and abstract thought. Towards making sense of this fundamentally unsolved problem, this work identifies and analyzes the phenomenon of attention glitches, in which the Transformer architecture's inductive biases intermittently fail to capture robust reasoning. To isolate the issue, we introduce flip-flop language modeling (FFLM), a parametric family of synthetic benchmarks designed to probe the extrapolative behavior of neural language models. This simple generative task requires a model to copy binary symbols over long-range dependencies, ignoring the tokens in between. We find that Transformer FFLMs suffer from a long tail of sporadic reasoning errors, some of which we can eliminate using various regularization techniques. Our preliminary mechanistic analyses show why the remaining errors may be very difficult to diagnose and resolve. We hypothesize that attention glitches account for (some of) the closed-domain hallucinations in natural LLMs.

----

## [0] Information Design in Multi-Agent Reinforcement Learning

**Authors**: *Yue Lin, Wenhao Li, Hongyuan Zha, Baoxiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/511d7c4e61878cf08ece6351ea3c529e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/511d7c4e61878cf08ece6351ea3c529e-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) is inspired by the way human infants and animals learn from the environment. The setting is somewhat idealized because, in actual tasks, other agents in the environment have their own goals and behave adaptively to the ego agent. To thrive in those environments, the agent needs to influence other agents so their actions become more helpful and less harmful. Research in computational economics distills two ways to influence others directly: by providing tangible goods (mechanism design) and by providing information (information design). This work investigates information design problems for a group of RL agents. The main challenges are two-fold. One is the information provided will immediately affect the transition of the agent trajectories, which introduces additional non-stationarity. The other is the information can be ignored, so the sender must provide information that the receiver is willing to respect. We formulate the Markov signaling game, and develop the notions of signaling gradient and the extended obedience constraints that address these challenges. Our algorithm is efficient on various mixed-motive tasks and provides further insights into computational economics. Our code is publicly available at https://github.com/YueLin301/InformationDesignMARL.

----

## [0] Gaussian Mixture Solvers for Diffusion Models

**Authors**: *Hanzhong Guo, Cheng Lu, Fan Bao, Tianyu Pang, Shuicheng Yan, Chao Du, Chongxuan Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51373b6499708b6fcc38f1e8f8f5b376-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51373b6499708b6fcc38f1e8f8f5b376-Abstract-Conference.html)

**Abstract**:

Recently, diffusion models have achieved great success in generative tasks. Sampling from diffusion models is equivalent to solving the reverse diffusion stochastic differential equations (SDEs) or the corresponding probability flow ordinary differential equations (ODEs). In comparison, SDE-based solvers can generate samples of higher quality and are suited for image translation tasks like stroke-based synthesis. During inference, however, existing SDE-based solvers are severely constrained by the efficiency-effectiveness dilemma. Our investigation suggests that this is because the Gaussian assumption in the reverse transition kernel is frequently violated (even in the case of simple mixture data) given a limited number of discretization steps. To overcome this limitation, we introduce a novel class of SDE-based solvers called \emph{Gaussian Mixture Solvers (GMS)} for diffusion models. Our solver estimates the first three-order moments and optimizes the parameters of a Gaussian mixture transition kernel using generalized methods of moments in each step during sampling. Empirically, our solver outperforms numerous SDE-based solvers in terms of sample quality in image generation and stroke-based synthesis in various diffusion models, which validates the motivation and effectiveness of GMS. Our code is available at https://github.com/Guohanzhong/GMS.

----

## [0] Trade-off Between Efficiency and Consistency for Removal-based Explanations

**Authors**: *Yifan Zhang, Haowei He, Zhiquan Tan, Yang Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51484744337f4bf5fea0e4dd92ddab0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51484744337f4bf5fea0e4dd92ddab0b-Abstract-Conference.html)

**Abstract**:

In the current landscape of explanation methodologies, most predominant approaches, such as SHAP and LIME, employ removal-based techniques to evaluate the impact of individual features by simulating various scenarios with specific features omitted. Nonetheless, these methods primarily emphasize efficiency in the original context, often resulting in general inconsistencies. In this paper, we demonstrate that such inconsistency is an inherent aspect of these approaches by establishing the Impossible Trinity Theorem, which posits that interpretability, efficiency, and consistency cannot hold simultaneously. Recognizing that the attainment of an ideal explanation remains elusive, we propose the utilization of interpretation error as a metric to gauge inefficiencies and inconsistencies. To this end, we present two novel algorithms founded on the standard polynomial basis, aimed at minimizing interpretation error. Our empirical findings indicate that the proposed methods achieve a substantial reduction in interpretation error, up to 31.8 times lower when compared to alternative techniques.

----

## [0] QuACK: Accelerating Gradient-Based Quantum Optimization with Koopman Operator Learning

**Authors**: *Di Luo, Jiayu Shen, Rumen Dangovski, Marin Soljacic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5159aaee380391c366b27994ed225e4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5159aaee380391c366b27994ed225e4f-Abstract-Conference.html)

**Abstract**:

Quantum optimization, a key application of quantum computing, has traditionally been stymied by the linearly increasing complexity of gradient calculations with an increasing number of parameters. This work bridges the gap between Koopman operator theory, which has found utility in applications because it allows for a linear representation of nonlinear dynamical systems, and natural gradient methods in quantum optimization, leading to a significant acceleration of gradient-based quantum optimization. We present Quantum-circuit Alternating Controlled Koopman learning (QuACK), a novel framework that leverages an alternating algorithm for efficient prediction of gradient dynamics on quantum computers. We demonstrate QuACK's remarkable ability to accelerate gradient-based optimization across a range of applications in quantum optimization and machine learning. In fact, our empirical studies, spanning quantum chemistry, quantum condensed matter, quantum machine learning, and noisy environments, have shown accelerations of more than 200x speedup in the overparameterized regime, 10x speedup in the smooth regime, and 3x speedup in the non-smooth regime. With QuACK, we offer a robust advancement that harnesses the advantage of gradient-based quantum optimization for practical benefits.

----

## [0] Provably Robust Temporal Difference Learning for Heavy-Tailed Rewards

**Authors**: *Semih Cayci, Atilla Eryilmaz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/516ca2e9e7bffbb4027a25d9f8838bc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/516ca2e9e7bffbb4027a25d9f8838bc9-Abstract-Conference.html)

**Abstract**:

In a broad class of reinforcement learning applications, stochastic rewards have heavy-tailed distributions, which lead to infinite second-order moments for stochastic (semi)gradients in policy evaluation and direct policy optimization. In such instances, the existing RL methods may fail miserably due to frequent statistical outliers. In this work, we establish that temporal difference (TD) learning with a dynamic gradient clipping mechanism, and correspondingly operated natural actor-critic (NAC), can be provably robustified against heavy-tailed reward distributions. It is shown in the framework of linear function approximation that a favorable tradeoff between bias and variability of the stochastic gradients can be achieved with this dynamic gradient clipping mechanism. In particular, we prove that robust versions of TD learning achieve sample complexities of order $\mathcal{O}(\varepsilon^{-\frac{1}{p}})$ and $\mathcal{O}(\varepsilon^{-1-\frac{1}{p}})$ with and without the full-rank assumption on the feature matrix, respectively, under heavy-tailed rewards with finite moments of order $(1+p)$ for some $p\in(0,1]$, both in expectation and with high probability. We show that a robust variant of NAC based on Robust TD learning achieves $\tilde{\mathcal{O}}(\varepsilon^{-4-\frac{2}{p}})$ sample complexity. We corroborate our theoretical results with numerical experiments.

----

## [0] Train Faster, Perform Better: Modular Adaptive Training in Over-Parameterized Models

**Authors**: *Yubin Shi, Yixuan Chen, Mingzhi Dong, Xiaochen Yang, Dongsheng Li, Yujiang Wang, Robert P. Dick, Qin Lv, Yingying Zhao, Fan Yang, Tun Lu, Ning Gu, Li Shang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/516fd05dc408fd6d6374940a83930193-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/516fd05dc408fd6d6374940a83930193-Abstract-Conference.html)

**Abstract**:

Despite their prevalence in deep-learning communities, over-parameterized models convey high demands of computational costs for proper training. This work studies the fine-grained, modular-level learning dynamics of over-parameterized models to attain a more efficient and fruitful training strategy. Empirical evidence reveals that when scaling down into network modules, such as heads in self-attention models, we can observe varying learning patterns implicitly associated with each module's trainability. To describe such modular-level learning capabilities, we introduce a novel concept dubbed modular neural tangent kernel (mNTK), and we demonstrate that the quality of a module's learning is tightly associated with its mNTK's principal eigenvalue $\lambda_{\max}$. A large $\lambda_{\max}$  indicates that the module learns features with better convergence, while those miniature ones may impact generalization negatively. Inspired by the discovery, we propose a novel training strategy termed Modular Adaptive Training (MAT) to update those modules with their $\lambda_{\max}$ exceeding a dynamic threshold selectively, concentrating the model on learning common features and ignoring those inconsistent ones. Unlike most existing training schemes with a complete BP cycle across all network modules, MAT can significantly save computations by its partially-updating strategy and can further improve performance. Experiments show that MAT nearly halves the computational cost of model training and outperforms the accuracy of baselines.

----

## [0] Class-Distribution-Aware Pseudo-Labeling for Semi-Supervised Multi-Label Learning

**Authors**: *Ming-Kun Xie, Jiahao Xiao, Hao-Zhe Liu, Gang Niu, Masashi Sugiyama, Sheng-Jun Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5195825ee60d7efc1e42b7f3f3137040-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5195825ee60d7efc1e42b7f3f3137040-Abstract-Conference.html)

**Abstract**:

Pseudo-labeling has emerged as a popular and effective approach for utilizing unlabeled data. However, in the context of semi-supervised multi-label learning (SSMLL), conventional pseudo-labeling methods encounter difficulties when dealing with instances associated with multiple labels and an unknown label count. These limitations often result in the introduction of false positive labels or the neglect of true positive ones. To overcome these challenges, this paper proposes a novel solution called Class-Aware Pseudo-Labeling (CAP) that performs pseudo-labeling in a class-aware manner. The proposed approach introduces a regularized learning framework incorporating class-aware thresholds, which effectively control the assignment of positive and negative pseudo-labels for each class. Notably, even with a small proportion of labeled examples, our observations demonstrate that the estimated class distribution serves as a reliable approximation. Motivated by this finding, we develop a class-distribution-aware thresholding strategy to ensure the alignment of pseudo-label distribution with the true distribution. The correctness of the estimated class distribution is theoretically verified, and a generalization error bound is provided for our proposed method. Extensive experiments on multiple benchmark datasets confirm the efficacy of CAP in addressing the challenges of SSMLL problems.

----

## [0] Adaptive Data Analysis in a Balanced Adversarial Model

**Authors**: *Kobbi Nissim, Uri Stemmer, Eliad Tsfadia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51ba8a68f471d952af625d1faf55e6c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51ba8a68f471d952af625d1faf55e6c6-Abstract-Conference.html)

**Abstract**:

In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $\cal{D}$, andis required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $\cal{D}$.Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions. However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $\cal{D}$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $\cal{D}$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $\cal{D}$.We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but has no prior knowledge of the underlying distribution (and hence has no a priori advantage with respect to the mechanism). We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.

----

## [0] Accelerating Molecular Graph Neural Networks via Knowledge Distillation

**Authors**: *Filip Ekström Kelvinius, Dimitar Georgiev, Artur P. Toshev, Johannes Gasteiger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51ec452ca04d8ec7160e5bbaf76153f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51ec452ca04d8ec7160e5bbaf76153f6-Abstract-Conference.html)

**Abstract**:

Recent advances in graph neural networks (GNNs) have enabled more comprehensive modeling of molecules and molecular systems, thereby enhancing the precision of molecular property prediction and molecular simulations. Nonetheless, as the field has been progressing to bigger and more complex architectures, state-of-the-art GNNs have become largely prohibitive for many large-scale applications. In this paper, we explore the utility of knowledge distillation (KD) for accelerating molecular GNNs. To this end, we devise KD strategies that facilitate the distillation of hidden representations in directional and equivariant GNNs, and evaluate their performance on the regression task of energy and force prediction. We validate our protocols across different teacher-student configurations and datasets, and demonstrate that they can consistently boost the predictive accuracy of student models without any modifications to their architecture. Moreover, we conduct comprehensive optimization of various components of our framework, and investigate the potential of data augmentation to further enhance performance. All in all, we manage to close the gap in predictive accuracy between teacher and student models by as much as 96.7\% and 62.5\% for energy and force prediction respectively, while fully preserving the inference throughput of the more lightweight models.

----

## [0] No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models

**Authors**: *Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, Matt J. Kusner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51f3d6252706100325ddc435ba0ade0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51f3d6252706100325ddc435ba0ade0e-Abstract-Conference.html)

**Abstract**:

The computation necessary for training Transformer-based language models has skyrocketed in recent years.This trend has motivated research on efficient training algorithms designed to improve training, validation, and downstream performance faster than standard training. In this work, we revisit three categories of such algorithms: dynamic architectures (layer stacking, layer dropping), batch selection (selective backprop., RHO-loss), and efficient optimizers (Lion, Sophia). When pre-training BERT and T5 with a fixed computation budget using such methods, we find that their training, validation, and downstream gains vanish compared to a baseline with a fully-decayed learning rate. We define an evaluation protocol that enables computation to be done on arbitrary machines by mapping all computation time to a reference machine which we call reference system time. We discuss the limitations of our proposed protocol and release our code to encourage rigorous research in efficient training procedures: https://github.com/JeanKaddour/NoTrainNoGain.

----

## [0] Layer-Neighbor Sampling - Defusing Neighborhood Explosion in GNNs

**Authors**: *Muhammed Fatih Balin, Ümit V. Çatalyürek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51f9036d5e7ae822da8f6d4adda1fb39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51f9036d5e7ae822da8f6d4adda1fb39-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have received significant attention recently, but training them at a large scale remains a challenge.Mini-batch training coupled with sampling is used to alleviate this challenge.However, existing approaches either suffer from the neighborhood explosion phenomenon or have suboptimal performance. To address these issues, we propose a new sampling algorithm called LAyer-neighBOR sampling (LABOR). It is designed to be a direct replacement for Neighbor Sampling (NS) with the same fanout hyperparameter while sampling up to 7 times fewer vertices, without sacrificing quality.By design, the variance of the estimator of each vertex matches NS from the point of view of a single vertex.Moreover, under the same vertex sampling budget constraints, LABOR converges faster than existing layer sampling approaches and can use up to 112 times larger batch sizes compared to NS.

----

## [0] Undirected Probabilistic Model for Tensor Decomposition

**Authors**: *Zerui Tao, Toshihisa Tanaka, Qibin Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51f9d542dea8bed1f66c8add6ec23c69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51f9d542dea8bed1f66c8add6ec23c69-Abstract-Conference.html)

**Abstract**:

Tensor decompositions (TDs) serve as a powerful tool for analyzing multiway data. Traditional TDs incorporate prior knowledge about the data into the model, such as a directed generative process from latent factors to observations. In practice, selecting proper structural or distributional assumptions beforehand is crucial for obtaining a promising TD representation. However, since such prior knowledge is typically unavailable in real-world applications, choosing an appropriate TD model can be challenging. This paper aims to address this issue by introducing a flexible TD framework that discards the structural and distributional assumptions, in order to learn as much information from the data. Specifically, we construct a TD model that captures the joint probability of the data and latent tensor factors through a deep energy-based model (EBM). Neural networks are then employed to parameterize the joint energy function of tensor factors and tensor entries. The flexibility of EBM and neural networks enables the learning of underlying structures and distributions. In addition, by designing the energy function, our model unifies the learning process of different types of tensors, such as static tensors and dynamic tensors with time stamps. The resulting model presents a doubly intractable nature due to the presence of latent tensor factors and the unnormalized probability function. To efficiently train the model, we derive a variational upper bound of the conditional noise-contrastive estimation objective that learns the unnormalized joint probability by distinguishing data from conditional noises. We show advantages of our model on both synthetic and several real-world datasets.

----

## [0] Rethinking Tokenizer and Decoder in Masked Graph Modeling for Molecules

**Authors**: *Zhiyuan Liu, Yaorui Shi, An Zhang, Enzhi Zhang, Kenji Kawaguchi, Xiang Wang, Tat-Seng Chua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/51fd9a7d1706023cb9f8210cc6ac357c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/51fd9a7d1706023cb9f8210cc6ac357c-Abstract-Conference.html)

**Abstract**:

Masked graph modeling excels in the self-supervised representation learning of molecular graphs. Scrutinizing previous studies, we can reveal a common scheme consisting of three key components: (1) graph tokenizer, which breaks a molecular graph into smaller fragments (\ie subgraphs) and converts them into tokens; (2) graph masking, which corrupts the graph with masks; (3) graph autoencoder, which first applies an encoder on the masked graph to generate the representations, and then employs a decoder on the representations to recover the tokens of the original graph. However, the previous MGM studies focus extensively on graph masking and encoder, while there is limited understanding of tokenizer and decoder. To bridge the gap, we first summarize popular molecule tokenizers at the granularity of node, edge, motif, and Graph Neural Networks (GNNs), and then examine their roles as the MGM's reconstruction targets. Further, we explore the potential of adopting an expressive decoder in MGM. Our results show that a subgraph-level tokenizer and a sufficiently expressive decoder with remask decoding have a \yuan{large impact on the encoder's representation learning}. Finally, we propose a novel MGM method SimSGT, featuring a Simple GNN-based Tokenizer (SGT) and an effective decoding strategy. We empirically validate that our method outperforms the existing molecule self-supervised learning methods. Our codes and checkpoints are available at https://github.com/syr-cn/SimSGT.

----

## [0] Shared Adversarial Unlearning: Backdoor Mitigation by Unlearning Shared Adversarial Examples

**Authors**: *Shaokui Wei, Mingda Zhang, Hongyuan Zha, Baoyuan Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/520425a5a4c2fb7f7fc345078b188201-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/520425a5a4c2fb7f7fc345078b188201-Abstract-Conference.html)

**Abstract**:

Backdoor attacks are serious security threats to machine learning models where an adversary can inject poisoned samples into the training set, causing a backdoored model which predicts poisoned samples with particular triggers to particular target classes, while behaving normally on benign samples. In this paper, we explore the task of purifying a backdoored model using a small clean dataset. By establishing the connection between backdoor risk and adversarial risk, we derive a novel upper bound for backdoor risk, which mainly captures the risk on the shared adversarial examples (SAEs) between the backdoored model and the purified model. This upper bound further suggests a novel bi-level optimization problem for mitigating backdoor using adversarial training techniques. To solve it, we propose Shared Adversarial Unlearning (SAU). Specifically, SAU first generates SAEs, and then, unlearns the generated SAEs such that they are either correctly classified by the purified model and/or differently classified by the two models, such that the backdoor effect in the backdoored model will be mitigated in the purified model. Experiments on various benchmark datasets and network architectures show that our proposed method achieves state-of-the-art performance for backdoor defense. The code is available at https://github.com/SCLBD/BackdoorBench (PyTorch) and https://github.com/shawkui/MindTrojan (MindSpore).

----

## [0] Calibration by Distribution Matching: Trainable Kernel Calibration Metrics

**Authors**: *Charlie Marx, Sofian Zalouk, Stefano Ermon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/52493d82db00e73abb2858a5a5f28717-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/52493d82db00e73abb2858a5a5f28717-Abstract-Conference.html)

**Abstract**:

Calibration ensures that probabilistic forecasts meaningfully capture uncertainty by requiring that predicted probabilities align with empirical frequencies. However, many existing calibration methods are specialized for post-hoc recalibration, which can worsen the sharpness of forecasts. Drawing on the insight that calibration can be viewed as a distribution matching task, we introduce kernel-based calibration metrics that unify and generalize popular forms of calibration for both classification and regression. These metrics admit differentiable sample estimates, making it easy to incorporate a calibration objective into empirical risk minimization. Furthermore, we provide intuitive mechanisms to tailor calibration metrics to a decision task, and enforce accurate loss estimation and no regret decisions. Our empirical evaluation demonstrates that employing these metrics as regularizers enhances calibration, sharpness, and decision-making across a range of regression and classification tasks, outperforming methods relying solely on post-hoc recalibration.

----

## [0] Polynomially Over-Parameterized Convolutional Neural Networks Contain Structured Strong Winning Lottery Tickets

**Authors**: *Arthur C. W. da Cunha, Francesco D'Amore, Emanuele Natale*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/525338e0d98401a62950bc7c454eb83d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/525338e0d98401a62950bc7c454eb83d-Abstract-Conference.html)

**Abstract**:

The Strong Lottery Ticket Hypothesis (SLTH) states that randomly-initialised neural networks likely contain subnetworks that perform well without any training. Although unstructured pruning has been extensively studied in this context, its structured counterpart, which can deliver significant computational and memory efficiency gains, has been largely unexplored. One of the main reasons for this gap is the limitations of the underlying mathematical tools used in formal analyses of the SLTH.In this paper, we overcome these limitations: we leverage recent advances in the multidimensional generalisation of the Random Subset-Sum Problem and obtain a variant that admits the stochastic dependencies that arise when addressing structured pruning in the SLTH. We apply this result to prove, for a wide class of random Convolutional Neural Networks, the existence of structured subnetworks that can approximate any sufficiently smaller network.This result provides the first sub-exponential bound around the SLTH for structured pruning, opening up new avenues for further research on the hypothesis and contributing to the understanding of the role of over-parameterization in deep learning.

----

## [0] Robustifying Generalizable Implicit Shape Networks with a Tunable Non-Parametric Model

**Authors**: *Amine Ouasfi, Adnane Boukhayma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/525c95ffca1f57a10e3527d3584f3cf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/525c95ffca1f57a10e3527d3584f3cf1-Abstract-Conference.html)

**Abstract**:

Feedforward generalizable models for implicit shape reconstruction from unoriented point cloud present multiple advantages, including high performance and inference speed. However, they still suffer from generalization issues, ranging from underfitting the input point cloud, to misrepresenting samples outside of the training data distribution, or with toplogies unseen at training.  We propose here an efficient mechanism to remedy some of these limitations at test time. We combine the inter-shape data prior of the network with an intra-shape regularization prior of a Nystr√∂m Kernel Ridge Regression, that we further adapt by fitting its hyperprameters to the current shape. The resulting shape function defined in a shape specific Reproducing Kernel Hilbert Space benefits from desirable stability and efficiency properties and grants a shape adaptive expressiveness-robustness trade-off. We demonstrate the improvement obtained through our method  with respect to baselines and the state-of-the-art using synthetic and real data.

----

## [0] Segment Anything in 3D with NeRFs

**Authors**: *Jiazhong Cen, Zanwei Zhou, Jiemin Fang, Chen Yang, Wei Shen, Lingxi Xie, Dongsheng Jiang, Xiaopeng Zhang, Qi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/525d24400247f884c3419b0b7b1c4829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/525d24400247f884c3419b0b7b1c4829-Abstract-Conference.html)

**Abstract**:

Recently, the Segment Anything Model (SAM) emerged as a powerful vision foundation model which is capable to segment anything in 2D images. This paper aims to generalize SAM to segment 3D objects. Rather than replicating the data acquisition and annotation procedure which is costly in 3D, we design an efficient solution, leveraging the Neural Radiance Field (NeRF) as a cheap and off-the-shelf prior that connects multi-view 2D images to the 3D space. We refer to the proposed solution as SA3D, for Segment Anything in 3D. It is only required to provide a manual segmentation prompt (e.g., rough points) for the target object in a single view, which is used to generate its 2D mask in this view with SAM. Next, SA3D alternately performs mask inverse rendering and cross-view self-prompting across various views to iteratively complete the 3D mask of the target object constructed with voxel grids. The former projects the 2D mask obtained by SAM in the current view onto 3D mask with guidance of the density distribution learned by the NeRF; The latter extracts reliable prompts automatically as the input to SAM from the NeRF-rendered 2D mask in another view. We show in experiments that SA3D adapts to various scenes and achieves 3D segmentation within minutes. Our research offers a generic and efficient methodology to lift a 2D vision foundation model to 3D, as long as the 2D model can steadily address promptable segmentation across multiple views.

----

## [0] Every Parameter Matters: Ensuring the Convergence of Federated Learning with Dynamic Heterogeneous Models Reduction

**Authors**: *Hanhan Zhou, Tian Lan, Guru Venkataramani, Wenbo Ding*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/526356453b7301c9b29aa0533f62bdef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/526356453b7301c9b29aa0533f62bdef-Abstract-Conference.html)

**Abstract**:

Cross-device Federated Learning (FL) faces significant challenges where low-end clients that could potentially make unique contributions are excluded from training large models due to their resource bottlenecks. Recent research efforts have focused on model-heterogeneous FL, by extracting reduced-size models from the global model and applying them to local clients accordingly. Despite the empirical success, general theoretical guarantees of convergence on this method remain an open question. This paper presents a unifying framework for heterogeneous FL algorithms with online model extraction and provides a general convergence analysis for the first time. In particular, we prove that under certain sufficient conditions and for both IID and non-IID data, these algorithms converge to a stationary point of standard FL for general smooth cost functions. Moreover, we introduce the concept of minimum coverage index, together with model reduction noise, which will determine the convergence of heterogeneous federated learning, and therefore we advocate for a holistic approach that considers both factors to enhance the efficiency of heterogeneous federated learning.

----

## [0] Small batch deep reinforcement learning

**Authors**: *Johan S. Obando-Ceron, Marc G. Bellemare, Pablo Samuel Castro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/528388f1ad3a481249a97cbb698d2fe6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/528388f1ad3a481249a97cbb698d2fe6-Abstract-Conference.html)

**Abstract**:

In value-based deep reinforcement learning with replay memories, the batch size parameter specifies how many transitions to sample for each gradient update. Although critical to the learning process, this value is typically not adjusted when proposing new algorithms. In this work we present a broad empirical study that suggests reducing the batch size can result in a number of significant performance gains; this is surprising, as the general tendency when training neural networks is towards larger batch sizes for improved performance. We complement our experimental findings with a set of empirical analyses towards better understanding this phenomenon.

----

## [0] A Deep Instance Generative Framework for MILP Solvers Under Limited Data Availability

**Authors**: *Zijie Geng, Xijun Li, Jie Wang, Xiao Li, Yongdong Zhang, Feng Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5297e56ac65ba2bfa70ee9fc4818c042-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5297e56ac65ba2bfa70ee9fc4818c042-Abstract-Conference.html)

**Abstract**:

In the past few years, there has been an explosive surge in the use of machine learning (ML) techniques to address combinatorial optimization (CO) problems, especially mixed-integer linear programs (MILPs). Despite the achievements, the limited availability of real-world instances often leads to sub-optimal decisions and biased solver assessments, which motivates a suite of synthetic MILP instance generation techniques. However, existing methods either rely heavily on expert-designed formulations or struggle to capture the rich features of real-world instances. To tackle this problem, we propose G2MILP, the first deep generative framework for MILP instances. Specifically, G2MILP represents MILP instances as bipartite graphs, and applies a masked variational autoencoder to iteratively corrupt and replace parts of the original graphs to generate new ones. The appealing feature of G2MILP is that it can learn to generate novel and realistic MILP instances without prior expert-designed formulations, while preserving the structures and computational hardness of real-world datasets, simultaneously. Thus the generated instances can facilitate downstream tasks for enhancing MILP solvers under limited data availability. We design a suite of benchmarks to evaluate the quality of the generated MILP instances. Experiments demonstrate that our method can produce instances that closely resemble real-world datasets in terms of both structures and computational hardness. The deliverables are released at https://miralab-ustc.github.io/L2O-G2MILP.

----

## [0] WordScape: a Pipeline to extract multilingual, visually rich Documents with Layout Annotations from Web Crawl Data

**Authors**: *Maurice Weber, Carlo Siebenschuh, Rory Butler, Anton Alexandrov, Valdemar Thanner, Georgios Tsolakis, Haris Jabbar, Ian T. Foster, Bo Li, Rick Stevens, Ce Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/52c1ce1a0eaf61e8b6e3a899c1b9c61f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/52c1ce1a0eaf61e8b6e3a899c1b9c61f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce WordScape, a novel pipeline for the creation of cross-disciplinary, multilingual corpora comprising millions of pages with annotations for document layout detection. Relating visual and textual items on document pages has gained further significance with the advent of multimodal models. Various approaches proved effective for visual question answering or layout segmentation. However, the interplay of text, tables, and visuals remains challenging for a variety of document understanding tasks. In particular, many models fail to generalize well to diverse domains and new languages due to insufficient availability of training data. WordScape addresses these limitations. Our automatic annotation pipeline parses the Open XML structure of Word documents obtained from the web, jointly providing layout-annotated document images and their textual representations. In turn, WordScape offers unique properties as it (1) leverages the ubiquity of the Word file format on the internet, (2) is readily accessible through the Common Crawl web corpus, (3) is adaptive to domain-specific documents, and (4) offers culturally and linguistically diverse document pages with natural semantic structure and high-quality text. Together with the pipeline, we will additionally release 9.5M urls to word documents which can be processed using WordScape to create a dataset of over 40M pages. Finally, we investigate the quality of text and layout annotations extracted by WordScape, assess the impact on document understanding benchmarks, and demonstrate that manual labeling costs can be substantially reduced.

----

## [0] Multi-Swap k-Means++

**Authors**: *Lorenzo Beretta, Vincent Cohen-Addad, Silvio Lattanzi, Nikos Parotsidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/52d63f9e4b81f866bf69fb3c834aad47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/52d63f9e4b81f866bf69fb3c834aad47-Abstract-Conference.html)

**Abstract**:

The $k$-means++ algorithm of Arthur and Vassilvitskii (SODA 2007) is often the practitioners' choice algorithm for optimizing the popular $k$-means clustering objective and is known to give an $O(\log k)$-approximation in expectation. To obtain higher quality solutions, Lattanzi and Sohler (ICML 2019) proposed augmenting $k$-means++ with $O(k \log \log k)$ local-search steps obtained through the $k$-means++ sampling distribution to yield a $c$-approximation to the $k$-means clustering problem, where $c$ is a large absolute constant. Here we generalize and extend their local-search algorithm by considering larger and more sophisticated local-search neighborhoods hence allowing to  swap multiple centers at the same time. Our algorithm achieves a $9 + \varepsilon$ approximation ratio, which is the best possible for local search. Importantly we show that our algorithm is practical, namely easy to implement and fast enough to run on a variety of classic datasets, and outputs solutions of better cost.

----

## [0] A Unified Discretization Framework for Differential Equation Approach with Lyapunov Arguments for Convex Optimization

**Authors**: *Kansei Ushiyama, Shun Sato, Takayasu Matsuo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/52da50b1ef221e4b1793e3bf44dd973d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/52da50b1ef221e4b1793e3bf44dd973d-Abstract-Conference.html)

**Abstract**:

The differential equation (DE) approach for convex optimization, which relates optimization methods to specific continuous DEs with rate-revealing Lyapunov functionals, has gained increasing interest since the seminal paper by Su--Boyd--Cand√®s (2014).However, the approach still lacks a crucial component to make it truly useful: there is no general, consistent way to transition back to discrete optimization methods. Consequently, even if we derive insights from continuous DEs, we still need to perform individualized and tedious calculations for the analysis of each method.This paper aims to bridge this gap by introducing a new concept called  ``weak discrete gradient'' (wDG), which consolidates the conditions required for discrete versions of gradients in the DE approach arguments.We then define abstract optimization methods using wDG and provide abstract convergence theories that parallel those in continuous DEs.We demonstrate that many typical optimization methods and their convergence rates can be derived as special cases of this abstract theory.The proposed unified discretization framework for the differential equation approach to convex optimization provides an easy environment for developing new optimization methods and achieving competitive convergence rates with state-of-the-art methods, such as Nesterov's accelerated gradient.

----

## [0] SARAMIS: Simulation Assets for Robotic Assisted and Minimally Invasive Surgery

**Authors**: *Nina Montaña Brown, Shaheer U. Saeed, Ahmed Abdulaal, Thomas Dowrick, Yakup Kilic, Sophie Wilkinson, Jack Gao, Meghavi Mashar, Chloe He, Alkisti Stavropoulou, Emma Thomson, Zachary M. C. Baum, Simone Foti, Brian R. Davidson, Yipeng Hu, Matthew J. Clarkson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/52e78a95d8baa6d082fb2d0e9499b661-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/52e78a95d8baa6d082fb2d0e9499b661-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Minimally-invasive surgery (MIS) and robot-assisted minimally invasive (RAMIS) surgery offer well-documented benefits to patients such as reduced post-operative pain and shorter hospital stays.However, the automation of MIS and RAMIS through the use of AI has been slow due to difficulties in data acquisition and curation, partially caused by the ethical considerations of training, testing and deploying AI models in medical environments.We introduce \texttt{SARAMIS}, the first large-scale dataset of anatomically derived 3D rendering assets of the human abdominal anatomy.Using previously existing, open-source CT datasets of the human anatomy, we derive novel 3D meshes, tetrahedral volumes, textures and diffuse maps for over 104 different anatomical targets in the human body, representing the largest, open-source dataset of 3D rendering assets for synthetic simulation of vision tasks in MIS+RAMIS, increasing the availability of openly available 3D meshes in the literature by three orders of magnitude.We supplement our dataset with a series of GPU-enabled rendering environments, which can be used to generate datasets for realistic MIS/RAMIS tasks.Finally, we present an example of the use of \texttt{SARAMIS} assets for an autonomous navigation task in colonoscopy from CT abdomen-pelvis scans for the first time in the literature.\texttt{SARAMIS} is publically made available at https://github.com/NMontanaBrown/saramis/, with assets released under a CC-BY-NC-SA license.

----

## [0] Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator

**Authors**: *Hanzhuo Huang, Yufan Feng, Cheng Shi, Lan Xu, Jingyi Yu, Sibei Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/52f050499cf82fa8efb588e263f6f3a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/52f050499cf82fa8efb588e263f6f3a7-Abstract-Conference.html)

**Abstract**:

Text-to-video is a rapidly growing research area that aims to generate a semantic, identical, and temporal coherence sequence of frames that accurately align with the input text prompt. This study focuses on zero-shot text-to-video generation considering the data- and cost-efficient. To generate a semantic-coherent video, exhibiting a rich portrayal of temporal semantics such as the whole process of flower blooming rather than a set of ``moving images'', we propose a novel Free-Bloom pipeline that harnesses large language models (LLMs) as the director to generate a semantic-coherence prompt sequence, while pre-trained latent diffusion models (LDMs) as the animator to generate the high fidelity frames. Furthermore, to ensure temporal and identical coherence while maintaining semantic coherence, we propose a series of annotative modifications to adapting LDMs in the reverse process, including joint noise sampling, step-aware attention shift, and dual-path interpolation. Without any video data and training requirements, Free-Bloom generates vivid and high-quality videos, awe-inspiring in generating complex scenes with semantic meaningful frame sequences.  In addition, Free-Bloom is naturally compatible with LDMs-based extensions.

----

## [0] NeRF Revisited: Fixing Quadrature Instability in Volume Rendering

**Authors**: *Mikaela Angelina Uy, Kiyohiro Nakayama, Guandao Yang, Rahul Krishna Thomas, Leonidas J. Guibas, Ke Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5301c49207917c5c870131959971851c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5301c49207917c5c870131959971851c-Abstract-Conference.html)

**Abstract**:

Neural radiance fields (NeRF) rely on volume rendering to synthesize novel views. Volume rendering requires evaluating an integral along each ray, which is numerically approximated with a finite sum that corresponds to the exact integral along the ray under piecewise constant volume density. As a consequence, the rendered result is unstable w.r.t. the choice of samples along the ray, a phenomenon that we dub quadrature instability. We propose a mathematically principled solution by reformulating the sample-based rendering equation so that it corresponds to the exact integral under piecewise linear volume density. This simultaneously resolves multiple issues: conflicts between samples along different rays, imprecise hierarchical sampling, and non-differentiability of quantiles of ray termination distances w.r.t. model parameters. We demonstrate several benefits over the classical sample-based rendering equation, such as sharper textures, better geometric reconstruction, and stronger depth supervision. Our proposed formulation can be also be used as a drop-in replacement to the volume rendering equation of existing NeRF-based methods. Our project page can be found at pl-nerf.github.io.

----

## [0] Practical Sharpness-Aware Minimization Cannot Converge All the Way to Optima

**Authors**: *Dongkuk Si, Chulhee Yun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5305b7891e1098dd9773d35cd9333180-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5305b7891e1098dd9773d35cd9333180-Abstract-Conference.html)

**Abstract**:

Sharpness-Aware Minimization (SAM) is an optimizer that takes a descent step based on the gradient at a perturbation $y_t = x_t + \rho \frac{\nabla f(x_t)}{\lVert \nabla f(x_t) \rVert}$ of the current point $x_t$. Existing studies prove convergence of SAM for smooth functions, but they do so by assuming decaying perturbation size $\rho$ and/or no gradient normalization in $y_t$, which is detached from practice. To address this gap, we study deterministic/stochastic versions of SAM with practical configurations (i.e., constant $\rho$ and gradient normalization in $y_t$) and explore their convergence properties on smooth functions with (non)convexity assumptions.Perhaps surprisingly, in many scenarios, we find out that SAM has limited capability to converge to global minima or stationary points.For smooth strongly convex functions, we show that while deterministic SAM enjoys tight global convergence rates of $\tilde \Theta(\frac{1}{T^2})$, the convergence bound of stochastic SAM suffers an inevitable additive term $\mathcal O(\rho^2)$, indicating convergence only up to neighborhoods of optima.In fact, such $\mathcal O(\rho^2)$ factors arise for stochastic SAM in all the settings we consider, and also for deterministic SAM in nonconvex cases; importantly, we prove by examples that such terms are unavoidable.Our results highlight vastly different characteristics of SAM with vs. without decaying perturbation size or gradient normalization, and suggest that the intuitions gained from one version may not apply to the other.

----

## [0] Online Convex Optimization with Unbounded Memory

**Authors**: *Raunak Kumar, Sarah Dean, Robert Kleinberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/531230cfac80c65017ad0f85d3031edc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/531230cfac80c65017ad0f85d3031edc-Abstract-Conference.html)

**Abstract**:

Online convex optimization (OCO) is a widely used framework in online  learning. In each round, the learner chooses a decision in a convex set and an  adversary chooses a convex loss function, and then the learner suffers the  loss associated with their current decision. However, in many applications the  learner's loss depends not only on the current decision but on the entire  history of decisions until that point. The OCO framework and its existing  generalizations do not capture this, and they can only be applied to many  settings of interest after a long series of approximation arguments. They also  leave open the question of whether the dependence on memory is tight because  there are no non-trivial lower bounds. In this work we introduce a  generalization of the OCO framework, ``Online Convex Optimization with  Unbounded Memory'', that captures long-term dependence on past decisions. We  introduce the notion of $p$-effective memory capacity, $H_p$, that quantifies  the maximum influence of past decisions on present losses. We prove an  $O(\sqrt{H_p T})$ upper bound on the policy regret and a matching (worst-case)  lower bound. As a special case, we prove the first non-trivial lower bound for  OCO with finite memory~\citep{anavaHM2015online}, which could be of  independent interest, and also improve existing upper bounds. We demonstrate  the broad applicability of our framework by using it to derive regret bounds,  and to improve and simplify existing regret bound derivations, for a variety  of online learning problems including online linear control and an online  variant of performative prediction.

----

## [0] Making Scalable Meta Learning Practical

**Authors**: *Sang Keun Choe, Sanket Vaibhav Mehta, Hwijeen Ahn, Willie Neiswanger, Pengtao Xie, Emma Strubell, Eric Xing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/531998dc1fc858b5857a90b74d96ecab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/531998dc1fc858b5857a90b74d96ecab-Abstract-Conference.html)

**Abstract**:

Despite its flexibility to learn diverse inductive biases in machine learning programs, meta learning (i.e.,\ learning to learn) has long been recognized to suffer from poor scalability due to its tremendous compute/memory costs, training instability, and a lack of efficient distributed training support. In this work, we focus on making scalable meta learning practical by introducing SAMA, which combines advances in both implicit differentiation algorithms and systems. Specifically, SAMA is designed to flexibly support a broad range of adaptive optimizers in the base level of meta learning programs, while reducing computational burden by avoiding explicit computation of second-order gradient information, and exploiting efficient distributed training techniques implemented for first-order gradients. Evaluated on multiple large-scale meta learning benchmarks, SAMA showcases up to 1.7/4.8x increase in throughput and 2.0/3.8x decrease in memory consumption respectively on single-/multi-GPU setups compared to other baseline meta learning algorithms. Furthermore, we show that SAMA-based data optimization leads to consistent improvements in text classification accuracy with BERT and RoBERTa large language models, and achieves state-of-the-art results in both small- and large-scale data pruning on image classification tasks, demonstrating the practical applicability of scalable meta learning across language and vision domains.

----

## [0] Dynamic Prompt Learning: Addressing Cross-Attention Leakage for Text-Based Image Editing

**Authors**: *Kai Wang, Fei Yang, Shiqi Yang, Muhammad Atif Butt, Joost van de Weijer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5321b1dabcd2be188d796c21b733e8c7-Abstract-Conference.html)

**Abstract**:

Large-scale text-to-image generative models have been a ground-breaking development in generative AI, with diffusion models showing their astounding ability to synthesize convincing images following an input text prompt. The goal of image editing research is to give users control over the generated images by modifying the text prompt. Current image editing techniques are susceptible to unintended modifications of regions outside the targeted area, such as on the background or on distractor objects which have some semantic or visual relationship with the targeted object. According to our experimental findings, inaccurate cross-attention maps are at the root of this problem. Based on this observation, we propose $\textit{Dynamic Prompt Learning}$ ($DPL$) to force cross-attention maps to focus on correct $\textit{noun}$ words in the text prompt. By updating the dynamic tokens for nouns in the textual input with the proposed leakage repairment losses, we achieve fine-grained image editing over particular objects while preventing undesired changes to other image regions. Our method $DPL$, based on the publicly available $\textit{Stable Diffusion}$, is extensively evaluated on a wide range of images, and consistently obtains superior results both quantitatively (CLIP score, Structure-Dist) and qualitatively (on user-evaluation). We show improved prompt editing results for Word-Swap, Prompt Refinement, and Attention Re-weighting, especially for complex multi-object scenes.

----

## [0] Brant: Foundation Model for Intracranial Neural Signal

**Authors**: *Daoze Zhang, Zhizhang Yuan, Yang Yang, Junru Chen, Jingjing Wang, Yafeng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/535915d26859036410b0533804cee788-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/535915d26859036410b0533804cee788-Abstract-Conference.html)

**Abstract**:

We propose a foundation model named Brant for modeling intracranial recordings, which learns powerful representations of intracranial neural signals by pre-training, providing a large-scale, off-the-shelf model for medicine. Brant is the largest model in the field of brain signals and is pre-trained on a large corpus of intracranial data collected by us. The design of Brant is to capture long-term temporal dependency and spatial correlation from neural signals, combining the information in both time and frequency domains. As a foundation model, Brant achieves SOTA performance on various downstream tasks (i.e. neural signal forecasting, frequency-phase forecasting, imputation and seizure detection), showing the generalization ability to a broad range of tasks. The low-resource label analysis and representation visualization further illustrate the effectiveness of our pre-training strategy. In addition, we explore the effect of model size to show that a larger model with a higher capacity can lead to performance improvements on our dataset. The source code and pre-trained weights are available at: https://zju-brainnet.github.io/Brant.github.io/.

----

## [0] Wasserstein distributional robustness of neural networks

**Authors**: *Xingjian Bai, Guangyi He, Yifan Jiang, Jan Oblój*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/53be3798fcc46e68ca0819c29a004652-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/53be3798fcc46e68ca0819c29a004652-Abstract-Conference.html)

**Abstract**:

Deep neural networks are known to be vulnerable to adversarial attacks (AA). For an image recognition task, this means that a small perturbation of the original can result in the image being misclassified. Design of such attacks as well as methods of adversarial training against them are subject of intense research. We re-cast the problem using techniques of Wasserstein distributionally robust optimization (DRO) and obtain novel contributions leveraging recent insights from DRO sensitivity analysis. We consider a set of distributional threat models. Unlike the traditional pointwise attacks, which assume a uniform bound on perturbation of each input data point, distributional threat models allow attackers to perturb inputs in a non-uniform way. We link these more general attacks with questions of out-of-sample performance and Knightian uncertainty. To evaluate the distributional robustness of neural networks, we propose a first-order AA algorithm and its multistep version. Our attack algorithms include Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) as special cases. Furthermore, we provide a new asymptotic estimate of the adversarial accuracy against distributional threat models. The bound is fast to compute and first-order accurate, offering new insights even for the pointwise AA. It also naturally yields out-of-sample performance guarantees. We conduct numerical experiments on CIFAR-10, CIFAR-100, ImageNet datasets using DNNs on RobustBench to illustrate our theoretical results. Our code is available at https://github.com/JanObloj/W-DRO-Adversarial-Methods.

----

## [0] High dimensional, tabular deep learning with an auxiliary knowledge graph

**Authors**: *Camilo Ruiz, Hongyu Ren, Kexin Huang, Jure Leskovec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/53dd219b6b11abc8ce523921c18c7a3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/53dd219b6b11abc8ce523921c18c7a3e-Abstract-Conference.html)

**Abstract**:

Machine learning models exhibit strong performance on datasets with abundant labeled samples. However, for tabular datasets with extremely high $d$-dimensional features but limited $n$ samples (i.e. $d \gg n$), machine learning models struggle to achieve strong performance due to the risk of overfitting. Here, our key insight is that there is often abundant, auxiliary domain information describing input features which can be structured as a heterogeneous knowledge graph (KG). We propose PLATO, a method that achieves strong performance on tabular data with $d \gg n$ by using an auxiliary KG describing input features to regularize a multilayer perceptron (MLP). In PLATO, each input feature corresponds to a node in the auxiliary KG. In the MLPâ€™s first layer, each input feature also corresponds to a weight vector. PLATO is based on the inductive bias that two input features corresponding to similar nodes in the auxiliary KG should have similar weight vectors in the MLP's first layer. PLATO captures this inductive bias by inferring the weight vector for each input feature from its corresponding node in the KG via a trainable message-passing function. Across 6 $d \gg n$ datasets, PLATO outperforms 13 state-of-the-art baselines by up to 10.19%.

----

## [0] Learning Space-Time Continuous Latent Neural PDEs from Partially Observed States

**Authors**: *Valerii Iakovlev, Markus Heinonen, Harri Lähdesmäki*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/53e9b4152ca09d5f1228157e752651dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/53e9b4152ca09d5f1228157e752651dd-Abstract-Conference.html)

**Abstract**:

We introduce a novel grid-independent model for learning partial differential equations (PDEs) from noisy and partial observations on irregular spatiotemporal grids. We propose a space-time continuous latent neural PDE model with an efficient probabilistic framework and a novel encoder design for improved data efficiency and grid independence. The latent state dynamics are governed by a PDE model that combines the collocation method and the method of lines. We employ amortized variational inference for approximate posterior estimation and utilize a multiple shooting technique for enhanced training speed and stability. Our model demonstrates state-of-the-art performance on complex synthetic and real-world datasets, overcoming limitations of previous approaches and effectively handling partially-observed data. The proposed model outperforms recent methods, showing its potential to advance data-driven PDE modeling and enabling robust, grid-independent modeling of complex partially-observed dynamic processes across various domains.

----

## [0] Adaptive SGD with Polyak stepsize and Line-search: Robust Convergence and Variance Reduction

**Authors**: *Xiaowen Jiang, Sebastian U. Stich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/540eb9e0ee35d525231c3fd22d1dcbf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/540eb9e0ee35d525231c3fd22d1dcbf2-Abstract-Conference.html)

**Abstract**:

The recently proposed stochastic Polyak stepsize (SPS) and stochastic line-search (SLS) for SGD have shown remarkable effectiveness when training over-parameterized models. However, two issues remain unsolved in this line of work. First, in non-interpolation settings, both algorithms only guarantee convergence to a neighborhood of a solution which may result in a worse output than the initial guess. While artificially decreasing the adaptive stepsize has been proposed to address this issue (Orvieto et al.), this approach results in slower convergence rates under interpolation. Second, intuitive line-search methods equipped with variance-reduction (VR) fail to converge (Dubois-Taine et al.). So far, no VR methods successfully accelerate these two stepsizes with a convergence guarantee.In this work, we make two contributions:Firstly, we propose two new robust variants of SPS and SLS, called AdaSPS and AdaSLS, which achieve optimal asymptotic rates in both strongly-convex or convex and interpolation or non-interpolation settings, except for the case when we have both strong convexity and non-interpolation. AdaSLS requires no knowledge of problem-dependent parameters, and AdaSPS requires only a lower bound of the optimal function value as input. Secondly, we propose a novel VR method that can use Polyak stepsizes or line-search to achieve acceleration. When it is equipped with AdaSPS or AdaSLS, the resulting algorithms obtain the optimal ratefor optimizing convex smooth functions. Finally, numerical experiments on synthetic and real datasets validate our theory and demonstrate the effectiveness and robustness of our algorithms.

----

## [0] SOC: Semantic-Assisted Object Cluster for Referring Video Object Segmentation

**Authors**: *Zhuoyan Luo, Yicheng Xiao, Yong Liu, Shuyan Li, Yitong Wang, Yansong Tang, Xiu Li, Yujiu Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/542c14ff4622e45384df40dc97b9cf90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/542c14ff4622e45384df40dc97b9cf90-Abstract-Conference.html)

**Abstract**:

This paper studies referring video object segmentation (RVOS) by boosting video-level visual-linguistic alignment. Recent approaches model the RVOS task as a sequence prediction problem and perform multi-modal interaction as well as segmentation for each frame separately. However, the lack of a global view of video content leads to difficulties in effectively utilizing inter-frame relationships and understanding textual descriptions of object temporal variations. To address this issue, we propose Semantic-assisted Object Cluster (SOC), which aggregates video content and textual guidance for unified temporal modeling and cross-modal alignment. By associating a group of frame-level object embeddings with language tokens, SOC facilitates joint space learning across modalities and time steps. Moreover, we present multi-modal contrastive supervision to help construct well-aligned joint space at the video level. We conduct extensive experiments on popular RVOS benchmarks, and our method outperforms state-of-the-art competitors on all benchmarks by a remarkable margin. Besides, the emphasis on temporal coherence enhances the segmentation stability and adaptability of our method in processing text expressions with temporal variations. Code is available at https://github.com/RobertLuo1/NeurIPS2023_SOC.

----

## [0] Posthoc privacy guarantees for collaborative inference with modified Propose-Test-Release

**Authors**: *Abhishek Singh, Praneeth Vepakomma, Vivek Sharma, Ramesh Raskar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5433b79562b9fa85bd5da0c95a78c907-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5433b79562b9fa85bd5da0c95a78c907-Abstract-Conference.html)

**Abstract**:

Cloud-based machine learning inference is an emerging paradigm where users query by sending their data through a service provider who runs an ML model on that data and returns back the answer. Due to increased concerns over data privacy, recent works have proposed Collaborative Inference (CI) to learn a privacy-preserving encoding of sensitive user data before it is shared with an untrusted service provider. Existing works so far evaluate the privacy of these encodings through empirical reconstruction attacks. In this work, we develop a new framework that provides formal privacy guarantees for an arbitrarily trained neural network by linking its local Lipschitz constant with its local sensitivity. To guarantee privacy using local sensitivity, we extend the Propose-Test-Release (PTR) framework to make it tractable for neural network queries. We verify the efficacy of our framework experimentally on real-world datasets and elucidate the role of Adversarial Representation Learning (ARL) in improving the privacy-utility trade-off.

----

## [0] Strategic Classification under Unknown Personalized Manipulation

**Authors**: *Han Shao, Avrim Blum, Omar Montasser*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/543924fdf260ba990f2ef84f940f3db2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/543924fdf260ba990f2ef84f940f3db2-Abstract-Conference.html)

**Abstract**:

We study the fundamental mistake bound and sample complexity in the strategic classification, where agents can strategically manipulate their feature vector up to an extent in order to be predicted as positive. For example, given a classifier determining college admission, student candidates may try to take easier classes to improve their GPA, retake SAT and change schools in an effort to fool the classifier. *Ball manipulations* are a widely studied class of manipulations in the literature, where agents can modify their feature vector within a bounded radius ball. Unlike most prior work, our work consider manipulations to be *personalized*, meaning that agents can have different levels of manipulation abilities (e.g., varying radii for ball manipulations), and *unknown* to the learner.We formalize the learning problem in an interaction model where the learner first deploys a classifier and the agent manipulates the feature vector within their manipulation set to game the deployed classifier. We investigate various scenarios in terms of the information available to the learner during the interaction, such as observing the original feature vector before or after deployment, observing the manipulated feature vector, or not seeing either the original or the manipulated feature vector. We begin by providing online mistake bounds and PAC sample complexity in these scenarios for ball manipulations. We also explore non-ball manipulations and show that, even in the simplest scenario where both the original and the manipulated feature vectors are revealed, the mistake bounds and sample complexity are lower bounded by $\Omega(|\mathcal H|)$ when the target function belongs to a known class $\mathcal H$.

----

## [0] EPIC Fields: Marrying 3D Geometry and Video Understanding

**Authors**: *Vadim Tschernezki, Ahmad Darkhalil, Zhifan Zhu, David Fouhey, Iro Laina, Diane Larlus, Dima Damen, Andrea Vedaldi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/543d4e171150cb931f1d401cacc3d7af-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/543d4e171150cb931f1d401cacc3d7af-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Neural rendering is fuelling a unification of learning, 3D geometry and video understanding that has been waiting for more than two decades. Progress, however, is still hampered by a lack of suitable datasets and benchmarks. To address this gap, we introduce EPIC Fields, an augmentation of EPIC-KITCHENS with 3D camera information. Like other datasets for neural rendering, EPIC Fields removes the complex and expensive step of reconstructing cameras using photogrammetry, and allows researchers to focus on modelling problems. We illustrate the challenge of photogrammetry in egocentric videos of dynamic actions and propose innovations to address them. Compared to other neural rendering datasets, EPIC Fields is better tailored to video understanding because it is paired with labelled action segments and the recent VISOR segment annotations. To further motivate the community, we also evaluate two benchmark tasks in neural rendering and segmenting dynamic objects, with strong baselines that showcase what is not possible today. We also highlight the advantage of geometry in semi-supervised video object segmentations on the VISOR annotations. EPIC Fields reconstructs 96\% of videos in EPIC-KITCHENS, registering 19M frames in 99 hours recorded in 45 kitchens, and is available from: http://epic-kitchens.github.io/epic-fields

----

## [0] On the Ability of Graph Neural Networks to Model Interactions Between Vertices

**Authors**: *Noam Razin, Tom Verbin, Nadav Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/543ec10715d964122ab7cb15f648772b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/543ec10715d964122ab7cb15f648772b-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are widely used for modeling complex interactions between entities represented as vertices of a graph. Despite recent efforts to theoretically analyze the expressive power of GNNs, a formal characterization of their ability to model interactions is lacking. The current paper aims to address this gap. Formalizing strength of interactions through an established measure known as separation rank, we quantify the ability of certain GNNs to model interaction between a given subset of vertices and its complement, i.e. between the sides of a given partition of input vertices. Our results reveal that the ability to model interaction is primarily determined by the partition's walk index --- a graph-theoretical characteristic defined by the number of walks originating from the boundary of the partition. Experiments with common GNN architectures corroborate this finding. As a practical application of our theory, we design an edge sparsification algorithm named Walk Index Sparsification (WIS), which preserves the ability of a GNN to model interactions when input edges are removed. WIS is simple, computationally efficient, and in our experiments has markedly outperformed alternative methods in terms of induced prediction accuracy. More broadly, it showcases the potential of improving GNNs by theoretically analyzing the interactions they can model.

----

## [0] Learning from Both Structural and Textual Knowledge for Inductive Knowledge Graph Completion

**Authors**: *Kunxun Qi, Jianfeng Du, Hai Wan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/544242770e8333875325d013328b2079-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/544242770e8333875325d013328b2079-Abstract-Conference.html)

**Abstract**:

Learning rule-based systems plays a pivotal role in knowledge graph completion (KGC). Existing rule-based systems restrict the input of the system to structural knowledge only, which may omit some useful knowledge for reasoning, e.g., textual knowledge. In this paper, we propose a two-stage framework that imposes both structural and textual knowledge to learn rule-based systems. In the first stage, we compute a set of triples with confidence scores (called \emph{soft triples}) from a text corpus by distant supervision, where a textual entailment model with multi-instance learning is exploited to estimate whether a given triple is entailed by a set of sentences. In the second stage, these soft triples are used to learn a rule-based model for KGC. To mitigate the negative impact of noise from soft triples, we propose a new formalism for rules to be learnt, named \emph{text enhanced rules} or \emph{TE-rules} for short. To effectively learn TE-rules, we propose a neural model that simulates the inference of TE-rules. We theoretically show that any set of TE-rules can always be interpreted by a certain parameter assignment of the neural model. We introduce three new datasets to evaluate the effectiveness of our method. Experimental results demonstrate that the introduction of soft triples and TE-rules results in significant performance improvements in inductive link prediction.

----

## [0] Sorting with Predictions

**Authors**: *Xingjian Bai, Christian Coester*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/544696ef4847c903376ed6ec58f3a703-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/544696ef4847c903376ed6ec58f3a703-Abstract-Conference.html)

**Abstract**:

We explore the fundamental problem of sorting through the lens of learning-augmented algorithms, where algorithms can leverage possibly erroneous predictions to improve their efficiency. We consider two different settings: In the first setting, each item is provided a prediction of its position in the sorted list. In the second setting, we assume there is a ``quick-and-dirty'' way of comparing items, in addition to slow-and-exact comparisons. For both settings, we design new and simple algorithms using only $O(\sum_i \log \eta_i)$ exact comparisons, where $\eta_i$ is a suitably defined prediction error for the $i$th element. In particular, as the quality of predictions deteriorates, the number of comparisons degrades smoothly from $O(n)$ to $O(n\log n)$. We prove that this comparison complexity is theoretically optimal with respect to the examined error measures. An experimental evaluation against existing adaptive and non-adaptive sorting algorithms demonstrates the potential of applying learning-augmented algorithms in sorting tasks.

----

## [0] Posterior Sampling for Competitive RL: Function Approximation and Partial Observation

**Authors**: *Shuang Qiu, Ziyu Dai, Han Zhong, Zhaoran Wang, Zhuoran Yang, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/545a674417b8c4bcae96eceffad1c4f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/545a674417b8c4bcae96eceffad1c4f0-Abstract-Conference.html)

**Abstract**:

This paper investigates posterior sampling algorithms for competitive reinforcement learning (RL) in the context of general function approximations. Focusing on zero-sum Markov games (MGs) under two critical settings, namely self-play and adversarial learning, we first propose the self-play and adversarial generalized eluder coefficient (GEC) as complexity measures for function approximation, capturing the exploration-exploitation trade-off in MGs. Based on self-play GEC, we propose a model-based self-play posterior sampling method to control both players to learn Nash equilibrium, which can successfully handle the partial observability of states. Furthermore, we identify a set of partially observable MG models fitting MG learning with the adversarial policies of the opponent. Incorporating the adversarial GEC, we propose a model-based posterior sampling method for learning adversarial MG with potential partial observability. We further provide low regret bounds for proposed algorithms that can scale sublinearly with the proposed GEC and the number of episodes $T$. To the best of our knowledge, we for the first time develop generic model-based posterior sampling algorithms for competitive RL that can be applied to a majority of tractable zero-sum MG classes in both fully observable and partially observable MGs with self-play and adversarial learning.

----

## [0] Towards Test-Time Refusals via Concept Negation

**Authors**: *Peiran Dong, Song Guo, Junxiao Wang, Bingjie Wang, Jiewei Zhang, Ziming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54801e196796134a2b0ae5e8adef502f-Abstract-Conference.html)

**Abstract**:

Generative models produce unbounded outputs, necessitating the use of refusal techniques to confine their output space. Employing generative refusals is crucial in upholding the ethical and copyright integrity of synthesized content, particularly when working with widely adopted diffusion models. "Concept negation'' presents a promising paradigm to achieve generative refusals, as it effectively defines and governs the model's output space based on concepts, utilizing natural language interfaces that are readily comprehensible to humans. However, despite the valuable contributions of prior research to the field of concept negation, it still suffers from significant limitations. The existing concept negation methods, which operate based on the composition of score or noise predictions from the diffusion process, are limited to independent concepts (e.g., ``a blonde girl`` without ``glasses``) and fail to consider the interconnected nature of concepts in reality (e.g., ``Mickey mouse eats ice cream`` without ``Disney characters``). Keeping the limitations in mind, we propose a novel framework, called $ProtoRe$, to improve the flexibility of concept negation via test-time negative concept identification along with purification in the feature space. $ProtoRe$ works by incorporating CLIP's language-contrastive knowledge to identify the prototype of negative concepts, extract the negative features from outputs using the prototype as a prompt, and further refine the attention maps by retrieving negative features. Our evaluation on multiple benchmarks shows that $ProtoRe$ outperforms state-of-the-art methods under various settings, in terms of the effectiveness of purification and the fidelity of generative images.

----

## [0] LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark

**Authors**: *Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiyong Wang, Lu Sheng, Lei Bai, Jing Shao, Wanli Ouyang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/548a41b9cac6f50dccf7e63e9e1b1b9b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/548a41b9cac6f50dccf7e63e9e1b1b9b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large language models have emerged as a promising approach towards achieving general-purpose AI agents. The thriving open-source LLM community has greatly accelerated the development of agents that support human-machine dialogue interaction through natural language processing. However, human interaction with the world extends beyond only text as a modality, and other modalities such as vision are also crucial. Recent works on multi-modal large language models, such as GPT-4V and Bard, have demonstrated their effectiveness in handling visual modalities. However, the transparency of these works is limited and insufficient to support academic research. To the best of our knowledge, we present one of the very first open-source endeavors in the field, LAMM, encompassing a Language-Assisted Multi-Modal instruction tuning dataset, framework, and benchmark. Our aim is to establish LAMM as a growing ecosystem for training and evaluating MLLMs, with a specific focus on facilitating AI agents capable of bridging the gap between ideas and execution, thereby enabling seamless human-AI interaction. Our main contribution is three-fold: 1) We present a comprehensive dataset and benchmark, which cover a wide range of vision tasks for 2D and 3D vision. Extensive experiments validate the effectiveness of our dataset and benchmark. 2) We outline the detailed methodology of constructing multi-modal instruction tuning datasets and benchmarks for MLLMs, enabling rapid scaling and extension of MLLM research to diverse domains, tasks, and modalities. 3) We provide a primary but potential MLLM training framework optimized for modality extension. We also provide baseline models, comprehensive experimental observations, and analysis to accelerate future research. Our baseline model is trained within 24 A100 GPU hours, framework supports training with V100 and RTX3090 is available thanks to the open-source society. Codes and data are now available at https://openlamm.github.io.

----

## [0] Likelihood Ratio Confidence Sets for Sequential Decision Making

**Authors**: *Nicolas Emmenegger, Mojmir Mutny, Andreas Krause*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5491280797f3192b895bce84eb83df8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5491280797f3192b895bce84eb83df8d-Abstract-Conference.html)

**Abstract**:

Certifiable, adaptive uncertainty estimates for unknown quantities are an essential ingredient of sequential decision-making algorithms. Standard approaches rely on problem-dependent concentration results and are limited to a specific combination of parameterization, noise family, and estimator. In this paper, we revisit the likelihood-based inference principle and propose to use \emph{likelihood ratios} to construct \emph{any-time valid} confidence sequences without requiring specialized treatment in each application scenario. Our method is especially suitable for problems with well-specified likelihoods, and the resulting sets always maintain the prescribed coverage in a model-agnostic manner. The size of the sets depends on a choice of estimator sequence in the likelihood ratio. We discuss how to provably choose the best sequence of estimators and shed light on connections to online convex optimization with algorithms such as Follow-the-Regularized-Leader. To counteract the initially large bias of the estimators, we propose a reweighting scheme that also opens up deployment in non-parametric settings such as RKHS function classes. We provide a \emph{non-asymptotic} analysis of the likelihood ratio confidence sets size for generalized linear models, using insights from convex duality and online learning. We showcase the practical strength of our method on generalized linear bandit problems, survival analysis, and bandits with various additive noise distributions.

----

## [0] Uncertainty Quantification over Graph with Conformalized Graph Neural Networks

**Authors**: *Kexin Huang, Ying Jin, Emmanuel J. Candès, Jure Leskovec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54a1495b06c4ee2f07184afb9a37abda-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) are powerful machine learning prediction models on graph-structured data. However, GNNs lack rigorous uncertainty estimates, limiting their reliable deployment in settings where the cost of errors is significant. We propose conformalized GNN (CF-GNN), extending conformal prediction (CP) to graph-based models for guaranteed uncertainty estimates. Given an entity in the graph, CF-GNN produces a prediction set/interval that provably contains the true label with pre-defined coverage probability (e.g. 90%). We establish a permutation invariance condition that enables the validity of CP on graph data and provide an exact characterization of the test-time coverage. Moreover, besides valid coverage, it is crucial to reduce the prediction set size/interval length for practical use. We observe a key connection between non-conformity scores and network structures, which motivates us to develop a topology-aware output correction model that learns to update the prediction and produces more efficient prediction sets/intervals. Extensive experiments show that CF-GNN achieves any pre-defined target marginal coverage while significantly reducing the prediction set/interval size by up to 74% over the baselines. It also empirically achieves satisfactory conditional coverage over various raw and network features.

----

## [0] EMBERSim: A Large-Scale Databank for Boosting Similarity Search in Malware Analysis

**Authors**: *Dragos-Georgian Corlatescu, Alexandru Dinu, Mihaela Gaman, Paul Sumedrea*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54bf430f5d3090502ea021941e9cb18e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/54bf430f5d3090502ea021941e9cb18e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In recent years there has been a shift from heuristics based malware detection towards machine learning, which proves to be more robust in the current heavily adversarial threat landscape. While we acknowledge machine learning to be better equipped to mine for patterns in the increasingly high amounts of similar-looking files, we also note a remarkable scarcity of the data available for similarity targeted research. Moreover, we observe that the focus in the few related works falls on quantifying similarity in malware, often overlooking the clean data. This one-sided quantification is especially dangerous in the context of detection bypass. We propose to address the deficiencies in the space of similarity research on binary files, starting from EMBER â€” one of the largest malware classification datasets. We enhance EMBER with similarity information as well as malware class tags, to enable further research in the similarity space. Our contribution is threefold: (1) we publish EMBERSim, an augmented version of EMBER, that includes similarity informed tags; (2) we enrich EMBERSim with automatically determined malware class tags using the open-source tool AVClass on VirusTotal data and (3) we describe and share the implementation for our class scoring technique and leaf similarity method.

----

## [0] VPP: Efficient Conditional 3D Generation via Voxel-Point Progressive Representation

**Authors**: *Zekun Qi, Muzhou Yu, Runpei Dong, Kaisheng Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54d2d38a56a74387d5916ee40e462295-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54d2d38a56a74387d5916ee40e462295-Abstract-Conference.html)

**Abstract**:

Conditional 3D generation is undergoing a significant advancement, enabling the free creation of 3D content from inputs such as text or 2D images. However, previous approaches have suffered from low inference efficiency, limited generation categories, and restricted downstream applications. In this work, we revisit the impact of different 3D representations on generation quality and efficiency. We propose a progressive generation method through Voxel-Point Progressive Representation (VPP). VPP leverages structured voxel representation in the proposed Voxel Semantic Generator and the sparsity of unstructured point representation in the Point Upsampler, enabling efficient generation of multi-category objects. VPP can generate high-quality 8K point clouds within 0.2 seconds. Additionally, the masked generation Transformer allows for various 3D downstream tasks, such as generation, editing, completion, and pre-training. Extensive experiments demonstrate that VPP efficiently generates high-fidelity and diverse 3D shapes across different categories, while also exhibiting excellent representation transfer performance. Codes will be released at https://github.com/qizekun/VPP.

----

## [0] Multi Time Scale World Models

**Authors**: *Vaisakh Shaj, Saleh Gholam Zadeh, Ozan Demir, Luiz R. Douat, Gerhard Neumann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54d8aab579b5a9ed3395764c7341ebec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54d8aab579b5a9ed3395764c7341ebec-Abstract-Conference.html)

**Abstract**:

Intelligent agents use internal world models to reason and make predictions about different courses of their actions at many scales. Devising learning paradigms and architectures that allow machines to learn world models that operate at multiple levels of temporal abstractions while dealing with complex uncertainty predictions is a major technical hurdle. In this work, we propose a probabilistic formalism to learn multi-time scale world models which we call the Multi Time Scale State Space (MTS3) model. Our model uses a  computationally efficient inference scheme on multiple time scales for highly accurate long-horizon predictions and uncertainty estimates over several seconds into the future. Our experiments, which focus on action conditional long horizon future predictions, show that MTS3 outperforms recent methods on several system identification benchmarks including complex simulated and real-world dynamical systems. Code is available at this repository:https://github.com/ALRhub/MTS3.

----

## [0] How many samples are needed to leverage smoothness?

**Authors**: *Vivien Cabannes, Stefano Vigogna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54dcf25318f9de5a7a01f0a4125c541e-Abstract-Conference.html)

**Abstract**:

A core principle in statistical learning is that smoothness of target functions allows to break the curse of dimensionality. However, learning a smooth function seems to require enough samples close to one another to get meaningful estimate of high-order derivatives, which would be hard in machine learning problems where the ratio between number of data and input dimension is relatively small. By deriving new lower bounds on the generalization error, this paper formalizes such an intuition, before investigating the role of constants and transitory regimes which are usually not depicted beyond classical learning theory statements while they play a dominant role in practice.

----

## [0] Causal Imitability Under Context-Specific Independence Relations

**Authors**: *Fateme Jamshidi, Sina Akbari, Negar Kiyavash*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54e13b23fa2f399cea6e67acf9063c40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54e13b23fa2f399cea6e67acf9063c40-Abstract-Conference.html)

**Abstract**:

Drawbacks of ignoring the causal mechanisms when performing imitation learning have recently been acknowledged. Several approaches both to assess the feasibility of imitation and to circumvent causal confounding and causal misspecifications have been proposed in the literature.However, the potential benefits of the incorporation of additional information about the underlying causal structure are left unexplored.An example of such overlooked information is context-specific independence (CSI), i.e., independence that holds only in certain contexts.We consider the problem of causal imitation learning when CSI relations are known.We prove that the decision problem pertaining to the feasibility of imitation in this setting is NP-hard.Further, we provide a necessary graphical criterion for imitation learning under CSI and show that under a structural assumption, this criterion is also sufficient.Finally, we propose a sound algorithmic approach for causal imitation learning which takes both CSI relations and data into account.

----

## [0] A Finite-Particle Convergence Rate for Stein Variational Gradient Descent

**Authors**: *Jiaxin Shi, Lester Mackey*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54e5d7af6250ccab796ad7fe75663ba5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54e5d7af6250ccab796ad7fe75663ba5-Abstract-Conference.html)

**Abstract**:

We provide the first finite-particle convergence rate for Stein variational gradient descent (SVGD), a popular algorithm for approximating a probability distribution with a collection of particles. Specifically, whenever the target distribution is sub-Gaussian with a Lipschitz score, SVGD with $n$ particles and an appropriate step size sequence drives the kernel Stein discrepancy to zero at an order ${1/}{\sqrt{\log\log n}}$ rate. We suspect that the dependence on $n$ can be improved, and we hope that our explicit, non-asymptotic proof strategy will serve as a template for future refinements.

----

## [0] Bringing regularized optimal transport to lightspeed: a splitting method adapted for GPUs

**Authors**: *Jacob Lindbäck, Zesen Wang, Mikael Johansson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54f7125dee9b8b3dc798bb9a082b09e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54f7125dee9b8b3dc798bb9a082b09e2-Abstract-Conference.html)

**Abstract**:

We present an efficient algorithm for regularized optimal transport. In contrast toprevious methods, we use the Douglas-Rachford splitting technique to developan efficient solver that can handle a broad class of regularizers. The algorithmhas strong global convergence guarantees, low per-iteration cost, and can exploitGPU parallelization, making it considerably faster than the state-of-the-art formany problems. We illustrate its competitiveness in several applications, includingdomain adaptation and learning of generative models.

----

## [0] Use perturbations when learning from explanations

**Authors**: *Juyeon Heo, Vihari Piratla, Matthew Wicker, Adrian Weller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/54f82cdae821aad5c2888d61a6515170-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/54f82cdae821aad5c2888d61a6515170-Abstract-Conference.html)

**Abstract**:

Machine learning from explanations (MLX) is an approach to learning that uses human-provided explanations of relevant or irrelevant features for each input to ensure that model predictions are right for the right reasons. Existing MLX approaches rely on local model interpretation methods and require strong model smoothing to align model and human explanations, leading to sub-optimal performance. We recast MLX as a robustness problem, where human explanations specify a lower dimensional manifold from which perturbations can be drawn, and show both theoretically and empirically how this approach alleviates the need for strong model smoothing. We consider various approaches to achieving robustness, leading to improved performance over prior MLX methods. Finally, we show how to combine robustness with an earlier MLX method, yielding state-of-the-art results on both synthetic and real-world benchmarks.

----

## [0] VisIT-Bench: A Dynamic Benchmark for Evaluating Instruction-Following Vision-and-Language Models

**Authors**: *Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5503389dbe070cdae9b48086c4996a59-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/5503389dbe070cdae9b48086c4996a59-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluating instruction-following vision-language models for real-world use. Our starting point is curating 70 "instruction families" that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps between models and references using both human and automatic evaluations; e.g., the top-performing instruction-following model wins against the GPT-4 reference in just 27% of the comparison. VisIT-Bench is dynamic to participate, practitioners simply submit their model's response on the project website; Data, code and leaderboard is available at https://visit-bench.github.io/.

----

## [0] The Memory-Perturbation Equation: Understanding Model's Sensitivity to Data

**Authors**: *Peter Nickl, Lu Xu, Dharmesh Tailor, Thomas Möllenhoff, Mohammad Emtiyaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/550ab405d0addd3de5b70e57b44878df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/550ab405d0addd3de5b70e57b44878df-Abstract-Conference.html)

**Abstract**:

Understanding model’s sensitivity to its training data is crucial but can also be challenging and costly, especially during training. To simplify such issues, we present the Memory-Perturbation Equation (MPE) which relates model's sensitivity to perturbation in its training data. Derived using Bayesian principles, the MPE unifies existing sensitivity measures, generalizes them to a wide-variety of models and algorithms, and unravels useful properties regarding sensitivities. Our empirical results show that sensitivity estimates obtained during training can be used to faithfully predict generalization on unseen test data. The proposed equation is expected to be useful for future research on robust and adaptive learning.

----

## [0] Contextual Gaussian Process Bandits with Neural Networks

**Authors**: *Haoting Zhang, Jinghai He, Rhonda Righter, Zuo-Jun Max Shen, Zeyu Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5526c73e3ff4f2a34009e13d15f52fcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5526c73e3ff4f2a34009e13d15f52fcb-Abstract-Conference.html)

**Abstract**:

Contextual decision-making problems have witnessed extensive applications in various fields such as online content recommendation, personalized healthcare, and autonomous vehicles, where a core practical challenge is to select a suitable surrogate model for capturing unknown complicated reward functions. It is often the case that both high approximation accuracy and explicit uncertainty quantification are desired. In this work, we propose a neural network-accompanied Gaussian process (NN-AGP) model, which leverages neural networks to approximate the unknown and potentially complicated reward function regarding the contextual variable, and maintains a Gaussian process surrogate model with respect to the decision variable. Our model is shown to outperform existing approaches by offering better approximation accuracy thanks to the use of neural networks and possessing explicit uncertainty quantification from the Gaussian process. We also analyze the maximum information gain of the NN-AGP model and prove regret bounds for the corresponding algorithms. Moreover, we conduct experiments on both synthetic and practical problems, illustrating the effectiveness of our approach.

----

## [0] Auxiliary Losses for Learning Generalizable Concept-based Models

**Authors**: *Ivaxi Sheth, Samira Ebrahimi Kahou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/555479a201da27c97aaeed842d16ca49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/555479a201da27c97aaeed842d16ca49-Abstract-Conference.html)

**Abstract**:

The increasing use of neural networks in various applications has lead to increasing apprehensions, underscoring the necessity to understand their operations beyond mere final predictions. As a solution to enhance model transparency, Concept Bottleneck Models (CBMs) have gained popularity since their introduction. CBMs essentially limit the latent space of a model to human-understandable high-level concepts.  While beneficial, CBMs have been reported to often learn irrelevant concept representations that consecutively damage model performance. To overcome the performance trade-off, we propose a cooperative-Concept Bottleneck Model (coop-CBM). The concept representation of our model is particularly meaningful when fine-grained concept labels are absent. Furthermore, we introduce the concept orthogonal loss (COL) to encourage the separation between the concept representations and to reduce the intra-concept distance. This paper presents extensive experiments on real-world datasets for image classification tasks, namely CUB, AwA2, CelebA and TIL. We also study the performance of coop-CBM models under various distributional shift settings. We show that our proposed method achieves higher accuracy in all distributional shift settings even compared to the black-box models with the highest concept accuracy.

----

## [0] Make the U in UDA Matter: Invariant Consistency Learning for Unsupervised Domain Adaptation

**Authors**: *Zhongqi Yue, Qianru Sun, Hanwang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5555cc3fb226ed067fa946e35355f938-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5555cc3fb226ed067fa946e35355f938-Abstract-Conference.html)

**Abstract**:

Domain Adaptation (DA) is always challenged by the spurious correlation between the domain-invariant features (e.g., class identity) and the domain-specific ones (e.g., environment) that does not generalize to the target domain. Unfortunately, even enriched with additional unsupervised target domains, existing Unsupervised DA (UDA) methods still suffer from it. This is because the source domain supervision only considers the target domain samples as auxiliary data (e.g., by pseudo-labeling), yet the inherent distribution in the target domain---where the valuable de-correlation clues hide---is disregarded. We propose to make the U in UDA matter by giving equal status to the two domains. Specifically, we learn an invariant classifier whose prediction is simultaneously consistent with the labels in the source domain and clusters in the target domain, hence the spurious correlation inconsistent in the target domain is removed. We dub our approach "Invariant CONsistency learning" (ICON). Extensive experiments show that ICON achieves the state-of-the-art performance on the classic UDA benchmarks: Office-Home and VisDA-2017, and outperforms all the conventional methods on the challenging WILDS 2.0 benchmark. Codes are in https://github.com/yue-zhongqi/ICON.

----

## [0] Hyper-HMM: aligning human brains and semantic features in a common latent event space

**Authors**: *Caroline Lee, Jane Han, Feilong Ma, Jiahui Guo, James V. Haxby, Christopher Baldassano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/558a100caa93422df215fadb9e9b1dd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/558a100caa93422df215fadb9e9b1dd7-Abstract-Conference.html)

**Abstract**:

Naturalistic stimuli evoke complex neural responses with spatial and temporal properties that differ across individuals. Current alignment methods focus on either spatial hyperalignment (assuming exact temporal correspondence) or temporal alignment (assuming exact spatial correspondence). Here, we propose a hybrid model, the Hyper-HMM, that simultaneously aligns both temporal and spatial features across brains. The model learns to linearly project voxels to a reduced-dimension latent space, in which timecourses are segmented into corresponding temporal events. This approach allows tracking of each individual's mental trajectory through an event sequence, and also allows for alignment with other feature spaces such as stimulus content. Using an fMRI dataset in which students watch videos of class lectures, we demonstrate that the Hyper-HMM can be used to map all participants and the semantic content of the videos into a common low-dimensional space, and that these mappings generalize to held-out data. Our model provides a new window into individual cognitive dynamics evoked by complex naturalistic stimuli.

----

## [0] Active Learning for Semantic Segmentation with Multi-class Label Query

**Authors**: *Sehyun Hwang, Sohyun Lee, Hoyoung Kim, Minhyeon Oh, Jungseul Ok, Suha Kwak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/559a0998fab1d19b80e7e43a5852401c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/559a0998fab1d19b80e7e43a5852401c-Abstract-Conference.html)

**Abstract**:

This paper proposes a new active learning method for semantic segmentation. The core of our method lies in a new annotation query design. It samples informative local image regions ($\textit{e.g.}$, superpixels), and for each of such regions, asks an oracle for a multi-hot vector indicating all classes existing in the region. This multi-class labeling strategy is substantially more efficient than existing ones like segmentation, polygon, and even dominant class labeling in terms of annotation time per click. However, it introduces the class ambiguity issue in training as it assigns partial labels ($\textit{i.e.}$, a set of candidate classes) to individual pixels. We thus propose a new algorithm for learning semantic segmentation while disambiguating the partial labels in two stages. In the first stage, it trains a segmentation model directly with the partial labels through two new loss functions motivated by partial label learning and multiple instance learning. In the second stage, it disambiguates the partial labels by generating pixel-wise pseudo labels, which are used for supervised learning of the model. Equipped with a new acquisition function dedicated to the multi-class labeling, our method outperforms previous work on Cityscapes and PASCAL VOC 2012 while spending less annotation cost. Our code and results are available at [https://github.com/sehyun03/MulActSeg](https://github.com/sehyun03/MulActSeg).

----

## [0] Aleatoric and Epistemic Discrimination: Fundamental Limits of Fairness Interventions

**Authors**: *Hao Wang, Luxi He, Rui Gao, Flávio P. Calmon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/55a49718689fdecef31b6a2386df6fe1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/55a49718689fdecef31b6a2386df6fe1-Abstract-Conference.html)

**Abstract**:

Machine learning (ML) models can underperform on certain population groups due to choices made during model development and bias inherent in the data. We categorize sources of discrimination in the ML pipeline into two classes: aleatoric discrimination, which is inherent in the data distribution, and epistemic discrimination, which is due to decisions made during model development. We quantify aleatoric discrimination by determining the performance limits of a model under fairness constraints, assuming perfect knowledge of the data distribution. We demonstrate how to characterize aleatoric discrimination by applying Blackwell's results on comparing statistical experiments. We then quantify epistemic discrimination as the gap between a model's accuracy when fairness constraints are applied and the limit posed by aleatoric discrimination. We apply this approach to benchmark existing fairness interventions and investigate fairness risks in data with missing values. Our results indicate that state-of-the-art fairness interventions are effective at removing epistemic discrimination on standard (overused) tabular datasets. However, when data has missing values, there is still significant room for improvement in handling aleatoric discrimination.

----

## [0] DISCOVER: Making Vision Networks Interpretable via Competition and Dissection

**Authors**: *Konstantinos P. Panousis, Sotirios Chatzis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/55aeba84b402008d3ed10440d906b4e1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/55aeba84b402008d3ed10440d906b4e1-Abstract-Conference.html)

**Abstract**:

Modern deep networks are highly complex and their inferential outcome very hard to interpret. This is a serious obstacle to their transparent deployment in safety-critical or bias-aware applications. This work contributes to *post-hoc* interpretability, and specifically Network Dissection. Our goal is to present a framework that makes it easier to *discover* the individual functionality of each neuron in a network trained on a vision task; discovery is performed in terms of textual description generation. To achieve this objective, we leverage: (i) recent advances in multimodal vision-text models and (ii) network layers founded upon the novel concept of stochastic local competition between linear units. In this setting, only a *small subset* of layer neurons are activated *for a given input*, leading to extremely high activation sparsity (as low as only $\approx 4\%$). Crucially, our proposed method infers (sparse) neuron activation patterns that enables the neurons to activate/specialize to inputs with specific characteristics, diversifying their individual functionality. This capacity of our method supercharges the potential of dissection processes: human understandable descriptions are generated only for the very few active neurons, thus facilitating the direct investigation of the network's decision process. As we experimentally show, our approach: (i) yields Vision Networks that retain or improve classification performance, and (ii) realizes a principled framework for text-based description and examination of the generated neuronal representations.

----

## [0] Adapting Neural Link Predictors for Data-Efficient Complex Query Answering

**Authors**: *Erik Arakelyan, Pasquale Minervini, Daniel Daza, Michael Cochez, Isabelle Augenstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/55c518a17bd17dcb69aa14d69d085994-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/55c518a17bd17dcb69aa14d69d085994-Abstract-Conference.html)

**Abstract**:

Answering complex queries on incomplete knowledge graphs is a challenging task where a model needs to answer complex logical queries in the presence of missing knowledge. Prior work in the literature has proposed to address this problem by designing architectures trained end-to-end for the complex query answering task with a reasoning process that is hard to interpret while requiring data and resource-intensive training.  Other lines of research have proposed re-using simple neural link predictors to answer complex queries, reducing the amount of training data by orders of magnitude while providing interpretable answers. The neural link predictor used in such approaches is not explicitly optimised for the complex query answering task, implying that its scores are not calibrated to interact together. We propose to address these problems via CQD$^{\mathcal{A}}$, a parameter-efficient score \emph{adaptation} model optimised to re-calibrate neural link prediction scores for the complex query answering task. While the neural link predictor is frozen, the adaptation component -- which only increases the number of model parameters by $0.03\%$ -- is trained on the downstream complex query answering task. Furthermore, the calibration component enables us to support reasoning over queries that include atomic negations, which was previously impossible with link predictors. In our experiments, CQD$^{\mathcal{A}}$ produces significantly more accurate results than current state-of-the-art methods, improving from $34.4$ to $35.1$ Mean Reciprocal Rank values averaged across all datasets and query types while using $\leq 30\%$ of the available training query types. We further show that CQD$^{\mathcal{A}}$ is data-efficient, achieving competitive results with only $1\%$ of the complex training queries and robust in out-of-domain evaluations. Source code and datasets are available at https://github.com/EdinburghNLP/adaptive-cqd.

----

## [0] DataComp: In search of the next generation of multimodal datasets

**Authors**: *Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, Eyal Orgad, Rahim Entezari, Giannis Daras, Sarah M. Pratt, Vivek Ramanujan, Yonatan Bitton, Kalyani Marathe, Stephen Mussmann, Richard Vencu, Mehdi Cherti, Ranjay Krishna, Pang Wei Koh, Olga Saukh, Alexander J. Ratner, Shuran Song, Hannaneh Hajishirzi, Ali Farhadi, Romain Beaumont, Sewoong Oh, Alex Dimakis, Jenia Jitsev, Yair Carmon, Vaishaal Shankar, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/56332d41d55ad7ad8024aac625881be7-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Multimodal datasets are a critical component in recent breakthroughs such as CLIP, Stable Diffusion and GPT-4, yet their design does not receive the same research attention as model architectures or training algorithms. To address this shortcoming in the machine learning ecosystem, we introduce DataComp, a testbed for dataset experiments centered around a new candidate pool of 12.8 billion image-text pairs from Common Crawl. Participants in our benchmark design new filtering techniques or curate new data sources and then evaluate their new dataset by running our standardized CLIP training code and testing the resulting model on 38 downstream test sets. Our benchmark consists of multiple compute scales spanning four orders of magnitude, which enables the study of scaling trends and makes the benchmark accessible to researchers with varying resources. Our baseline experiments show that the DataComp workflow leads to better training sets. Our best baseline, DataComp-1B, enables training a CLIP ViT-L/14 from scratch to 79.2% zero-shot accuracy on ImageNet, outperforming OpenAI's CLIP ViT-L/14 by 3.7 percentage points while using the same training procedure and compute. We release \datanet and all accompanying code at www.datacomp.ai.

----

## [0] p-value Adjustment for Monotonous, Unbiased, and Fast Clustering Comparison

**Authors**: *Kai Klede, Thomas Altstidl, Dario Zanca, Bjoern M. Eskofier*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/563d94819f68cb73d6a382809e587b54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/563d94819f68cb73d6a382809e587b54-Abstract-Conference.html)

**Abstract**:

Popular metrics for clustering comparison, like the Adjusted Rand Index and the Adjusted Mutual Information, are type II biased. The Standardized Mutual Information removes this bias but suffers from counterintuitive non-monotonicity and poor computational efficiency. We introduce the $p$-value adjusted Rand Index ($\operatorname{PMI}_2$), the first cluster comparison method that is type II unbiased and provably monotonous. The $\operatorname{PMI}_2$ has fast approximations that outperform the Standardized Mutual information. We demonstrate its unbiased clustering selection, approximation quality, and runtime efficiency on synthetic benchmarks. In experiments on image and social network datasets, we show how the $\operatorname{PMI}_2$ can help practitioners choose better clustering and community detection algorithms.

----

## [0] On Computing Pairwise Statistics with Local Differential Privacy

**Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Adam Sealfon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5642b9811a9ac5281be1cc84c275f251-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5642b9811a9ac5281be1cc84c275f251-Abstract-Conference.html)

**Abstract**:

We study the problem of computing pairwise statistics, i.e., ones of the form $\binom{n}{2}^{-1} \sum_{i \ne j} f(x_i, x_j)$, where $x_i$ denotes the input to the $i$th user, with differential privacy (DP) in the local model. This formulation captures important metrics such as Kendall's $\tau$ coefficient, Area Under Curve, Gini's mean difference, Gini's entropy, etc. We give several novel and generic algorithms for the problem, leveraging techniques from DP algorithms for linear queries.

----

## [0] STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning

**Authors**: *Weipu Zhang, Gang Wang, Jian Sun, Yetian Yuan, Gao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5647763d4245b23e6a1cb0a8947b38c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5647763d4245b23e6a1cb0a8947b38c9-Abstract-Conference.html)

**Abstract**:

Recently, model-based reinforcement learning algorithms have demonstrated remarkable efficacy  in visual input environments. These approaches begin by constructing a parameterized simulation world model of the real environment through self-supervised learning. By leveraging the imagination of the world model, the agent's policy is enhanced without the constraints of sampling from the real environment. The performance of these algorithms heavily relies on the sequence modeling and generation capabilities of the world model. However, constructing a perfectly accurate model of a complex unknown environment is nearly impossible. Discrepancies between the model and reality may cause the agent to pursue virtual goals, resulting in subpar performance in the real environment. Introducing random noise into model-based reinforcement learning has been proven beneficial.In this work, we introduce Stochastic Transformer-based wORld Model (STORM), an efficient world model architecture that combines the strong sequence modeling and generation capabilities of Transformers with the stochastic nature of variational autoencoders. STORM achieves a mean human performance of $126.7\%$ on the Atari $100$k benchmark, setting a new record among state-of-the-art methods that do not employ lookahead search techniques. Moreover, training an agent with $1.85$ hours of real-time interaction experience on a single NVIDIA GeForce RTX 3090 graphics card requires only $4.3$ hours, showcasing improved efficiency compared to previous methodologies.

----

## [0] Is Heterogeneity Notorious? Taming Heterogeneity to Handle Test-Time Shift in Federated Learning

**Authors**: *Yue Tan, Chen Chen, Weiming Zhuang, Xin Dong, Lingjuan Lyu, Guodong Long*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is an effective machine learning paradigm where multiple clients can train models based on heterogeneous data in a decentralized manner without accessing their private data. However, existing FL systems undergo performance deterioration due to feature-level test-time shifts, which are well investigated in centralized settings but rarely studied in FL. The common non-IID issue in FL usually refers to inter-client heterogeneity during training phase, while the test-time shift refers to the intra-client heterogeneity during test phase. Although the former is always deemed to be notorious for FL, there is still a wealth of useful information delivered by heterogeneous data sources, which may potentially help alleviate the latter issue. To explore the possibility of using inter-client heterogeneity in handling intra-client heterogeneity, we firstly propose a contrastive learning-based FL framework, namely FedICON, to capture invariant knowledge among heterogeneous clients and consistently tune the model to adapt to test data. In FedICON, each client performs sample-wise supervised contrastive learning during the local training phase, which enhances sample-wise invariance encoding ability. Through global aggregation, the invariance extraction ability can be mutually boosted among inter-client heterogeneity. During the test phase, our test-time adaptation procedure leverages unsupervised contrastive learning to guide the model to smoothly generalize to test data under intra-client heterogeneity. Extensive experiments validate the effectiveness of the proposed FedICON in taming heterogeneity to handle test-time shift problems.

----

## [0] Self-Predictive Universal AI

**Authors**: *Elliot Catt, Jordi Grau-Moya, Marcus Hutter, Matthew Aitchison, Tim Genewein, Grégoire Delétang, Kevin Li, Joel Veness*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/56a225639da77e8f7c0409f6d5ba996b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/56a225639da77e8f7c0409f6d5ba996b-Abstract-Conference.html)

**Abstract**:

Reinforcement Learning (RL) algorithms typically utilize learning and/or planning techniques to derive effective policies. The integration of both approaches has proven to be highly successful in addressing complex sequential decision-making challenges, as evidenced by algorithms such as AlphaZero and MuZero, which consolidate the planning process into a parametric search-policy. AIXI, the most potent theoretical universal agent, leverages planning through comprehensive search as its primary means to find an optimal policy. Here we define an alternative universal agent, which we call Self-AIXI, that on the contrary to AIXI, maximally exploits learning to obtain good policies. It does so by self-predicting its own stream of action data, which is generated, similarly to other TD(0) agents, by taking an action maximization step over the current on-policy (universal mixture-policy) Q-value estimates. We prove that Self-AIXI converges to AIXI, and inherits a series of properties like maximal Legg-Hutter intelligence and the self-optimizing property.

----

## [0] Addressing Negative Transfer in Diffusion Models

**Authors**: *Hyojun Go, JinYoung Kim, Yunsung Lee, Seunghyun Lee, Shinhyeok Oh, Hyeongdon Moon, Seungtaek Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/56a7b9a07ae01ddea762dcc51280298b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/56a7b9a07ae01ddea762dcc51280298b-Abstract-Conference.html)

**Abstract**:

Diffusion-based generative models have achieved remarkable success in various domains. It trains a shared model on denoising tasks that encompass different noise levels simultaneously, representing a form of multi-task learning (MTL). However, analyzing and improving diffusion models from an MTL perspective remains under-explored. In particular, MTL can sometimes lead to the well-known phenomenon of $\textit{negative transfer}$, which results in the performance degradation of certain tasks due to conflicts between tasks. In this paper, we first aim to analyze diffusion training from an MTL standpoint, presenting two key observations: $\textbf{(O1)}$ the task affinity between denoising tasks diminishes as the gap between noise levels widens, and $\textbf{(O2)}$ negative transfer can arise even in diffusion training. Building upon these observations, we aim to enhance diffusion training by mitigating negative transfer. To achieve this, we propose leveraging existing MTL methods, but the presence of a huge number of denoising tasks makes this computationally expensive to calculate the necessary per-task loss or gradient. To address this challenge, we propose clustering the denoising tasks into small task clusters and applying MTL methods to them. Specifically, based on $\textbf{(O2)}$, we employ interval clustering to enforce temporal proximity among denoising tasks within clusters. We show that interval clustering can be solved using dynamic programming, utilizing signal-to-noise ratio, timestep, and task affinity for clustering objectives. Through this, our approach addresses the issue of negative transfer in diffusion models by allowing for efficient computation of MTL methods. We validate the efficacy of proposed clustering and its integration with MTL methods through various experiments, demonstrating 1) improved generation quality and 2) faster training convergence of diffusion models. Our project page is available at https://gohyojun15.github.io/ANT_diffusion/.

----

## [0] The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks

**Authors**: *Ziqian Zhong, Ziming Liu, Max Tegmark, Jacob Andreas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/56cbfbf49937a0873d451343ddc8c57d-Abstract-Conference.html)

**Abstract**:

Do neural networks, trained on well-understood algorithmic tasks, reliably rediscover known algorithms? Several recent studies, on tasks ranging from group operations to in-context linear regression, have suggested that the answer is yes. Using modular addition as a prototypical problem, we show that algorithm discovery in neural networks is sometimes more complex: small changes to model hyperparameters and initializations can induce discovery of qualitatively different algorithms from a fixed training set, and even learning of multiple different solutions in parallel. In modular addition, we specifically show that models learn a known Clock algorithm, a previously undescribed, less intuitive, but comprehensible procedure we term the Pizza algorithm, and a variety of even more complex procedures. Our results show that even simple learning problems can admit a surprising diversity of solutions, motivating the development of new tools for mechanistically characterizing the behavior of neural networks across the algorithmic phase space.

----

## [0] Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems

**Authors**: *Samuel Hurault, Ulugbek Kamilov, Arthur Leclaire, Nicolas Papadakis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/56db53e53db1b29ae658e53fb764f067-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/56db53e53db1b29ae658e53fb764f067-Abstract-Conference.html)

**Abstract**:

Plug-and-Play (PnP) methods are efficient iterative algorithms for solving ill-posed image inverse problems. PnP methods are obtained by using deep Gaussian denoisers instead of the proximal operator or the gradient-descent step within proximal algorithms. Current PnP schemes rely on data-fidelity terms that have either Lipschitz gradients or closed-form proximal operators, which is not applicable to Poisson inverse problems. Based on the observation that the Gaussian noise is not the adequate noise model in this setting, we propose to generalize PnP using the Bregman Proximal Gradient (BPG) method. BPG replaces the Euclidean distance with a Bregman divergence that can better capture the smoothness properties of the problem. We introduce the Bregman Score Denoiser specifically parametrized and trained for the new Bregman geometry and prove that it corresponds to the proximal operator of a nonconvex potential. We propose two PnP algorithms based on the Bregman Score Denoiser for solving Poisson inverse problems. Extending the convergence results of BPG in the nonconvex settings, we show that the proposed methods converge, targeting stationary points of an explicit global functional. Experimental evaluations conducted on various Poisson inverse problems validate the convergence results and showcase effective restoration performance.

----

## [0] SQ Lower Bounds for Learning Mixtures of Linear Classifiers

**Authors**: *Ilias Diakonikolas, Daniel Kane, Yuxin Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/571fcbd2ad4273d9df51d7abc1172112-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/571fcbd2ad4273d9df51d7abc1172112-Abstract-Conference.html)

**Abstract**:

We study the problem of learning mixtures of linear classifiers under Gaussian covariates.Given sample access to a mixture of $r$ distributions on $\mathbb{R}^n$ of the form $(\mathbf{x},y_{\ell})$, $\ell \in [r]$,where $\mathbf{x}\sim\mathcal{N}(0,\mathbf{I}_n)$ and$y_\ell=\mathrm{sign}(\langle\mathbf{v}_{\ell},\mathbf{x}\rangle)$for an unknown unit vector $\mathbf{v}_{\ell}$,the goal is to learn the underlying distribution in total variation distance. Our main result is a Statistical Query (SQ) lower bound suggesting that known algorithms for this problem are essentially best possible,even for the special case of uniform mixtures.In particular, we show that the complexity of any SQ algorithm for the problem is $n^{\mathrm{poly}(1/\Delta) \log(r)}$,where $\Delta$ is a lower bound on the pairwise $\ell_2$-separation between the $\mathbf{v}_{\ell}$'s.The key technical ingredient underlying our result is a new construction of spherical designs on the unit sphere that may be of independent interest.

----

## [0] Canonical normalizing flows for manifold learning

**Authors**: *Kyriakos Flouris, Ender Konukoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/572a6f16ec44f794fb3e0f8a310acbc6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/572a6f16ec44f794fb3e0f8a310acbc6-Abstract-Conference.html)

**Abstract**:

Manifold learning flows are a class of generative modelling techniques that assume a low-dimensional manifold description of the data. The embedding of such a manifold into the high-dimensional space of the data is achieved via learnable invertible transformations. Therefore, once the manifold is properly aligned via a reconstruction loss, the probability density is tractable on the manifold and maximum likelihood can be used to optimize the network parameters. Naturally, the lower-dimensional representation of the data requires an injective-mapping. Recent approaches were able to enforce that the density aligns with the modelled manifold, while efficiently calculating the density volume-change term when embedding to the higher-dimensional space. However, unless the injective-mapping is analytically predefined, the learned manifold is not necessarily an \emph{efficient representation} of the data. Namely, the latent dimensions of such models frequently learn an entangled intrinsic basis, with degenerate information being stored in each dimension. Alternatively, if a locally orthogonal and/or sparse basis is to be learned, here coined canonical intrinsic basis, it can serve in learning a more compact latent space representation. Toward this end, we propose a canonical manifold learning flow method, where a novel optimization objective enforces the transformation matrix to have few prominent and non-degenerate basis functions. We demonstrate that by minimizing the off-diagonal manifold metric elements $\ell_1$-norm, we can achieve such a basis, which is simultaneously sparse and/or orthogonal. Canonical manifold flow yields a more efficient use of the latent space, automatically generating fewer prominent and distinct dimensions to represent data, and consequently a better approximation of target distributions than other manifold flow methods in most experiments we conducted, resulting in lower FID scores.

----

## [0] Regularizing Neural Networks with Meta-Learning Generative Models

**Authors**: *Shin'ya Yamaguchi, Daiki Chijiwa, Sekitoshi Kanai, Atsutoshi Kumagai, Hisashi Kashima*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/572cd21bd5dea96b065476b77d21b3c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/572cd21bd5dea96b065476b77d21b3c6-Abstract-Conference.html)

**Abstract**:

This paper investigates methods for improving generative data augmentation for deep learning. Generative data augmentation leverages the synthetic samples produced by generative models as an additional dataset for classification with small dataset settings. A key challenge of generative data augmentation is that the synthetic data contain uninformative samples that degrade accuracy. This can be caused by the synthetic samples not perfectly representing class categories in real data and uniform sampling not necessarily providing useful samples for tasks. In this paper, we present a novel strategy for generative data augmentation called meta generative regularization (MGR). To avoid the degradation of generative data augmentation, MGR utilizes synthetic samples for regularizing feature extractors instead of training classifiers. These synthetic samples are dynamically determined to minimize the validation losses through meta-learning. We observed that MGR can avoid the performance degradation of naive generative data augmentation and boost the baselines. Experiments on six datasets showed that MGR is effective particularly when datasets are smaller and stably outperforms baselines by up to 7 percentage points on test accuracy.

----

## [0] Landscape Surrogate: Learning Decision Losses for Mathematical Optimization Under Partial Information

**Authors**: *Arman Zharmagambetov, Brandon Amos, Aaron M. Ferber, Taoan Huang, Bistra Dilkina, Yuandong Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/574f145eac328cc4aaf9358e27120eb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/574f145eac328cc4aaf9358e27120eb5-Abstract-Conference.html)

**Abstract**:

Recent works in learning-integrated optimization have shown promise in settings where the optimization problem is only partially observed or where general-purpose optimizers perform poorly without expert tuning. By learning an optimizer $\mathbf{g}$ to tackle these challenging problems with $f$ as the objective, the optimization process can be substantially accelerated by leveraging past experience. The optimizer can be trained with supervision from known optimal solutions or implicitly by optimizing the compound function $f\circ \mathbf{g}$. The implicit approach may not require optimal solutions as labels and is capable of handling problem uncertainty; however, it is slow to train and deploy due to frequent calls to optimizer $\mathbf{g}$ during both training and testing. The training is further challenged by sparse gradients of $\mathbf{g}$, especially for combinatorial solvers. To address these challenges, we propose using a smooth and learnable **Landscape Surrogate** $\mathcal{M}$ as a replacement for $f\circ \mathbf{g}$. This surrogate, learnable by neural networks, can be computed faster than the solver $\mathbf{g}$, provides dense and smooth gradients during training, can generalize to unseen optimization problems, and is efficiently learned via alternating optimization. We test our approach on both synthetic problems, including shortest path and multidimensional knapsack, and real-world problems such as portfolio optimization, achieving comparable or superior objective values compared to state-of-the-art baselines while reducing the number of calls to $\mathbf{g}$. Notably, our approach outperforms existing methods for computationally expensive high-dimensional problems.

----

## [0] Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework

**Authors**: *Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard J. Chen, Zihao Deng, Nicholas B. Allen, Randy Auerbach, Faisal Mahmood, Russ Salakhutdinov, Louis-Philippe Morency*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html)

**Abstract**:

The recent explosion of interest in multimodal applications has resulted in a wide selection of datasets and methods for representing and integrating information from different modalities. Despite these empirical advances, there remain fundamental research questions: How can we quantify the interactions that are necessary to solve a multimodal task? Subsequently, what are the most suitable multimodal models to capture these interactions? To answer these questions, we propose an information-theoretic approach to quantify the degree of redundancy, uniqueness, and synergy relating input modalities with an output task. We term these three measures as the PID statistics of a multimodal distribution (or PID for short), and introduce two new estimators for these PID statistics that scale to high-dimensional distributions. To validate PID estimation, we conduct extensive experiments on both synthetic datasets where the PID is known and on large-scale multimodal benchmarks where PID estimations are compared with human annotations. Finally, we demonstrate their usefulness in (1) quantifying interactions within multimodal datasets, (2) quantifying interactions captured by multimodal models, (3) principled approaches for model selection, and (4) three real-world case studies engaging with domain experts in pathology, mood prediction, and robotic perception where our framework helps to recommend strong multimodal models for each application.

----

## [0] A Single 2D Pose with Context is Worth Hundreds for 3D Human Pose Estimation

**Authors**: *Qitao Zhao, Ce Zheng, Mengyuan Liu, Chen Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/5752f9fd2d5c40174738d6f02c202e72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/5752f9fd2d5c40174738d6f02c202e72-Abstract-Conference.html)

**Abstract**:

The dominant paradigm in 3D human pose estimation that lifts a 2D pose sequence to 3D heavily relies on long-term temporal clues (i.e., using a daunting number of video frames) for improved accuracy, which incurs performance saturation, intractable computation and the non-causal problem. This can be attributed to their inherent inability to perceive spatial context as plain 2D joint coordinates carry no visual cues. To address this issue, we propose a straightforward yet powerful solution: leveraging the $\textit{readily available}$ intermediate visual representations produced by off-the-shelf (pre-trained) 2D pose detectors -- no finetuning on the 3D task is even needed. The key observation is that, while the pose detector learns to localize 2D joints, such representations (e.g., feature maps) implicitly encode the joint-centric spatial context thanks to the regional operations in backbone networks. We design a simple baseline named $\textbf{Context-Aware PoseFormer}$ to showcase its effectiveness. $\textit{Without access to any temporal information}$, the proposed method significantly outperforms its context-agnostic counterpart, PoseFormer, and other state-of-the-art methods using up to $\textit{hundreds of}$ video frames regarding both speed and precision. $\textit{Project page:}$ https://qitaozhao.github.io/ContextAware-PoseFormer

----

## [0] On the Stability-Plasticity Dilemma in Continual Meta-Learning: Theory and Algorithm

**Authors**: *Qi Chen, Changjian Shui, Ligong Han, Mario Marchand*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57587d8d6a7ede0e5302fc22d0878c53-Abstract-Conference.html)

**Abstract**:

We focus on Continual Meta-Learning (CML), which targets accumulating and exploiting meta-knowledge on a sequence of non-i.i.d. tasks. The primary challenge is to strike a balance between stability and plasticity, where a model should be stable to avoid catastrophic forgetting in previous tasks and plastic to learn generalizable concepts from new tasks. To address this, we formulate the CML objective as controlling the average excess risk upper bound of the task sequence, which reflects the trade-off between forgetting and generalization. Based on the objective, we introduce a unified theoretical framework for CML in both static and shifting environments, providing guarantees for various task-specific learning algorithms. Moreover, we first present a rigorous analysis of a bi-level trade-off in shifting environments. To approach the optimal trade-off, we propose a novel algorithm that dynamically adjusts the meta-parameter and its learning rate w.r.t environment change. Empirical evaluations on synthetic and real datasets illustrate the effectiveness of the proposed theory and algorithm.

----

## [0] Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense

**Authors**: *Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, Mohit Iyyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/575c450013d0e99e4b0ecf82bd1afaa4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/575c450013d0e99e4b0ecf82bd1afaa4-Abstract-Conference.html)

**Abstract**:

The rise in malicious usage of large language models, such as fake content creation and academic plagiarism, has motivated the development of approaches that identify AI-generated text, including those based on watermarking or outlier detection. However, the robustness of these detection algorithms to paraphrases of AI-generated text remains unclear. To stress test these detectors, we build a 11B parameter paraphrase generation model (DIPPER) that can paraphrase paragraphs, condition on surrounding context, and control lexical diversity and content reordering. Paraphrasing text generated by three large language models (including GPT3.5-davinci-003) with DIPPER successfully evades several detectors, including watermarking, GPTZero, DetectGPT, and OpenAI's text classifier. For example, DIPPER drops detection accuracy of DetectGPT from 70.3% to 4.6% (at a constant false positive rate of 1%), without appreciably modifying the input semantics.To increase the robustness of AI-generated text detection to paraphrase attacks, we introduce a simple defense that relies on retrieving semantically-similar generations and must be maintained by a language model API provider. Given a candidate text, our algorithm searches a database of sequences previously generated by the API, looking for sequences that match the candidate text within a certain threshold. We empirically verify our defense using a database of 15M generations from a fine-tuned T5-XXL model and find that it can detect 80% to 97% of paraphrased generations across different settings while only classifying 1% of human-written sequences as AI-generated. We open-source our models, code and data.

----

## [0] ChimpACT: A Longitudinal Dataset for Understanding Chimpanzee Behaviors

**Authors**: *Xiaoxuan Ma, Stephan P. Kaufhold, Jiajun Su, Wentao Zhu, Jack Terwilliger, Andres Meza, Yixin Zhu, Federico Rossano, Yizhou Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57a95cd3898bf4912269848a01f53620-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/57a95cd3898bf4912269848a01f53620-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Understanding the behavior of non-human primates is crucial for improving animal welfare, modeling social behavior, and gaining insights into distinctively human and phylogenetically shared behaviors. However, the lack of datasets on non-human primate behavior hinders in-depth exploration of primate social interactions, posing challenges to research on our closest living relatives. To address these limitations, we present ChimpACT, a comprehensive dataset for quantifying the longitudinal behavior and social relations of chimpanzees within a social group. Spanning from 2015 to 2018, ChimpACT features videos of a group of over 20 chimpanzees residing at the Leipzig Zoo, Germany, with a particular focus on documenting the developmental trajectory of one young male, Azibo. ChimpACT is both comprehensive and challenging, consisting of 163 videos with a cumulative 160,500 frames, each richly annotated with detection, identification, pose estimation, and fine-grained spatiotemporal behavior labels. We benchmark representative methods of three tracks on ChimpACT: (i) tracking and identification, (ii) pose estimation, and (iii) spatiotemporal action detection of the chimpanzees. Our experiments reveal that ChimpACT offers ample opportunities for both devising new methods and adapting existing ones to solve fundamental computer vision tasks applied to chimpanzee groups, such as detection, pose estimation, and behavior analysis, ultimately deepening our comprehension of communication and sociality in non-human primates.

----

## [0] Energy Transformer

**Authors**: *Benjamin Hoover, Yuchen Liang, Bao Pham, Rameswar Panda, Hendrik Strobelt, Duen Horng Chau, Mohammed J. Zaki, Dmitry Krotov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html)

**Abstract**:

Our work combines aspects of three promising paradigms in machine learning, namely, attention mechanism, energy-based models, and associative memory. Attention is the power-house driving modern deep learning successes, but it lacks clear theoretical foundations. Energy-based models allow a principled approach to discriminative and generative tasks, but the design of the energy functional is not straightforward. At the same time, Dense Associative Memory models or Modern Hopfield Networks have a well-established theoretical foundation, and allow an intuitive design of the energy function. We propose a novel architecture, called the Energy Transformer (or ET for short), that uses a sequence of attention layers that are purposely designed to minimize a specifically engineered energy function, which is responsible for representing the relationships between the tokens. In this work, we introduce the theoretical foundations of ET, explore its empirical capabilities using the image completion task, and obtain strong quantitative results on the graph anomaly detection and graph classification tasks.

----

## [0] Theoretical and Practical Perspectives on what Influence Functions Do

**Authors**: *Andrea Schioppa, Katja Filippova, Ivan Titov, Polina Zablotskaia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57bb27b9be6ad04019ae3cea2b540872-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57bb27b9be6ad04019ae3cea2b540872-Abstract-Conference.html)

**Abstract**:

Influence functions (IF) have been seen as a technique for explaining model predictions through the lens of the training data. Their utility is assumed to be in identifying training examples "responsible" for a prediction so that, for example, correcting a prediction is possible by intervening on those examples (removing or editing them) and retraining the model. However, recent empirical studies have shown that the existing methods of estimating IF predict the leave-one-out-and-retrain effect poorly. In order to understand the mismatch between the theoretical promise and the practical results, we analyse five assumptions made by IF methods which are problematic for modern-scale deep neural networks and which concern convexity, numeric stability, training trajectory and parameter divergence. This allows us to clarify what can be expected theoretically from IF. We show that while most assumptions can be addressed successfully, the parameter divergence poses a clear limitation on the predictive power of IF: influence fades over training time even with deterministic training. We illustrate this theoretical result with BERT and ResNet models.Another conclusion from the theoretical analysis is that IF are still useful for model debugging and correcting even though some of the assumptions made in prior work do not hold: using natural language processing and computer vision tasks, we verify that mis-predictions can be successfully corrected by taking only a few fine-tuning steps on influential examples.

----

## [0] trajdata: A Unified Interface to Multiple Human Trajectory Datasets

**Authors**: *Boris Ivanovic, Guanyu Song, Igor Gilitschenski, Marco Pavone*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57bb67dbe17bfb660c8c63d089ea05b9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/57bb67dbe17bfb660c8c63d089ea05b9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The field of trajectory forecasting has grown significantly in recent years, partially owing to the release of numerous large-scale, real-world human trajectory datasets for autonomous vehicles (AVs) and pedestrian motion tracking. While such datasets have been a boon for the community, they each use custom and unique data formats and APIs, making it cumbersome for researchers to train and evaluate methods across multiple datasets. To remedy this, we present trajdata: a unified interface to multiple human trajectory datasets. At its core, trajdata provides a simple, uniform, and efficient representation and API for trajectory and map data. As a demonstration of its capabilities, in this work we conduct a comprehensive empirical evaluation of existing trajectory datasets, providing users with a rich understanding of the data underpinning much of current pedestrian and AV motion forecasting research, and proposing suggestions for future datasets from these insights. trajdata is permissively licensed (Apache 2.0) and can be accessed online at https://github.com/NVlabs/trajdata.

----

## [0] On Sparse Modern Hopfield Model

**Authors**: *Jerry Yao-Chieh Hu, Donglin Yang, Dennis Wu, Chenwei Xu, Bo-Yu Chen, Han Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/57bc0a850255e2041341bf74c7e2b9fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/57bc0a850255e2041341bf74c7e2b9fa-Abstract-Conference.html)

**Abstract**:

We introduce the sparse modern Hopfield model as a sparse extension of the modern Hopfield model.Like its dense counterpart, the sparse modern Hopfield model equips a memory-retrieval dynamics whose one-step approximation corresponds to the sparse attention mechanism. Theoretically, our key contribution is a principled derivation of a closed-form sparse Hopfield energy using the convex conjugate of the sparse entropic regularizer.Building upon this, we derive the sparse memory retrieval dynamics from the sparse energy function and show its one-step approximation is equivalent to the sparse-structured attention.Importantly, we provide a sparsity-dependent memory retrieval error bound which is provably tighter than its dense analog.The conditions for the benefits of sparsity to arise are therefore identified and discussed.In addition, we show that the sparse modern Hopfield model maintains the robust theoretical properties of its dense counterpart, including rapid fixed point convergence and exponential memory capacity.Empirically, we use both synthetic and real-world datasets to demonstrate that the sparse Hopfield model outperforms its dense counterpart in many situations.

----



[Go to the previous page](NIPS-2023-list1100.md)

[Go to the next page](NIPS-2023-list1102.md)

[Go to the catalog section](README.md)