## [1400] Decentralized Q-learning in Zero-sum Markov Games

        **Authors**: *Muhammed O. Sayin, Kaiqing Zhang, David S. Leslie, Tamer Basar, Asuman E. Ozdaglar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/985e9a46e10005356bbaf194249f6856-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/985e9a46e10005356bbaf194249f6856-Abstract.html)

        **Abstract**:

        We study multi-agent  reinforcement learning (MARL) in infinite-horizon discounted zero-sum Markov games. We focus on the practical but  challenging setting of decentralized MARL, where agents make decisions without coordination by a centralized controller, but only based on their own payoffs and local actions executed. The agents need not observe the opponent's actions or payoffs, possibly being even  oblivious to the presence of the opponent, nor be aware of the zero-sum structure of the underlying game, a setting also referred to as radically uncoupled in the literature of learning in games. In this paper, we develop a radically uncoupled Q-learning dynamics that is both rational and convergent: the learning dynamics converges to the best response to the opponent's strategy when the opponent follows an asymptotically stationary strategy;  when both agents adopt the learning dynamics, they converge to the Nash equilibrium of the game. The key challenge in this decentralized setting is the non-stationarity of the environment from an agent's perspective, since both her own payoffs and the system evolution depend on the actions of other agents, and each agent adapts her policies simultaneously and independently. To address this issue, we develop a two-timescale learning dynamics where each agent updates her local Q-function and value function estimates concurrently, with the latter happening at a slower timescale.

        ----

        ## [1401] Fast Certified Robust Training with Short Warmup

        **Authors**: *Zhouxing Shi, Yihan Wang, Huan Zhang, Jinfeng Yi, Cho-Jui Hsieh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/988f9153ac4fd966ea302dd9ab9bae15-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/988f9153ac4fd966ea302dd9ab9bae15-Abstract.html)

        **Abstract**:

        Recently, bound propagation based certified robust training methods have been proposed for training neural networks with certifiable robustness guarantees. Despite that state-of-the-art (SOTA) methods including interval bound propagation (IBP) and CROWN-IBP have per-batch training complexity similar to standard neural network training, they usually use a long warmup schedule with hundreds or thousands epochs to reach SOTA performance and are thus still costly. In this paper, we identify two important issues in existing methods, namely exploded bounds at initialization, and the imbalance in ReLU activation states and improve IBP training. These two issues make certified training difficult and unstable, and thereby long warmup schedules were needed in prior works. To mitigate these issues and conduct faster certified training with shorter warmup, we propose three improvements based on IBP training: 1) We derive a new weight initialization method for IBP training; 2) We propose to fully add Batch Normalization (BN) to each layer in the model, since we find BN can reduce the imbalance in ReLU activation states; 3) We also design regularization to explicitly tighten certified bounds and balance ReLU activation states during wamrup. We are able to obtain 65.03% verified error on CIFAR-10 ($\epsilon=\frac{8}{255}$) and 82.36% verified error on TinyImageNet ($\epsilon=\frac{1}{255}$) using very short training schedules (160 and 80 total epochs, respectively), outperforming literature SOTA trained with hundreds or thousands epochs under the same network architecture. The code is available at https://github.com/shizhouxing/Fast-Certified-Robust-Training.

        ----

        ## [1402] Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices

        **Authors**: *Federico López, Beatrice Pozzetti, Steve Trettel, Michael Strube, Anna Wienhard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/98c39996bf1543e974747a2549b3107c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/98c39996bf1543e974747a2549b3107c-Abstract.html)

        **Abstract**:

        We propose the use of the vector-valued distance to compute distances and extract geometric information from the manifold of symmetric positive definite matrices (SPD), and develop gyrovector calculus, constructing analogs of vector space operations in this curved space. We implement these operations and showcase their versatility in the tasks of knowledge graph completion, item recommendation, and question answering. In experiments, the SPD models outperform their equivalents in Euclidean and hyperbolic space. The vector-valued distance allows us to visualize embeddings, showing that the models learn to disentangle representations of positive samples from negative ones.

        ----

        ## [1403] Improved Transformer for High-Resolution GANs

        **Authors**: *Long Zhao, Zizhao Zhang, Ting Chen, Dimitris N. Metaxas, Han Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/98dce83da57b0395e163467c9dae521b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/98dce83da57b0395e163467c9dae521b-Abstract.html)

        **Abstract**:

        Attention-based models, exemplified by the Transformer, can effectively model long range dependency, but suffer from the quadratic complexity of self-attention operation, making them difficult to be adopted for high-resolution image generation based on Generative Adversarial Networks (GANs). In this paper, we introduce two key ingredients to Transformer to address this challenge. First, in low-resolution stages of the generative process, standard global self-attention is replaced with the proposed multi-axis blocked self-attention which allows efficient mixing of local and global attention. Second, in high-resolution stages, we drop self-attention while only keeping multi-layer perceptrons reminiscent of the implicit neural function. To further improve the performance, we introduce an additional self-modulation component based on cross-attention. The resulting model, denoted as HiT, has a nearly linear computational complexity with respect to the image size and thus directly scales to synthesizing high definition images. We show in the experiments that the proposed HiT achieves state-of-the-art FID scores of 30.83 and 2.95 on unconditional ImageNet $128 \times 128$ and FFHQ $256 \times 256$, respectively, with a reasonable throughput. We believe the proposed HiT is an important milestone for generators in GANs which are completely free of convolutions. Our code is made publicly available at https://github.com/google-research/hit-gan.

        ----

        ## [1404] Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence

        **Authors**: *Xue Yang, Xiaojiang Yang, Jirui Yang, Qi Ming, Wentao Wang, Qi Tian, Junchi Yan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html)

        **Abstract**:

        Existing rotated object detectors are mostly inherited from the horizontal detection paradigm, as the latter has evolved into a well-developed area. However, these detectors are difficult to perform prominently in high-precision detection due to the limitation of current regression loss design, especially for objects with large aspect ratios. Taking the perspective that horizontal detection is a special case for rotated object detection, in this paper, we are motivated to change the design of rotation regression loss from induction paradigm to deduction methodology, in terms of the relation between rotation and horizontal detection. We show that one essential challenge is how to modulate the coupled parameters in the rotation regression loss, as such the estimated parameters can influence to each other during the dynamic joint optimization, in an adaptive and synergetic way. Specifically, we first convert the rotated bounding box into a 2-D Gaussian distribution, and then calculate the Kullback-Leibler Divergence (KLD) between the Gaussian distributions as the regression loss. By analyzing the gradient of each parameter, we show that KLD (and its derivatives) can dynamically adjust the parameter gradients according to the characteristics of the object. For instance, it will adjust the importance (gradient weight) of the angle parameter according to the aspect ratio. This mechanism can be vital for high-precision detection as a slight angle error would cause a serious accuracy drop for large aspect ratios objects. More importantly, we have proved that KLD is scale invariant. We further show that the KLD loss can be degenerated into the popular Ln-norm loss for horizontal detection. Experimental results on seven datasets using different detectors show its consistent superiority, and codes are available at https://github.com/yangxue0827/RotationDetection.

        ----

        ## [1405] On Locality of Local Explanation Models

        **Authors**: *Sahra Ghalebikesabi, Lucile Ter-Minassian, Karla DiazOrdaz, Chris C. Holmes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/995665640dc319973d3173a74a03860c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/995665640dc319973d3173a74a03860c-Abstract.html)

        **Abstract**:

        Shapley values provide model agnostic feature attributions for model outcome at a particular instance by simulating feature absence under a global population distribution. The use of a global population can lead to potentially misleading results when local model behaviour is of interest. Hence we consider the formulation of  neighbourhood reference distributions that improve the local interpretability of Shapley values. By doing so, we find that the Nadaraya-Watson estimator, a well-studied kernel regressor, can be expressed as a self-normalised importance sampling estimator. Empirically, we observe that Neighbourhood Shapley values identify meaningful sparse  feature relevance attributions that provide insight into local model behaviour, complimenting conventional Shapley analysis. They also increase on-manifold explainability and robustness to the construction of adversarial classifiers.

        ----

        ## [1406] FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling

        **Authors**: *Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, Takahiro Shinozaki*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/995693c15f439e3d189b06e89d145dd5-Abstract.html)

        **Abstract**:

        The recently proposed FixMatch achieved state-of-the-art results on most semi-supervised learning (SSL) benchmarks. However, like other modern SSL algorithms, FixMatch uses a pre-defined constant threshold for all classes to select unlabeled data that contribute to the training, thus failing to consider different learning status and learning difficulties of different classes. To address this issue, we propose Curriculum Pseudo Labeling (CPL), a curriculum learning approach to leverage unlabeled data according to the model's learning status. The core of CPL is to flexibly adjust thresholds for different classes at each time step to let pass informative unlabeled data and their pseudo labels. CPL does not introduce additional parameters or computations (forward or backward propagation). We apply CPL to FixMatch and call our improved algorithm FlexMatch. FlexMatch achieves state-of-the-art performance on a variety of SSL benchmarks, with especially strong performances when the labeled data are extremely limited or when the task is challenging. For example, FlexMatch achieves 13.96% and 18.96% error rate reduction over FixMatch on CIFAR-100 and STL-10 datasets respectively, when there are only 4 labels per class. CPL also significantly boosts the convergence speed, e.g., FlexMatch can use only 1/5 training time of FixMatch to achieve even better performance. Furthermore, we show that CPL can be easily adapted to other SSL algorithms and remarkably improve their performances. We open-source our code at https://github.com/TorchSSL/TorchSSL.

        ----

        ## [1407] Relative Flatness and Generalization

        **Authors**: *Henning Petzka, Michael Kamp, Linara Adilova, Cristian Sminchisescu, Mario Boley*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/995f5e03890b029865f402e83a81c29d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/995f5e03890b029865f402e83a81c29d-Abstract.html)

        **Abstract**:

        Flatness of the loss curve is conjectured to be connected to the generalization ability of machine learning models, in particular neural networks. While it has been empirically observed that flatness measures consistently correlate strongly with generalization, it is still an open theoretical problem why and under which circumstances flatness is connected to generalization, in particular in light of reparameterizations that change certain flatness measures but leave generalization unchanged. We investigate the connection between flatness and generalization by relating it to the interpolation from representative data, deriving notions of representativeness, and feature robustness. The notions allow us to rigorously connect flatness and generalization and to identify conditions under which the connection holds. Moreover, they give rise to a novel, but natural relative flatness measure that correlates strongly with generalization, simplifies to ridge regression for ordinary least squares, and solves the reparameterization issue.

        ----

        ## [1408] The Image Local Autoregressive Transformer

        **Authors**: *Chenjie Cao, Yuxin Hong, Xiang Li, Chengrong Wang, Chengming Xu, Yanwei Fu, Xiangyang Xue*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9996535e07258a7bbfd8b132435c5962-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9996535e07258a7bbfd8b132435c5962-Abstract.html)

        **Abstract**:

        Recently, AutoRegressive (AR) models for the whole image generation empowered by transformers have achieved comparable or even better performance compared to Generative Adversarial Networks (GANs). Unfortunately, directly applying such AR models to edit/change local image regions, may suffer from the problems of missing global information, slow inference speed, and information leakage of local guidance. To address these limitations, we propose a novel model -- image Local Autoregressive Transformer (iLAT), to better facilitate the locally guided image synthesis. Our iLAT learns the novel local discrete representations, by the newly proposed local autoregressive (LA) transformer of the attention mask and convolution mechanism. Thus iLAT can efficiently synthesize the local image regions by key guidance information. Our iLAT is evaluated on various locally guided image syntheses, such as pose-guided person image synthesis and face editing. Both quantitative and qualitative results show the efficacy of our model.

        ----

        ## [1409] Towards Multi-Grained Explainability for Graph Neural Networks

        **Authors**: *Xiang Wang, Ying-Xin Wu, An Zhang, Xiangnan He, Tat-Seng Chua*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/99bcfcd754a98ce89cb86f73acc04645-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/99bcfcd754a98ce89cb86f73acc04645-Abstract.html)

        **Abstract**:

        When a graph neural network (GNN) made a prediction, one raises question about explainability: “Which fraction of the input graph is most inﬂuential to the model’s decision?” Producing an answer requires understanding the model’s inner workings in general and emphasizing the insights on the decision for the instance at hand. Nonetheless, most of current approaches focus only on one aspect: (1) local explainability, which explains each instance independently, thus hardly exhibits the class-wise patterns; and (2) global explainability, which systematizes the globally important patterns, but might be trivial in the local context. This dichotomy limits the ﬂexibility and effectiveness of explainers greatly. A performant paradigm towards multi-grained explainability is until-now lacking and thus a focus of our work. In this work, we exploit the pre-training and ﬁne-tuning idea to develop our explainer and generate multi-grained explanations. Speciﬁcally, the pre-training phase accounts for the contrastivity among different classes, so as to highlight the class-wise characteristics from a global view; afterwards, the ﬁne-tuning phase adapts the explanations in the local context. Experiments on both synthetic and real-world datasets show the superiority of our explainer, in terms of AUC on explaining graph classiﬁcation over the leading baselines. Our codes and datasets are available at https://github.com/Wuyxin/ReFine.

        ----

        ## [1410] Behavior From the Void: Unsupervised Active Pre-Training

        **Authors**: *Hao Liu, Pieter Abbeel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/99bf3d153d4bf67d640051a1af322505-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/99bf3d153d4bf67d640051a1af322505-Abstract.html)

        **Abstract**:

        We introduce a new unsupervised pre-training method for reinforcement learning called APT, which stands for Active Pre-Training. APT learns behaviors and representations by actively searching for novel states in reward-free environments. The key novel idea is to explore the environment by maximizing a non-parametric entropy computed in an abstract representation space, which avoids challenging density modeling and consequently allows our approach to scale much better in environments that have high-dimensional observations (e.g., image observations). We empirically evaluate APT by exposing task-specific reward after a long unsupervised pre-training phase. In Atari games, APT achieves human-level performance on 12 games and obtains highly competitive performance compared to canonical fully supervised RL algorithms. On DMControl suite, APT beats all baselines in terms of asymptotic performance and data efficiency and dramatically improves performance on tasks that are extremely difficult to train from scratch.

        ----

        ## [1411] Autonomous Reinforcement Learning via Subgoal Curricula

        **Authors**: *Archit Sharma, Abhishek Gupta, Sergey Levine, Karol Hausman, Chelsea Finn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/99c83c904d0d64fbef50d919a5c66a80-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/99c83c904d0d64fbef50d919a5c66a80-Abstract.html)

        **Abstract**:

        Reinforcement learning (RL) promises to enable autonomous acquisition of complex behaviors for diverse agents. However, the success of current reinforcement learning algorithms is predicated on an often under-emphasised requirement -- each trial needs to start from a fixed initial state distribution. Unfortunately, resetting the environment to its initial state after each trial requires substantial amount of human supervision and extensive instrumentation of the environment which defeats the goal of autonomous acquisition of complex behaviors. In this work, we propose Value-accelerated Persistent Reinforcement Learning (VaPRL), which generates a curriculum of initial states such that the agent can bootstrap on the success of easier tasks to efficiently learn harder tasks. The agent also learns to reach the initial states proposed by the curriculum, minimizing the reliance on human interventions into the learning. We observe that VaPRL reduces the interventions required by three orders of magnitude compared to episodic RL while outperforming prior state-of-the art methods for reset-free RL both in terms of sample efficiency and asymptotic performance on a variety of simulated robotics problems.

        ----

        ## [1412] Statistically and Computationally Efficient Linear Meta-representation Learning

        **Authors**: *Kiran Koshy Thekumparampil, Prateek Jain, Praneeth Netrapalli, Sewoong Oh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/99e7e6ce097324aceb45f98299ceb621-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/99e7e6ce097324aceb45f98299ceb621-Abstract.html)

        **Abstract**:

        In typical few-shot learning, each task is not equipped with enough data to be learned in isolation. To cope with such data scarcity, meta-representation learning methods train across many related tasks to find a shared (lower-dimensional) representation of the data where all tasks can be solved accurately. It is hypothesized that any new arriving tasks can be rapidly trained on this low-dimensional representation using only a few samples. Despite the practical successes of this approach, its statistical and computational properties are less understood. Moreover, the prescribed algorithms in these studies have little resemblance to those used in practice or they are computationally intractable. To understand and explain the success of popular meta-representation learning approaches such as ANIL, MetaOptNet, R2D2, and OML, we study a alternating gradient-descent minimization (AltMinGD) method (and its variant alternating minimization (AltMin)) which underlies the aforementioned methods. For a simple but canonical setting of shared linear representations, we show that AltMinGD achieves nearly-optimal estimation error, requiring only $\Omega(\mathrm{polylog}\,d)$ samples per task. This agrees with the observed efficacy of this algorithm in the practical few-shot learning scenarios.

        ----

        ## [1413] Decentralized Learning in Online Queuing Systems

        **Authors**: *Flore Sentenac, Etienne Boursier, Vianney Perchet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/99ef04eb612baf0e86671a5109e22154-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/99ef04eb612baf0e86671a5109e22154-Abstract.html)

        **Abstract**:

        Motivated by packet routing in computer networks, online queuing systems are composed of queues receiving packets at different rates. Repeatedly, they send packets to servers, each of them treating only at most one packet at a time. In the centralized case, the number of accumulated packets remains bounded (i.e., the system is stable) as long as the ratio between service rates and arrival rates is larger than $1$. In the decentralized case, individual no-regret strategies ensures stability when this ratio is larger than $2$. Yet, myopically minimizing  regret disregards the long term effects due to the carryover of packets to further rounds. On the other hand, minimizing long term costs leads to stable Nash equilibria as soon as the ratio exceeds $\frac{e}{e-1}$. Stability with decentralized learning strategies  with a ratio below $2$ was a major remaining question. We first argue that for ratios up to $2$, cooperation is required for stability of learning strategies, as  selfish minimization of policy regret, a patient notion of regret, might indeed still be unstable in this case. We therefore consider cooperative queues and propose the first learning decentralized algorithm guaranteeing stability of the system as long as the ratio of rates is larger than $1$, thus reaching performances comparable to centralized strategies.

        ----

        ## [1414] Explainable Semantic Space by Grounding Language to Vision with Cross-Modal Contrastive Learning

        **Authors**: *Yizhen Zhang, Minkyu Choi, Kuan Han, Zhongming Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a1335ef5ffebb0de9d089c4182e4868-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a1335ef5ffebb0de9d089c4182e4868-Abstract.html)

        **Abstract**:

        In natural language processing, most models try to learn semantic representations merely from texts. The learned representations encode the “distributional semantics” but fail to connect to any knowledge about the physical world. In contrast, humans learn language by grounding concepts in perception and action and the brain encodes “grounded semantics” for cognition. Inspired by this notion and recent work in vision-language learning, we design a two-stream model for grounding language learning in vision. The model includes a VGG-based visual stream and a Bert-based language stream. The two streams merge into a joint representational space. Through cross-modal contrastive learning, the model first learns to align visual and language representations with the MS COCO dataset. The model further learns to retrieve visual objects with language queries through a cross-modal attention module and to infer the visual relations between the retrieved objects through a bilinear operator with the Visual Genome dataset. After training, the model’s language stream is a stand-alone language model capable of embedding concepts in a visually grounded semantic space. This semantic space manifests principal dimensions explainable with human intuition and neurobiological knowledge. Word embeddings in this semantic space are predictive of human-defined norms of semantic features and are segregated into perceptually distinctive clusters. Furthermore, the visually grounded language model also enables compositional language understanding based on visual knowledge and multimodal image search with queries based on images, texts, or their combinations.

        ----

        ## [1415] BulletTrain: Accelerating Robust Neural Network Training via Boundary Example Mining

        **Authors**: *Weizhe Hua, Yichi Zhang, Chuan Guo, Zhiru Zhang, G. Edward Suh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a1756fd0c741126d7bbd4b692ccbd91-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a1756fd0c741126d7bbd4b692ccbd91-Abstract.html)

        **Abstract**:

        Neural network robustness has become a central topic in machine learning in recent years. Most training algorithms that improve the model's robustness to adversarial and common corruptions also introduce a large computational overhead, requiring as many as ten times the number of forward and backward passes in order to converge. To combat this inefficiency, we propose BulletTrain, a boundary example mining technique to drastically reduce the computational cost of robust training. Our key observation is that only a small fraction of examples are beneficial for improving robustness. BulletTrain dynamically predicts these important examples and optimizes robust training algorithms to focus on the important examples. We apply our technique to several existing robust training algorithms and achieve a 2.2x speed-up for TRADES and MART on CIFAR-10 and a 1.7x speed-up for AugMix on CIFAR-10-C and CIFAR-100-C without any reduction in clean and robust accuracy.

        ----

        ## [1416] Neural Distance Embeddings for Biological Sequences

        **Authors**: *Gabriele Corso, Zhitao Ying, Michal Pándy, Petar Velickovic, Jure Leskovec, Pietro Liò*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a1de01f893e0d2551ecbb7ce4dc963e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a1de01f893e0d2551ecbb7ce4dc963e-Abstract.html)

        **Abstract**:

        The development of data-dependent heuristics and representations for biological sequences that reflect their evolutionary distance is critical for large-scale biological research. However, popular machine learning approaches, based on continuous Euclidean spaces, have struggled with the discrete combinatorial formulation of the edit distance that models evolution and the hierarchical relationship that characterises real-world datasets. We present Neural Distance Embeddings (NeuroSEED), a general framework to embed sequences in geometric vector spaces, and illustrate the effectiveness of the hyperbolic space that captures the hierarchical structure and provides an average 38% reduction in embedding RMSE against the best competing geometry. The capacity of the framework and the significance of these improvements are then demonstrated devising supervised and unsupervised NeuroSEED approaches to multiple core tasks in bioinformatics. Benchmarked with common baselines, the proposed approaches display significant accuracy and/or runtime improvements on real-world datasets. As an example for hierarchical clustering, the proposed pretrained and from-scratch methods match the quality of competing baselines with 30x and 15x runtime reduction, respectively.

        ----

        ## [1417] Fitting summary statistics of neural data with a differentiable spiking network simulator

        **Authors**: *Guillaume Bellec, Shuqi Wang, Alireza Modirshanechi, Johanni Brea, Wulfram Gerstner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a32ff36c65e8ba30915a21b7bd76506-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a32ff36c65e8ba30915a21b7bd76506-Abstract.html)

        **Abstract**:

        Fitting network models to neural activity is an important tool in neuroscience. A popular approach is to model a brain area with a probabilistic recurrent spiking network whose parameters maximize the likelihood of the recorded activity. Although this is widely used, we show that the resulting model does not produce realistic neural activity. To correct for this, we suggest to augment the log-likelihood with terms that measure the dissimilarity between simulated and recorded activity. This dissimilarity is defined via summary statistics commonly used in neuroscience and the optimization is efficient because it relies on back-propagation through the stochastically simulated spike trains. We analyze this method theoretically and show empirically that it generates more realistic activity statistics. We find that it improves upon other fitting algorithms for spiking network models like GLMs (Generalized Linear Models) which do not usually rely on back-propagation. This new fitting algorithm also enables the consideration of hidden neurons which is otherwise notoriously hard, and we show that it can be crucial when trying to infer the network connectivity from spike recordings.

        ----

        ## [1418] PerSim: Data-Efficient Offline Reinforcement Learning with Heterogeneous Agents via Personalized Simulators

        **Authors**: *Anish Agarwal, Abdullah Alomar, Varkey Alumootil, Devavrat Shah, Dennis Shen, Zhi Xu, Cindy Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a3f263a5e5f63006098a05cd7491997-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a3f263a5e5f63006098a05cd7491997-Abstract.html)

        **Abstract**:

        We consider offline reinforcement learning (RL) with heterogeneous agents under severe data scarcity, i.e., we only observe a single historical trajectory for every agent under an unknown, potentially sub-optimal policy. We find that the performance of state-of-the-art offline and model-based RL methods degrade significantly given such limited data availability, even for commonly perceived "solved" benchmark settings such as "MountainCar" and "CartPole". To address this challenge, we propose PerSim, a model-based offline RL approach which first learns a personalized simulator for each agent by collectively using the historical trajectories across all agents, prior to learning a policy. We do so by positing that the transition dynamics across agents can be represented as a latent function of latent factors associated with agents, states, and actions; subsequently, we theoretically establish that this function is well-approximated by a "low-rank" decomposition of separable agent, state, and action latent functions. This representation suggests a simple, regularized neural network architecture to effectively learn the transition dynamics per agent, even with scarce, offline data. We perform extensive experiments across several benchmark environments and RL methods. The consistent improvement of our approach, measured in terms of both state dynamics prediction and eventual reward, confirms the efficacy of our framework in leveraging limited historical data to simultaneously learn personalized policies across agents.

        ----

        ## [1419] Online Sign Identification: Minimization of the Number of Errors in Thresholding Bandits

        **Authors**: *Reda Ouhamma, Odalric-Ambrym Maillard, Vianney Perchet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a3f54913bf27e648d1759c18d007165-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a3f54913bf27e648d1759c18d007165-Abstract.html)

        **Abstract**:

        In the fixed budget thresholding bandit problem, an algorithm sequentially allocates a budgeted number of samples to different distributions. It then predicts whether the mean of each distribution is larger or lower than a given threshold. We introduce a large family of algorithms (containing most existing relevant ones), inspired by the Frank-Wolfe algorithm, and provide a thorough yet generic analysis of their performance. This allowed us to construct new explicit algorithms, for a broad class of problems, whose losses are within a small constant factor of the non-adaptive oracle ones. Quite interestingly, we observed that adaptive methodsempirically greatly out-perform non-adaptive oracles, an uncommon behavior in standard online learning settings, such as regret minimization. We explain this surprising phenomenon on an insightful toy problem.

        ----

        ## [1420] All Tokens Matter: Token Labeling for Training Better Vision Transformers

        **Authors**: *Zihang Jiang, Qibin Hou, Li Yuan, Daquan Zhou, Yujun Shi, Xiaojie Jin, Anran Wang, Jiashi Feng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a49a25d845a483fae4be7e341368e36-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a49a25d845a483fae4be7e341368e36-Abstract.html)

        **Abstract**:

        In this paper, we present token labeling---a new training objective for training high-performance vision transformers (ViTs). Different from the standard training objective of ViTs that computes the classification loss on an additional trainable class token, our proposed one takes advantage of all the image patch tokens to compute the training loss in a dense manner. Specifically, token labeling reformulates the image classification problem into multiple token-level recognition problems and assigns each patch token with an individual location-specific supervision generated by a machine annotator. Experiments show that token labeling can clearly and consistently improve the performance of various ViT models across a wide spectrum. For a vision transformer with 26M learnable parameters serving as an example, with token labeling, the model can achieve 84.4% Top-1 accuracy on ImageNet. The result can be further increased to 86.4% by slightly scaling the model size up to 150M, delivering the minimal-sized model among previous models (250M+) reaching 86%. We also show that token labeling can clearly improve the generalization capability of the pretrained models on downstream tasks with dense prediction, such as semantic segmentation.  Our code and model are publiclyavailable at https://github.com/zihangJiang/TokenLabeling.

        ----

        ## [1421] Partition and Code: learning how to compress graphs

        **Authors**: *Giorgos Bouritsas, Andreas Loukas, Nikolaos Karalias, Michael M. Bronstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a4d6e8685bd057e4f68930bd7c8ecc0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a4d6e8685bd057e4f68930bd7c8ecc0-Abstract.html)

        **Abstract**:

        Can we use machine learning to compress graph data? The absence of ordering in graphs poses a significant challenge to conventional compression algorithms, limiting their attainable gains as well as their ability to discover relevant patterns. On the other hand, most graph compression approaches rely on domain-dependent handcrafted representations and cannot adapt to different underlying graph distributions. This work aims to establish the necessary principles a lossless graph compression method should follow to approach the entropy storage lower bound. Instead of making rigid assumptions about the graph distribution, we formulate the compressor as a probabilistic model that can be learned from data and generalise to unseen instances. Our “Partition and Code” framework entails three steps: first, a partitioning algorithm decomposes the graph into subgraphs, then these are mapped to the elements of a small dictionary on which we learn a probability distribution, and finally,  an entropy encoder translates the representation into bits.  All the components (partitioning, dictionary and distribution) are parametric and can be trained with gradient descent. We theoretically compare the compression quality of several graph encodings and prove, under mild conditions, that PnC achieves compression gains that grow either linearly or quadratically with the number of vertices. Empirically, PnC yields significant compression improvements on diverse real-world networks.

        ----

        ## [1422] Knowledge-inspired 3D Scene Graph Prediction in Point Cloud

        **Authors**: *Shoulong Zhang, Shuai Li, Aimin Hao, Hong Qin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a555403384fc12f931656dea910e334-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a555403384fc12f931656dea910e334-Abstract.html)

        **Abstract**:

        Prior knowledge integration helps identify semantic entities and their relationships in a graphical representation, however, its meaningful abstraction and intervention remain elusive. This paper advocates a knowledge-inspired 3D scene graph prediction method solely based on point clouds. At the mathematical modeling level, we formulate the task as two sub-problems: knowledge learning and scene graph prediction with learned prior knowledge. Unlike conventional methods that learn knowledge embedding and regular patterns from encoded visual information, we propose to suppress the misunderstandings caused by appearance similarities and other perceptual confusion. At the network design level, we devise a graph auto-encoder to automatically extract class-dependent representations and topological patterns from the one-hot class labels and their intrinsic graphical structures, so that the prior knowledge can avoid perceptual errors and noises. We further devise a scene graph prediction model to predict credible relationship triplets by incorporating the related prototype knowledge with perceptual information. Comprehensive experiments confirm that, our method can successfully learn representative knowledge embedding, and the obtained prior knowledge can effectively enhance the accuracy of relationship predictions. Our thorough evaluations indicate the new method can achieve the state-of-the-art performance compared with other scene graph prediction methods.

        ----

        ## [1423] Online Variational Filtering and Parameter Learning

        **Authors**: *Andrew Campbell, Yuyang Shi, Thomas Rainforth, Arnaud Doucet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a6a1aaafe73c572b7374828b03a1881-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a6a1aaafe73c572b7374828b03a1881-Abstract.html)

        **Abstract**:

        We present a variational method for online state estimation and parameter learning in state-space models (SSMs), a ubiquitous class of latent variable models for sequential data. As per standard batch variational techniques, we use stochastic gradients to simultaneously optimize a lower bound on the log evidence with respect to both model parameters and a variational approximation of the states' posterior distribution. However, unlike existing approaches, our method is able to operate in an entirely online manner, such that historic observations do not require revisitation after being incorporated and the cost of updates at each time step remains constant, despite the growing dimensionality of the joint posterior distribution of the states. This is achieved by utilizing backward decompositions of this joint posterior distribution and of its variational approximation, combined with Bellman-type recursions for the evidence lower bound and its gradients. We demonstrate the performance of this methodology across several examples, including high-dimensional SSMs and sequential Variational Auto-Encoders.

        ----

        ## [1424] Heavy Ball Neural Ordinary Differential Equations

        **Authors**: *Hedi Xia, Vai Suliafu, Hangjie Ji, Tan M. Nguyen, Andrea L. Bertozzi, Stanley J. Osher, Bao Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9a86d531e19ec6f5937ad1373bb118bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9a86d531e19ec6f5937ad1373bb118bd-Abstract.html)

        **Abstract**:

        We propose heavy ball neural ordinary differential equations (HBNODEs), leveraging the continuous limit of the classical momentum accelerated gradient descent, to improve neural ODEs (NODEs) training and inference. HBNODEs have two properties that imply practical advantages over NODEs: (i) The adjoint state of an HBNODE also satisfies an HBNODE, accelerating both forward and backward ODE solvers, thus significantly reducing the number of function evaluations (NFEs) and improving the utility of the trained models. (ii) The spectrum of HBNODEs is well structured, enabling effective learning of long-term dependencies from complex sequential data. We verify the advantages of HBNODEs over NODEs on benchmark tasks, including image classification, learning complex dynamics, and sequential modeling. Our method requires remarkably fewer forward and backward NFEs, is more accurate, and learns long-term dependencies more effectively than the other ODE-based neural network models. Code is available at \url{https://github.com/hedixia/HeavyBallNODE}.

        ----

        ## [1425] Structure learning in polynomial time: Greedy algorithms, Bregman information, and exponential families

        **Authors**: *Goutham Rajendran, Bohdan Kivva, Ming Gao, Bryon Aragam*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9ab8a8a9349eb1dd73ce155ce64c80fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9ab8a8a9349eb1dd73ce155ce64c80fa-Abstract.html)

        **Abstract**:

        Greedy algorithms have long been a workhorse for learning graphical models, and more broadly for learning statistical models with sparse structure. In the context of learning directed acyclic graphs, greedy algorithms are popular despite their worst-case exponential runtime. In practice, however, they are very efficient. We provide new insight into this phenomenon by studying a general greedy score-based algorithm for learning DAGs. Unlike edge-greedy algorithms such as the popular GES and hill-climbing algorithms, our approach is vertex-greedy and requires at most a polynomial number of score evaluations. We then show how recent polynomial-time algorithms for learning DAG models are a special case of this algorithm, thereby illustrating how these order-based algorithms can be rigourously interpreted as score-based algorithms. This observation suggests new score functions and optimality conditions based on the duality between Bregman divergences and exponential families, which we explore in detail. Explicit sample and computational complexity bounds are derived. Finally, we provide extensive experiments suggesting that this algorithm indeed optimizes the score in a variety of settings.

        ----

        ## [1426] On the Sample Complexity of Learning under Geometric Stability

        **Authors**: *Alberto Bietti, Luca Venturi, Joan Bruna*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9ac5a6d86e8924182271bd820acbce0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9ac5a6d86e8924182271bd820acbce0e-Abstract.html)

        **Abstract**:

        Many supervised learning problems involve high-dimensional data such as images, text, or graphs. In order to make efficient use of data, it is often useful to leverage certain geometric priors in the problem at hand, such as invariance to translations, permutation subgroups, or stability to small deformations. We study the sample complexity of learning problems where the target function presents such invariance and stability properties, by considering spherical harmonic decompositions of such functions on the sphere. We provide non-parametric rates of convergence for kernel methods, and show improvements in sample complexity by a factor equal to the size of the group when using an invariant kernel over the group, compared to the corresponding non-invariant kernel. These improvements are valid when the sample size is large enough, with an asymptotic behavior that depends on spectral properties of the group. Finally, these gains are extended beyond invariance groups to also cover geometric stability to small deformations, modeled here as subsets (not necessarily subgroups) of permutations.

        ----

        ## [1427] SIMILAR: Submodular Information Measures Based Active Learning In Realistic Scenarios

        **Authors**: *Suraj Kothawade, Nathan Beck, KrishnaTeja Killamsetty, Rishabh K. Iyer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9af08cda54faea9adf40a201794183cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9af08cda54faea9adf40a201794183cf-Abstract.html)

        **Abstract**:

        Active  learning  has  proven  to  be  useful  for  minimizing  labeling  costs  by selecting  the  most  informative  samples. However,  existing  active  learning methods do not work well in realistic scenarios such as imbalance or rare classes,out-of-distribution data in the unlabeled set, and redundancy.  In this work, we propose SIMILAR (Submodular Information Measures based actIve LeARning), a unified active learning framework using recently proposed submodular information measures (SIM) as acquisition functions. We argue that SIMILAR not only works in standard active learning but also easily extends to the realistic settings considered above and acts as a one-stop solution for active learning that is scalable to large real-world datasets. Empirically, we show that SIMILAR significantly outperforms existing active learning algorithms by as much as ~5%−18%in the case of rare classes and ~5%−10%in the case of out-of-distribution data on several image classification tasks like CIFAR-10, MNIST, and ImageNet.

        ----

        ## [1428] Monte Carlo Tree Search With Iteratively Refining State Abstractions

        **Authors**: *Samuel Sokota, Caleb Ho, Zaheen Farraz Ahmad, J. Zico Kolter*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9b0ead00a217ea2c12e06a72eec4923f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9b0ead00a217ea2c12e06a72eec4923f-Abstract.html)

        **Abstract**:

        Decision-time planning is the process of constructing a transient, local policy with the intent of using it to make the immediate decision. Monte Carlo tree search (MCTS), which has been leveraged to great success in Go, chess, shogi, Hex, Atari, and other settings, is perhaps the most celebrated decision-time planning algorithm. Unfortunately, in its original form, MCTS can degenerate to one-step search in domains with stochasticity. Progressive widening is one way to ameliorate this issue, but we argue that it possesses undesirable properties for some settings. In this work, we present a method, called abstraction refining, for extending MCTS to stochastic environments which, unlike progressive widening, leverages the geometry of the state space. We argue that leveraging the geometry of the space can offer advantages. To support this claim, we present a series of experimental examples in which abstraction refining outperforms progressive widening, given equal simulation budgets.

        ----

        ## [1429] Flattening Sharpness for Dynamic Gradient Projection Memory Benefits Continual Learning

        **Authors**: *Danruo Deng, Guangyong Chen, Jianye Hao, Qiong Wang, Pheng-Ann Heng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9b16759a62899465ab21e2e79d2ef75c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9b16759a62899465ab21e2e79d2ef75c-Abstract.html)

        **Abstract**:

        The backpropagation networks are notably susceptible to catastrophic forgetting, where networks tend to forget previously learned skills upon learning new ones. To address such the 'sensitivity-stability' dilemma, most previous efforts have been contributed to minimizing the empirical risk with different parameter regularization terms and episodic memory, but rarely exploring the usages of the weight loss landscape. In this paper, we investigate the relationship between the weight loss landscape and sensitivity-stability in the continual learning scenario, based on which, we propose a novel method, Flattening Sharpness for Dynamic Gradient Projection Memory (FS-DGPM). In particular, we introduce a soft weight to represent the importance of each basis representing past tasks in GPM, which can be adaptively learned during the learning process, so that less important bases can be dynamically released to improve the sensitivity of new skill learning. We further introduce Flattening Sharpness (FS) to reduce the generalization gap by explicitly regulating the flatness of the weight loss landscape of all seen tasks. As demonstrated empirically, our proposed method consistently outperforms baselines with the superior ability to learn new skills while alleviating forgetting effectively.

        ----

        ## [1430] Taxonomizing local versus global structure in neural network loss landscapes

        **Authors**: *Yaoqing Yang, Liam Hodgkinson, Ryan Theisen, Joe Zou, Joseph E. Gonzalez, Kannan Ramchandran, Michael W. Mahoney*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9b72e31dac81715466cd580a448cf823-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9b72e31dac81715466cd580a448cf823-Abstract.html)

        **Abstract**:

        Viewing neural network models in terms of their loss landscapes has a long history in the statistical mechanics approach to learning, and in recent years it has received attention within machine learning proper. Among other things, local metrics (such as the smoothness of the loss landscape) have been shown to correlate with global properties of the model (such as good generalization performance). Here, we perform a detailed empirical analysis of the loss landscape structure of thousands of neural network models, systematically varying learning tasks, model architectures, and/or quantity/quality of data. By considering a range of metrics that attempt to capture different aspects of the loss landscape, we demonstrate that the best test accuracy is obtained when: the loss landscape is globally well-connected; ensembles of trained models are more similar to each other; and models converge to locally smooth regions. We also show that globally poorly-connected landscapes can arise when models are small or when they are trained to lower quality data; and that, if the loss landscape is globally poorly-connected, then training to zero loss can actually lead to worse test accuracy. Our detailed empirical results shed light on phases of learning (and consequent double descent behavior), fundamental versus incidental determinants of good generalization, the role of load-like and temperature-like parameters in the learning process, different influences on the loss landscape from model and data, and the relationships between local and global metrics, all topics of recent interest.

        ----

        ## [1431] Learning Models for Actionable Recourse

        **Authors**: *Alexis Ross, Himabindu Lakkaraju, Osbert Bastani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9b82909c30456ac902e14526e63081d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9b82909c30456ac902e14526e63081d4-Abstract.html)

        **Abstract**:

        As machine learning models are increasingly deployed in high-stakes domains such as legal and financial decision-making, there has been growing interest in post-hoc methods for generating counterfactual explanations. Such explanations provide individuals adversely impacted by predicted outcomes (e.g., an applicant denied a loan) with recourse---i.e., a description of how they can change their features to obtain a positive outcome. We propose a novel algorithm that leverages adversarial training and PAC confidence sets to learn models that theoretically guarantee recourse to affected individuals with high probability without sacrificing accuracy. We demonstrate the efficacy of our approach via extensive experiments on real data.

        ----

        ## [1432] Efficient and Accurate Gradients for Neural SDEs

        **Authors**: *Patrick Kidger, James Foster, Xuechen Li, Terry J. Lyons*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9ba196c7a6e89eafd0954de80fc1b224-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9ba196c7a6e89eafd0954de80fc1b224-Abstract.html)

        **Abstract**:

        Neural SDEs combine many of the best qualities of both RNNs and SDEs, and as such are a natural choice for modelling many types of temporal dynamics. They offer memory efficiency, high-capacity function approximation, and strong priors on model space. Neural SDEs may be trained as VAEs or as GANs; in either case it is necessary to backpropagate through the SDE solve. In particular this may be done by constructing a backwards-in-time SDE whose solution is the desired parameter gradients. However, this has previously suffered from severe speed and accuracy issues, due to high computational complexity, numerical errors in the SDE solve, and the cost of reconstructing Brownian motion. Here, we make several technical innovations to overcome these issues. First, we introduce the \textit{reversible Heun method}: a new SDE solver that is algebraically reversible -- which reduces numerical gradient errors to almost zero, improving several test metrics by substantial margins over state-of-the-art. Moreover it requires half as many function evaluations as comparable solvers, giving up to a $1.98\times$ speedup. Next, we introduce the \textit{Brownian interval}. This is a new and computationally efficient way of exactly sampling \textit{and reconstructing} Brownian motion; this is in contrast to previous reconstruction techniques that are both approximate and relatively slow. This gives up to a $10.6\times$ speed improvement over previous techniques. After that, when specifically training Neural SDEs as GANs (Kidger et al. 2021), we demonstrate how SDE-GANs may be trained through careful weight clipping and choice of activation function. This reduces computational cost (giving up to a $1.87\times$ speedup), and removes the truncation errors of the double adjoint required for gradient penalty, substantially improving several test metrics. Altogether these techniques offer substantial improvements over the state-of-the-art, with respect to both training speed and with respect to classification, prediction, and MMD test metrics. We have contributed implementations of all of our techniques to the \texttt{torchsde} library to help facilitate their adoption.

        ----

        ## [1433] EIGNN: Efficient Infinite-Depth Graph Neural Networks

        **Authors**: *Juncheng Liu, Kenji Kawaguchi, Bryan Hooi, Yiwei Wang, Xiaokui Xiao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9bd5ee6fe55aaeb673025dbcb8f939c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9bd5ee6fe55aaeb673025dbcb8f939c1-Abstract.html)

        **Abstract**:

        Graph neural networks (GNNs) are widely used for modelling graph-structured data in numerous applications. However, with their inherently finite aggregation layers, existing GNN models may not be able to effectively capture long-range dependencies in the underlying graphs. Motivated by this limitation, we propose a GNN model with infinite depth, which we call Efficient Infinite-Depth Graph Neural Networks (EIGNN), to efficiently capture very long-range dependencies. We theoretically derive a closed-form solution of EIGNN which makes training an infinite-depth GNN model tractable. We then further show that we can achieve more efficient computation for training EIGNN by using eigendecomposition. The empirical results of comprehensive experiments on synthetic and real-world datasets show that EIGNN has a better ability to capture long-range dependencies than recent baselines, and consistently achieves state-of-the-art performance. Furthermore, we show that our model is also more robust against both noise and adversarial perturbations on node features.

        ----

        ## [1434] Fractal Structure and Generalization Properties of Stochastic Optimization Algorithms

        **Authors**: *Alexander Camuto, George Deligiannidis, Murat A. Erdogdu, Mert Gürbüzbalaban, Umut Simsekli, Lingjiong Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9bdb8b1faffa4b3d41779bb495d79fb9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9bdb8b1faffa4b3d41779bb495d79fb9-Abstract.html)

        **Abstract**:

        Understanding generalization in deep learning has been one of the major challenges in statistical learning theory over the last decade. While recent work has illustrated that the dataset and the training algorithm must be taken into account in order to obtain meaningful generalization bounds, it is still theoretically not clear which properties of the data and the algorithm determine the generalization performance. In this study, we approach this problem from a dynamical systems theory perspective and represent stochastic optimization algorithms as \emph{random iterated function systems} (IFS). Well studied in the dynamical systems literature, under mild assumptions, such IFSs can be shown to be ergodic with an invariant measure that is often supported on sets with a \emph{fractal structure}. As our main contribution, we prove that the generalization error of a stochastic optimization algorithm can be bounded based on the `complexity' of the fractal structure that underlies its invariant measure. Then, by leveraging results from dynamical systems theory, we show that the generalization error can be explicitly linked to the choice of the algorithm (e.g., stochastic gradient descent -- SGD), algorithm hyperparameters (e.g., step-size, batch-size), and the geometry of the problem (e.g., Hessian of the loss). We further specialize our results to specific problems (e.g., linear/logistic regression, one hidden-layered neural networks) and algorithms (e.g., SGD and preconditioned variants), and obtain analytical estimates for our bound. For modern neural networks, we develop an efficient algorithm to compute the developed bound and support our theory with various experiments on neural networks.

        ----

        ## [1435] An Infinite-Feature Extension for Bayesian ReLU Nets That Fixes Their Asymptotic Overconfidence

        **Authors**: *Agustinus Kristiadi, Matthias Hein, Philipp Hennig*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9be40cee5b0eee1462c82c6964087ff9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9be40cee5b0eee1462c82c6964087ff9-Abstract.html)

        **Abstract**:

        A Bayesian treatment can mitigate overconfidence in ReLU nets around the training data. But far away from them, ReLU Bayesian neural networks (BNNs) can still underestimate uncertainty and thus be asymptotically overconfident. This issue arises since the output variance of a BNN with finitely many features is quadratic in the distance from the data region. Meanwhile, Bayesian linear models with ReLU features converge, in the infinite-width limit, to a particular Gaussian process (GP) with a variance that grows cubically so that no asymptotic overconfidence can occur. While this may seem of mostly theoretical interest, in this work, we show that it can be used in practice to the benefit of BNNs. We extend finite ReLU BNNs with infinite ReLU features via the GP and show that the resulting model is asymptotically maximally uncertain far away from the data while the BNNs' predictive power is unaffected near the data. Although the resulting model approximates a full GP posterior, thanks to its structure, it can be applied post-hoc to any pre-trained ReLU BNN at a low cost.

        ----

        ## [1436] Bandit Phase Retrieval

        **Authors**: *Tor Lattimore, Botao Hao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9c36b930df0e0e8b05d4e1fcb4cdef27-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9c36b930df0e0e8b05d4e1fcb4cdef27-Abstract.html)

        **Abstract**:

        We study a bandit version of phase retrieval where the learner chooses actions $(A_t)_{t=1}^n$ in the $d$-dimensional unit ball and the expected reward is $\langle A_t, \theta_\star \rangle^2$ with $\theta_\star \in \mathbb R^d$ an unknown parameter vector. We prove an upper bound on the minimax cumulative regret in this problem of $\smash{\tilde \Theta(d \sqrt{n})}$, which matches known lower bounds up to logarithmic factors and improves on the best known upper bound by a factor of $\smash{\sqrt{d}}$. We also show that the minimax simple regret is $\smash{\tilde \Theta(d / \sqrt{n})}$ and that this is only achievable by an adaptive algorithm. Our analysis shows that an apparently convincing heuristic for guessing lower bounds can be misleading and that uniform bounds on the information ratio for information-directed sampling (Russo and Van Roy, 2014) are not sufficient for optimal regret.

        ----

        ## [1437] Lower Bounds on Metropolized Sampling Methods for Well-Conditioned Distributions

        **Authors**: *Yin Tat Lee, Ruoqi Shen, Kevin Tian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9c4e6233c6d5ff637e7984152a3531d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9c4e6233c6d5ff637e7984152a3531d5-Abstract.html)

        **Abstract**:

        We give lower bounds on the performance of two of the most popular sampling methods in practice, the Metropolis-adjusted Langevin algorithm (MALA) and multi-step Hamiltonian Monte Carlo (HMC) with a leapfrog integrator, when applied to well-conditioned distributions. Our main result is a nearly-tight lower bound of $\widetilde{\Omega}(\kappa d)$ on the mixing time of MALA from an exponentially warm start, matching a line of algorithmic results \cite{DwivediCW018, ChenDWY19, LeeST20a} up to logarithmic factors and answering an open question of \cite{ChewiLACGR20}. We also show that a polynomial dependence on dimension is necessary for the relaxation time of HMC under any number of leapfrog steps, and bound the gains achievable by changing the step count. Our HMC analysis draws upon a novel connection between leapfrog integration and Chebyshev polynomials, which may be of independent interest.

        ----

        ## [1438] Taming Communication and Sample Complexities in Decentralized Policy Evaluation for Cooperative Multi-Agent Reinforcement Learning

        **Authors**: *Xin Zhang, Zhuqing Liu, Jia Liu, Zhengyuan Zhu, Songtao Lu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9c51a13764ca629f439f6accbb4ec413-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9c51a13764ca629f439f6accbb4ec413-Abstract.html)

        **Abstract**:

        Cooperative multi-agent reinforcement learning (MARL) has received increasing attention in recent years and has found many scientific and engineering applications. However, a key challenge arising from many cooperative MARL algorithm designs (e.g., the actor-critic framework) is the policy evaluation problem, which can only be conducted in a {\em decentralized} fashion. In this paper, we focus on decentralized MARL policy evaluation with nonlinear function approximation, which is often seen in deep MARL. We first show that the empirical decentralized MARL policy evaluation problem can be reformulated as a decentralized nonconvex-strongly-concave minimax saddle point problem. We then develop a decentralized gradient-based descent ascent algorithm called GT-GDA that enjoys a convergence rate of $\mathcal{O}(1/T)$. To further reduce the sample complexity, we propose two decentralized stochastic optimization algorithms called GT-SRVR and GT-SRVRI, which enhance GT-GDA by variance reduction techniques. We show that all algorithms all enjoy an $\mathcal{O}(1/T)$ convergence rate to a stationary point of the reformulated minimax problem. Moreover, the fast convergence rates of GT-SRVR and GT-SRVRI imply $\mathcal{O}(\epsilon^{-2})$ communication complexity and $\mathcal{O}(m\sqrt{n}\epsilon^{-2})$ sample complexity, where $m$ is the number of agents and $n$ is the length of trajectories. To our knowledge, this paper is the first work that achieves both $\mathcal{O}(\epsilon^{-2})$ sample complexity and $\mathcal{O}(\epsilon^{-2})$ communication complexity in decentralized policy evaluation for cooperative MARL. Our extensive experiments also corroborate the theoretical performance of our proposed decentralized policy evaluation algorithms.

        ----

        ## [1439] Federated Graph Classification over Non-IID Graphs

        **Authors**: *Han Xie, Jing Ma, Li Xiong, Carl Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9c6947bd95ae487c81d4e19d3ed8cd6f-Abstract.html)

        **Abstract**:

        Federated learning has emerged as an important paradigm for training machine learning models in different domains. For graph-level tasks such as graph classification, graphs can also be regarded as a special type of data samples, which can be collected and stored in separate local systems. Similar to other domains, multiple local systems, each holding a small set of graphs, may benefit from collaboratively training a powerful graph mining model, such as the popular graph neural networks (GNNs). To provide more motivation towards such endeavors, we analyze real-world graphs from different domains to confirm that they indeed share certain graph properties that are statistically significant compared with random graphs. However, we also find that different sets of graphs, even from the same domain or same dataset, are non-IID regarding both graph structures and node features. To handle this, we propose a graph clustered federated learning (GCFL) framework that dynamically finds clusters of local systems based on the gradients of GNNs, and theoretically justify that such clusters can reduce the structure and feature heterogeneity among graphs owned by the local systems. Moreover, we observe the gradients of GNNs to be rather fluctuating in GCFL which impedes high-quality clustering, and design a gradient sequence-based clustering mechanism based on dynamic time warping (GCFL+). Extensive experimental results and in-depth analysis demonstrate the effectiveness of our proposed frameworks.

        ----

        ## [1440] SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning

        **Authors**: *Talip Ucar, Ehsan Hajiramezanali, Lindsay Edwards*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9c8661befae6dbcd08304dbf4dcaf0db-Abstract.html)

        **Abstract**:

        Self-supervised learning has been shown to be very effective in learning useful representations, and yet much of the success is achieved in data types such as images, audio, and text. The success is mainly enabled by taking advantage of spatial, temporal, or semantic structure in the data through augmentation. However, such structure may not exist in tabular datasets commonly used in fields such as healthcare, making it difficult to design an effective augmentation method, and hindering a similar progress in tabular data setting. In this paper, we introduce a new framework, Subsetting features of Tabular data (SubTab), that turns the task of learning from tabular data into a multi-view representation learning problem by dividing the input features to multiple subsets. We argue that reconstructing the data from the subset of its features rather than its corrupted version in an autoencoder setting can better capture its underlying latent representation. In this framework, the joint representation can be expressed as the aggregate of latent variables of the subsets at test time, which we refer to as collaborative inference. Our experiments show that the SubTab achieves the state of the art (SOTA) performance of 98.31% on MNIST in tabular setting, on par with CNN-based SOTA models, and surpasses existing baselines on three other real-world datasets by a significant margin.

        ----

        ## [1441] Convergence Rates of Stochastic Gradient Descent under Infinite Noise Variance

        **Authors**: *Hongjian Wang, Mert Gürbüzbalaban, Lingjiong Zhu, Umut Simsekli, Murat A. Erdogdu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9cdf26568d166bc6793ef8da5afa0846-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9cdf26568d166bc6793ef8da5afa0846-Abstract.html)

        **Abstract**:

        Recent studies have provided both empirical and theoretical evidence illustrating that heavy tails can emerge in stochastic gradient descent (SGD) in various scenarios. Such heavy tails potentially result in iterates with diverging variance, which hinders the use of conventional convergence analysis techniques that rely on the existence of the second-order moments. In this paper, we provide convergence guarantees for SGD under a state-dependent and heavy-tailed noise with a potentially infinite variance, for a class of strongly convex objectives. In the case where the $p$-th moment of the noise exists for some $p\in [1,2)$, we first identify a condition on the Hessian, coined `$p$-positive (semi-)definiteness', that leads to an interesting interpolation between the positive semi-definite cone ($p=2$) and the cone of diagonally dominant matrices with non-negative diagonal entries ($p=1$). Under this condition, we provide a convergence rate for the distance to the global optimum in $L^p$. Furthermore, we provide a generalized central limit theorem, which shows that the properly scaled Polyak-Ruppert averaging converges weakly to a multivariate $\alpha$-stable random vector.Our results indicate that even under heavy-tailed noise with infinite variance, SGD can converge to the global optimum without necessitating any modification neither to the loss function nor to the algorithm itself, as typically required in robust statistics.We demonstrate the implications of our resultsover misspecified models, in the presence of heavy-tailed data.

        ----

        ## [1442] Conflict-Averse Gradient Descent for Multi-task learning

        **Authors**: *Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, Qiang Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)

        **Abstract**:

        The goal of multi-task learning is to enable more efficient learning than single task learning by sharing model structures for a diverse set of tasks. A standard multi-task learning objective is to minimize the average loss across all tasks. While straightforward, using this objective often results in much worse final performance for each task than learning them independently. A major challenge in optimizing a multi-task model is the conflicting gradients, where gradients of different task objectives are not well aligned so that following the average gradient direction can be detrimental to specific tasks' performance. Previous work has proposed several heuristics to manipulate the task gradients for mitigating this problem. But most of them lack convergence guarantee and/or could converge to any Pareto-stationary point.In this paper, we introduce Conflict-Averse Gradient descent (CAGrad) which minimizes the average loss function, while leveraging the worst local improvement of individual tasks to regularize the algorithm trajectory. CAGrad balances the objectives automatically and still provably converges to a minimum over the average loss. It includes the regular gradient descent (GD) and the multiple gradient descent algorithm (MGDA) in the multi-objective optimization (MOO) literature as special cases. On a series of challenging multi-task supervised learning and reinforcement learning tasks, CAGrad achieves improved performance over prior state-of-the-art multi-objective gradient manipulation methods.

        ----

        ## [1443] Amortized Synthesis of Constrained Configurations Using a Differentiable Surrogate

        **Authors**: *Xingyuan Sun, Tianju Xue, Szymon Rusinkiewicz, Ryan P. Adams*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9d38e6eab92b2aeb0a83b570188d5a1a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9d38e6eab92b2aeb0a83b570188d5a1a-Abstract.html)

        **Abstract**:

        In design, fabrication, and control problems, we are often faced with the task of synthesis, in which we must generate an object or configuration that satisfies a set of constraints while maximizing one or more objective functions. The synthesis problem is typically characterized by a physical process in which many different realizations may achieve the goal. This many-to-one map presents challenges to the supervised learning of feed-forward synthesis, as the set of viable designs may have a complex structure. In addition, the non-differentiable nature of many physical simulations prevents efficient direct optimization. We address both of these problems with a two-stage neural network architecture that we may consider to be an autoencoder. We first learn the decoder: a differentiable surrogate that approximates the many-to-one physical realization process. We then learn the encoder, which maps from goal to design, while using the fixed decoder to evaluate the quality of the realization. We evaluate the approach on two case studies: extruder path planning in additive manufacturing and constrained soft robot inverse kinematics. We compare our approach to direct optimization of the design using the learned surrogate, and to supervised learning of the synthesis problem. We find that our approach produces higher quality solutions than supervised learning, while being competitive in quality with direct optimization, at a greatly reduced computational cost.

        ----

        ## [1444] Efficient First-Order Contextual Bandits: Prediction, Allocation, and Triangular Discrimination

        **Authors**: *Dylan J. Foster, Akshay Krishnamurthy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9d684c589d67031a627ad33d59db65e5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9d684c589d67031a627ad33d59db65e5-Abstract.html)

        **Abstract**:

        A recurring theme in statistical learning, online learning, and beyond is that faster convergence rates are possible for problems with low noise, often quantified by the performance of the best hypothesis; such results are known as first-order or small-loss guarantees. While first-order guarantees are relatively well understood in statistical and online learning, adapting to low noise in contextual bandits (and more broadly, decision making) presents major algorithmic challenges. In a COLT 2017 open problem, Agarwal, Krishnamurthy, Langford, Luo, and Schapire asked whether first-order guarantees are even possible for contextual bandits and---if so---whether they can be attained by efficient algorithms. We give a resolution to this question by providing an optimal and efficient reduction from contextual bandits to online regression with the logarithmic (or, cross-entropy) loss. Our algorithm is simple and practical, readily accommodates rich function classes, and requires no distributional assumptions beyond realizability. In a large-scale empirical evaluation, we find that our approach typically outperforms  comparable non-first-order methods.On the technical side, we show that the logarithmic loss and an information-theoretic quantity called the triangular discrimination play a fundamental role in obtaining first-order guarantees, and we combine this observation with new refinements to the regression oracle reduction framework of Foster and Rakhlin (2020). The use of triangular discrimination yields novel results even for the classical statistical learning model, and we anticipate that it will find broader use.

        ----

        ## [1445] Distributed Estimation with Multiple Samples per User: Sharp Rates and Phase Transition

        **Authors**: *Jayadev Acharya, Clément L. Canonne, Yuhan Liu, Ziteng Sun, Himanshu Tyagi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9d740bd0f36aaa312c8d504e28c42163-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9d740bd0f36aaa312c8d504e28c42163-Abstract.html)

        **Abstract**:

        We obtain tight minimax rates for the problem of distributed estimation of discrete distributions under communication constraints, where $n$ users observing $m $ samples each can broadcast only $\ell$  bits. Our main result is a tight characterization (up to logarithmic factors) of the error rate as a function of $m$, $\ell$, the domain size, and the number of users under most regimes of interest. While previous work focused on the setting where each user only holds one sample, we show that as $m$ grows the $\ell_1$ error rate gets reduced by a factor of $\sqrt{m}$ for small $m$. However, for large $m$ we observe an interesting phase transition: the dependence of the error rate on the communication constraint $\ell$ changes from $1/\sqrt{2^{\ell}}$ to $1/\sqrt{\ell}$.

        ----

        ## [1446] Revisiting Deep Learning Models for Tabular Data

        **Authors**: *Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)

        **Abstract**:

        The existing literature on deep learning for tabular data proposes a wide range of novel architectures and reports competitive results on various datasets. However, the proposed models are usually not properly compared to each other and existing works often use different benchmarks and experiment protocols. As a result, it is unclear for both researchers and practitioners what models perform best. Additionally, the field still lacks effective baselines, that is, the easy-to-use models that provide competitive performance across different problems.In this work, we perform an overview of the main families of DL architectures for tabular data and raise the bar of baselines in tabular DL by identifying two simple and powerful deep architectures. The first one is a ResNet-like architecture which turns out to be a strong baseline that is often missing in prior works. The second model is our simple adaptation of the Transformer architecture for tabular data, which outperforms other solutions on most tasks. Both models are compared to many existing architectures on a diverse set of tasks under the same training and tuning protocols. We also compare the best DL models with Gradient Boosted Decision Trees and conclude that there is still no universally superior solution. The source code is available at https://github.com/yandex-research/rtdl.

        ----

        ## [1447] Backdoor Attack with Imperceptible Input and Latent Modification

        **Authors**: *Khoa D. Doan, Yingjie Lao, Ping Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9d99197e2ebf03fc388d09f1e94af89b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9d99197e2ebf03fc388d09f1e94af89b-Abstract.html)

        **Abstract**:

        Recent studies have shown that deep neural networks (DNN) are vulnerable to various adversarial attacks. In particular, an adversary can inject a stealthy backdoor into a model such that the compromised model will behave normally without the presence of the trigger. Techniques for generating backdoor images that are visually imperceptible from clean images have also been developed recently, which further enhance the stealthiness of the backdoor attacks from the input space. Along with the development of attacks, defense against backdoor attacks is also evolving. Many existing countermeasures found that backdoor tends to leave tangible footprints in the latent or feature space, which can be utilized to mitigate backdoor attacks.In this paper, we extend the concept of imperceptible backdoor from the input space to the latent representation, which significantly improves the effectiveness against the existing defense mechanisms, especially those relying on the distinguishability between clean inputs and backdoor inputs in latent space. In the proposed framework, the trigger function will learn to manipulate the input by injecting imperceptible input noise while matching the latent representations of the clean and manipulated inputs via a Wasserstein-based regularization of the corresponding empirical distributions. We formulate such an objective as a non-convex and constrained optimization problem and solve the problem with an efficient stochastic alternating optimization procedure. We name the proposed backdoor attack as Wasserstein Backdoor (WB), which achieves a high attack success rate while being stealthy from both the input and latent spaces, as tested in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImagenet.

        ----

        ## [1448] SOPE: Spectrum of Off-Policy Estimators

        **Authors**: *Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, Scott Niekum*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9dd16e049becf4d5087c90a83fea403b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9dd16e049becf4d5087c90a83fea403b-Abstract.html)

        **Abstract**:

        Many sequential decision making problems are high-stakes and require off-policy evaluation (OPE) of a new policy using historical data collected using some other policy. One of the most common OPE techniques that provides unbiased estimates is trajectory based importance sampling (IS). However, due to the high variance of trajectory IS estimates, importance sampling methods based on state-action visitation distributions (SIS) have recently been adopted. Unfortunately, while SIS often provides lower variance estimates for long horizons, estimating the state-action distribution ratios can be challenging and lead to biased estimates. In this paper, we present a new perspective on this bias-variance trade-off and show the existence of a spectrum of estimators whose endpoints are SIS and IS. Additionally, we also establish a spectrum for doubly-robust and weighted version of these estimators. We provide empirical evidence that estimators in this spectrum can be used to trade-off between the bias and variance of IS and SIS and can achieve lower mean-squared error than both IS and SIS.

        ----

        ## [1449] Label-Imbalanced and Group-Sensitive Classification under Overparameterization

        **Authors**: *Ganesh Ramachandra Kini, Orestis Paraskevas, Samet Oymak, Christos Thrampoulidis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9dfcf16f0adbc5e2a55ef02db36bac7f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9dfcf16f0adbc5e2a55ef02db36bac7f-Abstract.html)

        **Abstract**:

        The goal in label-imbalanced and group-sensitive classification is to optimize relevant metrics such as balanced error and equal opportunity. Classical methods, such as weighted cross-entropy, fail when training deep nets to the terminal phase of training (TPT), that is training beyond zero training error. This observation has motivated recent flurry of activity in developing heuristic alternatives following the intuitive mechanism of promoting larger margin for minorities. In contrast to previous heuristics, we follow a principled analysis explaining how different loss adjustments affect margins. First, we prove that for all linear classifiers trained in TPT, it is necessary to introduce multiplicative, rather than additive, logit adjustments so that the interclass margins change appropriately. To show this, we discover a connection of the multiplicative CE modification to the cost-sensitive support-vector machines. Perhaps counterintuitively, we also find that, at the start of training, the same multiplicative weights can actually harm the minority classes. Thus, while additive adjustments are ineffective in the TPT, we show that they can speed up convergence by countering the initial negative effect of the multiplicative weights. Motivated by these findings, we formulate the vector-scaling (VS) loss, that captures existing techniques as special cases. Moreover, we introduce a natural extension of the VS-loss to group-sensitive classification, thus treating the two common types of imbalances (label/group) in a unifying way. Importantly, our experiments on state-of-the-art datasets are fully consistent with our theoretical insights and confirm the superior performance of our algorithms. Finally, for imbalanced Gaussian-mixtures data, we perform a generalization analysis, revealing tradeoffs between balanced / standard error and equal opportunity.

        ----

        ## [1450] Neural Program Generation Modulo Static Analysis

        **Authors**: *Rohan Mukherjee, Yeming Wen, Dipak Chaudhari, Thomas W. Reps, Swarat Chaudhuri, Christopher M. Jermaine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9e1a36515d6704d7eb7a30d783400e5d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9e1a36515d6704d7eb7a30d783400e5d-Abstract.html)

        **Abstract**:

        State-of-the-art neural models of source code tend to be evaluated on the generation of individual expressions and lines of code, and commonly fail on long-horizon tasks such as the generation of entire method bodies. We propose to address this deficiency using weak supervision from a static program analyzer. Our neurosymbolic method allows a deep generative model to symbolically compute, using calls to a static analysis tool, long-distance semantic relationships in the code that it has already generated. During training, the model observes these relationships and learns to generate programs conditioned on them. We apply our approach to the problem of generating entire Java methods given the remainder of the class that contains the method. Our experiments show that the approach substantially outperforms a state-of-the-art transformer and a model that explicitly tries to learn program semantics on this task, both in terms of producing programs free of basic semantic errors and in terms of syntactically matching the ground truth.

        ----

        ## [1451] Unfolding Taylor's Approximations for Image Restoration

        **Authors**: *Man Zhou, Xueyang Fu, Zeyu Xiao, Gang Yang, Aiping Liu, Zhiwei Xiong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9e3cfc48eccf81a0d57663e129aef3cb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9e3cfc48eccf81a0d57663e129aef3cb-Abstract.html)

        **Abstract**:

        Deep learning provides a new avenue for image restoration, which demands a delicate balance between fine-grained details and high-level contextualized information during recovering the latent clear image. In practice, however, existing methods empirically construct encapsulated end-to-end mapping networks without deepening into the rationality, and neglect the intrinsic prior knowledge of restoration task. To solve the above problems, inspired by Taylor’s Approximations, we unfold Taylor’s Formula to construct a novel framework for image restoration. We find the main part and the derivative part of Taylor’s Approximations take the same effect as the two competing goals of high-level contextualized information and spatial details of image restoration respectively. Specifically, our framework consists of two steps, which are correspondingly responsible for the mapping and derivative functions. The former first learns the high-level contextualized information and the later combines it with the degraded input to progressively recover local high-order spatial details. Our proposed framework is orthogonal to existing methods and thus can be easily integrated with them for further improvement, and extensive experiments demonstrate the effectiveness and scalability of our proposed framework.

        ----

        ## [1452] Metropolis-Hastings Data Augmentation for Graph Neural Networks

        **Authors**: *Hyeon-Jin Park, Seunghun Lee, Sihyeon Kim, Jinyoung Park, Jisu Jeong, Kyung-Min Kim, Jung-Woo Ha, Hyunwoo J. Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9e7ba617ad9e69b39bd0c29335b79629-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9e7ba617ad9e69b39bd0c29335b79629-Abstract.html)

        **Abstract**:

        Graph Neural Networks (GNNs) often suffer from weak-generalization due to sparsely labeled data despite their promising results on various graph-based tasks. Data augmentation is a prevalent remedy to improve the generalization ability of models in many domains. However, due to the non-Euclidean nature of data space and the dependencies between samples, designing effective augmentation on graphs is challenging. In this paper, we propose a novel framework Metropolis-Hastings Data Augmentation (MH-Aug) that draws augmented graphs from an explicit target distribution for semi-supervised learning. MH-Aug produces a sequence of augmented graphs from the target distribution enables flexible control of the strength and diversity of augmentation. Since the direct sampling from the complex target distribution is challenging, we adopt the Metropolis-Hastings algorithm to obtain the augmented samples. We also propose a simple and effective semi-supervised learning strategy with generated samples from MH-Aug. Our extensive experiments demonstrate that MH-Aug can generate a sequence of samples according to the target distribution to significantly improve the performance of GNNs.

        ----

        ## [1453] Strategic Behavior is Bliss: Iterative Voting Improves Social Welfare

        **Authors**: *Joshua Kavner, Lirong Xia*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9edcc1391c208ba0b503fe9a22574251-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9edcc1391c208ba0b503fe9a22574251-Abstract.html)

        **Abstract**:

        Recent work in iterative voting has defined the additive dynamic price of anarchy (ADPoA) as the difference in social welfare between the truthful and worst-case equilibrium profiles resulting from repeated strategic manipulations. While iterative plurality has been shown to only return alternatives with at most one less initial votes than the truthful winner, it is less understood how agents' welfare changes in equilibrium. To this end, we differentiate agents' utility from their manipulation mechanism and determine iterative plurality's ADPoA in the worst- and average-cases. We first prove that the worst-case ADPoA is linear in the number of agents. To overcome this negative result, we study the average-case ADPoA and prove that equilibrium winners have a constant order welfare advantage over the truthful winner in expectation. Our positive results illustrate the prospect for social welfare to increase due to strategic manipulation.

        ----

        ## [1454] Agnostic Reinforcement Learning with Low-Rank MDPs and Rich Observations

        **Authors**: *Ayush Sekhari, Christoph Dann, Mehryar Mohri, Yishay Mansour, Karthik Sridharan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9eed867b73ab1eab60583c9d4a789b1b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9eed867b73ab1eab60583c9d4a789b1b-Abstract.html)

        **Abstract**:

        There have been many recent advances on provably efficient Reinforcement Learning (RL) in problems with rich observation spaces. However, all these works share a strong realizability assumption about the optimal value function of the true MDP. Such realizability assumptions are often too strong to hold in practice. In this work, we consider the more realistic setting of agnostic RL with rich observation spaces and a fixed class of policies $\Pi$ that may not contain any near-optimal policy. We provide an algorithm for this setting whose error is bounded in terms of the rank $d$ of the underlying MDP.  Specifically, our algorithm enjoys a sample complexity bound of $\widetilde{O}\left((H^{4d} K^{3d} \log |\Pi|)/\epsilon^2\right)$ where $H$ is the length of episodes, $K$ is the number of actions and $\epsilon>0$ is the desired sub-optimality.  We also provide a nearly matching lower bound for this agnostic setting that shows that the exponential dependence on rank is unavoidable, without further assumptions.

        ----

        ## [1455] Functional Regularization for Reinforcement Learning via Learned Fourier Features

        **Authors**: *Alexander C. Li, Deepak Pathak*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f0609b9d45dd55bed75f892cf095fcf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f0609b9d45dd55bed75f892cf095fcf-Abstract.html)

        **Abstract**:

        We propose a simple architecture for deep reinforcement learning by embedding inputs into a learned Fourier basis and show that it improves the sample efficiency of both state-based and image-based RL. We perform infinite-width analysis of our architecture using the Neural Tangent Kernel and theoretically show that tuning the initial variance of the Fourier basis is equivalent to functional regularization of the learned deep network. That is, these learned Fourier features allow for adjusting the degree to which networks underfit or overfit different frequencies in the training data, and hence provide a controlled mechanism to improve the stability and performance of RL optimization. Empirically, this allows us to prioritize learning low-frequency functions and speed up learning by reducing networks' susceptibility to noise in the optimization process, such as during Bellman updates. Experiments on standard state-based and image-based RL benchmarks show clear benefits of our architecture over the baselines.

        ----

        ## [1456] Adaptive First-Order Methods Revisited: Convex Minimization without Lipschitz Requirements

        **Authors**: *Kimon Antonakopoulos, Panayotis Mertikopoulos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f16b57bdd4400066a83cd8eaa151c41-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f16b57bdd4400066a83cd8eaa151c41-Abstract.html)

        **Abstract**:

        We propose a new family of adaptive first-order methods for a class of convex minimization problems that may fail to be Lipschitz continuous or smooth in the standard sense. Specifically, motivated by a recent flurry of activity on non-Lipschitz (NoLips) optimization, we consider problems that are continuous or smooth relative to a reference Bregman function – as opposed to a global, ambient norm (Euclidean or otherwise). These conditions encompass a wide range ofproblems with singular objective, such as Fisher markets, Poisson tomography, D-design, and the like. In this setting, the application of existing order-optimal adaptive methods – like UnixGrad or AcceleGrad – is not possible, especially in the presence of randomness and uncertainty. The proposed method, adaptive mirror descent (AdaMir), aims to close this gap by concurrently achieving min-max optimal rates in problems that are relatively continuous or smooth, including stochastic ones.

        ----

        ## [1457] Adapting to function difficulty and growth conditions in private optimization

        **Authors**: *Hilal Asi, Daniel Levy, John C. Duchi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f820adf84bf8a1c259f464ba89ea11f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f820adf84bf8a1c259f464ba89ea11f-Abstract.html)

        **Abstract**:

        We develop algorithms for private stochastic convex optimization that adapt to the hardness of the specific function we wish to optimize. While previous work provide worst-case bounds for arbitrary convex functions, it is often the case that the function at hand belongs to a smaller class that enjoys faster rates. Concretely, we show that for functions exhibiting $\kappa$-growth around the optimum, i.e., $f(x) \ge f(x^\star) + \lambda \kappa^{-1} \|x-x^\star\|_2^\kappa$ for $\kappa > 1$, our algorithms improve upon the standard ${\sqrt{d}}/{n\varepsilon}$ privacy rate to the faster $({\sqrt{d}}/{n\varepsilon})^{\tfrac{\kappa}{\kappa - 1}}$. Crucially, they achieve these rates without knowledge of the growth constant $\kappa$ of the function. Our algorithms build upon the inverse sensitivity mechanism, which adapts to instance difficulty [2], and recent localization techniques in private optimization [25]. We complement our algorithms with matching lower bounds for these function classes and demonstrate that our adaptive algorithm is simultaneously (minimax) optimal over all $\kappa \ge 1+c$ whenever $c = \Theta(1)$.

        ----

        ## [1458] Support Recovery of Sparse Signals from a Mixture of Linear Measurements

        **Authors**: *Soumyabrata Pal, Arya Mazumdar, Venkata Gandikota*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f8785c7f9b578bec2c09e616568d270-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f8785c7f9b578bec2c09e616568d270-Abstract.html)

        **Abstract**:

        Recovery of support of a sparse vector from simple measurements is a widely studied problem, considered under the frameworks of compressed sensing, 1-bit compressed sensing, and more general single index models. We consider generalizations of this problem: mixtures of linear regressions, and mixtures of linear classifiers, where the goal is to recover supports of multiple sparse vectors using only a small number of possibly noisy linear, and 1-bit measurements respectively. The key challenge is that the measurements from different vectors are randomly mixed. Both of these problems have also received attention recently. In mixtures of linear classifiers, an observation corresponds to the side of the queried hyperplane a random unknown vector lies in; whereas in mixtures of linear regressions we observe the projection of a random unknown vector on the queried hyperplane. The primary step in recovering the unknown vectors from the mixture is to first identify the support of all the individual component vectors. In this work, we study the number of measurements sufficient for recovering the supports of all the component vectors in a mixture in both these models. We provide algorithms that use a number of measurements polynomial in $k, \log n$ and quasi-polynomial in $\ell$, to recover the support of all the $\ell$ unknown vectors in the mixture with high probability when each individual component is a $k$-sparse $n$-dimensional vector.

        ----

        ## [1459] Stochastic Gradient Descent-Ascent and Consensus Optimization for Smooth Games: Convergence Analysis under Expected Co-coercivity

        **Authors**: *Nicolas Loizou, Hugo Berard, Gauthier Gidel, Ioannis Mitliagkas, Simon Lacoste-Julien*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f96f36b7aae3b1ff847c26ac94c604e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f96f36b7aae3b1ff847c26ac94c604e-Abstract.html)

        **Abstract**:

        Two of the most prominent algorithms for solving unconstrained smooth games are the classical stochastic gradient descent-ascent (SGDA) and the recently introduced stochastic consensus optimization (SCO) [Mescheder et al., 2017]. SGDA is known to converge to a stationary point for specific classes of games, but current convergence analyses require a bounded variance assumption. SCO is used successfully for solving large-scale adversarial problems, but its convergence guarantees are limited to its deterministic variant. In this work, we introduce the expected co-coercivity condition, explain its benefits, and provide the first last-iterate convergence guarantees of SGDA and SCO under this condition for solving a class of stochastic variational inequality problems that are potentially non-monotone. We prove linear convergence of both methods to a neighborhood of the solution when they use constant step-size, and we propose insightful stepsize-switching rules to guarantee convergence to the exact solution. In addition, our convergence guarantees hold under the arbitrary sampling paradigm, and as such, we give insights into the complexity of minibatching.

        ----

        ## [1460] Tighter Expected Generalization Error Bounds via Wasserstein Distance

        **Authors**: *Borja Rodríguez Gálvez, Germán Bassi, Ragnar Thobaben, Mikael Skoglund*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f975093da0252e2c0ae181d74c90dc6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f975093da0252e2c0ae181d74c90dc6-Abstract.html)

        **Abstract**:

        This work presents several expected generalization error bounds based on the Wasserstein distance. More specifically, it introduces full-dataset, single-letter, and random-subset bounds, and their analogous in the randomized subsample setting from Steinke and Zakynthinou [1]. Moreover, when the loss function is bounded and the geometry of the space is ignored by the choice of the metric in the Wasserstein distance, these bounds recover from below (and thus, are tighter than) current bounds based on the relative entropy. In particular, they generate new, non-vacuous bounds based on the relative entropy. Therefore, these results can be seen as a bridge between works that account for the geometry of the hypothesis space and those based on the relative entropy, which is agnostic to such geometry. Furthermore, it is shown how to produce various new bounds based on different information measures (e.g., the lautum information or several $f$-divergences) based on these bounds and how to derive similar bounds with respect to the backward channel using the presented proof techniques.

        ----

        ## [1461] Unifying Width-Reduced Methods for Quasi-Self-Concordant Optimization

        **Authors**: *Deeksha Adil, Brian Bullins, Sushant Sachdeva*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9f9e8cba3700df6a947a8cf91035ab84-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9f9e8cba3700df6a947a8cf91035ab84-Abstract.html)

        **Abstract**:

        We provide several algorithms for constrained optimization of a large class of convex problems, including softmax, $\ell_p$ regression, and logistic regression. Central to our approach is the notion of width reduction, a technique which has proven immensely useful in the context of maximum flow [Christiano et al., STOC'11] and, more recently, $\ell_p$ regression [Adil et al., SODA'19], in terms of improving the iteration complexity from $O(m^{1/2})$ to $\tilde{O}(m^{1/3})$, where $m$ is the number of rows of the design matrix, and where each iteration amounts to a linear system solve. However, a considerable drawback is that these methods require both problem-specific potentials and individually tailored analyses.As our main contribution, we initiate a new direction of study by presenting the first \emph{unified} approach to achieving $m^{1/3}$-type rates. Notably, our method goes beyond these previously considered problems to more broadly capture \emph{quasi-self-concordant} losses, a class which has recently generated much interest and includes the well-studied problem of logistic regression, among others. In order to do so, we develop a unified width reduction method for carefully handling these losses based on a more general set of potentials. Additionally, we directly achieve $m^{1/3}$-type rates in the constrained setting without the need for any explicit acceleration schemes, thus naturally complementing recent work based on a ball-oracle approach [Carmon et al., NeurIPS'20].

        ----

        ## [1462] Bridging the Imitation Gap by Adaptive Insubordination

        **Authors**: *Luca Weihs, Unnat Jain, Iou-Jen Liu, Jordi Salvador, Svetlana Lazebnik, Aniruddha Kembhavi, Alexander G. Schwing*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9fc664916bce863561527f06a96f5ff3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9fc664916bce863561527f06a96f5ff3-Abstract.html)

        **Abstract**:

        In practice, imitation learning is preferred over pure reinforcement learning whenever it is possible to design a teaching agent to provide expert supervision. However, we show that when the teaching agent makes decisions with access to privileged information that is unavailable to the student, this information is marginalized during imitation learning, resulting in an "imitation gap" and, potentially, poor results. Prior work bridges this gap via a progression from imitation learning to reinforcement learning. While often successful, gradual progression fails for tasks that require frequent switches between exploration and memorization. To better address these tasks and alleviate the imitation gap we propose 'Adaptive Insubordination' (ADVISOR). ADVISOR dynamically weights imitation and reward-based reinforcement learning losses during training, enabling on-the-fly switching between imitation and exploration. On a suite of challenging tasks set within gridworlds, multi-agent particle environments, and high-fidelity 3D simulators, we show that on-the-fly switching with ADVISOR outperforms pure imitation, pure reinforcement learning, as well as their sequential and parallel combinations.

        ----

        ## [1463] Adversarial Robustness with Non-uniform Perturbations

        **Authors**: *Ecenaz Erdemir, Jeffrey Bickford, Luca Melis, Sergül Aydöre*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9fd98f856d3ca2086168f264a117ed7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9fd98f856d3ca2086168f264a117ed7c-Abstract.html)

        **Abstract**:

        Robustness of machine learning models is critical for security related applications, where real-world adversaries are uniquely focused on evading neural network based detectors. Prior work mainly focus on crafting adversarial examples (AEs) with small uniform norm-bounded perturbations across features to maintain the requirement of imperceptibility. However, uniform perturbations do not result in realistic AEs in domains such as malware, finance, and social networks. For these types of applications, features typically have some semantically meaningful dependencies. The key idea of our proposed approach is to enable non-uniform perturbations that can adequately represent these feature dependencies during adversarial training. We propose using characteristics of the empirical data distribution, both on correlations between the features and the importance of the features themselves. Using experimental datasets for malware classification, credit risk prediction, and spam detection, we show that our approach is more robust to real-world attacks. Finally, we present robustness certification utilizing non-uniform perturbation bounds, and show that non-uniform bounds achieve better certification.

        ----

        ## [1464] Container: Context Aggregation Networks

        **Authors**: *Peng Gao, Jiasen Lu, Hongsheng Li, Roozbeh Mottaghi, Aniruddha Kembhavi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/9fe77ac7060e716f2d42631d156825c0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/9fe77ac7060e716f2d42631d156825c0-Abstract.html)

        **Abstract**:

        Convolutional neural networks (CNNs) are ubiquitous in computer vision, with a myriad of effective and efficient variations. Recently, Transformers -- originally introduced in natural language processing -- have been increasingly adopted in computer vision. While early adopters continued to employ CNN backbones, the latest networks are end-to-end CNN-free Transformer solutions. A recent surprising finding now shows that a simple MLP based solution without any traditional convolutional or Transformer components can produce effective visual representations. While CNNs, Transformers and MLP-Mixers may be considered as completely disparate architectures, we provide a unified view showing that they are in fact special cases of a more general method to aggregate spatial context in a neural network stack. We present the \model (CONText AggregatIon NEtwoRk), a general-purpose building block for multi-head context aggregation that can exploit long-range interactions \emph{a la} Transformers while still exploiting the inductive bias of the local convolution operation leading to faster convergence speeds, often seen in CNNs. Our \model architecture achieves 82.7 \% Top-1 accuracy on ImageNet using 22M parameters, +2.8 improvement compared with DeiT-Small, and can converge to 79.9 \% Top-1 accuracy in just 200 epochs. In contrast to Transformer-based methods that do not scale well to downstream tasks that rely on larger input image resolutions, our efficient network, named \modellight, can be employed in object detection and instance segmentation networks such as DETR, RetinaNet and Mask-RCNN to obtain an impressive detection mAP of 38.9, 43.8, 45.1 and mask mAP of 41.3, providing large improvements of 6.6, 7.3, 6.9 and 6.6 pts respectively, compared to a ResNet-50 backbone with a comparable compute and parameter size. Our method also achieves promising results on self-supervised learning compared to DeiT on the DINO framework. Code is released at https://github.com/allenai/container.

        ----

        ## [1465] ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs

        **Authors**: *Zhanqiu Zhang, Jie Wang, Jiajun Chen, Shuiwang Ji, Feng Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a0160709701140704575d499c997b6ca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a0160709701140704575d499c997b6ca-Abstract.html)

        **Abstract**:

        Query embedding (QE)---which aims to embed entities and first-order logical (FOL) queries in low-dimensional spaces---has shown great power in multi-hop reasoning over knowledge graphs. Recently, embedding entities and queries with geometric shapes becomes a promising direction, as geometric shapes can naturally represent answer sets of queries and logical relationships among them. However, existing geometry-based models have difficulty in modeling queries with negation, which significantly limits their applicability. To address this challenge, we propose a novel query embedding model, namely \textbf{Con}e \textbf{E}mbeddings (ConE), which is the first geometry-based QE model that can handle all the FOL operations, including conjunction, disjunction, and negation. Specifically, ConE represents entities and queries as Cartesian products of two-dimensional cones, where the intersection and union of cones naturally model the conjunction and disjunction operations. By further noticing that the closure of complement of cones remains cones, we design geometric complement operators in the embedding space for the negation operations. Experiments demonstrate that ConE significantly outperforms existing state-of-the-art methods on benchmark datasets.

        ----

        ## [1466] Federated Hyperparameter Tuning: Challenges, Baselines, and Connections to Weight-Sharing

        **Authors**: *Mikhail Khodak, Renbo Tu, Tian Li, Liam Li, Maria-Florina Balcan, Virginia Smith, Ameet Talwalkar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a0205b87490c847182672e8d371e9948-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a0205b87490c847182672e8d371e9948-Abstract.html)

        **Abstract**:

        Tuning hyperparameters is a crucial but arduous part of the machine learning pipeline. Hyperparameter optimization is even more challenging in federated learning, where models are learned over a distributed network of heterogeneous devices; here, the need to keep data on device and perform local training makes it difficult to efficiently train and evaluate configurations. In this work, we investigate the problem of federated hyperparameter tuning. We first identify key challenges and show how standard approaches may be adapted to form baselines for the federated setting. Then, by making a novel connection to the neural architecture search technique of weight-sharing, we introduce a new method, FedEx, to accelerate federated hyperparameter tuning that is applicable to widely-used federated optimization methods such as FedAvg and recent variants. Theoretically, we show that a FedEx variant correctly tunes the on-device learning rate in the setting of online convex optimization across devices. Empirically, we show that FedEx can outperform natural baselines for federated hyperparameter tuning by several percentage points on the Shakespeare, FEMNIST, and CIFAR-10 benchmarksâ€”obtaining higher accuracy using the same training budget.

        ----

        ## [1467] Training for the Future: A Simple Gradient Interpolation Loss to Generalize Along Time

        **Authors**: *Anshul Nasery, Soumyadeep Thakur, Vihari Piratla, Abir De, Sunita Sarawagi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a02ef8389f6d40f84b50504613117f88-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a02ef8389f6d40f84b50504613117f88-Abstract.html)

        **Abstract**:

        In several real world applications, machine learning models are deployed to make predictions on data whose distribution changes gradually along time, leading to a drift between the train and test distributions. Such models are often re-trained on new data periodically, and they hence need to generalize to data not too far into the future. In this context, there is much prior work on enhancing temporal generalization, e.g. continuous transportation of past data, kernel smoothed time-sensitive parameters and more recently, adversarial learning of time-invariant features. However, these methods share several limitations, e.g, poor scalability, training instability, and dependence on unlabeled data from the future. Responding to the above limitations, we propose a simple method that starts with a model with time-sensitive parameters but regularizes its temporal complexity using a Gradient Interpolation  (GI) loss. GI allows the decision boundary to change along time and can still prevent overfitting to the limited training time snapshots  by allowing task-specific control over changes along time. We compare our method to existing baselines on multiple real-world datasets, which show that GI outperforms more complicated generative and adversarial approaches on the one hand, and simpler gradient regularization methods on the other.

        ----

        ## [1468] Agent Modelling under Partial Observability for Deep Reinforcement Learning

        **Authors**: *Georgios Papoudakis, Filippos Christianos, Stefano V. Albrecht*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a03caec56cd82478bf197475b48c05f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a03caec56cd82478bf197475b48c05f9-Abstract.html)

        **Abstract**:

        Modelling the behaviours of other agents is essential for understanding how agents interact and making effective decisions. Existing methods for agent modelling commonly assume knowledge of the local observations and chosen actions of the modelled agents during execution. To eliminate this assumption, we extract representations from the local information of the controlled agent using encoder-decoder architectures. Using the observations and actions of the modelled agents during training, our models learn to extract representations about the modelled agents conditioned only on the local observations of the controlled agent. The representations are used to augment the controlled agent's decision policy which is trained via deep reinforcement learning; thus, during execution, the policy does not require access to other agents' information. We provide a comprehensive evaluation and ablations studies in cooperative, competitive and mixed multi-agent environments, showing that our method achieves significantly higher returns than baseline methods which do not use the learned representations.

        ----

        ## [1469] Leveraging Distribution Alignment via Stein Path for Cross-Domain Cold-Start Recommendation

        **Authors**: *Weiming Liu, Jiajie Su, Chaochao Chen, Xiaolin Zheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a0443c8c8c3372d662e9173c18faaa2c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a0443c8c8c3372d662e9173c18faaa2c-Abstract.html)

        **Abstract**:

        Cross-Domain Recommendation (CDR) has been popularly studied to utilize different domain knowledge to solve the cold-start problem in recommender systems. In this paper, we focus on the Cross-Domain Cold-Start Recommendation (CDCSR) problem. That is, how to leverage the information from a source domain, where items are 'warm', to improve the recommendation performance of a target domain, where items are 'cold'. Unfortunately, previous approaches on cold-start and CDR cannot reduce the latent embedding discrepancy across domains efficiently and lead to model degradation. To address this issue, we propose DisAlign, a cross-domain recommendation framework for the CDCSR problem, which utilizes both rating and auxiliary representations from the source domain to improve the recommendation performance of the target domain. Specifically, we first propose Stein path alignment for aligning the latent embedding distributions across domains, and then further propose its improved version, i.e., proxy Stein path, which can reduce the operation consumption and improve efficiency. Our empirical study on Douban and Amazon datasets demonstrate that DisAlign significantly outperforms the state-of-the-art models under the CDCSR setting.

        ----

        ## [1470] Conservative Offline Distributional Reinforcement Learning

        **Authors**: *Yecheng Jason Ma, Dinesh Jayaraman, Osbert Bastani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a05d886123a54de3ca4b0985b718fb9b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a05d886123a54de3ca4b0985b718fb9b-Abstract.html)

        **Abstract**:

        Many reinforcement learning (RL) problems in practice are offline, learning purely from observational data. A key challenge is how to ensure the learned policy is safe, which requires quantifying the risk associated with different actions. In the online setting, distributional RL algorithms do so by learning the distribution over returns (i.e., cumulative rewards) instead of the expected return; beyond quantifying risk, they have also been shown to learn better representations for planning. We proposeConservative Offline Distributional Actor Critic (CODAC), an offline RL algorithm suitable for both risk-neutral and risk-averse domains. CODAC adapts distributional RL to the offline setting by penalizing the predicted quantiles of the return for out-of-distribution actions. We prove that CODAC learns a conservative return distribution---in particular, for finite MDPs, CODAC converges to an uniform lower bound on the quantiles of the return distribution; our proof relies on a novel analysis of the distributional Bellman operator. In our experiments, on two challenging robot navigation tasks, CODAC successfully learns risk-averse policies using offline data collected purely from risk-neutral agents. Furthermore, CODAC is state-of-the-art on the D4RL MuJoCo benchmark in terms of both expected and risk-sensitive performance.

        ----

        ## [1471] Separation Results between Fixed-Kernel and Feature-Learning Probability Metrics

        **Authors**: *Carles Domingo-Enrich, Youssef Mroueh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a081c174f5913958ba8c6443bacffcb9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a081c174f5913958ba8c6443bacffcb9-Abstract.html)

        **Abstract**:

        Several works in implicit and explicit generative modeling empirically observed that feature-learning discriminators  outperform  fixed-kernel discriminators in terms of the sample quality of the models. We provide separation results between probability metrics with fixed-kernel and feature-learning discriminators using the function classes $\mathcal{F}_2$  and $\mathcal{F}_1$ respectively, which were developed to study overparametrized two-layer neural networks. In particular, we construct pairs of distributions over hyper-spheres that can not be discriminated by  fixed kernel $(\mathcal{F}_2)$ integral probability metric (IPM) and Stein discrepancy (SD) in high dimensions, but that can be discriminated by their feature learning ($\mathcal{F}_1$) counterparts. To further study the separation we provide links between the $\mathcal{F}_1$ and $\mathcal{F}_2$ IPMs with sliced Wasserstein distances. Our work suggests that fixed-kernel discriminators perform worse than their feature learning counterparts because their corresponding metrics are weaker.

        ----

        ## [1472] Risk Minimization from Adaptively Collected Data: Guarantees for Supervised and Policy Learning

        **Authors**: *Aurélien Bibaut, Nathan Kallus, Maria Dimakopoulou, Antoine Chambaz, Mark J. van der Laan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a0ae15571eb4a97ac1c34a114f1bb179-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a0ae15571eb4a97ac1c34a114f1bb179-Abstract.html)

        **Abstract**:

        Empirical risk minimization (ERM) is the workhorse of machine learning, whether for classification and regression or for off-policy policy learning, but its model-agnostic guarantees can fail when we use adaptively collected data, such as the result of running a contextual bandit algorithm. We study a generic importance sampling weighted ERM algorithm for using adaptively collected data to minimize the average of a loss function over a hypothesis class and provide first-of-their-kind generalization guarantees and fast convergence rates. Our results are based on a new maximal inequality that carefully leverages the importance sampling structure to obtain rates with the good dependence on the exploration rate in the data. For regression, we provide fast rates that leverage the strong convexity of squared-error loss. For policy learning, we provide regret guarantees that close an open gap in the existing literature whenever exploration decays to zero, as is the case for bandit-collected data. An empirical investigation validates our theory.

        ----

        ## [1473] Bayesian Optimization with High-Dimensional Outputs

        **Authors**: *Wesley J. Maddox, Maximilian Balandat, Andrew Gordon Wilson, Eytan Bakshy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a0d3973ad100ad83a64c304bb58677dd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a0d3973ad100ad83a64c304bb58677dd-Abstract.html)

        **Abstract**:

        Bayesian optimization is a sample-efficient black-box optimization procedure that is typically applied to a small number of independent objectives. However, in practice we often wish to optimize objectives defined over many correlated outcomes (or “tasks”). For example, scientists may want to optimize the coverage of a cell tower network across a dense grid of locations. Similarly, engineers may seek to balance the performance of a robot across dozens of different environments via constrained or robust optimization. However, the Gaussian Process (GP) models typically used as probabilistic surrogates for multi-task Bayesian optimization scale poorly with the number of outcomes, greatly limiting applicability. We devise an efficient technique for exact multi-task GP sampling that combines exploiting Kronecker structure in the covariance matrices with Matheron’s identity, allowing us to perform Bayesian optimization using exact multi-task GP models with tens of thousands of correlated outputs. In doing so, we achieve substantial improvements in sample efficiency compared to existing approaches that model solely the outcome metrics. We demonstrate how this unlocks a new class of applications for Bayesian optimization across a range of tasks in science and engineering, including optimizing interference patterns of an optical interferometer with 65,000 outputs.

        ----

        ## [1474] Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks

        **Authors**: *Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a113c1ecd3cace2237256f4c712f61b5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a113c1ecd3cace2237256f4c712f61b5-Abstract.html)

        **Abstract**:

        One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online.

        ----

        ## [1475] Scalable Diverse Model Selection for Accessible Transfer Learning

        **Authors**: *Daniel Bolya, Rohit Mittapalli, Judy Hoffman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a1140a3d0df1c81e24ae954d935e8926-Abstract.html)

        **Abstract**:

        With the preponderance of pretrained deep learning models available off-the-shelf from model banks today, finding the best weights to fine-tune to your use-case can be a daunting task. Several methods have recently been proposed to find good models for transfer learning, but they either don't scale well to large model banks or don't perform well on the diversity of off-the-shelf models. Ideally the question we want to answer is, "given some data and a source model, can you quickly predict the model's accuracy after fine-tuning?" In this paper, we formalize this setting as "Scalable Diverse Model Selection" and propose several benchmarks for evaluating on this task. We find that existing model selection and transferability estimation methods perform poorly here and analyze why this is the case. We then introduce simple techniques to improve the performance and speed of these algorithms. Finally, we iterate on existing methods to create PARC, which outperforms all other methods on diverse model selection. We have released the benchmarks and method code in hope to inspire future work in model selection for accessible transfer learning.

        ----

        ## [1476] Light Field Networks: Neural Scene Representations with Single-Evaluation Rendering

        **Authors**: *Vincent Sitzmann, Semon Rezchikov, Bill Freeman, Josh Tenenbaum, Frédo Durand*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a11ce019e96a4c60832eadd755a17a58-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a11ce019e96a4c60832eadd755a17a58-Abstract.html)

        **Abstract**:

        Inferring representations of 3D scenes from 2D observations is a fundamental problem of computer graphics, computer vision, and artificial intelligence. Emerging 3D-structured neural scene representations are a promising approach to 3D scene understanding. In this work, we propose a novel neural scene representation, Light Field Networks or LFNs, which represent both geometry and appearance of the underlying 3D scene in a 360-degree, four-dimensional light field parameterized via a neural implicit representation.  Rendering a ray from an LFN requires only a single network evaluation, as opposed to hundreds of evaluations per ray for ray-marching or volumetric based renderers in 3D-structured neural scene representations.  In the setting of simple scenes, we leverage meta-learning to learn a prior over LFNs that enables multi-view consistent light field reconstruction from as little as a single image observation. This results in dramatic reductions in time and memory complexity, and enables real-time rendering. The cost of storing a 360-degree light field via an LFN is two orders of magnitude lower than conventional methods such as the Lumigraph.  Utilizing the analytical differentiability of neural implicit representations and a novel parameterization of light space, we further demonstrate the extraction of sparse depth maps from LFNs.

        ----

        ## [1477] ViSER: Video-Specific Surface Embeddings for Articulated 3D Shape Reconstruction

        **Authors**: *Gengshan Yang, Deqing Sun, Varun Jampani, Daniel Vlasic, Forrester Cole, Ce Liu, Deva Ramanan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a11f9e533f28593768ebf87075ab34f2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a11f9e533f28593768ebf87075ab34f2-Abstract.html)

        **Abstract**:

        We introduce ViSER, a method for recovering articulated 3D shapes and dense3D trajectories from monocular videos.  Previous work on high-quality reconstruction of dynamic 3D shapes typically relies on multiple camera views, strong category-specific priors, or 2D keypoint supervision. We show that none of these are required if one can reliably estimate long-range correspondences in a video, making use of only 2D object masks and two-frame optical flow as inputs. ViSER infers correspondences by matching 2D pixels to a canonical,  deformable 3D mesh via video-specific surface embeddings that capture the pixel appearance of each surface point.  These embeddings behave as a continuous set of keypoint descriptors defined over the mesh surface, which can be used to establish dense long-range correspondences across pixels.  The surface embeddings are implemented as coordinate-based MLPs that are fit to each video via self-supervised losses.Experimental results show that ViSER compares favorably against prior work on challenging videos of humans with loose clothing and unusual poses as well as animals videos from DAVIS and YTVOS. Project page: viser-shape.github.io.

        ----

        ## [1478] Understanding the Effect of Stochasticity in Policy Optimization

        **Authors**: *Jincheng Mei, Bo Dai, Chenjun Xiao, Csaba Szepesvári, Dale Schuurmans*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a12f69495f41bb3b637ba1b6238884d6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a12f69495f41bb3b637ba1b6238884d6-Abstract.html)

        **Abstract**:

        We study the effect of stochasticity in on-policy policy optimization, and make the following four contributions. \emph{First}, we show that the preferability of optimization methods depends critically on whether stochastic versus exact gradients are used. In particular, unlike the true gradient setting, geometric information cannot be easily exploited in the stochastic case for accelerating policy optimization without detrimental consequences or impractical assumptions. \emph{Second}, to explain these findings we introduce the concept of committal rate for stochastic policy optimization, and show that this can serve as a criterion for determining almost sure convergence to global optimality. \emph{Third}, we show that in the absence of external oracle information, which allows an algorithm to determine the difference between optimal and sub-optimal actions given only on-policy samples, there is an inherent trade-off between exploiting geometry to accelerate convergence versus achieving optimality almost surely. That is, an uninformed algorithm either converges to a globally optimal policy with probability $1$ but at a rate no better than $O(1/t)$, or it achieves faster than $O(1/t)$ convergence but then must fail to converge to the globally optimal policy with some positive probability. \emph{Finally}, we use the committal rate theory to explain why practical policy optimization methods are sensitive to random initialization, then develop an ensemble method that can be guaranteed to achieve near-optimal solutions with high probability.

        ----

        ## [1479] Fine-Grained Zero-Shot Learning with DNA as Side Information

        **Authors**: *Sarkhan Badirli, Zeynep Akata, George O. Mohler, Christine Picard, Murat Dundar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a18630ab1c3b9f14454cf70dc7114834-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a18630ab1c3b9f14454cf70dc7114834-Abstract.html)

        **Abstract**:

        Fine-grained zero-shot learning task requires some form of side-information to transfer discriminative information from seen to unseen classes. As manually annotated visual attributes are extremely costly and often impractical to obtain for a large number of classes, in this study we use DNA as a side information for the first time for fine-grained zero-shot classification of species. Mitochondrial DNA plays an important role as a genetic marker in evolutionary biology and has been used to achieve near perfect accuracy in species classification of living organisms. We implement a simple hierarchical Bayesian model that uses DNA information to establish the hierarchy in the image space and employs local priors to define surrogate classes for unseen ones. On the benchmark CUB dataset we show that DNA can be equally promising, yet in general a more accessible alternative than word vectors as a side information. This is especially important as obtaining robust word representations for fine-grained species names is not a practicable goal when information about these species in free-form text is limited. On a newly compiled fine-grained insect dataset that uses DNA information from over a thousand species we show that the Bayesian approach outperforms state-of-the-art by a wide margin.

        ----

        ## [1480] Optimal Underdamped Langevin MCMC Method

        **Authors**: *Zhengmian Hu, Feihu Huang, Heng Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a18aa23ee676d7f5ffb34cf16df3e08c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a18aa23ee676d7f5ffb34cf16df3e08c-Abstract.html)

        **Abstract**:

        In the paper, we study the underdamped Langevin diffusion (ULD) with strongly-convex potential consisting of finite summation of $N$ smooth components, and propose an efficient discretization method, which requires $O(N+d^\frac{1}{3}N^\frac{2}{3}/\varepsilon^\frac{2}{3})$ gradient evaluations to achieve $\varepsilon$-error (in $\sqrt{\mathbb{E}{\lVert{\cdot}\rVert_2^2}}$ distance) for approximating $d$-dimensional ULD. Moreover, we prove a lower bound of gradient complexity as $\Omega(N+d^\frac{1}{3}N^\frac{2}{3}/\varepsilon^\frac{2}{3})$, which indicates that our method is optimal in dependence of $N$, $\varepsilon$, and $d$. In particular, we apply our method to sample the strongly-log-concave distribution and obtain gradient complexity better than all existing gradient based sampling algorithms. Experimental results on both synthetic and real-world data show that our new method consistently outperforms the existing ULD approaches.

        ----

        ## [1481] Scheduling jobs with stochastic holding costs

        **Authors**: *Dabeen Lee, Milan Vojnovic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a19744e268754fb0148b017647355b7b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a19744e268754fb0148b017647355b7b-Abstract.html)

        **Abstract**:

        This paper proposes a learning and scheduling algorithm to minimize the expected cumulative holding cost incurred by jobs, where statistical parameters defining their individual holding costs are unknown a priori. In each time slot, the server can process a job while receiving the realized random holding costs of the jobs remaining in the system. Our algorithm is a learning-based variant of the $c\mu$ rule for scheduling: it starts with a preemption period of fixed length which serves as a learning phase, and after accumulating enough data about individual jobs, it switches to nonpreemptive scheduling mode. The algorithm is designed to handle instances with large or small gaps in jobs' parameters and achieves near-optimal performance guarantees. The performance of our algorithm is captured by its regret, where the benchmark is the minimum possible cost attained when the statistical parameters of jobs are fully known.  We prove upper bounds on the regret of our algorithm, and we derive a regret lower bound that is almost matching the proposed upper bounds. Our numerical results demonstrate the effectiveness of our algorithm and show that our theoretical regret analysis is nearly tight.

        ----

        ## [1482] REMIPS: Physically Consistent 3D Reconstruction of Multiple Interacting People under Weak Supervision

        **Authors**: *Mihai Fieraru, Mihai Zanfir, Teodor Alexandru Szente, Eduard Gabriel Bazavan, Vlad Olaru, Cristian Sminchisescu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a1a2c3fed88e9b3ba5bc3625c074a04e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a1a2c3fed88e9b3ba5bc3625c074a04e-Abstract.html)

        **Abstract**:

        The three-dimensional reconstruction of multiple interacting humans given a monocular image is crucial for the general task of scene understanding, as capturing the subtleties of interaction is often the very reason for taking a picture. Current 3D human reconstruction methods either treat each person independently, ignoring most of the context, or reconstruct people jointly, but cannot recover interactions correctly when people are in close proximity. In this work, we introduce \textbf{REMIPS}, a model for 3D \underline{Re}construction of \underline{M}ultiple \underline{I}nteracting \underline{P}eople under Weak \underline{S}upervision. \textbf{REMIPS} can reconstruct a variable number of people directly from monocular images. At the core of our methodology stands a novel transformer network that combines unordered person tokens (one for each detected human) with positional-encoded tokens from image features patches. We introduce a novel unified model for self- and interpenetration-collisions based on a mesh approximation computed by applying decimation operators. We rely on self-supervised losses for flexibility and generalisation in-the-wild and incorporate self-contact and interaction-contact losses directly into the learning process. With \textbf{REMIPS}, we report state-of-the-art quantitative results on common benchmarks even in cases where no 3D supervision is used. Additionally, qualitative visual results show that our reconstructions are plausible in terms of pose and shape and coherent for challenging images, collected in-the-wild, where people are often interacting.

        ----

        ## [1483] Differentiable Annealed Importance Sampling and the Perils of Gradient Noise

        **Authors**: *Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger B. Grosse*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a1a609f1ac109d0be28d8ae112db1bbb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a1a609f1ac109d0be28d8ae112db1bbb-Abstract.html)

        **Abstract**:

        Annealed importance sampling (AIS) and related algorithms are highly effective tools for marginal likelihood estimation, but are not fully differentiable due to the use of Metropolis-Hastings correction steps. Differentiability is a desirable property as it would admit the possibility of optimizing marginal likelihood as an objective using gradient-based methods. To this end, we propose Differentiable AIS (DAIS), a variant of AIS which ensures differentiability by abandoning the Metropolis-Hastings corrections. As a further advantage, DAIS allows for mini-batch gradients. We provide a detailed convergence analysis for Bayesian linear regression which goes beyond previous analyses by explicitly accounting for the sampler not having reached equilibrium. Using this analysis, we prove that DAIS is consistent in the full-batch setting and provide a sublinear convergence rate. Furthermore, motivated by the problem of learning from large-scale datasets, we study a stochastic variant of DAIS that uses mini-batch gradients. Surprisingly, stochastic DAIS can be arbitrarily bad due to a fundamental incompatibility between the goals of last-iterate convergence to the posterior and elimination of the accumulated stochastic error. This is in stark contrast with other settings such as gradient-based optimization and Langevin dynamics, where the effect of gradient noise can be washed out by taking smaller steps. This indicates that annealing-based marginal likelihood estimation with stochastic gradients may require new ideas.

        ----

        ## [1484] PSD Representations for Effective Probability Models

        **Authors**: *Alessandro Rudi, Carlo Ciliberto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a1b63b36ba67b15d2f47da55cdb8018d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a1b63b36ba67b15d2f47da55cdb8018d-Abstract.html)

        **Abstract**:

        Finding a good way to model probability densities is key to probabilistic inference. An ideal model should be able to concisely approximate any probability while being also compatible with two main operations: multiplications of two models (product rule) and marginalization with respect to a subset of the random variables (sum rule). In this work, we show that a recently proposed class of positive semi-definite (PSD) models for non-negative functions is particularly suited to this end. In particular, we characterize both approximation and generalization capabilities of PSD models, showing that they enjoy strong theoretical guarantees. Moreover, we show that we can perform efficiently both sum and product rule in closed form via matrix operations, enjoying the same versatility of mixture models. Our results open the way to applications of PSD models to density estimation, decision theory, and inference.

        ----

        ## [1485] Exploiting a Zoo of Checkpoints for Unseen Tasks

        **Authors**: *Jiaji Huang, Qiang Qiu, Kenneth Church*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a1c3ae6c49a89d92aef2d423dadb477f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a1c3ae6c49a89d92aef2d423dadb477f-Abstract.html)

        **Abstract**:

        There are so many models in the literature that it is difficult for practitioners to decide which combinations are likely to be effective for a new task. This paper attempts to address this question by capturing relationships among checkpoints published on the web. We model the space of tasks as a Gaussian process. The covariance can be estimated from checkpoints and unlabeled probing data. With the Gaussian process, we can identify representative checkpoints by a maximum mutual information criterion. This objective is submodular. A greedy method identifies representatives that are likely to "cover'' the task space. These representatives generalize to new tasks with superior performance. Empirical evidence is provided for applications from both computational linguistics as well as computer vision.

        ----

        ## [1486] Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach

        **Authors**: *Qitian Wu, Chenxiao Yang, Junchi Yan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a1c5aff9679455a233086e26b72b9a06-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a1c5aff9679455a233086e26b72b9a06-Abstract.html)

        **Abstract**:

        We target open-world feature extrapolation problem where the feature space of input data goes through expansion and a model trained on partially observed features needs to handle new features in test data without further retraining. The problem is of much significance for dealing with features incrementally collected from different fields. To this end, we propose a new learning paradigm with graph representation and learning. Our framework contains two modules: 1) a backbone network (e.g., feedforward neural nets) as a lower model takes features as input and outputs predicted labels; 2) a graph neural network as an upper model learns to extrapolate embeddings for new features via message passing over a feature-data graph built from observed data. Based on our framework, we design two training strategies, a self-supervised approach and an inductive learning approach, to endow the model with extrapolation ability and alleviate feature-level over-fitting. We also provide theoretical analysis on the generalization error on test data with new features, which dissects the impact of training features and algorithms on generalization performance. Our experiments over several classification datasets and large-scale advertisement click prediction datasets demonstrate that our model can produce effective embeddings for unseen features and significantly outperforms baseline methods that adopt KNN and local aggregation.

        ----

        ## [1487] Adversarial Teacher-Student Representation Learning for Domain Generalization

        **Authors**: *Fu-En Yang, Yuan-Chia Cheng, Zu-Yun Shiau, Yu-Chiang Frank Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a2137a2ae8e39b5002a3f8909ecb88fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a2137a2ae8e39b5002a3f8909ecb88fe-Abstract.html)

        **Abstract**:

        Domain generalization (DG) aims to transfer the learning task from a single or multiple source domains to unseen target domains. To extract and leverage the information which exhibits sufficient generalization ability, we propose a simple yet effective approach of Adversarial Teacher-Student Representation Learning, with the goal of deriving the domain generalizable representations via generating and exploring out-of-source data distributions. Our proposed framework advances Teacher-Student learning in an adversarial learning manner, which alternates between knowledge-distillation based representation learning and novel-domain data augmentation. The former progressively updates the teacher network for deriving domain-generalizable representations, while the latter synthesizes data out-of-source yet plausible distributions. Extensive image classification experiments on benchmark datasets in multiple and single source DG settings confirm that, our model exhibits sufficient generalization ability and performs favorably against state-of-the-art DG methods.

        ----

        ## [1488] Stochastic bandits with groups of similar arms

        **Authors**: *Fabien Pesquerel, Hassan Saber, Odalric-Ambrym Maillard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a22c0238589078fb10b606ab62015744-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a22c0238589078fb10b606ab62015744-Abstract.html)

        **Abstract**:

        We consider a variant of the stochastic multi-armed bandit problem where arms are known to be organized into different groups having the same mean. The groups are unknown but a lower bound $q$ on their size is known. This situation typically appears when each arm can be described with a list of categorical attributes, and the (unknown) mean reward function only depends on a subset of them, the others being redundant. In this case, $q$ is linked naturally to the number of attributes considered redundant, and the number of categories of each attribute. For this structured problem of practical relevance, we first derive the asymptotic regret lower bound and corresponding constrained optimization problem. They reveal  the achievable regret can be substantially reduced when compared to the unstructured setup, possibly by a factor $q$. However, solving exactly the exact constrained optimization problem involves a combinatorial problem. We introduce a lower-bound inspired strategy involving a computationally efficient relaxation that is based on a sorting mechanism. We further prove it achieves a lower bound close to the optimal one up to a controlled factor, and achieves an asymptotic regret $q$ times smaller than the unstructured one. We believe this shows it is a valuable strategy for the practitioner. Last, we illustrate the performance of the considered strategy on numerical experiments involving a large number of arms.

        ----

        ## [1489] Tracking Without Re-recognition in Humans and Machines

        **Authors**: *Drew Linsley, Girik Malik, Junkyung Kim, Lakshmi Narasimhan Govindarajan, Ennio Mingolla, Thomas Serre*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a2557a7b2e94197ff767970b67041697-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a2557a7b2e94197ff767970b67041697-Abstract.html)

        **Abstract**:

        Imagine trying to track one particular fruitfly in a swarm of hundreds. Higher biological visual systems have evolved to track moving objects by relying on both their appearance and their motion trajectories. We investigate if state-of-the-art spatiotemporal deep neural networks are capable of the same. For this, we introduce PathTracker, a synthetic visual challenge that asks human observers and machines to track a target object in the midst of identical-looking "distractor" objects. While humans effortlessly learn PathTracker and generalize to systematic variations in task design, deep networks struggle. To address this limitation, we identify and model circuit mechanisms in biological brains that are implicated in tracking objects based on motion cues. When instantiated as a recurrent network, our circuit model learns to solve PathTracker with a robust visual strategy that rivals human performance and explains a significant proportion of their decision-making on the challenge. We also show that the success of this circuit model extends to object tracking in natural videos. Adding it to a transformer-based architecture for object tracking builds tolerance to visual nuisances that affect object appearance, establishing the new state of the art on the large-scale TrackingNet challenge. Our work highlights the importance of understanding human vision to improve computer vision.

        ----

        ## [1490] Rethinking conditional GAN training: An approach using geometrically structured latent manifolds

        **Authors**: *Sameera Ramasinghe, Moshiur R. Farazi, Salman H. Khan, Nick Barnes, Stephen Gould*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a267f936e54d7c10a2bb70dbe6ad7a89-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a267f936e54d7c10a2bb70dbe6ad7a89-Abstract.html)

        **Abstract**:

        Conditional GANs (cGAN), in their rudimentary form, suffer from critical drawbacks such as the lack of diversity in generated outputs and distortion between the latent and output manifolds.  Although efforts have been made to improve results, they can suffer from unpleasant side-effects such as the topology mismatch between latent and output spaces. In contrast, we tackle this problem from a geometrical perspective and propose a novel training mechanism that increases both the diversity and the visual quality of a vanilla cGAN, by systematically encouraging a bi-lipschitz mapping between the latent and the output manifolds. We validate the efficacy of our solution on a baseline cGAN (i.e., Pix2Pix) which lacks diversity, and show that by only modifying its training mechanism (i.e., with our proposed Pix2Pix-Geo), one can achieve more diverse and realistic outputs on a broad set of image-to-image translation tasks.

        ----

        ## [1491] How to transfer algorithmic reasoning knowledge to learn new algorithms?

        **Authors**: *Louis-Pascal A. C. Xhonneux, Andreea Deac, Petar Velickovic, Jian Tang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a2802cade04644083dcde1c8c483ed9a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a2802cade04644083dcde1c8c483ed9a-Abstract.html)

        **Abstract**:

        Learning to execute algorithms is a fundamental problem that has been widely studied. Prior work (Veličković et al., 2019) has shown that to enable systematic generalisation on graph algorithms it is critical to have access to the intermediate steps of the program/algorithm. In many reasoning tasks, where algorithmic-style reasoning is important, we only have access to the input and output examples. Thus, inspired by the success of pre-training on similar tasks or data in Natural Language Processing (NLP) and Computer vision, we set out to study how we can transfer algorithmic reasoning knowledge. Specifically, we investigate how we can use algorithms for which we have access to the execution trace to learn to solve similar tasks for which we do not. We investigate two major classes of graph algorithms, parallel algorithms such as breadth-first search and Bellman-Ford and sequential greedy algorithms such as Prims and Dijkstra. Due to the fundamental differences between algorithmic reasoning knowledge and feature extractors such as used in Computer vision or NLP, we hypothesis that standard transfer techniques will not be sufficient to achieve systematic generalisation. To investigate this empirically we create a dataset including 9 algorithms and 3 different graph types. We validate this empirically and show how instead multi-task learning can be used to achieve the transfer of algorithmic reasoning knowledge.

        ----

        ## [1492] Fast Axiomatic Attribution for Neural Networks

        **Authors**: *Robin Hesse, Simone Schaub-Meyer, Stefan Roth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a284df1155ec3e67286080500df36a9a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a284df1155ec3e67286080500df36a9a-Abstract.html)

        **Abstract**:

        Mitigating the dependence on spurious correlations present in the training dataset is a quickly emerging and important topic of deep learning. Recent approaches include priors on the feature attribution of a deep neural network (DNN) into the training process to reduce the dependence on unwanted features. However, until now one needed to trade off high-quality attributions, satisfying desirable axioms, against the time required to compute them. This in turn either led to long training times or ineffective attribution priors. In this work, we break this trade-off by considering a special class of efficiently axiomatically attributable DNNs for which an axiomatic feature attribution can be computed with only a single forward/backward pass. We formally prove that nonnegatively homogeneous DNNs, here termed $\mathcal{X}$-DNNs, are efficiently axiomatically attributable and show that they can be effortlessly constructed from a wide range of regular DNNs by simply removing the bias term of each layer. Various experiments demonstrate the advantages of $\mathcal{X}$-DNNs, beating state-of-the-art generic attribution methods on regular DNNs for training with attribution priors.

        ----

        ## [1493] OSOA: One-Shot Online Adaptation of Deep Generative Models for Lossless Compression

        **Authors**: *Chen Zhang, Shifeng Zhang, Fabio Maria Carlucci, Zhenguo Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a2915ad0d57ca8c644f99f9c3f20a918-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a2915ad0d57ca8c644f99f9c3f20a918-Abstract.html)

        **Abstract**:

        Explicit deep generative models (DGMs), e.g., VAEs and Normalizing Flows, have shown to offer an effective data modelling alternative for lossless compression. However, DGMs themselves normally require large storage space and thus contaminate the advantage brought by accurate data density estimation.To eliminate the requirement of saving separate models for different target datasets, we propose a novel setting that starts from a pretrained deep generative model and compresses the data batches while adapting the model with a dynamical system for only one epoch.We formalise this setting as that of One-Shot Online Adaptation (OSOA) of DGMs for lossless compression and propose a vanilla algorithm under this setting. Experimental results show that vanilla OSOA can save significant time versus training bespoke models and space versus using one model for all targets.With the same adaptation step number or adaptation time, it is shown vanilla OSOA can exhibit better space efficiency, e.g., $47\%$ less space, than fine-tuning the pretrained model and saving the fine-tuned model.Moreover, we showcase the potential of OSOA and motivate more sophisticated OSOA algorithms by showing further space or time efficiency with multiple updates per batch and early stopping.

        ----

        ## [1494] Compressive Visual Representations

        **Authors**: *Kuang-Huei Lee, Anurag Arnab, Sergio Guadarrama, John F. Canny, Ian Fischer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a29a5ba2cb7bdeabba22de8c83321b46-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a29a5ba2cb7bdeabba22de8c83321b46-Abstract.html)

        **Abstract**:

        Learning effective visual representations that generalize well without human supervision is a fundamental problem in order to apply Machine Learning to a wide variety of tasks. Recently, two families of self-supervised methods, contrastive learning and latent bootstrapping, exemplified by SimCLR and BYOL respectively, have made significant progress. In this work, we hypothesize that adding explicit information compression to these algorithms yields better and more robust representations. We verify this by developing SimCLR and BYOL formulations compatible with the Conditional Entropy Bottleneck (CEB) objective, allowing us to both measure and control the amount of compression in the learned representation, and observe their impact on downstream tasks. Furthermore, we explore the relationship between Lipschitz continuity and compression, showing a tractable lower bound on the Lipschitz constant of the encoders we learn. As Lipschitz continuity is closely related to robustness, this provides a new explanation for why compressed models are more robust. Our experiments confirm that adding compression to SimCLR and BYOL significantly improves linear evaluation accuracies and model robustness across a wide range of domain shifts. In particular, the compressed version of BYOL achieves 76.0% Top-1 linear evaluation accuracy on ImageNet with ResNet-50, and 78.8% with ResNet-50 2x.

        ----

        ## [1495] Multi-Armed Bandits with Bounded Arm-Memory: Near-Optimal Guarantees for Best-Arm Identification and Regret Minimization

        **Authors**: *Arnab Maiti, Vishakha Patil, Arindam Khan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a2f04745390fd6897d09772b2cd1f581-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a2f04745390fd6897d09772b2cd1f581-Abstract.html)

        **Abstract**:

        We study the Stochastic Multi-armed Bandit problem under bounded arm-memory. In this setting, the arms arrive in a stream, and the number of arms that can be stored in the memory at any time, is bounded. The decision-maker can only pull arms that are present in the memory.  We address the problem from the perspective of two standard objectives: 1) regret minimization, and 2) best-arm identification. For regret minimization, we settle an important open question by showing an almost tight guarantee. We show $\Omega(T^{2/3})$ cumulative regret in expectation for single-pass algorithms for arm-memory size of $(n-1)$, where $n$ is the number of arms. For best-arm identification, we provide an $(\varepsilon, \delta)$-PAC algorithm with arm memory size of $O(\log^*n)$ and $O(\frac{n}{\varepsilon^2}\cdot \log(\frac{1}{\delta}))$ optimal sample complexity.

        ----

        ## [1496] Grounding inductive biases in natural images: invariance stems from variations in data

        **Authors**: *Diane Bouchacourt, Mark Ibrahim, Ari S. Morcos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a2fe8c05877ec786290dd1450c3385cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a2fe8c05877ec786290dd1450c3385cd-Abstract.html)

        **Abstract**:

        To perform well on unseen and potentially out-of-distribution samples, it is desirable for machine learning models to have a predictable response with respect to transformations affecting the factors of variation of the input. Here, we study the relative importance of several types of inductive biases towards such predictable behavior: the choice of data, their augmentations, and model architectures. Invariance is commonly achieved through hand-engineered data augmentation, but do standard data augmentations address transformations that explain variations in real data? While prior work has focused on synthetic data, we attempt here to characterize the factors of variation in a real dataset, ImageNet, and study the invariance of both standard residual networks and the recently proposed vision transformer with respect to changes in these factors. We show standard augmentation relies on a precise combination of translation and scale, with translation recapturing most of the performance improvement---despite the (approximate) translation invariance built in to convolutional architectures, such as residual networks. In fact, we found that scale and translation invariance was similar across residual networks and vision transformer models despite their markedly different architectural inductive biases. We show the training data itself is the main source of invariance, and that data augmentation only further increases the learned invariances. Notably, the invariances learned during training align with the ImageNet factors of variation we found. Finally, we find that the main factors of variation in ImageNet mostly relate to appearance and are specific to each class.

        ----

        ## [1497] Directed Graph Contrastive Learning

        **Authors**: *Zekun Tong, Yuxuan Liang, Henghui Ding, Yongxing Dai, Xinke Li, Changhu Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a3048e47310d6efaa4b1eaf55227bc92-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a3048e47310d6efaa4b1eaf55227bc92-Abstract.html)

        **Abstract**:

        Graph Contrastive Learning (GCL) has emerged to learn generalizable representations from contrastive views. However, it is still in its infancy with two concerns: 1) changing the graph structure through data augmentation to generate contrastive views may mislead the message passing scheme, as such graph changing action deprives the intrinsic graph structural information, especially the directional structure in directed graphs; 2) since GCL usually uses predefined contrastive views with hand-picking parameters, it does not take full advantage of the contrastive information provided by data augmentation, resulting in incomplete structure information for models learning. In this paper, we design a directed graph data augmentation method called Laplacian perturbation and theoretically analyze how it provides contrastive information without changing the directed graph structure. Moreover, we present a directed graph contrastive learning framework, which dynamically learns from all possible contrastive views generated by Laplacian perturbation. Then we train it using multi-task curriculum learning to progressively learn from multiple easy-to-difficult contrastive views. We empirically show that our model can retain more structural features of directed graphs than other GCL models because of its ability to provide complete contrastive information. Experiments on various benchmarks reveal our dominance over the state-of-the-art approaches.

        ----

        ## [1498] Space-time Mixing Attention for Video Transformer

        **Authors**: *Adrian Bulat, Juan-Manuel Pérez-Rúa, Swathikiran Sudhakaran, Brais Martínez, Georgios Tzimiropoulos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a34bacf839b923770b2c360eefa26748-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a34bacf839b923770b2c360eefa26748-Abstract.html)

        **Abstract**:

        This paper is on video recognition using Transformers. Very recent attempts in this area have demonstrated promising results in terms of recognition accuracy, yet they have been also shown to induce, in many cases, significant computational overheads due to the additional modelling of the temporal information. In this work, we propose a Video Transformer model the complexity of which scales linearly with the number of frames in the video sequence and hence induces no overhead compared to an image-based Transformer model. To achieve this, our model makes two approximations to the full space-time attention used in Video Transformers: (a) It restricts time attention to a local temporal window and capitalizes on the Transformer's depth to obtain full temporal coverage of the video sequence. (b) It uses efficient space-time mixing to attend jointly spatial and temporal locations without inducing any additional cost on top of a spatial-only attention model. We also show how to integrate 2 very lightweight mechanisms for global temporal-only attention which provide additional accuracy improvements at minimal computational cost. We demonstrate that our model produces very high recognition accuracy on the most popular video recognition datasets while at the same time being significantly more efficient than other Video Transformer models.

        ----

        ## [1499] Particle Dual Averaging: Optimization of Mean Field Neural Network with Global Convergence Rate Analysis

        **Authors**: *Atsushi Nitanda, Denny Wu, Taiji Suzuki*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a34e1ddbb4d329167f50992ba59fe45a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a34e1ddbb4d329167f50992ba59fe45a-Abstract.html)

        **Abstract**:

        We propose the particle dual averaging (PDA) method, which generalizes the dual averaging method in convex optimization to the optimization over probability distributions with quantitative runtime guarantee. The algorithm consists of an inner loop and outer loop: the inner loop utilizes the Langevin algorithm to approximately solve for a stationary distribution, which is then optimized in the outer loop. The method can be interpreted as an extension of the Langevin algorithm to naturally handle nonlinear functional on the probability space. An important application of the proposed method is the optimization of neural network in the mean field regime, which is theoretically attractive due to the presence of nonlinear feature learning, but quantitative convergence rate can be challenging to obtain. By adapting finite-dimensional convex optimization theory into the space of measures, we not only establish global convergence of PDA for two-layer mean field neural networks under more general settings and simpler analysis, but also provide quantitative polynomial runtime guarantee. Our theoretical results are supported by numerical simulations on neural networks with reasonable size.

        ----

        ## [1500] Learning Tree Interpretation from Object Representation for Deep Reinforcement Learning

        **Authors**: *Guiliang Liu, Xiangyu Sun, Oliver Schulte, Pascal Poupart*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a35fe7f7fe8217b4369a0af4244d1fca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a35fe7f7fe8217b4369a0af4244d1fca-Abstract.html)

        **Abstract**:

        Interpreting Deep Reinforcement Learning (DRL) models is important to enhance trust and comply with transparency regulations. Existing methods typically explain a DRL model by visualizing the importance of low-level input features with super-pixels, attentions, or saliency maps. Our approach provides an interpretation based on high-level latent object features derived from a disentangled representation. We propose a Represent And Mimic (RAMi) framework for training 1) an identifiable latent representation to capture the independent factors of variation for the objects and 2) a mimic tree that extracts the causal impact of the latent features on DRL action values. To jointly optimize both the fidelity and the simplicity of a mimic tree, we derive a novel Minimum Description Length (MDL) objective based on the Information Bottleneck (IB) principle. Based on this objective, we describe a Monte Carlo Regression Tree Search (MCRTS) algorithm that explores different splits to find the IB-optimal mimic tree. Experiments show that our mimic tree achieves strong approximation performance with significantly fewer nodes than baseline models. We demonstrate the interpretability of our mimic tree by showing latent traversals, decision rules, causal impacts, and human evaluation results.

        ----

        ## [1501] Only Train Once: A One-Shot Neural Network Training And Pruning Framework

        **Authors**: *Tianyi Chen, Bo Ji, Tianyu Ding, Biyi Fang, Guanyi Wang, Zhihui Zhu, Luming Liang, Yixin Shi, Sheng Yi, Xiao Tu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a376033f78e144f494bfc743c0be3330-Abstract.html)

        **Abstract**:

        Structured pruning is a commonly used technique in deploying deep neural networks (DNNs) onto resource-constrained devices. However, the existing pruning methods are usually heuristic, task-specified, and require an extra fine-tuning procedure. To overcome these limitations, we propose a framework that compresses DNNs into slimmer architectures with competitive performances and significant FLOPs reductions by Only-Train-Once (OTO). OTO contains two key steps: (i) we partition the parameters of DNNs into zero-invariant groups, enabling us to prune zero groups without affecting the output; and (ii) to promote zero groups, we then formulate a structured-sparsity optimization problem, and propose a novel optimization algorithm, Half-Space Stochastic Projected Gradient (HSPG), to solve it, which outperforms the standard proximal methods on group sparsity exploration, and maintains comparable convergence. To demonstrate the effectiveness of OTO, we train and compress full models simultaneously from scratch without fine-tuning for inference speedup and parameter reduction, and achieve state-of-the-art results on VGG16 for CIFAR10, ResNet50 for CIFAR10 and Bert for SQuAD and competitive result on ResNet50 for ImageNet. The source code is available at https://github.com/tianyic/onlytrainonce.

        ----

        ## [1502] Referring Transformer: A One-step Approach to Multi-task Visual Grounding

        **Authors**: *Muchen Li, Leonid Sigal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a376802c0811f1b9088828288eb0d3f0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a376802c0811f1b9088828288eb0d3f0-Abstract.html)

        **Abstract**:

        As an important step towards visual reasoning, visual grounding (e.g., phrase localization, referring expression comprehension / segmentation) has been widely explored. Previous approaches to referring expression comprehension (REC) or segmentation (RES) either suffer from limited performance, due to a two-stage setup, or require the designing of complex task-specific one-stage architectures. In this paper, we propose a simple one-stage multi-task framework for visual grounding tasks. Specifically, we leverage a transformer architecture, where two modalities are fused in a visual-lingual encoder. In the decoder, the model learns to generate contextualized lingual queries which are then decoded and used to directly regress the bounding box and produce a segmentation mask for the corresponding referred regions. With this simple but highly contextualized model, we outperform state-of-the-art methods by a large margin on both REC and RES tasks. We also show that a simple pre-training schedule (on an external dataset) further improves the performance. Extensive experiments and ablations illustrate that our model benefits greatly from contextualized information and multi-task training.

        ----

        ## [1503] Decoupling the Depth and Scope of Graph Neural Networks

        **Authors**: *Hanqing Zeng, Muhan Zhang, Yinglong Xia, Ajitesh Srivastava, Andrey Malevich, Rajgopal Kannan, Viktor K. Prasanna, Long Jin, Ren Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a378383b89e6719e15cd1aa45478627c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a378383b89e6719e15cd1aa45478627c-Abstract.html)

        **Abstract**:

        State-of-the-art Graph Neural Networks (GNNs) have limited scalability with respect to the graph and model sizes. On large graphs, increasing the model depth often means exponential expansion of the scope (i.e., receptive field). Beyond just a few layers, two fundamental challenges emerge:  1. degraded expressivity due to oversmoothing, and 2. expensive computation due to neighborhood explosion. We propose a design principle to decouple the depth and scope of GNNs – to generate representation of a target entity (i.e., a node or an edge), we first extract a localized subgraph as the bounded-size scope, and then apply a GNN of arbitrary depth on top of the subgraph. A properly extracted subgraph consists of a small number of critical neighbors, while excluding irrelevant ones. The GNN, no matter how deep it is, smooths the local neighborhood into informative representation rather than oversmoothing the global graph into “white noise”. Theoretically, decoupling improves the GNN expressive power from the perspectives of graph signal processing (GCN), function approximation (GraphSAGE) and topological learning (GIN). Empirically, on seven graphs (with up to 110M nodes) and six backbone GNN architectures, our design achieves significant accuracy improvement with orders of magnitude reduction in computation and hardware cost.

        ----

        ## [1504] Fast and Memory Efficient Differentially Private-SGD via JL Projections

        **Authors**: *Zhiqi Bu, Sivakanth Gopi, Janardhan Kulkarni, Yin Tat Lee, Judy Hanwen Shen, Uthaipon Tantipongpipat*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a3842ed7b3d0fe3ac263bcabd2999790-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a3842ed7b3d0fe3ac263bcabd2999790-Abstract.html)

        **Abstract**:

        Differentially Private-SGD (DP-SGD) of Abadi et al. and its variations are the only known algorithms for private training of large scale neural networks. This algorithm requires computation of per-sample gradients norms which is extremely slow and memory intensive in practice. In this paper, we present a new framework to design differentially private optimizers called DP-SGD-JL and DP-Adam-JL. Our approach uses Johnsonâ€“Lindenstrauss (JL) projections to quickly approximate the per-sample gradient norms without exactly computing them, thus making the training time and memory requirements of our optimizers closer to that of their non-DP versions. Unlike previous attempts to make DP-SGD faster which work only on a subset of network architectures or use compiler techniques, we propose an algorithmic solution which works for any network in a black-box manner which is the main contribution of this paper. To illustrate this, on IMDb dataset, we train a Recurrent Neural Network (RNN) to achieve good privacy-vs-accuracy tradeoff, while being significantly faster than DP-SGD and with a similar memory footprint as non-private SGD.

        ----

        ## [1505] Formalizing Generalization and Adversarial Robustness of Neural Networks to Weight Perturbations

        **Authors**: *Yu-Lin Tsai, Chia-Yi Hsu, Chia-Mu Yu, Pin-Yu Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a3ab4ff8fa4deed2e3bae3a5077675f0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a3ab4ff8fa4deed2e3bae3a5077675f0-Abstract.html)

        **Abstract**:

        Studying the sensitivity of weight perturbation in neural networks and its impacts on model performance, including generalization and robustness, is an active research topic due to its implications on a wide range of machine learning tasks such as model compression, generalization gap assessment, and adversarial attacks. In this paper, we provide the first integral study and analysis for feed-forward neural networks in terms of the robustness in pairwise class margin and its generalization behavior under weight perturbation. We further design a new theory-driven loss function for training generalizable and robust neural networks against weight perturbations. Empirical experiments are conducted to validate our theoretical analysis. Our results offer fundamental insights for characterizing the generalization and robustness of neural networks against weight perturbations.

        ----

        ## [1506] Pipeline Combinators for Gradual AutoML

        **Authors**: *Guillaume Baudart, Martin Hirzel, Kiran Kate, Parikshit Ram, Avraham Shinnar, Jason Tsay*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a3b36cb25e2e0b93b5f334ffb4e4064e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a3b36cb25e2e0b93b5f334ffb4e4064e-Abstract.html)

        **Abstract**:

        Automated machine learning (AutoML) can make data scientists more productive.  But if machine learning is totally automated, that leaves no room for data scientists to apply their intuition.  Hence, data scientists often prefer not total but gradual automation, where they control certain choices and AutoML explores the rest.  Unfortunately, gradual AutoML is cumbersome with state-of-the-art tools, requiring large non-compositional code changes.  More concise compositional code can be achieved with combinators, a powerful concept from functional programming.  This paper introduces a small set of orthogonal combinators for composing machine-learning operators into pipelines.  It describes a translation scheme from pipelines and associated hyperparameter schemas to search spaces for AutoML optimizers.  On that foundation, this paper presents Lale, an open-source sklearn-compatible AutoML library, and evaluates it with a user study.

        ----

        ## [1507] Boost Neural Networks by Checkpoints

        **Authors**: *Feng Wang, Guoyizhe Wei, Qiao Liu, Jinxiang Ou, Xian Wei, Hairong Lv*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a40511cad8383e5ae8ddd8b855d135da-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a40511cad8383e5ae8ddd8b855d135da-Abstract.html)

        **Abstract**:

        Training multiple deep neural networks (DNNs) and averaging their outputs is a simple way to improve the predictive performance. Nevertheless, the multiplied training cost prevents this ensemble method to be practical and efficient. Several recent works attempt to save and ensemble the checkpoints of DNNs, which only requires the same computational cost as training a single network. However, these methods suffer from either marginal accuracy improvements due to the low diversity of checkpoints or high risk of divergence due to the cyclical learning rates they adopted. In this paper, we propose a novel method to ensemble the checkpoints, where a boosting scheme is utilized to accelerate model convergence and maximize the checkpoint diversity. We theoretically prove that it converges by reducing exponential loss. The empirical evaluation also indicates our proposed ensemble outperforms single model and existing ensembles in terms of accuracy and efficiency. With the same training budget, our method achieves 4.16% lower error on Cifar-100 and 6.96% on Tiny-ImageNet with ResNet-110 architecture. Moreover, the adaptive sample weights in our method make it an effective solution to address the imbalanced class distribution. In the experiments, it yields up to 5.02% higher accuracy over single EfficientNet-B0 on the imbalanced datasets.

        ----

        ## [1508] Model Selection for Bayesian Autoencoders

        **Authors**: *Ba-Hien Tran, Simone Rossi, Dimitrios Milios, Pietro Michiardi, Edwin V. Bonilla, Maurizio Filippone*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a41db61e2728ef963614a8c8755b9b9a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a41db61e2728ef963614a8c8755b9b9a-Abstract.html)

        **Abstract**:

        We develop a novel method for carrying out model selection for Bayesian autoencoders (BAEs) by means of prior hyper-parameter optimization. Inspired by the common practice of type-II maximum likelihood optimization and its equivalence to Kullback-Leibler divergence minimization, we propose to optimize the distributional sliced-Wasserstein distance (DSWD) between the output of the autoencoder and the empirical data distribution. The advantages of this formulation are that we can estimate the DSWD based on samples and handle high-dimensional problems. We carry out posterior estimation of the BAE parameters via stochastic gradient Hamiltonian Monte Carlo and turn our BAE into a generative model by fitting a flexible Dirichlet mixture model in the latent space. Thanks to this approach, we obtain a powerful alternative to variational autoencoders, which are the preferred choice in modern application of autoencoders for representation learning with uncertainty. We evaluate our approach qualitatively and quantitatively using a vast experimental campaign on a number of unsupervised learning tasks and show that, in small-data regimes where priors matter,  our approach provides state-of-the-art results, outperforming multiple competitive baselines.

        ----

        ## [1509] Three Operator Splitting with Subgradients, Stochastic Gradients, and Adaptive Learning Rates

        **Authors**: *Alp Yurtsever, Alex Gu, Suvrit Sra*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a4267159aa970aa5a6542bcbb7ef575e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a4267159aa970aa5a6542bcbb7ef575e-Abstract.html)

        **Abstract**:

        Three Operator Splitting (TOS) (Davis & Yin, 2017) can minimize the sum of multiple convex functions effectively when an efficient gradient oracle or proximal operator is available for each term. This requirement often fails in machine learning applications: (i) instead of full gradients only stochastic gradients may be available; and (ii) instead of proximal operators, using subgradients to handle complex penalty functions may be more efficient and realistic. Motivated by these concerns, we analyze three potentially valuable extensions of TOS. The first two permit using subgradients and stochastic gradients, and are shown to ensure a $\mathcal{O}(1/\sqrt{t})$ convergence rate. The third extension AdapTOS endows TOS with adaptive step-sizes. For the important setting of optimizing a convex loss over the intersection of convex sets AdapTOS attains universal convergence rates, i.e., the rate adapts to the unknown smoothness degree of the objective. We compare our proposed methods with competing methods on various applications.

        ----

        ## [1510] Knowledge-Adaptation Priors

        **Authors**: *Mohammad Emtiyaz Khan, Siddharth Swaroop*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a4380923dd651c195b1631af7c829187-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a4380923dd651c195b1631af7c829187-Abstract.html)

        **Abstract**:

        Humans and animals have a natural ability to quickly adapt to their surroundings, but machine-learning models, when subjected to changes, often require a complete retraining from scratch. We present Knowledge-adaptation priors (K-priors) to reduce the cost of retraining by enabling quick and accurate adaptation for a wide-variety of tasks and models. This is made possible by a combination of weight and function-space priors to reconstruct the gradients of the past, which recovers and generalizes many existing, but seemingly-unrelated, adaptation strategies. Training with simple first-order gradient methods can often recover the exact retrained model to an arbitrary accuracy by choosing a sufficiently large memory of the past data. Empirical results show that adaptation with K-priors achieves performance similar to full retraining, but only requires training on a handful of past examples.

        ----

        ## [1511] Provably efficient multi-task reinforcement learning with model transfer

        **Authors**: *Chicheng Zhang, Zhi Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a440a3d316c5614c7a9310e902f4a43e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a440a3d316c5614c7a9310e902f4a43e-Abstract.html)

        **Abstract**:

        We study multi-task reinforcement learning (RL) in tabular episodic Markov decision processes (MDPs). We formulate a heterogeneous multi-player RL problem, in which a group of players concurrently face similar but not necessarily identical MDPs, with a goal of improving their collective performance through inter-player information sharing. We design and analyze a model-based algorithm, and provide gap-dependent and gap-independent regret upper and lower bounds that characterize the intrinsic complexity of the problem.

        ----

        ## [1512] Predicting Molecular Conformation via Dynamic Graph Score Matching

        **Authors**: *Shitong Luo, Chence Shi, Minkai Xu, Jian Tang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html)

        **Abstract**:

        Predicting stable 3D conformations from 2D molecular graphs has been a long-standing challenge in computational chemistry. Recently, machine learning approaches have demonstrated very promising results compared to traditional experimental and physics-based simulation methods. These approaches mainly focus on modeling the local interactions between neighboring atoms on the molecular graphs and overlook the long-range interactions between non-bonded atoms. However, these non-bonded atoms may be proximal to each other in 3D space, and modeling their interactions is of crucial importance to accurately determine molecular conformations, especially for large molecules and multi-molecular complexes. In this paper, we propose a new approach called Dynamic Graph Score Matching (DGSM) for molecular conformation prediction, which models both the local and long-range interactions by dynamically constructing graph structures between atoms according to their spatial proximity during both training and inference. Specifically, the DGSM directly estimates the gradient fields of the logarithm density of atomic coordinates according to the dynamically constructed graphs using score matching methods. The whole framework can be efficiently trained in an end-to-end fashion. Experiments across multiple tasks show that the DGSM outperforms state-of-the-art baselines by a large margin, and it is capable of generating conformations for a broader range of systems such as proteins and multi-molecular complexes.

        ----

        ## [1513] When in Doubt: Neural Non-Parametric Uncertainty Quantification for Epidemic Forecasting

        **Authors**: *Harshavardhan Kamarthi, Lingkai Kong, Alexander Rodríguez, Chao Zhang, B. Aditya Prakash*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a4a1108bbcc329a70efa93d7bf060914-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a4a1108bbcc329a70efa93d7bf060914-Abstract.html)

        **Abstract**:

        Accurate and trustworthy epidemic forecasting is an important problem for public health planning and disease mitigation. Most existing epidemic forecasting models disregard uncertainty quantification, resulting in mis-calibrated predictions. Recent works in deep neural models for uncertainty-aware time-series forecasting also have several limitations; e.g., it is difficult to specify proper priors in Bayesian NNs, while methods like deep ensembling can be computationally expensive. In this paper, we propose to use neural functional processes to fill this gap. We model epidemic time-series with a probabilistic generative process and propose a functional neural process model called EpiFNP, which directly models the probability distribution of the forecast value in a non-parametric way. In EpiFNP, we use a dynamic stochastic correlation graph to model the correlations between sequences, and design different stochastic latent variables to capture functional uncertainty from different perspectives. Our experiments in a real-time flu forecasting setting show that EpiFNP significantly outperforms state-of-the-art models in both accuracy and calibration metrics, up to 2.5x in accuracy and 2.4x in calibration. Additionally, as EpiFNP learns the relations between the current season and similar patterns of historical seasons, it enables interpretable forecasts. Beyond epidemic forecasting, EpiFNP can be of independent interest for advancing uncertainty quantification in deep sequential models for predictive analytics.

        ----

        ## [1514] Bounds all around: training energy-based models with bidirectional bounds

        **Authors**: *Cong Geng, Jia Wang, Zhiyong Gao, Jes Frellsen, Søren Hauberg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a4d8e2a7e0d0c102339f97716d2fdfb6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a4d8e2a7e0d0c102339f97716d2fdfb6-Abstract.html)

        **Abstract**:

        Energy-based models (EBMs) provide an elegant framework for density estimation, but they are notoriously difficult to train. Recent work has established links to generative adversarial networks, where the EBM is trained through a minimax game with a variational value function. We propose a bidirectional bound on the EBM log-likelihood, such that we maximize a lower bound and minimize an upper bound when solving the minimax game. We link one bound to a gradient penalty that stabilizes training, thereby provide grounding for best engineering practice. To evaluate the bounds we develop a new and efficient estimator of the Jacobi-determinant of the EBM generator. We demonstrate that these developments stabilize training and yield high-quality density estimation and sample generation.

        ----

        ## [1515] CogView: Mastering Text-to-Image Generation via Transformers

        **Authors**: *Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, Jie Tang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a4d92e2cd541fca87e4620aba658316d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a4d92e2cd541fca87e4620aba658316d-Abstract.html)

        **Abstract**:

        Text-to-Image generation in the general domain has long been an open problem, which requires both a powerful generative model and cross-modal understanding. We propose CogView, a 4-billion-parameter Transformer with VQ-VAE tokenizer to advance this problem. We also demonstrate the finetuning strategies for various downstream tasks, e.g. style learning, super-resolution, text-image ranking and fashion design, and methods to stabilize pretraining, e.g. eliminating NaN losses. CogView achieves the state-of-the-art FID on the blurred MS COCO dataset, outperforming previous GAN-based models and a recent similar work DALL-E.

        ----

        ## [1516] Time-independent Generalization Bounds for SGLD in Non-convex Settings

        **Authors**: *Tyler Farghly, Patrick Rebeschini*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a4ee59dd868ba016ed2de90d330acb6a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a4ee59dd868ba016ed2de90d330acb6a-Abstract.html)

        **Abstract**:

        We establish generalization error bounds for stochastic gradient Langevin dynamics (SGLD) with constant learning rate under the assumptions of dissipativity and smoothness, a setting that has received increased attention in the sampling/optimization literature. Unlike existing bounds for SGLD in non-convex settings, ours are time-independent and decay to zero as the sample size increases. Using the framework of uniform stability, we establish time-independent bounds by exploiting the Wasserstein contraction property of the Langevin diffusion, which also allows us to circumvent the need to bound gradients using Lipschitz-like assumptions. Our analysis also supports variants of SGLD that use different discretization methods, incorporate Euclidean projections, or use non-isotropic noise.

        ----

        ## [1517] Nonuniform Negative Sampling and Log Odds Correction with Rare Events Data

        **Authors**: *HaiYing Wang, Aonan Zhang, Chong Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a51c896c9cb81ecb5a199d51ac9fc3c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a51c896c9cb81ecb5a199d51ac9fc3c5-Abstract.html)

        **Abstract**:

        We investigate the issue of parameter estimation with nonuniform negative sampling for imbalanced data. We first prove that, with imbalanced data, the available information about unknown parameters is only tied to the relatively small number of positive instances, which justifies the usage of negative sampling. However, if the negative instances are subsampled to the same level of the positive cases, there is information loss. To maintain more information, we derive the asymptotic distribution of a general inverse probability weighted (IPW) estimator and obtain the optimal sampling probability that minimizes its variance. To further improve the estimation efficiency over the IPW method, we propose a likelihood-based estimator by correcting log odds for the sampled data and prove that the improved estimator has the smallest asymptotic variance among a large class of estimators. It is also more robust to pilot misspecification. We validate our approach on simulated data as well as a real click-through rate dataset with more than 0.3 trillion instances, collected over a period of a month. Both theoretical and empirical results demonstrate the effectiveness of our method.

        ----

        ## [1518] Algorithmic stability and generalization of an unsupervised feature selection algorithm

        **Authors**: *Xinxing Wu, Qiang Cheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a546203962b88771bb06faf8d6ec065e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a546203962b88771bb06faf8d6ec065e-Abstract.html)

        **Abstract**:

        Feature selection, as a vital dimension reduction technique, reduces data dimension by identifying an essential subset of input features, which can facilitate interpretable insights into learning and inference processes. Algorithmic stability is a key characteristic of an algorithm regarding its sensitivity to perturbations of input samples. In this paper, we propose an innovative unsupervised feature selection algorithm attaining this stability with provable guarantees. The architecture of our algorithm consists of a feature scorer and a feature selector. The scorer trains a neural network (NN) to globally score all the features, and the selector adopts a dependent sub-NN to locally evaluate the representation abilities for selecting features. Further, we present algorithmic stability analysis and show that our algorithm has a performance guarantee via a generalization error bound. Extensive experimental results on real-world datasets demonstrate superior generalization performance of our proposed algorithm to strong baseline methods. Also, the properties revealed by our theoretical analysis and the stability of our algorithm-selected features are empirically confirmed.

        ----

        ## [1519] On learning sparse vectors from mixture of responses

        **Authors**: *Nikita Polyanskii*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a5481cd6d7517aa3fc6476dc7d9019ab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a5481cd6d7517aa3fc6476dc7d9019ab-Abstract.html)

        **Abstract**:

        In this paper, we address two learning problems. Suppose a family of $\ell$ unknown sparse vectors is fixed, where each vector has at most $k$ non-zero elements. In the first problem, we concentrate on robust learning the supports of all vectors from the family using a sequence of noisy responses. Each response to a query vector shows the sign of the inner product between a randomly chosen vector from the family and the query vector. In the second problem, we aim at designing queries such that all sparse vectors from the family can be approximately reconstructed based on the error-free responses.  This learning model was introduced in the work of Gandikota et al., 2020, and these problems can be seen as generalizations of support recovery and approximate recovery problems, well-studied under the framework of  1-bit compressed sensing.  As the main contribution of the paper, we prove the existence of learning algorithms for the first problem which work without any assumptions. Under a mild structural assumption on the unknown vectors, we also show the existence of learning algorithms for the second problem and rigorously analyze their query complexity.

        ----

        ## [1520] Convergence and Alignment of Gradient Descent with Random Backpropagation Weights

        **Authors**: *Ganlin Song, Ruitu Xu, John Lafferty*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a576eafbce762079f7d1f77fca1c5cc2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a576eafbce762079f7d1f77fca1c5cc2-Abstract.html)

        **Abstract**:

        Stochastic gradient descent with backpropagation is the workhorse of artificial neural networks. It has long been recognized that backpropagation fails to be a biologically plausible algorithm. Fundamentally, it is a non-local procedure---updating one neuron's synaptic weights requires knowledge of synaptic weights or receptive fields of downstream neurons. This limits the use of artificial neural networks as a tool for understanding the biological principles of information processing in the brain. Lillicrap et al. (2016) propose a more biologically plausible "feedback alignment" algorithm that uses random and fixed backpropagation weights, and show promising simulations. In this paper we study the mathematical properties of the feedback alignment procedure by analyzing convergence and alignment for two-layer networks under squared error loss. In the overparameterized setting, we prove that the error converges to zero exponentially fast, and also that regularization is necessary in order for the Â parameters to become aligned with the random backpropagation weights. Simulations are given that are consistent with this analysis and suggest further generalizations. These results contribute to our understanding of how biologically plausible algorithms might carry out weight learning in a manner different from Hebbian learning, with performance that is comparable with the full non-local backpropagation algorithm.

        ----

        ## [1521] Adder Attention for Vision Transformer

        **Authors**: *Han Shu, Jiahao Wang, Hanting Chen, Lin Li, Yujiu Yang, Yunhe Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a57e8915461b83adefb011530b711704-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a57e8915461b83adefb011530b711704-Abstract.html)

        **Abstract**:

        Transformer is a new kind of calculation paradigm for deep learning which has shown strong performance on a large variety of computer vision tasks. However, compared with conventional deep models (e.g., convolutional neural networks), vision transformers require more computational resources which cannot be easily deployed on mobile devices. To this end, we present to reduce the energy consumptions using adder neural network (AdderNet). We first theoretically analyze the mechanism of self-attention and the difficulty for applying adder operation into this module. Specifically, the feature diversity, i.e., the rank of attention map using only additions cannot be well preserved. Thus, we develop an adder attention layer that includes an additional identity mapping. With the new operation, vision transformers constructed using additions can also provide powerful feature representations. Experimental results on several benchmarks demonstrate that the proposed approach can achieve highly competitive performance to that of the baselines while achieving an about 2~3Ã— reduction on the energy consumption.

        ----

        ## [1522] Reverse engineering learned optimizers reveals known and novel mechanisms

        **Authors**: *Niru Maheswaranathan, David Sussillo, Luke Metz, Ruoxi Sun, Jascha Sohl-Dickstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a57ecd54d4df7d999bd9c5e3b973ec75-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a57ecd54d4df7d999bd9c5e3b973ec75-Abstract.html)

        **Abstract**:

        Learned optimizers are parametric algorithms that can themselves be trained to solve optimization problems. In contrast to baseline optimizers (such as momentum or Adam) that use simple update rules derived from theoretical principles, learned optimizers use flexible, high-dimensional, nonlinear parameterizations. Although this can lead to better performance, their inner workings remain a mystery. How is a given learned optimizer able to outperform a well tuned baseline? Has it learned a sophisticated combination of existing optimization techniques, or is it implementing completely new behavior? In this work, we address these questions by careful analysis and visualization of learned optimizers. We study learned optimizers trained from scratch on four disparate tasks, and discover that they have learned interpretable behavior, including: momentum, gradient clipping, learning rate schedules, and new forms of learning rate adaptation. Moreover, we show how dynamics and mechanisms inside of learned optimizers orchestrate these computations. Our results help elucidate the previously murky understanding of how learned optimizers work, and establish tools for interpreting future learned optimizers.

        ----

        ## [1523] Matching a Desired Causal State via Shift Interventions

        **Authors**: *Jiaqi Zhang, Chandler Squires, Caroline Uhler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a5a61717dddc3501cfdf7a4e22d7dbaa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a5a61717dddc3501cfdf7a4e22d7dbaa-Abstract.html)

        **Abstract**:

        Transforming a causal system from a given initial state to a desired target state is an important task permeating multiple fields including control theory, biology, and materials science. In causal models, such transformations can be achieved by performing a set of interventions. In this paper, we consider the problem of identifying a shift intervention that matches the desired mean of a system through active learning. We define the Markov equivalence class that is identifiable from shift interventions and propose two active learning strategies that are guaranteed to exactly match a desired mean. We then derive a worst-case lower bound for the number of interventions required and show that these strategies are optimal for certain classes of graphs. In particular, we show that our strategies may require exponentially fewer interventions than the previously considered approaches, which optimize for structure learning in the underlying causal graph. In line with our theoretical results, we also demonstrate experimentally that our proposed active learning strategies require fewer interventions compared to several baselines.

        ----

        ## [1524] Unsupervised Noise Adaptive Speech Enhancement by Discriminator-Constrained Optimal Transport

        **Authors**: *Hsin-Yi Lin, Huan-Hsin Tseng, Xugang Lu, Yu Tsao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a5c7b30fb632c92feb59154517223dc9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a5c7b30fb632c92feb59154517223dc9-Abstract.html)

        **Abstract**:

        This paper presents a novel discriminator-constrained optimal transport network (DOTN) that performs unsupervised domain adaptation for speech enhancement (SE), which is an essential regression task in speech processing. The DOTN aims to estimate clean references of noisy speech in a target domain, by exploiting the knowledge available from the source domain. The domain shift between training and testing data has been reported to be an obstacle to learning problems in diverse fields. Although rich literature exists on unsupervised domain adaptation for classification, the methods proposed, especially in regressions, remain scarce and often depend on additional information regarding the input data. The proposed DOTN approach tactically fuses the optimal transport (OT) theory from mathematical analysis with generative adversarial frameworks, to help evaluate continuous labels in the target domain. The experimental results on two SE tasks demonstrate that by extending the classical OT formulation, our proposed DOTN outperforms previous adversarial domain adaptation frameworks in a purely unsupervised manner.

        ----

        ## [1525] Optimality of variational inference for stochasticblock model with missing links

        **Authors**: *Solenne Gaucher, Olga Klopp*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a5e308070bd6dd3cc56283f2313522de-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a5e308070bd6dd3cc56283f2313522de-Abstract.html)

        **Abstract**:

        Variational methods are extremely popular in the analysis of network data. Statistical guarantees obtained for these methods typically provide asymptotic normality for the problem of estimation of global model parameters under the stochastic block model. In the present work, we consider the case of networks with missing links that is important in application and show that the variational approximation to the maximum likelihood estimator converges at the minimax rate. This provides the first minimax optimal and tractable estimator for the problem of parameter estimation for the stochastic block model with missing links. We complement our results with numerical studies of simulated and real networks, which confirm the advantages of this estimator over current methods.

        ----

        ## [1526] Policy Learning Using Weak Supervision

        **Authors**: *Jingkang Wang, Hongyi Guo, Zhaowei Zhu, Yang Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a613863f6a3ada47ae5bca2a558872d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a613863f6a3ada47ae5bca2a558872d1-Abstract.html)

        **Abstract**:

        Most existing policy learning solutions require the learning agents to receive high-quality supervision signals, e.g., rewards in reinforcement learning (RL) or high-quality expert demonstrations in behavioral cloning (BC). These quality supervisions are either infeasible or prohibitively expensive to obtain in practice. We aim for a unified framework that leverages the available cheap weak supervisions to perform policy learning efficiently. To handle this problem, we treat the weak supervision'' as imperfect information coming from a peer agent, and evaluate the learning agent's policy based on a  correlated agreement'' with the peer agent's policy (instead of simple agreements). Our approach explicitly punishes a policy for overfitting to the weak supervision. In addition to theoretical guarantees, extensive evaluations on tasks including RL with noisy reward, BC with weak demonstrations, and standard policy co-training (RL + BC) show that our method leads to substantial performance improvements, especially when the complexity or the noise of the learning environments is high.

        ----

        ## [1527] Chasing Sparsity in Vision Transformers: An End-to-End Exploration

        **Authors**: *Tianlong Chen, Yu Cheng, Zhe Gan, Lu Yuan, Lei Zhang, Zhangyang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a61f27ab2165df0e18cc9433bd7f27c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a61f27ab2165df0e18cc9433bd7f27c5-Abstract.html)

        **Abstract**:

        Vision transformers (ViTs) have recently received explosive popularity, but their enormous model sizes and training costs remain daunting. Conventional post-training pruning often incurs higher training budgets. In contrast, this paper aims to trim down both the training memory overhead and the inference complexity, without sacrificing the achievable accuracy. We carry out the first-of-its-kind comprehensive exploration, on taking a unified approach of integrating sparsity in ViTs "from end to end''. Specifically, instead of training full ViTs, we dynamically extract and train sparse subnetworks, while sticking to a fixed small parameter budget. Our approach jointly optimizes model parameters and explores connectivity throughout training, ending up with one sparse network as the final output. The approach is seamlessly extended from unstructured to structured sparsity, the latter by considering to guide the prune-and-grow of self-attention heads inside ViTs. We further co-explore data and architecture sparsity for additional efficiency gains by plugging in a novel learnable token selector to adaptively determine the currently most vital patches. Extensive results on ImageNet with diverse ViT backbones validate the effectiveness of our proposals which obtain significantly reduced computational cost and almost unimpaired generalization. Perhaps most surprisingly, we find that the proposed sparse (co-)training can sometimes \textit{improve the ViT accuracy} rather than compromising it, making sparsity a tantalizing "free lunch''. For example, our sparsified DeiT-Small at ($5\%$, $50\%$) sparsity for (data, architecture), improves $\mathbf{0.28\%}$ top-1 accuracy, and meanwhile enjoys $\mathbf{49.32\%}$ FLOPs and $\mathbf{4.40\%}$ running time savings. Our codes are available at https://github.com/VITA-Group/SViTE.

        ----

        ## [1528] Graphical Models in Heavy-Tailed Markets

        **Authors**: *José Vinícius de Miranda Cardoso, Jiaxi Ying, Daniel P. Palomar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a64a034c3cb8eac64eb46ea474902797-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a64a034c3cb8eac64eb46ea474902797-Abstract.html)

        **Abstract**:

        Heavy-tailed statistical distributions have long been considered a more realistic statistical model for the data generating process in financial markets in comparison to their Gaussian counterpart. Nonetheless, mathematical nuisances, including nonconvexities, involved in estimating graphs in heavy-tailed settings pose a significant challenge to the practical design of algorithms for graph learning. In this work, we present graph learning estimators based on the Markov random field framework that assume a Student-$t$ data generating process. We design scalable numerical algorithms, via the alternating direction method of multipliers, to learn both connected and $k$-component graphs along with their theoretical convergence guarantees. The proposed methods outperform state-of-the-art benchmarks in an extensive series of practical experiments with publicly available data from the S\&P500 index, foreign exchanges, and cryptocurrencies.

        ----

        ## [1529] A Shading-Guided Generative Implicit Model for Shape-Accurate 3D-Aware Image Synthesis

        **Authors**: *Xingang Pan, Xudong Xu, Chen Change Loy, Christian Theobalt, Bo Dai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a64c94baaf368e1840a1324e839230de-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a64c94baaf368e1840a1324e839230de-Abstract.html)

        **Abstract**:

        The advancement of generative radiance fields has pushed the boundary of 3D-aware image synthesis. Motivated by the observation that a 3D object should look realistic from multiple viewpoints, these methods introduce a multi-view constraint as regularization to learn valid 3D radiance fields from 2D images. Despite the progress, they often fall short of capturing accurate 3D shapes due to the shape-color ambiguity, limiting their applicability in downstream tasks. In this work, we address this ambiguity by proposing a novel shading-guided generative implicit model that is able to learn a starkly improved shape representation. Our key insight is that an accurate 3D shape should also yield a realistic rendering under different lighting conditions. This multi-lighting constraint is realized by modeling illumination explicitly and performing shading with various lighting conditions. Gradients are derived by feeding the synthesized images to a discriminator. To compensate for the additional computational burden of calculating surface normals, we further devise an efficient volume rendering strategy via surface tracking, reducing the training and inference time by 24% and 48%, respectively. Our experiments on multiple datasets show that the proposed approach achieves photorealistic 3D-aware image synthesis while capturing accurate underlying 3D shapes. We demonstrate improved performance of our approach on 3D shape reconstruction against existing methods, and show its applicability on image relighting. Our code is available at https://github.com/XingangPan/ShadeGAN.

        ----

        ## [1530] XCiT: Cross-Covariance Image Transformers

        **Authors**: *Alaaeldin Ali, Hugo Touvron, Mathilde Caron, Piotr Bojanowski, Matthijs Douze, Armand Joulin, Ivan Laptev, Natalia Neverova, Gabriel Synnaeve, Jakob Verbeek, Hervé Jégou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a655fbe4b8d7439994aa37ddad80de56-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a655fbe4b8d7439994aa37ddad80de56-Abstract.html)

        **Abstract**:

        Following their success in natural language processing, transformers have recently shown much promise for computer vision. The self-attention operation underlying transformers yields global interactions between all tokens ,i.e. words or image patches, and enables flexible modelling of image data beyond the local interactions of convolutions. This flexibility, however, comes with a quadratic complexity in time and memory, hindering application to long sequences and high-resolution images. We propose a “transposed” version of self-attention that operates across feature channels rather than tokens, where the interactions are based on the cross-covariance matrix between keys and queries. The resulting cross-covariance attention (XCA) has linear complexity in the number of tokens, and allows efficient processing of high-resolution images.Our cross-covariance image transformer (XCiT) is built upon XCA. It combines the accuracy of conventional transformers with the scalability of convolutional architectures. We validate the effectiveness and generality of XCiT by reporting excellent results on multiple vision benchmarks, including image classification and self-supervised feature learning on ImageNet-1k, object detection and instance segmentation on COCO, and semantic segmentation on ADE20k.We will opensource our code and trained models to reproduce the reported results.

        ----

        ## [1531] Row-clustering of a Point Process-valued Matrix

        **Authors**: *Lihao Yin, Ganggang Xu, Huiyan Sang, Yongtao Guan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a6a38989dc7e433f1f42388e7afca318-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a6a38989dc7e433f1f42388e7afca318-Abstract.html)

        **Abstract**:

        Structured point process data harvested from various platforms poses new challenges to the machine learning community. To cluster repeatedly observed marked point processes, we propose a novel mixture model of multi-level marked point processes for identifying potential heterogeneity in the observed data. Specifically, we study a matrix whose entries are marked log-Gaussian Cox processes and cluster rows of such a matrix.  An efficient semi-parametric Expectation-Solution (ES) algorithm combined with functional principal component analysis (FPCA) of point processes is proposed for model estimation. The effectiveness of the proposed framework is demonstrated through simulation studies and real data analyses.

        ----

        ## [1532] Fine-Grained Neural Network Explanation by Identifying Input Features with Predictive Information

        **Authors**: *Yang Zhang, Ashkan Khakzar, Yawei Li, Azade Farshad, Seong Tae Kim, Nassir Navab*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a6d259bfbfa2062843ef543e21d7ec8e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a6d259bfbfa2062843ef543e21d7ec8e-Abstract.html)

        **Abstract**:

        One principal approach for illuminating a black-box neural network is feature attribution, i.e. identifying the importance of input features for the networkâ€™s prediction. The predictive information of features is recently proposed as a proxy for the measure of their importance. So far, the predictive information is only identified for latent features by placing an information bottleneck within the network. We propose a method to identify features with predictive information in the input domain. The method results in fine-grained identification of input features' information and is agnostic to network architecture. The core idea of our method is leveraging a bottleneck on the input that only lets input features associated with predictive latent features pass through. We compare our method with several feature attribution methods using mainstream feature attribution evaluation experiments. The code is publicly available.

        ----

        ## [1533] Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints

        **Authors**: *Maura Pintor, Fabio Roli, Wieland Brendel, Battista Biggio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html)

        **Abstract**:

        Evaluating adversarial robustness amounts to finding the minimum perturbation needed to have an input sample misclassified. The inherent complexity of the underlying optimization requires current gradient-based attacks to be carefully tuned, initialized, and possibly executed for many computationally-demanding iterations, even if specialized to a given perturbation model.In this work, we overcome these limitations by proposing a fast minimum-norm (FMN) attack that works with different $\ell_p$-norm perturbation models ($p=0, 1, 2, \infty$), is robust to hyperparameter choices, does not require adversarial starting points, and converges within few lightweight steps. It works by iteratively finding the sample misclassified with maximum confidence within an $\ell_p$-norm constraint of size $\epsilon$, while adapting $\epsilon$ to minimize the distance of the current sample to the decision boundary.Extensive experiments show that FMN significantly outperforms existing $\ell_0$, $\ell_1$, and $\ell_\infty$-norm attacks in terms of perturbation size, convergence speed and computation time, while reporting comparable performances with state-of-the-art $\ell_2$-norm attacks. Our open-source code is available at: https://github.com/pralab/Fast-Minimum-Norm-FMN-Attack.

        ----

        ## [1534] Uncertainty Quantification and Deep Ensembles

        **Authors**: *Rahul Rahaman, Alexandre H. Thiéry*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a70dc40477bc2adceef4d2c90f47eb82-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a70dc40477bc2adceef4d2c90f47eb82-Abstract.html)

        **Abstract**:

        Deep Learning methods are known to suffer from calibration issues: they typically produce over-confident estimates. These problems are exacerbated in the low data regime. Although the calibration of probabilistic models is well studied, calibrating extremely over-parametrized models in the low-data regime presents unique challenges. We show that deep-ensembles do not necessarily lead to improved calibration properties. In fact, we show that standard ensembling methods, when used in conjunction with modern techniques such as mixup regularization, can lead to less calibrated models. This text examines the interplay between three of the most simple and commonly used approaches to leverage deep learning when data is scarce: data-augmentation, ensembling, and post-processing calibration methods. We demonstrate that, although standard ensembling techniques certainly help to boost accuracy, the calibration of deep ensembles relies on subtle trade-offs. We also find that calibration methods such as temperature scaling need to be slightly tweaked when used with deep-ensembles and, crucially, need to be executed after the averaging process. Our simulations indicate that, in the low data regime, this simple strategy can halve the Expected Calibration Error (ECE) on a range of benchmark classification problems when compared to standard deep-ensembles.

        ----

        ## [1535] Directed Probabilistic Watershed

        **Authors**: *Enrique Fita Sanmartin, Sebastian Damrich, Fred A. Hamprecht*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a73d9b34d6f7c322fa3e34c633b1297d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a73d9b34d6f7c322fa3e34c633b1297d-Abstract.html)

        **Abstract**:

        The Probabilistic Watershed is a semi-supervised learning algorithm applied on undirected graphs. Given a set of labeled nodes (seeds), it defines a Gibbs probability distribution over all possible spanning forests disconnecting the seeds. It calculates, for every node, the probability of sampling a forest connecting a certain seed with the considered node. We propose the "Directed Probabilistic Watershed", an extension of the Probabilistic Watershed algorithm to directed graphs. Building on the Probabilistic Watershed, we apply the Matrix Tree Theorem for directed graphs and define a Gibbs probability distribution over all incoming directed forests rooted at the seeds. Similar to the undirected case, this turns out to be equivalent to the Directed Random Walker. Furthermore, we show that in the limit case in which the Gibbs distribution has infinitely low temperature, the labeling of the Directed Probabilistic Watershed is equal to the one induced by the incoming directed forest of minimum cost. Finally, for illustration, we compare the empirical performance of the proposed method with other semi-supervised segmentation methods for directed graphs.

        ----

        ## [1536] Laplace Redux - Effortless Bayesian Deep Learning

        **Authors**: *Erik Daxberger, Agustinus Kristiadi, Alexander Immer, Runa Eschenhagen, Matthias Bauer, Philipp Hennig*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a7c9585703d275249f30a088cebba0ad-Abstract.html)

        **Abstract**:

        Bayesian formulations of deep learning have been shown to have compelling theoretical properties and offer practical functional benefits, such as improved predictive uncertainty quantification and model selection. The Laplace approximation (LA) is a classic, and arguably the simplest family of approximations for the intractable posteriors of deep neural networks. Yet, despite its simplicity, the LA is not as popular as alternatives like variational Bayes or deep ensembles. This may be due to assumptions that the LA is expensive due to the involved Hessian computation, that it is difficult to implement, or that it yields inferior results. In this work we show that these are misconceptions: we (i) review the range of variants of the LA including versions with minimal cost overhead; (ii) introduce "laplace", an easy-to-use software library for PyTorch offering user-friendly access to all major flavors of the LA; and (iii) demonstrate through extensive experiments that the LA is competitive with more popular alternatives in terms of performance, while excelling in terms of computational cost. We hope that this work will serve as a catalyst to a wider adoption of the LA in practical deep learning, including in domains where Bayesian approaches are not typically considered at the moment.

        ----

        ## [1537] Hessian Eigenspectra of More Realistic Nonlinear Models

        **Authors**: *Zhenyu Liao, Michael W. Mahoney*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a7d8ae4569120b5bec12e7b6e9648b86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a7d8ae4569120b5bec12e7b6e9648b86-Abstract.html)

        **Abstract**:

        Given an optimization problem, the Hessian matrix and its eigenspectrum can be used in many ways, ranging from designing more efficient second-order algorithms to performing model analysis and regression diagnostics. When nonlinear models and non-convex problems are considered, strong simplifying assumptions are often made to make Hessian spectral analysis more tractable.This leads to the question of how relevant the conclusions of such analyses are for realistic nonlinear models. In this paper, we exploit tools from random matrix theory to make a precise characterization of the Hessian eigenspectra for a broad family of nonlinear models that extends the classical generalized linear models, without relying on strong simplifying assumptions used previously. We show that, depending on the data properties, the nonlinear response model, and the loss function, the Hessian can have qualitatively different spectral behaviors: of bounded or unbounded support, with single- or multi-bulk, and with isolated eigenvalues on the left- or right-hand side of the main eigenvalue bulk. By focusing on such a simple but nontrivial model, our analysis takes a step forward to unveil the theoretical origin of many visually striking features observed in more realistic machine learning models.

        ----

        ## [1538] Explicable Reward Design for Reinforcement Learning Agents

        **Authors**: *Rati Devidze, Goran Radanovic, Parameswaran Kamalaruban, Adish Singla*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a7f0d2b95c60161b3f3c82f764b1d1c9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a7f0d2b95c60161b3f3c82f764b1d1c9-Abstract.html)

        **Abstract**:

        We study the design of explicable reward functions for a reinforcement learning agent while guaranteeing that an optimal policy induced by the function belongs to a set of target policies. By being explicable, we seek to capture two properties: (a) informativeness so that the rewards speed up the agent's convergence, and (b) sparseness as a proxy for ease of interpretability of the rewards. The key challenge is that higher informativeness typically requires dense rewards for many learning tasks, and existing techniques do not allow one to balance these two properties appropriately. In this paper, we investigate the problem from the perspective of discrete optimization and introduce a novel framework, ExpRD, to design explicable reward functions. ExpRD builds upon an informativeness criterion that captures the (sub-)optimality of target policies at different time horizons in terms of actions taken from any given starting state. We provide a  mathematical analysis of ExpRD, and show its connections to existing reward design techniques, including potential-based reward shaping. Experimental results on two navigation tasks demonstrate the effectiveness of ExpRD in designing explicable reward functions.

        ----

        ## [1539] A Minimalist Approach to Offline Reinforcement Learning

        **Authors**: *Scott Fujimoto, Shixiang Shane Gu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html)

        **Abstract**:

        Offline reinforcement learning (RL) defines the task of learning from a fixed batch of data. Due to errors in value estimation from out-of-distribution actions, most offline RL algorithms take the approach of constraining or regularizing the policy with the actions contained in the dataset. Built on pre-existing RL algorithms, modifications to make an RL algorithm work offline comes at the cost of additional complexity. Offline RL algorithms introduce new hyperparameters and often leverage secondary components such as generative models, while adjusting the underlying RL algorithm. In this paper we aim to make a deep RL algorithm work while making minimal changes. We find that we can match the performance of state-of-the-art offline RL algorithms by simply adding a behavior cloning term to the policy update of an online RL algorithm and normalizing the data. The resulting algorithm is a simple to implement and tune baseline, while more than halving the overall run time by removing the additional computational overheads of previous methods.

        ----

        ## [1540] SIMONe: View-Invariant, Temporally-Abstracted Object Representations via Unsupervised Video Decomposition

        **Authors**: *Rishabh Kabra, Daniel Zoran, Goker Erdogan, Loic Matthey, Antonia Creswell, Matt M. Botvinick, Alexander Lerchner, Christopher P. Burgess*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a860a7886d7c7e2a8d3eaac96f76dc0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a860a7886d7c7e2a8d3eaac96f76dc0d-Abstract.html)

        **Abstract**:

        To help agents reason about scenes in terms of their building blocks, we wish to extract the compositional structure of any given scene (in particular, the configuration and characteristics of objects comprising the scene). This problem is especially difficult when scene structure needs to be inferred while also estimating the agent’s location/viewpoint, as the two variables jointly give rise to the agent’s observations. We present an unsupervised variational approach to this problem. Leveraging the shared structure that exists across different scenes, our model learns to infer two sets of latent representations from RGB video input alone: a set of "object" latents, corresponding to the time-invariant, object-level contents of the scene, as well as a set of "frame" latents, corresponding to global time-varying elements such as viewpoint. This factorization of latents allows our model, SIMONe, to represent object attributes in an allocentric manner which does not depend on viewpoint. Moreover, it allows us to disentangle object dynamics and summarize their trajectories as time-abstracted, view-invariant, per-object properties. We demonstrate these capabilities, as well as the model's performance in terms of view synthesis and instance segmentation, across three procedurally generated video datasets.

        ----

        ## [1541] Simple Stochastic and Online Gradient Descent Algorithms for Pairwise Learning

        **Authors**: *Zhenhuan Yang, Yunwen Lei, Puyu Wang, Tianbao Yang, Yiming Ying*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a87d27f712df362cd22c7a8ef823e987-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a87d27f712df362cd22c7a8ef823e987-Abstract.html)

        **Abstract**:

        Pairwise learning refers to learning tasks where  the loss function depends on a pair of  instances. It instantiates many important machine learning tasks such as bipartite ranking and metric learning. A popular approach to handle streaming data in pairwise learning is an online gradient descent (OGD) algorithm, where one needs to pair the current instance with a buffering set of previous instances with a sufficiently large size and therefore suffers from a scalability issue. In this paper, we propose simple stochastic and online gradient descent methods for pairwise learning. A notable difference from the existing studies  is that we only pair the current instance with the previous one in building a gradient direction, which is efficient in both the storage and computational complexity. We develop novel stability results, optimization, and generalization error bounds for both convex and nonconvex as well as both smooth and nonsmooth problems. We introduce novel techniques to decouple the dependency of models and the previous instance in both the optimization and generalization analysis. Our study resolves an open question on developing meaningful generalization bounds for OGD using a buffering set with a very small fixed size. We also extend our algorithms and stability analysis to develop differentially private SGD algorithms for pairwise learning which significantly improves the existing results.

        ----

        ## [1542] User-Level Differentially Private Learning via Correlated Sampling

        **Authors**: *Badih Ghazi, Ravi Kumar, Pasin Manurangsi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a89cf525e1d9f04d16ce31165e139a4b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a89cf525e1d9f04d16ce31165e139a4b-Abstract.html)

        **Abstract**:

        Most works in learning with differential privacy (DP) have focused on the setting where each user has a single sample. In this work, we consider the setting where each user holds $m$ samples and the privacy protection is enforced at the level of each user's data.  We show that, in this setting, we may learn with a much fewer number of users. Specifically, we show that, as long as each user receives sufficiently many samples, we can learn any privately learnable class via an $(\epsilon, \delta)$-DP algorithm using only $O(\log(1/\delta)/\epsilon)$ users. For $\epsilon$-DP algorithms, we show that we can learn using only $O_{\epsilon}(d)$ users even in the local model, where $d$ is the probabilistic representation dimension. In both cases, we show a nearly-matching lower bound on the number of users required.A crucial component of our results is a generalization of global stability [Bun, Livni, Moran, FOCS 2020]  that allows the use of public randomness. Under this relaxed notion, we employ a correlated sampling strategy to show that the global stability can be boosted to be arbitrarily close to one, at a polynomial expense in the number of samples.

        ----

        ## [1543] Asynchronous Decentralized Online Learning

        **Authors**: *Jiyan Jiang, Wenpeng Zhang, Jinjie Gu, Wenwu Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a8e864d04c95572d1aece099af852d0a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a8e864d04c95572d1aece099af852d0a-Abstract.html)

        **Abstract**:

        Most existing algorithms in decentralized online learning are conducted in the synchronous setting. However, synchronization makes these algorithms suffer from the straggler problem, i.e., fast learners have to wait for slow learners, which significantly reduces such algorithms' overall efficiency. To overcome this problem, we study decentralized online learning in the asynchronous setting, which allows different learners to work at their own pace. We first formulate the framework of Asynchronous Decentralized Online Convex Optimization, which specifies the whole process of asynchronous decentralized online learning using a sophisticated event indexing system. Then we propose the Asynchronous Decentralized Online Gradient-Push (AD-OGP) algorithm, which performs asymmetric gossiping communication and instantaneous model averaging. We further derive a regret bound of AD-OGP, which is a function of the network topology, the levels of processing delays, and the levels of communication delays. Extensive experiments show that AD-OGP runs significantly faster than its synchronous counterpart and also verify the theoretical results.

        ----

        ## [1544] Multi-Step Budgeted Bayesian Optimization with Unknown Evaluation Costs

        **Authors**: *Raul Astudillo, Daniel R. Jiang, Maximilian Balandat, Eytan Bakshy, Peter I. Frazier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a8ecbabae151abacba7dbde04f761c37-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a8ecbabae151abacba7dbde04f761c37-Abstract.html)

        **Abstract**:

        Bayesian optimization (BO) is a sample-efficient approach to optimizing costly-to-evaluate black-box functions. Most BO methods ignore how evaluation costs may vary over the optimization domain. However, these costs can be highly heterogeneous and are often unknown in advance in many practical settings, such as hyperparameter tuning of machine learning algorithms or physics-based simulation optimization. Moreover, those few existing methods that acknowledge cost heterogeneity do not naturally accommodate a budget constraint on the total evaluation cost. This combination of unknown costs and a budget constraint introduces a new dimension to the exploration-exploitation trade-off, where learning about the cost incurs a cost itself. Existing methods do not reason about the various trade-offs of this problem in a principled way, leading often to poor performance. We formalize this claim by proving that the expected improvement and the expected improvement per unit of cost, arguably the two most widely used acquisition functions in practice, can be arbitrarily inferior with respect to the optimal non-myopic policy. To overcome the shortcomings of existing approaches,  we propose the budgeted multi-step expected improvement, a non-myopic acquisition function that generalizes classical expected improvement to the setting of heterogeneous and unknown evaluation costs. We show that our acquisition function outperforms existing methods in a variety of synthetic and real problems.

        ----

        ## [1545] Model-Based Domain Generalization

        **Authors**: *Alexander Robey, George J. Pappas, Hamed Hassani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a8f12d9486cbcc2fe0cfc5352011ad35-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a8f12d9486cbcc2fe0cfc5352011ad35-Abstract.html)

        **Abstract**:

        Despite remarkable success in a variety of applications, it is well-known that deep learning can fail catastrophically when presented with out-of-distribution data.  Toward addressing this challenge, we consider the \emph{domain generalization} problem, wherein predictors are trained using data drawn from a family of related training domains and then evaluated on a distinct and unseen test domain.  We show that under a natural model of data generation and a concomitant invariance condition, the domain generalization problem is equivalent to an infinite-dimensional constrained statistical learning problem; this problem forms the basis of our approach, which we call Model-Based Domain Generalization.  Due to the inherent challenges in solving constrained optimization problems in deep learning, we exploit nonconvex duality theory to develop unconstrained relaxations of this statistical problem with tight bounds on the duality gap.  Based on this theoretical motivation, we propose a novel domain generalization algorithm with convergence guarantees.   In our experiments, we report improvements of up to 30% over state-of-the-art domain generalization baselines on several benchmarks including ColoredMNIST, Camelyon17-WILDS, FMoW-WILDS, and PACS.

        ----

        ## [1546] $\alpha$-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression

        **Authors**: *Jiabo He, Sarah M. Erfani, Xingjun Ma, James Bailey, Ying Chi, Xian-Sheng Hua*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a8f15eda80c50adb0e71943adc8015cf-Abstract.html)

        **Abstract**:

        Bounding box (bbox) regression is a fundamental task in computer vision. So far, the most commonly used loss functions for bbox regression are the Intersection over Union (IoU) loss and its variants. In this paper, we generalize existing IoU-based losses to a new family of power IoU losses that have a power IoU term and an additional power regularization term with a single power parameter $\alpha$. We call this new family of losses the $\alpha$-IoU losses and analyze properties such as order preservingness and loss/gradient reweighting. Experiments on multiple object detection benchmarks and models demonstrate that $\alpha$-IoU losses, 1) can surpass existing IoU-based losses by a noticeable performance margin; 2) offer detectors more flexibility in achieving different levels of bbox regression accuracy by modulating $\alpha$; and 3) are more robust to small datasets and noisy bboxes.

        ----

        ## [1547] Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient

        **Authors**: *David L. Applegate, Mateo Díaz, Oliver Hinder, Haihao Lu, Miles Lubin, Brendan O'Donoghue, Warren Schudy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a8fbbd3b11424ce032ba813493d95ad7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a8fbbd3b11424ce032ba813493d95ad7-Abstract.html)

        **Abstract**:

        We present PDLP, a practical first-order method for linear programming (LP) that can solve to the high levels of accuracy that are expected in traditional LP applications. In addition, it can scale to very large problems because its core operation is matrix-vector multiplications. PDLP is derived by applying the primal-dual hybrid gradient (PDHG) method, popularized by Chambolle and Pock (2011), to a saddle-point formulation of LP. PDLP enhances PDHG for LP by combining several new techniques with older tricks from the literature; the enhancements include diagonal preconditioning, presolving, adaptive step sizes, and adaptive restarting. PDLP improves the state of the art for first-order methods applied to LP. We compare PDLP with SCS, an ADMM-based solver, on a set of 383 LP instances derived from MIPLIB 2017. With a target of $10^{-8}$ relative accuracy and 1 hour time limit, PDLP achieves a 6.3x reduction in the geometric mean of solve times and a 4.6x reduction in the number of instances unsolved (from 227 to 49). Furthermore, we highlight standard benchmark instances and a large-scale application (PageRank) where our open-source prototype of PDLP, written in Julia, outperforms a commercial LP solver.

        ----

        ## [1548] On the Provable Generalization of Recurrent Neural Networks

        **Authors**: *Lifu Wang, Bo Shen, Bo Hu, Xing Cao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a928731e103dfc64c0027fa84709689e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a928731e103dfc64c0027fa84709689e-Abstract.html)

        **Abstract**:

        Recurrent Neural Network (RNN) is a fundamental structure in deep learning. Recently, some works study the training process of over-parameterized neural networks, and show that over-parameterized networks can learn functions in some notable concept classes with a provable generalization error bound. In this paper, we analyze the training and generalization for RNNs with random initialization, and provide the following improvements over recent works:(1) For a RNN with input sequence $x=(X_1,X_2,...,X_L)$, previous works study to learn functions that are summation of $f(\beta^T_lX_l)$ and require normalized conditions that $||X_l||\leq\epsilon$ with some very small $\epsilon$ depending on the complexity of $f$. In this paper, using detailed analysis about the neural tangent kernel matrix, we prove a generalization error bound to learn such functions without normalized conditions and show that some notable concept classes are learnable with  the numbers of iterations and samples scaling almost-polynomially in the input length $L$.(2) Moreover, we prove a novel result to learn N-variables functions of input sequence  with the form $f(\beta^T[X_{l_1},...,X_{l_N}])$, which do not belong to the ``additive'' concept class, i,e., the summation of function $f(X_l)$. And we show that when either $N$ or $l_0=\max(l_1,..,l_N)-\min(l_1,..,l_N)$ is small, $f(\beta^T[X_{l_1},...,X_{l_N}])$ will be learnable with the number iterations and samples scaling  almost-polynomially in the input length $L$.

        ----

        ## [1549] Differentiable Spline Approximations

        **Authors**: *Minsu Cho, Aditya Balu, Ameya Joshi, Anjana Deva Prasad, Biswajit Khara, Soumik Sarkar, Baskar Ganapathysubramanian, Adarsh Krishnamurthy, Chinmay Hegde*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a952ddeda0b7e2c20744e52e728e5594-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a952ddeda0b7e2c20744e52e728e5594-Abstract.html)

        **Abstract**:

        The paradigm of differentiable programming has significantly enhanced the scope of machine learning via the judicious use of gradient-based optimization. However, standard differentiable programming methods (such as autodiff) typically require that the machine learning models be differentiable, limiting their applicability. Our goal in this paper is to use a new, principled approach to extend gradient-based optimization to functions well modeled by splines, which encompass a large family of piecewise polynomial models. We derive the form of the (weak) Jacobian of such functions and show that it exhibits a block-sparse structure that can be computed implicitly and efficiently. Overall, we show that leveraging this redesigned Jacobian in the form of a differentiable "layer'' in predictive models leads to improved performance in diverse applications such as image segmentation, 3D point cloud reconstruction, and finite element analysis. We also open-source the code at \url{https://github.com/idealab-isu/DSA}.

        ----

        ## [1550] Rate-Optimal Subspace Estimation on Random Graphs

        **Authors**: *Zhixin Zhou, Fan Zhou, Ping Li, Cun-Hui Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a97f6e2fedcabc887911dc9b5fd3ccc3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a97f6e2fedcabc887911dc9b5fd3ccc3-Abstract.html)

        **Abstract**:

        We study the theory of random bipartite graph whose adjacency matrix is generated according to a connectivity matrix $M$. We consider the bipartite graph to be sparse, i.e., the entries of $M$ are upper bounded by certain sparsity parameter. We show that the performance of estimating the connectivity matrix $M$ depends on the sparsity of the graph. We focus on two measurement of performance of estimation: the error of estimating $M$ and the error of estimating the column space of $M$. In the first case, we consider the operator norm and Frobenius norm of the difference between the estimation and the true connectivity matrix. In the second case, the performance will be measured by the difference between the estimated projection matrix and the true projection matrix in operator norm and Frobenius norm. We will show that the estimators we propose achieve the minimax optimal rate.

        ----

        ## [1551] Estimating the Unique Information of Continuous Variables

        **Authors**: *Ari Pakman, Amin Nejatbakhsh, Dar Gilboa, Abdullah Makkeh, Luca Mazzucato, Michael Wibral, Elad Schneidman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a9a1d5317a33ae8cef33961c34144f84-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a9a1d5317a33ae8cef33961c34144f84-Abstract.html)

        **Abstract**:

        The integration and transfer of information from multiple sources to multiple targets is a core motive of neural systems. The emerging field of partial information decomposition (PID) provides a novel information-theoretic lens into these mechanisms by identifying synergistic, redundant, and unique contributions to the mutual information between one and several variables. While many works have studied aspects of PID for Gaussian and discrete distributions, the case of general continuous distributions is still uncharted territory. In this work we present a method for estimating the unique information in continuous distributions, for the case of one versus two variables. Our method solves the associated optimization problem over the space of distributions with fixed bivariate  marginals by combining copula decompositions and techniques developed to optimize variational autoencoders. We obtain excellent agreement with known analytic results for Gaussians, and  illustrate the power of our new approach in several brain-inspired neural models. Our method is capable of recovering the effective connectivity of a chaotic network of rate neurons, and uncovers a complex trade-off between redundancy, synergy and unique information in recurrent networks trained to solve a generalized XOR~task.

        ----

        ## [1552] Reliable Causal Discovery with Improved Exact Search and Weaker Assumptions

        **Authors**: *Ignavier Ng, Yujia Zheng, Jiji Zhang, Kun Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a9b4ec2eb4ab7b1b9c3392bb5388119d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a9b4ec2eb4ab7b1b9c3392bb5388119d-Abstract.html)

        **Abstract**:

        Many of the causal discovery methods rely on the faithfulness assumption to guarantee asymptotic correctness. However, the assumption can be approximately violated in many ways, leading to sub-optimal solutions. Although there is a line of research in Bayesian network structure learning that focuses on weakening the assumption, such as exact search methods with well-defined score functions, they do not scale well to large graphs. In this work, we introduce several strategies to improve the scalability of exact score-based methods in the linear Gaussian setting. In particular, we develop a super-structure estimation method based on the support of inverse covariance matrix which requires assumptions that are strictly weaker than faithfulness, and apply it to restrict the search space of exact search. We also propose a local search strategy that performs exact search on the local clusters formed by each variable and its neighbors within two hops in the super-structure. Numerical experiments validate the efficacy of the proposed procedure, and demonstrate that it scales up to hundreds of nodes with a high accuracy.

        ----

        ## [1553] Node Dependent Local Smoothing for Scalable Graph Learning

        **Authors**: *Wentao Zhang, Mingyu Yang, Zeang Sheng, Yang Li, Wen Ouyang, Yangyu Tao, Zhi Yang, Bin Cui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/a9eb812238f753132652ae09963a05e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/a9eb812238f753132652ae09963a05e9-Abstract.html)

        **Abstract**:

        Recent works reveal that feature or label smoothing lies at the core of Graph Neural Networks (GNNs). Concretely, they show feature smoothing combined with simple linear regression achieves comparable performance with the carefully designed GNNs, and a simple MLP model with label smoothing of its prediction can outperform the vanilla GCN. Though an interesting finding, smoothing has not been well understood, especially regarding how to control the extent of smoothness. Intuitively, too small or too large smoothing iterations may cause under-smoothing or over-smoothing and can lead to sub-optimal performance. Moreover, the extent of smoothness is node-specific, depending on its degree and local structure. To this end, we propose a novel algorithm called node-dependent local smoothing (NDLS), which aims to control the smoothness of every node by setting a node-specific smoothing iteration. Specifically, NDLS computes influence scores based on the adjacency matrix and selects the iteration number by setting a threshold on the scores. Once selected, the iteration number can be applied to both feature smoothing and label smoothing. Experimental results demonstrate that NDLS enjoys high accuracy -- state-of-the-art performance on node classifications tasks, flexibility -- can be incorporated with any models, scalability and efficiency -- can support large scale graphs with fast training.

        ----

        ## [1554] Parallel and Efficient Hierarchical k-Median Clustering

        **Authors**: *Vincent Cohen-Addad, Silvio Lattanzi, Ashkan Norouzi-Fard, Christian Sohler, Ola Svensson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/aa495e18c7e3a21a4e48923b92048a61-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/aa495e18c7e3a21a4e48923b92048a61-Abstract.html)

        **Abstract**:

        As a fundamental unsupervised learning task, hierarchical clustering has been extensively studied in the past decade. In particular, standard metric formulations as hierarchical $k$-center, $k$-means,  and $k$-median received a lot of attention and the problems have been studied extensively in different models of computation. Despite all this interest, not many efficient parallel algorithms are known for these problems. In this paper we introduce a new parallel algorithm for the Euclidean hierarchical $k$-median problem that, when using machines with memory $s$ (for $s\in \Omega(\log^2 (n+\Delta+d))$), outputs a hierarchical clustering such that for every fixed value of $k$ the cost of the solution is at most an $O(\min\{d, \log n\} \log \Delta)$ factor larger in expectation than that of an optimal solution. Furthermore, we also get that for all $k$ simultanuously the cost of the solution is at most an $O(\min\{d, \log n\} \log \Delta \log (\Delta d n))$ factor bigger that the corresponding optimal solution.  The algorithm requires in $O\left(\log_{s} (nd\log(n+\Delta))\right)$ rounds. Here $d$ is the dimension of the data set and $\Delta$ is  the ratio between the maximum and minimum distance of two points in the input dataset. To the best of our knowledge, this is the first \emph{parallel} algorithm  for the hierarchical $k$-median problem with theoretical guarantees. We further complement our theoretical results with an empirical study of our algorithm that shows its effectiveness in practice.

        ----

        ## [1555] Human-Adversarial Visual Question Answering

        **Authors**: *Sasha Sheng, Amanpreet Singh, Vedanuj Goswami, Jose Alberto Lopez Magana, Tristan Thrush, Wojciech Galuba, Devi Parikh, Douwe Kiela*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/aa97d584861474f4097cf13ccb5325da-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/aa97d584861474f4097cf13ccb5325da-Abstract.html)

        **Abstract**:

        Performance on the most commonly used Visual Question Answering dataset (VQA v2) is starting to approach human accuracy. However, in interacting with state-of-the-art VQA models, it is clear that the problem is far from being solved. In order to stress test VQA models, we benchmark them against human-adversarial examples. Human subjects interact with a state-of-the-art VQA model, and for each image in the dataset, attempt to find a question where the modelâ€™s predicted answer is incorrect. We find that a wide range of state-of-the-art models perform poorly when evaluated on these examples. We conduct an extensive analysis of the collected adversarial examples and provide guidance on future research directions. We hope that this Adversarial VQA (AdVQA) benchmark can help drive progress in the field and advance the state of the art.

        ----

        ## [1556] Across-animal odor decoding by probabilistic manifold alignment

        **Authors**: *Pedro Herrero-Vidal, Dmitry Rinberg, Cristina Savin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/aad64398a969ec3186800d412fa7ab31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/aad64398a969ec3186800d412fa7ab31-Abstract.html)

        **Abstract**:

        Identifying the common structure of neural dynamics across subjects is key for extracting unifying principles of brain computation and for many brain machine interface applications. Here, we propose a novel probabilistic approach for aligning stimulus-evoked responses from multiple animals in a common low dimensional manifold and use hierarchical inference to identify which stimulus drives neural activity in any given trial. Our probabilistic decoder is robust to a range of features of the neural responses and significantly outperforms existing neural alignment procedures. When applied to recordings from the mouse olfactory bulb, our approach reveals low-dimensional population dynamics that are odor specific and have consistent structure across animals. Thus, our decoder can be used for increasing the robustness and scalability of neural-based chemical detection.

        ----

        ## [1557] Excess Capacity and Backdoor Poisoning

        **Authors**: *Naren Manoj, Avrim Blum*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/aaebdb8bb6b0e73f6c3c54a0ab0c6415-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/aaebdb8bb6b0e73f6c3c54a0ab0c6415-Abstract.html)

        **Abstract**:

        A backdoor data poisoning attack is an adversarial attack wherein the attacker injects several watermarked, mislabeled training examples into a training set. The watermark does not impact the test-time performance of the model on typical data; however, the model reliably errs on watermarked examples.To gain a better foundational understanding of backdoor data poisoning attacks, we present a formal theoretical framework within which one can discuss backdoor data poisoning attacks for classification problems. We then use this to analyze important statistical and computational issues surrounding these attacks.On the statistical front, we identify a parameter we call the memorization capacity that captures the intrinsic vulnerability of a learning problem to a backdoor attack. This allows us to argue about the robustness of several natural learning problems to backdoor attacks. Our results favoring the attacker involve presenting explicit constructions of backdoor attacks, and our robustness results show that some natural problem settings cannot yield successful backdoor attacks.From a computational standpoint, we show that under certain assumptions, adversarial training can detect the presence of backdoors in a training set. We then show that under similar assumptions, two closely related problems we call backdoor filtering and robust generalization are nearly equivalent. This implies that it is both asymptotically necessary and sufficient to design algorithms that can identify watermarked examples in the training set in order to obtain a learning algorithm that both generalizes well to unseen data and is robust to backdoors.

        ----

        ## [1558] A Convergence Analysis of Gradient Descent on Graph Neural Networks

        **Authors**: *Pranjal Awasthi, Abhimanyu Das, Sreenivas Gollapudi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/aaf2979785deb27864047e0ea40ef1b7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/aaf2979785deb27864047e0ea40ef1b7-Abstract.html)

        **Abstract**:

        Graph Neural Networks~(GNNs) are a powerful class of architectures for solving learning problems on graphs. While many variants of GNNs have been proposed in the literature and have achieved strong empirical performance, their theoretical properties are less well understood. In this work we study the convergence properties of the gradient descent algorithm when used to train GNNs. In particular, we consider the realizable setting where the data is generated from a network with unknown weights and our goal is to study conditions under which gradient descent on a GNN architecture can recover near optimal solutions. While such analysis has been performed in recent years for other architectures such as fully connected feed-forward networks, the message passing nature of the updates in a GNN poses a new challenge in understanding the nature of the gradient descent updates. We take a step towards overcoming this by proving that for the case of deep linear GNNs gradient descent provably recovers solutions up to error $\epsilon$ in $O(\text{log}(1/\epsilon))$ iterations, under natural assumptions on the data distribution. Furthermore, for the case of one-round GNNs with ReLU activations, we show that gradient descent provably recovers solutions up to error $\epsilon$ in $O(\frac{1}{\epsilon^2} \log(\frac{1}{\epsilon}))$ iterations.

        ----

        ## [1559] Differentiable rendering with perturbed optimizers

        **Authors**: *Quentin Le Lidec, Ivan Laptev, Cordelia Schmid, Justin Carpentier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ab233b682ec355648e7891e66c54191b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ab233b682ec355648e7891e66c54191b-Abstract.html)

        **Abstract**:

        Reasoning about 3D scenes from their 2D image projections is one of the core problems in computer vision. Solutions to this inverse and ill-posed problem typically involve a search for models that best explain observed image data. Notably, images depend both on the properties of observed scenes and on the process of image formation. Hence, if optimization techniques should be used to explain images, it is crucial to design differentable functions for the projection of 3D scenes into images, also known as differentiable rendering. Previous approaches to differentiable rendering typically replace non-differentiable operations by smooth approximations, impacting the subsequent 3D estimation. In this paper, we take a more general approach and study differentiable renderers through the prism of randomized optimization and the related notion of perturbed optimizers. In particular, our work highlights the link between some well-known differentiable renderer formulations and randomly smoothed optimizers, and introduces differentiable perturbed renderers. We also propose a variance reduction mechanism to alleviate the computational burden inherent to perturbed optimizers and introduce an adaptive scheme to automatically adjust the smoothing parameters of the rendering process. We apply our method to 3D scene reconstruction and demonstrate its advantages on the tasks of 6D pose estimation and 3D mesh reconstruction.  By providing informative gradients that can be used as a strong supervisory signal, we demonstrate the benefits of perturbed renderers to obtain more accurate solutions when compared to the state-of-the-art alternatives using smooth gradient approximations.

        ----

        ## [1560] BCORLE(λ): An Offline Reinforcement Learning and Evaluation Framework for Coupons Allocation in E-commerce Market

        **Authors**: *Yang Zhang, Bo Tang, Qingyu Yang, Dou An, Hongyin Tang, Chenyang Xi, Xueying Li, Feiyu Xiong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ab452534c5ce28c4fbb0e102d4a4fb2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ab452534c5ce28c4fbb0e102d4a4fb2e-Abstract.html)

        **Abstract**:

        Coupons allocation is an important tool for enterprises to increase the activity and loyalty of users on the e-commerce market. One fundamental problem related is how to allocate coupons within a fixed budget while maximizing users' retention on the e-commerce platform. The online e-commerce environment is complicated and ever changing, so it requires the coupons allocation policy learning can quickly adapt to the changes of the company's business strategy. Unfortunately, existing studies with a huge computation overhead can hardly satisfy the requirements of real-time and fast-response in the real world. Specifically, the problem of coupons allocation within a fixed budget is usually formulated as a Lagrangian problem. Existing solutions need to re-learn the policy once the value of Lagrangian multiplier variable $\lambda$ is updated, causing a great computation overhead. Besides, a mature e-commerce market often faces tens of millions of users and dozens of types of coupons which construct the huge policy space, further increasing the difficulty of solving the problem. To tackle with above problems, we propose a budget constrained offline reinforcement learning and evaluation with $\lambda$-generalization (BCORLE($\lambda$)) framework. The proposed method can help enterprises develop a coupons allocation policy which greatly improves users' retention rate on the platform while ensuring the cost does not exceed the budget. Specifically, $\lambda$-generalization method is proposed to lead the policy learning process can be executed according to different $\lambda$ values adaptively, avoiding re-learning new polices from scratch. Thus the computation overhead is greatly reduced. Further, a novel offline reinforcement learning method and an off-policy evaluation algorithm are proposed for policy learning and policy evaluation, respectively. Finally, experiments on the simulation platform and real-world e-commerce market validate the effectiveness of our approach.

        ----

        ## [1561] Nested Variational Inference

        **Authors**: *Heiko Zimmermann, Hao Wu, Babak Esmaeili, Jan-Willem van de Meent*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ab49b208848abe14418090d95df0d590-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ab49b208848abe14418090d95df0d590-Abstract.html)

        **Abstract**:

        We develop nested variational inference (NVI), a family of methods that learn proposals for nested importance samplers by minimizing an forward or reverse KL divergence at each level of nesting. NVI is applicable to many commonly-used importance sampling strategies and provides a mechanism for learning intermediate densities, which can serve as heuristics to guide the sampler. Our experiments apply NVI to (a) sample from a multimodal distribution using a learned annealing path (b) learn heuristics that approximate the likelihood of future observations in a hidden Markov model and (c) to perform amortized inference in hierarchical deep generative models. We observe that optimizing nested objectives leads to improved sample quality in terms of log average weight and effective sample size.

        ----

        ## [1562] Exponential Bellman Equation and Improved Regret Bounds for Risk-Sensitive Reinforcement Learning

        **Authors**: *Yingjie Fei, Zhuoran Yang, Yudong Chen, Zhaoran Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ab6439fa2daf0246f92eea433bca5ac4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ab6439fa2daf0246f92eea433bca5ac4-Abstract.html)

        **Abstract**:

        We study risk-sensitive reinforcement learning (RL) based on the entropic risk measure. Although existing works have established non-asymptotic regret guarantees for this problem, they leave open an exponential gap between the upper and lower bounds. We identify the deficiencies in existing algorithms and their analysis that result in such a gap. To remedy these deficiencies, we investigate a simple transformation of the risk-sensitive Bellman equations, which we call the exponential Bellman equation. The exponential Bellman equation inspires us to develop a novel analysis of Bellman backup procedures in risk-sensitive RL algorithms, and further motivates the design of a novel exploration mechanism. We show that these analytic and algorithmic innovations together lead to improved regret upper bounds over existing ones.

        ----

        ## [1563] On sensitivity of meta-learning to support data

        **Authors**: *Mayank Agarwal, Mikhail Yurochkin, Yuekai Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ab73f542b6d60c4de151800b8abc0a6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ab73f542b6d60c4de151800b8abc0a6c-Abstract.html)

        **Abstract**:

        Meta-learning algorithms are widely used for few-shot learning. For example, image recognition systems that readily adapt to unseen classes after seeing only a few labeled examples. Despite their success, we show that modern meta-learning algorithms are extremely sensitive to the data used for adaptation, i.e. support data. In particular, we demonstrate the existence of (unaltered, in-distribution, natural) images that, when used for adaptation, yield accuracy as low as 4\% or as high as 95\% on standard few-shot image classification benchmarks. We explain our empirical findings in terms of class margins, which in turn suggests that robust and safe meta-learning requires larger margins than supervised learning.

        ----

        ## [1564] On Large-Cohort Training for Federated Learning

        **Authors**: *Zachary Charles, Zachary Garrett, Zhouyuan Huo, Sergei Shmulyian, Virginia Smith*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ab9ebd57177b5106ad7879f0896685d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ab9ebd57177b5106ad7879f0896685d4-Abstract.html)

        **Abstract**:

        Federated learning methods typically learn a model by iteratively sampling updates from a population of clients. In this work, we explore how the number of clients sampled at each round (the cohort size) impacts the quality of the learned model and the training dynamics of federated learning algorithms. Our work poses three fundamental questions. First, what challenges arise when trying to scale federated learning to larger cohorts? Second, what parallels exist between cohort sizes in federated learning, and batch sizes in centralized learning? Last, how can we design federated learning methods that effectively utilize larger cohort sizes? We give partial answers to these questions based on extensive empirical evaluation. Our work highlights a number of challenges stemming from the use of larger cohorts. While some of these (such as generalization issues and diminishing returns) are analogs of large-batch training challenges, others (including catastrophic training failures and fairness concerns) are unique to federated learning.

        ----

        ## [1565] Generic Neural Architecture Search via Regression

        **Authors**: *Yuhong Li, Cong Hao, Pan Li, Jinjun Xiong, Deming Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/aba53da2f6340a8b89dc96d09d0d0430-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/aba53da2f6340a8b89dc96d09d0d0430-Abstract.html)

        **Abstract**:

        Most existing neural architecture search (NAS) algorithms are dedicated to and evaluated by the downstream tasks, e.g., image classification in computer vision. However, extensive experiments have shown that, prominent neural architectures, such as ResNet in computer vision and LSTM in natural language processing, are generally good at extracting patterns from the input data and perform well on different downstream tasks. In this paper, we attempt to answer two fundamental questions related to NAS. (1) Is it necessary to use the performance of specific downstream tasks to evaluate and search for good neural architectures? (2) Can we perform NAS effectively and efficiently while being agnostic to the downstream tasks? To answer these questions, we propose a novel and generic NAS framework, termed Generic NAS (GenNAS). GenNAS does not use task-specific labels but instead adopts regression on a set of manually designed synthetic signal bases for architecture evaluation. Such a self-supervised regression task can effectively evaluate the intrinsic power of an architecture to capture and transform the input signal patterns, and allow more sufficient usage of training samples. Extensive experiments across 13 CNN search spaces and one NLP space demonstrate the remarkable efficiency of GenNAS using regression, in terms of both evaluating the neural architectures (quantified by the ranking correlation Spearman's rho between the approximated performances and the downstream task performances) and the convergence speed for training (within a few seconds). For example, on NAS-Bench-101, GenNAS achieves 0.85 rho while the existing efficient methods only achieve 0.38. We then propose an automatic task search to optimize the combination of synthetic signals using limited downstream-task-specific labels, further improving the performance of GenNAS. We also thoroughly evaluate GenNAS's generality and end-to-end NAS performance on all search spaces, which outperforms almost all existing works with significant speedup. For example, on NASBench-201, GenNAS can find near-optimal architectures within 0.3 GPU hour.

        ----

        ## [1566] The best of both worlds: stochastic and adversarial episodic MDPs with unknown transition

        **Authors**: *Tiancheng Jin, Longbo Huang, Haipeng Luo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/abb9d15b3293a96a3ea116867b2b16d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/abb9d15b3293a96a3ea116867b2b16d5-Abstract.html)

        **Abstract**:

        We consider the best-of-both-worlds problem for learning an episodic Markov Decision Process through $T$ episodes, with the goal of achieving $\widetilde{\mathcal{O}}(\sqrt{T})$ regret when the losses are adversarial and simultaneously $\mathcal{O}(\log T)$ regret when the losses are (almost) stochastic. Recent work by [Jin and Luo, 2020]  achieves this goal when the fixed transition is known, and leaves the case of unknown transition as a major open question. In this work, we resolve this open problem by using the same Follow-the-Regularized-Leader (FTRL) framework together with a set of new techniques. Specifically, we first propose a loss-shifting trick in the FTRL analysis, which greatly simplifies the approach of [Jin and Luo, 2020] and already improves their results for the known transition case. Then, we extend this idea to the unknown transition case and develop a novel analysis which upper bounds the transition estimation error by the regret itself in the stochastic setting, a key property to ensure $\mathcal{O}(\log T)$ regret.

        ----

        ## [1567] Private learning implies quantum stability

        **Authors**: *Yihui Quek, Srinivasan Arunachalam, John A. Smolin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/abdbeb4d8dbe30df8430a8394b7218ef-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/abdbeb4d8dbe30df8430a8394b7218ef-Abstract.html)

        **Abstract**:

        Learning an unknown n-qubit quantum state rho is a fundamental challenge in quantum computing. Information-theoretically, it is known that tomography requires exponential in n many copies of rho to estimate its entries. Motivated by learning theory, Aaronson et al. introduced many (weaker) learning models: the PAC model of learning states (Proceedings of Royal Society A'07), shadow tomography (STOC'18) for learning shadows" of a state, a model that also requires learners to be differentially private (STOC'19) and the online model of learning states (NeurIPS'18). In these models it was shown that an unknown state can be learnedapproximately" using linear in n many copies of rho. But is there any relationship between these models? In this paper we prove a sequence of (information-theoretic) implications from differentially-private PAC learning to online learning and then to quantum stability.Our main result generalizes the recent work of Bun, Livni and Moran (Journal of the ACM'21) who showed that finite Littlestone dimension (of Boolean-valued concept classes) implies PAC learnability in the (approximate) differentially private (DP) setting. We first consider their work in the real-valued setting and further extend to their techniques to the setting of learning quantum states. Key to our results is our generic quantum online learner, Robust Standard Optimal Algorithm (RSOA), which is robust to adversarial imprecision. We then show information-theoretic implications between DP learning quantum states in the PAC model, learnability of quantum states in the one-way communication model, online learning of quantum states, quantum stability (which is our conceptual contribution), various combinatorial parameters and give further applications to gentle shadow tomography and noisy quantum state learning.

        ----

        ## [1568] Interesting Object, Curious Agent: Learning Task-Agnostic Exploration

        **Authors**: *Simone Parisi, Victoria Dean, Deepak Pathak, Abhinav Gupta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/abe8e03e3ac71c2ec3bfb0de042638d8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/abe8e03e3ac71c2ec3bfb0de042638d8-Abstract.html)

        **Abstract**:

        Common approaches for task-agnostic exploration learn tabula-rasa --the agent assumes isolated environments and no prior knowledge or experience. However, in the real world, agents learn in many environments and always come with prior experiences as they explore new ones. Exploration is a lifelong process. In this paper, we propose a paradigm change in the formulation and evaluation of task-agnostic exploration. In this setup, the agent first learns to explore across many environments without any extrinsic goal in a task-agnostic manner.Later on, the agent effectively transfers the learned exploration policy to better explore new environments when solving tasks. In this context, we evaluate several baseline exploration strategies and present a simple yet effective approach to learning task-agnostic exploration policies. Our key idea is that there are two components of exploration: (1) an agent-centric component encouraging exploration of unseen parts of the environment based on an agentâ€™s belief; (2) an environment-centric component encouraging exploration of inherently interesting objects. We show that our formulation is effective and provides the most consistent exploration across several training-testing environment pairs. We also introduce benchmarks and metrics for evaluating task-agnostic exploration strategies. The source code is available at https://github.com/sparisi/cbet/.

        ----

        ## [1569] SimiGrad: Fine-Grained Adaptive Batching for Large Scale Training using Gradient Similarity Measurement

        **Authors**: *Heyang Qin, Samyam Rajbhandari, Olatunji Ruwase, Feng Yan, Lei Yang, Yuxiong He*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/abea47ba24142ed16b7d8fbf2c740e0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/abea47ba24142ed16b7d8fbf2c740e0d-Abstract.html)

        **Abstract**:

        Large scale training requires massive parallelism to finish the training within a reasonable amount of time. To support massive parallelism, large batch training is the key enabler but often at the cost of generalization performance. Existing works explore adaptive batching or hand-tuned static large batching, in order to strike a balance between the computational efficiency and the performance. However, these methods can provide only coarse-grained adaption (e.g., at a epoch level) due to the intrinsic expensive calculation or hand tuning requirements. In this paper, we propose a fully automated and lightweight adaptive batching methodology to enable fine-grained batch size adaption (e.g., at a mini-batch level) that can achieve state-of-the-art performance with record breaking batch sizes. The core component of our method is a lightweight yet efficient representation of the critical gradient noise information. We open-source the proposed methodology by providing a plugin tool that supports mainstream machine learning frameworks. Extensive evaluations on popular benchmarks (e.g., CIFAR10, ImageNet, and BERT-Large) demonstrate that the proposed methodology outperforms state-of-the-art methodologies using adaptive batching approaches or hand-tuned static strategies in both performance and batch size. Particularly, we achieve a new state-of-the-art batch size of 78k in BERT-Large pretraining with SQuAD score 90.69 compared to 90.58 reported in previous state-of-the-art with 59k batch size.

        ----

        ## [1570] Variational Inference for Continuous-Time Switching Dynamical Systems

        **Authors**: *Lukas Köhs, Bastian Alt, Heinz Koeppl*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/abec16f483abb4f1810ca029aadf8446-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/abec16f483abb4f1810ca029aadf8446-Abstract.html)

        **Abstract**:

        Switching dynamical systems provide a powerful, interpretable modeling framework for inference in time-series data in, e.g., the natural sciences or engineering applications. Since many areas, such as biology or discrete-event systems, are naturally described in continuous time, we present a model based on a Markov jump process modulating a subordinated diffusion process. We provide the exact evolution equations for the prior and posterior marginal densities, the direct solutions of which are however computationally intractable. Therefore, we develop a new continuous-time variational inference algorithm, combining a Gaussian process approximation on the diffusion level with posterior inference for Markov jump processes. By minimizing the path-wise Kullback-Leibler divergence we obtain (i) Bayesian latent state estimates for arbitrary points on the real axis and (ii) point estimates of unknown system parameters, utilizing variational expectation maximization. We extensively evaluate our algorithm under the model assumption and for real-world examples.

        ----

        ## [1571] Implicit Regularization in Matrix Sensing via Mirror Descent

        **Authors**: *Fan Wu, Patrick Rebeschini*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/abf0931987f2f8eb7a8d26f2c21fe172-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/abf0931987f2f8eb7a8d26f2c21fe172-Abstract.html)

        **Abstract**:

        We study discrete-time mirror descent applied to the unregularized empirical risk in matrix sensing. In both the general case of rectangular matrices and the particular case of positive semidefinite matrices, a simple potential-based analysis in terms of the Bregman divergence allows us to establish convergence of mirror descent---with different choices of the mirror maps---to a matrix that, among all global minimizers of the empirical risk, minimizes a quantity explicitly related to the nuclear norm, the Frobenius norm, and the von Neumann entropy. In both cases, this characterization implies that mirror descent, a first-order algorithm minimizing the unregularized empirical risk, recovers low-rank matrices under the same set of assumptions that are sufficient to guarantee recovery for nuclear-norm minimization. When the sensing matrices are symmetric and commute, we show that gradient descent with full-rank factorized parametrization is a first-order approximation to mirror descent, in which case we obtain an explicit characterization of the implicit bias of gradient flow as a by-product.

        ----

        ## [1572] STORM+: Fully Adaptive SGD with Recursive Momentum for Nonconvex Optimization

        **Authors**: *Kfir Y. Levy, Ali Kavis, Volkan Cevher*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac10ff1941c540cd87c107330996f4f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac10ff1941c540cd87c107330996f4f6-Abstract.html)

        **Abstract**:

        In this work we investigate stochastic non-convex optimization problems where the objective is an expectation over smooth loss functions, and the goal is to find an approximate stationary point. The most popular approach to handling such problems is variance reduction techniques, which are also known to obtain tight convergence rates, matching the lower bounds in this case. Nevertheless, these techniques require a careful maintenance of anchor points in conjunction with appropriately selected ``mega-batchsizes". This leads to a challenging hyperparameter tuning problem, that weakens their practicality. Recently, [Cutkosky and Orabona, 2019] have shown that one can employ recursive momentum in order to avoid the use of anchor points and large batchsizes, and still obtain the optimal rate for this setting. Yet, their method called $\rm{STORM}$ crucially relies on the knowledge of the smoothness, as well a bound on the gradient norms. In this work we propose $\rm{STORM}^{+}$, a new method that is completely parameter-free, does not require large batch-sizes, and obtains the optimal $O(1/T^{1/3})$ rate for finding an approximate stationary point. Our work builds on the $\rm{STORM}$ algorithm, in conjunction with a novel approach to adaptively set the learning rate and momentum parameters.

        ----

        ## [1573] Skipping the Frame-Level: Event-Based Piano Transcription With Neural Semi-CRFs

        **Authors**: *Yujia Yan, Frank Cwitkowitz, Zhiyao Duan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac53fab47b547a0d47b77e424cf119ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac53fab47b547a0d47b77e424cf119ba-Abstract.html)

        **Abstract**:

        Piano transcription systems are typically optimized to estimate pitch activity at each frame of audio. They are often followed by carefully designed heuristics and post-processing algorithms to estimate note events from the frame-level predictions. Recent methods have also framed piano transcription as a multi-task learning problem, where the activation of different stages of a note event are estimated independently. These practices are not well aligned with the desired outcome of the task, which is the specification of note intervals as holistic events, rather than the aggregation of disjoint observations. In this work, we propose a novel formulation of piano transcription, which is optimized to directly predict note events. Our method is based on Semi-Markov Conditional Random Fields (semi-CRF), which produce scores for intervals rather than individual frames. When formulating piano transcription in this way, we eliminate the need to rely on disjoint frame-level estimates for different stages of a note event. We conduct experiments on the MAESTRO dataset and demonstrate that the proposed model surpasses the current state-of-the-art for piano transcription. Our results suggest that the semi-CRF output layer, while still quadratic in complexity, is a simple, fast and well-performing solution for event-based prediction, and may lead to similar success in other areas which currently rely on frame-level estimates.

        ----

        ## [1574] Deep Learning on a Data Diet: Finding Important Examples Early in Training

        **Authors**: *Mansheej Paul, Surya Ganguli, Gintare Karolina Dziugaite*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac56f8fe9eea3e4a365f29f0f1957c55-Abstract.html)

        **Abstract**:

        Recent success in deep learning has partially been driven by training increasingly overparametrized networks on ever larger datasets. It is therefore natural to ask: how much of the data is superfluous, which examples are important for generalization, and how do we find them? In this work, we make the striking observation that, in standard vision datasets, simple scores averaged over several weight initializations can be used to identify important examples very early in training. We propose two such scores—the Gradient Normed (GraNd) and the Error L2-Norm (EL2N) scores—and demonstrate their efficacy on a range of architectures and datasets by pruning significant fractions of training data without sacrificing test accuracy. In fact, using EL2N scores calculated a few epochs into training, we can prune half of the CIFAR10 training set while slightly improving test accuracy. Furthermore, for a given dataset, EL2N scores from one architecture or hyperparameter configuration generalize to other configurations. Compared to recent work that prunes data by discarding examples that are rarely forgotten over the course of training, our scores use only local information early in training. We also use our scores to detect noisy examples and study training dynamics through the lens of important examples—we investigate how the data distribution shapes the loss surface and identify subspaces of the model’s data representation that are relatively stable over training.

        ----

        ## [1575] BNS: Building Network Structures Dynamically for Continual Learning

        **Authors**: *Qi Qin, Wenpeng Hu, Han Peng, Dongyan Zhao, Bing Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac64504cc249b070772848642cffe6ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac64504cc249b070772848642cffe6ff-Abstract.html)

        **Abstract**:

        Continual learning (CL) of a sequence of tasks is often accompanied with the catastrophic forgetting(CF) problem. Existing research has achieved remarkable results in overcoming CF, especially for task continual learning. However, limited work has been done to achieve another important goal of CL,knowledge transfer.In this paper, we propose a technique (called BNS) to do both.  The novelty of BNS is that it dynamically builds a network to learn each new task to overcome CF and to transfer knowledge across tasks at the same time. Experimental results show that when the tasks are different (with little shared knowledge), BNS can already outperform the state-of-the-art baselines. When the tasks are similar and have shared knowledge, BNS outperforms the baselines substantially by a large margin due to its knowledge transfer capability.

        ----

        ## [1576] Auditing Black-Box Prediction Models for Data Minimization Compliance

        **Authors**: *Bashir Rastegarpanah, Krishna P. Gummadi, Mark Crovella*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac6b3cce8c74b2e23688c3e45532e2a7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac6b3cce8c74b2e23688c3e45532e2a7-Abstract.html)

        **Abstract**:

        In this paper, we focus on auditing black-box prediction models for compliance with the GDPR’s data minimization principle. This principle restricts prediction models to use the minimal information that is necessary for performing the task at hand. Given the challenge of the black-box setting, our key idea is to check if each of the prediction model’s input features is individually necessary by assigning it some constant value (i.e., applying a simple imputation) across all prediction instances, and measuring the extent to which the model outcomes would change. We introduce a metric for data minimization that is based on model instability under simple imputations. We extend the applicability of this metric from a finite sample model to a distributional setting by introducing a probabilistic data minimization guarantee, which we derive using a Bayesian approach. Furthermore, we address the auditing problem under a constraint on the number of queries to the prediction system. We formulate the problem of allocating a budget of system queries to feasible simple imputations (for investigating model instability) as a multi-armed bandit framework with probabilistic success metrics. We define two bandit problems for providing a probabilistic data minimization guarantee at a given confidence level: a decision problem given a data minimization level, and a measurement problem given a fixed query budget. We design efficient algorithms for these auditing problems using novel exploration strategies that expand classical bandit strategies. Our experiments with real-world prediction systems show that our auditing algorithms significantly outperform simpler benchmarks in both measurement and decision problems.

        ----

        ## [1577] Dueling Bandits with Team Comparisons

        **Authors**: *Lee Cohen, Ulrike Schmidt-Kraepelin, Yishay Mansour*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac73001b1d44f4925449ce09d9f5d5ca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac73001b1d44f4925449ce09d9f5d5ca-Abstract.html)

        **Abstract**:

        We introduce the dueling teams problem, a new online-learning setting in which the learner observes noisy comparisons of disjoint pairs of $k$-sized teams from a universe of $n$ players. The goal of the learner is to minimize the number of duels required to identify, with high probability, a Condorcet winning team, i.e., a team which wins against any other disjoint team (with probability at least $1/2$).Noisy comparisons are linked to a total order on the teams. We formalize our model by building upon the dueling bandits setting (Yue et al. 2012) and provide several algorithms, both for stochastic and deterministic settings. For the stochastic setting, we provide a reduction to the classical dueling bandits setting, yielding an algorithm that identifies a Condorcet winning team within $\mathcal{O}((n + k \log (k)) \frac{\max(\log\log n, \log k)}{\Delta^2})$ duels, where $\Delta$ is a gap parameter. For deterministic feedback, we additionally present a gap-independent algorithm that identifies a Condorcet winning team within $\mathcal{O}(nk\log(k)+k^5)$ duels.

        ----

        ## [1578] Meta Internal Learning

        **Authors**: *Raphael Bensadoun, Shir Gur, Tomer Galanti, Lior Wolf*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac796a52db3f16bbdb6557d3d89d1c5a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac796a52db3f16bbdb6557d3d89d1c5a-Abstract.html)

        **Abstract**:

        Internal learning for single-image generation is a framework, where a generator is trained to produce novel images based on a single image. Since these models are trained on a single image, they are limited in their scale and application. To overcome these issues, we propose a meta-learning approach that enables training over a collection of images, in order to model the internal statistics of the sample image more effectively.In the presented meta-learning approach, a single-image GAN model is generated given an input image, via a convolutional feedforward hypernetwork $f$. This network is trained over a dataset of images, allowing for feature sharing among different models, and for interpolation in the space of generative models. The generated single-image model contains a hierarchy of multiple generators and discriminators. It is therefore required to train the meta-learner in an adversarial manner, which requires careful design choices that we justify by a theoretical analysis. Our results show that the models obtained are as suitable as single-image GANs for many common image applications, {significantly reduce the training time per image without loss in performance}, and introduce novel capabilities, such as interpolation and feedforward modeling of novel images.

        ----

        ## [1579] Uniform Convergence of Interpolators: Gaussian Width, Norm Bounds and Benign Overfitting

        **Authors**: *Frederic Koehler, Lijia Zhou, Danica J. Sutherland, Nathan Srebro*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ac9815bef801f58de83804bce86984ad-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ac9815bef801f58de83804bce86984ad-Abstract.html)

        **Abstract**:

        We consider interpolation learning in high-dimensional linear regression with Gaussian data, and prove a generic uniform convergence guarantee on the generalization error of interpolators in an arbitrary hypothesis class in terms of the classâ€™s Gaussian width.  Applying the generic bound to Euclidean norm balls recovers the consistency result of Bartlett et al. (2020) for minimum-norm interpolators, and confirms a prediction of Zhou et al. (2020) for near-minimal-norm interpolators in the special case of Gaussian data.  We demonstrate the generality of the bound by applying it to the simplex, obtaining a novel consistency result for minimum $\ell_1$-norm interpolators (basis pursuit). Our results show how norm-based generalization bounds can explain and be used to analyze benign overfitting, at least in some settings.

        ----

        ## [1580] Adaptive wavelet distillation from neural networks through interpretations

        **Authors**: *Wooseok Ha, Chandan Singh, François Lanusse, Srigokul Upadhyayula, Bin Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/acaa23f71f963e96c8847585e71352d6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/acaa23f71f963e96c8847585e71352d6-Abstract.html)

        **Abstract**:

        Recent deep-learning models have achieved impressive prediction performance, but often sacrifice interpretability and computational efficiency. Interpretability is crucial in many disciplines, such as science and medicine, where models must be carefully vetted or where interpretation is the goal itself. Moreover, interpretable models are concise and often yield computational efficiency. Here, we propose adaptive wavelet distillation (AWD), a method which aims to distill information from a trained neural network into a wavelet transform. Specifically, AWD penalizes feature attributions of a neural network in the wavelet domain to learn an effective multi-resolution wavelet transform. The resulting model is highly predictive, concise, computationally efficient, and has properties (such as a multi-scale structure) which make it easy to interpret. In close collaboration with domain experts, we showcase how AWD addresses challenges in two real-world settings: cosmological parameter inference and molecular-partner prediction. In both cases, AWD yields a scientifically interpretable and concise model which gives predictive performance better than state-of-the-art neural networks. Moreover, AWD identifies predictive features that are scientifically meaningful in the context of respective domains. All code and models are released in a full-fledged package available on Github.

        ----

        ## [1581] Generative Occupancy Fields for 3D Surface-Aware Image Synthesis

        **Authors**: *Xudong Xu, Xingang Pan, Dahua Lin, Bo Dai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/acab0116c354964a558e65bdd07ff047-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/acab0116c354964a558e65bdd07ff047-Abstract.html)

        **Abstract**:

        The advent of generative radiance fields has significantly promoted the development of 3D-aware image synthesis. The cumulative rendering process in radiance fields makes training these generative models much easier since gradients are distributed over the entire volume, but leads to diffused object surfaces. In the meantime, compared to radiance fields occupancy representations could inherently ensure deterministic surfaces. However, if we directly apply occupancy representations to generative models, during training they will only receive sparse gradients located on object surfaces and eventually suffer from the convergence problem. In this paper, we propose Generative Occupancy Fields (GOF), a novel model based on generative radiance fields that can learn compact object surfaces without impeding its training convergence. The key insight of GOF is a dedicated transition from the cumulative rendering in radiance fields to rendering with only the surface points as the learned surface gets more and more accurate. In this way, GOF combines the merits of two representations in a unified framework. In practice, the training-time transition of start from radiance fields and march to occupancy representations is achieved in GOF by gradually shrinking the sampling region in its rendering process from the entire volume to a minimal neighboring region around the surface. Through comprehensive experiments on multiple datasets, we demonstrate that GOF can synthesize high-quality images with 3D consistency and simultaneously learn compact and smooth object surfaces. Our code is available at https://github.com/SheldonTsui/GOF_NeurIPS2021.

        ----

        ## [1582] Relaxed Marginal Consistency for Differentially Private Query Answering

        **Authors**: *Ryan McKenna, Siddhant Pradhan, Daniel Sheldon, Gerome Miklau*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/acb55f9af76808c5fd5522dcdb519fde-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/acb55f9af76808c5fd5522dcdb519fde-Abstract.html)

        **Abstract**:

        Many differentially private algorithms for answering database queries involve astep that reconstructs a discrete data distribution from noisy measurements. Thisprovides consistent query answers and reduces error, but often requires space thatgrows exponentially with dimension. PRIVATE-PGM is a recent approach that usesgraphical models to represent the data distribution, with complexity proportional tothat of exact marginal inference in a graphical model with structure determined bythe co-occurrence of variables in the noisy measurements. PRIVATE-PGM is highlyscalable for sparse measurements, but may fail to run in high dimensions with densemeasurements. We overcome the main scalability limitation of PRIVATE-PGMthrough a principled approach that relaxes consistency constraints in the estimationobjective. Our new approach works with many existing private query answeringalgorithms and improves scalability or accuracy with no privacy cost.

        ----

        ## [1583] Local policy search with Bayesian optimization

        **Authors**: *Sarah Müller, Alexander von Rohr, Sebastian Trimpe*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ad0f7a25211abc3889cb0f420c85e671-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ad0f7a25211abc3889cb0f420c85e671-Abstract.html)

        **Abstract**:

        Reinforcement learning (RL) aims to find an optimal policy by interaction with an environment. Consequently, learning complex behavior requires a vast number of samples, which can be prohibitive in practice. Nevertheless, instead of systematically reasoning and actively choosing informative samples, policy gradients for local search are often obtained from random perturbations. These random samples yield high variance estimates and hence are sub-optimal in terms of sample complexity. Actively selecting informative samples is at the core of Bayesian optimization, which constructs a probabilistic surrogate of the objective from past samples to reason about informative subsequent ones. In this paper, we propose to join both worlds. We develop an algorithm utilizing a probabilistic model of the objective function and its gradient. Based on the model, the algorithm decides where to query a noisy zeroth-order oracle to improve the gradient estimates. The resulting algorithm is a novel type of policy search method, which we compare to existing black-box algorithms. The comparison reveals improved sample complexity and reduced variance in extensive empirical evaluations on synthetic objectives. Further, we highlight the benefits of active sampling on popular RL benchmarks.

        ----

        ## [1584] DominoSearch: Find layer-wise fine-grained N: M sparse schemes from dense neural networks

        **Authors**: *Wei Sun, Aojun Zhou, Sander Stuijk, Rob G. J. Wijnhoven, Andrew Nelson, Hongsheng Li, Henk Corporaal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ad68473a64305626a27c32a5408552d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ad68473a64305626a27c32a5408552d7-Abstract.html)

        **Abstract**:

        Neural pruning is a widely-used compression technique for Deep Neural Networks (DNNs). Recent innovations in Hardware Architectures (e.g. Nvidia Ampere Sparse Tensor Core) and N:M fine-grained Sparse Neural Network algorithms (i.e. every M-weights contains N non-zero values) reveal a promising research line of neural pruning. However, the existing N:M algorithms only address the challenge of how to train N:M sparse neural networks in a uniform fashion (i.e. every layer has the same N:M sparsity) and suffer from a significant accuracy drop for high sparsity (i.e. when sparsity > 80\%). To tackle this problem, we present a novel technique -- \textbf{\textit{DominoSearch}} to find mixed N:M sparsity schemes from pre-trained dense deep neural networks to achieve higher accuracy than the uniform-sparsity scheme with equivalent complexity constraints (e.g. model size or FLOPs). For instance, for the same model size with 2.1M parameters (87.5\% sparsity), our layer-wise N:M sparse ResNet18 outperforms its uniform counterpart by 2.1\% top-1 accuracy, on the large-scale ImageNet dataset. For the same computational complexity of 227M FLOPs, our layer-wise sparse ResNet18 outperforms the uniform one by 1.3\% top-1 accuracy. Furthermore, our layer-wise fine-grained N:M sparse ResNet50 achieves 76.7\% top-1 accuracy with 5.0M parameters. {This is competitive to the results achieved by layer-wise unstructured sparsity} that is believed to be the upper-bound of Neural Network pruning with respect to the accuracy-sparsity trade-off. We believe that our work can build a strong baseline for further sparse DNN research and encourage future hardware-algorithm co-design work. Our code and models are publicly available at \url{https://github.com/NM-sparsity/DominoSearch}.

        ----

        ## [1585] Techniques for Symbol Grounding with SATNet

        **Authors**: *Sever Topan, David Rolnick, Xujie Si*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ad7ed5d47b9baceb12045a929e7e2f66-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ad7ed5d47b9baceb12045a929e7e2f66-Abstract.html)

        **Abstract**:

        Many experts argue that the future of artificial intelligence is limited by the fieldâ€™s ability to integrate symbolic logical reasoning into deep learning architectures. The recently proposed differentiable MAXSAT solver, SATNet, was a breakthrough in its capacity to integrate with a traditional neural network and solve visual reasoning problems. For instance, it can learn the rules of Sudoku purely from image examples. Despite its success, SATNet was shown to succumb to a key challenge in neurosymbolic systems known as the Symbol Grounding Problem: the inability to map visual inputs to symbolic variables without explicit supervision ("label leakage"). In this work, we present a self-supervised pre-training pipeline that enables SATNet to overcome this limitation, thus broadening the class of problems that SATNet architectures can solve to include datasets where no intermediary labels are available at all. We demonstrate that our method allows SATNet to attain full accuracy even with a harder problem setup that prevents any label leakage. We additionally introduce a proofreading method that further improves the performance of SATNet architectures, beating the state-of-the-art on Visual Sudoku.

        ----

        ## [1586] Object DGCNN: 3D Object Detection using Dynamic Graphs

        **Authors**: *Yue Wang, Justin M. Solomon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ade1d98c5ab2997e867b1151a5c5028d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ade1d98c5ab2997e867b1151a5c5028d-Abstract.html)

        **Abstract**:

        3D object detection often involves complicated training and testing pipelines, which require substantial domain knowledge about individual datasets. Inspired by recent non-maximum suppression-free 2D object detection models, we propose a 3D object detection architecture on point clouds. Our method models 3D object detection as message passing on a dynamic graph, generalizing the DGCNN framework to predict a set of objects. In our construction, we remove the necessity of post-processing via object confidence aggregation or non-maximum suppression. To facilitate object detection from sparse point clouds, we also propose a set-to-set distillation approach customized to 3D detection. This approach aligns the outputs of the teacher model and the student model in a permutation-invariant fashion, significantly simplifying knowledge distillation for the 3D detection task. Our method achieves state-of-the-art performance on autonomous driving benchmarks. We also provide abundant analysis of the detection model and distillation framework.

        ----

        ## [1587] Safe Policy Optimization with Local Generalized Linear Function Approximations

        **Authors**: *Akifumi Wachi, Yunyue Wei, Yanan Sui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/adf7e293599134777339fdc40ddfa818-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/adf7e293599134777339fdc40ddfa818-Abstract.html)

        **Abstract**:

        Safe exploration is a key to applying reinforcement learning (RL) in safety-critical systems. Existing safe exploration methods guaranteed safety under the assumption of regularity, and it has been difficult to apply them to large-scale real problems. We propose a novel algorithm, SPO-LF, that optimizes an agent's policy while learning the relation between a locally available feature obtained by sensors and environmental reward/safety using generalized linear function approximations. We provide theoretical guarantees on its safety and optimality. We experimentally show that our algorithm is 1) more efficient in terms of sample complexity and computational cost and 2) more applicable to large-scale problems than previous safe RL methods with theoretical guarantees, and 3) comparably sample-efficient and safer compared with existing advanced deep RL methods with safety constraints.

        ----

        ## [1588] Symplectic Adjoint Method for Exact Gradient of Neural ODE with Minimal Memory

        **Authors**: *Takashi Matsubara, Yuto Miyatake, Takaharu Yaguchi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/adf8d7f8c53c8688e63a02bfb3055497-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/adf8d7f8c53c8688e63a02bfb3055497-Abstract.html)

        **Abstract**:

        A neural network model of a differential equation, namely neural ODE, has enabled the learning of continuous-time dynamical systems and probabilistic distributions with high accuracy. The neural ODE uses the same network repeatedly during a numerical integration. The memory consumption of the backpropagation algorithm is proportional to the number of uses times the network size. This is true even if a checkpointing scheme divides the computation graph into sub-graphs. Otherwise, the adjoint method obtains a gradient by a numerical integration backward in time. Although this method consumes memory only for a single network use, it requires high computational cost to suppress numerical errors. This study proposes the symplectic adjoint method, which is an adjoint method solved by a symplectic integrator. The symplectic adjoint method obtains the exact gradient (up to rounding error) with memory proportional to the number of uses plus the network size. The experimental results demonstrate that the symplectic adjoint method consumes much less memory than the naive backpropagation algorithm and checkpointing schemes, performs faster than the adjoint method, and is more robust to rounding errors.

        ----

        ## [1589] Exponential Separation between Two Learning Models and Adversarial Robustness

        **Authors**: *Grzegorz Gluch, Rüdiger L. Urbanke*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae06fbdc519bddaa88aa1b24bace4500-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae06fbdc519bddaa88aa1b24bace4500-Abstract.html)

        **Abstract**:

        We prove an exponential separation for the sample/query complexity between the standard PAC-learning model and a version of the Equivalence-Query-learning model. In the PAC model all samples are provided at the beginning of the learning process. In the Equivalence-Query model the samples are acquired through an interaction between a teacher and a learner, where the teacher provides counterexamples to hypotheses given by the learner. It is intuitive that in an interactive setting fewer samples are needed. We make this formal and prove that in order to achieve an error $\epsilon$ {\em exponentially} (in $\epsilon$) fewer samples suffice than what the PAC bound requires. It was shown experimentally by Stutz, Hein, and Schiele that adversarial training with on-manifold adversarial examples aids generalization (compared to standard training). If we think of the adversarial examples as counterexamples to the current hypothesis then our result can be thought of as a theoretical confirmation of those findings.  We also discuss how our result relates to adversarial robustness. In the standard adversarial model one restricts the adversary by introducing a norm constraint. An alternative was pioneered by Goldwasser et. al. Rather than restricting the adversary the learner is enhanced. We pursue a third path. We require the adversary to return samples according to the Equivalance-Query model and show that this leads to robustness. Even though our model has its limitations it provides a fresh point of view on adversarial robustness.

        ----

        ## [1590] The balancing principle for parameter choice in distance-regularized domain adaptation

        **Authors**: *Werner Zellinger, Natalia Shepeleva, Marius-Constantin Dinu, Hamid Eghbal-zadeh, Hoan Duc Nguyen, Bernhard Nessler, Sergei V. Pereverzyev, Bernhard Alois Moser*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae0909a324fb2530e205e52d40266418-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae0909a324fb2530e205e52d40266418-Abstract.html)

        **Abstract**:

        We address the unsolved algorithm design problem of choosing a justified regularization parameter in unsupervised domain adaptation. This problem is intriguing as no labels are available in the target domain. Our approach starts with the observation that the widely-used method of minimizing the source error, penalized by a distance measure between source and target feature representations, shares characteristics with regularized ill-posed inverse problems. Regularization parameters in inverse problems are optimally chosen by the fundamental principle of balancing approximation and sampling errors. We use this principle to balance learning errors and domain distance in a target error bound. As a result, we obtain a theoretically justified rule for the choice of the regularization parameter. In contrast to the state of the art, our approach allows source and target distributions with disjoint supports. An empirical comparative study on benchmark datasets underpins the performance of our approach.

        ----

        ## [1591] Gaussian Kernel Mixture Network for Single Image Defocus Deblurring

        **Authors**: *Yuhui Quan, Zicong Wu, Hui Ji*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae1eaa32d10b6c886981755d579fb4d8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae1eaa32d10b6c886981755d579fb4d8-Abstract.html)

        **Abstract**:

        Defocus blur is one kind of blur effects often seen in images, which is challenging to remove due to its spatially variant amount. This paper presents  an end-to-end deep learning approach for removing defocus blur from a single image, so as to have an all-in-focus image for consequent vision tasks. First, a  pixel-wise Gaussian kernel mixture (GKM) model is  proposed for representing spatially variant defocus blur kernels in an efficient linear parametric form, with higher accuracy than existing models. Then, a deep neural network called GKMNet is developed by unrolling a fixed-point iteration of the GKM-based deblurring.  The GKMNet is built on a lightweight scale-recurrent architecture, with a scale-recurrent attention module for estimating the mixing coefficients in GKM for defocus deblurring. Extensive experiments show that the GKMNet not only noticeably outperforms existing defocus deblurring methods, but also has its advantages in terms of model complexity and computational efficiency.

        ----

        ## [1592] Cockpit: A Practical Debugging Tool for the Training of Deep Neural Networks

        **Authors**: *Frank Schneider, Felix Dangel, Philipp Hennig*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae3539867aaeec609a4260c6feb725f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae3539867aaeec609a4260c6feb725f4-Abstract.html)

        **Abstract**:

        When engineers train deep learning models, they are very much "flying blind". Commonly used methods for real-time training diagnostics, such as monitoring the train/test loss, are limited. Assessing a network's training process solely through these performance indicators is akin to debugging software without access to internal states through a debugger. To address this, we present Cockpit, a collection of instruments that enable a closer look into the inner workings of a learning machine, and a more informative and meaningful status report for practitioners. It facilitates the identification of learning phases and failure modes, like ill-chosen hyperparameters. These instruments leverage novel higher-order information about the gradient distribution and curvature, which has only recently become efficiently accessible. We believe that such a debugging tool, which we open-source for PyTorch, is a valuable help in troubleshooting the training process. By revealing new insights, it also more generally contributes to explainability and interpretability of deep nets.

        ----

        ## [1593] MEST: Accurate and Fast Memory-Economic Sparse Training Framework on the Edge

        **Authors**: *Geng Yuan, Xiaolong Ma, Wei Niu, Zhengang Li, Zhenglun Kong, Ning Liu, Yifan Gong, Zheng Zhan, Chaoyang He, Qing Jin, Siyue Wang, Minghai Qin, Bin Ren, Yanzhi Wang, Sijia Liu, Xue Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae3f4c649fb55c2ee3ef4d1abdb79ce5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae3f4c649fb55c2ee3ef4d1abdb79ce5-Abstract.html)

        **Abstract**:

        Recently, a new trend of exploring sparsity for accelerating neural network training has emerged, embracing the paradigm of training on the edge. This paper proposes a novel Memory-Economic Sparse Training (MEST) framework targeting for accurate and fast execution on edge devices. The proposed MEST framework consists of enhancements by Elastic Mutation (EM) and Soft Memory Bound (&S) that ensure superior accuracy at high sparsity ratios. Different from the existing works for sparse training, this current work reveals the importance of sparsity schemes on the performance of sparse training in terms of accuracy as well as training speed on real edge devices. On top of that, the paper proposes to employ data efficiency for further acceleration of sparse training. Our results suggest that unforgettable examples can be identified in-situ even during the dynamic exploration of sparsity masks in the sparse training process, and therefore can be removed for further training speedup on edge devices. Comparing with state-of-the-art (SOTA) works on accuracy, our MEST increases Top-1 accuracy significantly on ImageNet when using the same unstructured sparsity scheme. Systematical evaluation on accuracy, training speed, and memory footprint are conducted, where the proposed MEST framework consistently outperforms representative SOTA works. A reviewer strongly against our work based on his false assumptions and misunderstandings. On top of the previous submission, we employ data efficiency for further acceleration of sparse training. And we explore the impact of model sparsity, sparsity schemes, and sparse training algorithms on the number of removable training examples. Our codes are publicly available at: https://github.com/boone891214/MEST.

        ----

        ## [1594] Precise characterization of the prior predictive distribution of deep ReLU networks

        **Authors**: *Lorenzo Noci, Gregor Bachmann, Kevin Roth, Sebastian Nowozin, Thomas Hofmann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae4503ec3da32f5e9033604744ec45ae-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae4503ec3da32f5e9033604744ec45ae-Abstract.html)

        **Abstract**:

        Recent works on Bayesian neural networks (BNNs) have highlighted the need to better understand the implications of using Gaussian priors in combination with the compositional structure of the network architecture. Similar in spirit to the kind of analysis that has been developed to devise better initialization schemes for neural networks (cf. He- or Xavier initialization), we derive a precise characterization of the prior predictive distribution of finite-width ReLU networks with Gaussian weights.While theoretical results have been obtained for their heavy-tailedness,the full characterization of the prior predictive distribution (i.e. its density, CDF and moments), remained unknown prior to this work. Our analysis, based on the Meijer-G function, allows us to quantify the influence of architectural choices such as the width or depth of the network on the resulting shape of the prior predictive distribution. We also formally connect our results to previous work in the infinite width setting, demonstrating that the moments of the distribution converge to those of a normal log-normal mixture in the infinite depth limit. Finally, our results provide valuable guidance on prior design: for instance, controlling the predictive variance with depth- and width-informed priors on the weights of the network.

        ----

        ## [1595] RED : Looking for Redundancies for Data-FreeStructured Compression of Deep Neural Networks

        **Authors**: *Edouard Yvinec, Arnaud Dapogny, Matthieu Cord, Kevin Bailly*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae5e3ce40e0404a45ecacaaf05e5f735-Abstract.html)

        **Abstract**:

        Deep Neural Networks (DNNs) are ubiquitous in today's computer vision landscape, despite involving considerable computational costs. The mainstream approaches for runtime acceleration consist in pruning connections (unstructured pruning) or, better, filters (structured pruning), both often requiring data to retrain the model.  In this paper, we present RED, a data-free, unified approach to tackle structured pruning. First, we propose a novel adaptive hashing of the scalar DNN weight distribution densities to increase the number of identical neurons represented by their weight vectors. Second, we prune the network by merging redundant neurons based on their relative similarities, as defined by their distance. Third, we propose a novel uneven depthwise separation technique to further prune convolutional layers. We demonstrate through a large variety of benchmarks that RED largely outperforms other data-free pruning methods, often reaching performance similar to unconstrained, data-driven methods.

        ----

        ## [1596] TestRank: Bringing Order into Unlabeled Test Instances for Deep Learning Tasks

        **Authors**: *Yu Li, Min Li, Qiuxia Lai, Yannan Liu, Qiang Xu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae78510109d46b0a6eef9820a4ca95d6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae78510109d46b0a6eef9820a4ca95d6-Abstract.html)

        **Abstract**:

        Deep learning (DL) systems are notoriously difficult to test and debug due to the lack of correctness proof and the huge test input space to cover. Given the ubiquitous unlabeled test data and high labeling cost, in this paper, we propose a novel test prioritization technique, namely TestRank, which aims at revealing more model failures with less labeling effort. TestRank brings order into the unlabeled test data according to their likelihood of being a failure, i.e., their failure-revealing capabilities. Different from existing solutions, TestRank leverages both intrinsic and contextual attributes of the unlabeled test data when prioritizing them. To be specific, we first build a similarity graph on both unlabeled test samples and labeled samples (e.g., training or previously labeled test samples). Then, we conduct graph-based semi-supervised learning to extract contextual features from the correctness of similar labeled samples. For a particular test instance, the contextual features extracted with the graph neural network and the intrinsic features obtained with the DL model itself are combined to predict its failure-revealing capability. Finally, TestRank prioritizes unlabeled test inputs in descending order of the above probability value. We evaluate TestRank on three popular image classification datasets, and results show that TestRank significantly outperforms existing test prioritization techniques.

        ----

        ## [1597] Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods

        **Authors**: *Derek Lim, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, Ser-Nam Lim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ae816a80e4c1c56caa2eb4e1819cbb2f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ae816a80e4c1c56caa2eb4e1819cbb2f-Abstract.html)

        **Abstract**:

        Many widely used datasets for graph machine learning tasks have generally been homophilous, where nodes with similar labels connect to each other. Recently, new Graph Neural Networks (GNNs) have been developed that move beyond the homophily regime; however, their evaluation has often been conducted on small graphs with limited application domains. We collect and introduce diverse non-homophilous datasets from a variety of application areas that have up to 384x more nodes and 1398x more edges than prior datasets. We further show that existing scalable graph learning and graph minibatching techniques lead to performance degradation on these non-homophilous datasets, thus highlighting the need for further work on scalable non-homophilous methods. To address these concerns, we introduce LINKX --- a strong simple method that admits straightforward minibatch training and inference. Extensive experimental results with representative simple methods and GNNs across our proposed datasets show that LINKX achieves state-of-the-art performance for learning on non-homophilous graphs. Our codes and data are available at https://github.com/CUAI/Non-Homophily-Large-Scale.

        ----

        ## [1598] Reinforcement Learning based Disease Progression Model for Alzheimer's Disease

        **Authors**: *Krishnakant V. Saboo, Anirudh Choudhary, Yurui Cao, Gregory A. Worrell, David T. Jones, Ravishankar K. Iyer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/af1c25e88a9e818f809f6b5d18ca02e2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/af1c25e88a9e818f809f6b5d18ca02e2-Abstract.html)

        **Abstract**:

        We model Alzheimer’s disease (AD) progression by combining differential equations (DEs) and reinforcement learning (RL) with domain knowledge. DEs  provide relationships between some, but not all, factors relevant to AD. We assume that the missing relationships must satisfy general criteria about the working of the brain, for e.g., maximizing cognition while minimizing the cost of supporting cognition. This allows us to extract the missing relationships by using RL to optimize an objective (reward) function that captures the above criteria. We use our model consisting of DEs (as a simulator) and the trained RL agent to predict individualized 10-year AD progression using baseline (year 0) features on synthetic and real data. The model was comparable or better at predicting 10-year cognition trajectories than state-of-the-art learning-based models. Our interpretable model demonstrated, and provided insights into, "recovery/compensatory" processes that mitigate the effect of AD, even though those processes were not explicitly encoded in the model. Our framework combines DEs with RL for modelling AD progression and has broad applicability for understanding other neurological disorders.

        ----

        ## [1599] Catch-A-Waveform: Learning to Generate Audio from a Single Short Example

        **Authors**: *Gal Greshler, Tamar Rott Shaham, Tomer Michaeli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)

        **Abstract**:

        Models for audio generation are typically trained on hours of recordings. Here, we illustrate that capturing the essence of an audio source is typically possible from as little as a few tens of seconds from a single training signal. Specifically, we present a GAN-based generative model that can be trained on one short audio signal from any domain (e.g. speech, music, etc.) and does not require pre-training or any other form of external supervision. Once trained, our model can generate random  samples of arbitrary duration that maintain semantic similarity to the training waveform, yet exhibit new compositions of its audio primitives. This enables a long line of interesting applications, including generating new jazz improvisations or new a-cappella rap variants based on a single short example, producing coherent modifications to famous songs (e.g. adding a new verse to a Beatles song based solely on the original recording), filling-in of missing parts (inpainting), extending the bandwidth of a speech signal (super-resolution), and enhancing old recordings without access to any clean training example. We show that in all cases, no more than 20 seconds of training audio commonly suffice for our model to achieve state-of-the-art results. This is despite its complete lack of prior knowledge about the nature of audio signals in general.

        ----

        

[Go to the previous page](NIPS-2021-list07.md)

[Go to the next page](NIPS-2021-list09.md)

[Go to the catalog section](README.md)