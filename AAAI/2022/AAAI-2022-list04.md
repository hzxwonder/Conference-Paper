## [600] DeepVisualInsight: Time-Travelling Visualization for Spatio-Temporal Causality of Deep Classification Training

**Authors**: *Xianglin Yang, Yun Lin, Ruofan Liu, Zhenfeng He, Chao Wang, Jin Song Dong, Hong Mei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20473](https://doi.org/10.1609/aaai.v36i5.20473)

**Abstract**:

Understanding how the predictions of deep learning models are formed during the training process is crucial to improve model performance and fix model defects, especially when we need to investigate nontrivial training strategies such as active learning, and track the root cause of unexpected training results such as performance degeneration.

In this work, we propose a time-travelling visual solution DeepVisualInsight (DVI), aiming to manifest the spatio-temporal causality while training a deep learning image classifier. The spatio-temporal causality demonstrates how the gradient-descent algorithm and various training data sampling techniques can influence and reshape the layout of learnt input representation and the classification boundaries in consecutive epochs. Such causality allows us to observe and analyze the whole learning process in the visible low dimensional space. Technically, we propose four spatial and temporal properties and design our visualization solution to satisfy them. These properties preserve the most important information when projecting and inverse-projecting input samples between the visible low-dimensional and the invisible high-dimensional space, for causal analyses. Our extensive experiments show that, comparing to baseline approaches, we achieve the best visualization performance regarding the spatial/temporal properties and visualization efficiency. Moreover, our case study shows that our visual solution can well reflect the characteristics of various training scenarios, showing good potential of DVI as a debugging tool for analyzing deep learning training processes.

----

## [601] When Facial Expression Recognition Meets Few-Shot Learning: A Joint and Alternate Learning Framework

**Authors**: *Xinyi Zou, Yan Yan, Jing-Hao Xue, Si Chen, Hanzi Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20474](https://doi.org/10.1609/aaai.v36i5.20474)

**Abstract**:

Human emotions involve basic and compound facial expressions. However, current research on facial expression recognition (FER) mainly focuses on basic expressions, and thus fails to address the diversity of human emotions in practical scenarios. Meanwhile, existing work on compound FER relies heavily on abundant labeled compound expression training data, which are often laboriously collected under the professional instruction of psychology. In this paper, we study compound FER in the cross-domain few-shot learning setting, where only a few images of novel classes from the target domain are required as a reference. In particular, we aim to identify unseen compound expressions with the model trained on easily accessible basic expression datasets. To alleviate the problem of limited base classes in our FER task, we propose a novel Emotion Guided Similarity Network (EGS-Net), consisting of an emotion branch and a similarity branch, based on a two-stage learning framework. Specifically, in the first stage, the similarity branch is jointly trained with the emotion branch in a multi-task fashion. With the regularization of the emotion branch, we prevent the similarity branch from overfitting to sampled base classes that are highly overlapped across different episodes. In the second stage, the emotion branch and the similarity branch play a “two-student game” to alternately learn from each other, thereby further improving the inference ability of the similarity branch on unseen compound expressions. Experimental results on both in-the-lab and in-the-wild compound expression datasets demonstrate the superiority of our proposed method against several state-of-the-art methods.

----

## [602] Discovering State and Action Abstractions for Generalized Task and Motion Planning

**Authors**: *Aidan Curtis, Tom Silver, Joshua B. Tenenbaum, Tomás Lozano-Pérez, Leslie Pack Kaelbling*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20475](https://doi.org/10.1609/aaai.v36i5.20475)

**Abstract**:

Generalized planning accelerates classical planning by finding an algorithm-like policy that solves multiple instances of a task. A generalized plan can be learned from a few training examples and applied to an entire domain of problems. Generalized planning approaches perform well in discrete AI planning problems that involve large numbers of objects and extended action sequences to achieve the goal. In this paper, we propose an algorithm for learning features, abstractions, and generalized plans for continuous robotic task and motion planning (TAMP) and examine the unique difficulties that arise when forced to consider geometric and physical constraints as a part of the generalized plan. Additionally, we show that these simple generalized plans learned from only a handful of examples can be used to improve the search efficiency of TAMP solvers.

----

## [603] Recurrent Neural Network Controllers Synthesis with Stability Guarantees for Partially Observed Systems

**Authors**: *Fangda Gu, He Yin, Laurent El Ghaoui, Murat Arcak, Peter J. Seiler, Ming Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20476](https://doi.org/10.1609/aaai.v36i5.20476)

**Abstract**:

Neural network controllers have become popular in control tasks thanks to their flexibility and expressivity. Stability is a crucial property for safety-critical dynamical systems, while stabilization of partially observed systems, in many cases, requires controllers to retain and process long-term memories of the past. We consider the important class of recurrent neural networks (RNN) as dynamic controllers for nonlinear uncertain partially-observed systems, and derive convex stability conditions based on integral quadratic constraints, S-lemma and sequential convexification. To ensure stability during the learning and control process, we propose a projected policy gradient method that iteratively enforces the stability conditions in the reparametrized space taking advantage of mild additional information on system dynamics. Numerical experiments show that our method learns stabilizing controllers with fewer samples and achieves higher final performance compared with policy gradient.

----

## [604] Random Mapping Method for Large-Scale Terrain Modeling

**Authors**: *Xu Liu, Decai Li, Yuqing He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20477](https://doi.org/10.1609/aaai.v36i5.20477)

**Abstract**:

The vast amount of data captured by robots in large-scale environments brings the computing and storage bottlenecks to the typical methods of modeling the spaces the robots travel in. In order to efficiently construct a compact terrain model from uncertain, incomplete point cloud data of large-scale environments, in this paper, we first propose a novel feature mapping method, named random mapping, based on the fast random construction of base functions, which can efficiently project the messy points in the low-dimensional space into the high-dimensional space where the points are approximately linearly distributed. Then, in this mapped space, we propose to learn a continuous linear regression model to represent the terrain. We show that this method can model the environments in much less computation time, memory consumption, and access time, with high accuracy. Furthermore, the models possess the generalization capabilities comparable to the performances on the training set, and its inference accuracy gradually increases as the random mapping dimension increases. To better solve the large-scale environmental modeling problem, we adopt the idea of parallel computing to train the models. This strategy greatly reduces the wall-clock time of calculation without losing much accuracy. Experiments show the effectiveness of the random mapping method and the effects of some important parameters on its performance. Moreover, we evaluate the proposed terrain modeling method based on the random mapping method and compare its performances with popular typical methods and state-of-art methods.

----

## [605] Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning

**Authors**: *Yecheng Jason Ma, Andrew Shen, Osbert Bastani, Dinesh Jayaraman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20478](https://doi.org/10.1609/aaai.v36i5.20478)

**Abstract**:

Reinforcement Learning (RL) agents in the real world must satisfy safety constraints in addition to maximizing a reward objective. Model-based RL algorithms hold promise for reducing unsafe real-world actions: they may synthesize policies that obey all constraints using simulated samples from a learned model. However, imperfect models can result in real-world constraint violations even for actions that are predicted to satisfy all constraints. We propose Conservative and Adaptive Penalty (CAP), a model-based safe RL framework that accounts for potential modeling errors by capturing model uncertainty and adaptively exploiting it to balance the reward and the cost objectives. First, CAP inflates predicted costs using an uncertainty-based penalty. Theoretically, we show that policies that satisfy this conservative cost constraint are guaranteed to also be feasible in the true environment. We further show that 
 this guarantees the safety of all intermediate solutions during RL training. Further, CAP adaptively tunes this penalty during training using true cost feedback from the environment. We evaluate this conservative and adaptive penalty-based approach for model-based safe RL extensively on state and image-based environments. Our results demonstrate substantial gains in sample-efficiency while incurring fewer violations than prior safe RL algorithms. Code is available at: https://github.com/Redrew/CAP

----

## [606] CTIN: Robust Contextual Transformer Network for Inertial Navigation

**Authors**: *Bingbing Rao, Ehsan Kazemi, Yifan Ding, Devu M. Shila, Frank M. Tucker, Liqiang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20479](https://doi.org/10.1609/aaai.v36i5.20479)

**Abstract**:

Recently, data-driven inertial navigation approaches have demonstrated their capability of using well-trained neural networks to obtain accurate position estimates from inertial measurement units (IMUs) measurements. In this paper, we propose a novel robust Contextual Transformer-based network for Inertial Navigation (CTIN) to accurately predict velocity and trajectory. To this end, we first design a ResNet-based encoder enhanced by local and global multi-head self-attention to capture spatial contextual information from IMU measurements. Then we fuse these spatial representations with temporal knowledge by leveraging multi-head attention in the Transformer decoder. Finally, multi-task learning with uncertainty reduction is leveraged to improve learning efficiency and prediction accuracy of velocity and trajectory. Through extensive experiments over a wide range of inertial datasets (e.g., RIDI, OxIOD, RoNIN, IDOL, and our own), CTIN is very robust and outperforms state-of-the-art models.

----

## [607] Monocular Camera-Based Point-Goal Navigation by Learning Depth Channel and Cross-Modality Pyramid Fusion

**Authors**: *Tianqi Tang, Heming Du, Xin Yu, Yi Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20480](https://doi.org/10.1609/aaai.v36i5.20480)

**Abstract**:

For a monocular camera-based navigation system, if we could effectively explore scene geometric cues from RGB images, the geometry information will significantly facilitate the efficiency of the navigation system. Motivated by this, we propose a highly efficient point-goal navigation framework, dubbed Geo-Nav. In a nutshell, our Geo-Nav consists of two parts: a visual perception part and a navigation part. In the visual perception part, we firstly propose a Self-supervised Depth Estimation network (SDE) specially tailored for the monocular camera-based navigation agent. Our SDE learns a mapping from an RGB input image to its corresponding depth image by exploring scene geometric constraints in a self-consistency manner. Then, in order to achieve a representative visual representation from the RGB inputs and learned depth images, we propose a Cross-modality Pyramid Fusion module (CPF). Concretely, our CPF computes a patch-wise cross-modality correlation between different modal features and exploits the correlation to fuse and enhance features at each scale. Thanks to the patch-wise nature of our CPF, we can fuse feature maps at high resolution, allowing our visual network to perceive more image details. In the navigation part, our extracted visual representations are fed to a navigation policy network to learn how to map the visual representations to agent actions effectively. Extensive experiments on a widely-used multiple-room environment Gibson demonstrate that Geo-Nav outperforms the state-of-the-art in terms of efficiency and effectiveness.

----

## [608] Robust Adversarial Reinforcement Learning with Dissipation Inequation Constraint

**Authors**: *Peng Zhai, Jie Luo, Zhiyan Dong, Lihua Zhang, Shunli Wang, Dingkang Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20481](https://doi.org/10.1609/aaai.v36i5.20481)

**Abstract**:

Robust adversarial reinforcement learning is an effective method to train agents to manage uncertain disturbance and modeling errors in real environments. However, for systems that are sensitive to disturbances or those that are difficult to stabilize, it is easier to learn a powerful adversary than establish a stable control policy. An improper strong adversary can destabilize the system, introduce biases in the sampling process, make the learning process unstable, and even reduce the robustness of the policy. In this study, we consider the problem of ensuring system stability during training in the adversarial reinforcement learning architecture. The dissipative principle of robust H-inﬁnity control is extended to the Markov Decision Process, and robust stability constraints are obtained based on L2 gain performance in the reinforcement learning system. Thus, we propose a dissipation-inequation-constraint-based adversarial reinforcement learning architecture. This architecture ensures the stability of the system during training by imposing constraints on the normal and adversarial agents. Theoretically, this architecture can be applied to a large family of deep reinforcement learning algorithms. Results of experiments in MuJoCo and GymFc environments show that our architecture effectively improves the robustness of the controller against environmental changes and adapts to more powerful adversaries. Results of the flight experiments on a real quadcopter indicate that our method can directly deploy the policy trained in the simulation environment to the real environment, and our controller outperforms the PID controller based on hardware-in-the-loop. Both our theoretical and empirical results provide new and critical outlooks on the adversarial reinforcement learning architecture from a rigorous robust control perspective.

----

## [609] Sim2Real Object-Centric Keypoint Detection and Description

**Authors**: *Chengliang Zhong, Chao Yang, Fuchun Sun, Jinshan Qi, Xiaodong Mu, Huaping Liu, Wenbing Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20482](https://doi.org/10.1609/aaai.v36i5.20482)

**Abstract**:

Keypoint detection and description play a central role in computer vision. Most existing methods are in the form of scene-level prediction, without returning the object classes of different keypoints. In this paper, we propose the object-centric formulation, which, beyond the conventional setting, requires further identifying which object each interest point belongs to. With such fine-grained information, our framework enables more downstream potentials, such as object-level matching and pose estimation in a clustered environment. To get around the difficulty of label collection in the real world, we develop a sim2real contrastive learning mechanism that can generalize the model trained in simulation to real-world applications. The novelties of our training method are three-fold: (i) we integrate the uncertainty into the learning framework to improve feature description of hard cases, e.g., less-textured or symmetric patches; (ii) we decouple the object descriptor into two independent branches, intra-object salience and inter-object distinctness, resulting in a better pixel-wise description; (iii) we enforce cross-view semantic consistency for enhanced robustness in representation learning. Comprehensive experiments on image matching and 6D pose estimation verify the encouraging generalization ability of our method. Particularly for 6D pose estimation, our method significantly outperforms typical unsupervised/sim2real methods, achieving a closer gap with the fully supervised counterpart.

----

## [610] Incomplete Argumentation Frameworks: Properties and Complexity

**Authors**: *Gianvincenzo Alfano, Sergio Greco, Francesco Parisi, Irina Trubitsyna*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20483](https://doi.org/10.1609/aaai.v36i5.20483)

**Abstract**:

Dung’s Argumentation Framework (AF) has been extended in several directions, including the possibility of representing unquantified uncertainty about the existence of arguments and attacks.
The framework resulting from such an extension is called incomplete AF (iAF). In this paper, we first introduce three new satisfaction problems named totality, determinism and functionality, and investigate their computational complexity for both AF and iAF under several semantics.
We also investigate the complexity of credulous and skeptical acceptance in iAF under semi-stable semantics—a problem left open in the literature.
We then show that any iAF can be rewritten into an equivalent one where either only (unattacked) arguments or only attacks are uncertain. 
Finally, we relate iAF to probabilistic argumentation framework, where uncertainty is quantified.

----

## [611] Trading Complexity for Sparsity in Random Forest Explanations

**Authors**: *Gilles Audemard, Steve Bellart, Louenas Bounia, Frédéric Koriche, Jean-Marie Lagniez, Pierre Marquis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20484](https://doi.org/10.1609/aaai.v36i5.20484)

**Abstract**:

Random forests have long been considered as powerful model ensembles in machine learning. By training multiple decision trees, whose diversity is fostered through data and feature subsampling, the resulting random forest can lead to more stable and reliable predictions than a single decision tree. This however comes at the cost of decreased interpretability: while decision trees are often easily interpretable, the predictions made by random forests are much more difficult to understand, as they involve a majority vote over multiple decision trees. In this paper, we examine different types of reasons that explain "why" an input instance is classified as positive or negative by a Boolean random forest. Notably, as an alternative to prime-implicant explanations taking the form of subset-minimal implicants of the random forest, we introduce majoritary reasons which are subset-minimal implicants of a strict majority of decision trees. For these abductive explanations, the tractability of the generation problem (finding one reason) and the optimization problem (finding one minimum-sized reason) are investigated. Unlike prime-implicant explanations, majoritary reasons may contain redundant features. However, in practice, prime-implicant explanations - for which the identification problem is DP-complete - are slightly larger than majoritary reasons that can be generated using a simple linear-time greedy algorithm. They are also significantly larger than minimum-sized majoritary reasons which can be approached using an anytime Partial MaxSAT algorithm.

----

## [612] From Actions to Programs as Abstract Actual Causes

**Authors**: *Bita Banihashemi, Shakil M. Khan, Mikhail Soutchanski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20485](https://doi.org/10.1609/aaai.v36i5.20485)

**Abstract**:

Causality plays a central role in reasoning about observations. In many cases, it might be useful to define the conditions under which a non-deterministic program can be called an actual cause of an effect in a setting where a sequence of programs are executed one after another. There can be two perspectives, one where at least one execution of the program leads to the effect, and another where all executions do so. The former captures a ''weak'' notion of causation and is more general than the latter stronger notion. In this paper, we give a definition of weak potential causes. Our analysis is performed within the situation calculus basic action theories and we consider programs formulated in the logic programming language ConGolog. Within this setting, we show how one can utilize a recently developed abstraction framework to relate causes at various levels of abstraction, which facilitates reasoning about programs as causes.

----

## [613] Equivalence in Argumentation Frameworks with a Claim-Centric View - Classical Results with Novel Ingredients

**Authors**: *Ringo Baumann, Anna Rapberger, Markus Ulbricht*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20486](https://doi.org/10.1609/aaai.v36i5.20486)

**Abstract**:

A common feature of non-monotonic logics is that the classical notion of equivalence does not preserve the intended meaning in light of additional information. Consequently, the term strong equivalence was coined in the literature and thoroughly investigated. In the present paper, the knowledge representation formalism under consideration are claim-augmented argumentation frameworks (CAFs) which provide a formal basis to analyze conclusion-oriented problems in argumentation by adapting a claim-focused perspective. CAFs extend Dung AFs by associating a claim to each argument representing its conclusion. In this paper, we investigate both ordinary and strong equivalence in CAFs. Thereby, we take the fact into account that one might either be interested in the actual arguments or their claims only. The former point of view naturally yields an extension of strong equivalence for AFs to the claim-based setting while the latter gives rise to a novel equivalence notion which is genuine for CAFs. We tailor, examine and compare these notions and obtain a comprehensive study of this matter for CAFs. We conclude by investigating the computational complexity of naturally arising decision problems.

----

## [614] Finite Entailment of Local Queries in the Z Family of Description Logics

**Authors**: *Bartosz Bednarczyk, Emanuel Kieronski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20487](https://doi.org/10.1609/aaai.v36i5.20487)

**Abstract**:

In the last few years the field of logic-based knowledge representation took a lot of inspiration from database theory. A vital example is that the finite model semantics in description logics (DLs) is reconsidered as a desirable alternative to the classical one and that query entailment has replaced knowledge-base satisfiability (KBSat) checking as the key inference problem. However, despite the considerable effort, the overall picture concerning finite query answering in DLs is still incomplete. In this work we study the complexity of finite entailment of local queries (conjunctive queries and positive boolean combinations thereof) in the Z family of DLs, one of the most powerful KR formalisms, lying on the verge of decidability. Our main result is that the DLs ZOQ and ZOI are finitely controllable, i.e. that their finite and unrestricted entailment problems for local queries coincide. This allows us to reuse recently established upper bounds on querying these logics under the classical semantics. While we will not solve finite query entailment for the third main logic in the Z family, ZIQ, we provide a generic reduction from the finite entail- ment problem to the finite KBSat problem, working for ZIQ and some of its sublogics. Our proofs unify and solidify previously established results on finite satisfiability and finite query entailment for many known DLs.

----

## [615] The Price of Selfishness: Conjunctive Query Entailment for ALCSelf Is 2EXPTIME-Hard

**Authors**: *Bartosz Bednarczyk, Sebastian Rudolph*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20488](https://doi.org/10.1609/aaai.v36i5.20488)

**Abstract**:

In logic-based knowledge representation, query answering has essentially replaced mere satisfiability checking as the inferencing problem of primary interest. For knowledge bases in the basic description logic ALC, the computational complexity of conjunctive query (CQ) answering is well known to be EXPTIME-complete and hence not harder than satisfiability. This does not change when the logic is extended by certain features (such as counting or role hierarchies), whereas adding others (inverses, nominals or transitivity together with role-hierarchies) turns CQ answering exponentially harder.
 We contribute to this line of results by showing the surprising fact that even extending ALC by just the Self operator – which proved innocuous in many other contexts – increases the complexity of CQ entailment to 2EXPTIME. As common for this type of problem, our proof establishes a reduction from alternating Turing machines running in exponential space, but several novel ideas and encoding tricks are required to make the approach work in that specific, restricted setting.

----

## [616] Expressivity of Planning with Horn Description Logic Ontologies

**Authors**: *Stefan Borgwardt, Jörg Hoffmann, Alisa Kovtunova, Markus Krötzsch, Bernhard Nebel, Marcel Steinmetz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20489](https://doi.org/10.1609/aaai.v36i5.20489)

**Abstract**:

State constraints in AI Planning globally restrict the legal environment states. Standard planning languages make closed-domain and closed-world assumptions. Here we address open-world state constraints formalized by planning over a description logic (DL) ontology. Previously, this combination of DL and planning has been investigated for the light-weight DL DL-Lite. Here we propose a novel compilation scheme into standard PDDL with derived predicates, which applies to more expressive DLs and is based on the rewritability of DL queries into Datalog with stratified negation. We also provide a new rewritability result for the DL Horn-ALCHOIQ, which allows us to apply our compilation scheme to quite expressive ontologies. In contrast, we show that in the slight extension Horn-SROIQ no such compilation is possible unless the weak exponential hierarchy collapses. Finally, we show that our approach can outperform previous work on existing benchmarks for planning with DL ontologies, and is feasible on new benchmarks taking advantage of more expressive ontologies.

----

## [617] ER: Equivariance Regularizer for Knowledge Graph Completion

**Authors**: *Zongsheng Cao, Qianqian Xu, Zhiyong Yang, Qingming Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20490](https://doi.org/10.1609/aaai.v36i5.20490)

**Abstract**:

Tensor factorization and distanced based models play important roles in knowledge graph completion (KGC). However, the relational matrices in KGC methods often induce a high model complexity, bearing a high risk of overfitting. As a remedy, researchers propose a variety of different regularizers such as the tensor nuclear norm regularizer. Our motivation is based on the observation that the previous work only focuses on the “size” of the parametric space, while leaving the implicit semantic information widely untouched. To address this issue, we propose a new regularizer, namely, Equivariance Regularizer (ER), which can suppress overfitting by leveraging the implicit semantic information. Specifically, ER can enhance the generalization ability of the model by employing the semantic equivariance between the head and tail entities. Moreover, it is a generic solution for both distance based models and tensor factorization based models. Our experimental results indicate a clear and substantial improvement over the state-of-the-art relation prediction methods.

----

## [618] Geometry Interaction Knowledge Graph Embeddings

**Authors**: *Zongsheng Cao, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20491](https://doi.org/10.1609/aaai.v36i5.20491)

**Abstract**:

Knowledge graph (KG) embeddings have shown great power in learning representations of entities and relations for link prediction tasks. Previous work usually embeds KGs into a single geometric space such as Euclidean space (zero curved), hyperbolic space (negatively curved) or hyperspherical space (positively curved) to maintain their specific geometric structures (e.g., chain, hierarchy and ring structures). However, the topological structure of KGs appears to be complicated, since it may contain multiple types of geometric structures simultaneously. Therefore, embedding KGs in a single space, no matter the Euclidean space, hyperbolic space or hyperspheric space, cannot capture the complex structures of KGs accurately. To overcome this challenge, we propose Geometry Interaction knowledge graph Embeddings (GIE), which learns spatial structures interactively between the Euclidean, hyperbolic and hyperspherical spaces. Theoretically, our proposed GIE can capture a richer set of relational information, model key inference patterns, and enable expressive semantic matching across entities. Experimental results on three well-established knowledge graph completion benchmarks show that our GIE achieves the state-of-the-art performance with fewer parameters.

----

## [619] Multi-Relational Graph Representation Learning with Bayesian Gaussian Process Network

**Authors**: *Guanzheng Chen, Jinyuan Fang, Zaiqiao Meng, Qiang Zhang, Shangsong Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20492](https://doi.org/10.1609/aaai.v36i5.20492)

**Abstract**:

Learning effective representations of entities and relations for knowledge graphs (KGs) is critical to the success of many multi-relational learning tasks. Existing methods based on graph neural networks learn a deterministic embedding function, which lacks sufficient flexibility to explore better choices when dealing with the imperfect and noisy KGs such as the scarce labeled nodes and noisy graph structure. To this end, we propose a novel multi-relational graph Gaussian Process network (GGPN), which aims to improve the flexibility of deterministic methods by simultaneously learning a family of embedding functions, i.e., a stochastic embedding function. Specifically, a Bayesian Gaussian Process (GP) is proposed to model the distribution of this stochastic function and the resulting representations are obtained by aggregating stochastic function values, i.e., messages, from neighboring entities. The two problems incurred when leveraging GP in GGPN are the proper choice of kernel function and the cubic computational complexity. To address the first problem, we further propose a novel kernel function that can explicitly take the diverse relations between each pair of entities into account and be adaptively learned in a data-driven way. We address the second problem by reformulating GP as a Bayesian linear model, resulting in a linear computational complexity. With these two solutions, our GGPN can be efficiently trained in an end-to-end manner. We evaluate our GGPN in link prediction and entity classification tasks, and the experimental results demonstrate the superiority of our method. Our code is available at https://github.com/sysu-gzchen/GGPN.

----

## [620] ASP-Based Declarative Process Mining

**Authors**: *Francesco Chiariello, Fabrizio Maria Maggi, Fabio Patrizi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20493](https://doi.org/10.1609/aaai.v36i5.20493)

**Abstract**:

We put forward Answer Set Programming (ASP) as a solution approach for three classical problems in Declarative Process Mining: Log Generation, Query Checking, and Conformance Checking. These problems correspond to different ways of analyzing business processes under execution, starting from sequences of recorded events, a.k.a. event logs. We tackle them in their data-aware variant, i.e., by considering events that carry a payload (set of attribute-value pairs), in addition to the performed activity, specifying processes declaratively with an extension of linear-time temporal logic over finite traces (LTLf). The data-aware setting is significantly more challenging than the control-flow one: Query Checking is still open, while the existing approaches for the other two problems do not scale well. The contributions of the work include an ASP encoding schema for the three problems, their solution, and experiments showing the feasibility of the approach.

----

## [621] On Testing for Discrimination Using Causal Models

**Authors**: *Hana Chockler, Joseph Y. Halpern*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20494](https://doi.org/10.1609/aaai.v36i5.20494)

**Abstract**:

Consider a bank that uses an AI system to decide which loan applications to approve. We want to ensure that the system is fair, that is, it does not discriminate against applicants based on a predefined list of sensitive attributes, such as gender and ethnicity. We expect there to be a regulator whose job it is to certify the bank’s system as fair or unfair. We consider issues that the regulator will have to confront when making such a decision, including the precise definition of fairness, dealing with proxy variables, and dealing with what we call allowed variables, that is, variables such as salary on which the decision is allowed to depend, despite being correlated with sensitive variables. We show (among other things) that the problem of deciding fairness as we have defined it is co-NP-complete, but then argue that, despite that, in practice the problem should be manageable.

----

## [622] Monotone Abstractions in Ontology-Based Data Management

**Authors**: *Gianluca Cima, Marco Console, Maurizio Lenzerini, Antonella Poggi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20495](https://doi.org/10.1609/aaai.v36i5.20495)

**Abstract**:

In Ontology-Based Data Management (OBDM), an abstraction of a source query q is a query over the ontology capturing the semantics of q in terms of the concepts and the relations available in the ontology. Since a perfect characterization of a source query may not exist, the notions of best sound and complete approximations of an abstraction have been introduced and studied in the typical OBDM context, i.e., in the case where the ontology is expressed in DL-Lite, and source queries are expressed as unions of conjunctive queries (UCQs). Interestingly, if we restrict our attention to abstractions expressed as UCQs, even best approximations of abstractions are not guaranteed to exist. Thus, a natural question to ask is whether such limitations affect even larger classes of queries. In this paper, we answer this fundamental question for an essential class of queries, namely the class of monotone queries. We define a monotone query language based on disjunctive Datalog enriched with an epistemic operator, and show that its expressive power suffices for expressing the best approximations of monotone abstractions of UCQs.

----

## [623] Lower Bounds on Intermediate Results in Bottom-Up Knowledge Compilation

**Authors**: *Alexis de Colnet, Stefan Mengel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20496](https://doi.org/10.1609/aaai.v36i5.20496)

**Abstract**:

Bottom-up knowledge compilation is a paradigm for generating representations of functions by iteratively conjoining constraints using a so-called apply function. When the input is not efficiently compilable into a language - generally a class of circuits - because optimal compiled representations are provably large, the problem is not the compilation algorithm as much as the choice of a language too restrictive for the input. In contrast, in this paper, we look at CNF formulas for which very small circuits exists and look at the efficiency of their bottom-up compilation in one of the most general languages, namely that of structured decomposable negation normal forms (str-DNNF). We prove that, while the inputs have constant size representations as str-DNNF, any bottom-up compilation in the general setting where conjunction and structure modification are allowed takes exponential time and space, since large intermediate results have to be produced. This unconditionally proves that the inefficiency of bottom-up compilation resides in the bottom-up paradigm itself.

----

## [624] Enforcement Heuristics for Argumentation with Deep Reinforcement Learning

**Authors**: *Dennis Craandijk, Floris Bex*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20497](https://doi.org/10.1609/aaai.v36i5.20497)

**Abstract**:

In this paper, we present a learning-based approach to the symbolic reasoning problem of dynamic argumentation, where the knowledge about attacks between arguments is incomplete or evolving. Specifically, we employ deep reinforcement learning to learn which attack relations between arguments should be added or deleted in order to enforce the acceptability of (a set of) arguments. We show that our Graph Neural Network (GNN) architecture EGNN can learn a near optimal enforcement heuristic for all common argument-fixed enforcement problems, including problems for which no other (symbolic) solvers exist. We demonstrate that EGNN outperforms other GNN baselines and on enforcement problems with high computational complexity performs better than state-of-the-art symbolic solvers with respect to efficiency. Thus, we show our neuro-symbolic approach is able to learn heuristics without the expert knowledge of a human designer and offers a valid alternative to symbolic solvers. We publish our code at https://github.com/DennisCraandijk/DL-Abstract-Argumentation.

----

## [625] On the Computation of Necessary and Sufficient Explanations

**Authors**: *Adnan Darwiche, Chunxi Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20498](https://doi.org/10.1609/aaai.v36i5.20498)

**Abstract**:

The complete reason behind a decision is a Boolean formula that characterizes why the decision was made. This recently introduced notion has a number of applications, which include generating explanations, detecting decision bias and evaluating counterfactual queries. Prime implicants of the complete reason are known as sufficient reasons for the decision and they correspond to what is known as PI explanations and abductive explanations. In this paper, we refer to the prime implicates of a complete reason as necessary reasons for the decision. We justify this terminology semantically and show that necessary reasons correspond to what is known as contrastive explanations. We also study the computation of complete reasons for multi-class decision trees and graphs with nominal and numeric features for which we derive efficient, closed-form complete reasons. We further investigate the computation of shortest necessary and sufficient reasons for a broad class of complete reasons, which include the derived closed forms and the complete reasons for Sentential Decision Diagrams (SDDs). We provide an algorithm which can enumerate their shortest necessary reasons in output polynomial time. Enumerating shortest sufficient reasons for this class of complete reasons is hard even for a single reason. For this problem, we provide an algorithm that appears to be quite efficient as we show empirically.

----

## [626] Machine Learning for Utility Prediction in Argument-Based Computational Persuasion

**Authors**: *Ivan Donadello, Anthony Hunter, Stefano Teso, Mauro Dragoni*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20499](https://doi.org/10.1609/aaai.v36i5.20499)

**Abstract**:

Automated persuasion systems (APS) aim to persuade a user to believe something by entering into a dialogue in which arguments and counterarguments are exchanged. To maximize the probability that an APS is successful in persuading a user, it can identify a global policy that will allow it to select the best arguments it presents at each stage of the dialogue whatever arguments the user presents. However, in real applications, such as for healthcare, it is unlikely the utility of the outcome of the dialogue will be the same, or the exact opposite, for the APS and user. In order to deal with this situation, games in extended form have been harnessed for argumentation in Bi-party Decision Theory. This opens new problems that we address in this paper: (1) How can we use Machine Learning (ML) methods to predict utility functions for different subpopulations of users? and (2) How can we identify for a new user the best utility function from amongst those that we have learned. To this extent, we develop two ML methods, EAI and EDS, that leverage information coming from the users to predict their utilities. EAI is restricted to a fixed amount of information, whereas EDS can choose the information that best detects the subpopulations of a user. We evaluate EAI and EDS in a simulation setting and in a realistic case study concerning healthy eating habits. Results are promising in both cases, but EDS is more effective at predicting useful utility functions.

----

## [627] On the Complexity of Inductively Learning Guarded Clauses

**Authors**: *Andrei Draghici, Georg Gottlob, Matthias Lanzinger*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20500](https://doi.org/10.1609/aaai.v36i5.20500)

**Abstract**:

We investigate the computational complexity of mining guarded clauses from clausal datasets through the framework of inductive logic programming (ILP). We show that learning guarded clauses is NP-complete and thus one step below the Sigma2-complete task of learning Horn clauses on the polynomial hierarchy. Motivated by practical applications on large datasets we identify a natural tractable fragment of the problem. Finally, we also generalise all of our results to k-guarded clauses for constant k.

----

## [628] Tractable Abstract Argumentation via Backdoor-Treewidth

**Authors**: *Wolfgang Dvorák, Markus Hecher, Matthias König, André Schidler, Stefan Szeider, Stefan Woltran*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20501](https://doi.org/10.1609/aaai.v36i5.20501)

**Abstract**:

Argumentation frameworks (AFs) are a core formalism in the field of formal argumentation. As most standard computational tasks regarding AFs are hard for the first or second level of the Polynomial Hierarchy, a variety of algorithmic approaches to achieve manageable runtimes have been considered in the past. Among them, the backdoor-approach and the treewidth-approach turned out to yield fixed-parameter tractable fragments. However, many applications yield high parameter values for these methods, often rendering them infeasible in practice. We introduce the backdoor-treewidth approach for abstract argumentation, combining the best of both worlds with a guaranteed parameter value that does not exceed the minimum of the backdoor- and treewidth-parameter. In particular, we formally define backdoor-treewidth and establish fixed-parameter tractability for standard reasoning tasks of abstract argumentation. Moreover, we provide systems to find and exploit backdoors of small width, and conduct systematic experiments evaluating the new parameter.

----

## [629] Large-Neighbourhood Search for Optimisation in Answer-Set Solving

**Authors**: *Thomas Eiter, Tobias Geibinger, Nelson Higuera Ruiz, Nysret Musliu, Johannes Oetsch, Daria Stepanova*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20502](https://doi.org/10.1609/aaai.v36i5.20502)

**Abstract**:

While Answer-Set Programming (ASP) is a prominent approach to declarative problem solving, optimisation problems can still be a challenge for it. Large-Neighbourhood Search (LNS) is a metaheuristic for optimisation where parts of a solution are alternately destroyed and reconstructed that has high but untapped potential for ASP solving. We present a framework for LNS optimisation in answer-set solving, in which neighbourhoods can be specified either declaratively as part of the ASP encoding, or automatically generated by code. To effectively explore different neighbourhoods, we focus on multi-shot solving as it allows to avoid program regrounding. We illustrate the framework on different optimisation problems, some of which are notoriously difficult, including shift planning and a parallel machine scheduling problem from semi-conductor production which demonstrate the effectiveness of the LNS approach.

----

## [630] Answering Queries with Negation over Existential Rules

**Authors**: *Stefan Ellmauthaler, Markus Krötzsch, Stephan Mennicke*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20503](https://doi.org/10.1609/aaai.v36i5.20503)

**Abstract**:

Ontology-based query answering with existential rules is well understood and implemented for positive queries, in particular conjunctive queries. For queries with negation, however, there is no agreed-upon semantics or standard implementation. This problem is unknown for simpler rule languages, such as Datalog, where it is intuitive and practical to evaluate negative queries over the least model. This fails for existential rules, which instead of a single least model have multiple universal models that may not lead to the same results for negative queries. We therefore propose universal core models as a basis for a meaningful (non-monotonic) semantics for queries with negation. Since cores are hard to compute, we identify syntactic conditions (on rules and queries) under which our core-based semantics can equivalently be obtained for other universal models, such as those produced by practical chase algorithms. Finally, we use our findings to propose a semantics for a broad class of existential rules with negation.

----

## [631] Axiomatization of Aggregates in Answer Set Programming

**Authors**: *Jorge Fandinno, Zachary Hansen, Yuliya Lierler*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20504](https://doi.org/10.1609/aaai.v36i5.20504)

**Abstract**:

The paper presents a characterization of logic programs with aggregates based on many-sorted generalization of operator SM that refers neither to grounding nor to fixpoints. This characterization introduces new symbols for aggregate operations and aggregate elements, whose meaning is fixed by adding appropriate axioms to the result of the SM transformation. We prove that for programs without positive recursion through aggregates our semantics coincides with the semantics of the answer set solver Clingo.

----

## [632] Linear-Time Verification of Data-Aware Dynamic Systems with Arithmetic

**Authors**: *Paolo Felli, Marco Montali, Sarah Winkler*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20505](https://doi.org/10.1609/aaai.v36i5.20505)

**Abstract**:

Combined modeling and verification of dynamic systems and the data they operate on has gained momentum in AI and in several application domains. We investigate the expressive yet concise framework of data-aware dynamic systems (DDS), extending it with linear arithmetic, and providing the following contributions. 
 First, we introduce a new, semantic property of “finite summary”, which guarantees the existence of a faithful finite-state abstraction. We rely on this to show that checking whether a witness exists for a linear-time, finite-trace property is decidable for DDSs with finite summary. Second, we demonstrate that several decidability conditions studied in formal methods and database theory can be seen as concrete, checkable instances of this property. This also gives rise to new decidability results. Third, we show how the abstract, uniform property of finite summary leads to modularity results: a system enjoys finite summary if it can be partitioned appropriately into smaller systems that possess the property. Our results allow us to analyze systems that were out of reach in earlier approaches. Finally, we demonstrate the feasibility of our approach in a prototype implementation.

----

## [633] Rushing and Strolling among Answer Sets - Navigation Made Easy

**Authors**: *Johannes Klaus Fichte, Sarah Alice Gaggl, Dominik Rusovac*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20506](https://doi.org/10.1609/aaai.v36i5.20506)

**Abstract**:

Answer set programming (ASP) is a popular declarative programming paradigm
  with a wide range of applications in artificial intelligence. Oftentimes,
  when modeling an AI problem with ASP, and in particular when we are interested
  beyond simple search for optimal solutions, an actual solution, differences
  between solutions, or number of solutions of the ASP program matter. For
  example, when a user aims to identify a specific answer set according to her
  needs, or requires the total number of diverging solutions to comprehend
  probabilistic applications such as reasoning in medical domains. Then, there
  are only certain problem specific and handcrafted encoding techniques
  available to navigate the solution space of ASP programs, which is oftentimes
  not enough. In this paper, we propose a formal and general framework for
  interactive navigation toward desired subsets of answer sets analogous to
  faceted browsing. Our approach enables the user to explore the solution space
  by consciously zooming in or out of sub-spaces of solutions at a certain
  configurable pace. We illustrate that weighted faceted navigation is
  computationally hard. Finally, we provide an implementation of our approach
  that demonstrates the feasibility of our framework for incomprehensible
  solution spaces.

----

## [634] Sufficient Reasons for Classifier Decisions in the Presence of Domain Constraints

**Authors**: *Niku Gorji, Sasha Rubin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20507](https://doi.org/10.1609/aaai.v36i5.20507)

**Abstract**:

Recent work has unveiled a theory for reasoning about the decisions made by binary classifiers: a classifier describes a Boolean function, and the reasons behind an instance being classified as positive are the prime-implicants of the function that are satisfied by the instance. One drawback of these works is that they do not explicitly treat scenarios where the underlying data is known to be constrained, e.g., certain combinations of features may not exist, may not be observable, or may be required to be disregarded. We propose a more general theory, also based on prime-implicants, tailored to taking constraints into account. The main idea is to view classifiers as describing partial Boolean functions that are undefined on instances that do not satisfy the constraints. We prove that this simple idea results in more parsimonious reasons. That is, not taking constraints into account (e.g., ignoring, or taking them as negative instances) results in reasons that are subsumed by reasons that do take constraints into account. We illustrate this improved succinctness on synthetic classifiers and classifiers learnt from real data.

----

## [635] Reasoning about Causal Models with Infinitely Many Variables

**Authors**: *Joseph Y. Halpern, Spencer Peters*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20508](https://doi.org/10.1609/aaai.v36i5.20508)

**Abstract**:

Generalized structural equations models (GSEMs) (Peters and Halpern 2021), are, as the name suggests, a generalization of structural equations models (SEMs). They can deal with (among other things) infinitely many variables with infinite ranges, which is critical for capturing dynamical systems. We provide a sound and complete axiomatization of causal reasoning in GSEMs that is an extension of the sound and complete axiomatization provided by Halpern (2000) for SEMs. Considering GSEMs helps clarify what properties Halpern's axioms capture.

----

## [636] An Axiomatic Approach to Revising Preferences

**Authors**: *Adrian Haret, Johannes Peter Wallner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20509](https://doi.org/10.1609/aaai.v36i5.20509)

**Abstract**:

We study a model of preference revision in which a prior preference over a set of alternatives is adjusted in order to accommodate input from an authoritative source, while maintaining certain structural constraints (e.g., transitivity, completeness), and without giving up more information than strictly necessary. We analyze this model under two aspects: the first allows us to capture natural distance-based operators, at the cost of a mismatch between the input and output formats of the revision operator. Requiring the input and output to be aligned yields a second type of operator, which we characterize using preferences on the comparisons in the prior preference Prefence revision is set in a logic-based framework and using the formal machinery of belief change, along the lines of the well-known AGM approach: we propose rationality postulates for each of the two versions of our model and derive representation results, thus situating preference revision within the larger family of belief change operators.

----

## [637] BERTMap: A BERT-Based Ontology Alignment System

**Authors**: *Yuan He, Jiaoyan Chen, Denvar Antonyrajah, Ian Horrocks*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20510](https://doi.org/10.1609/aaai.v36i5.20510)

**Abstract**:

Ontology alignment (a.k.a ontology matching (OM)) plays a critical role in knowledge integration. Owing to the success of machine learning in many domains, it has been applied in OM. However, the existing methods, which often adopt ad-hoc feature engineering or non-contextual word embeddings, have not yet outperformed rule-based systems especially in an unsupervised setting. In this paper, we propose a novel OM system named BERTMap which can support both unsupervised and semi-supervised settings. It first predicts mappings using a classifier based on fine-tuning the contextual embedding model BERT on text semantics corpora extracted from ontologies, and then refines the mappings through extension and repair by utilizing the ontology structure and logic. Our evaluation with three alignment tasks on biomedical ontologies demonstrates that BERTMap can often perform better than the leading OM systems LogMap and AML.

----

## [638] Conditional Abstract Dialectical Frameworks

**Authors**: *Jesse Heyninck, Matthias Thimm, Gabriele Kern-Isberner, Tjitze Rienstra, Kenneth Skiba*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20511](https://doi.org/10.1609/aaai.v36i5.20511)

**Abstract**:

Abstract dialectical frameworks (in short, ADFs) are a unifying model of formal argumentation, where argumentative relations between arguments are represented by assigning acceptance conditions to atomic arguments. This idea is generalized by letting acceptance conditions being assigned to complex formulas, resulting in conditional abstract dialectical frameworks (in short, cADFs). We define the semantics of cADFs in terms of a non-truth-functional four-valued logic, and study the semantics in-depth, by showing existence results and proving that all semantics are generalizations of the corresponding semantics for ADFs.

----

## [639] MultiplexNet: Towards Fully Satisfied Logical Constraints in Neural Networks

**Authors**: *Nick Hoernle, Rafael-Michael Karampatsis, Vaishak Belle, Kobi Gal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20512](https://doi.org/10.1609/aaai.v36i5.20512)

**Abstract**:

We propose a novel way to incorporate expert knowledge into the training of deep neural networks. Many approaches encode domain constraints directly into the network architecture, requiring non-trivial or domain-specific engineering. In contrast, our approach, called MultiplexNet, represents domain knowledge as a quantifier-free logical formula in disjunctive normal form (DNF) which is easy to encode and to elicit from human experts. It introduces a latent Categorical variable that learns to choose which constraint term optimizes the error function of the network and it compiles the constraints directly into the output of existing learning algorithms. We demonstrate the efficacy of this approach empirically on several classical deep learning tasks, such as density estimation and classification in both supervised and unsupervised settings where prior knowledge about the domains was expressed as logical constraints. Our results show that the MultiplexNet approach learned to approximate unknown distributions well, often requiring fewer data samples than the alternative approaches. In some cases, MultiplexNet finds better solutions than the baselines; or solutions that could not be achieved with the alternative approaches. Our contribution is in encoding domain knowledge in a way that facilitates inference. We specifically focus on quantifier-free logical formulae that are specified over the output domain of a network. We show that this approach is both efficient and general; and critically, our approach guarantees 100% constraint satisfaction in a network's output.

----

## [640] Towards Explainable Action Recognition by Salient Qualitative Spatial Object Relation Chains

**Authors**: *Hua Hua, Dongxu Li, Ruiqi Li, Peng Zhang, Jochen Renz, Anthony G. Cohn*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20513](https://doi.org/10.1609/aaai.v36i5.20513)

**Abstract**:

In order to be trusted by humans, Artificial Intelligence agents should be able to describe rationales behind their decisions. One such application is human action recognition in critical or sensitive scenarios, where trustworthy and explainable action recognizers are expected. For example, reliable pedestrian action recognition is essential for self-driving cars and explanations for real-time decision making are critical for investigations if an accident happens. In this regard, learning-based approaches, despite their popularity and accuracy, are disadvantageous due to their limited interpretability. 
 
 This paper presents a novel neuro-symbolic approach that recognizes actions from videos with human-understandable explanations. Specifically, we first propose to represent videos symbolically by qualitative spatial relations between objects called qualitative spatial object relation chains. We further develop a neural saliency estimator to capture the correlation between such object relation chains and the occurrence of actions. Given an unseen video, this neural saliency estimator is able to tell which object relation chains are more important for the action recognized. We evaluate our approach on two real-life video datasets, with respect to recognition accuracy and the quality of generated action explanations. Experiments show that our approach achieves superior performance on both aspects to previous symbolic approaches, thus facilitating trustworthy intelligent decision making. Our approach can be used to augment state-of-the-art learning approaches with explainabilities.

----

## [641] Tractable Explanations for d-DNNF Classifiers

**Authors**: *Xuanxiang Huang, Yacine Izza, Alexey Ignatiev, Martin C. Cooper, Nicholas Asher, João Marques-Silva*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20514](https://doi.org/10.1609/aaai.v36i5.20514)

**Abstract**:

Compilation into propositional languages finds a growing number of practical uses, including in constraint programming, diagnosis and machine learning (ML), among others. One concrete example is the use of propositional languages as classifiers, and one natural question is how to explain the predictions made. This paper shows that for classifiers represented with some of the best-known propositional languages, different kinds of explanations can be computed in polynomial time. These languages include deterministic decomposable negation normal form (d-DNNF), and so any propositional language that is strictly less succinct than d-DNNF. Furthermore, the paper describes optimizations, specific to Sentential Decision Diagrams (SDDs), which are shown to yield more efficient algorithms in practice.

----

## [642] Understanding Enthymemes in Deductive Argumentation Using Semantic Distance Measures

**Authors**: *Anthony Hunter*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20515](https://doi.org/10.1609/aaai.v36i5.20515)

**Abstract**:

An argument can be regarded as some premises and a claim following from those premises. Normally, arguments exchanged by human agents are enthymemes, which generally means that some premises are implicit. So when an enthymeme is presented, the presenter expects that the recipient can identify the missing premises. An important kind of implicitness arises when a presenter assumes that two symbols denote the same, or nearly the same, concept (e.g. dad and father), and uses the symbols interchangeably. To model this process, we propose the use of semantic distance measures (e.g. based on a vector representation of word embeddings or a semantic network representation of words) to determine whether one symbol can be substituted by another. We present a theoretical framework for using substitutions, together with abduction of default knowledge, for understanding enthymemes based on deductive argumentation, and investigate how this could be used in practice.

----

## [643] Inferring Lexicographically-Ordered Rewards from Preferences

**Authors**: *Alihan Hüyük, William R. Zame, Mihaela van der Schaar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20516](https://doi.org/10.1609/aaai.v36i5.20516)

**Abstract**:

Modeling the preferences of agents over a set of alternatives is a principal concern in many areas. The dominant approach has been to find a single reward/utility function with the property that alternatives yielding higher rewards are preferred over alternatives yielding lower rewards. However, in many settings, preferences are based on multiple—often competing—objectives; a single reward function is not adequate to represent such preferences. This paper proposes a method for inferring multi-objective reward-based representations of an agent's observed preferences. We model the agent's priorities over different objectives as entering lexicographically, so that objectives with lower priorities matter only when the agent is indifferent with respect to objectives with higher priorities. We offer two example applications in healthcare—one inspired by cancer treatment, the other inspired by organ transplantation—to illustrate how the lexicographically-ordered rewards we learn can provide a better understanding of a decision-maker's preferences and help improve policies when used in reinforcement learning.

----

## [644] Towards Fine-Grained Reasoning for Fake News Detection

**Authors**: *Yiqiao Jin, Xiting Wang, Ruichao Yang, Yizhou Sun, Wei Wang, Hao Liao, Xing Xie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20517](https://doi.org/10.1609/aaai.v36i5.20517)

**Abstract**:

The detection of fake news often requires sophisticated reasoning skills, such as logically combining information by considering word-level subtle clues. In this paper, we move towards fine-grained reasoning for fake news detection by better reflecting the logical processes of human thinking and enabling the modeling of subtle clues. In particular, we propose a fine-grained reasoning framework by following the human’s information-processing model, introduce a mutual-reinforcement-based method for incorporating human knowledge about which evidence is more important, and design a prior-aware bi-channel kernel graph network to model subtle differences between pieces of evidence. Extensive experiments show that our model outperforms the state-of-the-art methods and demonstrate the explainability of our approach.

----

## [645] ApproxASP - a Scalable Approximate Answer Set Counter

**Authors**: *Mohimenul Kabir, Flavio O. Everardo, Ankit K. Shukla, Markus Hecher, Johannes Klaus Fichte, Kuldeep S. Meel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20518](https://doi.org/10.1609/aaai.v36i5.20518)

**Abstract**:

Answer Set Programming (ASP) is a framework in artificial intelligence and knowledge representation for declarative modeling and problem solving. Modern ASP solvers focus on the computation or enumeration of answer sets. However, a variety of probabilistic applications in reasoning or logic programming require counting answer sets. While counting can be done by enumeration, simple enumeration becomes immediately infeasible if the number of solutions is high. On the other hand, approaches to exact counting are of high worst-case complexity. In fact, in propositional model counting, exact counting becomes impractical. In this work, we present a scalable approach to approximate counting for answer set programming. Our approach is based on systematically adding XOR constraints to ASP programs, which divide the search space. We prove that adding random XOR constraints partitions the answer sets of an ASP program. In practice, we use a Gaussian elimination-based approach by lifting ideas from SAT to ASP and integrating it into a state of the art ASP solver, which we call ApproxASP. Finally, our experimental evaluation shows the scalability of our approach over the existing ASP systems.

----

## [646] Unit Selection with Causal Diagram

**Authors**: *Ang Li, Judea Pearl*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20519](https://doi.org/10.1609/aaai.v36i5.20519)

**Abstract**:

The unit selection problem aims to identify a set of individuals who are most likely to exhibit a desired mode of behavior, for example, selecting individuals who would respond one way if encouraged and a different way if not encouraged. Using a combination of experimental and observational data, Li and Pearl derived tight bounds on the "benefit function" - the payoff/cost associated with selecting an individual with given characteristics. This paper shows that these bounds can be narrowed significantly (enough to change decisions) when structural information is available in the form of a causal model. We address the problem of estimating the benefit function using observational and experimental data when specific graphical criteria are assumed to hold.

----

## [647] Bounds on Causal Effects and Application to High Dimensional Data

**Authors**: *Ang Li, Judea Pearl*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20520](https://doi.org/10.1609/aaai.v36i5.20520)

**Abstract**:

This paper addresses the problem of estimating causal effects when adjustment variables in the back-door or front-door criterion are partially observed. For such scenarios, we derive bounds on the causal effects by solving two non-linear optimization problems, and demonstrate that the bounds are sufficient. Using this optimization method, we propose a framework for dimensionality reduction that allows one to trade bias for estimation power, and demonstrate its performance using simulation studies.

----

## [648] How Does Knowledge Graph Embedding Extrapolate to Unseen Data: A Semantic Evidence View

**Authors**: *Ren Li, Yanan Cao, Qiannan Zhu, Guanqun Bi, Fang Fang, Yi Liu, Qian Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20521](https://doi.org/10.1609/aaai.v36i5.20521)

**Abstract**:

Knowledge Graph Embedding (KGE) aims to learn representations for entities and relations. Most KGE models have gained great success, especially on extrapolation scenarios. Specifically, given an unseen triple (h, r, t), a trained model can still correctly predict t from (h, r, ?), or h from (?, r, t), such extrapolation ability is impressive. However, most existing KGE works focus on the design of delicate triple modeling function, which mainly tells us how to measure the plausibility of observed triples, but offers limited explanation of why the methods can extrapolate to unseen data, and what are the important factors to help KGE extrapolate. Therefore in this work, we attempt to study the KGE extrapolation of two problems: 1. How does KGE extrapolate to unseen data? 2. How to design the KGE model with better extrapolation ability? 
For the problem 1, we first discuss the impact factors for extrapolation and from relation, entity and triple level respectively, propose three Semantic Evidences (SEs), which can be observed from train set and provide important semantic information for extrapolation. Then we verify the effectiveness of SEs through extensive experiments on several typical KGE methods.
For the problem 2, to make better use of the three levels of SE, we propose a novel GNN-based KGE model, called Semantic Evidence aware Graph Neural Network (SE-GNN). In SE-GNN, each level of SE is modeled explicitly by the corresponding neighbor pattern, and merged sufficiently by the multi-layer aggregation, which contributes to obtaining more extrapolative knowledge representation. 
Finally, through extensive experiments on FB15k-237 and WN18RR datasets, we show that SE-GNN achieves state-of-the-art performance on Knowledge Graph Completion task and performs a better extrapolation ability. Our code is available at https://github.com/renli1024/SE-GNN.

----

## [649] Multi-View Graph Representation for Programming Language Processing: An Investigation into Algorithm Detection

**Authors**: *Ting Long, Yutong Xie, Xianyu Chen, Weinan Zhang, Qinxiang Cao, Yong Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20522](https://doi.org/10.1609/aaai.v36i5.20522)

**Abstract**:

Program representation, which aims at converting program source code into vectors with automatically extracted features, is a fundamental problem in programming language processing (PLP). Recent work tries to represent programs with neural networks based on source code structures. However, such methods often focus on the syntax and consider only one single perspective of programs, limiting the representation power of models. This paper proposes a multi-view graph (MVG) program representation method. MVG pays more attention to code semantics and simultaneously includes both data flow and control flow as multiple views. These views are then combined and processed by a graph neural network (GNN) to obtain a comprehensive program representation that covers various aspects. We thoroughly evaluate our proposed MVG approach in the context of algorithm detection, an important and challenging subfield of PLP. Specifically, we use a public dataset POJ-104 and also construct a new challenging dataset ALG-109 to test our method. In experiments, MVG outperforms previous methods significantly, demonstrating our model's strong capability of representing source code.

----

## [650] Automated Synthesis of Generalized Invariant Strategies via Counterexample-Guided Strategy Refinement

**Authors**: *Kailun Luo, Yongmei Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20523](https://doi.org/10.1609/aaai.v36i5.20523)

**Abstract**:

Strategy synthesis for multi-agent systems has proved to be a hard task, even when limited to two-player games with safety objectives. Generalized strategy synthesis, an extension of generalized planning which aims to produce a single solution for multiple (possibly infinitely many) planning instances, is a promising direction to deal with the state-space explosion problem. In this paper, we formalize the problem of generalized strategy synthesis in the situation calculus. The synthesis task involves second-order theorem proving generally. Thus we consider strategies aiming to maintain invariants; such strategies can be verified with first-order theorem proving. We propose a sound but incomplete approach to synthesize invariant strategies by adapting the framework of counterexample-guided refinement. The key idea for refinement is to generate a strategy using a model checker for a game constructed from the counterexample, and use it to refine the candidate general strategy. We implemented our method and did experiments with a number of game problems. Our system can successfully synthesize solutions for most of the domains within a reasonable amount of time.

----

## [651] Using Conditional Independence for Belief Revision

**Authors**: *Matthew James Lynn, James P. Delgrande, Pavlos Peppas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20524](https://doi.org/10.1609/aaai.v36i5.20524)

**Abstract**:

We present an approach to incorporating qualitative assertions of conditional irrelevance into belief revision, in order to address the limitations of existing work which considers only unconditional irrelevance. These assertions serve to enforce the requirement of minimal change to existing beliefs, while also suggesting a route to reducing the computational cost of belief revision by excluding irrelevant beliefs from consideration. In our approach, a knowledge engineer specifies a collection of multivalued dependencies that encode domain-dependent assertions of conditional irrelevance in the knowledge base. We consider these as capturing properties of the underlying domain which should be taken into account during belief revision. We introduce two related notions of what it means for a multivalued dependency to be taken into account by a belief revision operator: partial and full compliance. We provide characterisations of partially and fully compliant belief revision operators in terms of semantic conditions on their associated faithful rankings. Using these characterisations, we show that the constraints for partially and fully compliant belief revision operators are compatible with the AGM postulates. Finally, we compare our approach to existing work on unconditional irrelevance in belief revision.

----

## [652] Weighted Model Counting in FO2 with Cardinality Constraints and Counting Quantifiers: A Closed Form Formula

**Authors**: *Sagar Malhotra, Luciano Serafini*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20525](https://doi.org/10.1609/aaai.v36i5.20525)

**Abstract**:

Weighted First-Order Model Counting (WFOMC) computes the weighted sum of the models of a first-order logic theory on a given finite domain. First-Order Logic theories that admit polynomial-time WFOMC w.r.t domain cardinality are called domain liftable. We introduce the concept of lifted interpretations as a tool for formulating closed forms for WFOMC. Using lifted interpretations, we reconstruct the closed-form formula for polynomial-time FOMC in the universally quantified fragment of FO2, earlier proposed by Beame et al. We then expand this closed-form to incorporate cardinality constraints, existential quantifiers, and counting quantifiers (a.k.a C2) without losing domain-liftability. Finally, we show that the obtained closed-form motivates a natural definition of a family of weight functions strictly larger than symmetric weight functions.

----

## [653] TempoQR: Temporal Question Reasoning over Knowledge Graphs

**Authors**: *Costas Mavromatis, Prasanna Lakkur Subramanyam, Vassilis N. Ioannidis, Adesoji Adeshina, Phillip Ryan Howard, Tetiana Grinberg, Nagib Hakim, George Karypis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20526](https://doi.org/10.1609/aaai.v36i5.20526)

**Abstract**:

Knowledge Graph Question Answering (KGQA) involves retrieving facts from a Knowledge Graph (KG) using natural language queries. A KG is a curated set of facts consisting of entities linked by relations. Certain facts include also temporal information forming a Temporal KG (TKG). Although many natural questions involve explicit or implicit time constraints, question answering (QA) over TKGs has been a relatively unexplored area. Existing solutions are mainly designed for simple temporal questions that can be answered directly by a single TKG fact.
 This paper puts forth a comprehensive embedding-based framework for answering complex questions over TKGs. Our method termed temporal question reasoning (TempoQR) exploits TKG embeddings to ground the question to the specific entities and time scope it refers to. It does so by augmenting the question embeddings with context, entity and time-aware information by employing three specialized modules. The first computes a textual representation of a given question, the second combines it with the entity embeddings for entities involved in the question, and the third generates question-specific time embeddings. Finally, a transformer-based encoder learns to fuse the generated temporal information with the question representation, which is used for answer predictions. Extensive experiments show that TempoQR improves accuracy by 25--45 percentage points on complex temporal questions over state-of-the-art approaches and it generalizes better to unseen question types.

----

## [654] Compilation of Aggregates in ASP Systems

**Authors**: *Giuseppe Mazzotta, Francesco Ricca, Carmine Dodaro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20527](https://doi.org/10.1609/aaai.v36i5.20527)

**Abstract**:

Answer Set Programming (ASP) is a well-known declarative AI formalism for knowledge representation and reasoning. State-of-the-art ASP implementations employ the ground&solve approach, and they were successfully applied to industrial and academic problems. Nonetheless there are classes of ASP programs whose evaluation is not efficient (sometimes not feasible) due to the combinatorial blow-up of the program produced by the grounding step. Recent researches suggest that compilation-based techniques can mitigate the grounding bottleneck problem. However, no compilation-based technique has been developed for ASP programs that contain aggregates, which are one of the most relevant and commonly-employed constructs of ASP. In this paper, we propose a compilation-based approach for ASP programs with aggregates. We implement it on top of a state-of-the-art ASP system, and evaluate the performance on publicly-available benchmarks. Experiments show our approach is effective on ground-intensive ASP programs.

----

## [655] Prevailing in the Dark: Information Walls in Strategic Games

**Authors**: *Pavel Naumov, Wenxuan Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20528](https://doi.org/10.1609/aaai.v36i5.20528)

**Abstract**:

The paper studies strategic abilities that rise from restrictions on the information sharing in multi-agent systems. The main technical result is a sound and complete logical system that describes the interplay between the knowledge and the strategic ability modalities.

----

## [656] Knowledge Compilation Meets Logical Separability

**Authors**: *Junming Qiu, Wenqing Li, Zhanhao Xiao, Quanlong Guan, Liangda Fang, Zhao-Rong Lai, Qian Dong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20529](https://doi.org/10.1609/aaai.v36i5.20529)

**Abstract**:

Knowledge compilation is an alternative solution to address demanding reasoning tasks with high complexity via converting knowledge bases into a suitable target language. Interestingly, the notion of logical separability, proposed by Levesque, offers a general explanation for the tractability of clausal entailment for two remarkable languages: decomposable negation normal form and prime implicates. It is interesting to explore what role logical separability on earth plays in problem tractability. In this paper, we apply the notion of logical separability in three reasoning problems within the context of propositional logic: satisfiability check (CO), clausal entailment check (CE) and model counting (CT), contributing to three corresponding polytime procedures. We provide three logical separability based properties: CO- logical separability, CE-logical separability and CT-logical separability. We then identify three novel normal forms: CO-LSNNF, CE-LSNNF and CT-LSNNF based on the above properties. Besides, we show that every normal form is the necessary and sufficient condition under which the corresponding procedure is correct. We finally integrate the above four normal forms into the knowledge compilation map.

----

## [657] Propositional Encodings of Acyclicity and Reachability by Using Vertex Elimination

**Authors**: *Masood Feyzbakhsh Rankooh, Jussi Rintanen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20530](https://doi.org/10.1609/aaai.v36i5.20530)

**Abstract**:

We introduce novel methods for encoding acyclicity and s-t-reachability constraints for propositional formulas with underlying directed graphs, based on vertex elimination graphs, which makes them suitable for cases where the underlying graph has a low directed elimination width. In contrast to solvers with ad hoc constraint propagators for graph constraints such as GraphSAT, our methods encode these constraints as standard propositional clauses, making them directly applicable with any SAT solver. An empirical study demonstrates that our methods do often outperform both earlier encodings of these constraints as well as GraphSAT especially when underlying graphs have a low directed elimination width.

----

## [658] Random vs Best-First: Impact of Sampling Strategies on Decision Making in Model-Based Diagnosis

**Authors**: *Patrick Rodler*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20531](https://doi.org/10.1609/aaai.v36i5.20531)

**Abstract**:

Statistical samples, in order to be representative, have to be drawn from a population in a random and unbiased way. Nevertheless, it is common practice in the ﬁeld of model-based diagnosis to make estimations from (biased) best-ﬁrst samples. One example is the computation of a few most probable fault explanations for a defective system and the use of these to assess which aspect of the system, if measured, would bring the highest information gain. In this work, we scrutinize whether these statistically not well-founded conventions, that both diagnosis researchers and practitioners have adhered to for decades, are indeed reasonable. To this end, we empirically analyze various sampling methods that generate fault explanations. We study the representativeness of the produced samples in terms of their estimations about fault explanations and how well they guide diagnostic decisions, and we investigate the impact of sample size, the optimal trade-off between sampling efﬁciency and effectivity, and how approximate sampling techniques compare to exact ones.

----

## [659] On Paraconsistent Belief Revision in LP

**Authors**: *Nicolas Schwind, Sébastien Konieczny, Ramón Pino Pérez*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20532](https://doi.org/10.1609/aaai.v36i5.20532)

**Abstract**:

Belief revision aims at incorporating, in a rational way, a new piece of information into the beliefs of an agent. Most works in belief revision suppose a classical logic setting, where the beliefs of the agent are consistent. Moreover, the consistency postulate states that the result of the revision should be consistent if the new piece of information is consistent. But in real applications it may easily happen that (some parts of) the beliefs of the agent are not consistent. In this case then it seems reasonable to use paraconsistent logics to derive sensible conclusions from these inconsistent beliefs. However, in this context, the standard belief revision postulates trivialize the revision process. In this work we discuss how to adapt these postulates when the underlying logic is Priest's LP logic, in order to model a rational change, while being a conservative extension of AGM/KM belief revision. This implies, in particular, to adequately adapt the notion of expansion. We provide a representation theorem and some examples of belief revision operators in this setting.

----

## [660] Weakly Supervised Neural Symbolic Learning for Cognitive Tasks

**Authors**: *Jidong Tian, Yitian Li, Wenqing Chen, Liqiang Xiao, Hao He, Yaohui Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20533](https://doi.org/10.1609/aaai.v36i5.20533)

**Abstract**:

Despite the recent success of end-to-end deep neural networks, there are growing concerns about their lack of logical reasoning abilities, especially on cognitive tasks with perception and reasoning processes. A solution is the neural symbolic learning (NeSyL) method that can effectively utilize pre-defined logic rules to constrain the neural architecture making it perform better on cognitive tasks. However, it is challenging to apply NeSyL to these cognitive tasks because of the lack of supervision, the non-differentiable manner of the symbolic system, and the difficulty to probabilistically constrain the neural network. In this paper, we propose WS-NeSyL, a weakly supervised neural symbolic learning model for cognitive tasks with logical reasoning. First, WS-NeSyL employs a novel back search algorithm to sample the possible reasoning process through logic rules. This sampled process can supervise the neural network as the pseudo label. Based on this algorithm, we can backpropagate gradients to the neural network of WS-NeSyL in a weakly supervised manner. Second, we introduce a probabilistic logic regularization into WS-NeSyL to help the neural network learn probabilistic logic. To evaluate WS-NeSyL, we have conducted experiments on three cognitive datasets, including temporal reasoning, handwritten formula recognition, and relational reasoning datasets. Experimental results show that WS-NeSyL not only outperforms the end-to-end neural model but also beats the state-of-the-art neural symbolic learning models.

----

## [661] First Order Rewritability in Ontology-Mediated Querying in Horn Description Logics

**Authors**: *David Toman, Grant E. Weddell*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20534](https://doi.org/10.1609/aaai.v36i5.20534)

**Abstract**:

We consider first-order (FO) rewritability for query answering in ontology mediated querying (OMQ) in which ontologies are formulated in Horn fragments of description logics (DLs). In general, OMQ approaches for such logics rely on non-FO rewriting of the query and/or on non-FO completion of the data, called a ABox. Specifically, we consider the problem of FO rewritability in terms of Beth definability, and show how Craig interpolation can then be used to effectively construct the rewritings, when they exist, from the Clark’s completion of Datalog-like programs encoding a given DL TBox and optionally a query. We show how this approach to FO rewritability can also be used to (a) capture integrity constraints commonly available in backend relational data sources, (b) capture constraints inherent in mapping such sources to an ABox , and (c) can be used an alternative to deriving so-called perfect rewritings of queries in the case of DL-Lite ontologies.

----

## [662] MeTeoR: Practical Reasoning in Datalog with Metric Temporal Operators

**Authors**: *Dingmin Wang, Pan Hu, Przemyslaw Andrzej Walega, Bernardo Cuenca Grau*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20535](https://doi.org/10.1609/aaai.v36i5.20535)

**Abstract**:

DatalogMTL is an extension of Datalog with operators from metric temporal logic which has received significant attention in recent years. It is a highly expressive knowledge representation language that is well-suited for applications in temporal ontology-based query answering and stream processing. Reasoning in DatalogMTL is, however, of high computational complexity, making implementation challenging and hindering its adoption in applications. In this paper, we present a novel approach for practical reasoning in DatalogMTL which combines materialisation (a.k.a. forward chaining) with automata-based techniques. We have implemented this approach in a reasoner called MeTeoR and evaluated its performance using a temporal extension of the Lehigh University Benchmark and a benchmark based on real-world meteorological data. Our experiments show that MeTeoR is a scalable system which enables reasoning over complex temporal rules and datasets involving tens of millions of temporal facts.

----

## [663] SGEITL: Scene Graph Enhanced Image-Text Learning for Visual Commonsense Reasoning

**Authors**: *Zhecan Wang, Haoxuan You, Liunian Harold Li, Alireza Zareian, Suji Park, Yiqing Liang, Kai-Wei Chang, Shih-Fu Chang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20536](https://doi.org/10.1609/aaai.v36i5.20536)

**Abstract**:

Answering complex questions about images is an ambitious goal for machine intelligence, which requires a joint understanding of images, text, and commonsense knowledge, as well as a strong reasoning ability. Recently, multimodal Transformers have made a great progress in the task of Visual Commonsense Reasoning (VCR), by jointly understanding visual objects and text tokens through layers of cross-modality attention. However, these approaches do not utilize the rich structure of the scene and the interactions between objects which are essential in answering complex commonsense questions. We propose a
Scene Graph Enhanced  Image-Text  Learning  (SGEITL) framework to incorporate visual scene graph in commonsense reasoning. In order to exploit the scene graph structure, at the model structure level, we propose a multihop graph transformer for regularizing attention interaction among hops. As for pre-training, a scene-graph-aware pre-training method is proposed to leverage structure knowledge extracted in visual scene graph. Moreover, we introduce a method to train and generate domain relevant visual scene graph using textual annotations in a weakly-supervised manner. Extensive experiments on VCR and other tasks show significant performance boost compared with the state-of-the-art methods, and prove the efficacy of each proposed component.

----

## [664] Inductive Relation Prediction by BERT

**Authors**: *Hanwen Zha, Zhiyu Chen, Xifeng Yan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20537](https://doi.org/10.1609/aaai.v36i5.20537)

**Abstract**:

Relation prediction in knowledge graphs is dominated by embedding based methods which mainly focus on the transductive setting. Unfortunately, they are not able to handle inductive learning where unseen entities and relations are present and cannot take advantage of prior knowledge. Furthermore, their inference process is not easily explainable. In this work, we propose an all-in-one solution, called BERTRL (BERT-based Relational Learning), which leverages pre-trained language model and fine-tunes it by taking relation instances and their possible reasoning paths as training samples. BERTRL outperforms the SOTAs in 15 out of 18 cases in both inductive and transductive settings. Meanwhile, it demonstrates strong generalization capability in few-shot learning and is explainable. The data and code can be found at https://github.com/zhw12/BERTRL.

----

## [665] Learning to Walk with Dual Agents for Knowledge Graph Reasoning

**Authors**: *Denghui Zhang, Zixuan Yuan, Hao Liu, Xiaodong Lin, Hui Xiong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20538](https://doi.org/10.1609/aaai.v36i5.20538)

**Abstract**:

Graph walking based on reinforcement learning (RL) has shown great success in navigating an agent to automatically complete various reasoning tasks over an incomplete knowledge graph (KG) by exploring multi-hop relational paths. However, existing multi-hop reasoning approaches only work well on short reasoning paths and tend to miss the target entity with the increasing path length. This is undesirable for many reasoning tasks in real-world scenarios, where short paths connecting the source and target entities are not available in incomplete KGs, and thus the reasoning performances drop drastically unless the agent is able to seek out more clues from longer paths. To address the above challenge, in this paper, we propose a dual-agent reinforcement learning framework, which trains two agents (Giant and Dwarf) to walk over a KG jointly and search for the answer collaboratively. Our approach tackles the reasoning challenge in long paths by assigning one of the agents (Giant) searching on cluster-level paths quickly and providing stage-wise hints for another agent (Dwarf). Finally, experimental results on several KG reasoning benchmarks show that our approach can search answers more accurately and efficiently, and outperforms existing RL-based methods for long path queries by a large margin.

----

## [666] Residual Similarity Based Conditional Independence Test and Its Application in Causal Discovery

**Authors**: *Hao Zhang, Shuigeng Zhou, Kun Zhang, Jihong Guan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20539](https://doi.org/10.1609/aaai.v36i5.20539)

**Abstract**:

Recently, many regression based conditional independence (CI) test methods have been proposed to solve the problem of causal discovery. These methods provide alternatives to test CI by first removing the information of the controlling set from the two target variables, and then testing the independence between the corresponding residuals Res1 and Res2. When the residuals are linearly uncorrelated, the independence test between them is nontrivial. With the ability to calculate inner product in high-dimensional space, kernel-based methods are usually used to achieve this goal, but still consume considerable time. In this paper, we investigate the independence between two linear combinations under linear non-Gaussian structural equation model. We show that the dependence between the two residuals can be captured by the difference between the similarity of (Res1, Res2) and that of (Res1, Res3) (Res3 is generated by random permutation) in high-dimensional space. With this result, we design a new method called SCIT for CI test, where permutation test is performed to control Type I error rate. The proposed method is simpler yet more efficient and effective than the existing ones. When applied to causal discovery, the proposed method outperforms the counterparts in terms of both speed and Type II error rate, especially in the case of small sample size, which is validated by our extensive experiments on various datasets.

----

## [667] Characterizing the Program Expressive Power of Existential Rule Languages

**Authors**: *Heng Zhang, Guifei Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i5.20540](https://doi.org/10.1609/aaai.v36i5.20540)

**Abstract**:

Existential rule languages are a family of ontology languages that have been widely used in ontology-mediated query answering (OMQA). However, for most of them, the expressive power of representing domain knowledge for OMQA, known as the program expressive power, is not well-understood yet. In this paper, we establish a number of novel characterizations for the program expressive power of several important existential rule languages, including tuple-generating dependencies (TGDs), linear TGDs, as well as disjunctive TGDs. The characterizations employ natural model-theoretic properties, and automata-theoretic properties sometimes, which thus provide powerful tools for identifying the definability of domain knowledge for OMQA in these languages.

----

## [668] Context-Specific Representation Abstraction for Deep Option Learning

**Authors**: *Marwa Abdulhai, Dong-Ki Kim, Matthew Riemer, Miao Liu, Gerald Tesauro, Jonathan P. How*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20541](https://doi.org/10.1609/aaai.v36i6.20541)

**Abstract**:

Hierarchical reinforcement learning has focused on discovering temporally extended actions, such as options, that can provide benefits in problems requiring extensive exploration. One promising approach that learns these options end-to-end is the option-critic (OC) framework. We examine and show in this paper that OC does not decompose a problem into simpler sub-problems, but instead increases the size of the search over policy space with each option considering the entire state space during learning. This issue can result in practical limitations of this method, including sample inefficient learning. To address this problem, we introduce Context-Specific Representation Abstraction for Deep Option Learning (CRADOL), a new framework that considers both temporal abstraction and context-specific representation abstraction to effectively reduce the size of the search over policy space. Specifically, our method learns a factored belief state representation that enables each option to learn a policy over only a subsection of the state space. We test our method against hierarchical, non-hierarchical, and modular recurrent neural network baselines, demonstrating significant sample efficiency improvements in challenging partially observable environments.

----

## [669] FisheyeHDK: Hyperbolic Deformable Kernel Learning for Ultra-Wide Field-of-View Image Recognition

**Authors**: *Ola Ahmad, Freddy Lécué*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20542](https://doi.org/10.1609/aaai.v36i6.20542)

**Abstract**:

Conventional convolution neural networks (CNNs) trained on narrow Field-of-View (FoV) images are the state-of-the art approaches for object recognition tasks. Some methods proposed the adaptation of CNNs to ultra-wide FoV images by learning deformable kernels. However, they are limited by the Euclidean geometry and their accuracy degrades under strong distortions caused by fisheye projections. In this work, we demonstrate that learning the shape of convolution kernels in non-Euclidean spaces is better than existing deformable kernel methods. In particular, we propose a new approach that learns deformable kernel parameters (positions) in hyperbolic space. FisheyeHDK is a hybrid CNN architecture combining hyperbolic and Euclidean convolution layers for positions and features learning. First, we provide intuition of hyperbolic space for wide FoV images. Using synthetic distortion profiles, we demonstrate the effectiveness of our approach. We select two datasets - Cityscapes and BDD100K 2020 - of perspective images which we transform to fisheye equivalents at different scaling factors (analogue to focal lengths). Finally, we provide an experiment on data collected by a real fisheye camera. Validations and experiments show that our approach improves existing deformable kernel methods for CNN adaptation on fisheye images.

----

## [670] Distributed Learning with Strategic Users: A Repeated Game Approach

**Authors**: *Abdullah Basar Akbay, Junshan Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20543](https://doi.org/10.1609/aaai.v36i6.20543)

**Abstract**:

We consider a distributed learning setting where strategic users are incentivized by a fusion center, to train a learning model based on local data. The users are not obliged to provide their true gradient updates and the fusion center is not capable of validating the authenticity of reported updates. Thus motivated, we formulate the interactions between the fusion center and the users as repeated games, manifesting an under-explored interplay between machine learning and game theory. We then develop an incentive mechanism for the fusion center based on a joint gradient estimation and user action classification scheme, and study its impact on the convergence performance of distributed learning. Further, we devise adaptive zero-determinant (ZD) strategies, thereby generalizing the classical ZD strategies to the repeated games with time-varying stochastic errors. Theoretical and empirical analysis show that the fusion center can incentivize the strategic users to cooperate and report informative gradient updates, thus ensuring the convergence.

----

## [671] Private Rank Aggregation in Central and Local Models

**Authors**: *Daniel Alabi, Badih Ghazi, Ravi Kumar, Pasin Manurangsi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20544](https://doi.org/10.1609/aaai.v36i6.20544)

**Abstract**:

In social choice theory, (Kemeny) rank aggregation is a well-studied problem where the goal is to combine rankings from multiple voters into a single ranking on the same set of items. Since rankings can reveal preferences of voters (which a voter might like to keep private), it is important to aggregate preferences in such a way to preserve privacy. In this work, we present differentially private algorithms for rank aggregation in the pure and approximate settings along with distribution-independent utility upper and lower bounds. In addition to bounds in the central model, we also present utility bounds for the local model of differential privacy.

----

## [672] Combating Adversaries with Anti-adversaries

**Authors**: *Motasem Alfarra, Juan C. Pérez, Ali K. Thabet, Adel Bibi, Philip H. S. Torr, Bernard Ghanem*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20545](https://doi.org/10.1609/aaai.v36i6.20545)

**Abstract**:

Deep neural networks are vulnerable to small input perturbations known as adversarial attacks. Inspired by the fact that these adversaries are constructed by iteratively minimizing the confidence of a network for the true class label, we propose the anti-adversary layer, aimed at countering this effect. In particular, our layer generates an input perturbation in the opposite direction of the adversarial one and feeds the classifier a perturbed version of the input. Our approach is training-free and theoretically supported. We verify the effectiveness of our approach by combining our layer with both nominally and robustly trained models and conduct large-scale experiments from black-box to adaptive attacks on CIFAR10, CIFAR100, and ImageNet. Our layer significantly enhances model robustness while coming at no cost on clean accuracy.

----

## [673] DeformRS: Certifying Input Deformations with Randomized Smoothing

**Authors**: *Motasem Alfarra, Adel Bibi, Naeemullah Khan, Philip H. S. Torr, Bernard Ghanem*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20546](https://doi.org/10.1609/aaai.v36i6.20546)

**Abstract**:

Deep neural networks are vulnerable to input deformations in the form of vector fields of pixel displacements and to other parameterized geometric deformations e.g. translations, rotations, etc. Current input deformation certification methods either (i) do not scale to deep networks on large input datasets, or (ii) can only certify a specific class of deformations, e.g. only rotations. We reformulate certification in randomized smoothing setting for both general vector field and parameterized deformations and propose DeformRS-VF and DeformRS-Par, respectively. Our new formulation scales to large networks on large input datasets. For instance, DeformRS-Par certifies rich deformations, covering translations, rotations, scaling, affine deformations, and other visually aligned deformations such as ones parameterized by Discrete-Cosine-Transform basis. Extensive experiments on MNIST, CIFAR10, and ImageNet show competitive performance of DeformRS-Par achieving a certified accuracy of 39\% against perturbed rotations in the set [-10 degree, 10 degree] on ImageNet.

----

## [674] Latent Time Neural Ordinary Differential Equations

**Authors**: *Srinivas Anumasa, P. K. Srijith*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20547](https://doi.org/10.1609/aaai.v36i6.20547)

**Abstract**:

Neural ordinary differential equations (NODE) have been proposed as a continuous depth generalization to popular deep learning models such as Residual networks (ResNets). They provide parameter efficiency and automate the model selection process in deep learning models to some extent. However, they lack the much-required uncertainty modelling and robustness capabilities which are crucial for their use in several real-world applications such as autonomous driving and healthcare. We propose a novel and unique approach to model uncertainty in NODE by considering a distribution over the end-time T of the ODE solver. The proposed approach, latent time NODE (LT-NODE), treats T as a latent variable and apply Bayesian learning to obtain a posterior distribution over T from the data. In particular, we use variational inference to learn an approximate posterior and the model parameters. Prediction is done by considering the NODE representations from different samples of the posterior and can be done efficiently using a single forward pass. As T implicitly defines the depth of a NODE, posterior distribution over T would also help in model selection in NODE. We also propose, adaptive latent time NODE (ALT-NODE), which allow each data point to have a distinct posterior distribution over end-times. ALT-NODE uses amortized variational inference to learn an approximate posterior using inference networks. We demonstrate the effectiveness of the proposed approaches in modelling uncertainty and robustness through experiments on synthetic and several real-world image classification data.

----

## [675] Beyond GNNs: An Efficient Architecture for Graph Problems

**Authors**: *Pranjal Awasthi, Abhimanyu Das, Sreenivas Gollapudi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20548](https://doi.org/10.1609/aaai.v36i6.20548)

**Abstract**:

Despite their popularity for graph structured data, existing Graph Neural Networks (GNNs) have inherent limitations for fundamental graph problems such as shortest paths, k-connectivity, minimum spanning tree and minimum cuts. In these instances, it is known that one needs GNNs of high depth, scaling at a polynomial rate with the number of nodes n, to provably encode the solution space, in turn affecting their statistical efficiency.
 
 In this work we propose a new hybrid architecture to overcome this limitation. Our proposed architecture that we call as GNNplus networks involve a combination of multiple parallel low depth GNNs along with simple pooling layers involving low depth fully connected networks. We provably demonstrate that for many graph problems, the solution space can be encoded by GNNplus networks using depth that scales only poly-logarithmically in the number of nodes. This also has statistical advantages that we demonstrate via generalization bounds for GNNplus networks. We empirically show the effectiveness of our proposed architecture for a variety of graph problems and real world classification problems.

----

## [676] Programmatic Modeling and Generation of Real-Time Strategic Soccer Environments for Reinforcement Learning

**Authors**: *Abdus Salam Azad, Edward Kim, Qiancheng Wu, Kimin Lee, Ion Stoica, Pieter Abbeel, Alberto L. Sangiovanni-Vincentelli, Sanjit A. Seshia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20549](https://doi.org/10.1609/aaai.v36i6.20549)

**Abstract**:

The capability of a reinforcement learning (RL) agent heavily depends on the diversity of the learning scenarios generated by the environment. Generation of diverse realistic scenarios is challenging for real-time strategy (RTS) environments. The RTS environments are characterized by intelligent entities/non-RL agents cooperating and competing with the RL agents with large state and action spaces over a long period of time, resulting in an infinite space of feasible, but not necessarily realistic, scenarios involving complex interaction among different RL and non-RL agents. Yet, most of the existing simulators rely on randomly generating the environments based on predefined settings/layouts and offer limited flexibility and control over the environment dynamics for researchers to generate diverse, realistic scenarios as per their demand. To address this issue, for the first time, we formally introduce the benefits of adopting an existing formal scenario specification language, SCENIC, to assist researchers to model and generate diverse scenarios in an RTS environment in a flexible, systematic, and programmatic manner. To showcase the benefits, we interfaced SCENIC to an existing RTS environment Google Research Football (GRF) simulator and introduced a benchmark consisting of 32 realistic scenarios, encoded in SCENIC, to train RL agents and testing their generalization capabilities. We also show how researchers/RL practitioners can incorporate their domain knowledge to expedite the training process by intuitively modeling stochastic programmatic policies with SCENIC.

----

## [677] Admissible Policy Teaching through Reward Design

**Authors**: *Kiarash Banihashem, Adish Singla, Jiarui Gan, Goran Radanovic*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20550](https://doi.org/10.1609/aaai.v36i6.20550)

**Abstract**:

We study reward design strategies for incentivizing a reinforcement learning agent to adopt a policy from a set of admissible policies. The goal of the reward designer is to modify the underlying reward function cost-efficiently while ensuring that any approximately optimal deterministic policy under the new reward function is admissible and performs well under the original reward function. This problem can be viewed as a dual to the problem of optimal reward poisoning attacks: instead of forcing an agent to adopt a specific policy, the reward designer incentivizes an agent to avoid taking actions that are inadmissible in certain states. Perhaps surprisingly, and in contrast to the problem of optimal reward poisoning attacks, we first show that the reward design problem for admissible policy teaching is computationally challenging, and it is NP-hard to find an approximately optimal reward modification. We then proceed by formulating a surrogate problem whose optimal solution approximates the optimal solution to the reward design problem in our setting, but is more amenable to optimization techniques and analysis. For this surrogate problem, we present characterization results that provide bounds on the value of the optimal solution. Finally, we design a local search algorithm to solve the surrogate problem and showcase its utility using simulation-based experiments.

----

## [678] Entropy-Based Logic Explanations of Neural Networks

**Authors**: *Pietro Barbiero, Gabriele Ciravegna, Francesco Giannini, Pietro Lió, Marco Gori, Stefano Melacci*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20551](https://doi.org/10.1609/aaai.v36i6.20551)

**Abstract**:

Explainable artificial intelligence has rapidly emerged since lawmakers have started requiring interpretable models for safety-critical domains. Concept-based neural networks have arisen as explainable-by-design methods as they leverage human-understandable symbols (i.e. concepts) to predict class memberships. However, most of these approaches focus on the identification of the most relevant concepts but do not provide concise, formal explanations of how such concepts are leveraged by the classifier to make predictions. In this paper, we propose a novel end-to-end differentiable approach enabling the extraction of logic explanations from neural networks using the formalism of First-Order Logic. The method relies on an entropy-based criterion which automatically identifies the most relevant concepts. We consider four different case studies to demonstrate that: (i) this entropy-based criterion enables the distillation of concise logic explanations in safety-critical domains from clinical data to computer vision; (ii) the proposed approach outperforms state-of-the-art white-box models in terms of classification accuracy.

----

## [679] Training Robust Deep Models for Time-Series Domain: Novel Algorithms and Theoretical Analysis

**Authors**: *Taha Belkhouja, Yan Yan, Janardhan Rao Doppa*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20552](https://doi.org/10.1609/aaai.v36i6.20552)

**Abstract**:

Despite the success of deep neural networks (DNNs) for real-world applications over time-series data such as mobile health, little is known about how to train robust DNNs for time-series domain due to its unique characteristics compared to images and text data. In this paper, we fill this gap by proposing a novel algorithmic framework referred as RObust Training for Time-Series (RO-TS) to create robust deep models for time-series classification tasks. Specifically, we formulate a min-max optimization problem over the model parameters by explicitly reasoning about the robustness criteria in terms of additive perturbations to time-series inputs measured by the global alignment kernel (GAK) based distance. 
We also show the generality and advantages of our formulation using the summation structure over time-series alignments by relating both GAK and dynamic time warping (DTW).
This problem is an instance of a family of compositional min-max optimization problems, which are challenging and open with unclear theoretical guarantee. 
We propose a principled stochastic compositional alternating gradient descent ascent (SCAGDA) algorithm for this family of optimization problems. Unlike traditional methods for time-series that require approximate computation of distance measures, SCAGDA approximates the GAK based distance on-the-fly using a moving average approach. We theoretically analyze the convergence rate of SCAGDA and provide strong theoretical support for the estimation of GAK based distance.
Our experiments on real-world benchmarks demonstrate that RO-TS creates more robust deep models when compared to adversarial training using prior methods that rely on data augmentation or new definitions of loss functions. We also demonstrate the importance of GAK for time-series data over the Euclidean distance.

----

## [680] A Fast Algorithm for PAC Combinatorial Pure Exploration

**Authors**: *Noa Ben-David, Sivan Sabato*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20553](https://doi.org/10.1609/aaai.v36i6.20553)

**Abstract**:

We consider the problem of Combinatorial Pure Exploration (CPE), which deals with finding a combinatorial set of arms with a high reward, when the rewards of individual arms are unknown in advance and must be estimated using arm pulls. Previous algorithms for this problem, while obtaining sample complexity reductions in many cases, are highly computationally intensive, thus making them impractical even for mildly large problems. In this work, we propose a new CPE algorithm in the PAC setting, which is computationally light weight, and so can easily be applied to problems with tens of thousands of arms. This is achieved since the proposed algorithm requires a very small number of combinatorial oracle calls. The algorithm is based on successive acceptance of arms, along with elimination which is based on the combinatorial structure of the problem. We provide sample complexity guarantees for our algorithm, and demonstrate in experiments its usefulness on large problems, whereas previous algorithms are impractical to run on problems of even a few dozen arms. The code is provided at https://github.com/noabdavid/csale. The full version of this paper is available at https://arxiv.org/abs/2112.04197.

----

## [681] Modeling Attrition in Recommender Systems with Departing Bandits

**Authors**: *Omer Ben-Porat, Lee Cohen, Liu Leqi, Zachary C. Lipton, Yishay Mansour*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20554](https://doi.org/10.1609/aaai.v36i6.20554)

**Abstract**:

Traditionally, when recommender systems are formalized as multi-armed bandits, the policy of the recommender system influences the rewards accrued, but not the length of interaction. However, in real-world systems, dissatisfied users may depart (and never come back). In this work, we propose a novel multi-armed bandit setup that captures such policy-dependent horizons. Our setup consists of a finite set of user types, and multiple arms with Bernoulli payoffs.
Each (user type, arm) tuple corresponds to an (unknown) reward probability. Each user's type is initially unknown 
and can only be inferred through their response to recommendations. Moreover, if a user is dissatisfied with their recommendation, they might depart the system. We first address the case where all users share the same type, 
demonstrating that a recent UCB-based algorithm is optimal. We then move forward to the more challenging case,
where users are divided among two types. While naive approaches cannot handle this setting, 
we provide an efficient learning algorithm that achieves O(sqrt(T)ln(T)) regret, where T is the number of users.

----

## [682] Federated Dynamic Sparse Training: Computing Less, Communicating Less, Yet Learning Better

**Authors**: *Sameer Bibikar, Haris Vikalo, Zhangyang Wang, Xiaohan Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20555](https://doi.org/10.1609/aaai.v36i6.20555)

**Abstract**:

Federated learning (FL) enables distribution of machine learning workloads from the cloud to resource-limited edge devices. Unfortunately, current deep networks remain not only too compute-heavy for inference and training on edge devices, but also too large for communicating updates over bandwidth-constrained networks. In this paper, we develop, implement, and experimentally validate a novel FL framework termed Federated Dynamic Sparse Training (FedDST) by which complex neural networks can be deployed and trained with substantially improved efficiency in both on-device computation and in-network communication. At the core of FedDST is a dynamic process that extracts and trains sparse sub-networks from the target full network. With this scheme, "two birds are killed with one stone:'' instead of full models, each client performs efficient training of its own sparse networks, and only sparse networks are transmitted between devices and the cloud. Furthermore, our results reveal that the dynamic sparsity during FL training more flexibly accommodates local heterogeneity in FL agents than the fixed, shared sparse masks. Moreover, dynamic sparsity naturally introduces an "in-time self-ensembling effect'' into the training dynamics, and improves the FL performance even over dense training. In a realistic and challenging non i.i.d. FL setting, FedDST consistently outperforms competing algorithms in our experiments: for instance, at any fixed upload data cap on non-iid CIFAR-10, it gains an impressive accuracy advantage of 10% over FedAvgM when given the same upload data cap; the accuracy gap remains 3% even when FedAvgM is given 2 times the upload data cap, further demonstrating efficacy of FedDST. Code is available at: https://github.com/bibikar/feddst.

----

## [683] Robust and Resource-Efficient Data-Free Knowledge Distillation by Generative Pseudo Replay

**Authors**: *Kuluhan Binici, Shivam Aggarwal, Nam Trung Pham, Karianto Leman, Tulika Mitra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20556](https://doi.org/10.1609/aaai.v36i6.20556)

**Abstract**:

Data-Free Knowledge Distillation (KD) allows knowledge transfer from a trained neural network (teacher) to a more compact one (student) in the absence of original training data. Existing works use a validation set to monitor the accuracy of the student over real data and report the highest performance throughout the entire process. However, validation data may not be available at distillation time either, making it infeasible to record the student snapshot that achieved the peak accuracy. Therefore, a practical data-free KD method should be robust and ideally provide monotonically increasing student accuracy during distillation. This is challenging because the student experiences knowledge degradation due to the distribution shift of the synthetic data. A straightforward approach to overcome this issue is to store and rehearse the generated samples periodically, which increases the memory footprint and creates privacy concerns. We propose to model the distribution of the previously observed synthetic samples with a generative network. In particular, we design a Variational Autoencoder (VAE) with a training objective that is customized to learn the synthetic data representations optimally. The student is rehearsed by the generative pseudo replay technique, with samples produced by the VAE. Hence knowledge degradation can be prevented without storing any samples. Experiments on image classification benchmarks show that our method optimizes the expected value of the distilled model accuracy while eliminating the large memory overhead incurred by the sample-storing methods.

----

## [684] ErfAct and Pserf: Non-monotonic Smooth Trainable Activation Functions

**Authors**: *Koushik Biswas, Sandeep Kumar, Shilpak Banerjee, Ashish Kumar Pandey*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20557](https://doi.org/10.1609/aaai.v36i6.20557)

**Abstract**:

An activation function is a crucial component of a neural network that introduces non-linearity in the network. The state-of-the-art performance of a neural network depends also on the perfect choice of an activation function. We propose two novel non-monotonic smooth trainable activation functions, called ErfAct and Pserf. Experiments suggest that the proposed functions improve the network performance significantly compared to the widely used activations like ReLU, Swish, and Mish. Replacing ReLU by ErfAct and Pserf, we have 5.68% and 5.42% improvement for top-1 accuracy on Shufflenet V2 (2.0x) network in CIFAR100 dataset, 2.11% and 1.96% improvement for top-1 accuracy on Shufflenet V2 (2.0x) network in CIFAR10 dataset, 1.0%, and 1.0% improvement on mean average precision (mAP) on SSD300 model in Pascal VOC dataset.

----

## [685] Feedback Gradient Descent: Efficient and Stable Optimization with Orthogonality for DNNs

**Authors**: *Fanchen Bu, Dong Eui Chang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20558](https://doi.org/10.1609/aaai.v36i6.20558)

**Abstract**:

The optimization with orthogonality has been shown useful in training deep neural networks (DNNs). 
 To impose orthogonality on DNNs, both computational efficiency and stability are important.
 However, existing methods utilizing Riemannian optimization or hard constraints can only ensure stability while those using soft constraints can only improve efficiency.
 In this paper, we propose a novel method, named Feedback Gradient Descent (FGD), to our knowledge, the first work showing high efficiency and stability simultaneously.
 FGD induces orthogonality based on the simple yet indispensable Euler discretization of a continuous-time dynamical system on the tangent bundle of the Stiefel manifold.
 In particular, inspired by a numerical integration method on manifolds called Feedback Integrators, we propose to instantiate it on the tangent bundle of the Stiefel manifold for the first time.
 In our extensive image classification experiments, FGD comprehensively outperforms the existing state-of-the-art methods in terms of accuracy, efficiency, and stability.

----

## [686] Breaking the Convergence Barrier: Optimization via Fixed-Time Convergent Flows

**Authors**: *Param Budhraja, Mayank Baranwal, Kunal Garg, Ashish R. Hota*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20559](https://doi.org/10.1609/aaai.v36i6.20559)

**Abstract**:

Accelerated gradient methods are the cornerstones of large-scale, data-driven optimization problems that arise naturally in machine learning and other fields concerning data analysis. We introduce a gradient-based optimization framework for achieving acceleration, based on the recently introduced notion of fixed-time stability of dynamical systems. The method presents itself as a generalization of simple gradient-based methods suitably scaled to achieve convergence to the optimizer in a fixed-time, independent of the initialization. We achieve this by first leveraging a continuous-time framework for designing fixed-time stable dynamical systems, and later providing a consistent discretization strategy, such that the equivalent discrete-time algorithm tracks the optimizer in a practically fixed number of iterations. We also provide a theoretical analysis of the convergence behavior of the proposed gradient flows, and their robustness to additive disturbances for a range of functions obeying strong convexity, strict convexity, and possibly nonconvexity but satisfying the Polyak-Łojasiewicz inequality. We also show that the regret bound on the convergence rate is constant by virtue of the fixed-time convergence. The hyperparameters have intuitive interpretations and can be tuned to fit the requirements on the desired convergence rates. We validate the accelerated convergence properties of the proposed schemes on a range of numerical examples against the state-of-the-art optimization algorithms. Our work provides insights on developing novel optimization algorithms via discretization of continuous-time flows.

----

## [687] Shrub Ensembles for Online Classification

**Authors**: *Sebastian Buschjäger, Sibylle Hess, Katharina Morik*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20560](https://doi.org/10.1609/aaai.v36i6.20560)

**Abstract**:

Online learning algorithms have become a ubiquitous tool in the machine learning toolbox and are frequently used in small, resource-constraint environments. Among the most successful online learning methods are Decision Tree (DT) ensembles. DT ensembles provide excellent performance while adapting to changes in the data, but they are not resource efficient. Incremental tree learners keep adding new nodes to the tree but never remove old ones increasing the memory consumption over time. Gradient-based tree learning, on the other hand, requires the computation of gradients over the entire tree which is costly for even moderately sized trees.
 In this paper, we propose a novel memory-efficient online classification ensemble called shrub ensembles for resource-constraint systems. Our algorithm trains small to medium-sized decision trees on small windows and uses stochastic proximal gradient descent to learn the ensemble weights of these `shrubs'. We provide a theoretical analysis of our algorithm and include an extensive discussion on the behavior of our approach in the online setting. In a series of 2~959 experiments on 12 different datasets, we compare our method against 8 state-of-the-art methods. Our Shrub Ensembles retain an excellent performance even when only little memory is available. We show that SE offers a better accuracy-memory trade-off in 7 of 12 cases, while having a statistically significant better performance than most other methods. Our implementation is available under https://github.com/sbuschjaeger/se-online .

----

## [688] NoiseGrad - Enhancing Explanations by Introducing Stochasticity to Model Weights

**Authors**: *Kirill Bykov, Anna Hedström, Shinichi Nakajima, Marina M.-C. Höhne*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20561](https://doi.org/10.1609/aaai.v36i6.20561)

**Abstract**:

Many efforts have been made for revealing the decision-making process of black-box learning machines such as deep neural networks, resulting in useful local and global explanation methods. For local explanation, stochasticity is known to help: a simple method, called SmoothGrad, has improved the visual quality of gradient-based attribution by adding noise to the input space and averaging the explanations of the noisy inputs. In this paper, we extend this idea and propose NoiseGrad that enhances both local and global explanation methods. Specifically, NoiseGrad introduces stochasticity in the weight parameter space, such that the decision boundary is perturbed. NoiseGrad is expected to enhance the local explanation, similarly to SmoothGrad, due to the dual relationship between the input perturbation and the decision boundary perturbation. We evaluate NoiseGrad and its fusion with SmoothGrad - FusionGrad - qualitatively and quantitatively with several evaluation criteria, and show that our novel approach significantly outperforms the baseline methods. Both NoiseGrad and FusionGrad are method-agnostic and as handy as SmoothGrad using a simple heuristic for the choice of the hyperparameter setting without the need of fine-tuning.

----

## [689] Leaping through Time with Gradient-Based Adaptation for Recommendation

**Authors**: *Nuttapong Chairatanakul, Hoang NT, Xin Liu, Tsuyoshi Murata*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20562](https://doi.org/10.1609/aaai.v36i6.20562)

**Abstract**:

Modern recommender systems are required to adapt to the change in user preferences and item popularity. Such a problem is known as the temporal dynamics problem, and it is one of the main challenges in recommender system modeling. Different from the popular recurrent modeling approach, we propose a new solution named LeapRec to the temporal dynamic problem by using trajectory-based meta-learning to model time dependencies. LeapRec characterizes temporal dynamics by two complement components named global time leap (GTL) and ordered time leap (OTL). By design, GTL learns long-term patterns by finding the shortest learning path across unordered temporal data. Cooperatively, OTL learns short-term patterns by considering the sequential nature of the temporal data. Our experimental results show that LeapRec consistently outperforms the state-of-the-art methods on several datasets and recommendation metrics. Furthermore, we provide an empirical study of the interaction between GTL and OTL, showing the effects of long- and short-term modeling.

----

## [690] Active Sampling for Text Classification with Subinstance Level Queries

**Authors**: *Shayok Chakraborty, Ankita Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20563](https://doi.org/10.1609/aaai.v36i6.20563)

**Abstract**:

Active learning algorithms are effective in identifying the salient and exemplar samples from large amounts of unlabeled data. This tremendously reduces the human annotation effort in inducing a machine learning model as only a few samples, which are identified by the algorithm, need to be labeled manually. In problem domains like text mining and video classification, human oracles peruse the data instances incrementally to derive an opinion about their class labels (such as reading a movie review progressively to assess its sentiment). In such applications, it is not necessary for the human oracles to review an unlabeled sample end-to-end in order to provide a label; it may be more efficient to identify an optimal subinstance size (percentage of the sample from the start) for each unlabeled sample, and request the human annotator to label the sample by analyzing only the subinstance, instead of the whole data sample. In this paper, we propose a novel framework to address this challenging problem, in an effort to further reduce the labeling burden on the human oracles and utilize the available labeling budget more efficiently. We pose the sample and subinstance size selection as a constrained optimization problem and derive a linear programming relaxation to select a batch of exemplar samples, together with the optimal subinstance size of each, which can potentially augment maximal information to the underlying classification model. Our extensive empirical studies on six challenging datasets from the text mining domain corroborate the practical usefulness of our framework over competing baselines.

----

## [691] A Unifying Theory of Thompson Sampling for Continuous Risk-Averse Bandits

**Authors**: *Joel Q. L. Chang, Vincent Y. F. Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20564](https://doi.org/10.1609/aaai.v36i6.20564)

**Abstract**:

This paper unifies the design and the analysis of risk-averse Thompson sampling algorithms for the multi-armed bandit problem for a class of risk functionals ρ that are continuous and dominant. We prove generalised concentration bounds for these continuous and dominant risk functionals and show that a wide class of popular risk functionals belong to this class. Using our newly developed analytical toolkits, we analyse the algorithm ρ-MTS (for multinomial distributions) and prove that they admit asymptotically optimal regret bounds of risk-averse algorithms under the CVaR, proportional hazard, and other ubiquitous risk measures. More generally, we prove the asymptotic optimality of ρ-MTS for Bernoulli distributions for a class of risk measures known as empirical distribution performance measures (EDPMs); this includes the well-known mean-variance. Numerical simulations show that the regret bounds incurred by our algorithms are reasonably tight vis-à-vis algorithm-independent lower bounds.

----

## [692] Locally Private k-Means Clustering with Constant Multiplicative Approximation and Near-Optimal Additive Error

**Authors**: *Anamay Chaturvedi, Matthew Jones, Huy Le Nguyen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20565](https://doi.org/10.1609/aaai.v36i6.20565)

**Abstract**:

Given a data set of size n in d'-dimensional Euclidean space, the k-means problem asks for a set of k points (called centers) such that the sum of the l_2^2-distances between the data points and the set of centers is minimized. Previous work on this problem in the local differential privacy setting shows how to achieve multiplicative approximation factors arbitrarily close to optimal, but suffers high additive error. The additive error has also been seen to be an issue in implementations of differentially private k-means clustering algorithms in both the central and local settings. In this work, we introduce a new locally private k-means clustering algorithm that achieves near-optimal additive error whilst retaining constant multiplicative approximation factors and round complexity. Concretely, given any c>√2, our algorithm achieves O(k^(1 + O(1/(2c^2-1))) √(d' n) log d' poly log n) additive error with an O(c^2) multiplicative approximation factor.

----

## [693] Safe Online Convex Optimization with Unknown Linear Safety Constraints

**Authors**: *Sapana Chaudhary, Dileep M. Kalathil*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20566](https://doi.org/10.1609/aaai.v36i6.20566)

**Abstract**:

We study the problem of safe online convex optimization, where the action at each time step must satisfy a set of linear safety constraints. The goal is to select a sequence of actions to minimize the regret without violating the safety constraints at any time step (with high probability). The parameters that specify the linear safety constraints are unknown to the algorithm. The algorithm has access to only the noisy observations of constraints for the chosen actions. We propose an algorithm, called the Safe Online Projected Gradient Descent (SO-PGD) algorithm, to address this problem. We show that, under the assumption of availability of a safe baseline action, the SO-PGD algorithm achieves a regret O(T^{2/3}). While there are many algorithms for online convex optimization (OCO) problems with safety constraints available in the literature, they allow constraint violations during learning/optimization, and the focus has been on characterizing the cumulative constraint violations. To the best of our knowledge, ours is the first work that provides an algorithm with provable guarantees on the regret, without violating the linear safety constraints (with high probability) at any time step.

----

## [694] Deconvolutional Density Network: Modeling Free-Form Conditional Distributions

**Authors**: *Bing Chen, Mazharul Islam, Jisuo Gao, Lin Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20567](https://doi.org/10.1609/aaai.v36i6.20567)

**Abstract**:

Conditional density estimation (CDE) is the task of estimating the probability of an event conditioned on some inputs. A neural network (NN) can also be used to compute the output distribution for continuous-domain, which can be viewed as an extension of regression task. Nevertheless, it is difficult to explicitly approximate a distribution without knowing the information of its general form a priori. In order to fit an arbitrary conditional distribution, discretizing the continuous domain into bins is an effective strategy, as long as we have sufficiently narrow bins and very large data. However, collecting enough data is often hard to reach and falls far short of that ideal in many circumstances, especially in multivariate CDE for the curse of dimensionality. In this paper, we demonstrate the benefits of modeling free-form conditional distributions using a deconvolution-based neural net framework, coping with data deficiency problems in discretization. It has the advantage of being flexible but also takes advantage of the hierarchical smoothness offered by the deconvolution layers. We compare our method to a number of other density-estimation approaches and show that our Deconvolutional Density Network (DDN) outperforms the competing methods on many univariate and multivariate tasks. The code of DDN is available at https://github.com/NBICLAB/DDN

----

## [695] Multiscale Generative Models: Improving Performance of a Generative Model Using Feedback from Other Dependent Generative Models

**Authors**: *Changyu Chen, Avinandan Bose, Shih-Fen Cheng, Arunesh Sinha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20568](https://doi.org/10.1609/aaai.v36i6.20568)

**Abstract**:

Realistic fine-grained multi-agent simulation of real-world complex systems is crucial for many downstream tasks such as reinforcement learning. Recent work has used generative models (GANs in particular) for providing high-fidelity simulation of real-world systems. However, such generative models are often monolithic and miss out on modeling the interaction in multi-agent systems. In this work, we take a first step towards building multiple interacting generative models (GANs) that reflects the interaction in real world. We build and analyze a hierarchical set-up where a higher-level GAN is conditioned on the output of multiple lower-level GANs. We present a technique of using feedback from the higher-level GAN to improve performance of lower-level GANs. We mathematically characterize the conditions under which our technique is impactful, including understanding the transfer learning nature of our set-up. We present three distinct experiments on synthetic data, time series data, and image domain, revealing the wide applicability of our technique.

----

## [696] Simultaneously Learning Stochastic and Adversarial Bandits under the Position-Based Model

**Authors**: *Cheng Chen, Canzhe Zhao, Shuai Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20569](https://doi.org/10.1609/aaai.v36i6.20569)

**Abstract**:

Online learning to rank (OLTR) interactively learns to choose lists of items from a large collection based on certain click models that describe users' click behaviors. Most recent works for this problem focus on the stochastic environment where the item attractiveness is assumed to be invariant during the learning process. In many real-world scenarios, however, the environment could be dynamic or even arbitrarily changing. This work studies the OLTR problem in both stochastic and adversarial environments under the position-based model (PBM). We propose a method based on the follow-the-regularized-leader (FTRL) framework with Tsallis entropy and develop a new self-bounding constraint especially designed for PBM. We prove the proposed algorithm simultaneously achieves O(log T) regret in the stochastic environment and O(m√nT) regret in the adversarial environment, where T is the number of rounds, n is the number of items and m is the number of positions. We also provide a lower bound of order Ω(m√nT) for adversarial PBM, which matches our upper bound and improves over the state-of-the-art lower bound. The experiments show that our algorithm could simultaneously learn in both stochastic and adversarial environments and is competitive compared to existing methods that are designed for a single environment.

----

## [697] Clustering Interval-Censored Time-Series for Disease Phenotyping

**Authors**: *Irene Y. Chen, Rahul G. Krishnan, David A. Sontag*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20570](https://doi.org/10.1609/aaai.v36i6.20570)

**Abstract**:

Unsupervised learning is often used to uncover clusters in data. However, different kinds of noise may impede the discovery of useful patterns from real-world time-series data. In this work, we focus on mitigating the interference of interval censoring in the task of clustering for disease phenotyping. We develop a deep generative, continuous-time model of time-series data that clusters time-series while correcting for censorship time. We provide conditions under which clusters and the amount of delayed entry may be identified from data under a noiseless model. On synthetic data, we demonstrate accurate, stable, and interpretable results that outperform several benchmarks. On real-world clinical datasets of heart failure and Parkinson's disease patients, we study how interval censoring can adversely affect the task of disease phenotyping. Our model corrects for this source of error and recovers known clinical subtypes.

----

## [698] Efficient Robust Training via Backward Smoothing

**Authors**: *Jinghui Chen, Yu Cheng, Zhe Gan, Quanquan Gu, Jingjing Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20571](https://doi.org/10.1609/aaai.v36i6.20571)

**Abstract**:

Adversarial training is so far the most effective strategy in defending against adversarial examples. However, it suffers from high computational costs due to the iterative adversarial attacks in each training step. Recent studies show that it is possible to achieve fast Adversarial Training by performing a single-step attack with random initialization. However, such an approach still lags behind state-of-the-art adversarial training algorithms on both stability and model robustness. In this work, we develop a new understanding towards Fast Adversarial Training, by viewing random initialization as performing randomized smoothing for better optimization of the inner maximization problem. Following this new perspective, we also propose a new initialization strategy, backward smoothing, to further improve the stability and model robustness over single-step robust training methods.
 Experiments on multiple benchmarks demonstrate that our method achieves similar model robustness as the original TRADES method while using much less training time (~3x improvement with the same training schedule).

----

## [699] An Online Learning Approach to Sequential User-Centric Selection Problems

**Authors**: *Junpu Chen, Hong Xie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20572](https://doi.org/10.1609/aaai.v36i6.20572)

**Abstract**:

This paper proposes a new variant of multi-play MAB model, to capture important factors of the sequential user-centric selection problem arising from mobile edge computing, ridesharing applications, etc. In the proposed model, each arm is associated with 
discrete units of resources, each play is associate with movement costs and multiple plays can pull the same arm simultaneously. To learn the optimal action profile (an action profile prescribes the arm that each play pulls), there are two challenges: (1) the number of action profiles is large, i.e., M^K, where K and M denote the number of plays and arms respectively; (2) feedbacks on action profiles are not available, but instead feedbacks on some model parameters can be observed.  To address the first challenge, we formulate a completed weighted bipartite graph to capture key factors of the offline decision problem with given model parameters. We identify the correspondence between action profiles and a special class of matchings of the graph. We also identify a dominance structure of this class of matchings. This correspondence and dominance structure enable us to design an algorithm named OffOptActPrf to locate the optimal action efficiently. To address the second challenge, we design an OnLinActPrf algorithm. We design estimators for model parameters and use these estimators to design a Quasi-UCB index for each action profile. The OnLinActPrf uses OffOptActPrf as a subroutine to select the action profile with the largest Quasi-UCB index. We conduct extensive experiments to validate the efficiency of OnLinActPrf.

----

## [700] Better Parameter-Free Stochastic Optimization with ODE Updates for Coin-Betting

**Authors**: *Keyi Chen, John Langford, Francesco Orabona*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20573](https://doi.org/10.1609/aaai.v36i6.20573)

**Abstract**:

Parameter-free stochastic gradient descent (PFSGD) algorithms do not require setting learning rates while achieving optimal theoretical performance. In practical applications, however, there remains an empirical gap between tuned stochastic gradient descent (SGD) and PFSGD. In this paper, we close the empirical gap with a new parameter-free algorithm based on continuous-time Coin-Betting on truncated models. The new update is derived through the solution of an Ordinary Differential Equation (ODE) and solved in a closed form. We show empirically that this new parameter-free algorithm outperforms algorithms with the ``best default'' learning rates and almost matches the performance of finely tuned baselines without anything to tune.

----

## [701] Mutual Nearest Neighbor Contrast and Hybrid Prototype Self-Training for Universal Domain Adaptation

**Authors**: *Liang Chen, Qianjin Du, Yihang Lou, Jianzhong He, Tao Bai, Minghua Deng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20574](https://doi.org/10.1609/aaai.v36i6.20574)

**Abstract**:

Universal domain adaptation (UniDA) aims to transfer knowledge learned from a labeled source domain to an unlabeled target domain under domain shift and category shift. Without prior category overlap information, it is challenging to simultaneously align the common categories between two domains and separate their respective private categories. Additionally, previous studies utilize the source classifier's prediction to obtain various known labels and one generic "unknown" label of target samples. However, over-reliance on learned classifier knowledge is inevitably biased to source data, ignoring the intrinsic structure of target domain. Therefore, in this paper, we propose a novel two-stage UniDA framework called MATHS based on the principle of mutual nearest neighbor contrast and hybrid prototype discrimination. In the first stage, we design an efficient mutual nearest neighbor contrastive learning scheme to achieve feature alignment, which exploits the instance-level affinity relationship to uncover the intrinsic structure of two domains. We introduce a bimodality hypothesis for the maximum discriminative probability distribution to detect the possible target private samples, and present a data-based statistical approach to separate the common and private categories. In the second stage, to obtain more reliable label predictions, we propose an incremental pseudo-classifier for target data only, which is driven by the hybrid representative prototypes. A confidence-guided prototype contrastive loss is designed to optimize the category allocation uncertainty via a self-training mechanism. Extensive experiments on three benchmarks demonstrate that MATHS outperforms previous state-of-the-arts on most UniDA settings.

----

## [702] Evidential Neighborhood Contrastive Learning for Universal Domain Adaptation

**Authors**: *Liang Chen, Yihang Lou, Jianzhong He, Tao Bai, Minghua Deng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20575](https://doi.org/10.1609/aaai.v36i6.20575)

**Abstract**:

Universal domain adaptation (UniDA) aims to transfer the knowledge learned from a labeled source domain to an unlabeled target domain without any constraints on the label sets. However, domain shift and category shift make UniDA extremely challenging, mainly attributed to the requirement of identifying both shared “known” samples and private “unknown” samples. Previous methods barely exploit the intrinsic manifold structure relationship between two domains for feature alignment, and they rely on the softmax-based scores with class competition nature to detect underlying “unknown” samples. Therefore, in this paper, we propose a novel evidential neighborhood contrastive learning framework called TNT to address these issues. Specifically, TNT first proposes a new domain alignment principle: semantically consistent samples should be geometrically adjacent to each other, whether within or across domains. From this criterion, a cross-domain multi-sample contrastive loss based on mutual nearest neighbors is designed to achieve common category matching and private category separation. Second, toward accurate “unknown” sample detection, TNT introduces a class competition-free uncertainty score from the perspective of evidential deep learning. Instead of setting a single threshold, TNT learns a category-aware heterogeneous threshold vector to reject diverse “unknown” samples. Extensive experiments on three benchmarks demonstrate that TNT significantly outperforms previous state-of-the-art UniDA methods.

----

## [703] Zero Stability Well Predicts Performance of Convolutional Neural Networks

**Authors**: *Liangming Chen, Long Jin, Mingsheng Shang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20576](https://doi.org/10.1609/aaai.v36i6.20576)

**Abstract**:

The question of what kind of convolutional neural network (CNN) structure performs well is fascinating. In this work, we move toward the answer with one more step by connecting zero stability and model performance. Specifically, we found that if a discrete solver of an ordinary differential equation is zero stable, the CNN corresponding to that solver performs well. We first give the interpretation of zero stability in the context of deep learning and then investigate the performance of existing first- and second-order CNNs under different zero-stable circumstances. Based on the preliminary observation, we provide a higher-order discretization to construct CNNs and then propose a zero-stable network (ZeroSNet). To guarantee zero stability of the ZeroSNet, we first deduce a structure that meets consistency conditions and then give a zero stable region of a training-free parameter. By analyzing the roots of a characteristic equation, we theoretically obtain the optimal coefficients of feature maps. Empirically, we present our results from three aspects: We provide extensive empirical evidence of different depth on different datasets to show that the moduli of the characteristic equation's roots are the keys for the performance of CNNs that require historical features; Our experiments show that ZeroSNet outperforms existing CNNs which is based on high-order discretization; ZeroSNets show better robustness against noises on the input. The source code is available at https://github.com/logichen/ZeroSNet.

----

## [704] Semi-supervised Learning with Multi-Head Co-Training

**Authors**: *Mingcai Chen, Yuntao Du, Yi Zhang, Shuwei Qian, Chongjun Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20577](https://doi.org/10.1609/aaai.v36i6.20577)

**Abstract**:

Co-training, extended from self-training, is one of the frameworks for semi-supervised learning. Without natural split of features, single-view co-training works at the cost of training extra classifiers, where the algorithm should be delicately designed to prevent individual classifiers from collapsing into each other. To remove these obstacles which deter the adoption of single-view co-training, we present a simple and efficient algorithm Multi-Head Co-Training. By integrating base learners into a multi-head structure, the model is in a minimal amount of extra parameters. Every classification head in the unified model interacts with its peers through a “Weak and Strong Augmentation” strategy, in which the diversity is naturally brought by the strong data augmentation. Therefore, the proposed method facilitates single-view co-training by 1). promoting diversity implicitly and 2). only requiring a small extra computational overhead. The effectiveness of Multi-Head Co-Training is demonstrated in an empirical study on standard semi-supervised learning benchmarks.

----

## [705] Instance Selection: A Bayesian Decision Theory Perspective

**Authors**: *Qingqiang Chen, Fuyuan Cao, Ying Xing, Jiye Liang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20578](https://doi.org/10.1609/aaai.v36i6.20578)

**Abstract**:

In this paper, we consider the problem of lacking theoretical foundation and low execution efficiency of the instance selection methods based on the k-nearest neighbour rule when processing large-scale data. We point out that the core idea of these methods can be explained from the perspective of Bayesian decision theory, that is, to find which instances are reducible, irreducible, and deleterious. Then, based on the percolation theory, we establish the relationship between these three types of instances and local homogeneous cluster (i.e., a set of instances with the same labels). Finally, we propose a method based on an accelerated k-means algorithm to construct local homogeneous clusters and remove the superfluous instances. The performance of our method is studied on extensive synthetic and benchmark data sets. Our proposed method can handle large-scale data more effectively than the state-of-the-art instance selection methods. All code and data results are available at https://github.com/CQQXY161120/Instance-Selection.

----

## [706] Input-Specific Robustness Certification for Randomized Smoothing

**Authors**: *Ruoxin Chen, Jie Li, Junchi Yan, Ping Li, Bin Sheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20579](https://doi.org/10.1609/aaai.v36i6.20579)

**Abstract**:

Although randomized smoothing has demonstrated high certified robustness and superior scalability to other certified defenses, the high computational overhead of the robustness certification bottlenecks the practical applicability, as it depends heavily on the large sample approximation for estimating the confidence interval. In existing works, the sample size for the confidence interval is universally set and agnostic to the input for prediction. This Input-Agnostic Sampling (IAS) scheme may yield a poor Average Certified Radius (ACR)-runtime trade-off which calls for improvement. In this paper, we propose Input-Specific Sampling (ISS) acceleration to achieve the cost-effectiveness for robustness certification, in an adaptive way of reducing the sampling size based on the input characteristic. Furthermore, our method universally controls the certified radius decline from the ISS sample size reduction. The empirical results on CIFAR-10 and ImageNet show that ISS can speed up the certification by more than three times at a limited cost of 0.05 certified radius. Meanwhile, ISS surpasses IAS on the average certified radius across the extensive hyperparameter settings. Specifically, ISS achieves ACR=0.958 on ImageNet in 250 minutes, compared to ACR=0.917 by IAS under the same condition. We release our code in https://github.com/roy-ch/Input-Specific-Certification.

----

## [707] Multimodal Adversarially Learned Inference with Factorized Discriminators

**Authors**: *Wenxue Chen, Jianke Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20580](https://doi.org/10.1609/aaai.v36i6.20580)

**Abstract**:

Learning from multimodal data is an important research topic in machine learning, which has the potential to obtain better representations. In this work, we propose a novel approach to generative modeling of multimodal data based on generative adversarial networks. To learn a coherent multimodal generative model, we show that it is necessary to align different encoder distributions with the joint decoder distribution simultaneously. To this end, we construct a specific form of the discriminator to enable our model to utilize data efficiently, which can be trained constrastively. By taking advantage of contrastive learning through factorizing the discriminator, we train our model on unimodal data. We have conducted experiments on the benchmark datasets, whose promising results show that our proposed approach outperforms the-state-ofthe-art methods on a variety of metrics. The source code is publicly available at https://github.com/6b5d/mmali.

----

## [708] Imbalance-Aware Uplift Modeling for Observational Data

**Authors**: *Xuanying Chen, Zhining Liu, Li Yu, Liuyi Yao, Wenpeng Zhang, Yi Dong, Lihong Gu, Xiaodong Zeng, Yize Tan, Jinjie Gu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20581](https://doi.org/10.1609/aaai.v36i6.20581)

**Abstract**:

Uplift modeling aims to model the incremental impact of a treatment on an individual outcome, which has attracted great interests of researchers and practitioners from different communities. Existing uplift modeling methods rely on either the data collected from randomized controlled trials (RCTs) or the observational data which is more realistic. However, we notice that on the observational data, it is often the case that only a small number of subjects receive treatment, but finally infer the uplift on a much large group of subjects. Such highly imbalanced data is common in various fields such as marketing and medical treatment but it is rarely handled by existing works. In this paper, we theoretically and quantitatively prove that the existing representative methods, transformed outcome (TOM) and doubly robust (DR), suffer from large bias and deviation on highly imbalanced datasets with skewed propensity scores, mainly because they are proportional to the reciprocal of the propensity score. To reduce the bias and deviation of uplift modeling with an imbalanced dataset, we propose an imbalance-aware uplift modeling (IAUM) method via constructing a robust proxy outcome, which adaptively combines the doubly robust estimator and the imputed treatment effects based on the propensity score. We theoretically prove that IAUM can obtain a better bias-variance trade-off than existing methods on a highly imbalanced dataset. We conduct extensive experiments on a synthetic dataset and two real-world datasets, and the experimental results well demonstrate the superiority of our method over state-of-the-art.

----

## [709] KAM Theory Meets Statistical Learning Theory: Hamiltonian Neural Networks with Non-zero Training Loss

**Authors**: *Yuhan Chen, Takashi Matsubara, Takaharu Yaguchi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20582](https://doi.org/10.1609/aaai.v36i6.20582)

**Abstract**:

Many physical phenomena are described by Hamiltonian mechanics using an energy function (Hamiltonian). Recently, the Hamiltonian neural network, which approximates the Hamiltonian by a neural network, and its extensions have attracted much attention. This is a very powerful method, but theoretical studies are limited. In this study, by combining the statistical learning theory and KAM theory, we provide a theoretical analysis of the behavior of Hamiltonian neural networks when the learning error is not completely zero. A Hamiltonian neural network with non-zero errors can be considered as a perturbation from the true dynamics, and the perturbation theory of the Hamilton equation is widely known as KAM theory. To apply KAM theory, we provide a generalization error bound for Hamiltonian neural networks by deriving an estimate of the covering number of the gradient of the multi-layer perceptron, which is the key ingredient of the model. This error bound gives a sup-norm bound on the Hamiltonian that is required in the application of KAM theory.

----

## [710] BScNets: Block Simplicial Complex Neural Networks

**Authors**: *Yuzhou Chen, Yulia R. Gel, H. Vincent Poor*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20583](https://doi.org/10.1609/aaai.v36i6.20583)

**Abstract**:

Simplicial neural networks (SNNs) have recently emerged as a new direction in graph learning which expands the idea of convolutional architectures from node space to simplicial complexes on graphs. Instead of predominantly assessing pairwise relations among nodes as in the current practice, simplicial complexes allow us to describe higher-order interactions and multi-node graph structures. By building upon connection between the convolution operation and the new block Hodge-Laplacian, we propose the first SNN for link prediction. Our new Block Simplicial Complex Neural Networks (BScNets) model generalizes existing graph convolutional network (GCN) frameworks by systematically incorporating salient interactions among multiple higher-order graph structures of different dimensions. We discuss theoretical foundations behind BScNets and illustrate its utility for link prediction on eight real-world and synthetic datasets. Our experiments indicate that BScNets outperforms the state-of-the-art models by a significant margin while maintaining low computation costs. Finally, we show utility of BScNets as a new promising alternative for tracking spread of infectious diseases such as COVID-19 and measuring the effectiveness of the healthcare risk mitigation strategies.

----

## [711] ASM2TV: An Adaptive Semi-supervised Multi-Task Multi-View Learning Framework for Human Activity Recognition

**Authors**: *Zekai Chen, Xiao Zhang, Xiuzhen Cheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20584](https://doi.org/10.1609/aaai.v36i6.20584)

**Abstract**:

Many real-world scenarios, such as human activity recognition (HAR) in IoT, can be formalized as a multi-task multi-view learning problem. Each specific task consists of multiple shared feature views collected from multiple sources, either homogeneous or heterogeneous. Common among recent approaches is to employ a typical hard/soft sharing strategy at the initial phase separately for each view across tasks to uncover common knowledge, underlying the assumption that all views are conditionally independent. On the one hand, multiple views across tasks possibly relate to each other under practical situations. On the other hand, supervised methods might be insufficient when labeled data is scarce. To tackle these challenges, we introduce a novel framework ASM2TV for semi-supervised multi-task multi-view learning. We present a new perspective named gating control policy, a learnable task-view-interacted sharing policy that adaptively selects the most desirable candidate shared block for any view across any task, which uncovers more fine-grained task-view-interacted relatedness and improves inference efficiency. Significantly, our proposed gathering consistency adaption procedure takes full advantage of large amounts of unlabeled fragmented time-series, making it a general framework that accommodates a wide range of applications. Experiments on two diverse real-world HAR benchmark datasets collected from various subjects and sources demonstrate our framework's superiority over other state-of-the-arts. Anonymous codes are available at https://github.com/zachstarkk/ASM2TV.

----

## [712] Identification of Linear Latent Variable Model with Arbitrary Distribution

**Authors**: *Zhengming Chen, Feng Xie, Jie Qiao, Zhifeng Hao, Kun Zhang, Ruichu Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20585](https://doi.org/10.1609/aaai.v36i6.20585)

**Abstract**:

An important problem across multiple disciplines is to infer and understand meaningful latent variables. One strategy commonly used is to model the measured variables in terms of the latent variables under suitable assumptions on the connectivity from the latents to the measured (known as measurement model). Furthermore, it might be even more interesting to discover the causal relations among the latent variables (known as structural model). Recently, some methods have been proposed to estimate the structural model by assuming that the noise terms in the measured and latent variables are non-Gaussian. However, they are not suitable when some of the noise terms become Gaussian. To bridge this gap, we investigate the problem of identification of the structural model with arbitrary noise distributions. We provide necessary and sufficient condition under which the structural model is identifiable: it is identifiable iff for each pair of adjacent latent variables Lx, Ly, (1) at least one of Lx and Ly has non-Gaussian noise, or (2) at least one of them has a non-Gaussian ancestor and is not d-separated from the non-Gaussian component of this ancestor by the common causes of Lx and Ly. This identifiability result relaxes the non-Gaussianity requirements to only a (hopefully small) subset of variables, and accordingly elegantly extends the application scope of the structural model. Based on the above identifiability result, we further propose a practical algorithm to learn the structural model. We verify the correctness of the identifiability result and the effectiveness of the proposed method through empirical studies.

----

## [713] DPNAS: Neural Architecture Search for Deep Learning with Differential Privacy

**Authors**: *Anda Cheng, Jiaxing Wang, Xi Sheryl Zhang, Qiang Chen, Peisong Wang, Jian Cheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20586](https://doi.org/10.1609/aaai.v36i6.20586)

**Abstract**:

Training deep neural networks (DNNs) for meaningful differential privacy (DP) guarantees severely degrades model utility. In this paper, we demonstrate that the architecture of DNNs has a significant impact on model utility in the context of private deep learning, whereas its effect is largely unexplored in previous studies. In light of this missing, we propose the very first framework that employs neural architecture search to automatic model design for private deep learning, dubbed as DPNAS. To integrate private learning with architecture search, a DP-aware approach is introduced for training candidate models composed on a delicately defined novel search space. We empirically certify the effectiveness of the proposed framework. The searched model DPNASNet achieves state-of-the-art privacy/utility trade-offs, e.g., for the privacy budget of (epsilon, delta)=(3, 1e-5), our model obtains test accuracy of 98.57% on MNIST, 88.09% on FashionMNIST, and 68.33% on CIFAR-10. Furthermore, by studying the generated architectures, we provide several intriguing findings of designing private-learning-friendly DNNs, which can shed new light on model design for deep learning with differential privacy.

----

## [714] Graph Neural Controlled Differential Equations for Traffic Forecasting

**Authors**: *Jeongwhan Choi, Hwangyong Choi, Jeehyun Hwang, Noseong Park*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20587](https://doi.org/10.1609/aaai.v36i6.20587)

**Abstract**:

Traffic forecasting is one of the most popular spatio-temporal tasks in the field of machine learning. A prevalent approach in the field is to combine graph convolutional networks and recurrent neural networks for the spatio-temporal processing. There has been fierce competition and many novel methods have been proposed. In this paper, we present the method of spatio-temporal graph neural controlled differential equation (STG-NCDE). Neural controlled differential equations (NCDEs) are a breakthrough concept for processing sequential data. We extend the concept and design two NCDEs: one for the temporal processing and the other for the spatial processing. After that, we combine them into a single framework. We conduct experiments with 6 benchmark datasets and 20 baselines. STG-NCDE shows the best accuracy in all cases, outperforming all those 20 baselines by non-trivial margins.

----

## [715] Differentially Private Regret Minimization in Episodic Markov Decision Processes

**Authors**: *Sayak Ray Chowdhury, Xingyu Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20588](https://doi.org/10.1609/aaai.v36i6.20588)

**Abstract**:

We study regret minimization in finite horizon tabular Markov decision processes (MDPs) under the constraints of differential privacy (DP). This is motivated by the widespread applications of reinforcement learning (RL) in real-world sequential decision making problems, where protecting users' sensitive and private information is becoming paramount. We consider two variants of DP -- joint DP (JDP), where a centralized agent is responsible for protecting users' sensitive data and local DP (LDP), where information needs to be protected directly on the user side. We first propose two general frameworks -- one for policy optimization and another for value iteration -- for designing private, optimistic RL algorithms. We then instantiate these frameworks with suitable privacy mechanisms to satisfy JDP and LDP requirements, and simultaneously obtain sublinear regret guarantees. The regret bounds show that under JDP, the cost of privacy is only a lower order additive term, while for a stronger privacy protection under LDP, the cost suffered is multiplicative. Finally, the regret bounds are obtained by a unified analysis, which, we believe, can be extended beyond tabular MDPs.

----

## [716] Learning by Competition of Self-Interested Reinforcement Learning Agents

**Authors**: *Stephen Chung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20589](https://doi.org/10.1609/aaai.v36i6.20589)

**Abstract**:

An artificial neural network can be trained by uniformly broadcasting a reward signal to units that implement a REINFORCE learning rule. Though this presents a biologically plausible alternative to backpropagation in training a network, the high variance associated with it renders it impractical to train deep networks. The high variance arises from the inefficient structural credit assignment since a single reward signal is used to evaluate the collective action of all units. To facilitate structural credit assignment, we propose replacing the reward signal to hidden units with the change in the L2 norm of the unit's outgoing weight. As such, each hidden unit in the network is trying to maximize the norm of its outgoing weight instead of the global reward, and thus we call this learning method Weight Maximization. We prove that Weight Maximization is approximately following the gradient of rewards in expectation. In contrast to backpropagation, Weight Maximization can be used to train both continuous-valued and discrete-valued units. Moreover, Weight Maximization solves several major issues of backpropagation relating to biological plausibility. Our experiments show that a network trained with Weight Maximization can learn significantly faster than REINFORCE and slightly slower than backpropagation. Weight Maximization illustrates an example of cooperative behavior automatically arising from a population of self-interested agents in a competitive game without any central coordination.

----

## [717] How to Distribute Data across Tasks for Meta-Learning?

**Authors**: *Alexandru Cioba, Michael Bromberg, Qian Wang, Ritwik Niyogi, Georgios Batzolis, Jezabel R. Garcia, Da-Shan Shiu, Alberto Bernacchia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20590](https://doi.org/10.1609/aaai.v36i6.20590)

**Abstract**:

Meta-learning models transfer the knowledge acquired from previous tasks to quickly learn new ones. They are trained on benchmarks with a fixed number of data points per task. This number is usually arbitrary and it is unknown how it affects performance at testing. Since labelling of data is expensive, finding the optimal allocation of labels across training tasks may reduce costs. Given a fixed budget of labels, should we use a small number of highly labelled tasks, or many tasks with few labels each? Should we allocate more labels to some tasks and less to others?
We show that: 1) If tasks are homogeneous, there is a uniform optimal allocation, whereby all tasks get the same amount of data; 2) At fixed budget, there is a trade-off between number of tasks and number of data points per task, with a unique solution for the optimum; 3) When trained separately, harder task should get more data, at the cost of a smaller number of tasks; 4) When training on a mixture of easy and hard tasks, more data should be allocated to easy tasks. Interestingly, Neuroscience experiments have shown that human visual skills also transfer better from easy tasks. We prove these results mathematically on mixed linear regression, and we show empirically that the same results hold for few-shot image classification on CIFAR-FS and mini-ImageNet. Our results provide guidance for allocating labels across tasks when collecting data for meta-learning.

----

## [718] Similarity Search for Efficient Active Learning and Search of Rare Concepts

**Authors**: *Cody Coleman, Edward Chou, Julian Katz-Samuels, Sean Culatana, Peter Bailis, Alexander C. Berg, Robert D. Nowak, Roshan Sumbaly, Matei Zaharia, I. Zeki Yalniz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20591](https://doi.org/10.1609/aaai.v36i6.20591)

**Abstract**:

Many active learning and search approaches are intractable for large-scale industrial settings with billions of unlabeled examples. Existing approaches search globally for the optimal examples to label, scaling linearly or even quadratically with the unlabeled data. In this paper, we improve the computational efficiency of active learning and search methods by restricting the candidate pool for labeling to the nearest neighbors of the currently labeled set instead of scanning over all of the unlabeled data. We evaluate several selection strategies in this setting on three large-scale computer vision datasets: ImageNet, OpenImages, and a de-identified and aggregated dataset of 10 billion publicly shared images provided by a large internet company. Our approach achieved similar mAP and recall as the traditional global approach while reducing the computational cost of selection by up to three orders of magnitude, enabling web-scale active learning.

----

## [719] Learning Influence Adoption in Heterogeneous Networks

**Authors**: *Vincent Conitzer, Debmalya Panigrahi, Hanrui Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20592](https://doi.org/10.1609/aaai.v36i6.20592)

**Abstract**:

We study the problem of learning influence adoption in networks. In this problem, a communicable entity (such as an infectious disease, a computer virus, or a social media meme) propagates through a network, and the goal is to learn the state of each individual node by sampling only a small number of nodes and observing/testing their states. We study this problem in heterogeneous networks, in which each individual node has a set of distinct features that determine how it is affected by the propagating entity. We give an efficient algorithm with nearly optimal sample complexity for two variants of this learning problem, corresponding to symptomatic and asymptomatic spread. In each case, the optimal sample complexity naturally generalizes the complexity of learning how nodes are affected in isolation, and the complexity of learning influence adoption in a homogeneous network.

----

## [720] Graph-Wise Common Latent Factor Extraction for Unsupervised Graph Representation Learning

**Authors**: *Thilini Cooray, Ngai-Man Cheung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20593](https://doi.org/10.1609/aaai.v36i6.20593)

**Abstract**:

Unsupervised graph-level representation learning plays a crucial role in a variety of tasks such as molecular property prediction and community analysis, especially when data annotation is expensive. Currently, most of the best-performing graph embedding methods are based on Infomax principle. The performance of these methods highly depends on the selection of negative samples and hurt the performance, if the samples were not carefully selected. Inter-graph similarity-based methods also suffer if the selected set of graphs for similarity matching is low in quality. To address this, we focus only on utilizing the current input graph for embedding learning. We are motivated by an observation from real-world graph generation processes where the graphs are formed based on one or more global factors which are common to all elements of the graph (e.g., topic of a discussion thread, solubility level of a molecule). We hypothesize extracting these common factors could be highly beneficial. Hence, this work proposes a new principle for unsupervised graph representation learning: Graph-wise Common latent Factor EXtraction (GCFX). We further propose a deep model for GCFX, deepGCFX, based on the idea of reversing the above-mentioned graph generation process which could explicitly extract common latent factors from an input graph and achieve improved results on downstream tasks to the current state-of-the-art. Through extensive experiments and analysis, we demonstrate that, while extracting common latent factors is beneficial for graph-level tasks to alleviate distractions caused by local variations of individual nodes or local neighbourhoods, it also benefits node-level tasks by enabling long-range node dependencies, especially for disassortative graphs.

----

## [721] Reinforcement Learning with Stochastic Reward Machines

**Authors**: *Jan Corazza, Ivan Gavran, Daniel Neider*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20594](https://doi.org/10.1609/aaai.v36i6.20594)

**Abstract**:

Reward machines are an established tool for dealing with reinforcement learning problems in which rewards are sparse and depend on complex sequences of actions.
 However, existing algorithms for learning reward machines assume an overly idealized setting where rewards have to be free of noise.
 To overcome this practical limitation, we introduce a novel type of reward machines, called stochastic reward machines, and an algorithm for learning them.
 Our algorithm, based on constraint solving, learns minimal stochastic reward machines from the explorations of a reinforcement learning agent.
 This algorithm can easily be paired with existing reinforcement learning algorithms for reward machines and guarantees to converge to an optimal policy in the limit.
 We demonstrate the effectiveness of our algorithm in two case studies and show that it outperforms both existing methods and a naive approach for handling noisy reward functions.

----

## [722] Sparse-RS: A Versatile Framework for Query-Efficient Sparse Black-Box Adversarial Attacks

**Authors**: *Francesco Croce, Maksym Andriushchenko, Naman D. Singh, Nicolas Flammarion, Matthias Hein*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20595](https://doi.org/10.1609/aaai.v36i6.20595)

**Abstract**:

We propose a versatile framework based on random search, Sparse-RS, for score-based sparse targeted and untargeted attacks in the black-box setting. Sparse-RS does not rely on substitute models and achieves state-of-the-art success rate and query efficiency for multiple sparse attack models: L0-bounded perturbations, adversarial patches, and adversarial frames. The L0-version of untargeted Sparse-RS outperforms all black-box and even all white-box attacks for different models on MNIST, CIFAR-10, and ImageNet. Moreover, our untargeted Sparse-RS achieves very high success rates even for the challenging settings of 20x20 adversarial patches and 2-pixel wide adversarial frames for 224x224 images. Finally, we show that Sparse-RS can be applied to generate targeted universal adversarial patches where it significantly outperforms the existing approaches. Our code is available at https://github.com/fra31/sparse-rs.

----

## [723] Learning Logic Programs Though Divide, Constrain, and Conquer

**Authors**: *Andrew Cropper*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20596](https://doi.org/10.1609/aaai.v36i6.20596)

**Abstract**:

We introduce an inductive logic programming approach that combines classical divide-and-conquer search with modern constraint-driven search. Our anytime approach can learn optimal, recursive, and large programs and supports predicate invention. Our experiments on three domains (classification, inductive general game playing, and program synthesis) show that our approach can increase predictive accuracies and reduce learning times.

----

## [724] Implicit Gradient Alignment in Distributed and Federated Learning

**Authors**: *Yatin Dandi, Luis Barba, Martin Jaggi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20597](https://doi.org/10.1609/aaai.v36i6.20597)

**Abstract**:

A major obstacle to achieving global convergence in distributed and federated learning is the misalignment of gradients across clients or mini-batches due to heterogeneity and stochasticity of the distributed data. In this work, we show that data heterogeneity can in fact be exploited to improve generalization performance through implicit regularization.
One way to alleviate the effects of heterogeneity is to encourage the alignment of gradients across different clients throughout training. Our analysis reveals that this goal can be accomplished by utilizing the right optimization method that replicates the implicit regularization effect of SGD, leading to gradient alignment as well as improvements in test accuracies.
Since the existence of this regularization in SGD completely relies on the sequential use of different mini-batches during training, it is inherently absent when training with large mini-batches.
To obtain the generalization benefits of this regularization while increasing parallelism, we propose a novel GradAlign algorithm that induces the same implicit regularization while allowing the use of arbitrarily large batches in each update. We experimentally validate the benefits of our algorithm in different distributed and federated learning settings.

----

## [725] How Good Are Low-Rank Approximations in Gaussian Process Regression?

**Authors**: *Constantinos Daskalakis, Petros Dellaportas, Aristeidis Panos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20598](https://doi.org/10.1609/aaai.v36i6.20598)

**Abstract**:

We provide guarantees for approximate Gaussian Process (GP) regression resulting from two common low-rank kernel approximations: based on random Fourier features, and based on truncating the kernel's Mercer expansion. In particular, we bound the Kullback–Leibler divergence between an exact GP and one resulting from one of the afore-described low-rank approximations to its 
 kernel, as well as between their corresponding predictive densities, and we also bound the error between predictive mean vectors and between predictive covariance matrices computed using the exact versus using the approximate GP. We provide experiments on both simulated data and standard benchmarks to evaluate the effectiveness of our theoretical bounds.

----

## [726] KOALA: A Kalman Optimization Algorithm with Loss Adaptivity

**Authors**: *Aram Davtyan, Sepehr Sameni, Llukman Cerkezi, Givi Meishvili, Adam Bielski, Paolo Favaro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20599](https://doi.org/10.1609/aaai.v36i6.20599)

**Abstract**:

Optimization is often cast as a deterministic problem, where the solution is found through some iterative procedure such as gradient descent. However, when training neural networks the loss function changes over (iteration) time due to the randomized selection of a subset of the samples. This randomization turns the optimization problem into a stochastic one. We propose to consider the loss as a noisy observation with respect to some reference optimum. This interpretation of the loss allows us to adopt Kalman filtering as an optimizer, as its recursive formulation is designed to estimate unknown parameters from noisy measurements. Moreover, we show that the Kalman Filter dynamical model for the evolution of the unknown parameters can be used to capture the gradient dynamics of advanced methods such as Momentum and Adam. We call this stochastic optimization method KOALA, which is short for Kalman Optimization Algorithm with Loss Adaptivity. KOALA is an easy to implement, scalable, and efficient method to train neural networks. We provide convergence analysis and show experimentally that it yields parameter estimates that are on par with or better than existing state of the art optimization algorithms across several neural network architectures and machine learning tasks, such as computer vision and language modeling. The project page with the code and the supplementary materials is available at https://araachie.github.io/koala/.

----

## [727] First-Order Convex Fitting and Its Application to Economics and Optimization

**Authors**: *Quinlan Dawkins, Minbiao Han, Haifeng Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20600](https://doi.org/10.1609/aaai.v36i6.20600)

**Abstract**:

This paper studies a function fitting problem which we coin  first-order convex fitting (FCF): given any two vector sequences x1, ..., xT and p1, ..., pT, when is it possible to  efficiently construct  a convex function f(x) that ``fits'' the two sequences in the first-order sense, i.e,   the (sub)gradient of f(xi) equals precisely pi, for all i = 1, ..., T? Despite a  basic question of convex analysis,  FCF has surprisingly been overlooked in the past literature. With an efficient constructive proof, we provide a clean answer to this question:  FCF   is possible  if and only if the two sequences are permutation stable: p1 * x1 + ... + pT * xT is greater than or equal to p1 * x’1 + ... + pT * x’T where x’1, ..., x’T is any permutation of x1, ..., xT.
 
We  demonstrate the usefulness of FCF in  two  applications. First, we study how it can be used as an empirical risk minimization procedure to learn the original convex function. We provide  efficient PAC-learnability bounds for special classes of convex functions  learned via FCF, and demonstrate its application to multiple economic problems where only function gradients (as opposed to function values)  can be observed. Second,  we empirically show how it can be used as a surrogate to significantly accelerate the minimization of the original convex function.

----

## [728] Gradient Temporal Difference with Momentum: Stability and Convergence

**Authors**: *Rohan Deb, Shalabh Bhatnagar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20601](https://doi.org/10.1609/aaai.v36i6.20601)

**Abstract**:

Gradient temporal difference (Gradient TD) algorithms are a popular class of stochastic approximation (SA) algorithms used for policy evaluation in reinforcement learning. Here, we consider Gradient TD algorithms with an additional heavy ball momentum term and provide choice of step size and momentum parameter that ensures almost sure convergence of these algorithms asymptotically. In doing so, we decompose the heavy ball Gradient TD iterates into three separate iterates with different step sizes. We first analyze these iterates under one-timescale SA setting using results from current literature. However, the one-timescale case is restrictive and a more general analysis can be provided by looking at a three-timescale decomposition of the iterates. In the process we provide the first conditions for stability and convergence of general three-timescale SA. We then prove that the heavy ball Gradient TD algorithm is convergent using our three-timescale SA analysis. Finally, we evaluate these algorithms on standard RL problems and report improvement in performance over the vanilla algorithms.

----

## [729] Distillation of RL Policies with Formal Guarantees via Variational Abstraction of Markov Decision Processes

**Authors**: *Florent Delgrange, Ann Nowé, Guillermo A. Pérez*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20602](https://doi.org/10.1609/aaai.v36i6.20602)

**Abstract**:

We consider the challenge of policy simplification and verification in the context of policies learned through reinforcement learning (RL) in continuous environments. In well-behaved settings, RL algorithms have convergence guarantees in the limit. While these guarantees are valuable, they are insufficient for safety-critical applications. Furthermore, they are lost when applying advanced techniques such as deep-RL. To recover guarantees when applying advanced RL algorithms to  more complex environments with (i) reachability, (ii) safety-constrained reachability, or (iii) discounted-reward objectives, we build upon the DeepMDP framework to derive new bisimulation bounds between the unknown environment and a learned discrete latent model of it. Our bisimulation bounds enable the application of formal methods for Markov decision processes. Finally, we show how one can use a policy obtained via state-of-the-art RL to efficiently train a variational autoencoder that yields a discrete latent model with provably approximately correct bisimulation guarantees. Additionally, we obtain a distilled version of the policy for the latent model.

----

## [730] Reducing Flipping Errors in Deep Neural Networks

**Authors**: *Xiang Deng, Yun Xiao, Bo Long, Zhongfei Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20603](https://doi.org/10.1609/aaai.v36i6.20603)

**Abstract**:

Deep neural networks (DNNs) have been widely applied in various domains in artificial intelligence including computer vision and natural language processing.
 A DNN is typically trained for many epochs and then a validation dataset is used to select the DNN in an epoch (we simply call this epoch ``the last epoch") as the final model for making predictions on unseen samples, while it usually cannot achieve a perfect accuracy on unseen samples. An interesting question is ``how many test (unseen) samples that a DNN misclassifies in the last epoch were ever correctly classified by the DNN before the last epoch?". In this paper, we empirically study this question and find on several benchmark datasets that the vast majority of the misclassified samples in the last epoch were ever classified correctly before the last epoch, which means that the predictions for these samples were flipped from ``correct" to ``wrong". Motivated by this observation, we propose to restrict the behavior changes of a DNN on the correctly-classified samples so that the correct local boundaries can be maintained and the flipping error on unseen samples can be largely reduced. Extensive experiments on different benchmark datasets with different modern network architectures demonstrate that the proposed flipping error reduction (FER) approach can substantially improve the generalization, the robustness, and the transferability of DNNs without introducing any additional network parameters or inference cost, only with a negligible training overhead.

----

## [731] Bayesian Optimization over Permutation Spaces

**Authors**: *Aryan Deshwal, Syrine Belakaria, Janardhan Rao Doppa, Dae Hyun Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20604](https://doi.org/10.1609/aaai.v36i6.20604)

**Abstract**:

Optimizing expensive to evaluate black-box functions over an input space consisting of all permutations of d objects is an important problem with many real-world applications. For example, placement of functional blocks in hardware design to optimize performance via simulations. The overall goal is to minimize the number of function evaluations to find high-performing permutations. The key challenge in solving this problem using the Bayesian optimization (BO) framework is to trade-off the complexity of statistical model and tractability of acquisition function optimization. In this paper, we propose and evaluate two algorithms for BO over Permutation Spaces (BOPS). First, BOPS-T employs Gaussian process (GP) surrogate model with Kendall kernels and a Tractable acquisition function optimization approach to select the sequence of permutations for evaluation. Second, BOPS-H employs GP surrogate model with Mallow kernels and a Heuristic search approach to optimize the acquisition function. We theoretically analyze the performance of BOPS-T to show that their regret grows sub-linearly. Our experiments on multiple synthetic and real-world benchmarks show that both BOPS-T and BOPS-H perform better than the state-of-the-art BO algorithm for combinatorial spaces. To drive future research on this important problem, we make new resources and real-world benchmarks available to the community.

----

## [732] Meta Propagation Networks for Graph Few-shot Semi-supervised Learning

**Authors**: *Kaize Ding, Jianling Wang, James Caverlee, Huan Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20605](https://doi.org/10.1609/aaai.v36i6.20605)

**Abstract**:

Inspired by the extensive success of deep learning, graph neural networks (GNNs) have been proposed to learn expressive node representations and demonstrated promising performance in various graph learning tasks. However, existing endeavors predominately focus on the conventional semi-supervised setting where relatively abundant gold-labeled nodes are provided. While it is often impractical due to the fact that data labeling is unbearably laborious and requires intensive domain knowledge, especially when considering the heterogeneity of graph-structured data. Under the few-shot semi-supervised setting, the performance of most of the existing GNNs is inevitably undermined by the overfitting and oversmoothing issues, largely owing to the shortage of labeled data. In this paper, we propose a decoupled network architecture equipped with a novel meta-learning algorithm to solve this problem. In essence, our framework Meta-PN infers high-quality pseudo labels on unlabeled nodes via a meta-learned label propagation strategy, which effectively augments the scarce labeled data while enabling large receptive fields during training. Extensive experiments demonstrate that our approach offers easy and substantial performance gains compared to existing techniques on various benchmark datasets. The implementation and extended manuscript of this work are publicly available at https://github.com/kaize0409/Meta-PN.

----

## [733] Online Certification of Preference-Based Fairness for Personalized Recommender Systems

**Authors**: *Virginie Do, Sam Corbett-Davies, Jamal Atif, Nicolas Usunier*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20606](https://doi.org/10.1609/aaai.v36i6.20606)

**Abstract**:

Recommender systems are facing scrutiny because of their growing impact on the opportunities we have access to. Current audits for fairness are limited to coarse-grained parity assessments at the level of sensitive groups. We propose to audit for envy-freeness, a more granular criterion aligned with individual preferences: every user should prefer their recommendations to those of other users. Since auditing for envy requires to estimate the preferences of users beyond their existing recommendations, we cast the audit as a new pure exploration problem in multi-armed bandits. We propose a sample-efficient algorithm with theoretical guarantees that it does not deteriorate user experience. We also study the trade-offs achieved on real-world recommendation datasets.

----

## [734] Disentangled Spatiotemporal Graph Generative Models

**Authors**: *Yuanqi Du, Xiaojie Guo, Hengning Cao, Yanfang Ye, Liang Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20607](https://doi.org/10.1609/aaai.v36i6.20607)

**Abstract**:

Spatiotemporal graph represents a crucial data structure where the nodes and edges are embedded in a geometric space and their attribute values can evolve dynamically over time. Nowadays, spatiotemporal graph data is becoming increasingly popular and important, ranging from microscale (e.g. protein folding), to middle-scale (e.g. dynamic functional connectivity), to macro-scale (e.g. human mobility network). Although disentangling and understanding the correlations among spatial, temporal, and graph aspects have been a long-standing key topic in network science, they typically rely on network processes hypothesized by human knowledge. They usually fit well towards the properties that the predefined principles are tailored for, but usually cannot do well for the others, especially for many key domains where the human has yet very limited knowledge such as protein folding and biological neuronal networks. In this paper, we aim at pushing forward the modeling and understanding of spatiotemporal graphs via new disentangled deep generative models. Specifically, a new Bayesian model is proposed that factorizes spatiotemporal graphs into spatial, temporal, and graph factors as well as the factors that explain the interplay among them. A variational objective function and new mutual information thresholding algorithms driven by information bottleneck theory have been proposed to maximize the disentanglement among the factors with theoretical guarantees. Qualitative and quantitative experiments on both synthetic and real-world datasets demonstrate the superiority of the proposed model over the state-of-the-arts by up to 69.2% for graph generation and 41.5% for interpretability.

----

## [735] Learning from the Dark: Boosting Graph Convolutional Neural Networks with Diverse Negative Samples

**Authors**: *Wei Duan, Junyu Xuan, Maoying Qiao, Jie Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20608](https://doi.org/10.1609/aaai.v36i6.20608)

**Abstract**:

Graph Convolutional Neural Networks (GCNs) have been generally accepted to be an effective tool for node representations learning. An interesting way to understand GCNs is to think of them as a message passing mechanism where each node updates its representation by accepting information from its neighbours (also known as positive samples). However, beyond these neighbouring nodes, graphs have a large, dark, all-but forgotten world in which we find the non-neighbouring nodes (negative samples). In this paper, we show that this great dark world holds a substantial amount of information that might be useful for representation learning. Most specifically, it can provide negative information about the node representations. Our overall idea is to select appropriate negative samples for each node and incorporate the negative information contained in these samples into the representation updates. Moreover, we show that the process of selecting the negative samples is not trivial. Our theme therefore begins by describing the criteria for a good negative sample, followed by a determinantal point process algorithm for efficiently obtaining such samples. A GCN, boosted by diverse negative samples, then jointly considers the positive and negative information when passing messages. Experimental evaluations show that this idea not only improves the overall performance of standard representation learning but also significantly alleviates over-smoothing problems.

----

## [736] Adaptive and Universal Algorithms for Variational Inequalities with Optimal Convergence

**Authors**: *Alina Ene, Huy Le Nguyen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20609](https://doi.org/10.1609/aaai.v36i6.20609)

**Abstract**:

We develop new adaptive algorithms for variational inequalities with monotone operators, which capture many problems of interest, notably convex optimization and convex-concave saddle point problems. Our algorithms automatically adapt to unknown problem parameters such as the smoothness and the norm of the operator, and the variance of the stochastic evaluation oracle. We show that our algorithms are universal and simultaneously achieve the optimal convergence rates in the non-smooth, smooth, and stochastic settings. The convergence guarantees of our algorithms improve over existing adaptive methods and match the optimal non-adaptive algorithms. Additionally, prior works require that the optimization domain is bounded. In this work, we remove this restriction and give algorithms for unbounded domains that are adaptive and universal. Our general proof techniques can be used for many variants of the algorithm using one or two operator evaluations per iteration. The classical methods based on the ExtraGradient/MirrorProx algorithm require two operator evaluations per iteration, which is the dominant factor in the running time in many settings.

----

## [737] Zero-Shot Out-of-Distribution Detection Based on the Pre-trained Model CLIP

**Authors**: *Sepideh Esmaeilpour, Bing Liu, Eric Robertson, Lei Shu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20610](https://doi.org/10.1609/aaai.v36i6.20610)

**Abstract**:

In an out-of-distribution (OOD) detection problem, samples of known classes (also called in-distribution classes) are used to train a special classifier. In testing, the classifier can (1) classify the test samples of known classes to their respective classes and also (2) detect samples that do not belong to any of the known classes (i.e., they belong to some unknown or OOD classes). This paper studies the problem of zero-shot out-of-distribution (OOD) detection, which still performs the same two tasks in testing but has no training except using the given known class names. This paper proposes a novel and yet simple method (called ZOC) to solve the problem. ZOC builds on top of the recent advances in zero-shot classification through multi-modal representation learning. It first extends the pre-trained language-vision model CLIP by training a text-based image description generator on top of CLIP. In testing, it uses the extended model to generate candidate unknown class names for each test sample and computes a confidence score based on both the known class names and candidate unknown class names for zero-shot OOD detection. Experimental results on 5 benchmark datasets for OOD detection demonstrate that ZOC outperforms the baselines by a large margin.

----

## [738] Gradient Flow in Sparse Neural Networks and How Lottery Tickets Win

**Authors**: *Utku Evci, Yani Ioannou, Cem Keskin, Yann N. Dauphin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20611](https://doi.org/10.1609/aaai.v36i6.20611)

**Abstract**:

Sparse Neural Networks (NNs) can match the generalization of dense NNs using a fraction of the compute/storage for inference, and have the potential to enable efficient training. However, naively training unstructured sparse NNs from random initialization results in significantly worse generalization, with the notable exceptions of Lottery Tickets (LTs) and Dynamic Sparse Training (DST). In this work, we attempt to answer: (1) why training unstructured sparse networks from random initialization performs poorly and; (2) what makes LTs and DST the exceptions? We show that sparse NNs have poor gradient flow at initialization and propose a modified initialization for unstructured connectivity. Furthermore, we find that DST methods significantly improve gradient flow during training over traditional sparse training methods. Finally, we show that LTs do not improve gradient flow, rather their success lies in re-learning the pruning solution they are derived from — however, this comes at the cost of learning novel solutions.

----

## [739] Dynamic Nonlinear Matrix Completion for Time-Varying Data Imputation

**Authors**: *Jicong Fan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20612](https://doi.org/10.1609/aaai.v36i6.20612)

**Abstract**:

Classical matrix completion methods focus on data with stationary latent structure and hence are not effective in missing value imputation when the latent structure changes with time.  This paper proposes a dynamic nonlinear matrix completion (D-NLMC) method, which is able to recover the missing values of streaming data when the low-dimensional nonlinear latent structure of the data changes with time.  The paper provides an efficient approach to updating the nonlinear model dynamically.  D-NLMC incorporates the information of new data and remove the information of earlier data recursively.  The paper shows that the missing data can be estimated if the change of latent structure is slow enough.  Different from existing online or adaptive low-rank matrix completion methods,  D-NLMC does not require the local low-rank assumption and is able to adaptively recover high-rank matrices with low-dimensional latent structures. Note that existing high-rank matrix completion methods have high-computational costs and are not applicable to streaming data with varying latent structures, which fortunately can be handled by D-NLMC efficiently and accurately.  Numerical results show that D-NLMC outperforms the baselines in real applications.

----

## [740] Up to 100x Faster Data-Free Knowledge Distillation

**Authors**: *Gongfan Fang, Kanya Mo, Xinchao Wang, Jie Song, Shitao Bei, Haofei Zhang, Mingli Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20613](https://doi.org/10.1609/aaai.v36i6.20613)

**Abstract**:

Data-free knowledge distillation (DFKD) has recently been attracting increasing attention from research communities, attributed to its capability to compress a model only using synthetic data. Despite the encouraging results achieved, state-of-the-art DFKD methods still suffer from the inefficiency of data synthesis, making the data-free training process extremely time-consuming and thus inapplicable for large-scale tasks. In this work, we introduce an efficacious scheme, termed as FastDFKD, that allows us to accelerate DFKD by a factor of orders of magnitude. At the heart of our approach is a novel strategy to reuse the shared common features in training data so as to synthesize different data instances. Unlike prior methods that optimize a set of data independently, we propose to learn a meta-synthesizer that seeks common features as the initialization for the fast data synthesis. As a result, FastDFKD achieves data synthesis within only a few steps,  significantly enhancing the efficiency of data-free training. Experiments over CIFAR, NYUv2, and ImageNet demonstrate that the proposed FastDFKD achieves 10x and even 100x acceleration while preserving  performances on par with state of the art. Code is available at https://github.com/zju-vipa/Fast-Datafree.

----

## [741] Learning Aligned Cross-Modal Representation for Generalized Zero-Shot Classification

**Authors**: *Zhiyu Fang, Xiaobin Zhu, Chun Yang, Zheng Han, Jingyan Qin, Xu-Cheng Yin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20614](https://doi.org/10.1609/aaai.v36i6.20614)

**Abstract**:

Learning a common latent embedding by aligning the latent spaces of cross-modal autoencoders is an effective strategy for Generalized Zero-Shot Classification (GZSC). However, due to the lack of fine-grained instance-wise annotations, it still easily suffer from the domain shift problem for the discrepancy between the visual representation of diversified images and the semantic representation of fixed attributes. In this paper, we propose an innovative autoencoder network by learning Aligned Cross-Modal Representations (dubbed ACMR) for GZSC. Specifically, we propose a novel Vision-Semantic Alignment (VSA) method to strengthen the alignment of cross-modal latent features on the latent subspaces guided by a learned classifier. In addition, we propose a novel Information Enhancement Module (IEM) to reduce the possibility of latent variables collapse meanwhile encouraging the discriminative ability of latent variables. Extensive experiments on publicly available datasets demonstrate the state-of-the-art performance of our method.

----

## [742] KerGNNs: Interpretable Graph Neural Networks with Graph Kernels

**Authors**: *Aosong Feng, Chenyu You, Shiqiang Wang, Leandros Tassiulas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20615](https://doi.org/10.1609/aaai.v36i6.20615)

**Abstract**:

Graph kernels are historically the most widely-used technique for graph classification tasks. However, these methods suffer from limited performance because of the hand-crafted combinatorial features of graphs. In recent years, graph neural networks (GNNs) have become the state-of-the-art method in downstream graph-related tasks due to their superior performance. Most GNNs are based on Message Passing Neural Network (MPNN) frameworks. However, recent studies show that MPNNs can not exceed the power of the Weisfeiler-Lehman (WL) algorithm in graph isomorphism test. To address the limitations of existing graph kernel and GNN methods, in this paper, we propose a novel GNN framework, termed Kernel Graph Neural Networks (KerGNNs), which integrates graph kernels into the message passing process of GNNs. Inspired by convolution filters in convolutional neural networks (CNNs), KerGNNs adopt trainable hidden graphs as graph filters which are combined with subgraphs to update node embeddings using graph kernels. In addition, we show that MPNNs can be viewed as special cases of KerGNNs. We apply KerGNNs to multiple graph-related tasks and use cross-validation to make fair comparisons with benchmarks. We show that our method achieves competitive performance compared with existing state-of-the-art methods, demonstrating the potential to increase the representation ability of GNNs. We also show that the trained graph filters in KerGNNs can reveal the local graph structures of the dataset, which significantly improves the model interpretability compared with conventional GNN models.

----

## [743] Scaling Neural Program Synthesis with Distribution-Based Search

**Authors**: *Nathanaël Fijalkow, Guillaume Lagarde, Théo Matricon, Kevin Ellis, Pierre Ohlmann, Akarsh Nayan Potta*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20616](https://doi.org/10.1609/aaai.v36i6.20616)

**Abstract**:

We consider the problem of automatically constructing computer programs from input-output examples. We investigate how to augment probabilistic and neural program synthesis methods with new search algorithms, proposing a framework called distribution-based search. Within this framework, we introduce two new search algorithms: Heap Search, an enumerative method, and SQRT Sampling, a probabilistic method. We prove certain optimality guarantees for both methods, show how they integrate with probabilistic and neural techniques, and demonstrate how they can operate at scale across parallel compute environments. Collectively these findings offer theoretical and applied studies of search algorithms for program synthesis that integrate with recent developments in machine-learned program synthesizers.

----

## [744] Modification-Fair Cluster Editing

**Authors**: *Vincent Froese, Leon Kellerhals, Rolf Niedermeier*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20617](https://doi.org/10.1609/aaai.v36i6.20617)

**Abstract**:

The classic Cluster Editing problem (also known as Correlation Clustering) asks to transform a given graph into a disjoint union of cliques (clusters) by a small number of edge modifications. When applied to vertex-colored graphs (the colors representing subgroups), standard algorithms for the NP-hard Cluster Editing problem may yield  solutions that are biased towards subgroups of data (e.g., demographic groups), measured in the number of modifications incident to the members of the subgroups.
We propose a modification fairness constraint which ensures that the number of edits incident to each subgroup is proportional to its size. To start with, we study Modification-Fair Cluster Editing for graphs with two vertex colors. We show that the problem is NP-hard even if one may only insert edges within a subgroup; note that in the classic "non-fair" setting, this case is trivially polynomial-time solvable. However, in the more general editing form, the modification-fair variant remains fixed-parameter tractable with respect to the number of edge edits. We complement these and further theoretical results with an empirical analysis of our model on real-world social networks where we find that the price of modification-fairness is surprisingly low, that is, the cost of optimal modification-fair differs from the cost of optimal "non-fair" solutions only by a small percentage.

----

## [745] Reinforcement Learning Based Dynamic Model Combination for Time Series Forecasting

**Authors**: *Yuwei Fu, Di Wu, Benoit Boulet*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20618](https://doi.org/10.1609/aaai.v36i6.20618)

**Abstract**:

Time series data appears in many real-world fields such as energy, transportation, communication systems. Accurate modelling and forecasting of time series data can be of significant importance to improve the efficiency of these systems. Extensive research efforts have been taken for time series problems. Different types of approaches, including both statistical-based methods and machine learning-based methods, have been investigated. Among these methods, ensemble learning has shown to be effective and robust. However, it is still an open question that how we should determine weights for base models in the ensemble. Sub-optimal weights may prevent the final model from reaching its full potential. To deal with this challenge, we propose a reinforcement learning (RL) based model combination (RLMC) framework for determining model weights in an ensemble for time series forecasting tasks. By formulating model selection as a sequential decision-making problem, RLMC learns a deterministic policy to output dynamic model weights for non-stationary time series data. RLMC further leverages deep learning to learn hidden features from raw time series data to adapt fast to the changing data distribution. Extensive experiments on multiple real-world datasets have been implemented to showcase the effectiveness of the proposed method.

----

## [746] JFB: Jacobian-Free Backpropagation for Implicit Networks

**Authors**: *Samy Wu Fung, Howard Heaton, Qiuwei Li, Daniel McKenzie, Stanley J. Osher, Wotao Yin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20619](https://doi.org/10.1609/aaai.v36i6.20619)

**Abstract**:

A promising trend in deep learning replaces traditional feedforward networks with implicit networks. Unlike traditional networks, implicit networks solve a fixed point equation to compute inferences. Solving for the fixed point varies in complexity, depending on provided data and an error tolerance. Importantly, implicit networks may be trained with fixed memory costs in stark contrast to feedforward networks, whose memory requirements scale linearly with depth. However, there is no free
 lunch --- backpropagation through implicit networks often requires solving a costly Jacobian-based equation arising from the implicit function theorem. We propose Jacobian-Free Backpropagation (JFB), a fixed-memory approach that circumvents the need to solve Jacobian-based equations. JFB makes implicit networks faster to train and significantly easier to implement, without sacrificing test accuracy. Our experiments show implicit networks trained with JFB are competitive with feedforward networks and prior implicit networks given the same number of parameters.

----

## [747] Smoothing Advantage Learning

**Authors**: *Yaozhong Gan, Zhe Zhang, Xiaoyang Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20620](https://doi.org/10.1609/aaai.v36i6.20620)

**Abstract**:

Advantage learning (AL) aims to improve the robustness of value-based reinforcement learning against estimation errors with action-gap-based regularization. Unfortunately, the method tends to be unstable in the case of function approximation. In this paper, we propose a simple variant of AL, named smoothing advantage learning (SAL), to alleviate this problem. The key to our method is to replace the original Bellman Optimal operator in AL with a smooth one so as to obtain more reliable estimation of the temporal difference target. We give a detailed account of the resulting action gap and the performance bound for approximate SAL. Further theoretical analysis reveals that the proposed value smoothing technique not only helps to stabilize the training procedure of AL by controlling the trade-off between convergence rate and the upper bound of the approximation errors, but is beneficial to increase the action gap between the optimal and sub-optimal action value as well.

----

## [748] Enhancing Counterfactual Classification Performance via Self-Training

**Authors**: *Ruijiang Gao, Max Biggs, Wei Sun, Ligong Han*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20621](https://doi.org/10.1609/aaai.v36i6.20621)

**Abstract**:

Unlike traditional supervised learning, in many settings only partial feedback is available. We may only observe outcomes for the chosen actions, but not the counterfactual outcomes associated with other alternatives. Such settings encompass a wide variety of applications including pricing, online marketing and precision medicine. A key challenge is that observational data are influenced by historical policies deployed in the system, yielding a biased data distribution. We approach this task as a domain adaptation problem and propose a self-training algorithm which imputes outcomes with categorical values for finite unseen actions in the observational data to simulate a randomized trial through pseudolabelling, which we refer to as Counterfactual Self-Training (CST). CST iteratively imputes pseudolabels and retrains the model. In addition, we show input consistency loss can further improve CST performance which is shown in recent theoretical analysis of pseudolabelling. We demonstrate the effectiveness of the proposed algorithms on both synthetic and real datasets.

----

## [749] Learning V1 Simple Cells with Vector Representation of Local Content and Matrix Representation of Local Motion

**Authors**: *Ruiqi Gao, Jianwen Xie, Siyuan Huang, Yufan Ren, Song-Chun Zhu, Ying Nian Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20622](https://doi.org/10.1609/aaai.v36i6.20622)

**Abstract**:

This paper proposes a representational model for image pairs such as consecutive video frames that are related by local pixel displacements, in the hope that the model may shed light on motion perception in primary visual cortex (V1). The model couples the following two components: (1) the vector representations of local contents of images and (2) the matrix representations of local pixel displacements caused by the relative motions between the agent and the objects in the 3D scene. When the image frame undergoes changes due to local pixel displacements, the vectors are multiplied by the matrices that represent the local displacements. Thus the vector representation is equivariant as it varies according to the local displacements. Our experiments show that our model can learn Gabor-like filter pairs of quadrature phases. The profiles of the learned filters match those of simple cells in Macaque V1. Moreover, we demonstrate that the model can learn to infer local motions in either a supervised or unsupervised manner. With such a simple model, we achieve competitive results on optical flow estimation.

----

## [750] Algorithmic Concept-Based Explainable Reasoning

**Authors**: *Dobrik Georgiev, Pietro Barbiero, Dmitry Kazhdan, Petar Velickovic, Pietro Lió*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20623](https://doi.org/10.1609/aaai.v36i6.20623)

**Abstract**:

Recent research on graph neural network (GNN) models successfully applied GNNs to classical graph algorithms and combinatorial optimisation problems. This has numerous benefits, such as allowing applications of algorithms when preconditions are not satisfied, or reusing learned models when sufficient training data is not available or can't be generated. Unfortunately, a key hindrance of these approaches is their lack of explainability, since GNNs are black-box models that cannot be interpreted directly. In this work, we address this limitation by applying existing work on concept-based explanations to GNN models. We introduce concept-bottleneck GNNs, which rely on a modification to the GNN readout mechanism. Using three case studies we demonstrate that: (i) our proposed model is capable of accurately learning concepts and extracting propositional formulas based on the learned concepts for each target class; (ii) our concept-based GNN models achieve comparative performance with state-of-the-art models; (iii) we can derive global graph concepts, without explicitly providing any supervision on graph-level concepts.

----

## [751] Recovering the Propensity Score from Biased Positive Unlabeled Data

**Authors**: *Walter Gerych, Thomas Hartvigsen, Luke Buquicchio, Emmanuel Agu, Elke A. Rundensteiner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20624](https://doi.org/10.1609/aaai.v36i6.20624)

**Abstract**:

Positive-Unlabeled (PU) learning methods train a classifier to distinguish between the positive and negative classes given only positive and unlabeled data. While traditional PU methods require the labeled positive samples to be an unbiased sample of the positive distribution, in practice the labeled sample is often a biased draw from the true distribution. Prior work shows that if we know the likelihood that each positive instance will be selected for labeling, referred to as the propensity score, then the biased sample can be used for PU learning. Unfortunately, no prior work has been proposed an inference strategy for which the propensity score is identifiable. In this work, we propose two sets of assumptions under which the propensity score can be uniquely determined: one in which no assumption is made on the functional form of the propensity score (requiring assumptions on the data distribution), and the second which loosens the data assumptions while assuming a functional form for the propensity score. We then propose inference strategies for each case. Our empirical study shows that our approach significantly outperforms the state-of-the-art propensity estimation methods on a rich variety of benchmark datasets.

----

## [752] DiPS: Differentiable Policy for Sketching in Recommender Systems

**Authors**: *Aritra Ghosh, Saayan Mitra, Andrew S. Lan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20625](https://doi.org/10.1609/aaai.v36i6.20625)

**Abstract**:

In sequential recommender system applications, it is important to develop models that can capture users' evolving interest over time to successfully recommend future items that they are likely to interact with. For users with long histories, typical models based on recurrent neural networks tend to forget important items in the distant past. Recent works have shown that storing a small sketch of past items can improve sequential recommendation tasks. However, these works all rely on static sketching policies, i.e., heuristics to select items to keep in the sketch, which are not necessarily optimal and cannot improve over time with more training data. In this paper, we propose a differentiable policy for sketching (DiPS), a framework that learns a data-driven sketching policy in an end-to-end manner together with the recommender system model to explicitly maximize recommendation quality in the future. 
 We also propose an approximate estimator of the gradient for optimizing the sketching algorithm parameters that is computationally efficient. We verify the effectiveness of DiPS on real-world datasets under various practical settings and show that it requires up to 50% fewer sketch items to reach the same predictive quality than existing sketching policies.

----

## [753] Learning Large DAGs by Combining Continuous Optimization and Feedback Arc Set Heuristics

**Authors**: *Pierre Gillot, Pekka Parviainen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20626](https://doi.org/10.1609/aaai.v36i6.20626)

**Abstract**:

Bayesian networks represent relations between variables using a directed acyclic graph (DAG). Learning the DAG is an NP-hard problem and exact learning algorithms are feasible only for small sets of variables. We propose two scalable heuristics for learning DAGs in the linear structural equation case. Our methods learn the DAG by alternating between unconstrained gradient descent-based step to optimize an objective function and solving a maximum acyclic subgraph problem to enforce acyclicity. Thanks to this decoupling, our methods scale up beyond thousands of variables.

----

## [754] Regularized Modal Regression on Markov-Dependent Observations: A Theoretical Assessment

**Authors**: *Tieliang Gong, Yuxin Dong, Hong Chen, Wei Feng, Bo Dong, Chen Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20627](https://doi.org/10.1609/aaai.v36i6.20627)

**Abstract**:

Modal regression, a widely used regression protocol, has been extensively investigated in statistical and machine learning communities due to its robustness to outlier and heavy-tailed noises. Understanding modal regression's theoretical behavior can be fundamental in learning theory. Despite significant progress in characterizing its statistical property, the majority results are based on the assumption that samples are independent and identical distributed (i.i.d.), which is too restrictive for real-world applications. This paper concerns about the statistical property of regularized modal regression (RMR) within an important dependence structure - Markov dependent. Specifically, we establish the upper bound for RMR estimator under moderate conditions and give an explicit learning rate. Our results show that the Markov dependence impacts on the generalization error in the way that sample size would be discounted by a multiplicative factor depending on the spectral gap of the underlying Markov chain. This result shed a new light on characterizing the theoretical underpinning for robust regression.

----

## [755] Partial Multi-Label Learning via Large Margin Nearest Neighbour Embeddings

**Authors**: *Xiuwen Gong, Dong Yuan, Wei Bao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20628](https://doi.org/10.1609/aaai.v36i6.20628)

**Abstract**:

To deal with ambiguities in partial multi-label learning (PML), existing popular PML research attempts to perform disambiguation by direct ground-truth label identification. However, these approaches can be easily misled by noisy false-positive labels in the iteration of updating the model parameter and the latent ground-truth label variables. When labeling information is ambiguous, we should depend more on underlying structure of data, such as label and feature correlations, to perform disambiguation for partially labeled data. Moreover, large margin nearest neighbour (LMNN) is a popular strategy that considers data structure in classification. However, due to the ambiguity of labeling information in PML, traditional LMNN cannot be used to solve the PML problem directly. In addition, embedding is an effective technology to decrease the noise information of data. Inspried by LMNN and embedding technology, we propose a novel PML paradigm called Partial Multi-label Learning via Large Margin Nearest Neighbour Embeddings (PML-LMNNE), which aims to conduct disambiguation by projecting labels and features into a lower-dimension embedding space and reorganize the underlying structure by LMNN in the embedding space simultaneously. An efficient algorithm is designed to implement the proposed method and the convergence rate of the algorithm is analyzed. Moreover, we present a theoretical analysis of the generalization error bound for the proposed PML-LMNNE, which shows that the generalization error converges to the sum of two times the Bayes error over the labels when the number of instances goes to infinity. Comprehensive experiments on artificial and real-world datasets demonstrate the superiorities of the proposed PML-LMNNE.

----

## [756] LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks

**Authors**: *Adam Goodge, Bryan Hooi, See-Kiong Ng, Wee Siong Ng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20629](https://doi.org/10.1609/aaai.v36i6.20629)

**Abstract**:

Many well-established anomaly detection methods use the distance of a sample to those in its local neighbourhood: so-called `local outlier methods', such as LOF and DBSCAN. They are popular for their simple principles and strong performance on unstructured, feature-based data that is commonplace in many practical applications. However, they cannot learn to adapt for a particular set of data due to their lack of trainable parameters. In this paper, we begin by unifying local outlier methods by showing that they are particular cases of the more general message passing framework used in graph neural networks. This allows us to introduce learnability into local outlier methods, in the form of a neural network, for greater flexibility and expressivity: specifically, we propose LUNAR, a novel, graph neural network-based anomaly detection method. LUNAR learns to use information from the nearest neighbours of each node in a trainable way to find anomalies. We show that our method performs significantly better than existing local outlier methods, as well as state-of-the-art deep baselines. We also show that the performance of our method is much more robust to different settings of the local neighbourhood size.

----

## [757] Semi-supervised Conditional Density Estimation with Wasserstein Laplacian Regularisation

**Authors**: *Olivier Graffeuille, Yun Sing Koh, Jörg Wicker, Moritz K. Lehmann*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20630](https://doi.org/10.1609/aaai.v36i6.20630)

**Abstract**:

Conditional Density Estimation (CDE) has wide-reaching applicability to various real-world problems, such as spatial density estimation and environmental modelling. CDE estimates the probability density of a random variable rather than a single value and can thus model uncertainty and inverse problems. This task is inherently more complex than regression, and many algorithms suffer from overfitting, particularly when modelled with few labelled data points. For applications where unlabelled data is abundant but labelled data is scarce, we propose Wasserstein Laplacian Regularisation, a semi-supervised learning framework that allows CDE algorithms to leverage these unlabelled data. The framework minimises an objective function which ensures that the learned model is smooth along the manifold of the underlying data, as measured by Wasserstein distance. When applying our framework to Mixture Density Networks, the resulting semi-supervised algorithm can achieve similar performance to a supervised model with up to three times as many labelled data points on baseline datasets. We additionally apply our technique to the problem of remote sensing for chlorophyll-a estimation in inland waters.

----

## [758] GoTube: Scalable Statistical Verification of Continuous-Depth Models

**Authors**: *Sophie A. Gruenbacher, Mathias Lechner, Ramin M. Hasani, Daniela Rus, Thomas A. Henzinger, Scott A. Smolka, Radu Grosu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20631](https://doi.org/10.1609/aaai.v36i6.20631)

**Abstract**:

We introduce a new statistical verification algorithm that formally quantifies the behavioral robustness of any time-continuous process formulated as a continuous-depth model. Our algorithm solves a set of global optimization (Go) problems over a given time horizon to construct a tight enclosure (Tube) of the set of all process executions starting from a ball of initial states. We call our algorithm GoTube. Through its construction, GoTube ensures that the bounding tube is conservative up to a desired probability and up to a desired tightness.
 GoTube is implemented in JAX and optimized to scale to complex continuous-depth neural network models. Compared to advanced reachability analysis tools for time-continuous neural networks, GoTube does not accumulate overapproximation errors between time steps and avoids the infamous wrapping effect inherent in symbolic techniques. We show that GoTube substantially outperforms state-of-the-art verification tools in terms of the size of the initial ball, speed, time-horizon, task completion, and scalability on a large set of experiments.
 GoTube is stable and sets the state-of-the-art in terms of its ability to scale to time horizons well beyond what has been previously possible.

----

## [759] Balanced Self-Paced Learning for AUC Maximization

**Authors**: *Bin Gu, Chenkang Zhang, Huan Xiong, Heng Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20632](https://doi.org/10.1609/aaai.v36i6.20632)

**Abstract**:

Learning to improve AUC performance is an important topic in machine learning. However, AUC maximization algorithms may decrease generalization performance due to the noisy data. Self-paced learning is an effective method for handling noisy data. However, existing self-paced learning methods are limited to pointwise learning, while AUC maximization is a pairwise learning problem. To solve this challenging problem, we innovatively propose a balanced self-paced AUC maximization algorithm (BSPAUC). Specifically, we first provide a statistical objective for self-paced AUC.
 Based on this, we propose our self-paced AUC maximization formulation, where a novel balanced self-paced regularization term is embedded to ensure that the selected positive and negative samples have proper proportions. Specially, the sub-problem with respect to all weight variables may be non-convex in our formulation, while the one is normally convex in existing self-paced problems. To address this, we propose a doubly cyclic block coordinate descent method.
 More importantly, we prove that the sub-problem with respect to all weight variables converges to a stationary point on the basis of closed-form solutions, and our BSPAUC converges to a stationary point of our fixed optimization objective under a mild assumption. Considering both the deep learning and kernel-based implementations, experimental results on several large-scale datasets demonstrate that our BSPAUC has a better generalization performance than existing state-of-the-art AUC maximization methods.

----

## [760] Theoretical Guarantees of Fictitious Discount Algorithms for Episodic Reinforcement Learning and Global Convergence of Policy Gradient Methods

**Authors**: *Xin Guo, Anran Hu, Junzi Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20633](https://doi.org/10.1609/aaai.v36i6.20633)

**Abstract**:

When designing algorithms for finite-time-horizon episodic reinforcement learning problems, a common approach is to introduce a fictitious discount factor and use stationary policies for approximations. Empirically, it has been shown that the fictitious discount factor helps reduce variance, and stationary policies serve to save the per-iteration computational cost. Theoretically, however, there is no existing work on convergence analysis for algorithms with this fictitious discount recipe. This paper takes the first step towards analyzing these algorithms. It focuses on two vanilla policy gradient (VPG) variants: the first being a widely used variant with discounted advantage estimations (DAE), the second with an additional fictitious discount factor in the score functions of the policy gradient estimators. Non-asymptotic convergence guarantees are established for both algorithms, and the additional discount factor is shown to reduce the bias introduced in DAE and thus improve the algorithm convergence asymptotically. A key ingredient of our analysis is to connect three settings of Markov decision processes (MDPs): the finite-time-horizon, the average reward and the discounted settings. To our best knowledge, this is the first theoretical guarantee on fictitious discount algorithms for the episodic reinforcement learning of finite-time-horizon MDPs, which also leads to the (first) global convergence of policy gradient methods for finite-time-horizon episodic reinforcement learning.

----

## [761] Adaptive Orthogonal Projection for Batch and Online Continual Learning

**Authors**: *Yiduo Guo, Wenpeng Hu, Dongyan Zhao, Bing Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20634](https://doi.org/10.1609/aaai.v36i6.20634)

**Abstract**:

Catastrophic forgetting is a key obstacle to continual learning. One of the state-of-the-art approaches is orthogonal projection. The idea of this approach is to learn each task by updating the network parameters or weights only in the direction orthogonal to the subspace spanned by all previous task inputs. This ensures no interference with tasks that have been learned. The system OWM that uses the idea performs very well against other state-of-the-art systems. In this paper, we first discuss an issue that we discovered in the mathematical derivation of this approach and then propose a novel method, called AOP (Adaptive Orthogonal Projection), to resolve it, which results in significant accuracy gains in empirical evaluations in both the batch and online continual learning settings without saving any previous training data as in replay-based methods.

----

## [762] Learning Action Translator for Meta Reinforcement Learning on Sparse-Reward Tasks

**Authors**: *Yijie Guo, Qiucheng Wu, Honglak Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20635](https://doi.org/10.1609/aaai.v36i6.20635)

**Abstract**:

Meta reinforcement learning (meta-RL) aims to learn a policy solving a set of training tasks simultaneously and quickly adapting to new tasks. It requires massive amounts of data drawn from training tasks to infer the common structure shared among tasks. Without heavy reward engineering, the sparse rewards in long-horizon tasks exacerbate the problem of sample efficiency in meta-RL. Another challenge in meta-RL is the discrepancy of difficulty level among tasks, which might cause one easy task dominating learning of the shared policy and thus preclude policy adaptation to new tasks. This work introduces a novel objective function to learn an action translator among training tasks. We theoretically verify that the value of the transferred policy with the action translator can be close to the value of the source policy and our objective function (approximately) upper bounds the value difference. We propose to combine the action translator with context-based meta-RL algorithms for better data collection and moreefficient exploration during meta-training. Our approach em-pirically improves the sample efficiency and performance ofmeta-RL algorithms on sparse-reward tasks.

----

## [763] Self-Supervised Pre-training for Protein Embeddings Using Tertiary Structures

**Authors**: *Yuzhi Guo, Jiaxiang Wu, Hehuan Ma, Junzhou Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20636](https://doi.org/10.1609/aaai.v36i6.20636)

**Abstract**:

The protein tertiary structure largely determines its interaction with other molecules. Despite its importance in various structure-related tasks, fully-supervised data are often time-consuming and costly to obtain. Existing pre-training models mostly focus on amino-acid sequences or multiple sequence alignments, while the structural information is not yet exploited. In this paper, we propose a self-supervised pre-training model for learning structure embeddings from protein tertiary structures. Native protein structures are perturbed with random noise, and the pre-training model aims at estimating gradients over perturbed 3D structures. Specifically, we adopt SE(3)-invariant features as model inputs and reconstruct gradients over 3D coordinates with SE(3)-equivariance preserved. Such paradigm avoids the usage of sophisticated SE(3)-equivariant models, and dramatically improves the computational efficiency of pre-training models. We demonstrate the effectiveness of our pre-training model on two downstream tasks, protein structure quality assessment (QA) and protein-protein interaction (PPI) site prediction. Hierarchical structure embeddings are extracted to enhance corresponding prediction models. Extensive experiments indicate that such structure embeddings consistently improve the prediction accuracy for both downstream tasks.

----

## [764] Improved Gradient-Based Adversarial Attacks for Quantized Networks

**Authors**: *Kartik Gupta, Thalaiyasingam Ajanthan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20637](https://doi.org/10.1609/aaai.v36i6.20637)

**Abstract**:

Neural network quantization has become increasingly popular due to efficient memory consumption and faster computation resulting from bitwise operations on the quantized networks. Even though they exhibit excellent generalization capabilities, their robustness properties are not well-understood. In this work, we systematically study the robustness of quantized networks against gradient based adversarial attacks and demonstrate that these quantized models suffer from gradient vanishing issues and show a fake sense of robustness. By attributing gradient vanishing to poor forward-backward signal propagation in the trained network, we introduce a simple temperature scaling approach to mitigate this issue while preserving the decision boundary. Despite being a simple modification to existing gradient based adversarial attacks, experiments on multiple image classification datasets with multiple network architectures demonstrate that our temperature scaled attacks obtain near-perfect success rate on quantized networks while outperforming original attacks on adversarially trained models as well as floating-point networks.

----

## [765] TIGGER: Scalable Generative Modelling for Temporal Interaction Graphs

**Authors**: *Shubham Gupta, Sahil Manchanda, Srikanta Bedathur, Sayan Ranu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20638](https://doi.org/10.1609/aaai.v36i6.20638)

**Abstract**:

There has been a recent surge in learning generative models for graphs. While impressive progress has been made on static graphs, work on generative modeling of temporal graphs is at a nascent stage with significant scope for improvement. First, existing generative models do not scale with either the time horizon or the number of nodes. Second, existing techniques are transductive in nature and thus do not facilitate knowledge transfer. Finally, due to relying on one-to-one node mapping from source to the generated graph, existing models leak node identity information and do not allow up-scaling/down-scaling the source graph size. In this paper, we bridge these gaps with a novel generative model called TIGGER. TIGGER derives its power through a combination of temporal point processes with auto-regressive modeling enabling both transductive and inductive variants. Through extensive experiments on real datasets, we establish TIGGER generates graphs of superior fidelity, while also being up to 3 orders of magnitude faster than the state-of-the-art.

----

## [766] A Generalized Bootstrap Target for Value-Learning, Efficiently Combining Value and Feature Predictions

**Authors**: *Anthony GX-Chen, Veronica Chelu, Blake A. Richards, Joelle Pineau*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20639](https://doi.org/10.1609/aaai.v36i6.20639)

**Abstract**:

Estimating value functions is a core component of reinforcement learning algorithms. Temporal difference (TD) learning algorithms use bootstrapping, i.e. they update the value function toward a learning target using value estimates at subsequent time-steps. Alternatively, the value function can be updated toward a learning target constructed by separately predicting successor features (SF)—a policy-dependent model—and linearly combining them with instantaneous rewards. 
 
We focus on bootstrapping targets used when estimating value functions, and propose a new backup target, the ?-return mixture, which implicitly combines value-predictive knowledge (used by TD methods) with (successor) feature-predictive knowledge—with a parameter ? capturing how much to rely on each. We illustrate that incorporating predictive knowledge through an ??-discounted SF model makes more efficient use of sampled experience, compared to either extreme, i.e. bootstrapping entirely on the value function estimate, or bootstrapping on the product of separately estimated successor features and instantaneous reward models. We empirically show this approach leads to faster policy evaluation and better control performance, for tabular and nonlinear function approximations, indicating scalability and generality.

----

## [767] Oscillatory Fourier Neural Network: A Compact and Efficient Architecture for Sequential Processing

**Authors**: *Bing Han, Cheng Wang, Kaushik Roy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20640](https://doi.org/10.1609/aaai.v36i6.20640)

**Abstract**:

Tremendous progress has been made in sequential processing with the recent advances in recurrent neural networks. However, recurrent architectures face the challenge of exploding/vanishing gradients during training, and require significant computational resources to execute back-propagation through time. Moreover, large models are typically needed for executing complex sequential tasks. To address these challenges, we propose a novel neuron model that has cosine activation with a time varying component for sequential processing. The proposed neuron provides an efficient building block for projecting sequential inputs into spectral domain, which helps to retain long-term dependencies with minimal extra model parameters and computation. A new type of recurrent network architecture, named Oscillatory Fourier Neural Network, based on the proposed neuron is presented and applied to various types of sequential tasks. We demonstrate that recurrent neural network with the proposed neuron model is mathematically equivalent to a simplified form of discrete Fourier transform applied onto periodical activation. In particular, the computationally intensive back-propagation through time in training is eliminated, leading to faster training while achieving the state of the art inference accuracy in a diverse group of sequential tasks. For instance, applying the proposed model to sentiment analysis on IMDB review dataset reaches 89.4% test accuracy within 5 epochs, accompanied by over 35x reduction in the model size compared to LSTM. The proposed novel RNN architecture is well poised for intelligent sequential processing in resource constrained hardware.

----

## [768] End-to-End Probabilistic Label-Specific Feature Learning for Multi-Label Classification

**Authors**: *Jun-Yi Hang, Min-Ling Zhang, Yanghe Feng, Xiaocheng Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20641](https://doi.org/10.1609/aaai.v36i6.20641)

**Abstract**:

Label-specific features serve as an effective strategy to learn from multi-label data with tailored features accounting for the distinct discriminative properties of each class label. Existing prototype-based label-specific feature transformation approaches work in a three-stage framework, where prototype acquisition, label-specific feature generation and classification model induction are performed independently. Intuitively, this separate framework is suboptimal due to its decoupling nature. In this paper, we make a first attempt towards a unified framework for prototype-based label-specific feature transformation, where the prototypes and the label-specific features are directly optimized for classification. To instantiate it, we propose modelling the prototypes probabilistically by the normalizing flows, which possess adaptive prototypical complexity to fully capture the underlying properties of each class label and allow for scalable stochastic optimization. Then, a label correlation regularized probabilistic latent metric space is constructed via jointly learning the prototypes and the metric-based label-specific features for classification. Comprehensive experiments on 14 benchmark data sets show that our approach outperforms the state-of-the-art counterparts.

----

## [769] Cross-Domain Few-Shot Graph Classification

**Authors**: *Kaveh Hassani*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20642](https://doi.org/10.1609/aaai.v36i6.20642)

**Abstract**:

We study the problem of few-shot graph classification across domains with nonequivalent feature spaces by introducing three new cross-domain benchmarks constructed from publicly available datasets. We also propose an attention-based graph encoder that uses three congruent views of graphs, one contextual and two topological views, to learn representations of task-specific information for fast adaptation, and task-agnostic information for knowledge transfer. We run exhaustive experiments to evaluate the performance of contrastive and meta-learning strategies. We show that when coupled with metric-based meta-learning frameworks, the proposed encoder achieves the best average meta-test classification accuracy across all benchmarks.

----

## [770] SpreadGNN: Decentralized Multi-Task Federated Learning for Graph Neural Networks on Molecular Data

**Authors**: *Chaoyang He, Emir Ceyani, Keshav Balasubramanian, Murali Annavaram, Salman Avestimehr*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20643](https://doi.org/10.1609/aaai.v36i6.20643)

**Abstract**:

Graph Neural Networks (GNNs) are the first choice methods for graph machine learning problems thanks to their ability to learn state-of-the-art level representations from graph-structured data. However, centralizing a massive amount of real-world graph data for GNN training is prohibitive due to user-side privacy concerns, regulation restrictions, and commercial competition. Federated Learning is the de-facto standard for collaborative training of machine learning models over many distributed edge devices without the need for centralization. Nevertheless, training graph neural networks in a federated setting is vaguely defined and brings statistical and systems challenges. This work proposes SpreadGNN, a novel multi-task federated training framework capable of operating in the presence of partial labels and absence of a central server for the first time in the literature. We provide convergence guarantees and empirically demonstrate the efficacy of our framework on a variety of non-I.I.D. distributed graph-level molecular property prediction datasets with partial labels. Our results show that SpreadGNN outperforms GNN models trained over a central server-dependent federated learning system, even in constrained topologies.

----

## [771] Not All Parameters Should Be Treated Equally: Deep Safe Semi-supervised Learning under Class Distribution Mismatch

**Authors**: *Rundong He, Zhongyi Han, Yang Yang, Yilong Yin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20644](https://doi.org/10.1609/aaai.v36i6.20644)

**Abstract**:

Deep semi-supervised learning (SSL) aims to utilize a sizeable unlabeled set to train deep networks, thereby reducing the dependence on labeled instances. However, the unlabeled set often carries unseen classes that cause the deep SSL algorithm to lose generalization. Previous works focus on the data level that they attempt to remove unseen class data or assign lower weight to them but could not eliminate their adverse effects on the SSL algorithm. Rather than focusing on the data level, this paper turns attention to the model parameter level. We find that only partial parameters are essential for seen-class classification, termed safe parameters. In contrast, the other parameters tend to fit irrelevant data, termed harmful parameters. Driven by this insight, we propose Safe Parameter Learning (SPL) to discover safe parameters and make the harmful parameters inactive, such that we can mitigate the adverse effects caused by unseen-class data. Specifically, we firstly design an effective strategy to divide all parameters in the pre-trained SSL model into safe and harmful ones. Then, we introduce a bi-level optimization strategy to update the safe parameters and kill the harmful parameters. Extensive experiments show that SPL outperforms the state-of-the-art SSL methods on all the benchmarks by a large margin. Moreover, experiments demonstrate that SPL can be integrated into the most popular deep SSL networks and be easily extended to handle other cases of class distribution mismatch.

----

## [772] Wasserstein Unsupervised Reinforcement Learning

**Authors**: *Shuncheng He, Yuhang Jiang, Hongchang Zhang, Jianzhun Shao, Xiangyang Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20645](https://doi.org/10.1609/aaai.v36i6.20645)

**Abstract**:

Unsupervised reinforcement learning aims to train agents to learn a handful of policies or skills in environments without external reward. These pre-trained policies can accelerate learning when endowed with external reward, and can also be used as primitive options in hierarchical reinforcement learning. Conventional approaches of unsupervised skill discovery feed a latent variable to the agent and shed its empowerment on agent’s behavior by mutual information (MI) maximization. However, the policies learned by MI-based methods cannot sufficiently explore the state space, despite they can be successfully identified from each other. Therefore we propose a new framework Wasserstein unsupervised reinforcement learning (WURL) where we directly maximize the distance of state distributions induced by different policies. Additionally, we overcome difficulties in simultaneously training N(N>2) policies, and amortizing the overall reward to each step. Experiments show policies learned by our approach outperform MI-based methods on the metric of Wasserstein distance while keeping high discriminability. Furthermore, the agents trained by WURL can sufficiently explore the state space in mazes and MuJoCo tasks and the pre-trained policies can be applied to downstream tasks by hierarchical learning.

----

## [773] Multi-Mode Tensor Space Clustering Based on Low-Tensor-Rank Representation

**Authors**: *Yicong He, George K. Atia*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20646](https://doi.org/10.1609/aaai.v36i6.20646)

**Abstract**:

Traditional subspace clustering aims to cluster data lying in a union of linear subspaces. The vectorization of high-dimensional data to 1-D vectors to perform clustering ignores much of the structure intrinsic to such data. To preserve said structure, in this work we exploit clustering in a high-order tensor space rather than a vector space. We develop a novel low-tensor-rank representation (LTRR) for unfolded matrices of tensor data lying in a low-rank tensor space. The representation coefficient matrix of an unfolding matrix is tensorized to a 3-order tensor, and the low-tensor-rank constraint is imposed on the transformed coefficient tensor to exploit the self-expressiveness property. Then, inspired by the multi-view clustering framework, we develop a multi-mode tensor space clustering algorithm (MMTSC) that can deal with tensor space clustering with or without missing entries. The tensor is unfolded along each mode, and the coefficient matrices are obtained for each unfolded matrix. The low tensor rank constraint is imposed on a tensor combined from transformed coefficient tensors of each mode, such that the proposed method can simultaneously capture the low rank property for the data within each tensor space and maintain cluster consistency across different modes. Experimental results demonstrate that the proposed MMTSC algorithm can outperform existing clustering algorithms in many cases.

----

## [774] Toward Physically Realizable Quantum Neural Networks

**Authors**: *Mohsen Heidari, Ananth Grama, Wojciech Szpankowski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20647](https://doi.org/10.1609/aaai.v36i6.20647)

**Abstract**:

There has been significant recent interest in quantum neural networks (QNNs), along with their applications in diverse domains. Current solutions for QNNs pose significant challenges concerning their scalability, ensuring that the postulates of quantum mechanics are satisfied and that the networks are physically realizable. The exponential state space of QNNs poses challenges for the scalability of training procedures. The no-cloning principle prohibits making multiple copies of training samples, and the measurement postulates lead to non-deterministic loss functions. Consequently, the physical realizability and efficiency of existing approaches that rely on repeated measurement of several copies of each sample for training QNNs are unclear. This paper presents a new model for QNNs that relies on band-limited Fourier expansions of transfer functions of quantum perceptrons (QPs) to design scalable training procedures. This training procedure is augmented with a randomized quantum stochastic gradient descent technique that eliminates the need for sample replication. We show that this training procedure converges to the true minima in expectation, even in the presence of non-determinism due to quantum measurement. Our solution has a number of important benefits: (i) using QPs with concentrated Fourier power spectrum, we show that the training procedure for QNNs can be made scalable; (ii) it eliminates the need for resampling, thus staying consistent with the no-cloning rule; and (iii) enhanced data efficiency for the overall training process since each data sample is processed once per epoch. We present a detailed theoretical foundation for our models and methods' scalability, accuracy, and data efficiency. We also validate the utility of our approach through a series of numerical experiments.

----

## [775] Reinforcement Learning of Causal Variables Using Mediation Analysis

**Authors**: *Tue Herlau, Rasmus Larsen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20648](https://doi.org/10.1609/aaai.v36i6.20648)

**Abstract**:

We consider the problem of acquiring causal representations and concepts in a reinforcement learning setting. 
 Our approach defines a causal variable as being both manipulable by a policy, and able to predict the outcome. 
 We thereby obtain a parsimonious causal graph in which interventions occur at the level of policies.
 The approach avoids defining a generative model of the data, prior pre-processing, or learning the transition kernel of the Markov decision process. 
 Instead, causal variables and policies are determined by maximizing a new optimization target inspired by mediation analysis, which differs from the expected return. 
 The maximization is accomplished using a generalization of Bellman's equation which is shown to converge, and the method finds meaningful causal representations in a simulated environment.

----

## [776] Anytime Guarantees under Heavy-Tailed Data

**Authors**: *Matthew J. Holland*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20649](https://doi.org/10.1609/aaai.v36i6.20649)

**Abstract**:

Under data distributions which may be heavy-tailed, many stochastic gradient-based learning algorithms are driven by feedback queried at points with almost no performance guarantees on their own. Here we explore a modified "anytime online-to-batch" mechanism which for smooth objectives admits high-probability error bounds while requiring only lower-order moment bounds on the stochastic gradients. Using this conversion, we can derive a wide variety of "anytime robust" procedures, for which the task of performance analysis can be effectively reduced to regret control, meaning that existing regret bounds (for the bounded gradient case) can be robustified and leveraged in a straightforward manner. As a direct takeaway, we obtain an easily implemented stochastic gradient-based algorithm for which all queried points formally enjoy sub-Gaussian error bounds, and in practice show noteworthy gains on real-world data applications.

----

## [777] Adversarial Examples Can Be Effective Data Augmentation for Unsupervised Machine Learning

**Authors**: *Chia-Yi Hsu, Pin-Yu Chen, Songtao Lu, Sijia Liu, Chia-Mu Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20650](https://doi.org/10.1609/aaai.v36i6.20650)

**Abstract**:

Adversarial examples causing evasive predictions are widely used to evaluate and improve the robustness of machine learning models. However, current studies focus on supervised learning tasks, relying on the ground truth data label, a targeted objective, or supervision from a trained classifier. In this paper, we propose a framework of generating adversarial examples for unsupervised models and demonstrate novel applications to data augmentation. Our framework exploits a mutual information neural estimator as an information theoretic similarity measure to generate adversarial examples without supervision. We propose a new MinMax algorithm with provable convergence guarantees for the efficient generation of unsupervised adversarial examples. Our framework can also be extended to supervised adversarial examples. When using unsupervised adversarial examples as a simple plugin data augmentation tool for model retraining, significant improvements are consistently observed across different unsupervised tasks and datasets, including data reconstruction, representation learning, and contrastive learning. Our results show novel methods and considerable advantages in studying and improving unsupervised machine learning via adversarial examples.

----

## [778] Towards Automating Model Explanations with Certified Robustness Guarantees

**Authors**: *Mengdi Huai, Jinduo Liu, Chenglin Miao, Liuyi Yao, Aidong Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20651](https://doi.org/10.1609/aaai.v36i6.20651)

**Abstract**:

Providing model explanations has gained significant popularity recently. In contrast with the traditional feature-level model explanations, concept-based explanations can provide explanations in the form of high-level human concepts. However, existing concept-based explanation methods implicitly follow a two-step procedure that involves human intervention. Specifically, they first need the human to be involved to define (or extract) the high-level concepts, and then manually compute the importance scores of these identified concepts in a post-hoc way. This laborious process requires significant human effort and resource expenditure due to manual work, which hinders their large-scale deployability. In practice, it is challenging to automatically generate the concept-based explanations without human intervention due to the subjectivity of defining the units of concept-based interpretability. In addition, due to its data-driven nature, the interpretability itself is also potentially susceptible to malicious manipulations. Hence, our goal in this paper is to free human from this tedious process, while ensuring that the generated explanations are provably robust to adversarial perturbations. We propose a novel concept-based interpretation method, which can not only automatically provide the prototype-based concept explanations but also provide certified robustness guarantees for the generated prototype-based explanations. We also conduct extensive experiments on real-world datasets to verify the desirable properties of the proposed method.

----

## [779] Multi-View Clustering on Topological Manifold

**Authors**: *Shudong Huang, Ivor W. Tsang, Zenglin Xu, Jiancheng Lv, Quanhui Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20652](https://doi.org/10.1609/aaai.v36i6.20652)

**Abstract**:

Multi-view clustering has received a lot of attentions in data mining recently. Though plenty of works have been investigated on this topic, it is still a severe challenge due to the complex nature of the multiple heterogeneous features. Particularly, existing multi-view clustering algorithms fail to consider the topological structure in the data, which is essential for clustering data on manifold. In this paper, we propose to exploit the implied data manifold by learning the topological relationship between data points. Our method coalesces multiple view-wise graphs with the topological relevance considered, and learns the weights as well as the consensus graph interactively in a unified framework. Furthermore, we manipulate the consensus graph by a connectivity constraint such that the data points from the same cluster are precisely connected into the same component. Substantial experiments on both toy data and real datasets are conducted to validate the effectiveness of the proposed method, compared to the state-of-the-art algorithms over the clustering performance.

----

## [780] Achieving Counterfactual Fairness for Causal Bandit

**Authors**: *Wen Huang, Lu Zhang, Xintao Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20653](https://doi.org/10.1609/aaai.v36i6.20653)

**Abstract**:

In online recommendation, customers arrive in a sequential and stochastic manner from an underlying distribution and the online decision model recommends a chosen item for each arriving individual based on some strategy. We study how to recommend an item at each step to maximize the expected reward while achieving user-side fairness for customers, i.e., customers who share similar profiles will receive a similar reward regardless of their sensitive attributes and items being recommended. By incorporating causal inference into bandits and adopting soft intervention to model the arm selection strategy, we first propose the d-separation based UCB algorithm (D-UCB) to explore the utilization of the d-separation set in reducing the amount of exploration needed to achieve low cumulative regret. Based on that, we then propose the fair causal bandit (F-UCB) for achieving the counterfactual individual fairness. Both theoretical analysis and empirical evaluation demonstrate effectiveness of our algorithms.

----

## [781] Uncertainty-Aware Learning against Label Noise on Imbalanced Datasets

**Authors**: *Yingsong Huang, Bing Bai, Shengwei Zhao, Kun Bai, Fei Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20654](https://doi.org/10.1609/aaai.v36i6.20654)

**Abstract**:

Learning against label noise is a vital topic to guarantee a reliable performance for deep neural networks.Recent research usually refers to dynamic noise modeling with model output probabilities and loss values, and then separates clean and noisy samples.These methods have gained notable success. However, unlike cherry-picked data, existing approaches often cannot perform well when facing imbalanced datasets, a common scenario in the real world.We thoroughly investigate this phenomenon and point out two major issues that hinder the performance, i.e., inter-class loss distribution discrepancy and misleading predictions due to uncertainty.The first issue is that existing methods often perform class-agnostic noise modeling. However, loss distributions show a significant discrepancy among classes under class imbalance, and class-agnostic noise modeling can easily get confused with noisy samples and samples in minority classes.The second issue refers to that models may output misleading predictions due to epistemic uncertainty and aleatoric uncertainty, thus existing methods that rely solely on the output probabilities may fail to distinguish confident samples. Inspired by our observations, we propose an Uncertainty-aware Label Correction framework(ULC) to handle label noise on imbalanced datasets. First, we perform epistemic uncertainty-aware class-specific noise modeling to identify trustworthy clean samples and refine/discard highly confident true/corrupted labels.Then, we introduce aleatoric uncertainty in the subsequent learning process to prevent noise accumulation in the label noise modeling process. We conduct experiments on several synthetic and real-world datasets. The results demonstrate the effectiveness of the proposed method, especially on imbalanced datasets.

----

## [782] Globally Optimal Hierarchical Reinforcement Learning for Linearly-Solvable Markov Decision Processes

**Authors**: *Guillermo Infante, Anders Jonsson, Vicenç Gómez*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20655](https://doi.org/10.1609/aaai.v36i6.20655)

**Abstract**:

We present a novel approach to hierarchical reinforcement learning for linearly-solvable Markov decision processes. Our approach assumes that the state space is partitioned, and defines subtasks for moving between the partitions. We represent value functions on several levels of abstraction, and use the compositionality of subtasks to estimate the optimal values of the states in each partition. The policy is implicitly defined on these optimal value estimates, rather than being decomposed among the subtasks. As a consequence, our approach can learn the globally optimal policy, and does not suffer from non-stationarities induced by high-level decisions. If several partitions have equivalent dynamics, the subtasks of those partitions can be shared. We show that our approach is significantly more sample efficient than that of a flat learner and similar hierarchical approaches when the set of boundary states is smaller than the entire state space.

----

## [783] Causal Discovery in Hawkes Processes by Minimum Description Length

**Authors**: *Amirkasra Jalaldoust, Katerina Hlavácková-Schindler, Claudia Plant*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20656](https://doi.org/10.1609/aaai.v36i6.20656)

**Abstract**:

Hawkes processes are a special class of temporal point processes which exhibit a natural notion of causality, as occurrence of events in the past may increase the probability of events in the future. Discovery of the underlying inﬂuence network among the dimensions of multi-dimensional temporal processes is of high importance in disciplines where a high-frequency data is to model, e.g. in ﬁnancial data or in seismological data. This paper approaches the problem of learning Granger-causal network in multi-dimensional Hawkes processes. We formulate this problem as a model selection task in which we follow the minimum description length (MDL) principle. Moreover, we propose a general algorithm for MDL-based inference using a Monte-Carlo method and we use it for our causal discovery problem. We compare our algorithm with the state-of-the-art baseline methods on synthetic and real-world ﬁnancial data. The synthetic experiments demonstrate superiority of our method in causal graph discovery compared to the baseline methods with respect to the size of the data. The results of experiments with the G-7 bonds price data are consistent with the experts’ knowledge.

----

## [784] Group-Aware Threshold Adaptation for Fair Classification

**Authors**: *Taeuk Jang, Pengyi Shi, Xiaoqian Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20657](https://doi.org/10.1609/aaai.v36i6.20657)

**Abstract**:

The fairness in machine learning is getting increasing attention, as its applications in different fields continue to expand and diversify. To mitigate the discriminated model behaviors between different demographic groups, we introduce a novel post-processing method to optimize over multiple fairness constraints through group-aware threshold adaptation. We propose to learn adaptive classification thresholds for each demographic group by optimizing the confusion matrix estimated from the probability distribution of a classification model output. As we only need an estimated probability distribution of model output instead of the classification model structure, our post-processing model can be applied to a wide range of classification models and improve fairness in a model-agnostic manner and ensure privacy. This even allows us to post-process existing fairness methods to further improve the trade-off between accuracy and fairness. Moreover, our model has low computational cost. We provide rigorous theoretical analysis on the convergence of our optimization algorithm and the trade-off between accuracy and fairness. Our method theoretically enables a better upper bound in near optimality than previous method under the same condition. Experimental results demonstrate that our method outperforms state-of-the-art methods and obtains the result that is closest to the theoretical accuracy-fairness trade-off boundary.

----

## [785] Towards Discriminant Analysis Classifiers Using Online Active Learning via Myoelectric Interfaces

**Authors**: *Andres G. Jaramillo-Yanez, Marco E. Benalcázar, Sebastian Sardiña, Fabio Zambetta*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20658](https://doi.org/10.1609/aaai.v36i6.20658)

**Abstract**:

We propose a discriminant analysis (DA) classifier that uses online active learning to address the need for the frequent training of myoelectric interfaces due to covariate shift. This online classifier is initially trained using a small set of examples, and then updated over time using streaming data that are interactively labeled by a user or pseudo-labeled by a soft-labeling technique. We prove, theoretically, that this yields the same model as training a DA classifier via full batch learning. We then provide experimental evidence that our approach improves the performance of DA classifiers and is robust to mislabeled data, and that our soft-labeling technique has better performance than existing state-of-the-art methods. We argue that our proposal is suitable for real-time applications, as its time complexity w.r.t. the streaming data remains constant.

----

## [786] Label Hallucination for Few-Shot Classification

**Authors**: *Yiren Jian, Lorenzo Torresani*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20659](https://doi.org/10.1609/aaai.v36i6.20659)

**Abstract**:

Few-shot classification requires adapting knowledge learned from a large annotated base dataset to recognize novel unseen classes, each represented by few labeled examples. In such a scenario, pretraining a network with high capacity on the large dataset and then finetuning it on the few examples causes severe overfitting. At the same time, training a simple linear classifier on top of ``frozen'' features learned from the large labeled dataset fails to adapt the model to the properties of the novel classes, effectively inducing underfitting. In this paper we propose an alternative approach to both of these two popular strategies. First, our method pseudo-labels the entire large dataset using the linear classifier trained on the novel classes. This effectively ``hallucinates'' the novel classes in the large dataset, despite the novel categories not being present in the base database (novel and base classes are disjoint). Then, it finetunes the entire model with a distillation loss on the pseudo-labeled base examples, in addition to the standard cross-entropy loss on the novel dataset. This step effectively trains the network to recognize contextual and appearance cues that are useful for the novel-category recognition but using the entire large-scale base dataset and thus overcoming the inherent data-scarcity problem of few-shot learning. Despite the simplicity of the approach, we show that that our method outperforms the state-of-the-art on four well-established few-shot classification benchmarks. The code is available at https://github.com/yiren-jian/LabelHalluc.

----

## [787] Learning Expected Emphatic Traces for Deep RL

**Authors**: *Ray Jiang, Shangtong Zhang, Veronica Chelu, Adam White, Hado van Hasselt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20660](https://doi.org/10.1609/aaai.v36i6.20660)

**Abstract**:

Off-policy sampling and experience replay are key for improving sample efficiency and scaling model-free temporal difference learning methods. When combined with function approximation, such as neural networks, this combination is known as the deadly triad and is potentially unstable. Recently, it has been shown that stability and good performance at scale can be achieved by combining emphatic weightings and multi-step updates. This approach, however, is generally limited to sampling complete trajectories in order, to compute the required emphatic weighting. In this paper we investigate how to combine emphatic weightings with non-sequential, off-line data sampled from a replay buffer. We develop a multi-step emphatic weighting that can be combined with replay, and a time-reversed n-step TD learning algorithm to learn the required emphatic weighting. We show that these state weightings reduce variance compared with prior approaches, while providing convergence guarantees. We tested the approach at scale on Atari 2600 video games, and observed that the new X-ETD(n) agent improved over baseline agents, highlighting both the scalability and broad applicability of our approach.

----

## [788] Delving into Sample Loss Curve to Embrace Noisy and Imbalanced Data

**Authors**: *Shenwang Jiang, Jianan Li, Ying Wang, Bo Huang, Zhang Zhang, Tingfa Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20661](https://doi.org/10.1609/aaai.v36i6.20661)

**Abstract**:

Corrupted labels and class imbalance are commonly encountered in practically collected training data, which easily leads to over-fitting of deep neural networks (DNNs).  Existing approaches alleviate these issues  by adopting a sample re-weighting strategy, which is to re-weight sample by designing weighting function. However, it is only applicable for training data containing only either one type of data biases.
In practice, however, biased samples with corrupted labels and of tailed classes commonly co-exist in training data.
How to handle them simultaneously is a key but under-explored problem. In this paper, we find that these two types of biased samples, though have similar transient loss, have distinguishable trend and characteristics in loss curves, which could provide valuable priors for sample weight assignment. Motivated by this, we delve into the loss curves and propose a novel probe-and-allocate training strategy: In the probing stage, we train the network on the whole biased training data without intervention, and record the loss curve of each sample as an additional attribute; In the allocating stage, we feed the resulting attribute to a newly designed curve-perception network, named CurveNet, to learn to identify the bias type of each sample and assign proper weights through meta-learning adaptively. 
The training speed of meta learning also blocks its application.
To solve it, we propose a method named skip layer meta optimization (SLMO)  to accelerate training speed by skipping the bottom layers.
Extensive synthetic and real experiments well validate the proposed method, which achieves state-of-the-art performance on multiple challenging benchmarks.

----

## [789] Fast Graph Neural Tangent Kernel via Kronecker Sketching

**Authors**: *Shunhua Jiang, Yunze Man, Zhao Song, Zheng Yu, Danyang Zhuo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20662](https://doi.org/10.1609/aaai.v36i6.20662)

**Abstract**:

Many deep learning tasks need to deal with graph data (e.g., social networks, protein structures, code ASTs). Due to the importance of these tasks, people turned to Graph Neural Networks (GNNs) as the de facto method for machine learning on graph data. GNNs have become widely applied due to their convincing performance. Unfortunately, one major barrier to using GNNs is that GNNs require substantial time and resources to train. Recently, a new method for learning on graph data is Graph Neural Tangent Kernel (GNTK). GNTK is an application of Neural Tangent Kernel (NTK) (a kernel method) on graph data, and solving NTK regression is equivalent to using gradient descent to train an infinite-wide neural network. The key benefit of using GNTK is that, similar to any kernel method, GNTK's parameters can be solved directly in a single step, avoiding time-consuming gradient descent. Meanwhile, sketching has become increasingly used in speeding up various optimization problems, including solving kernel regression. Given a kernel matrix of n graphs, using sketching in solving kernel regression can reduce the running time to o(n^3). But unfortunately such methods usually require extensive knowledge about the kernel matrix beforehand, while in the case of GNTK we find that the construction of the kernel matrix is already O(n^2N^4), assuming each graph has N nodes. The kernel matrix construction time can be a major performance bottleneck when the size of graphs N increases. A natural question to ask is thus whether we can speed up the kernel matrix construction to improve GNTK regression's end-to-end running time. This paper provides the first algorithm to construct the kernel matrix in o(n^2N^3) running time.

----

## [790] Creativity of AI: Automatic Symbolic Option Discovery for Facilitating Deep Reinforcement Learning

**Authors**: *Mu Jin, Zhihao Ma, Kebing Jin, Hankz Hankui Zhuo, Chen Chen, Chao Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20663](https://doi.org/10.1609/aaai.v36i6.20663)

**Abstract**:

Despite of achieving great success in real life, Deep Reinforcement Learning (DRL) is still suffering from three critical issues, which are data efficiency, lack of the interpretability and transferability. Recent research shows that embedding symbolic knowledge into DRL is promising in addressing those challenges. Inspired by this, we introduce a novel deep reinforcement learning framework with symbolic options. This framework features a loop training procedure, which enables guiding the improvement of policy by planning with action models and symbolic options learned from interactive trajectories automatically. The learned symbolic options help doing the dense requirement of expert domain knowledge and provide inherent interpretabiliy of policies. Moreover, the transferability and data efficiency can be further improved by planning with the action models. To validate the effectiveness of this framework, we conduct experiments on two domains, Montezuma's Revenge and Office World respectively, and the results demonstrate the comparable performance, improved data efficiency, interpretability and transferability.

----

## [791] Adaptive Kernel Graph Neural Network

**Authors**: *Mingxuan Ju, Shifu Hou, Yujie Fan, Jianan Zhao, Yanfang Ye, Liang Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20664](https://doi.org/10.1609/aaai.v36i6.20664)

**Abstract**:

Graph neural networks (GNNs) have demonstrated great success in representation learning for graph-structured data. The layer-wise graph convolution in GNNs is shown to be powerful at capturing graph topology. During this process, GNNs are usually guided by pre-defined kernels such as Laplacian matrix, adjacency matrix, or their variants. However, the adoptions of pre-defined kernels may restrain the generalities to different graphs: mismatch between graph and kernel would entail sub-optimal performance. For example, GNNs that focus on low-frequency information may not achieve satisfactory performance when high-frequency information is significant for the graphs, and vice versa. To solve this problem, in this paper, we propose a novel framework - i.e., namely Adaptive Kernel Graph Neural Network (AKGNN) - which learns to adapt to the optimal graph kernel in a unified manner at the first attempt. In the proposed AKGNN, we first design a data-driven graph kernel learning mechanism, which adaptively modulates the balance between all-pass and low-pass filters by modifying the maximal eigenvalue of the graph Laplacian. Through this process, AKGNN learns the optimal threshold between high and low frequency signals to relieve the generality problem. Later, we further reduce the number of parameters by a parameterization trick and enhance the expressive power by a global readout function. Extensive experiments are conducted on acknowledged benchmark datasets and promising results demonstrate the outstanding performance of our proposed AKGNN by comparison with state-of-the-art GNNs. The source code is publicly available at: https://github.com/jumxglhf/AKGNN.

----

## [792] Fully Spiking Variational Autoencoder

**Authors**: *Hiromichi Kamata, Yusuke Mukuta, Tatsuya Harada*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i6.20665](https://doi.org/10.1609/aaai.v36i6.20665)

**Abstract**:

Spiking neural networks (SNNs) can be run on neuromorphic devices with ultra-high speed and ultra-low energy consumption because of their binary and event-driven nature. Therefore, SNNs are expected to have various applications, including as generative models being running on edge devices to create high-quality images. In this study, we build a variational autoencoder (VAE) with SNN to enable image generation. VAE is known for its stability among generative models; recently, its quality advanced. In vanilla VAE, the latent space is represented as a normal distribution, and floating-point calculations are required in sampling. However, this is not possible in SNNs because all features must be binary time series data. Therefore, we constructed the latent space with an autoregressive SNN model, and randomly selected samples from its output to sample the latent variables. This allows the latent variables to follow the Bernoulli process and allows variational learning. Thus, we build the Fully Spiking Variational Autoencoder where all modules are constructed with SNN. To the best of our knowledge, we are the first to build a VAE only with SNN layers. We experimented with several datasets, and confirmed that it can generate images with the same or better quality compared to conventional ANNs. The code is available at https://github.com/kamata1729/FullySpikingVAE.

----

## [793] Classifying Emails into Human vs Machine Category

**Authors**: *Changsung Kang, Hongwei Shang, Jean-Marc Langlois*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20666](https://doi.org/10.1609/aaai.v36i7.20666)

**Abstract**:

It is an essential product requirement of Yahoo Mail to distinguish between personal and machine-generated emails. The old production classifier in Yahoo Mail was based on a simple logistic regression model. That model was trained by aggregating features at the SMTP address level. We propose building deep learning models at the message level. We train four individual CNN models: (1) a content model with subject and content as input, (2) a sender model with sender email address and name as input, (3) an action model by analyzing email recipients’ action patterns and generating target labels based on senders’ opening/deleting behaviors and (4) a salutation model by utilizing senders’ "explicit salutation" signal as positive labels. Next, we train a final full model after exploring different combinations of the above four models. Experimental results on editorial data show that our full model improves the adjusted-recall from 70.5% to 78.8% and the precision from 94.7% to 96.0% compared to the old production model. Also, our full model significantly outperforms a state-of-the-art BERT model at this task. Our new model has been deployed to the current production system (Yahoo Mail 6).

----

## [794] Self-Supervised Enhancement of Latent Discovery in GANs

**Authors**: *Adarsh Kappiyath, Silpa Vadakkeeveetil Sreelatha, S. Sumitra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20667](https://doi.org/10.1609/aaai.v36i7.20667)

**Abstract**:

Several methods for discovering interpretable directions in the latent space of pre-trained GANs have been proposed. Latent semantics discovered by unsupervised methods are less disentangled than supervised methods since they do not use pre-trained attribute classifiers. We propose Scale Ranking Estimator (SRE), which is trained using self-supervision. SRE enhances the disentanglement in directions obtained by existing unsupervised disentanglement techniques. These directions are updated to preserve the ordering of variation within each direction in latent space. Qualitative and quantitative evaluation of the discovered directions demonstrates that our proposed method significantly improves disentanglement in various datasets. We also show that the learned SRE can be used to perform Attribute-based image retrieval task without any training.

----

## [795] Multiple-Source Domain Adaptation via Coordinated Domain Encoders and Paired Classifiers

**Authors**: *Payam Karisani*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20668](https://doi.org/10.1609/aaai.v36i7.20668)

**Abstract**:

We present a novel multiple-source unsupervised model for text classification under domain shift. Our model exploits the update rates in document representations to dynamically integrate domain encoders. It also employs a probabilistic heuristic to infer the error rate in the target domain in order to pair source classifiers. Our heuristic exploits data transformation cost and the classifier accuracy in the target feature space. We have used real world scenarios of Domain Adaptation to evaluate the efficacy of our algorithm. We also used pretrained multi-layer transformers as the document encoder in the experiments to demonstrate whether the improvement achieved by domain adaptation models can be delivered by out-of-the-box language model pretraining. The experiments testify that our model is the top performing approach in this setting.

----

## [796] Instance-Sensitive Algorithms for Pure Exploration in Multinomial Logit Bandit

**Authors**: *Nikolai Karpov, Qin Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20669](https://doi.org/10.1609/aaai.v36i7.20669)

**Abstract**:

Motivated by real-world applications such as fast fashion retailing and online advertising, the Multinomial Logit Bandit (MNL-bandit) is a popular model in online learning and operations research, and has attracted much attention in the past decade. In this paper, we give efficient algorithms for pure exploration in MNL-bandit. Our algorithms achieve instance-sensitive pull complexities. We also complement the upper bounds by an almost matching lower bound.

----

## [797] iDECODe: In-Distribution Equivariance for Conformal Out-of-Distribution Detection

**Authors**: *Ramneet Kaur, Susmit Jha, Anirban Roy, Sangdon Park, Edgar Dobriban, Oleg Sokolsky, Insup Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20670](https://doi.org/10.1609/aaai.v36i7.20670)

**Abstract**:

Machine learning methods such as deep neural networks (DNNs), despite their success across different domains, are known to often generate incorrect predictions with high confidence on inputs outside their training distribution. The deployment of DNNs in safety-critical domains requires detection of out-of-distribution (OOD) data so that DNNs can abstain from making predictions on those. A number of methods have been recently developed for OOD detection, but there is still room for improvement. We propose the new method iDECODe, leveraging in-distribution equivariance for conformal OOD detection. It relies on a novel base non-conformity measure and a new aggregation method, used in the inductive conformal anomaly detection framework, thereby guaranteeing a bounded false detection rate. We demonstrate the efficacy of iDECODe by experiments on image and audio datasets, obtaining state-of-the-art results. We also show that iDECODe can detect adversarial examples. Code, pre-trained models, and data are available at  https://github.com/ramneetk/iDECODe.

----

## [798] Partial Wasserstein Covering

**Authors**: *Keisuke Kawano, Satoshi Koide, Keisuke Otaki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20671](https://doi.org/10.1609/aaai.v36i7.20671)

**Abstract**:

We consider a general task called partial Wasserstein covering with the goal of providing information on what patterns are not being taken into account in a dataset (e.g., dataset used during development) compared to another (e.g., dataset obtained from actual applications). We model this task as a discrete optimization problem with partial Wasserstein divergence as an objective function. Although this problem is NP-hard, we prove that it satisfies the submodular property, allowing us to use a greedy algorithm with a 0.63 approximation. However, the greedy algorithm is still inefficient because it requires solving linear programming for each objective function evaluation. To overcome this inefficiency, we propose quasi-greedy algorithms, which consist of a series of techniques for acceleration such as sensitivity analysis based on strong duality and the so-called C-transform in the optimal transport field. Experimentally, we demonstrate that we can efficiently fill in the gaps between the two datasets, and find missing scene in real driving scene datasets.

----

## [799] Optimal Tensor Transport

**Authors**: *Tanguy Kerdoncuff, Rémi Emonet, Michaël Perrot, Marc Sebban*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i7.20672](https://doi.org/10.1609/aaai.v36i7.20672)

**Abstract**:

Optimal Transport (OT) has become a popular tool in machine learning to align finite datasets typically lying in the same vector space. To expand the range of possible applications, Co-Optimal Transport (Co-OT) jointly estimates two distinct transport plans, one for the rows (points) and one for the columns (features), to match two data matrices that might use different features. On the other hand, Gromov Wasserstein (GW) looks for a single transport plan from two pairwise intra-domain distance matrices. Both Co-OT and GW can be seen as specific extensions of OT to more complex data. In this paper, we propose a unified framework, called Optimal Tensor Transport (OTT), which takes the form of a generic formulation that encompasses OT, GW and Co-OT and can handle tensors of any order by learning possibly multiple transport plans. We derive theoretical results for the resulting new distance and present an efficient way for computing it. We further illustrate the interest of such a formulation in Domain Adaptation and Comparison-based Clustering.

----



[Go to the previous page](AAAI-2022-list03.md)

[Go to the next page](AAAI-2022-list05.md)

[Go to the catalog section](README.md)