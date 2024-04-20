## [1200] Unsupervised Neighborhood Propagation Kernel Layers for Semi-supervised Node Classification

**Authors**: *Sonny Achten, Francesco Tonin, Panagiotis Patrinos, Johan A. K. Suykens*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28949](https://doi.org/10.1609/aaai.v38i10.28949)

**Abstract**:

We present a deep Graph Convolutional Kernel Machine (GCKM) for semi-supervised node classification in graphs. The method is built of two main types of blocks: (i) We introduce unsupervised kernel machine layers propagating the node features in a one-hop neighborhood, using implicit node feature mappings. (ii) We specify a semi-supervised classification kernel machine through the lens of the Fenchel-Young inequality. We derive an effective initialization scheme and efficient end-to-end training algorithm in the dual variables for the full architecture. The main idea underlying GCKM is that, because of the unsupervised core, the final model can achieve higher performance in semi-supervised node classification when few labels are available for training. Experimental results demonstrate the effectiveness of the proposed framework.

----

## [1201] No Prejudice! Fair Federated Graph Neural Networks for Personalized Recommendation

**Authors**: *Nimesh Agrawal, Anuj Kumar Sirohi, Sandeep Kumar, Jayadeva*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28950](https://doi.org/10.1609/aaai.v38i10.28950)

**Abstract**:

Ensuring fairness in Recommendation Systems (RSs) across demographic groups is critical due to the increased integration of RSs in applications such as personalized healthcare, finance, and e-commerce. Graph-based RSs play a crucial role in capturing intricate higher-order interactions among entities. However, integrating these graph models into the Federated Learning (FL) paradigm with fairness constraints poses formidable challenges as this requires access to the entire interaction graph and sensitive user information (such as gender, age, etc.) at the central server. This paper addresses the pervasive issue of inherent bias within RSs for different demographic groups without compromising the privacy of sensitive user attributes in FL environment with the graph-based model. To address the group bias, we propose F2PGNN (Fair Federated Personalized Graph Neural Network), a novel framework that leverages the power of Personalized Graph Neural Network (GNN) coupled with fairness considerations. Additionally, we use differential privacy techniques to fortify privacy protection. Experimental evaluation on three publicly available datasets showcases the efficacy of F2PGNN in mitigating group unfairness by 47% ∼ 99% compared to the state-of-the-art while preserving privacy and maintaining the utility. The results validate the significance of our framework in achieving equitable and personalized recommendations using GNN within the FL landscape. Source code is at: https://github.com/nimeshagrawal/F2PGNN-AAAI24

----

## [1202] Pareto Front-Diverse Batch Multi-Objective Bayesian Optimization

**Authors**: *Alaleh Ahmadianshalchi, Syrine Belakaria, Janardhan Rao Doppa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28951](https://doi.org/10.1609/aaai.v38i10.28951)

**Abstract**:

We consider the problem of multi-objective optimization (MOO) of expensive black-box functions with the goal of discovering high-quality and diverse Pareto fronts where we are allowed to evaluate a batch of inputs. This problem arises in many real-world applications including penicillin production where diversity of solutions is critical. We solve this problem in the framework of Bayesian optimization (BO) and propose a novel approach referred to as Pareto front-Diverse Batch Multi-Objective BO (PDBO). PDBO tackles two important challenges: 1) How to automatically select the best acquisition function in each BO iteration, and 2) How to select a diverse batch of inputs by considering multiple objectives. We propose principled solutions to address these two challenges. First, PDBO employs a multi-armed bandit approach to select one acquisition function from a given library. We solve a cheap MOO problem by assigning the selected acquisition function for each expensive objective function to obtain a candidate set of inputs for evaluation. Second, it utilizes Determinantal Point Processes (DPPs) to choose a Pareto-front-diverse batch of inputs for evaluation from the candidate set obtained from the first step. The key parameters for the methods behind these two steps are updated after each round of function evaluations. Experiments on multiple MOO benchmarks demonstrate that PDBO outperforms prior methods in terms of both the quality and diversity of Pareto solutions.

----

## [1203] SimCS: Simulation for Domain Incremental Online Continual Segmentation

**Authors**: *Motasem Alfarra, Zhipeng Cai, Adel Bibi, Bernard Ghanem, Matthias Müller*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28952](https://doi.org/10.1609/aaai.v38i10.28952)

**Abstract**:

Continual Learning is a step towards lifelong intelligence where models continuously learn from recently collected data without forgetting previous knowledge. Existing continual learning approaches mostly focus on image classification in the class-incremental setup with clear task boundaries and unlimited computational budget. This work explores the problem of Online Domain-Incremental Continual Segmentation (ODICS), where the model is continually trained over batches of densely labeled images from different domains, with limited computation and no information about the task boundaries. ODICS arises in many practical applications. In autonomous driving, this may correspond to the realistic scenario of training a segmentation model over time on a sequence of cities. We analyze several existing continual learning methods and show that they perform poorly in this setting despite working well in class-incremental segmentation. We propose SimCS, a parameter-free method complementary to existing ones that uses simulated data to regularize continual learning. Experiments show that SimCS provides consistent improvements when combined with different CL methods.

----

## [1204] Sample Efficient Reinforcement Learning with Partial Dynamics Knowledge

**Authors**: *Meshal Alharbi, Mardavij Roozbehani, Munther A. Dahleh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28953](https://doi.org/10.1609/aaai.v38i10.28953)

**Abstract**:

The problem of sample complexity of online reinforcement learning is often studied in the literature without taking into account any partial knowledge about the system dynamics that could potentially accelerate the learning process. In this paper, we study the sample complexity of online Q-learning methods when some prior knowledge about the dynamics is available or can be learned efficiently. We focus on systems that evolve according to an additive disturbance model of the form S_{h+1} = ƒ(S_h, A_h) + W_h, where ƒ represents the underlying system dynamics, and W_h are unknown disturbances independent of states and actions. In the setting of finite episodic Markov decision processes with S states, A actions, and episode length H, we present an optimistic Q-learning algorithm that achieves Õ(Poly(H)√T) regret under perfect knowledge of ƒ, where T is the total number of interactions with the system. This is in contrast to the typical Õ(Poly(H)√SAT) regret for existing Q-learning methods. Further, if only a noisy estimate ƒ_hat of ƒ is available, our method can learn an approximately optimal policy in a number of samples that is independent of the cardinalities of state and action spaces. The sub-optimality gap depends on the approximation error ƒ_hat − ƒ, as well as the Lipschitz constant of the corresponding optimal value function. Our approach does not require modeling of the transition probabilities and enjoys the same memory complexity as model-free methods.

----

## [1205] Understanding and Improving Optimization in Predictive Coding Networks

**Authors**: *Nicholas Alonso, Jeffrey L. Krichmar, Emre Neftci*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28954](https://doi.org/10.1609/aaai.v38i10.28954)

**Abstract**:

Backpropagation (BP), the standard learning algorithm for artificial neural networks, is often considered biologically implausible. In contrast, the standard learning algorithm for predictive coding (PC) models in neuroscience, known as the inference learning algorithm (IL), is a promising, bio-plausible alternative. However, several challenges and questions hinder IL's application to real-world problems. For example, IL is computationally demanding, and without memory-intensive optimizers like Adam, IL may converge to poor local minima. Moreover, although IL can reduce loss more quickly than BP, the reasons for these speedups or their robustness remains unclear. In this paper, we tackle these challenges by 1) altering the standard implementation of PC circuits to substantially reduce computation, 2) developing a novel optimizer that improves the convergence of IL without increasing memory usage, and 3) establishing theoretical results that help elucidate the conditions under which IL is sensitive to second and higher-order information.

----

## [1206] Limited Memory Online Gradient Descent for Kernelized Pairwise Learning with Dynamic Averaging

**Authors**: *Hilal AlQuabeh, William de Vazelhes, Bin Gu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28955](https://doi.org/10.1609/aaai.v38i10.28955)

**Abstract**:

Pairwise learning, an important domain within machine learning, addresses loss functions defined on pairs of training examples, including those in metric learning and AUC maximization. Acknowledging the quadratic growth in computation complexity accompanying pairwise loss as the sample size grows, researchers have turned to online gradient descent (OGD) methods for enhanced scalability. Recently, an OGD algorithm emerged, employing gradient computation involving prior and most recent examples, a step that effectively reduces algorithmic complexity to O(T), with T being the number of received examples. This approach, however, confines itself to linear models while assuming the independence of example arrivals. We introduce a lightweight OGD algorithm that does not require the independence of examples and generalizes to kernel pairwise learning. Our algorithm builds the gradient based on a random example and a moving average representing the past data, which results in a sub-linear regret bound with a complexity of O(T). Furthermore, through the integration of O(√T logT) random Fourier features, the complexity of kernel calculations is effectively minimized. Several experiments with real-world datasets show that the proposed technique outperforms kernel and linear algorithms in offline and online scenarios.

----

## [1207] Faithful Model Explanations through Energy-Constrained Conformal Counterfactuals

**Authors**: *Patrick Altmeyer, Mojtaba Farmanbar, Arie van Deursen, Cynthia C. S. Liem*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28956](https://doi.org/10.1609/aaai.v38i10.28956)

**Abstract**:

Counterfactual explanations offer an intuitive and straightforward way to explain black-box models and offer algorithmic recourse to individuals. To address the need for plausible explanations, existing work has primarily relied on surrogate models to learn how the input data is distributed. This effectively reallocates the task of learning realistic explanations for the data from the model itself to the surrogate. Consequently, the generated explanations may seem plausible to humans but need not necessarily describe the behaviour of the black-box model faithfully. We formalise this notion of faithfulness through the introduction of a tailored evaluation metric and propose a novel algorithmic framework for generating Energy-Constrained Conformal Counterfactuals that are only as plausible as the model permits. Through extensive empirical studies, we demonstrate that ECCCo reconciles the need for faithfulness and plausibility. In particular, we show that for models with gradient access, it is possible to achieve state-of-the-art performance without the need for surrogate models. To do so, our framework relies solely on properties defining the black-box model itself by leveraging recent advances in energy-based modelling and conformal prediction. To our knowledge, this is the first venture in this direction for generating faithful counterfactual explanations. Thus, we anticipate that ECCCo can serve as a baseline for future research. We believe that our work opens avenues for researchers and practitioners seeking tools to better distinguish trustworthy from unreliable models.

----

## [1208] Optimal Transport with Tempered Exponential Measures

**Authors**: *Ehsan Amid, Frank Nielsen, Richard Nock, Manfred K. Warmuth*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28957](https://doi.org/10.1609/aaai.v38i10.28957)

**Abstract**:

In the field of optimal transport, two prominent subfields face each other: (i) unregularized optimal transport, ``a-la-Kantorovich'', which leads to extremely sparse plans but with algorithms that scale poorly, and (ii) entropic-regularized optimal transport, ``a-la-Sinkhorn-Cuturi'', which gets near-linear approximation algorithms but leads to maximally un-sparse plans. In this paper, we show that an extension of the latter to tempered exponential measures, a generalization of exponential families with indirect measure normalization, gets to a very convenient middle ground, with both very fast approximation algorithms and sparsity, which is under control up to sparsity patterns. In addition, our formulation fits naturally in the unbalanced optimal transport problem setting.

----

## [1209] Elijah: Eliminating Backdoors Injected in Diffusion Models via Distribution Shift

**Authors**: *Shengwei An, Sheng-Yen Chou, Kaiyuan Zhang, Qiuling Xu, Guanhong Tao, Guangyu Shen, Siyuan Cheng, Shiqing Ma, Pin-Yu Chen, Tsung-Yi Ho, Xiangyu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28958](https://doi.org/10.1609/aaai.v38i10.28958)

**Abstract**:

Diffusion models (DM) have become state-of-the-art generative models because of their capability of generating high-quality images from noises without adversarial training. However, they are vulnerable to backdoor attacks as reported by recent studies.  When a data input (e.g., some Gaussian noise) is stamped with a trigger (e.g., a white patch), the backdoored model always generates the target image  (e.g., an improper photo). However, effective defense strategies to mitigate backdoors from DMs are underexplored. To bridge this gap, we propose the first backdoor detection and removal framework for DMs. We evaluate our framework Elijah on over hundreds of DMs of 3 types including DDPM, NCSN and LDM, with 13 samplers against 3 existing backdoor attacks. Extensive experiments show that our approach can have close to 100% detection accuracy and reduce the backdoor effects to close to zero without significantly sacrificing the model utility.

----

## [1210] Transfer and Alignment Network for Generalized Category Discovery

**Authors**: *Wenbin An, Feng Tian, Wenkai Shi, Yan Chen, Yaqiang Wu, Qianying Wang, Ping Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28959](https://doi.org/10.1609/aaai.v38i10.28959)

**Abstract**:

Generalized Category Discovery (GCD) is a crucial real-world task that aims to recognize both known and novel categories from an unlabeled dataset by leveraging another labeled dataset with only known categories. Despite the improved performance on known categories, current methods perform poorly on novel categories. We attribute the poor performance to two reasons: biased knowledge transfer between labeled and unlabeled data and noisy representation learning on the unlabeled data. The former leads to unreliable estimation of learning targets for novel categories and the latter hinders models from learning discriminative features. To mitigate these two issues, we propose a Transfer and Alignment Network (TAN), which incorporates two knowledge transfer mechanisms to calibrate the biased knowledge and two feature alignment mechanisms to learn discriminative features.
Specifically, we model different categories with prototypes and transfer the prototypes in labeled data to correct model bias towards known categories. On the one hand, we pull instances with known categories in unlabeled data closer to these prototypes to form more compact clusters and avoid boundary overlap between known and novel categories. On the other hand, we use these prototypes to calibrate noisy prototypes estimated from unlabeled data based on category similarities, which allows for more accurate estimation of prototypes for novel categories that can be used as reliable learning targets later. After knowledge transfer, we further propose two feature alignment mechanisms to acquire both instance- and category-level knowledge from unlabeled data by aligning instance features with both augmented features and the calibrated prototypes, which can boost model performance on both known and novel categories with less noise. Experiments on three benchmark datasets show that our model outperforms SOTA methods, especially on novel categories. Theoretical analysis is provided for an in-depth understanding of our model in general.
Our code and data are available at https://github.com/Lackel/TAN.

----

## [1211] Fluctuation-Based Adaptive Structured Pruning for Large Language Models

**Authors**: *Yongqi An, Xu Zhao, Tao Yu, Ming Tang, Jinqiao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28960](https://doi.org/10.1609/aaai.v38i10.28960)

**Abstract**:

Network Pruning is a promising way to address the huge computing resource demands of the deployment and inference of Large Language Models (LLMs). Retraining-free is important for LLMs' pruning methods. However, almost all of the existing retraining-free pruning approaches for LLMs focus on unstructured pruning, which requires specific hardware support for acceleration. In this paper, we propose a novel retraining-free structured pruning framework for LLMs, named FLAP (FLuctuation-based Adaptive
Structured Pruning). It is hardware-friendly by effectively reducing storage and enhancing inference speed. For effective structured pruning of LLMs, we highlight three critical elements that demand the utmost attention: formulating structured importance metrics, adaptively searching the global compressed model, and implementing compensation mechanisms to mitigate performance loss. First, FLAP determines whether the output feature map is easily recoverable when a column of weight is removed, based on the fluctuation pruning metric. Then it standardizes the importance scores to adaptively determine the global compressed model structure. At last, FLAP adds additional bias terms to recover the output feature maps using the baseline values. We thoroughly evaluate our approach on a variety of language benchmarks. Without any retraining, our method significantly outperforms the state-of-the-art methods, including LLM-Pruner and the extension of Wanda in structured pruning. The code is released at https://github.com/CASIA-IVA-Lab/FLAP.

----

## [1212] Active Learning Guided by Efficient Surrogate Learners

**Authors**: *Yunpyo An, Suyeong Park, Kwang In Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28961](https://doi.org/10.1609/aaai.v38i10.28961)

**Abstract**:

Re-training a deep learning model each time a single data point receives a new label is impractical due to the inherent complexity of the training process. Consequently, existing active learning (AL) algorithms tend to adopt a batch-based approach where, during each AL iteration, a set of data points is collectively chosen for annotation. However, this strategy frequently leads to redundant sampling, ultimately eroding the efficacy of the labeling procedure. In this paper, we introduce a new AL algorithm that harnesses the power of a Gaussian process surrogate in conjunction with the neural network principal learner. Our proposed model adeptly updates the surrogate learner for every new data instance, enabling it to emulate and capitalize on the continuous learning dynamics of the neural network without necessitating a complete retraining of the principal model for each individual label. Experiments on four benchmark datasets demonstrate that this approach yields significant enhancements, either rivaling or aligning with the performance of state-of-the-art techniques.

----

## [1213] Formal Logic Enabled Personalized Federated Learning through Property Inference

**Authors**: *Ziyan An, Taylor T. Johnson, Meiyi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28962](https://doi.org/10.1609/aaai.v38i10.28962)

**Abstract**:

Recent advancements in federated learning (FL) have greatly facilitated the development of decentralized collaborative applications, particularly in the domain of Artificial Intelligence of Things (AIoT). However, a critical aspect missing from the current research landscape is the ability to enable data-driven client models with symbolic reasoning capabilities. Specifically, the inherent heterogeneity of participating client devices poses a significant challenge, as each client exhibits unique logic reasoning properties. Failing to consider these device-specific specifications can result in critical properties being missed in the client predictions, leading to suboptimal performance. In this work, we propose a new training paradigm that leverages temporal logic reasoning to address this issue. Our approach involves enhancing the training process by incorporating mechanically generated logic expressions for each FL client. Additionally, we introduce the concept of aggregation clusters and develop a partitioning algorithm to effectively group clients based on the alignment of their temporal reasoning properties. We evaluate the proposed method on two tasks: a real-world traffic volume prediction task consisting of sensory data from fifteen states and a smart city multi-task prediction utilizing synthetic data. The evaluation results exhibit clear improvements, with performance accuracy improved by up to 54% across all sequential prediction models.

----

## [1214] Generating Universal Adversarial Perturbations for Quantum Classifiers

**Authors**: *Gautham Anil, Vishnu Vinod, Apurva Narayan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28963](https://doi.org/10.1609/aaai.v38i10.28963)

**Abstract**:

Quantum Machine Learning (QML) has emerged as a promising field of research, aiming to leverage the capabilities of quantum computing to enhance existing machine learning methodologies. Recent studies have revealed that, like their classical counterparts, QML models based on Parametrized Quantum Circuits (PQCs) are also vulnerable to adversarial attacks. Moreover, the existence of Universal Adversarial Perturbations (UAPs) in the quantum domain has been demonstrated theoretically in the context of quantum classifiers. In this work, we introduce QuGAP: a novel framework for generating UAPs for quantum classifiers. We conceptualize the notion of additive UAPs for PQC-based classifiers and theoretically demonstrate their existence. We then utilize generative models (QuGAP-A) to craft additive UAPs and experimentally show that quantum classifiers are susceptible to such attacks. Moreover, we formulate a new method for generating unitary UAPs (QuGAP-U) using quantum generative models and a novel loss function based on fidelity constraints. We evaluate the performance of the proposed framework and show that our method achieves state-of-the-art misclassification rates, while maintaining high fidelity between legitimate and adversarial samples.

----

## [1215] Enhancing Training of Spiking Neural Network with Stochastic Latency

**Authors**: *Srinivas Anumasa, Bhaskar Mukhoty, Velibor Bojkovic, Giulia De Masi, Huan Xiong, Bin Gu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28964](https://doi.org/10.1609/aaai.v38i10.28964)

**Abstract**:

Spiking neural networks (SNNs) have garnered significant attention for their low power consumption when deployed on neuromorphic hardware that operates in orders of magnitude lower power than general-purpose hardware. Direct training methods for SNNs come with an inherent latency for which the SNNs are optimized, and in general, the higher the latency, the better the predictive powers of the models, but at the same time, the higher the energy consumption during training and inference. Furthermore, an SNN model optimized for one particular latency does not necessarily perform well in lower latencies, which becomes relevant in scenarios where it is necessary to switch to a lower latency because of the depletion of onboard energy or other operational requirements. In this work, we propose Stochastic Latency Training (SLT), a direct training method for SNNs that optimizes the model for the given latency but simultaneously offers a minimum reduction of predictive accuracy when shifted to lower inference latencies. We provide heuristics for our approach with partial theoretical justification and experimental evidence showing the state-of-the-art performance of our models on datasets such as CIFAR-10, DVS-CIFAR-10,  CIFAR-100, and DVS-Gesture. Our code is available at https://github.com/srinuvaasu/SLT

----

## [1216] Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning against Attribute Inference Attacks

**Authors**: *Caridad Arroyo Arevalo, Sayedeh Leila Noorbakhsh, Yun Dong, Yuan Hong, Binghui Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28965](https://doi.org/10.1609/aaai.v38i10.28965)

**Abstract**:

Federated learning (FL)  has been widely studied recently due to its property to collaboratively train data from different devices without sharing the raw  data. Nevertheless, recent studies show that an adversary can still be possible to infer private information about devices' data, e.g., sensitive attributes such as income, race, and sexual orientation. To mitigate the attribute inference attacks, various existing privacy-preserving FL methods can be adopted/adapted. However, all these existing methods have key limitations: they need to know the FL task in advance, or have intolerable computational overheads or utility losses, or do not have provable privacy guarantees. 

We address these issues and design a task-agnostic privacy-preserving presentation learning method for FL (TAPPFL) against attribute inference attacks. TAPPFL is formulated via information theory. Specifically, 
TAPPFL has two mutual information goals, where one goal learns task-agnostic data representations that contain the least information about the private attribute in each device's data, and the other goal ensures the learnt data representations include as much information as possible about the device data to maintain FL utility. We also derive privacy guarantees of TAPPFL against worst-case attribute inference attacks, as well as the inherent tradeoff between utility preservation and privacy protection. Extensive results on multiple datasets and applications validate the effectiveness of TAPPFL to protect data privacy, maintain the FL utility, and be efficient as well. 
Experimental results also show that TAPPFL outperforms the existing defenses.

----

## [1217] Neural Network Approximators for Marginal MAP in Probabilistic Circuits

**Authors**: *Shivvrat Arya, Tahrima Rahman, Vibhav Gogate*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28966](https://doi.org/10.1609/aaai.v38i10.28966)

**Abstract**:

Probabilistic circuits (PCs) such as sum-product networks efficiently represent large multi-variate probability distributions. They are preferred in practice over other probabilistic representations, such as Bayesian and Markov networks, because PCs can solve marginal inference (MAR) tasks in time that scales linearly in the size of the network. Unfortunately, the most probable explanation (MPE) task and its generalization, the marginal maximum-a-posteriori (MMAP) inference task remain NP-hard in these models. Inspired by the recent work on using neural networks for generating near-optimal solutions to optimization problems such as integer linear programming, we propose an approach that uses neural networks to approximate MMAP inference in PCs. The key idea in our approach is to approximate the cost of an assignment to the query variables using a continuous multilinear function and then use the latter as a loss function. The two main benefits of our new method are that it is self-supervised, and after the neural network is learned, it requires only linear time to output a solution. We evaluate our new approach on several benchmark datasets and show that it outperforms three competing linear time approximations: max-product inference, max-marginal inference, and sequential estimation, which are used in practice to solve MMAP tasks in PCs.

----

## [1218] Generator Assisted Mixture of Experts for Feature Acquisition in Batch

**Authors**: *Vedang Asgaonkar, Aditya Jain, Abir De*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28967](https://doi.org/10.1609/aaai.v38i10.28967)

**Abstract**:

Given a set of observations, feature acquisition is about finding the subset of unobserved features which would enhance accuracy. Such problems has been explored in a sequential setting in prior work. Here, the model receives feedback from every new feature acquireed and chooses to explore more features or to predict. However, sequential acquisition is not feasible in some settings where time is of essence. We consider the problem of feature acquisition in batch, where the subset of features to be queried in batch is chosen based on the currently observed features, and then acquired as a batch, followed by prediction. We solve this problem using several technical innovations. First, we use a feature generator to draw a subset of the synthetic features for some examples, which reduces the cost of oracle queries. Second, to make the feature acquisition problem tractable for the large heterogeneous observed features, we partition the data into buckets, by borrowing tools from locality sensitive hashing and then train a mixture of experts model. Third, we design a tractable lower bound of the original objective.
We use a greedy algorithm combined with model training to solve the underlying problem. Experiments with four datasets shows that our approach outperforms these methods in terms of trade off between accuracy and feature acquisition cost.

----

## [1219] Taming Binarized Neural Networks and Mixed-Integer Programs

**Authors**: *Johannes Aspman, Georgios Korpas, Jakub Marecek*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28968](https://doi.org/10.1609/aaai.v38i10.28968)

**Abstract**:

There has been a great deal of recent interest in binarized neural networks, especially because of their explainability. At the same time, automatic differentiation algorithms such as backpropagation fail for binarized neural networks, which limits their applicability. 
We show that binarized neural networks admit a tame representation
by reformulating the problem of training binarized neural networks as a subadditive dual of a mixed-integer program, which we show to have nice properties. This makes it possible to use the framework of Bolte et al. for implicit differentiation, which offers the possibility for practical implementation of backpropagation in the context of binarized neural networks. 

This approach could also be used for a broader class of mixed-integer programs, beyond the training of binarized neural networks, as encountered in symbolic approaches to AI and beyond.

----

## [1220] Contextual Pandora's Box

**Authors**: *Alexia Atsidakou, Constantine Caramanis, Evangelia Gergatsouli, Orestis Papadigenopoulos, Christos Tzamos*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28969](https://doi.org/10.1609/aaai.v38i10.28969)

**Abstract**:

Pandora’s Box is a fundamental stochastic optimization problem, where the decision-maker must find a good alternative, while minimizing the search cost of exploring the value of each alternative. In the original formulation, it is assumed that accurate distributions are given for the values of all the alternatives, while recent work studies the online variant of Pandora’s Box where the distributions are originally unknown. In this work, we study Pandora’s Box in the online setting, while incorporating context. At each round, we are presented with a number of alternatives each having a context, an exploration cost and an unknown value drawn from an unknown distribution that may change at every round. Our main result is a no-regret algorithm that performs comparably well against the optimal algorithm which knows all prior distributions exactly. Our algorithm works even in the bandit setting where the algorithm never learns the values of the alternatives that were not explored. The key technique that enables our result is a novel modification of the realizability condition in contextual bandits that connects a context to a sufficient statistic of each alternative’s distribution (its reservation value) rather than its mean.

----

## [1221] Contextual Pre-planning on Reward Machine Abstractions for Enhanced Transfer in Deep Reinforcement Learning

**Authors**: *Guy Azran, Mohamad H. Danesh, Stefano V. Albrecht, Sarah Keren*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28970](https://doi.org/10.1609/aaai.v38i10.28970)

**Abstract**:

Recent studies show that deep reinforcement learning (DRL) agents tend to overfit to the task on which they were trained and fail to adapt to minor environment changes. To expedite learning when transferring to unseen tasks, we propose a novel approach to representing the current task using reward machines (RMs), state machine abstractions that induce subtasks based on the current task’s rewards and dynamics. Our method provides agents with symbolic representations of optimal transitions from their current abstract state and rewards them for achieving these transitions. These representations are shared across tasks, allowing agents to exploit knowledge of previously encountered symbols and transitions, thus enhancing transfer. Empirical results show that our representations improve sample efficiency and few-shot transfer in a variety of domains.

----

## [1222] FairTrade: Achieving Pareto-Optimal Trade-Offs between Balanced Accuracy and Fairness in Federated Learning

**Authors**: *Maryam Badar, Sandipan Sikdar, Wolfgang Nejdl, Marco Fisichella*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28971](https://doi.org/10.1609/aaai.v38i10.28971)

**Abstract**:

As Federated Learning (FL) gains prominence in distributed machine learning applications, achieving fairness without compromising predictive performance becomes paramount. The data being gathered from distributed clients in an FL environment often leads to class imbalance. In such scenarios, balanced accuracy rather than accuracy is the true representation of model performance. However, most state-of-the-art fair FL methods report accuracy as the measure of performance,  which can lead to misguided interpretations of the model's effectiveness to mitigate discrimination. To the best of our knowledge, this work presents the first attempt towards achieving Pareto-optimal trade-offs between balanced accuracy and fairness in a federated environment (FairTrade). By utilizing multi-objective optimization, the framework negotiates the intricate balance between model's balanced accuracy and fairness. The framework's agnostic design adeptly accommodates both statistical and causal fairness notions, ensuring its adaptability across diverse FL contexts. We provide empirical evidence of our framework's efficacy through extensive experiments on five real-world datasets and comparisons with six baselines. The empirical results underscore the potential of our framework in improving the trade-off between fairness and balanced accuracy in FL applications.

----

## [1223] Robustness-Guided Image Synthesis for Data-Free Quantization

**Authors**: *Jianhong Bai, Yuchen Yang, Huanpeng Chu, Hualiang Wang, Zuozhu Liu, Ruizhe Chen, Xiaoxuan He, Lianrui Mu, Chengfei Cai, Haoji Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28972](https://doi.org/10.1609/aaai.v38i10.28972)

**Abstract**:

Quantization has emerged as a promising direction for model compression. Recently, data-free quantization has been widely studied as a promising method to avoid privacy concerns, which synthesizes images as an alternative to real training data. Existing methods use classification loss to ensure the reliability of the synthesized images. Unfortunately, even if these images are well-classified by the pre-trained model, they still suffer from low semantics and homogenization issues. Intuitively, these low-semantic images are sensitive to perturbations, and the pre-trained model tends to have inconsistent output when the generator synthesizes an image with low semantics. To this end, we propose Robustness-Guided Image Synthesis (RIS), a simple but effective method to enrich the semantics of synthetic images and improve image diversity, further boosting the performance of data-free compression tasks. Concretely, we first introduce perturbations on input and model weight, then define the inconsistency metrics at feature and prediction levels before and after perturbations. On the basis of inconsistency on two levels, we design a robustness optimization objective to eliminate low-semantic images. Moreover, we also make our approach diversity-aware by forcing the generator to synthesize images with small correlations. With RIS, we achieve state-of-the-art performance for various settings on data-free quantization and can be extended to other data-free compression tasks.

----

## [1224] Regret Analysis of Policy Gradient Algorithm for Infinite Horizon Average Reward Markov Decision Processes

**Authors**: *Qinbo Bai, Washim Uddin Mondal, Vaneet Aggarwal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28973](https://doi.org/10.1609/aaai.v38i10.28973)

**Abstract**:

In this paper, we consider an infinite horizon average reward Markov Decision Process (MDP). Distinguishing itself from existing works within this context, our approach harnesses the power of the general policy gradient-based algorithm, liberating it from the constraints of assuming a linear MDP structure. We propose a vanilla policy gradient-based algorithm and show its global convergence property. We then prove that the proposed algorithm has O(T^3/4) regret. Remarkably, this paper marks a pioneering effort by presenting the first exploration into regret bound computation for the general parameterized policy gradient algorithm in the context of average reward scenarios.

----

## [1225] Combating Data Imbalances in Federated Semi-supervised Learning with Dual Regulators

**Authors**: *Sikai Bai, Shuaicheng Li, Weiming Zhuang, Jie Zhang, Kunlin Yang, Jun Hou, Shuai Yi, Shuai Zhang, Junyu Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28974](https://doi.org/10.1609/aaai.v38i10.28974)

**Abstract**:

Federated learning has become a popular method to learn from decentralized heterogeneous data. Federated semi-supervised learning (FSSL) emerges to train models from a small fraction of labeled data due to label scarcity on decentralized clients. Existing FSSL methods assume independent and identically distributed (IID) labeled data across clients and consistent class distribution between labeled and unlabeled data within a client. This work studies a more practical and challenging scenario of FSSL, where data distribution is different not only across clients but also within a client between labeled and unlabeled data. To address this challenge, we propose a novel FSSL framework with dual regulators, FedDure. FedDure lifts the previous assumption with a coarse-grained regulator (C-reg) and a fine-grained regulator (F-reg): C-reg regularizes the updating of the local model by tracking the learning effect on labeled data distribution; F-reg learns an adaptive weighting scheme tailored for unlabeled instances in each client. We further formulate the client model training as bi-level optimization that adaptively optimizes the model in the client with two regulators. Theoretically, we show the convergence guarantee of the dual regulators. Empirically, we demonstrate that FedDure is superior to the existing methods across a  wide range of settings, notably by more than 11% on CIFAR-10 and CINIC-10 datasets.

----

## [1226] SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation

**Authors**: *Malyaban Bal, Abhronil Sengupta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28975](https://doi.org/10.1609/aaai.v38i10.28975)

**Abstract**:

Large language Models (LLMs), though growing exceedingly powerful, comprises of orders of magnitude less neurons and synapses than the human brain. However, it requires significantly more power/energy to operate. In this work, we propose a novel bio-inspired spiking language model (LM) which aims to reduce the computational cost of conventional LMs by drawing motivation from the synaptic information flow in the brain. In this paper, we demonstrate a framework that leverages the average spiking rate of neurons at equilibrium to train a neuromorphic spiking LM using implicit differentiation technique, thereby overcoming the non-differentiability problem of spiking neural network (SNN) based algorithms without using any type of surrogate gradient. The steady-state convergence of the spiking neurons also allows us to design a spiking attention mechanism, which is critical in developing a scalable spiking LM. Moreover, the convergence of average spiking rate of neurons at equilibrium is utilized to develop a novel ANN-SNN knowledge distillation based technique wherein we use a pre-trained BERT model as “teacher” to train our “student” spiking architecture. While the primary architecture proposed in this paper is motivated by BERT, the technique can be potentially extended to different kinds of LLMs. Our work is the first one to demonstrate the performance of an operational spiking LM architecture on multiple different tasks in the GLUE benchmark. Our implementation source code is available at https://github.com/NeuroCompLab-psu/SpikingBERT.

----

## [1227] Disentangled Partial Label Learning

**Authors**: *Wei-Xuan Bao, Yong Rui, Min-Ling Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28976](https://doi.org/10.1609/aaai.v38i10.28976)

**Abstract**:

Partial label learning (PLL) induces a multi-class classifier from training examples each associated with a set of candidate labels, among which only one is valid. The formation of real-world data typically arises from heterogeneous entanglement of series latent explanatory factors, which are considered intrinsic properties for discriminating between different patterns. Though learning disentangled representation is expected to facilitate label disambiguation for partial-label (PL) examples, few existing works were dedicated to addressing this issue. In this paper, we make the first attempt towards disentangled PLL and propose a novel approach named TERIAL, which makes predictions according to derived disentangled representation of instances and label embeddings. The TERIAL approach formulates the PL examples as an undirected bipartite graph where instances are only connected with their candidate labels, and employs a tailored neighborhood routing mechanism to yield disentangled representation of nodes in the graph. Specifically, the proposed routing mechanism progressively infers the explanatory factors that contribute to the edge between adjacent nodes and augments the representation of the central node with factor-aware embedding information propagated from specific neighbors simultaneously via iteratively analyzing the promising subspace clusters formed by the node and its neighbors. The estimated labeling confidence matrix is also introduced to accommodate unreliable links owing to the inherent ambiguity of PLL. Moreover, we theoretically prove that the neighborhood routing mechanism will converge to the point estimate that maximizes the marginal likelihood of observed PL training examples. Comprehensive experiments over various datasets demonstrate that our approach outperforms the state-of-the-art counterparts.

----

## [1228] Efficient Target Propagation by Deriving Analytical Solution

**Authors**: *Yanhao Bao, Tatsukichi Shibuya, Ikuro Sato, Rei Kawakami, Nakamasa Inoue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28977](https://doi.org/10.1609/aaai.v38i10.28977)

**Abstract**:

Exploring biologically plausible algorithms as alternatives to error backpropagation (BP) is a challenging research topic in artificial intelligence. It also provides insights into the brain's learning methods. Recently, when combined with well-designed feedback loss functions such as Local Difference Reconstruction Loss (LDRL) and through hierarchical training of feedback pathway synaptic weights, Target Propagation (TP) has achieved performance comparable to BP in image classification tasks. However, with an increase in the number of network layers, the tuning and training cost of feedback weights escalates. Drawing inspiration from the work of Ernoult et al., we propose a training method that seeks the optimal solution for feedback weights. This method enhances the efficiency of feedback training by analytically minimizing feedback loss, allowing the feedback layer to skip certain local training iterations. More specifically, we introduce the Jacobian matching loss (JML) for feedback training. We also proactively implement layers designed to derive analytical solutions that minimize JML. Through experiments, we have validated the effectiveness of this approach. Using the CIFAR-10 dataset, our method showcases accuracy levels comparable to state-of-the-art TP methods. Furthermore, we have explored its effectiveness in more intricate network architectures.

----

## [1229] Strong Baselines for Parameter-Efficient Few-Shot Fine-Tuning

**Authors**: *Samyadeep Basu, Shell Xu Hu, Daniela Massiceti, Soheil Feizi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28978](https://doi.org/10.1609/aaai.v38i10.28978)

**Abstract**:

Few-shot classification (FSC) entails learning novel classes given only a few examples per class after a pre-training (or meta-training) phase on a set of base classes. Recent works have shown that simply fine-tuning a pre-trained Vision Transformer (ViT) on new test classes is a strong approach for FSC. Fine-tuning ViTs, however, is expensive in time, compute and storage. This has motivated the design of parameter efficient fine-tuning (PEFT) methods which fine-tune only a fraction of the Transformer's parameters. While these methods have shown promise, inconsistencies in experimental conditions make it difficult to disentangle their advantage from other experimental factors including the feature extractor architecture, pre-trained initialization and fine-tuning algorithm, amongst others. In our paper, we conduct a large-scale, experimentally consistent, empirical analysis to study PEFTs for few-shot image classification. Through a battery of over 1.8k controlled experiments on large-scale few-shot benchmarks including Meta-Dataset and ORBIT, we uncover novel insights on PEFTs that cast light on their efficacy in fine-tuning ViTs for few-shot classification. Through our controlled empirical study, we have two main findings: (i) Fine-tuning just the LayerNorm  parameters (which we call LN-Tune) during few-shot adaptation is an extremely strong baseline across ViTs pre-trained with both self-supervised and supervised objectives, (ii) For self-supervised ViTs, we find that simply learning a set of scaling parameters for each attention matrix (which we call Attn-Scale) along with a domain-residual adapter (DRA) module leads to state-of-the-art performance (while being ~9x more parameter-efficient) on Meta-Dataset. Our empirical findings set strong baselines and call for rethinking the current design of PEFT methods for FSC.

----

## [1230] TREE-G: Decision Trees Contesting Graph Neural Networks

**Authors**: *Maya Bechler-Speicher, Amir Globerson, Ran Gilad-Bachrach*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28979](https://doi.org/10.1609/aaai.v38i10.28979)

**Abstract**:

When dealing with tabular data, models based on decision
trees are a popular choice due to their high accuracy on these
data types, their ease of application, and explainability properties. However, when it comes to graph-structured data, it
is not clear how to apply them effectively, in a way that in-
corporates the topological information with the tabular data
available on the vertices of the graph. To address this challenge,
we introduce TREE-G. TREE-G modifies standard decision
trees, by introducing a novel split function that is specialized
for graph data. Not only does this split function incorporate
the node features and the topological information, but it also
uses a novel pointer mechanism that allows split nodes to
use information computed in previous splits. Therefore, the
split function adapts to the predictive task and the graph at
hand. We analyze the theoretical properties of TREE-G and
demonstrate its benefits empirically on multiple graph and
vertex prediction benchmarks. In these experiments, TREE-G
consistently outperforms other tree-based models and often
outperforms other graph-learning algorithms such as Graph
Neural Networks (GNNs) and Graph Kernels, sometimes by
large margins. Moreover, TREE-Gs models and their predic
tions can be explained and visualized.

----

## [1231] Scores for Learning Discrete Causal Graphs with Unobserved Confounders

**Authors**: *Alexis Bellot, Junzhe Zhang, Elias Bareinboim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28980](https://doi.org/10.1609/aaai.v38i10.28980)

**Abstract**:

Structural learning is arguably one of the most challenging and pervasive tasks found throughout the data sciences. There exists a growing literature that studies structural learning in non-parametric settings where conditional independence constraints are taken to define the equivalence class. In the presence of unobserved confounders, it is understood that non-conditional independence constraints are imposed over the observational distribution, including certain equalities and inequalities between functionals of the joint distribution. In this paper, we develop structural learning methods that leverage additional constraints beyond conditional independences. Specifically, we first introduce a score for arbitrary graphs combining Watanabe's asymptotic expansion of the marginal likelihood and new bounds over the cardinality of the exogenous variables. Second, we show that the new score has desirable properties in terms of expressiveness and computability. In terms of expressiveness, we prove that the score captures distinct constraints imprinted in the data, including Verma's and inequalities'. In terms of computability, we show properties of score equivalence and decomposability, which allows, in principle, to break the problem of structural learning in smaller and more manageable pieces. Third, we implement this score using an MCMC sampling algorithm and test its properties in several simulation scenarios.

----

## [1232] Simplicity Bias in Overparameterized Machine Learning

**Authors**: *Yakir Berchenko*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28981](https://doi.org/10.1609/aaai.v38i10.28981)

**Abstract**:

A thorough theoretical understanding of the surprising generalization ability of deep networks (and other overparameterized models) is still lacking. Here we demonstrate that simplicity bias is a major phenomenon to be reckoned with in overparameterized machine learning. In addition to explaining the outcome of simplicity bias, we also study its source: following concrete rigorous examples, we argue that (i) simplicity bias can explain generalization in overparameterized learning models such as neural networks; (ii) simplicity bias and excellent generalization are optimizer-independent, as our example shows, and although the optimizer affects training, it is not the driving force behind simplicity bias; (iii) simplicity bias in pre-training models, and subsequent posteriors, is universal and stems from the subtle fact that uniformly-at-random constructed priors are not uniformly-at-random sampled ; and (iv) in neural network models, the biasing mechanism in wide (and shallow) networks is different from the biasing mechanism in deep (and narrow) networks.

----

## [1233] Maximizing the Success Probability of Policy Allocations in Online Systems

**Authors**: *Artem Betlei, Mariia Vladimirova, Mehdi Sebbar, Nicolas Urien, Thibaud Rahier, Benjamin Heymann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28982](https://doi.org/10.1609/aaai.v38i10.28982)

**Abstract**:

The effectiveness of advertising in e-commerce largely depends on the ability of merchants to bid on and win impressions for their targeted users. The bidding procedure is highly complex due to various factors such as market competition, user behavior, and the diverse objectives of advertisers. In this paper we consider the problem at the level of user timelines instead of individual bid requests, manipulating full policies (i.e. pre-defined bidding strategies) and not bid values. In order to optimally allocate policies to users, typical multiple treatments allocation methods solve knapsack-like problems which aim at maximizing an expected value under constraints. In the specific context of online advertising, we argue that optimizing for the probability of success is a more suited objective than expected value maximization, and we introduce the SuccessProbaMax algorithm that aims at finding the policy allocation which is the most likely to outperform a fixed reference policy. Finally, we conduct comprehensive experiments both on synthetic and real-world data to evaluate its performance. The results demonstrate that our proposed algorithm outperforms conventional expected-value maximization algorithms in terms of success rate.

----

## [1234] DGCLUSTER: A Neural Framework for Attributed Graph Clustering via Modularity Maximization

**Authors**: *Aritra Bhowmick, Mert Kosan, Zexi Huang, Ambuj K. Singh, Sourav Medya*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28983](https://doi.org/10.1609/aaai.v38i10.28983)

**Abstract**:

Graph clustering is a fundamental and challenging task in the field of graph mining where the objective is to group the nodes into clusters taking into consideration the topology of the graph. It has several applications in diverse domains spanning social network analysis, recommender systems, computer vision, and bioinformatics.  In this work, we propose a novel method, DGCluster, which primarily optimizes the modularity objective using graph neural networks and scales linearly with the graph size. Our method does not require the number of clusters to be specified as a part of the input and can also leverage the availability of auxiliary node level information. We extensively test DGCluster on several real-world datasets of varying sizes, across multiple popular cluster quality metrics. Our approach consistently outperforms the state-of-the-art methods, demonstrating significant performance gains in almost all settings.

----

## [1235] MEPSI: An MDL-Based Ensemble Pruning Approach with Structural Information

**Authors**: *Xiao-Dong Bi, Shao-Qun Zhang, Yuan Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28984](https://doi.org/10.1609/aaai.v38i10.28984)

**Abstract**:

Ensemble pruning that combines a subset of individual learners generated in parallel to make predictions is an important topic in ensemble learning. Past decades have developed a lot of pruning algorithms that focus on the external behavior of learners on samples, which may lead to over-fitting. In this paper, we conjecture that the generalization performance of an ensemble is not only related to its external behavior on samples but also dependent on the internal structure of individual learners. We propose the general MEPSI approach based on Kolmogorov complexity and the Minimum Description Length (MDL) principle, which formulates the ensemble pruning task as the two-objective optimization problem that comprises the empirical error and structural information among individual learners. We also provide a concrete implementation of MEPSI on decision trees. The theoretical results provide generalization bounds for both the general MEPSI approach and tree-based implementation. The comparative experiments conducted on multiple real-world data sets demonstrate the effectiveness of our proposed method.

----

## [1236] Constraint Latent Space Matters: An Anti-anomalous Waveform Transformation Solution from Photoplethysmography to Arterial Blood Pressure

**Authors**: *Cheng Bian, Xiaoyu Li, Qi Bi, Guangpu Zhu, Jiegeng Lyu, Weile Zhang, Yelei Li, Zijing Zeng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28985](https://doi.org/10.1609/aaai.v38i10.28985)

**Abstract**:

Arterial blood pressure (ABP) holds substantial promise for proactive cardiovascular health management. Notwithstanding its potential, the invasive nature of ABP measurements confines their utility primarily to clinical environments, limiting their applicability for continuous monitoring beyond medical facilities. The conversion of photoplethysmography (PPG) signals into ABP equivalents has garnered significant attention due to its potential in revolutionizing cardiovascular disease management. Recent strides in PPG-to-ABP prediction encompass the integration of generative and discriminative models. Despite these advances, the efficacy of these models is curtailed by the latent space shift predicament, stemming from alterations in PPG data distribution across disparate hardware and individuals, potentially leading to distorted ABP waveforms. To tackle this problem, we present an innovative solution named the Latent Space Constraint Transformer (LSCT), leveraging a quantized codebook to yield robust latent spaces by employing multiple discretizing bases. To facilitate improved reconstruction, the Correlation-boosted Attention Module (CAM) is introduced to systematically query pertinent bases on a global scale. Furthermore, to enhance expressive capacity, we propose the Multi-Spectrum Enhancement Knowledge (MSEK), which fosters local information flow within the channels of latent code and provides additional embedding for reconstruction. Through comprehensive experimentation on both publicly available datasets and a private downstream task dataset, the proposed approach demonstrates noteworthy performance enhancements compared to existing methods. Extensive ablation studies further substantiate the effectiveness of each introduced module.

----

## [1237] Axiomatic Aggregations of Abductive Explanations

**Authors**: *Gagan Biradar, Yacine Izza, Elita Lobo, Vignesh Viswanathan, Yair Zick*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28986](https://doi.org/10.1609/aaai.v38i10.28986)

**Abstract**:

The recent criticisms of the robustness of post hoc model approximation explanation methods (like LIME and SHAP) have led to the rise of model-precise abductive explanations. For each data point, abductive explanations provide a minimal subset of features that are sufficient to generate the outcome. While theoretically sound and rigorous, abductive explanations suffer from a major issue --- there can be several valid abductive explanations for the same data point. In such cases, providing a single abductive explanation can be insufficient; on the other hand, providing all valid abductive explanations can be incomprehensible due to their size. In this work, we solve this issue by aggregating the many possible abductive explanations into feature importance scores. We propose three aggregation methods: two based on power indices from cooperative game theory and a third based on a well-known measure of causal strength. We characterize these three methods axiomatically, showing that each of them uniquely satisfies a set of desirable properties. We also evaluate them on multiple datasets and show that these explanations are robust to the attacks that fool SHAP and LIME.

----

## [1238] MIND: Multi-Task Incremental Network Distillation

**Authors**: *Jacopo Bonato, Francesco Pelosin, Luigi Sabetta, Alessandro Nicolosi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28987](https://doi.org/10.1609/aaai.v38i10.28987)

**Abstract**:

The recent surge of pervasive devices that generate dynamic data streams has underscored the necessity for learning systems to adapt continually to data distributional shifts. To tackle this challenge, the research community has put forth a spectrum of methodologies, including the demanding pursuit of class-incremental learning without replay data. In this study, we present MIND, a parameter isolation method that aims to significantly enhance the performance of replay-free solutions and achieve state-of-the-art results on several widely studied datasets. Our approach introduces two main contributions: two alternative distillation procedures that significantly improve the efficiency of MIND increasing the accumulated knowledge of each sub-network, and the optimization of the BachNorm layers across tasks inside the sub-networks. Overall, MIND outperforms all the state-of-the-art methods for rehearsal-free Class-Incremental learning (with an increment in classification accuracy of approx. +6% on CIFAR-100/10 and +10% on TinyImageNet/10) reaching up to approx. +40% accuracy in Domain-Incremental scenarios. Moreover, we ablated each contribution to demonstrate its impact on performance improvement. Our results showcase the superior performance of MIND indicating its potential for addressing the challenges posed by Class-incremental and Domain-Incremental learning in resource-constrained environments.

----

## [1239] HyperFast: Instant Classification for Tabular Data

**Authors**: *David Bonet, Daniel Mas Montserrat, Xavier Giró-i-Nieto, Alexander G. Ioannidis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28988](https://doi.org/10.1609/aaai.v38i10.28988)

**Abstract**:

Training deep learning models and performing hyperparameter tuning can be computationally demanding and time-consuming. Meanwhile, traditional machine learning methods like gradient-boosting algorithms remain the preferred choice for most tabular data applications, while neural network alternatives require extensive hyperparameter tuning or work only in toy datasets under limited settings. In this paper, we introduce HyperFast, a meta-trained hypernetwork designed for instant classification of tabular data in a single forward pass. HyperFast generates a task-specific neural network tailored to an unseen dataset that can be directly used for classification inference, removing the need for training a model. We report extensive experiments with OpenML and genomic data, comparing HyperFast to competing tabular data neural networks, traditional ML methods, AutoML systems, and boosting machines. HyperFast shows highly competitive results, while being significantly faster. Additionally, our approach demonstrates robust adaptability across a variety of classification tasks with little to no fine-tuning, positioning HyperFast as a strong solution for numerous applications and rapid model deployment. HyperFast introduces a promising paradigm for fast classification, with the potential to substantially decrease the computational burden of deep learning. Our code, which offers a scikit-learn-like interface, along with the trained HyperFast model, can be found at https://github.com/AI-sandbox/HyperFast.

----

## [1240] A Theory of Non-acyclic Generative Flow Networks

**Authors**: *Leo Maxime Brunswic, Yinchuan Li, Yushun Xu, Yijun Feng, Shangling Jui, Lizhuang Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28989](https://doi.org/10.1609/aaai.v38i10.28989)

**Abstract**:

GFlowNets is a novel flow-based method for learning a stochastic policy to generate objects via a sequence of actions and with probability proportional to a given positive reward. We contribute to relaxing hypotheses limiting the application range of GFlowNets, in particular: acyclicity (or lack thereof). To this end, we extend the theory of GFlowNets on measurable spaces which includes continuous state spaces without cycle restrictions, and provide a generalization of cycles in this generalized context. We show that losses used so far push flows to get stuck into cycles and we define a family of losses solving this issue. Experiments on graphs and continuous tasks validate those principles.

----

## [1241] Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples

**Authors**: *Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28990](https://doi.org/10.1609/aaai.v38i10.28990)

**Abstract**:

Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted adversarial examples, which are generated through either well-conceived L_p-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer where to attack. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate Counterfactual ADversarial Examples to answer how to attack. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.

----

## [1242] MSGNet: Learning Multi-Scale Inter-series Correlations for Multivariate Time Series Forecasting

**Authors**: *Wanlin Cai, Yuxuan Liang, Xianggen Liu, Jianshuai Feng, Yuankai Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28991](https://doi.org/10.1609/aaai.v38i10.28991)

**Abstract**:

Multivariate time series forecasting poses an ongoing challenge across various disciplines. Time series data often exhibit diverse intra-series and inter-series correlations, contributing to intricate and interwoven dependencies that have been the focus of numerous studies. Nevertheless, a significant research gap remains in comprehending the varying inter-series correlations across different time scales among multiple time series, an area that has received limited attention in the literature. To bridge this gap, this paper introduces MSGNet, an advanced deep learning model designed to capture the varying inter-series correlations across multiple time scales using frequency domain analysis and adaptive graph convolution. By leveraging frequency domain analysis, MSGNet effectively extracts salient periodic patterns and decomposes the time series into distinct time scales. The model incorporates a self-attention mechanism to capture intra-series dependencies, while introducing an adaptive mixhop graph convolution layer to autonomously learn diverse inter-series correlations within each time scale. Extensive experiments are conducted on several real-world datasets to showcase the effectiveness of MSGNet. Furthermore, MSGNet possesses the ability to automatically learn explainable multi-scale inter-series correlations, exhibiting strong generalization capabilities even when applied to out-of-distribution samples.

----

## [1243] Kernelized Normalizing Constant Estimation: Bridging Bayesian Quadrature and Bayesian Optimization

**Authors**: *Xu Cai, Jonathan Scarlett*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28992](https://doi.org/10.1609/aaai.v38i10.28992)

**Abstract**:

In this paper, we study the problem of estimating the normalizing constant through queries to the black-box function f, which is the integration of the exponential function of f scaled by a problem parameter lambda.  We assume f belongs to a reproducing kernel Hilbert space (RKHS), and show that to estimate the normalizing constant within a small relative error, the level of difficulty depends on the value of lambda: When lambda approaches zero, the problem is similar to Bayesian quadrature (BQ), while when lambda approaches infinity, the problem is similar to Bayesian optimization (BO).  More generally, the problem varies between BQ and BO.   We find that this pattern holds true even when the function evaluations are noisy, bringing new aspects to this topic.  Our findings are supported by both algorithm-independent lower bounds and algorithmic upper bounds, as well as simulation studies conducted on a variety of benchmark functions.

----

## [1244] EG-NAS: Neural Architecture Search with Fast Evolutionary Exploration

**Authors**: *Zicheng Cai, Lei Chen, Peng Liu, Tongtao Ling, Yutao Lai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28993](https://doi.org/10.1609/aaai.v38i10.28993)

**Abstract**:

Differentiable Architecture Search (DARTS) has achieved a rapid search for excellent architectures by optimizing architecture parameters through gradient descent. However, this efficiency comes with a significant challenge: the risk of premature convergence to local optima, resulting in subpar performance that falls short of expectations. To address this issue, we propose a novel and effective method called Evolutionary Gradient-Based Neural Architecture Search (EG-NAS). Our approach combines the strengths of both gradient descent and evolutionary strategy, allowing for the exploration of various optimization directions during the architecture search process. To begin with, we continue to employ gradient descent for updating network parameters to ensure efficiency. Subsequently, to mitigate the risk of premature convergence, we introduce an evolutionary strategy with global search capabilities to optimize the architecture parameters. By leveraging the best of both worlds, our method strikes a balance between efficient exploration and exploitation of the search space. Moreover, we have redefined the fitness function to not only consider accuracy but also account for individual similarity. This inclusion enhances the diversity and accuracy of the optimized directions identified by the evolutionary strategy. Extensive experiments on various datasets and search spaces demonstrate that EG-NAS achieves highly competitive performance at significantly low search costs compared to state-of-the-art methods. The code is available at https://github.com/caicaicheng/EG-NAS.

----

## [1245] Mixup-Induced Domain Extrapolation for Domain Generalization

**Authors**: *Meng Cao, Songcan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28994](https://doi.org/10.1609/aaai.v38i10.28994)

**Abstract**:

Domain generalization aims to learn a well-performed classifier on multiple source domains for unseen target domains under domain shift. Domain-invariant representation (DIR) is an intuitive approach and has been of great concern. In practice, since the targets are variant and agnostic, only a few sources are not sufficient to reflect the entire domain population, leading to biased DIR. Derived from PAC-Bayes framework, we provide a novel generalization bound involving the number of domains sampled from the environment (N) and the radius of the Wasserstein ball centred on the target (r), which have rarely been considered before. Herein, we can obtain two natural and significant findings: when N increases, 1) the gap between the source and target sampling environments can be gradually mitigated; 2) the target can be better approximated within the Wasserstein ball. These findings prompt us to collect adequate domains against domain shift. For seeking convenience, we design a novel yet simple Extrapolation Domain strategy induced by the Mixup scheme, namely EDM. Through a reverse Mixup scheme to generate the extrapolated domains, combined with the interpolated domains, we expand the interpolation space spanned by the sources, providing more abundant domains to increase sampling intersections to shorten r. Moreover, EDM is easy to implement and be plugged-and-played. In experiments, EDM has been plugged into several methods in both closed and open set settings, achieving up to 5.73% improvement.

----

## [1246] Continuous-Time Graph Representation with Sequential Survival Process

**Authors**: *Abdulkadir Çelikkanat, Nikolaos Nakis, Morten Mørup*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28995](https://doi.org/10.1609/aaai.v38i10.28995)

**Abstract**:

Over the past two decades, there has been a tremendous increase in the growth of representation learning methods for graphs, with numerous applications across various fields, including bioinformatics, chemistry, and the social sciences. However, current dynamic network approaches focus on discrete-time networks or treat links in continuous-time networks as instantaneous events. Therefore, these approaches have limitations in capturing the persistence or absence of links that continuously emerge and disappear over time for particular durations. To address this, we propose a novel stochastic process relying on survival functions to model the durations of links and their absences over time. This forms a generic new likelihood specification explicitly accounting for intermittent edge-persistent networks, namely GraSSP: Graph Representation with Sequential Survival Process. We apply the developed framework to a recent continuous time dynamic latent distance model characterizing network dynamics in terms of a sequence of piecewise linear movements of nodes in latent space. We quantitatively assess the developed framework in various downstream tasks, such as link prediction and network completion, demonstrating that the developed modeling framework accounting for link persistence and absence well tracks the intrinsic trajectories of nodes in a latent space and captures the underlying characteristics of evolving network structure.

----

## [1247] Learning to Unlearn: Instance-Wise Unlearning for Pre-trained Classifiers

**Authors**: *Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, Moontae Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28996](https://doi.org/10.1609/aaai.v38i10.28996)

**Abstract**:

Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we consider instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.

----

## [1248] Variable Importance in High-Dimensional Settings Requires Grouping

**Authors**: *Ahmad Chamma, Bertrand Thirion, Denis A. Engemann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28997](https://doi.org/10.1609/aaai.v38i10.28997)

**Abstract**:

Explaining the decision process of machine learning algorithms is nowadays crucial for both model’s performance enhancement and human comprehension. This can be achieved by assessing the variable importance of single variables, even for high-capacity non-linear methods, e.g. Deep Neural Networks (DNNs). While only removal-based approaches, such as Permutation Importance (PI), can bring statistical validity, they return misleading results when variables are correlated. Conditional Permutation Importance (CPI) bypasses PI’s limitations in such cases. However, in high-dimensional settings, where high correlations between the variables cancel their conditional importance, the use of CPI as well as other methods leads to unreliable results, besides prohibitive computation costs. Grouping variables statistically via clustering or some prior knowledge gains some power back and leads to better interpretations. In this work, we introduce BCPI (Block-Based Conditional Permutation Importance), a new generic framework for variable importance computation with statistical guarantees handling both single and group cases. Furthermore, as handling groups with high cardinality (such as a set of observations of a given modality) are both time-consuming and resource-intensive, we also introduce a new stacking approach extending the DNN architecture with sub-linear layers adapted to the group structure. We show that the ensuing approach extended with stacking controls the type-I error even with highly-correlated groups and shows top accuracy across benchmarks. Furthermore, we perform a real-world data analysis in a large-scale medical dataset where we aim to show the consistency between our results and the literature for a biomarker prediction.

----

## [1249] Privacy Amplification by Iteration for ADMM with (Strongly) Convex Objective Functions

**Authors**: *T.-H. Hubert Chan, Hao Xie, Mengshi Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28998](https://doi.org/10.1609/aaai.v38i10.28998)

**Abstract**:

We examine a private ADMM variant for (strongly) convex objectives which is a primal-dual iterative method. Each iteration has a user with a private function used to update the primal variable, masked by Gaussian noise for local privacy, without directly adding noise to the dual variable. Privacy amplification by iteration explores if noises from later iterations can enhance the privacy guarantee when releasing final variables after the last iteration.

Cyffers et al. explored privacy amplification by iteration for the proximal ADMM variant, where a user's entire private function is accessed and noise is added to the primal variable. In contrast, we examine a private ADMM variant requiring just one gradient access to a user's function, but both primal and dual variables must be passed between successive iterations.

To apply Balle et al.'s coupling framework to the gradient ADMM variant, we tackle technical challenges with novel ideas. First, we address the non-expansive mapping issue in ADMM iterations by using a customized norm. Second, because the dual variables are not masked with any noise directly, their privacy guarantees are achieved by treating two consecutive noisy ADMM iterations as a Markov operator.

Our main result is that the privacy guarantee for the gradient ADMM variant can be amplified proportionally to the number of iterations. For strongly convex objective functions, this amplification exponentially increases with the number of iterations. These amplification results align with the previously studied special case of stochastic gradient descent.

----

## [1250] Understanding Distributed Representations of Concepts in Deep Neural Networks without Supervision

**Authors**: *Wonjoon Chang, Dahee Kwon, Jaesik Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.28999](https://doi.org/10.1609/aaai.v38i10.28999)

**Abstract**:

Understanding intermediate representations of the concepts learned by deep learning classifiers is indispensable for interpreting general model behaviors. Existing approaches to reveal learned concepts often rely on human supervision, such as pre-defined concept sets or segmentation processes. In this paper, we propose a novel unsupervised method for discovering distributed representations of concepts by selecting a principal subset of neurons. Our empirical findings demonstrate that instances with similar neuron activation states tend to share coherent concepts. Based on the observations, the proposed method selects principal neurons that construct an interpretable region, namely a Relaxed Decision Region (RDR), encompassing instances with coherent concepts in the feature space. It can be utilized to identify unlabeled subclasses within data and to detect the causes of misclassifications. Furthermore, the applicability of our method across various layers discloses distinct distributed representations over the layers, which provides deeper insights into the internal mechanisms of the deep learning model.

----

## [1251] Incomplete Contrastive Multi-View Clustering with High-Confidence Guiding

**Authors**: *Guoqing Chao, Yi Jiang, Dianhui Chu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29000](https://doi.org/10.1609/aaai.v38i10.29000)

**Abstract**:

Incomplete multi-view clustering becomes an important research problem, since multi-view data with missing values are ubiquitous in real-world applications. Although great efforts have been made for incomplete multi-view clustering, there are still some challenges: 1) most existing methods didn't make full use of multi-view information to deal with missing values; 2) most methods just employ the consistent information within multi-view data but ignore the complementary information; 3) For the existing incomplete multi-view clustering methods, incomplete multi-view representation learning and clustering are treated as independent processes, which leads to performance gap. In this work, we proposed a novel Incomplete Contrastive Multi-View Clustering method with high-confidence guiding (ICMVC). Firstly, we proposed a multi-view consistency relation transfer plus graph convolutional network to tackle missing values problem. Secondly, instance-level attention fusion and high-confidence guiding are proposed to exploit the complementary information while instance-level contrastive learning for latent representation is designed to employ the consistent information. Thirdly, an end-to-end framework is proposed to integrate multi-view missing values handling, multi-view representation learning and clustering assignment for joint optimization. Experiments compared with state-of-the-art approaches demonstrated the effectiveness and superiority of our method. Our code is publicly available at https://github.com/liunian-Jay/ICMVC. The version with supplementary material can be found at http://arxiv.org/abs/2312.08697.

----

## [1252] Offline Model-Based Optimization via Policy-Guided Gradient Search

**Authors**: *Yassine Chemingui, Aryan Deshwal, Trong Nghia Hoang, Janardhan Rao Doppa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29001](https://doi.org/10.1609/aaai.v38i10.29001)

**Abstract**:

Offline optimization is an emerging problem in many experimental engineering domains including protein, drug or aircraft design, where online experimentation to collect evaluation data is too expensive or dangerous. To avoid that, one has to optimize an unknown function given only its offline evaluation at a fixed set of inputs. A naive solution to this problem is to learn a surrogate model of the unknown function and optimize this surrogate instead. However, such a naive optimizer is prone to erroneous overestimation of the surrogate (possibly due to over-fitting on a biased sample of function evaluation) on inputs outside the offline dataset. Prior approaches addressing this challenge have primarily focused on learning robust surrogate models. However, their search strategies are derived from the surrogate model rather than the actual offline data. To fill this important gap, we introduce a new learning-to-search perspective for offline optimization by reformulating it as an offline reinforcement learning problem. Our proposed policy-guided gradient search approach explicitly learns the best policy for a given surrogate model created from the offline data. Our empirical results on multiple benchmarks demonstrate that the learned optimization policy can be combined with existing offline surrogates to significantly improve the optimization performance.

----

## [1253] Focus-Then-Decide: Segmentation-Assisted Reinforcement Learning

**Authors**: *Chao Chen, Jiacheng Xu, Weijian Liao, Hao Ding, Zongzhang Zhang, Yang Yu, Rui Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29002](https://doi.org/10.1609/aaai.v38i10.29002)

**Abstract**:

Visual Reinforcement Learning (RL) is a promising approach to achieve human-like intelligence. However, it currently faces challenges in learning efficiently within noisy environments. In contrast, humans can quickly identify task-relevant objects in distraction-filled surroundings by applying previously acquired common knowledge. Recently, foundational models in natural language processing and computer vision have achieved remarkable successes, and the common knowledge within these models can significantly benefit downstream task training. Inspired by these achievements, we aim to incorporate common knowledge from foundational models into visual RL. We propose a novel Focus-Then-Decide (FTD) framework, allowing the agent to make decisions based solely on task-relevant objects. To achieve this, we introduce an attention mechanism to select task-relevant objects from the object set returned by a foundational segmentation model, and only use the task-relevant objects for the subsequent training of the decision module. Additionally, we specifically employed two generic self-supervised objectives to facilitate the rapid learning of this attention mechanism. Experimental results on challenging tasks based on DeepMind Control Suite and Franka Emika Robotics demonstrate that our method can quickly and accurately pinpoint objects of interest in noisy environments. Consequently, it achieves a significant performance improvement over current state-of-the-art algorithms.
Project Page: https://www.lamda.nju.edu.cn/chenc/FTD.html
Code: https://github.com/LAMDA-RL/FTD

----

## [1254] Data Shunt: Collaboration of Small and Large Models for Lower Costs and Better Performance

**Authors**: *Dong Chen, Yueting Zhuang, Shuo Zhang, Jinfeng Liu, Su Dong, Siliang Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29003](https://doi.org/10.1609/aaai.v38i10.29003)

**Abstract**:

Pretrained large models, particularly large language models, have garnered increasing attention, as they have demonstrated remarkable abilities through contextual learning. Pretrained large models are increasingly recognized as fundamental tools for solving various tasks. However, the substantial computational demands of large models have dissuaded most product teams and individuals from running them. In such scenarios, to leverage the exceptional performance of large models, one must solely depend on costly APIs, further burdening product teams and individuals. On the other hand, despite the overall inferior performance of small models compared to large models, there are certain distributions where small models can achieve comparable or even superior results. For instance, during training, small models may become trapped in a local optimum that is unique to certain distributions, leading to superior performance. Hence, we propose Data Shunt (DS), a general paradigm for collaboration of small and large models. DS not only substantially reduces the cost associated with deploying large models but also effectively enhances overall performance. Specifically, DS determines the shunting direction by evaluating the confidence level of small models. When the confidence level falls below a specific threshold, the input data is forwarded to large models. To further leverage the advantages of the small and large models, we introduce Prompt Pruning (PP) and 2-Stage Confidence Distillation (2CD), which facilitate mutual collaboration, leading to better results and less cost. 
The remarkable performance across diverse modalities and tasks demonstrates the superiority of the proposed DS over large models. For instance, ChatGPT achieves an accuracy of 94.43% on Amazon Product sentiment analysis, and DS achieves an accuracy of 95.64%, while the cost has been reduced to only 31.18%. The code for the proposed method are provided for research purposes https://github.com/Anfeather/Data-Shunt.

----

## [1255] EPSD: Early Pruning with Self-Distillation for Efficient Model Compression

**Authors**: *Dong Chen, Ning Liu, Yichen Zhu, Zhengping Che, Rui Ma, Fachao Zhang, Xiaofeng Mou, Yi Chang, Jian Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29004](https://doi.org/10.1609/aaai.v38i10.29004)

**Abstract**:

Neural network compression techniques, such as knowledge distillation (KD) and network pruning, have received increasing attention. Recent work `Prune, then Distill' reveals that a pruned student-friendly teacher network can benefit the performance of KD. However, the conventional teacher-student pipeline, which entails cumbersome pre-training of the teacher and complicated compression steps, makes pruning with KD less efficient. In addition to compressing models, recent compression techniques also emphasize the aspect of efficiency. Early pruning demands significantly less computational cost in comparison to the conventional pruning methods as it does not require a large pre-trained model. Likewise, a special case of KD, known as self-distillation (SD), is more efficient since it requires no pre-training or student-teacher pair selection. This inspires us to collaborate early pruning with SD for efficient model compression. In this work, we propose the framework named Early Pruning with Self-Distillation (EPSD), which identifies and preserves distillable weights in early pruning for a given SD task. EPSD efficiently combines early pruning and self-distillation in a two-step process, maintaining the pruned network's trainability for compression. Instead of a simple combination of pruning and SD, EPSD enables the pruned network to favor SD by keeping more distillable weights before training to ensure better distillation of the pruned network. We demonstrated that EPSD improves the training of pruned networks, supported by visual and quantitative analyses. Our evaluation covered diverse benchmarks (CIFAR-10/100, Tiny-ImageNet, full ImageNet, CUB-200-2011, and Pascal VOC), with EPSD outperforming advanced pruning and SD techniques.

----

## [1256] A Generalized Shuffle Framework for Privacy Amplification: Strengthening Privacy Guarantees and Enhancing Utility

**Authors**: *E. Chen, Yang Cao, Yifei Ge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29005](https://doi.org/10.1609/aaai.v38i10.29005)

**Abstract**:

The shuffle model of local differential privacy is an advanced method of privacy amplification designed to enhance privacy protection with high utility. 
It achieves this by randomly shuffling  sensitive data, making linking individual data points to specific individuals more challenging.
However, most existing studies have focused on the shuffle model based on
(ε0,0)-Locally Differentially Private (LDP) randomizers, with limited consideration for complex scenarios such as (ε0,δ0)-LDP or personalized LDP (PLDP). 
This hinders a comprehensive understanding of the shuffle model's potential and limits its application in various settings.
To bridge this research gap, we propose a generalized shuffle framework that can be applied to PLDP setting. This generalization allows for a broader exploration of the privacy-utility trade-off and facilitates the design of privacy-preserving analyses in diverse contexts.
We prove that the shuffled PLDP process approximately preserves μ-Gaussian Differential Privacy with 
μ = O(1/√n).
This approach allows us to avoid the limitations and potential inaccuracies associated with inequality estimations.
To strengthen the privacy guarantee, we improve the lower bound by utilizing hypothesis testing instead of relying on rough estimations like the Chernoff bound or Hoeffding's inequality.
Furthermore, extensive comparative evaluations clearly show that our approach outperforms existing methods in achieving strong central privacy guarantees while preserving the utility of the global model.
We have also carefully designed corresponding algorithms for average function, frequency estimation, and stochastic gradient descent.

----

## [1257] Adaptive Discovering and Merging for Incremental Novel Class Discovery

**Authors**: *Guangyao Chen, Peixi Peng, Yangru Huang, Mengyue Geng, Yonghong Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29006](https://doi.org/10.1609/aaai.v38i10.29006)

**Abstract**:

One important desideratum of lifelong learning aims to discover novel classes from unlabelled data in a continuous manner. The central challenge is twofold: discovering and learning novel classes while mitigating the issue of catastrophic forgetting of established knowledge. To this end, we introduce a new paradigm called Adaptive Discovering and Merging (ADM) to discover novel categories adaptively in the incremental stage and integrate novel knowledge into the model without affecting the original knowledge. To discover novel classes adaptively, we decouple representation learning and novel class discovery, and use Triple Comparison (TC) and Probability Regularization (PR) to constrain the probability discrepancy and diversity for adaptive category assignment. To merge the learned novel knowledge adaptively, we propose a hybrid structure with base and novel branches named Adaptive Model Merging (AMM), which reduces the interference of the novel branch on the old classes to preserve the previous knowledge, and merges the novel branch to the base model without performance loss and parameter growth. Extensive experiments on several datasets show that ADM significantly outperforms existing class-incremental Novel Class Discovery (class-iNCD) approaches. Moreover, our AMM also benefits the class-incremental Learning (class-IL) task by alleviating the catastrophic forgetting problem. The source code is included in the supplementary materials.

----

## [1258] FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning

**Authors**: *Haokun Chen, Yao Zhang, Denis Krompass, Jindong Gu, Volker Tresp*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29007](https://doi.org/10.1609/aaai.v38i10.29007)

**Abstract**:

Recently, foundation models have exhibited remarkable advancements in multi-modal learning. These models, equipped with millions (or billions) of parameters, typically require a substantial amount of data for finetuning. However, collecting and centralizing training data from diverse sectors becomes challenging due to distinct privacy regulations. Federated Learning (FL) emerges as a promising solution, enabling multiple clients to collaboratively train neural networks without centralizing their local data. To alleviate client computation burdens and communication overheads, previous works have adapted Parameter-efficient Finetuning (PEFT) methods for FL. Hereby, only a small fraction of the model parameters are optimized and communicated during federated communications. Nevertheless, most previous works have focused on a single modality and neglected one common phenomenon, i.e., the presence of data heterogeneity across the clients. Therefore, in this work, we propose a finetuning framework tailored to heterogeneous multi-modal FL, called Federated Dual-Aadapter Teacher (FedDAT). Specifically, our approach leverages a Dual-Adapter Teacher (DAT) to address data heterogeneity by regularizing the client local updates and applying Mutual Knowledge Distillation (MKD) for an efficient knowledge transfer. FedDAT is the first approach that enables an efficient distributed finetuning of foundation models for a variety of heterogeneous Vision-Language tasks. To demonstrate its effectiveness, we conduct extensive experiments on four multi-modality FL benchmarks with different types of data heterogeneity, where FedDAT substantially outperforms the existing centralized PEFT methods adapted for FL.

----

## [1259] Uncertainty Quantification for Data-Driven Change-Point Learning via Cross-Validation

**Authors**: *Hui Chen, Yinxu Jia, Guanghui Wang, Changliang Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29008](https://doi.org/10.1609/aaai.v38i10.29008)

**Abstract**:

Accurately detecting multiple change-points is critical for various applications, but determining the optimal number of change-points remains a challenge. Existing approaches based on information criteria attempt to balance goodness-of-fit and model complexity, but their performance varies depending on the model. Recently, data-driven selection criteria based on cross-validation has been proposed, but these methods can be prone to slight overfitting in finite samples. In this paper, we introduce a method that controls the probability of overestimation and provides uncertainty quantification for learning multiple change-points via cross-validation. We frame this problem as a sequence of model comparison problems and leverage high-dimensional inferential procedures. We demonstrate the effectiveness of our approach through experiments on finite-sample data, showing superior uncertainty quantification for overestimation compared to existing methods. Our approach has broad applicability and can be used in diverse change-point models.

----

## [1260] Bridging the Semantic Latent Space between Brain and Machine: Similarity Is All You Need

**Authors**: *Jiaxuan Chen, Yu Qi, Yueming Wang, Gang Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29009](https://doi.org/10.1609/aaai.v38i10.29009)

**Abstract**:

How our brain encodes complex concepts has been a longstanding mystery in neuroscience. The answer to this problem can lead to new understandings about how the brain retrieves information in large-scale data with high efficiency and robustness. Neuroscience studies suggest the brain represents concepts in a locality-sensitive hashing (LSH) strategy, i.e., similar concepts will be represented by similar responses. This finding has inspired the design of similarity-based algorithms, especially in contrastive learning. Here, we hypothesize that the brain and large neural network models, both using similarity-based learning rules, could contain a similar semantic embedding space. To verify that, this paper proposes a functional Magnetic Resonance Imaging (fMRI) semantic learning network named BrainSem, aimed at seeking a joint semantic latent space that bridges the brain and a Contrastive Language-Image Pre-training (CLIP) model. Given that our perception is inherently cross-modal, we introduce a fuzzy (one-to-many) matching loss function to encourage the models to extract high-level semantic components from neural signals. Our results claimed that using only a small set of fMRI recordings for semantic space alignment, we could obtain shared embedding valid for unseen categories out of the training set, which provided potential evidence for the semantic representation similarity between the brain and large neural networks. In a zero-shot classification task, our BrainSem achieves an 11.6% improvement over the state-of-the-art.

----

## [1261] On Disentanglement of Asymmetrical Knowledge Transfer for Modality-Task Agnostic Federated Learning

**Authors**: *Jiayi Chen, Aidong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29010](https://doi.org/10.1609/aaai.v38i10.29010)

**Abstract**:

There has been growing concern regarding data privacy during the development and deployment of Multimodal Foundation Models for Artificial General Intelligence (AGI), while Federated Learning (FL) allows multiple clients to collaboratively train models in a privacy-preserving manner. This paper formulates and studies Modality-task Agnostic Federated Learning (AFL) to pave the way toward privacy-preserving AGI. A unique property of AFL is the asymmetrical knowledge relationships among clients due to modality gaps, task gaps, and domain shifts between clients. This raises a challenge in learning an optimal inter-client information-sharing scheme that maximizes positive transfer and minimizes negative transfer for AFL. However, prior FL methods, mostly focusing on symmetrical knowledge transfer, tend to exhibit insufficient positive transfer and fail to fully avoid negative transfer during inter-client collaboration. To address this issue, we propose DisentAFL, which leverages a two-stage Knowledge Disentanglement and Gating mechanism to explicitly decompose the original asymmetrical inter-client information-sharing scheme into several independent symmetrical inter-client information-sharing schemes, each of which corresponds to certain semantic knowledge type learned from the local tasks. Experimental results demonstrate the superiority of our method on AFL than baselines.

----

## [1262] Accelerate Multi-Agent Reinforcement Learning in Zero-Sum Games with Subgame Curriculum Learning

**Authors**: *Jiayu Chen, Zelai Xu, Yunfei Li, Chao Yu, Jiaming Song, Huazhong Yang, Fei Fang, Yu Wang, Yi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29011](https://doi.org/10.1609/aaai.v38i10.29011)

**Abstract**:

Learning Nash equilibrium (NE) in complex zero-sum games with multi-agent reinforcement learning (MARL) can be extremely computationally expensive. Curriculum learning is an effective way to accelerate learning, but an under-explored dimension for generating a curriculum is the difficulty-to-learn of the subgames –games induced by starting from a specific state. In this work, we present a novel subgame curriculum learning framework for zero-sum games. It adopts an adaptive initial state distribution by resetting agents to some previously visited states where they can quickly learn to improve performance. Building upon this framework, we derive a subgame selection metric that approximates the squared distance to NE values and further adopt a particle-based state sampler for subgame generation. Integrating these techniques leads to our new algorithm, Subgame Automatic Curriculum Learning (SACL), which is a realization of the subgame curriculum learning framework. SACL can be combined with any MARL algorithm such as MAPPO. Experiments in the particle-world environment and Google Research Football environment show SACL produces much stronger policies than baselines. In the challenging hide-and-seek quadrant environment, SACL produces all four emergent stages and uses only half the samples of MAPPO with self-play. The project website is at https://sites.google.com/view/sacl-neurips.

----

## [1263] Watch Your Head: Assembling Projection Heads to Save the Reliability of Federated Models

**Authors**: *Jinqian Chen, Jihua Zhu, Qinghai Zheng, Zhongyu Li, Zhiqiang Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29012](https://doi.org/10.1609/aaai.v38i10.29012)

**Abstract**:

Federated learning encounters substantial challenges with heterogeneous data, leading to performance degradation and convergence issues. While considerable progress has been achieved in mitigating such an impact, the reliability aspect of federated models has been largely disregarded. In this study, we conduct extensive experiments to investigate the reliability of both generic and personalized federated models. Our exploration uncovers a significant finding: federated models exhibit unreliability when faced with heterogeneous data, demonstrating poor calibration on in-distribution test data and low uncertainty levels on out-of-distribution data. This unreliability is primarily attributed to the presence of biased projection heads, which introduce miscalibration into the federated models. Inspired by this observation, we propose the "Assembled Projection Heads" (APH) method for enhancing the reliability of federated models. By treating the existing projection head parameters as priors, APH randomly samples multiple initialized parameters of projection heads from the prior and further performs targeted fine-tuning on locally available data under varying learning rates. Such a head ensemble introduces parameter diversity into the deterministic model, eliminating the bias and producing reliable predictions via head averaging. We evaluate the effectiveness of the proposed APH method across three prominent federated benchmarks. Experimental results validate the efficacy of APH in model calibration and uncertainty estimation. Notably, APH can be seamlessly integrated into various federated approaches but only requires less than 30% additional computation cost for 100x inferences within large models.

----

## [1264] Discriminative Forests Improve Generative Diversity for Generative Adversarial Networks

**Authors**: *Junjie Chen, Jiahao Li, Chen Song, Bin Li, Qingcai Chen, Hongchang Gao, Wendy Hui Wang, Zenglin Xu, Xinghua Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29013](https://doi.org/10.1609/aaai.v38i10.29013)

**Abstract**:

Improving the diversity of Artificial Intelligence Generated Content (AIGC) is one of the fundamental problems in the theory of generative models such as generative adversarial networks (GANs). Previous studies have demonstrated that the discriminator in GANs should have high capacity and robustness to achieve the diversity of generated data. However, a discriminator with high capacity tends to overfit and guide the generator toward collapsed equilibrium. In this study, we propose a novel discriminative forest GAN, named Forest-GAN, that replaces the discriminator to improve the capacity and robustness for modeling statistics in real-world data distribution. A discriminative forest is composed of multiple independent discriminators built on bootstrapped data. We prove that a discriminative forest has a generalization error bound, which is determined by the strength of individual discriminators and the correlations among them. Hence, a discriminative forest can provide very large capacity without any risk of overfitting, which subsequently improves the generative diversity. With the discriminative forest framework, we significantly improved the performance of AutoGAN with a new record FID of 19.27 from 30.71 on STL10 and improved the performance of StyleGAN2-ADA with a new record FID of 6.87 from 9.22 on LSUN-cat.

----

## [1265] Efficient Algorithms for Non-gaussian Single Index Models with Generative Priors

**Authors**: *Junren Chen, Zhaoqiang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29014](https://doi.org/10.1609/aaai.v38i10.29014)

**Abstract**:

In this work, we focus on high-dimensional single index models with non-Gaussian sensing vectors and generative priors. More specifically, our goal is to estimate the underlying signal from i.i.d. realizations of the semi-parameterized single index model, where the underlying signal is contained in (up to a constant scaling) the range of a Lipschitz continuous generative model with bounded low-dimensional inputs, the sensing vector follows a non-Gaussian distribution, the noise is a random variable that is independent of the sensing vector, and the unknown non-linear link function is differentiable. Using the first- and second-order Stein's identity, we introduce efficient algorithms to obtain estimated vectors that achieve the near-optimal statistical rate. Experimental results on image datasets are provided to support our theory.

----

## [1266] Audio Scanning Network: Bridging Time and Frequency Domains for Audio Classification

**Authors**: *Liangwei Chen, Xiren Zhou, Huanhuan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29015](https://doi.org/10.1609/aaai.v38i10.29015)

**Abstract**:

With the rapid growth of audio data, there's a pressing need for automatic audio classification. As a type of time-series data, audio exhibits waveform fluctuations in both the time and frequency domains that evolve over time, with similar instances sharing consistent patterns. This study introduces the Audio Scanning Network (ASNet), designed to leverage abundant information for achieving stable and effective audio classification. ASNet captures real-time changes in audio waveforms across both time and frequency domains through reservoir computing, supported by Reservoir Kernel Canonical Correlation Analysis (RKCCA) to explore correlations between time-domain and frequency-domain waveform fluctuations. This innovative approach empowers ASNet to comprehensively capture the changes and inherent correlations within the audio waveform, and without the need for time-consuming iterative training. Instead of converting audio into spectrograms, ASNet directly utilizes audio feature sequences to uncover associations between time and frequency fluctuations. Experiments on environmental sound and music genre classification tasks demonstrate ASNet's comparable performance to state-of-the-art methods.

----

## [1267] Deep Contrastive Graph Learning with Clustering-Oriented Guidance

**Authors**: *Mulin Chen, Bocheng Wang, Xuelong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29016](https://doi.org/10.1609/aaai.v38i10.29016)

**Abstract**:

Graph Convolutional Network (GCN) has exhibited remarkable potential in improving graph-based clustering. To handle the general clustering scenario without a prior graph, these models estimate an initial graph beforehand to apply GCN. Throughout the literature, we have witnessed that 1) most models focus on the initial graph while neglecting the original features. Therefore, the discriminability of the learned representation may be corrupted by a low-quality initial graph; 2) the training procedure lacks effective clustering guidance, which may lead to the incorporation of clustering-irrelevant information into the learned graph. To tackle these problems, the Deep Contrastive Graph Learning (DCGL) model is proposed for general data clustering. Specifically, we establish a pseudo-siamese network, which incorporates auto-encoder with GCN to emphasize both the graph structure and the original features. On this basis, feature-level contrastive learning is introduced to enhance the discriminative capacity, and the relationship between samples and centroids is employed as the clustering-oriented guidance. Afterward, a two-branch graph learning mechanism is designed to extract the local and global structural relationships, which are further embedded into a unified graph under the cluster-level contrastive guidance. Experimental results on several benchmark datasets demonstrate the superiority of DCGL against state-of-the-art algorithms.

----

## [1268] On the Unstable Convergence Regime of Gradient Descent

**Authors**: *Shuo Chen, Jiaying Peng, Xiaolong Li, Yao Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29017](https://doi.org/10.1609/aaai.v38i10.29017)

**Abstract**:

Traditional gradient descent (GD) has been fully investigated for convex or L-smoothness functions, and it is widely utilized in current neural network optimization. The classical descent lemma ensures that for a function with L-smoothness, the GD trajectory converges stably towards the minimum when the learning rate is below 2 / L. This convergence is marked by a consistent reduction in the loss function throughout the iterations. However, recent experimental studies have demonstrated that even when the L-smoothness condition is not met, or if the learning rate is increased leading to oscillations in the loss function during iterations, the GD trajectory still exhibits convergence over the long run. This phenomenon is referred to as the unstable convergence regime of GD. In this paper, we present a theoretical perspective to offer a qualitative analysis of this phenomenon. The unstable convergence is in fact an inherent property of GD for general twice differentiable functions. Specifically, the forwardinvariance of GD is established, i.e., it ensures that any point within a local region will always remain within this region under GD iteration. Then, based on the forward-invariance, for the initialization outside an open set containing the local minimum, the loss function will oscillate at the first several iterations and then become monotonely decreasing after the GD trajectory jumped into the open set. This work theoretically clarifies the unstable convergence phenomenon of GD discussed in previous experimental works. The unstable convergence of GD mainly depends on the selection of the initialization, and it is actually inevitable due to the complex nature of loss function.

----

## [1269] PG-LBO: Enhancing High-Dimensional Bayesian Optimization with Pseudo-Label and Gaussian Process Guidance

**Authors**: *Taicai Chen, Yue Duan, Dong Li, Lei Qi, Yinghuan Shi, Yang Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29018](https://doi.org/10.1609/aaai.v38i10.29018)

**Abstract**:

Variational Autoencoder based Bayesian Optimization (VAE-BO) has demonstrated its excellent performance in addressing high-dimensional structured optimization problems. However, current mainstream methods overlook the potential of utilizing a pool of unlabeled data to construct the latent space, while only concentrating on designing sophisticated models to leverage the labeled data. Despite their effective usage of labeled data, these methods often require extra network structures, additional procedure, resulting in computational inefficiency. To address this issue, we propose a novel method to effectively utilize unlabeled data with the guidance of labeled data. Specifically, we tailor the pseudo-labeling technique from semi-supervised learning to explicitly reveal the relative magnitudes of optimization objective values hidden within the unlabeled data. Based on this technique, we assign appropriate training weights to unlabeled data to enhance the construction of a discriminative latent space. Furthermore, we treat the VAE encoder and the Gaussian Process (GP) in Bayesian optimization as a unified deep kernel learning process, allowing the direct utilization of labeled data, which we term as Gaussian Process guidance. This directly and effectively integrates the goal of improving GP accuracy into the VAE training, thereby guiding the construction of the latent space. The extensive experiments demonstrate that our proposed method outperforms existing VAE-BO algorithms in various optimization scenarios. Our code will be published at https://github.com/TaicaiChen/PG-LBO.

----

## [1270] DGPO: Discovering Multiple Strategies with Diversity-Guided Policy Optimization

**Authors**: *Wentse Chen, Shiyu Huang, Yuan Chiang, Tim Pearce, Wei-Wei Tu, Ting Chen, Jun Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29019](https://doi.org/10.1609/aaai.v38i10.29019)

**Abstract**:

Most reinforcement learning algorithms seek a single optimal strategy that solves a given task. However, it can often be valuable to learn a diverse set of solutions, for instance, to make an agent's interaction with users more engaging, or improve the robustness of a policy to an unexpected perturbance. We propose Diversity-Guided Policy Optimization (DGPO), an on-policy algorithm that discovers multiple strategies for solving a given task. Unlike prior work, it achieves this with a shared policy network trained over a single run. Specifically, we design an intrinsic reward based on an information-theoretic diversity objective. Our final objective alternately constraints on the diversity of the strategies and on the extrinsic reward. We solve the constrained optimization problem by casting it as a probabilistic inference task and use policy iteration to maximize the derived lower bound. Experimental results show that our method efficiently discovers diverse strategies in a wide variety of reinforcement learning tasks. Compared to baseline methods, DGPO achieves comparable rewards, while discovering more diverse strategies, and often with better sample efficiency.

----

## [1271] Exploiting Symmetric Temporally Sparse BPTT for Efficient RNN Training

**Authors**: *Xi Chen, Chang Gao, Zuowen Wang, Longbiao Cheng, Sheng Zhou, Shih-Chii Liu, Tobi Delbruck*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29020](https://doi.org/10.1609/aaai.v38i10.29020)

**Abstract**:

Recurrent Neural Networks (RNNs) are useful in temporal sequence tasks. However, training RNNs involves dense matrix multiplications which require hardware that can support a large number of arithmetic operations and memory accesses. Implementing online training of RNNs on the edge calls for optimized algorithms for an efficient deployment on hardware. Inspired by the spiking neuron model, the Delta RNN exploits temporal sparsity during inference by skipping over the update of hidden states from those inactivated neurons whose change of activation across two timesteps is below a defined threshold. This work describes a training algorithm for Delta RNNs that exploits temporal sparsity in the backward propagation phase to reduce computational requirements for training on the edge. Due to the symmetric computation graphs of forward and backward propagation during training, the gradient computation of inactivated neurons can be skipped. Results show a reduction of ∼80% in matrix operations for training a 56k parameter Delta LSTM on the Fluent Speech Commands dataset with negligible accuracy loss. Logic simulations of a hardware accelerator designed for the training algorithm show 2-10X speedup in matrix computations for an activation sparsity range of 50%-90%. Additionally, we show that the proposed Delta RNN training will be useful for online incremental learning on edge devices with limited computing resources.

----

## [1272] Meta-Inverse Reinforcement Learning for Mean Field Games via Probabilistic Context Variables

**Authors**: *Yang Chen, Xiao Lin, Bo Yan, Libo Zhang, Jiamou Liu, Neset Özkan Tan, Michael Witbrock*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29021](https://doi.org/10.1609/aaai.v38i10.29021)

**Abstract**:

Designing suitable reward functions for numerous interacting intelligent agents is challenging in real-world applications. Inverse reinforcement learning (IRL) in mean field games (MFGs) offers a practical framework to infer reward functions from expert demonstrations. While promising, the assumption of agent homogeneity limits the capability of existing methods to handle demonstrations with heterogeneous and unknown objectives, which are common in practice. To this end, we propose a deep latent variable MFG model and an associated IRL method. Critically, our method can infer rewards from different yet structurally similar tasks without prior knowledge about underlying contexts or modifying the MFG model itself. Our experiments, conducted on simulated scenarios and a real-world spatial taxi-ride pricing problem, demonstrate the superiority of our approach over state-of-the-art IRL methods in MFGs.

----

## [1273] Exact Policy Recovery in Offline RL with Both Heavy-Tailed Rewards and Data Corruption

**Authors**: *Yiding Chen, Xuezhou Zhang, Qiaomin Xie, Xiaojin Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29022](https://doi.org/10.1609/aaai.v38i10.29022)

**Abstract**:

We study offline reinforcement learning (RL) with heavy-tailed reward distribution and data corruption: (i) Moving beyond subGaussian reward distribution, we allow the rewards to have infinite variances; (ii) We allow corruptions where an attacker can arbitrarily modify a small fraction of the rewards and transitions in the dataset. We first derive a sufficient optimality condition for generalized Pessimistic Value Iteration (PEVI), which allows various estimators with proper confidence bounds and can be applied to multiple learning settings. In order to handle the data corruption and heavy-tailed reward setting, we prove that the trimmed-mean estimation achieves the minimax optimal error rate for robust mean estimation under heavy-tailed distributions. In the PEVI algorithm, we plug in the trimmed mean estimation and the confidence bound to solve the robust offline RL problem. Standard analysis reveals that data corruption induces a bias term in the suboptimality gap, which gives the false impression that any data corruption prevents optimal policy learning. By using the optimality condition for the generalized PEVI, we show that as long as the bias term is less than the ``action gap'', the policy returned by PEVI achieves the optimal value given sufficient data.

----

## [1274] Progressive Poisoned Data Isolation for Training-Time Backdoor Defense

**Authors**: *Yiming Chen, Haiwei Wu, Jiantao Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29023](https://doi.org/10.1609/aaai.v38i10.29023)

**Abstract**:

Deep Neural Networks (DNN) are susceptible to backdoor attacks where malicious attackers manipulate the model's predictions via data poisoning. It is hence imperative to develop a strategy for training a clean model using a potentially poisoned dataset. Previous training-time defense mechanisms typically employ an one-time isolation process, often leading to suboptimal isolation outcomes. In this study, we present a novel and efficacious defense method, termed Progressive Isolation of Poisoned Data (PIPD), that progressively isolates poisoned data to enhance the isolation accuracy and mitigate the risk of benign samples being misclassified as poisoned ones. Once the poisoned portion of the dataset has been identified, we introduce a selective training process to train a clean model. Through the implementation of these techniques, we ensure that the trained model manifests a significantly diminished attack success rate against the poisoned data. Extensive experiments on multiple benchmark datasets and DNN models, assessed against nine state-of-the-art backdoor attacks, demonstrate the superior performance of our PIPD method for backdoor defense. For instance, our PIPD achieves an average True Positive Rate (TPR) of 99.95% and an average False Positive Rate (FPR) of 0.06% for diverse attacks over CIFAR-10 dataset, markedly surpassing the performance of state-of-the-art methods. The code is available at https://github.com/RorschachChen/PIPD.git.

----

## [1275] Pushing the Limit of Fine-Tuning for Few-Shot Learning: Where Feature Reusing Meets Cross-Scale Attention

**Authors**: *Ying-Yu Chen, Jun-Wei Hsieh, Xin Li, Ming-Ching Chang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29024](https://doi.org/10.1609/aaai.v38i10.29024)

**Abstract**:

Due to the scarcity of training samples, Few-Shot Learning (FSL) poses a significant challenge to capture discriminative object features effectively. The combination of transfer learning and meta-learning has recently been explored by pre-training the backbone features using labeled base data and subsequently fine-tuning the model with target data. However, existing meta-learning methods, which use embedding networks, suffer from scaling limitations when dealing with a few labeled samples, resulting in suboptimal results. Inspired by the latest advances in FSL, we further advance the approach of fine-tuning a pre-trained architecture by a strengthened hierarchical feature representation. The technical contributions of this work include: 1) a hybrid design named Intra-Block Fusion (IBF) to strengthen the extracted features within each convolution block; and 2) a novel Cross-Scale Attention (CSA) module to mitigate the scaling inconsistencies arising from the limited training samples, especially for cross-domain tasks.  We conducted comprehensive evaluations on standard benchmarks, including three in-domain tasks (miniImageNet, CIFAR-FS, and FC100), as well as two cross-domain tasks (CDFSL and Meta-Dataset). The results have improved significantly over existing state-of-the-art approaches on all benchmark datasets. In particular, the FSL performance on the in-domain FC100 dataset is more than three points better than the latest PMF (Hu et al. 2022).

----

## [1276] Fed-QSSL: A Framework for Personalized Federated Learning under Bitwidth and Data Heterogeneity

**Authors**: *Yiyue Chen, Haris Vikalo, Chianing Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29025](https://doi.org/10.1609/aaai.v38i10.29025)

**Abstract**:

Motivated by high resource costs of centralized machine learning schemes as well as data privacy concerns, federated learning (FL) emerged as an efficient alternative that relies on aggregating locally trained models rather than collecting clients' potentially private data. In practice, available resources and data distributions vary from one client to another, creating an inherent system heterogeneity that leads to deterioration of the performance of conventional FL algorithms. In this work, we present a federated quantization-based self-supervised learning scheme (Fed-QSSL) designed to address heterogeneity in FL systems. At clients' side, to tackle data heterogeneity we leverage distributed self-supervised learning while utilizing low-bit quantization to satisfy constraints imposed by local infrastructure and limited communication resources. At server's side, Fed-QSSL deploys de-quantization, weighted aggregation and re-quantization, ultimately creating models personalized to both data distribution as well as specific infrastructure of each client's device. We validated the proposed algorithm on real world datasets, demonstrating its efficacy, and theoretically analyzed impact of low-bit training on the convergence and robustness of the learned models.

----

## [1277] TopoGCL: Topological Graph Contrastive Learning

**Authors**: *Yuzhou Chen, José Frías, Yulia R. Gel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29026](https://doi.org/10.1609/aaai.v38i10.29026)

**Abstract**:

Graph contrastive learning (GCL) has recently emerged as a new concept which allows for capitalizing on the  strengths of graph neural networks (GNNs) to learn rich representations in a wide variety of applications which involve abundant unlabeled information. However, existing GCL approaches largely tend to overlook the important latent information on higher-order graph substructures. We address this limitation by introducing the concepts of topological invariance and extended persistence on graphs to GCL. In particular, we propose a new contrastive mode which targets topological representations of the two augmented views from the same graph, yielded by extracting latent shape properties of the graph at multiple resolutions. Along with the extended topological layer, we introduce a new extended persistence summary, namely, extended persistence landscapes (EPL) and derive its theoretical stability guarantees. Our extensive numerical results on biological, chemical, and social interaction graphs show that the new Topological Graph Contrastive Learning (TopoGCL) model delivers significant performance gains in unsupervised graph classification for 8 out of 12 considered datasets and also exhibits robustness under noisy scenarios.

----

## [1278] Continuous Rotation Group Equivariant Network Inspired by Neural Population Coding

**Authors**: *Zhiqiang Chen, Yang Chen, Xiaolong Zou, Shan Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29027](https://doi.org/10.1609/aaai.v38i10.29027)

**Abstract**:

Neural population coding can represent continuous information by neurons with a series of discrete preferred stimuli, and we find that the bell-shaped tuning curve plays an important role in this mechanism. Inspired by this, we incorporate a bell-shaped tuning curve into the discrete group convolution to achieve continuous group equivariance. Simply, we modulate group convolution kernels by Gauss functions to obtain bell-shaped tuning curves. Benefiting from the modulation, kernels also gain smooth gradients on geometric dimensions (e.g., location dimension and orientation dimension). It allows us to generate group convolution kernels from sparse weights with learnable geometric parameters, which can achieve both competitive performances and parameter efficiencies. Furthermore, we quantitatively prove that discrete group convolutions with proper tuning curves (bigger than 1x sampling step) can achieve continuous equivariance. Experimental results show that 1) our approach achieves very competitive performances on MNIST-rot with at least 75% fewer parameters compared with previous SOTA methods, which is efficient in parameter; 2) Especially with small sample sizes, our approach exhibits more pronounced performance improvements (up to 24%); 3) It also has excellent rotation generalization ability on various datasets such as MNIST, CIFAR, and ImageNet with both plain and ResNet architectures.

----

## [1279] Diagnosing and Rectifying Fake OOD Invariance: A Restructured Causal Approach

**Authors**: *Ziliang Chen, Yongsen Zheng, Zhao-Rong Lai, Quanlong Guan, Liang Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29028](https://doi.org/10.1609/aaai.v38i10.29028)

**Abstract**:

Invariant representation learning (IRL) encourages the prediction from invariant causal features to labels deconfounded from the environments, advancing the technical roadmap of out-of-distribution (OOD) generalization. Despite spotlights around, recent theoretical result verified that some causal features recovered by IRLs merely pretend domain-invariantly in the training environments but fail in unseen domains. The fake invariance severely endangers OOD generalization since the trustful objective can not be diagnosed and existing causal remedies are invalid to rectify. In this paper, we review a IRL family (InvRat) under the Partially and Fully Informative Invariant Feature Structural Causal Models (PIIF SCM /FIIF SCM) respectively, to certify their weaknesses in representing fake invariant features, then, unify their causal diagrams to propose ReStructured SCM (RS-SCM). RS-SCM can ideally rebuild the spurious and the fake invariant features simultaneously. Given this, we further develop an approach based on conditional mutual information with respect to RS-SCM, then rigorously rectify the spurious and fake invariant effects. It can be easily implemented by a small feature selection subnet introduced in the IRL family, which is alternatively optimized to achieve our goal. Experiments verified the superiority of our approach to fight against the fake invariant issue across a variety of OOD generalization benchmarks.

----

## [1280] Instrumental Variable Estimation for Causal Inference in Longitudinal Data with Time-Dependent Latent Confounders

**Authors**: *Debo Cheng, Ziqi Xu, Jiuyong Li, Lin Liu, Jixue Liu, Wentao Gao, Thuc Duy Le*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29029](https://doi.org/10.1609/aaai.v38i10.29029)

**Abstract**:

Causal inference from longitudinal observational data is a challenging problem due to the difficulty in correctly identifying the time-dependent confounders, especially in the presence of latent time-dependent confounders. Instrumental variable (IV) is a powerful tool for addressing the latent confounders issue, but the traditional IV technique cannot deal with latent time-dependent confounders in longitudinal studies. In this work, we propose a novel Time-dependent Instrumental Factor Model (TIFM) for time-varying causal effect estimation from data with latent time-dependent confounders. At each time-step, the proposed TIFM method employs the Recurrent Neural Network (RNN) architecture to infer latent IV, and then uses the inferred latent IV factor for addressing the confounding bias caused by the latent time-dependent confounders. We provide a theoretical analysis for the proposed TIFM method regarding causal effect estimation in longitudinal data. Extensive evaluation with synthetic datasets demonstrates the effectiveness of TIFM in addressing causal effect estimation over time. We further apply TIFM to a climate dataset to showcase the potential of the proposed method in tackling real-world problems.

----

## [1281] Hierarchize Pareto Dominance in Multi-Objective Stochastic Linear Bandits

**Authors**: *Ji Cheng, Bo Xue, Jiaxiang Yi, Qingfu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29030](https://doi.org/10.1609/aaai.v38i10.29030)

**Abstract**:

Multi-objective Stochastic Linear bandit (MOSLB) plays a critical role in the sequential decision-making paradigm, however, most existing methods focus on the Pareto dominance among different objectives without considering any priority. In this paper, we study bandit algorithms under mixed Pareto-lexicographic orders, which can reflect decision makers' preferences. We adopt the Grossone approach to deal with these orders and develop the notion of Pareto-lexicographic optimality to evaluate the learners' performance. Our work represents a first attempt to address these important and realistic orders in bandit algorithms. To design algorithms under these orders, the upper confidence bound (UCB) policy and the prior free lexicographical filter are adapted to approximate the optimal arms at each round. Moreover, the framework of the algorithms involves two stages in pursuit of the balance between exploration and exploitation. Theoretical analysis as well as numerical experiments demonstrate the effectiveness of our algorithms.

----

## [1282] FedGCR: Achieving Performance and Fairness for Federated Learning with Distinct Client Types via Group Customization and Reweighting

**Authors**: *Shu-Ling Cheng, Chin-Yuan Yeh, Ting-An Chen, Eliana Pastor, Ming-Syan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29031](https://doi.org/10.1609/aaai.v38i10.29031)

**Abstract**:

To achieve better performance and greater fairness in Federated Learning (FL), much of the existing research has centered on individual clients, using domain adaptation techniques and redesigned aggregation schemes to counteract client data heterogeneity. However, an overlooked scenario exists where clients belong to distinctive groups, or, client types, in which groups of clients share similar characteristics such as device specifications or data patterns. Despite being common in group collaborations, this scenario has been overlooked in previous research, potentially leading to performance degradation and systemic biases against certain client types. To bridge this gap, we introduce Federated learning with Group Customization and Reweighting (FedGCR). FedGCR enhances both performance and fairness for FL with Distinct Client Types, consisting of a Federated Group Customization (FedGC) model to provide customization via a novel prompt tuning technique to mitigate the data disparity across different client-types, and a Federated Group Reweighting (FedGR) aggregation scheme to ensure uniform and unbiased performances between clients and between client types by a novel reweighting approach. Extensive experiment comparisons with prior FL methods in domain adaptation and fairness demonstrate the superiority of FedGCR in all metrics, including the overall accuracy and performance uniformity in both the group and the individual level. FedGCR achieves 82.74% accuracy and 12.26(↓) in performance uniformity on the Digit-Five dataset and 81.88% and 14.88%(↓) on DomainNet with a domain imbalance factor of 10, which significantly outperforms the state-of-the-art. Code is available at https://github.com/celinezheng/fedgcr.

----

## [1283] Clarifying the Behavior and the Difficulty of Adversarial Training

**Authors**: *Xu Cheng, Hao Zhang, Yue Xin, Wen Shen, Quanshi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29032](https://doi.org/10.1609/aaai.v38i10.29032)

**Abstract**:

Adversarial training is usually difficult to optimize. This paper provides conceptual and analytic insights into the difficulty of adversarial training via a simple theoretical study, where we derive an approximate dynamics of a recursive multi-step attack in a simple setting. Despite the simplicity of our theory, it still reveals verifiable predictions about various phenomena in adversarial training under real-world settings. First, compared to vanilla training, adversarial training is more likely to boost the influence of input samples with large gradient norms in an exponential manner. Besides, adversarial training also strengthens the influence of the Hessian matrix of the loss w.r.t. network parameters, which is more likely to make network parameters oscillate and boosts the difficulty of adversarial training.

----

## [1284] Arithmetic Feature Interaction Is Necessary for Deep Tabular Learning

**Authors**: *Yi Cheng, Renjun Hu, Haochao Ying, Xing Shi, Jian Wu, Wei Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29033](https://doi.org/10.1609/aaai.v38i10.29033)

**Abstract**:

Until recently, the question of the effective inductive bias of deep models on tabular data has remained unanswered. This paper investigates the hypothesis that arithmetic feature interaction is necessary for deep tabular learning. To test this point, we create a synthetic tabular dataset with a mild feature interaction assumption and examine a modified transformer architecture enabling arithmetical feature interactions, referred to as AMFormer. Results show that AMFormer outperforms strong counterparts in fine-grained tabular data modeling, data efficiency in training, and generalization. This is attributed to its parallel additive and multiplicative attention operators and prompt-based optimization, which facilitate the separation of tabular samples in an extended space with arithmetically-engineered features. Our extensive experiments on real-world data also validate the consistent effectiveness, efficiency, and rationale of AMFormer, suggesting it has established a strong inductive bias for deep learning on tabular data. Code is available at https://github.com/aigc-apps/AMFormer.

----

## [1285] CUTS+: High-Dimensional Causal Discovery from Irregular Time-Series

**Authors**: *Yuxiao Cheng, Lianglong Li, Tingxiong Xiao, Zongren Li, Jinli Suo, Kunlun He, Qionghai Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29034](https://doi.org/10.1609/aaai.v38i10.29034)

**Abstract**:

Causal discovery in time-series is a fundamental problem in the machine learning community, enabling causal reasoning and decision-making in complex scenarios. Recently, researchers successfully discover causality by combining neural networks with Granger causality, but their performances degrade largely when encountering high-dimensional data because of the highly redundant network design and huge causal graphs. Moreover, the missing entries in the observations further hamper the causal structural learning. To overcome these limitations, We propose CUTS+, which is built on the Granger-causality-based causal discovery method CUTS and raises the scalability by introducing a technique called Coarse-to-fine-discovery (C2FD) and leveraging a message-passing-based graph neural network (MPGNN). Compared to previous methods on simulated, quasi-real, and real datasets, we show that CUTS+ largely improves the causal discovery performance on high-dimensional data with different types of irregular sampling.

----

## [1286] Generalized Variational Inference via Optimal Transport

**Authors**: *Jinjin Chi, Zhichao Zhang, Zhiyao Yang, Jihong Ouyang, Hongbin Pei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29035](https://doi.org/10.1609/aaai.v38i10.29035)

**Abstract**:

Variational Inference (VI) has gained popularity as a flexible approximate inference scheme for computing posterior distributions in Bayesian models. Original VI methods use Kullback-Leibler (KL) divergence to construct variational objectives. However, KL divergence has zero-forcing behavior and is completely agnostic to the metric of the underlying data distribution, resulting in bad approximations. To alleviate this issue, we propose a new variational objective by using Optimal Transport (OT) distance, which is a metric-aware divergence, to measure the difference between approximate posteriors and priors. The superior performance of OT distance enables us to learn more accurate approximations. We further enhance the objective by gradually including the OT term using a hyperparameter λ for over-parameterized models. We develop a Variational inference method with OT (VOT) which presents a gradient-based black-box framework for solving Bayesian models, even when the density function of approximate distribution is not available. We provide the consistency analysis of approximate posteriors and demonstrate the practical effectiveness on Bayesian neural networks and variational autoencoders.

----

## [1287] Operator-Learning-Inspired Modeling of Neural Ordinary Differential Equations

**Authors**: *Woojin Cho, Seunghyeon Cho, Hyundong Jin, Jinsung Jeon, Kookjin Lee, Sanghyun Hong, Dongeun Lee, Jonghyun Choi, Noseong Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29036](https://doi.org/10.1609/aaai.v38i10.29036)

**Abstract**:

Neural ordinary differential equations (NODEs), one of the most influential works of the differential equation-based deep learning, are to continuously generalize residual networks and opened a new field. They are currently utilized for various downstream tasks, e.g., image classification, time series classification, image generation, etc. Its key part is how to model the time-derivative of the hidden state, denoted dh(t)/dt. People have habitually used conventional neural network architectures, e.g., fully-connected layers followed by non-linear activations. In this paper, however, we present a neural operator-based method to define the time-derivative term. Neural operators were initially proposed to model the differential operator of partial differential equations (PDEs). Since the time-derivative of NODEs can be understood as a special type of the differential operator, our proposed method, called branched Fourier neural operator (BFNO), makes sense. In our experiments with general downstream tasks, our method significantly outperforms existing methods.

----

## [1288] Make Prompts Adaptable: Bayesian Modeling for Vision-Language Prompt Learning with Data-Dependent Prior

**Authors**: *Youngjae Cho, HeeSun Bae, Seungjae Shin, Yeo Dong Youn, Weonyoung Joo, Il-Chul Moon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29037](https://doi.org/10.1609/aaai.v38i10.29037)

**Abstract**:

Recent vision-language pre-trained (VLP) models have become the backbone for many downstream tasks, but they are utilized as frozen model without learning. Prompt learning is a method to improve the pre-trained VLP model by adding a learnable context vector to the inputs of the text encoder. In a few-shot learning scenario of the downstream task, MLE training can lead the context vector to over-fit dominant image features in the training data. This overfitting can potentially harm the generalization ability, especially in the presence of a distribution shift between the training and test dataset.  This paper presents a Bayesian-based framework of prompt tuning, which could alleviate the over-fitting issues on few-shot learning application and increase the adaptability of prompts on unobserved instances. Specifically, modeling data-dependent prior enhances the adaptability of text features for both seen and unseen image features without the trade-off of performance between them. Based on the Bayesian framework, we utilize the Wasserstein gradient flow in the estimation of our target posterior distribution, which enables our prompt to be flexible in capturing the complex modes of image features. We demonstrate the effectiveness of our method on benchmark datasets for several experiments by showing statistically significant improvements on performance compared to existing methods.

----

## [1289] SNN-PDE: Learning Dynamic PDEs from Data with Simplicial Neural Networks

**Authors**: *Jae Choi, Yuzhou Chen, Huikyo Lee, Hyun Kim, Yulia R. Gel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29038](https://doi.org/10.1609/aaai.v38i10.29038)

**Abstract**:

Dynamics of many complex systems, from weather and climate to spread of infectious diseases, can be described by partial differential equations (PDEs). Such PDEs involve unknown function(s),  partial derivatives, and typically multiple independent variables. The traditional numerical methods for solving PDEs assume that the data are observed on a regular grid. However, in many applications, for example, weather and air pollution monitoring delivered by the arbitrary located weather stations of the National Weather Services, data records are irregularly spaced. Furthermore, in problems involving prediction analytics such as forecasting wildfire smoke plumes, the primary focus may be on a set of irregular locations associated with urban development. In recent years, deep learning (DL) methods and, in particular, graph neural networks (GNNs) have emerged as a new promising tool that can complement traditional PDE solvers in scenarios of the irregular spaced data, contributing to the newest research trend of physics informed machine learning (PIML). However, most existing PIML methods tend to be limited in their ability to describe higher dimensional structural properties exhibited by real world phenomena, especially, ones that live on manifolds. To address this fundamental challenge, we bring the elements of the Hodge theory and, in particular, simplicial convolution defined on the Hodge Laplacian to the emerging nexus of DL and PDEs. In contrast to conventional Laplacian and the associated convolution operation, the simplicial convolution allows us to rigorously describe diffusion across higher order structures and to better approximate the complex underlying topology and geometry of the data. The new approach, Simplicial Neural Networks for Partial Differential Equations (SNN PDE) offers a computationally efficient yet effective solution for time dependent PDEs. Our studies of a broad range of synthetic data and wildfire processes demonstrate that SNN PDE improves upon state of the art baselines in handling unstructured grids and irregular time intervals of complex physical systems and offers competitive forecasting capabilities for weather and air quality forecasting.

----

## [1290] Unsupervised Object Interaction Learning with Counterfactual Dynamics Models

**Authors**: *Jongwook Choi, Sungtae Lee, Xinyu Wang, Sungryull Sohn, Honglak Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29039](https://doi.org/10.1609/aaai.v38i10.29039)

**Abstract**:

We present COIL (Counterfactual Object Interaction Learning), a novel way of learning skills of object interactions on entity-centric environments. The goal is to learn primitive behaviors that can induce interactions without external reward or any supervision. Existing skill discovery methods are limited to locomotion, simple navigation tasks, or single-object manipulation tasks, mostly not inducing interaction between objects. Unlike a monolithic representation usually used in prior skill learning methods, we propose to use a structured goal representation that can query and scope which objects to interact with, which can serve as a basis for solving more complex downstream tasks. We design a novel counterfactual intrinsic reward through the use of either a forward model or successor features that can learn an interaction skill between a pair of objects given as a goal. Through experiments on continuous control environments such as Magnetic Block and 2.5-D Stacking Box, we demonstrate that an agent can learn object interaction behaviors (e.g., attaching or stacking one block to another) without any external rewards or domain-specific knowledge.

----

## [1291] DUEL: Duplicate Elimination on Active Memory for Self-Supervised Class-Imbalanced Learning

**Authors**: *Won-Seok Choi, Hyundo Lee, Dong-Sig Han, Junseok Park, Heeyeon Koo, Byoung-Tak Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29040](https://doi.org/10.1609/aaai.v38i10.29040)

**Abstract**:

Recent machine learning algorithms have been developed using well-curated datasets, which often require substantial cost and resources. On the other hand, the direct use of raw data often leads to overfitting towards frequently occurring class information. To address class imbalances cost-efficiently, we propose an active data filtering process during self-supervised pre-training in our novel framework, Duplicate Elimination (DUEL). This framework integrates an active memory inspired by human working memory and introduces distinctiveness information, which measures the diversity of the data in the memory, to optimize both the feature extractor and the memory. The DUEL policy, which replaces the most duplicated data with new samples, aims to enhance the distinctiveness information in the memory and thereby mitigate class imbalances. We validate the effectiveness of the DUEL framework in class-imbalanced environments, demonstrating its robustness and providing reliable results in downstream tasks. We also analyze the role of the DUEL policy in the training process through various metrics and visualizations.

----

## [1292] Consistency-Guided Temperature Scaling Using Style and Content Information for Out-of-Domain Calibration

**Authors**: *Wonjeong Choi, Jungwuk Park, Dong-Jun Han, Younghyun Park, Jaekyun Moon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29041](https://doi.org/10.1609/aaai.v38i10.29041)

**Abstract**:

Research interests in the robustness of deep neural networks  against domain shifts have been rapidly increasing in recent years. Most existing works, however, focus on improving the accuracy of the model, not the calibration performance which is another important requirement for trustworthy AI systems. Temperature scaling (TS), an accuracy-preserving post-hoc calibration method, has been proven to be effective in in-domain settings, but not in out-of-domain (OOD) due to the difficulty in obtaining a validation set for the unseen domain beforehand. In this paper, we propose consistency-guided temperature scaling (CTS), a new temperature scaling strategy that can significantly enhance the OOD calibration performance by providing mutual supervision among data samples in the source domains. Motivated by our observation that over-confidence stemming from inconsistent sample predictions  is the main obstacle to OOD calibration, we propose to guide the scaling process by taking consistencies into account in terms of two different aspects - style and content - which are the key components that can well-represent data samples in multi-domain settings. Experimental results demonstrate that our proposed strategy outperforms existing works, achieving superior OOD calibration performance on various datasets. This can be accomplished by employing only the source domains without compromising accuracy, making our scheme directly applicable to various trustworthy AI systems.

----

## [1293] A Provably Accurate Randomized Sampling Algorithm for Logistic Regression

**Authors**: *Agniva Chowdhury, Pradeep Ramuhalli*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29042](https://doi.org/10.1609/aaai.v38i10.29042)

**Abstract**:

In statistics and machine learning, logistic regression is a widely-used supervised learning technique primarily employed for binary classification tasks. When the number of observations greatly exceeds the number of predictor variables, we present a simple, randomized sampling-based algorithm for logistic regression problem that guarantees high-quality approximations to both the estimated probabilities and the overall discrepancy of the model. Our analysis builds upon two simple structural conditions that boil down to randomized matrix multiplication, a fundamental and well-understood primitive of randomized numerical linear algebra. We analyze the properties of estimated probabilities of logistic regression when leverage scores are used to sample observations, and prove that accurate approximations can be achieved with a sample whose size is much smaller than the total number of observations. To further validate our theoretical findings, we conduct comprehensive empirical evaluations. Overall, our work sheds light on the potential of using randomized sampling approaches to efficiently approximate the estimated probabilities in logistic regression, offering a practical and computationally efficient solution for large-scale datasets.

----

## [1294] Graph-Based Prediction and Planning Policy Network (GP3Net) for Scalable Self-Driving in Dynamic Environments Using Deep Reinforcement Learning

**Authors**: *Jayabrata Chowdhury, Venkataramanan Shivaraman, Suresh Sundaram, P. B. Sujit*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29043](https://doi.org/10.1609/aaai.v38i10.29043)

**Abstract**:

Recent advancements in motion planning for Autonomous Vehicles (AVs) show great promise in using expert driver behaviors in non-stationary driving environments. However, learning only through expert drivers needs more generalizability to recover from domain shifts and near-failure scenarios due to the dynamic behavior of traffic participants and weather conditions. A deep Graph-based Prediction and Planning Policy Network (GP3Net) framework is proposed for non-stationary environments that encodes the interactions between traffic participants with contextual information and provides a decision for safe maneuver for AV. A spatio-temporal graph models the interactions between traffic participants for predicting the future trajectories of those participants. The predicted trajectories are utilized to generate a future occupancy map around the AV with uncertainties embedded to anticipate the evolving non-stationary driving environments. Then the contextual information and future occupancy maps are input to the policy network of the GP3Net framework and trained using Proximal Policy Optimization (PPO) algorithm. The proposed GP3Net performance is evaluated on standard CARLA benchmarking scenarios with domain shifts of traffic patterns (urban, highway, and mixed). The results show that the GP3Net outperforms previous state-of-the-art imitation learning-based planning models for different towns. Further, in unseen new weather conditions, GP3Net completes the desired route with fewer traffic infractions. Finally, the results emphasize the advantage of including the prediction module to enhance safety measures in non-stationary environments.

----

## [1295] Lyapunov-Stable Deep Equilibrium Models

**Authors**: *Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao, Yuto Miyatake*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29044](https://doi.org/10.1609/aaai.v38i10.29044)

**Abstract**:

Deep equilibrium (DEQ) models have emerged as a promising class of implicit layer models, which abandon traditional depth by solving for the fixed points of a single nonlinear layer. Despite their success, the stability of the fixed points for these models remains poorly understood. By considering DEQ models as nonlinear dynamic systems, we propose a robust DEQ model named LyaDEQ with guaranteed provable stability via Lyapunov theory. The crux of our method is ensuring the Lyapunov stability of the DEQ model's fixed points, which enables the proposed model to resist minor initial perturbations. To avoid poor adversarial defense due to Lyapunov-stable fixed points being located near each other, we orthogonalize the layers after the Lyapunov stability module to separate different fixed points. We evaluate LyaDEQ models under well-known adversarial attacks, and experimental results demonstrate significant improvement in robustness. Furthermore, we show that the LyaDEQ model can be combined with other defense methods, such as adversarial training, to achieve even better adversarial robustness.

----

## [1296] Make RepVGG Greater Again: A Quantization-Aware Approach

**Authors**: *Xiangxiang Chu, Liang Li, Bo Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29045](https://doi.org/10.1609/aaai.v38i10.29045)

**Abstract**:

The tradeoff between performance and inference speed is critical for practical applications. Architecture reparameterization obtains better tradeoffs and it is becoming an increasingly popular ingredient in modern convolutional neural networks. Nonetheless, its quantization performance is usually too poor to deploy (e.g. more than 20% top-1 accuracy drop on ImageNet) when INT8 inference is desired. In this paper, we dive into the underlying mechanism of this failure, where the original design inevitably enlarges quantization error. We propose a simple, robust, and effective remedy to have a quantization-friendly structure that also enjoys reparameterization benefits. Our method greatly bridges the gap between INT8 and FP32 accuracy for RepVGG. Without bells and whistles, the top-1 accuracy drop on ImageNet is reduced within 2% by standard post-training quantization. Extensive experiments on detection and semantic segmentation tasks verify its generalization.

----

## [1297] Meta-Reinforcement Learning via Exploratory Task Clustering

**Authors**: *Zhendong Chu, Renqin Cai, Hongning Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29046](https://doi.org/10.1609/aaai.v38i10.29046)

**Abstract**:

Meta-reinforcement learning (meta-RL) aims to quickly solve new RL tasks by leveraging knowledge from prior tasks. Previous studies often assume a single-mode homogeneous task distribution, ignoring possible structured heterogeneity among tasks. Such an oversight can hamper effective exploration and adaptation, especially with limited samples. In this work, we harness the structured heterogeneity among tasks via clustering to improve meta-RL, which facilitates knowledge sharing at the cluster level. To facilitate exploration, we also develop a dedicated cluster-level exploratory policy to discover task clusters via divide-and-conquer. The knowledge from the discovered clusters helps to narrow the search space of task-specific policy learning, leading to more sample-efficient policy adaptation. We evaluate the proposed method on environments with parametric clusters (e.g., rewards and state dynamics in the MuJoCo suite) and non-parametric clusters (e.g., control skills in the Meta-World suite). The results demonstrate strong advantages of our solution against a set of representative meta-RL methods.

----

## [1298] Task-Driven Causal Feature Distillation: Towards Trustworthy Risk Prediction

**Authors**: *Zhixuan Chu, Mengxuan Hu, Qing Cui, Longfei Li, Sheng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29047](https://doi.org/10.1609/aaai.v38i10.29047)

**Abstract**:

Since artificial intelligence has seen tremendous recent successes in many areas, it has sparked great interest in its potential for trustworthy and interpretable risk prediction. However, most models lack causal reasoning and struggle with class imbalance, leading to poor precision and recall. To address this, we propose a Task-Driven Causal Feature Distillation model (TDCFD) to transform original feature values into causal feature attributions for the specific risk prediction task. The causal feature attribution helps describe how much contribution the value of this feature can make to the risk prediction result. After the causal feature distillation, a deep neural network is applied to produce trustworthy prediction results with causal interpretability and high precision/recall. We evaluate the performance of our TDCFD method on several synthetic and real datasets, and the results demonstrate its superiority over the state-of-the-art methods regarding precision, recall, interpretability, and causality.

----

## [1299] Resource Efficient Deep Learning Hardware Watermarks with Signature Alignment

**Authors**: *Joseph Clements, Yingjie Lao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29048](https://doi.org/10.1609/aaai.v38i10.29048)

**Abstract**:

Deep learning intellectual properties (IPs) are high-value assets that are frequently susceptible to theft. This vulnerability has led to significant interest in defending the field's intellectual properties from theft. Recently, watermarking techniques have been extended to protect deep learning hardware from privacy. These technique embed modifications that change the hardware's behavior when activated. In this work, we propose the first method for embedding watermarks in deep learning hardware that incorporates the owner's key samples into the embedding methodology. This improves our watermarks' reliability and efficiency in identifying the hardware over those generated using randomly selected key samples. Our experimental results demonstrate that by considering the target key samples when generating the hardware modifications, we can significantly increase the embedding success rate while targeting fewer functional blocks, decreasing the required hardware overhead needed to defend it.

----

## [1300] RLfOLD: Reinforcement Learning from Online Demonstrations in Urban Autonomous Driving

**Authors**: *Daniel Coelho, Miguel Oliveira, Vitor Santos*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29049](https://doi.org/10.1609/aaai.v38i10.29049)

**Abstract**:

Reinforcement Learning from Demonstrations (RLfD) has emerged as an effective method by fusing expert demonstrations into Reinforcement Learning (RL) training, harnessing the strengths of both Imitation Learning (IL) and RL. However, existing algorithms rely on offline demonstrations, which can introduce a distribution gap between the demonstrations and the actual training environment, limiting their performance. In this paper, we propose a novel approach, Reinforcement Learning from Online Demonstrations (RLfOLD), that leverages online demonstrations to address this limitation, ensuring the agent learns from relevant and up-to-date scenarios, thus effectively bridging the distribution gap. Unlike conventional policy networks used in typical actor-critic algorithms, RLfOLD introduces a policy network that outputs two standard deviations: one for exploration and the other for IL training. This novel design allows the agent to adapt to varying levels of uncertainty inherent in both RL and IL. Furthermore, we introduce an exploration process guided by an online expert, incorporating an uncertainty-based technique. Our experiments on the CARLA NoCrash benchmark demonstrate the effectiveness and efficiency of RLfOLD. Notably, even with a significantly smaller encoder and a single camera setup, RLfOLD surpasses state-of-the-art methods in this evaluation. These results, achieved with limited resources, highlight RLfOLD as a highly promising solution for real-world applications.

----

## [1301] Big Learning Expectation Maximization

**Authors**: *Yulai Cong, Sijia Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29050](https://doi.org/10.1609/aaai.v38i10.29050)

**Abstract**:

Mixture models serve as one fundamental tool with versatile applications. However, their training techniques, like the popular Expectation Maximization (EM) algorithm, are notoriously sensitive to parameter initialization and often suffer from bad local optima that could be arbitrarily worse than the optimal. To address the long-lasting bad-local-optima challenge, we draw inspiration from the recent ground-breaking foundation models and propose to leverage their underlying big learning principle to upgrade the EM. Specifically, we present the Big Learning EM (BigLearn-EM), an EM upgrade that simultaneously performs joint, marginal, and orthogonally transformed marginal matchings between data and model distributions. Through simulated experiments, we empirically show that the BigLearn-EM is capable of delivering the optimal with high probability; comparisons on benchmark clustering datasets further demonstrate its effectiveness and advantages over existing techniques. The code is available at https://github.com/YulaiCong/Big-Learning-Expectation-Maximization.

----

## [1302] Time-Aware Knowledge Representations of Dynamic Objects with Multidimensional Persistence

**Authors**: *Baris Coskunuzer, Ignacio Segovia-Dominguez, Yuzhou Chen, Yulia R. Gel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29051](https://doi.org/10.1609/aaai.v38i10.29051)

**Abstract**:

Learning time-evolving objects such as multivariate time series and dynamic networks requires the development of novel knowledge representation mechanisms and neural network architectures, which allow for capturing implicit time-dependent information contained in the data. Such information is typically not directly observed but plays a key role in the learning task performance. In turn, lack of time dimension in knowledge encoding mechanisms for time-dependent data leads to frequent model updates, poor learning performance, and, as a result, subpar decision-making.  Here we propose a new approach to a time-aware knowledge representation mechanism that notably focuses on implicit time-dependent topological information along multiple geometric dimensions. In particular, we propose a new approach, named Temporal MultiPersistence (TMP), which produces multidimensional topological fingerprints of the data by using the existing single parameter topological summaries. The main idea behind TMP is to merge the two newest directions in topological representation learning, that is, multi-persistence which simultaneously describes data shape evolution along multiple key parameters, and zigzag persistence to enable us to extract the most salient data shape information over time.    

We derive theoretical guarantees of TMP vectorizations and show its utility, in application to forecasting on benchmark traffic flow, Ethereum blockchain, and electrocardiogram datasets, demonstrating the competitive performance, especially, in scenarios of limited data records. In addition, our TMP method improves the computational efficiency of the state-of-the-art multipersistence summaries up to 59.5 times.

----

## [1303] BadRL: Sparse Targeted Backdoor Attack against Reinforcement Learning

**Authors**: *Jing Cui, Yufei Han, Yuzhe Ma, Jianbin Jiao, Junge Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29052](https://doi.org/10.1609/aaai.v38i10.29052)

**Abstract**:

Backdoor attacks in reinforcement learning (RL) have previously employed intense attack strategies to ensure attack success. However, these methods suffer from high attack costs and increased detectability. In this work, we propose a novel approach, BadRL, which focuses on conducting highly sparse backdoor poisoning efforts during training and testing while maintaining successful attacks. Our algorithm, BadRL, strategically chooses state observations with high attack values to inject triggers during training and testing, thereby reducing the chances of detection. In contrast to the previous methods that utilize sample-agnostic trigger patterns, BadRL dynamically generates distinct trigger patterns based on targeted state observations, thereby enhancing its effectiveness. Theoretical analysis shows that the targeted backdoor attack is always viable and remains stealthy under specific assumptions. Empirical results on various classic RL tasks illustrate that BadRL can substantially degrade the performance of a victim agent with minimal poisoning efforts (0.003% of total training steps) during training and infrequent attacks during testing. Code is available at: https://github.com/7777777cc/code.

----

## [1304] Deletion-Robust Submodular Maximization with Knapsack Constraints

**Authors**: *Shuang Cui, Kai Han, He Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29053](https://doi.org/10.1609/aaai.v38i10.29053)

**Abstract**:

Submodular maximization algorithms have found wide applications in various fields such as data summarization, recommendation systems, and active learning. In recent years, deletion-robust submodular maximization algorithms have garnered attention due to their significant implications in scenarios where some data points may be removed due to user preferences or privacy concerns, such as in recommendation systems and influence maximization. In this paper, we study the fundamental problem of submodular maximization with knapsack constraints and propose a robust streaming algorithm for it. To the best of our knowledge, our algorithm is the first to solve this problem for non-monotone submodular functions and can achieve an approximation ratio of 1/(6.82+2.63d)-ϵ under a near-optimal summary size of O(k+r), where k denotes the maximum cardinality of any feasible solution, d denotes the number of the knapsack constraints and r is the robustness parameter. For monotone submodular functions, our algorithm can achieve an approximation ratio of 1/(2+2d)-ϵ under a near-optimal summary size of O(k+r), significantly improving upon the best-known ratio of Ω((1/d-ϵ)^2). The empirical performance of our algorithm is extensively evaluated in several applications including influence maximization and recommendation systems, and the experimental results demonstrate the effectiveness of our algorithm.

----

## [1305] Continual Vision-Language Retrieval via Dynamic Knowledge Rectification

**Authors**: *Zhenyu Cui, Yuxin Peng, Xun Wang, Manyu Zhu, Jiahuan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29054](https://doi.org/10.1609/aaai.v38i10.29054)

**Abstract**:

The recent large-scale pre-trained models like CLIP have aroused great concern in vision-language tasks. However, when required to match image-text data collected in a streaming manner, namely Continual Vision-Language Retrieval (CVRL), their performances are still limited due to the catastrophic forgetting of the learned old knowledge. To handle this issue, advanced methods are proposed to distill the affinity knowledge between images and texts from the old model to the new one for anti-forgetting. Unfortunately, existing approaches neglect the impact of incorrect affinity, which prevents the balance between the anti-forgetting of old knowledge and the acquisition of new knowledge. Therefore, we propose a novel framework called Dynamic Knowledge Rectification (DKR) that simultaneously achieves incorrect knowledge filtering and rectification. Specifically, we first filter the incorrect affinity knowledge calculated by the old model on the new data. Then, a knowledge rectification method is designed to rectify the incorrect affinities while preserving the correct ones. In particular, for the new data that can only be correctly retrieved by the new model, we rectify them with the corresponding new affinity to protect them from negative transfer. Additionally, for those that can not be retrieved by either the old or the new model, we introduce paired ground-truth labels to promote the acquisition of both old and new knowledge. Extensive experiments on several benchmark datasets demonstrate the effectiveness of our DKR and its superiority against state-of-the-art methods.

----

## [1306] Inverse Weight-Balancing for Deep Long-Tailed Learning

**Authors**: *Wenqi Dang, Zhou Yang, Weisheng Dong, Xin Li, Guangming Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29055](https://doi.org/10.1609/aaai.v38i10.29055)

**Abstract**:

The performance of deep learning models often degrades rapidly when faced with imbalanced data characterized by a long-tailed distribution. Researchers have found that the fully connected layer trained by cross-entropy loss has large weight-norms for classes with many samples, but not for classes with few samples. How to address the data imbalance problem with both the encoder and the classifier seems an under-researched problem. In this paper, we propose an inverse weight-balancing (IWB) approach to guide model training and alleviate the data imbalance problem in two stages. In the first stage, an encoder and classifier (the fully connected layer) are trained using conventional cross-entropy loss. In the second stage, with a fixed encoder, the classifier is finetuned through an adaptive distribution for IWB in the decision space. Unlike existing inverse image frequency that implements a multiplicative margin adjustment transformation in the classification layer, our approach can be interpreted as an adaptive distribution alignment strategy using not only the class-wise number distribution but also the sample-wise difficulty distribution in both encoder and classifier. Experiments show that our method can greatly improve performance on imbalanced datasets such as CIFAR100-LT with different imbalance factors, ImageNet-LT, and iNaturelists2018.

----

## [1307] Learn the Force We Can: Enabling Sparse Motion Control in Multi-Object Video Generation

**Authors**: *Aram Davtyan, Paolo Favaro*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29056](https://doi.org/10.1609/aaai.v38i10.29056)

**Abstract**:

We propose a novel unsupervised method to autoregressively generate videos from a single frame and a sparse motion input. Our trained model can generate unseen realistic object-to-object interactions. Although our model has never been given the explicit segmentation and motion of each object in the scene during training, it is able to implicitly separate their dynamics and extents. Key components in our method are the randomized conditioning scheme, the encoding of the input motion control, and the randomized and sparse sampling to enable generalization to out of distribution but realistic correlations. Our model, which we call YODA, has therefore the ability to move objects without physically touching them. Through extensive qualitative and quantitative evaluations on several datasets, we show that YODA is on par with or better than state of the art video generation prior work in terms of both controllability and video quality.

----

## [1308] Iterative Regularization with k-support Norm: An Important Complement to Sparse Recovery

**Authors**: *William de Vazelhes, Bhaskar Mukhoty, Xiao-Tong Yuan, Bin Gu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29057](https://doi.org/10.1609/aaai.v38i10.29057)

**Abstract**:

Sparse recovery is ubiquitous in machine learning and signal processing. Due to the NP-hard nature of sparse recovery, existing methods are known to suffer either from restrictive (or even unknown) applicability conditions, or high computational cost. Recently, iterative regularization methods have emerged as a promising fast approach because they can achieve sparse recovery in one pass through early stopping, rather than the tedious grid-search used in the traditional methods.
However, most of those iterative methods are based on the l1 norm which requires restrictive applicability conditions and could fail in many cases. Therefore, achieving sparse recovery with iterative regularization methods under a wider range of conditions has yet to be further explored.
To address this issue, we propose a novel iterative regularization algorithm, IRKSN, based on the k-support norm regularizer rather than the l1 norm. We provide conditions for sparse recovery with IRKSN, and compare them with traditional conditions for recovery with l1 norm regularizers. Additionally, we give an early stopping bound on the model error of IRKSN with explicit constants, achieving the standard linear rate for sparse recovery. Finally, we illustrate the applicability of our algorithm on several experiments, including a support recovery experiment with a correlated design matrix.

----

## [1309] SEA-GWNN: Simple and Effective Adaptive Graph Wavelet Neural Network

**Authors**: *Swakshar Deb, Sejuti Rahman, Shafin Rahman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29058](https://doi.org/10.1609/aaai.v38i10.29058)

**Abstract**:

The utilization of wavelet-based techniques in graph neural networks (GNNs) has gained considerable attention, particularly in the context of node classification. Although existing wavelet-based approaches have shown promise, they are constrained by their reliance on pre-defined wavelet filters, rendering them incapable of effectively adapting to signals that reside on graphs based on tasks at hand. Recent research endeavors address this issue through the introduction of a wavelet lifting transform. However, this technique necessitates the use of bipartite graphs, causing a transformation of the original graph structure into a bipartite configuration. This alteration of graph topology results in the generation of undesirable wavelet filters, thereby undermining the effectiveness of the method. In response to these challenges, we propose a novel simple and effective adaptive graph wavelet neural network (SEA-GWNN) class that employs the lifting scheme on arbitrary graph structures while upholding the original graph topology by leveraging multi-hop computation trees. A noteworthy aspect of the approach is the focus on local substructures represented as acyclic trees, wherein the lifting strategy is applied in a localized manner. This locally defined lifting scheme effectively combines high-pass and low-pass frequency information to enhance node representations. Furthermore, to reduce computing costs, we propose to decouple the higher- order lifting operators and induce them from the lower-order structures.  Finally, we benchmark our model on several real- world datasets spanning four distinct categories, including citation networks, webpages, the film industry, and large-scale graphs and the experimental results showcase the efficacy of the proposed SEA-GWNN.

----

## [1310] Self-Interpretable Graph Learning with Sufficient and Necessary Explanations

**Authors**: *Jiale Deng, Yanyan Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29059](https://doi.org/10.1609/aaai.v38i10.29059)

**Abstract**:

Self-interpretable graph learning methods provide insights to unveil the black-box nature of GNNs by providing predictions with built-in explanations. However, current works suffer from performance degradation compared to GNNs trained without built-in explanations. We argue the main reason is that they fail to generate explanations satisfying both sufficiency and necessity, and the biased explanations further hurt GNNs' performance. In this work, we propose a novel framework for generating SUfficient aNd NecessarY explanations (SUNNY-GNN for short) that benefit GNNs' predictions. The key idea is to conduct augmentations by structurally perturbing given explanations and employ a contrastive loss to guide the learning of explanations toward sufficiency and necessity directions. SUNNY-GNN introduces two coefficients to generate hard and reliable contrastive samples. We further extend SUNNY-GNN to heterogeneous graphs. Empirical results on various GNNs and real-world graphs show that SUNNY-GNN yields accurate predictions and faithful explanations, outperforming the state-of-the-art methods by improving 3.5% prediction accuracy and 13.1% explainability fidelity on average. Our code and data are available at https://github.com/SJTU-Quant/SUNNY-GNN.

----

## [1311] Semi-supervised TEE Segmentation via Interacting with SAM Equipped with Noise-Resilient Prompting

**Authors**: *Sen Deng, Yidan Feng, Haoneng Lin, Yiting Fan, Alex Pui-Wai Lee, Xiaowei Hu, Jing Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29060](https://doi.org/10.1609/aaai.v38i10.29060)

**Abstract**:

Semi-supervised learning (SSL) is a powerful tool to address the challenge of insufficient annotated data in medical segmentation problems. However, existing semi-supervised methods mainly rely on internal knowledge for pseudo labeling, which is biased due to the distribution mismatch between the highly imbalanced labeled and unlabeled data. Segmenting left atrial appendage (LAA) from transesophageal echocardiogram (TEE) images is a typical medical image segmentation task featured by scarcity of professional annotations and diverse data distributions, for which existing SSL models cannot achieve satisfactory performance. In this paper, we propose a novel strategy to mitigate the inherent challenge of distribution mismatch in SSL by, for the first time, incorporating a large foundation model (i.e. SAM in our implementation) into an SSL model to improve the quality of pseudo labels. We further propose a new self-reconstruction mechanism to generate both noise-resilient prompts to demonically improve SAM’s generalization capability over TEE images and self-perturbations to stabilize the training process and reduce the impact of noisy labels. We conduct extensive experiments on an in-house TEE dataset; experimental results demonstrate that our method achieves better performance than state-of-the-art SSL models.

----

## [1312] Peer Learning: Learning Complex Policies in Groups from Scratch via Action Recommendations

**Authors**: *Cedric Derstroff, Mattia Cerrato, Jannis Brugger, Jan Peters, Stefan Kramer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29061](https://doi.org/10.1609/aaai.v38i10.29061)

**Abstract**:

Peer learning is a novel high-level reinforcement learning framework for agents learning in groups. While standard reinforcement learning trains an individual agent in trial-and-error fashion, all on its own, peer learning addresses a related setting in which a group of agents, i.e., peers, learns to master a task simultaneously together from scratch. Peers are allowed to communicate only about their own states and actions recommended by others: "What would you do in my situation?". Our motivation is to study the learning behavior of these agents.
We formalize the teacher selection process in the action advice setting as a multi-armed bandit problem and therefore highlight the need for exploration. Eventually, we analyze the learning behavior of the peers and observe their ability to rank the agents' performance within the study group and understand which agents give reliable advice. Further, we compare peer learning with single agent learning and a state-of-the-art action advice baseline. We show that peer learning is able to outperform single-agent learning and the baseline in several challenging discrete and continuous OpenAI Gym domains. Doing so, we also show that within such a framework complex policies from action recommendations beyond discrete action spaces can evolve.

----

## [1313] Conformal Autoregressive Generation: Beam Search with Coverage Guarantees

**Authors**: *Nicolas Deutschmann, Marvin Alberts, María Rodríguez Martínez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29062](https://doi.org/10.1609/aaai.v38i10.29062)

**Abstract**:

We introduce two new extensions to the beam search algorithm based on conformal predictions (CP) to produce sets of sequences with theoretical coverage guarantees. 
The first method is very simple and proposes dynamically-sized subsets of beam search results but, unlike typical CP proceedures, has an upper bound on the achievable guarantee depending on a post-hoc calibration measure. 
Our second algorithm introduces the conformal set prediction procedure as part of the decoding process, producing a variable beam width which adapts to the current uncertainty. 
While more complex, this procedure can achieve coverage guarantees selected a priori. We provide marginal coverage bounds as well as calibration-conditional guarantees for each method, and evaluate them empirically on a selection of tasks drawing from natural language processing and chemistry.

----

## [1314] Exploiting Label Skews in Federated Learning with Model Concatenation

**Authors**: *Yiqun Diao, Qinbin Li, Bingsheng He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29063](https://doi.org/10.1609/aaai.v38i10.29063)

**Abstract**:

Federated Learning (FL) has emerged as a promising solution to perform deep learning on different data owners without exchanging raw data. However, non-IID data has been a key challenge in FL, which could significantly degrade the accuracy of the final model. Among different non-IID types, label skews have been challenging and common in image classification and other tasks. Instead of averaging the local models in most previous studies, we propose FedConcat, a simple and effective approach that concatenates these local models as the base of the global model to effectively aggregate the local knowledge. To reduce the size of the global model, we adopt the clustering technique to group the clients by their label distributions and collaboratively train a model inside each cluster. We theoretically analyze the advantage of concatenation over averaging by analyzing the information bottleneck of deep neural networks. Experimental results demonstrate that FedConcat achieves significantly higher accuracy than previous state-of-the-art FL methods in various heterogeneous label skew distribution settings and meanwhile has lower communication costs. Our code is publicly available at https://github.com/sjtudyq/FedConcat.

----

## [1315] Multi-View Randomized Kernel Classification via Nonconvex Optimization

**Authors**: *Xiaojian Ding, Fan Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29064](https://doi.org/10.1609/aaai.v38i10.29064)

**Abstract**:

Multi kernel learning (MKL) is a representative supervised multi-view learning method widely applied in multi-modal and multi-view applications.
MKL aims to classify data by integrating complementary information from predefined kernels.
Although existing MKL methods achieve promising performance, they fail to consider the tradeoff between diversity and classification accuracy of kernels, preventing further improvement of classification performance.
In this paper, we tackle this problem by generating a number of high-quality base learning kernels and selecting a kernel subset with maximum pairwise diversity and minimum generalization errors.
We first formulate this idea as a nonconvex quadratic integer programming problem.
Then we transform this nonconvex problem into a convex optimization problem and prove it is equivalent to a semidefinite relaxation problem, which a semidefinite-based branch-and-bound algorithm can quickly solve.
Experimental results on the real-world datasets demonstrate the superiority of the proposed method.
The results also show that our method works for the support vector machine (SVM) classifier and other state-of-the-art kernel classifiers.

----

## [1316] Turning Waste into Wealth: Leveraging Low-Quality Samples for Enhancing Continuous Conditional Generative Adversarial Networks

**Authors**: *Xin Ding, Yongwei Wang, Zuheng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29065](https://doi.org/10.1609/aaai.v38i10.29065)

**Abstract**:

Continuous Conditional Generative Adversarial Networks (CcGANs) enable generative modeling conditional on continuous scalar variables (termed regression labels). However, they can produce subpar fake images due to limited training data. Although Negative Data Augmentation (NDA) effectively enhances unconditional and class-conditional GANs by introducing anomalies into real training images, guiding the GANs away from low-quality outputs, its impact on CcGANs is limited, as it fails to replicate negative samples that may occur during the CcGAN sampling. We present a novel NDA approach called Dual-NDA specifically tailored for CcGANs to address this problem. Dual-NDA employs two types of negative samples: visually unrealistic images generated from a pre-trained CcGAN and label-inconsistent images created by manipulating real images' labels. Leveraging these negative samples, we introduce a novel discriminator objective alongside a modified CcGAN training algorithm. Empirical analysis on UTKFace and Steering Angle reveals that Dual-NDA consistently enhances the visual fidelity and label consistency of fake images generated by CcGANs, exhibiting a substantial performance gain over the vanilla NDA. Moreover, by applying Dual-NDA, CcGANs demonstrate a remarkable advancement beyond the capabilities of state-of-the-art conditional GANs and diffusion models, establishing a new pinnacle of performance. Our codes can be found at https://github.com/UBCDingXin/Dual-NDA.

----

## [1317] Shrinking Your TimeStep: Towards Low-Latency Neuromorphic Object Recognition with Spiking Neural Networks

**Authors**: *Yongqi Ding, Lin Zuo, Mengmeng Jing, Pei He, Yongjun Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29066](https://doi.org/10.1609/aaai.v38i10.29066)

**Abstract**:

Neuromorphic object recognition with spiking neural networks (SNNs) is the cornerstone of low-power neuromorphic computing. However, existing SNNs suffer from significant latency, utilizing 10 to 40 timesteps or more, to recognize neuromorphic objects. At low latencies, the performance of existing SNNs is drastically degraded. In this work, we propose the Shrinking SNN (SSNN) to achieve low-latency neuromorphic object recognition without reducing performance. Concretely, we alleviate the temporal redundancy in SNNs by dividing SNNs into multiple stages with progressively shrinking timesteps, which significantly reduces the inference latency. During timestep shrinkage, the temporal transformer smoothly transforms the temporal scale and preserves the information maximally. Moreover, we add multiple early classifiers to the SNN during training to mitigate the mismatch between the surrogate gradient and the true gradient, as well as the gradient vanishing/exploding, thus eliminating the performance degradation at low latency. Extensive experiments on neuromorphic datasets, CIFAR10-DVS, N-Caltech101, and DVS-Gesture have revealed that SSNN is able to improve the baseline accuracy by 6.55% ~ 21.41%. With only 5 average timesteps and without any data augmentation, SSNN is able to achieve an accuracy of 73.63% on CIFAR10-DVS. This work presents a heterogeneous temporal scale SNN and provides valuable insights into the development of high-performance, low-latency SNNs.

----

## [1318] DGA-GNN: Dynamic Grouping Aggregation GNN for Fraud Detection

**Authors**: *Mingjiang Duan, Tongya Zheng, Yang Gao, Gang Wang, Zunlei Feng, Xinyu Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29067](https://doi.org/10.1609/aaai.v38i10.29067)

**Abstract**:

Fraud detection has increasingly become a prominent research field due to the dramatically increased incidents of fraud. The complex connections involving thousands, or even millions of nodes, present challenges for fraud detection tasks. Many researchers have developed various graph-based methods to detect fraud from these intricate graphs. However, those methods neglect two distinct characteristics of the fraud graph: the non-additivity of certain attributes and the distinguishability of grouped messages from neighbor nodes.
This paper introduces the Dynamic Grouping Aggregation Graph Neural Network (DGA-GNN) for fraud detection, which addresses these two characteristics by dynamically grouping attribute value ranges and neighbor nodes. In DGA-GNN, we initially propose the decision tree binning encoding to transform non-additive node attributes into bin vectors. This approach aligns well with the GNN’s aggregation operation and avoids nonsensical feature generation. Furthermore, we devise a feedback dynamic grouping strategy to classify graph nodes into two distinct groups and then employ a hierarchical aggregation. This method extracts more discriminative features for fraud detection tasks. Extensive experiments on five datasets suggest that our proposed method achieves a 3% ~ 16% improvement over existing SOTA methods. Code is available at https://github.com/AtwoodDuan/DGA-GNN.

----

## [1319] Roll with the Punches: Expansion and Shrinkage of Soft Label Selection for Semi-supervised Fine-Grained Learning

**Authors**: *Yue Duan, Zhen Zhao, Lei Qi, Luping Zhou, Lei Wang, Yinghuan Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29068](https://doi.org/10.1609/aaai.v38i10.29068)

**Abstract**:

While semi-supervised learning (SSL) has yielded promising results, the more realistic SSL scenario remains to be explored, in which the unlabeled data exhibits extremely high recognition difficulty, e.g., fine-grained visual classification in the context of SSL (SS-FGVC). The increased recognition difficulty on fine-grained unlabeled data spells disaster for pseudo-labeling accuracy, resulting in poor performance of the SSL model. To tackle this challenge, we propose Soft Label Selection with Confidence-Aware Clustering based on Class Transition Tracking (SoC) by reconstructing the pseudo-label selection process by jointly optimizing Expansion Objective and Shrinkage Objective, which is based on a soft label manner. Respectively, the former objective encourages soft labels to absorb more candidate classes to ensure the attendance of ground-truth class, while the latter  encourages soft labels to reject more noisy classes, which is theoretically proved to be equivalent to entropy minimization. In comparisons with various state-of-the-art methods, our approach demonstrates its superior performance in SS-FGVC. Checkpoints and source code are available at https://github.com/NJUyued/SoC4SS-FGVC.

----

## [1320] Provably Powerful Graph Neural Networks for Directed Multigraphs

**Authors**: *Béni Egressy, Luc von Niederhäusern, Jovan Blanusa, Erik Altman, Roger Wattenhofer, Kubilay Atasu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29069](https://doi.org/10.1609/aaai.v38i10.29069)

**Abstract**:

This paper analyses a set of simple adaptations that transform standard message-passing Graph Neural Networks (GNN) into provably powerful directed multigraph neural networks. The adaptations include multigraph port numbering, ego IDs, and reverse message passing. We prove that the combination of these theoretically enables the detection of any directed subgraph pattern. To validate the effectiveness of our proposed adaptations in practice, we conduct experiments on synthetic subgraph detection tasks, which demonstrate outstanding performance with almost perfect results. 
Moreover, we apply our proposed adaptations to two financial crime analysis tasks. We observe dramatic improvements in detecting money laundering transactions, improving the minority-class F1 score of a standard message-passing GNN by up to 30%, and closely matching or outperforming tree-based and GNN baselines. Similarly impressive results are observed on a real-world phishing detection dataset, boosting three standard GNNs’ F1 scores by around 15% and outperforming all baselines. An extended version with appendices can be found on arXiv: https://arxiv.org/abs/2306.11586.

----

## [1321] Causal Adversarial Perturbations for Individual Fairness and Robustness in Heterogeneous Data Spaces

**Authors**: *Ahmad-Reza Ehyaei, Kiarash Mohammadi, Amir-Hossein Karimi, Samira Samadi, Golnoosh Farnadi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29070](https://doi.org/10.1609/aaai.v38i10.29070)

**Abstract**:

As responsible AI gains importance in machine learning algorithms, properties like fairness, adversarial robustness, and causality have received considerable attention in recent years. However, despite their individual significance, there remains a critical gap in simultaneously exploring and integrating these properties. In this paper, we propose a novel approach that examines the relationship between individual fairness, adversarial robustness, and structural causal models (SCMs) in heterogeneous data spaces, particularly when dealing with discrete sensitive attributes. We use SCMs and sensitive attributes to create a fair metric and apply it to measure semantic similarity among individuals. By introducing a novel causal adversarial perturbation (CAP) and applying adversarial training, we create a new regularizer that combines individual fairness, causality, and robustness in the classifier. Our method is evaluated on both real-world and synthetic datasets, demonstrating its effectiveness in achieving an accurate classifier that simultaneously exhibits fairness, adversarial robustness, and causal awareness.

----

## [1322] Double-Descent Curves in Neural Networks: A New Perspective Using Gaussian Processes

**Authors**: *Ouns El Harzli, Bernardo Cuenca Grau, Guillermo Valle Pérez, Ard A. Louis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i10.29071](https://doi.org/10.1609/aaai.v38i10.29071)

**Abstract**:

Double-descent curves in neural networks describe the phenomenon that the generalisation error initially descends with increasing parameters, then grows after reaching an optimal number of parameters which is less than the number of data points, but then descends again in the overparameterized regime. In this paper, we use techniques from random matrix theory to characterize the spectral distribution of the empirical feature covariance matrix as a width-dependent perturbation of the spectrum of the neural network Gaussian process (NNGP) kernel, thus establishing a novel connection between the NNGP literature and the random matrix theory literature in the context of neural networks. Our analytical expressions allow us to explore the generalisation behavior of the corresponding kernel and GP regression. Furthermore, they offer a new interpretation of  double-descent in terms of the discrepancy between the width-dependent empirical kernel and the width-independent NNGP kernel.

----

## [1323] A Score-Based Deterministic Diffusion Algorithm with Smooth Scores for General Distributions

**Authors**: *Karthik Elamvazhuthi, Xuechen Zhang, Matthew Jacobs, Samet Oymak, Fabio Pasqualetti*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29072](https://doi.org/10.1609/aaai.v38i11.29072)

**Abstract**:

Score matching based diffusion has shown to achieve the state of art results in generation modeling. In the original score matching based diffusion algorithm, the forward equation is a differential equation for which the probability density equation evolves according to a linear partial differential equation, the Fokker-Planck equation. A drawback of this approach is that one needs the data distribution to have a Lipschitz logarithmic gradient. This excludes a large class of data distributions that have a compact support. We present a deterministic diffusion process for which the vector fields are always Lipschitz and hence the score does not explode for probability measures with compact support. This deterministic diffusion process can be seen as a regularization of the porous media equation equation, which enables one to guarantee long term convergence of the forward process to the noise distribution. Though the porous media equation is itself not always guaranteed to have a Lipschitz vector field, it can be used to understand the closeness of the output of the algorithm to the data distribution as a function of the the time horizon and score matching error. This analysis enables us to show that the algorithm has better dependence on the score matching error than approaches based on stochastic diffusions. Using numerical experiments we verify our theoretical results on example one and two dimensional data distributions which are compactly supported. Additionally, we validate the approach on a modified MNIST data set for which the distribution is concentrated on a compact set. In each of the experiments, the approach using deterministic diffusion performs better that the diffusion algorithm with stochastic forward process, when considering the FID scores of the generated samples.

----

## [1324] Feature Transportation Improves Graph Neural Networks

**Authors**: *Moshe Eliasof, Eldad Haber, Eran Treister*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29073](https://doi.org/10.1609/aaai.v38i11.29073)

**Abstract**:

Graph neural networks (GNNs) have shown remarkable success in learning representations for graph-structured data. However, GNNs still face challenges in modeling complex phenomena that involve feature transportation. In this paper, we propose a novel GNN architecture inspired by Advection-Diffusion-Reaction systems, called ADR-GNN.
Advection models feature transportation, while diffusion captures the local smoothing of features, and reaction represents the non-linear transformation between feature channels. We provide an analysis of the qualitative behavior of ADR-GNN, that shows the benefit of combining advection, diffusion, and reaction.
To demonstrate its efficacy, we evaluate ADR-GNN on real-world node classification and spatio-temporal datasets, and show that it improves or offers competitive performance compared to state-of-the-art networks.

----

## [1325] Exact Inference for Continuous-Time Gaussian Process Dynamics

**Authors**: *Katharina Ensinger, Nicholas Tagliapietra, Sebastian Ziesche, Sebastian Trimpe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29074](https://doi.org/10.1609/aaai.v38i11.29074)

**Abstract**:

Many physical systems can be described as a continuous-time dynamical system. In practice, the true system is often unknown and has to be learned from measurement data. Since data is typically collected in discrete time, e.g. by sensors, most methods in Gaussian process (GP) dynamics model learning are trained on one-step ahead predictions. While this scheme is mathematically tempting, it can become problematic in several scenarios, e.g. if measurements are provided at irregularly-sampled time steps or physical system properties have to be conserved. Thus, we aim for a GP model of the true continuous-time dynamics. We tackle this task by leveraging higher-order numerical integrators. These integrators provide the necessary tools to discretize dynamical systems with arbitrary accuracy. However, most higher-order integrators require dynamics evaluations at intermediate time steps, making exact GP inference intractable. In previous work, this problem is often addressed by approximate inference techniques. However, exact GP inference is preferable in many scenarios, e.g. due to its mathematical guarantees. In order to enable direct inference, we propose to leverage multistep and Taylor integrators. We demonstrate how exact inference schemes can be derived for these types of integrators. Further, we derive tailored sampling schemes that allow one to draw consistent dynamics functions from the posterior. The learned model can thus be integrated with arbitrary integrators, just like a standard dynamical system. We show empirically and theoretically that our approach yields an accurate representation of the continuous-time system.

----

## [1326] Learning Hybrid Dynamics Models with Simulator-Informed Latent States

**Authors**: *Katharina Ensinger, Sebastian Ziesche, Sebastian Trimpe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29075](https://doi.org/10.1609/aaai.v38i11.29075)

**Abstract**:

Dynamics model learning deals with the task of inferring unknown dynamics from measurement data and predicting the future behavior of the system. A typical approach to address this problem is to train recurrent models. However, predictions with these models are often not physically meaningful. Further, they suffer from deteriorated behavior over time due to accumulating errors. Often, simulators building on first principles are available being physically meaningful by design. However, modeling simplifications typically cause inaccuracies in these models. Consequently, hybrid modeling is an emerging trend that aims to combine the best of both worlds. In this paper, we propose a new approach to hybrid modeling, where we inform the latent states of a learned model via a black-box simulator. This allows to control the predictions via the simulator preventing them from accumulating errors. This is especially challenging since, in contrast to previous approaches, access to the simulator's latent states is not available. We tackle the task by leveraging observers, a well-known concept from control theory, inferring unknown latent states from observations and dynamics over time. In our learning-based setting, we jointly learn the dynamics and an observer that infers the latent states via the simulator. Thus, the simulator constantly corrects the latent states, compensating for modeling mismatch caused by learning. To maintain flexibility, we train an RNN-based residuum for the latent states that cannot be informed by the simulator.

----

## [1327] PAC-Bayes Generalisation Bounds for Dynamical Systems including Stable RNNs

**Authors**: *Deividas Eringis, John Leth, Zheng-Hua Tan, Rafael Wisniewski, Mihály Petreczky*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29076](https://doi.org/10.1609/aaai.v38i11.29076)

**Abstract**:

In this paper, we derive a PAC-Bayes bound on the generalisation gap, in a supervised time-series setting for a special class of discrete-time non-linear dynamical systems. This class includes stable recurrent neural networks (RNN), and the motivation for this work was its application to RNNs. In order to achieve the results, we impose some stability constraints, on the allowed models. 
Here, stability is understood in the sense of dynamical systems. For RNNs, these stability conditions can be expressed in terms of conditions on the weights. 
We assume the processes involved are essentially bounded and the loss functions are Lipschitz. The proposed bound on the generalisation gap depends on the mixing coefficient of the data distribution, and the essential supremum of the data. Furthermore, the bound converges to zero as the dataset size increases.
In this paper, we 1) formalize the learning problem, 2) derive a PAC-Bayesian error bound for such systems, 3) discuss various consequences of this error bound, and 4) show an illustrative example, with discussions on computing the proposed bound. Unlike other available bounds  the derived bound holds for non i.i.d. data (time-series) and it does not grow with the number of steps of the RNN.

----

## [1328] Non-parametric Representation Learning with Kernels

**Authors**: *Pascal Mattia Esser, Maximilian Fleissner, Debarghya Ghoshdastidar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29077](https://doi.org/10.1609/aaai.v38i11.29077)

**Abstract**:

Unsupervised and self-supervised representation learning has become popular in recent years for learning useful features from unlabelled data. Representation learning has been mostly developed in the neural network literature, and other models for representation learning are surprisingly unexplored. In this work, we introduce and analyze several kernel-based representation learning approaches: Firstly, we define two kernel Self-Supervised Learning (SSL) models using contrastive loss functions and secondly, a Kernel Autoencoder (AE) model based on the idea of embedding and reconstructing data. We argue that the classical representer theorems for supervised kernel machines are not always applicable for (self-supervised) representation learning, and present new representer theorems, which  show that the representations learned by our kernel models can be expressed in terms of kernel matrices. We further derive generalisation error bounds for representation learning with kernel SSL and AE, and empirically evaluate the performance of these methods in both small data regimes as well as in comparison with neural network based models.

----

## [1329] Neural Gaussian Similarity Modeling for Differential Graph Structure Learning

**Authors**: *Xiaolong Fan, Maoguo Gong, Yue Wu, Zedong Tang, Jieyi Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29078](https://doi.org/10.1609/aaai.v38i11.29078)

**Abstract**:

Graph Structure Learning (GSL) has demonstrated considerable potential in the analysis of graph-unknown non-Euclidean data across a wide range of domains. However, constructing an end-to-end graph structure learning model poses a challenge due to the impediment of gradient flow caused by the nearest neighbor sampling strategy. In this paper, we construct a differential graph structure learning model by replacing the non-differentiable nearest neighbor sampling with a differentiable sampling using the reparameterization trick. Under this framework, we argue that the act of sampling nearest neighbors may not invariably be essential, particularly in instances where node features exhibit a significant degree of similarity. To alleviate this issue, the bell-shaped Gaussian Similarity (GauSim) modeling is proposed to sample non-nearest neighbors. To adaptively model the similarity, we further propose Neural Gaussian Similarity (NeuralGauSim) with learnable parameters featuring flexible sampling behaviors. In addition, we develop a scalable method by transferring the large-scale graph to the transition graph to significantly reduce the complexity. Experimental results demonstrate the effectiveness of the proposed methods.

----

## [1330] Dynamic Sub-graph Distillation for Robust Semi-supervised Continual Learning

**Authors**: *Yan Fan, Yu Wang, Pengfei Zhu, Qinghua Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29079](https://doi.org/10.1609/aaai.v38i11.29079)

**Abstract**:

Continual learning (CL) has shown promising results and comparable performance to learning at once in a fully supervised manner. However, CL strategies typically require a large number of labeled samples, making their real-life deployment challenging. In this work, we focus on semi-supervised continual learning (SSCL), where the model progressively learns from partially labeled data with unknown categories. We provide a comprehensive analysis of SSCL and demonstrate that unreliable distributions of unlabeled data lead to unstable training and refinement of the progressing stages. This problem severely impacts the performance of SSCL. To address the limitations, we propose a novel approach called Dynamic Sub-Graph Distillation (DSGD) for semi-supervised continual learning, which leverages both semantic and structural information to achieve more stable knowledge distillation on unlabeled data and exhibit robustness against distribution bias. Firstly, we formalize a general model of structural distillation and design a dynamic graph construction for the continual learning progress. Next, we define a structure distillation vector and design a dynamic sub-graph distillation algorithm, which enables end-to-end training and adaptability to scale up tasks. The entire proposed method is adaptable to various CL methods and supervision settings. Finally, experiments conducted on three datasets CIFAR10, CIFAR100, and ImageNet-100, with varying supervision ratios, demonstrate the effectiveness of our proposed approach in mitigating the catastrophic forgetting problem in semi-supervised continual learning scenarios. Our code is available: https://github.com/fanyan0411/DSGD.

----

## [1331] Selective Focus: Investigating Semantics Sensitivity in Post-training Quantization for Lane Detection

**Authors**: *Yunqian Fan, Xiuying Wei, Ruihao Gong, Yuqing Ma, Xiangguo Zhang, Qi Zhang, Xianglong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29080](https://doi.org/10.1609/aaai.v38i11.29080)

**Abstract**:

Lane detection (LD) plays a crucial role in enhancing the L2+ capabilities of autonomous driving, capturing widespread attention. The Post-Processing Quantization (PTQ) could facilitate the practical application of LD models, enabling fast speeds and limited memories without labeled data. However, prior PTQ methods do not consider the complex LD outputs that contain physical semantics, such as offsets, locations, etc., and thus cannot be directly applied to LD models. In this paper, we pioneeringly investigate semantic sensitivity to post-processing for lane detection with a novel Lane Distortion Score. Moreover, we identify two main factors impacting the LD performance after quantization, namely intra-head sensitivity and inter-head sensitivity, where a small quantization error in specific semantics can cause significant lane distortion. Thus, we propose a Selective Focus framework deployed with Semantic Guided Focus and Sensitivity Aware Selection modules, to incorporate post-processing information into PTQ reconstruction. Based on the observed intra-head sensitivity, Semantic Guided Focus is introduced to prioritize foreground-related semantics using a practical proxy. For inter-head sensitivity, we present Sensitivity Aware Selection, efficiently recognizing influential prediction heads and refining the optimization objectives at runtime. Extensive experiments have been done on a wide variety of models including keypoint-, anchor-, curve-, and segmentation-based ones. Our method produces quantized models in minutes on a single GPU and can achieve 6.4\% F1 Score improvement on the CULane dataset. Code and supplementary statement can be found at https://github.com/PannenetsF/SelectiveFocus.

----

## [1332] Backdoor Adjustment via Group Adaptation for Debiased Coupon Recommendations

**Authors**: *Junpeng Fang, Gongduo Zhang, Qing Cui, Caizhi Tang, Lihong Gu, Longfei Li, Jinjie Gu, Jun Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29081](https://doi.org/10.1609/aaai.v38i11.29081)

**Abstract**:

Accurate prediction of coupon usage is crucial for promoting user consumption through targeted coupon recommendations. However, in real-world coupon recommendations, the coupon allocation process is not solely determined by the model trained with the history interaction data but is also interfered with by marketing tactics desired to fulfill specific commercial goals.This interference creates an imbalance in the interactions, which causes the data to deviate from the user's natural preferences. We refer to this deviation as the matching bias. Such biased interaction data affects the efficacy of the model, and thus it is necessary to employ debiasing techniques to prevent any negative impact.
We investigate the mitigation of matching bias in coupon recommendations from a causal-effect perspective. By treating the attributes of users and coupons associated with marketing tactics as confounders, we find the confounders open the backdoor path between user-coupon matching and the conversion, which introduces spurious correlation. To remove the bad effect, we propose a novel training paradigm named Backdoor Adjustment via Group Adaptation (BAGA) for debiased coupon recommendations, which performs intervened training and inference, i.e., separately modeling each user-coupon group pair. However, modeling all possible group pairs greatly increases the computational complexity and cost. To address the efficiency challenge, we further present a simple but effective dual-tower multi-task framework and leverage the Customized Gate Control (CGC) model architecture, which separately models each user and coupon group with a separate expert module. We instantiate BAGA on five representative models: FM, DNN, NCF, MASKNET, and DEEPFM, and conduct comprehensive offline and online experiments to demonstrate the efficacy of our proposed paradigm.

----

## [1333] Improving GNN Calibration with Discriminative Ability: Insights and Strategies

**Authors**: *Yujie Fang, Xin Li, Qianyu Chen, Mingzhong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29082](https://doi.org/10.1609/aaai.v38i11.29082)

**Abstract**:

The widespread adoption of Graph Neural Networks (GNNs) has led to an increasing focus on their reliability. To address the issue of underconfidence in GNNs, various calibration methods have been developed to gain notable reductions in calibration error. However, we observe that existing approaches generally fail to enhance consistently, and in some cases even deteriorate, GNNs' ability to discriminate between correct and incorrect predictions. In this study, we advocate the significance of discriminative ability and the inclusion of relevant evaluation metrics. Our rationale is twofold: 1) Overlooking discriminative ability can inadvertently compromise the overall quality of the model; 2) Leveraging discriminative ability can significantly inform and improve calibration outcomes. Therefore, we thoroughly explore the reasons why existing calibration methods have ineffectiveness and even degradation regarding the discriminative ability of GNNs. Building upon these insights, we conduct GNN calibration experiments across multiple datasets using a straightforward example model, denoted as DC(GNN). Its excellent performance confirms the potential of integrating discriminative ability as a key consideration in the calibration of GNNs, thereby establishing a pathway toward more effective and reliable network calibration.

----

## [1334] SUF: Stabilized Unconstrained Fine-Tuning for Offline-to-Online Reinforcement Learning

**Authors**: *Jiaheng Feng, Mingxiao Feng, Haolin Song, Wengang Zhou, Houqiang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29083](https://doi.org/10.1609/aaai.v38i11.29083)

**Abstract**:

Offline-to-online reinforcement learning (RL) provides a promising solution to improving suboptimal offline pre-trained policies through online fine-tuning. However, one efficient method, unconstrained fine-tuning, often suffers from severe policy collapse due to excessive distribution shift. To ensure stability, existing methods retain offline constraints and employ additional techniques during fine-tuning, which hurts efficiency. In this work, we introduce a novel perspective: eliminating the policy collapse without imposing constraints. We observe that such policy collapse arises from the mismatch between unconstrained fine-tuning and the conventional RL training framework. To this end, we propose Stabilized Unconstrained Fine-tuning (SUF), a streamlined framework that benefits from the efficiency of unconstrained fine-tuning while ensuring stability by modifying the Update-To-Data ratio. With just a few lines of code adjustments, SUF demonstrates remarkable adaptability to diverse backbones and superior performance over state-of-the-art baselines.

----

## [1335] BaCon: Boosting Imbalanced Semi-supervised Learning via Balanced Feature-Level Contrastive Learning

**Authors**: *Qianhan Feng, Lujing Xie, Shijie Fang, Tong Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29084](https://doi.org/10.1609/aaai.v38i11.29084)

**Abstract**:

Semi-supervised Learning (SSL) reduces the need for extensive annotations in deep learning, but the more realistic challenge of imbalanced data distribution in SSL remains largely unexplored. In Class Imbalanced Semi-supervised Learning (CISSL), the bias introduced by unreliable pseudo-labels can be exacerbated by imbalanced data distributions. Most existing methods address this issue at instance-level through reweighting or resampling, but the performance is heavily limited by their reliance on biased backbone representation. Some other methods do perform feature-level adjustments like feature blending but might introduce unfavorable noise. In this paper, we discuss the bonus of a more balanced feature distribution for the CISSL problem, and further propose a Balanced Feature-Level Contrastive Learning method (BaCon). Our method directly regularizes the distribution of instances' representations in a well-designed contrastive manner. Specifically, class-wise feature centers are computed as the positive anchors, while negative anchors are selected by a straightforward yet effective mechanism. A distribution-related temperature adjustment is leveraged to control the class-wise contrastive degrees dynamically. Our method demonstrates its effectiveness through comprehensive experiments on the CIFAR10-LT, CIFAR100-LT, STL10-LT, and SVHN-LT datasets across various settings. For example, BaCon surpasses instance-level method FixMatch-based ABC on CIFAR10-LT with a 1.21% accuracy improvement, and outperforms state-of-the-art feature-level method CoSSL on CIFAR100-LT with a 0.63% accuracy improvement. When encountering more extreme imbalance degree, BaCon also shows better robustness than other methods.

----

## [1336] Latent Diffusion Transformer for Probabilistic Time Series Forecasting

**Authors**: *Shibo Feng, Chunyan Miao, Zhong Zhang, Peilin Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29085](https://doi.org/10.1609/aaai.v38i11.29085)

**Abstract**:

The probability prediction of multivariate time series is a notoriously challenging but practical task. This research proposes to condense high-dimensional multivariate time series forecasting into a problem of latent space time series generation, to improve the expressiveness of each timestamp and make forecasting more manageable. To solve the problem that the existing work is hard to extend to high-dimensional multivariate time series, we present a latent multivariate time series diffusion framework called Latent Diffusion Transformer (LDT), which consists of a symmetric statistics-aware autoencoder and a diffusion-based conditional generator, to implement this idea. Through careful design, the time series autoencoder can compress multivariate timestamp patterns into a concise latent representation by considering dynamic statistics. Then, the diffusion-based conditional generator is able to efficiently generate realistic multivariate timestamp values on a continuous latent space under a novel self-conditioning guidance which is modeled in a non-autoregressive way. Extensive experiments demonstrate that our model achieves state-of-the-art performance on many popular high-dimensional multivariate time series datasets.

----

## [1337] Partial Multi-View Clustering via Self-Supervised Network

**Authors**: *Wei Feng, Guoshuai Sheng, Qianqian Wang, Quanxue Gao, Zhiqiang Tao, Bo Dong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29086](https://doi.org/10.1609/aaai.v38i11.29086)

**Abstract**:

Partial multi-view clustering is a challenging and practical research problem for data analysis in real-world applications, due to the potential data missing issue in different views. However, most existing methods have not fully explored the correlation information among various incomplete views. In addition, these existing clustering methods always ignore discovering discriminative features inside the data itself in this unsupervised task. To tackle these challenges, we propose Partial Multi-View Clustering via Self-Supervised \textbf{N}etwork (PVC-SSN) in this paper. 
Specifically, we employ contrastive learning to obtain a more discriminative and consistent subspace representation, which is guided by a self-supervised module. Self-supervised learning can exploit effective cluster information through the data itself to guide the learning process of clustering tasks. Thus, it can pull together embedding features from the same cluster and push apart these from different clusters. Extensive experiments on several benchmark datasets show that the proposed PVC-SCN method outperforms several state-of-the-art clustering methods.

----

## [1338] Harnessing Manycore Processors with Distributed Memory for Accelerated Training of Sparse and Recurrent Models

**Authors**: *Jan Finkbeiner, Thomas Gmeinder, Mark Pupilli, Alexander Titterton, Emre Neftci*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29087](https://doi.org/10.1609/aaai.v38i11.29087)

**Abstract**:

Current AI training infrastructure is dominated by single instruction multiple data (SIMD) and systolic array architectures, such as Graphics Processing Units (GPUs) and Tensor Processing Units (TPUs), that excel at accelerating parallel workloads and dense vector matrix multiplications. Potentially more efficient neural network models utilizing sparsity and recurrence cannot leverage the full power of SIMD processor and are thus at a severe disadvantage compared to today's prominent parallel architectures like Transformers and CNNs, thereby hindering the path towards more sustainable AI. 
To overcome this limitation, we explore sparse and recurrent model training on a massively parallel multiple instruction multiple data (MIMD) architecture with distributed local memory. We implement a training routine based on backpropagation though time (BPTT) for the brain-inspired class of Spiking Neural Networks (SNNs) that feature binary sparse activations. We observe a massive advantage in using sparse activation tensors with a MIMD processor, the Intelligence Processing Unit (IPU) compared to GPUs. On training workloads, our results demonstrate 5-10x throughput gains compared to A100 GPUs and up to 38x gains for higher levels of activation sparsity, without a significant slowdown in training convergence or reduction in final model performance. Furthermore, our results show highly promising trends for both single and multi IPU configurations as we scale up to larger model sizes. 
Our work paves the way towards more efficient, non-standard models via AI training hardware beyond GPUs, and competitive large scale SNN models.

----

## [1339] Graph Learning in 4D: A Quaternion-Valued Laplacian to Enhance Spectral GCNs

**Authors**: *Stefano Fiorini, Stefano Coniglio, Michele Ciavotta, Enza Messina*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29088](https://doi.org/10.1609/aaai.v38i11.29088)

**Abstract**:

We introduce QuaterGCN, a spectral Graph Convolutional Network (GCN) with quaternion-valued weights at whose core lies the Quaternionic Laplacian, a quaternion-valued Laplacian matrix by whose proposal we generalize two widely-used Laplacian matrices: the classical Laplacian (defined for undirected graphs) and the complex-valued Sign-Magnetic Laplacian (proposed within the spectral GCN SigMaNet to handle digraphs with weights of arbitrary sign). In addition to its generality, QuaterGCN is the only Laplacian to completely preserve the (di)graph topology that we are aware of, as it can handle graphs and digraphs containing antiparallel pairs of edges (digons) of different weight without reducing them to a single (directed or undirected) edge as done by other Laplacians. Experimental results show the superior performance of QuaterGCN compared to other state-of-the-art GCNs, particularly in scenarios where the information the digons carry is crucial to successfully address the task at hand.

----

## [1340] Learning Broadcast Protocols

**Authors**: *Dana Fisman, Noa Izsak, Swen Jacobs*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29089](https://doi.org/10.1609/aaai.v38i11.29089)

**Abstract**:

The problem of learning a computational model from examples has been receiving growing attention. For the particularly challenging problem of learning models of distributed systems, existing results are restricted to models with a fixed number of interacting processes. In this work we look for the first time (to the best of our knowledge) at the problem of learning a distributed system with an arbitrary number of processes, assuming only that there exists a cutoff, i.e., a number of processes that is sufficient to produce all observable behaviors. Specifically, we consider fine broadcast protocols, these are broadcast protocols (BPs) with a finite cutoff and no hidden states. We provide a learning algorithm that can infer a correct BP from a sample that is consistent with a fine BP, and a minimal equivalent BP if the sample is sufficiently complete. On the negative side we show that (a) characteristic sets of exponential size are unavoidable, (b)  the consistency problem for fine BPs is NP hard, and (c) that fine BPs are not polynomially predictable.

----

## [1341] Beyond Expected Return: Accounting for Policy Reproducibility When Evaluating Reinforcement Learning Algorithms

**Authors**: *Manon Flageat, Bryan Lim, Antoine Cully*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29090](https://doi.org/10.1609/aaai.v38i11.29090)

**Abstract**:

Many applications in Reinforcement Learning (RL) usually have noise or stochasticity present in the environment. Beyond their impact on learning, these uncertainties lead the exact same policy to perform differently, i.e. yield different return, from one roll-out to another. Common evaluation procedures in RL summarise the consequent return distributions using solely the expected return, which does not account for the spread of the distribution. Our work defines this spread as the policy reproducibility: the ability of a policy to obtain similar performance when rolled out many times, a crucial property in some real-world applications. We highlight that existing procedures that only use the expected return are limited on two fronts: first an infinite number of return distributions with a wide range of performance-reproducibility trade-offs can have the same expected return, limiting its effectiveness when used for comparing policies; second, the expected return metric does not leave any room for practitioners to choose the best trade-off value for considered applications. In this work, we address these limitations by recommending the use of Lower Confidence Bound, a metric taken from Bayesian optimisation that provides the user with a preference parameter to choose a desired performance-reproducibility trade-off.
We also formalise and quantify policy reproducibility, and demonstrate the benefit of our metrics using extensive experiments of popular RL algorithms on common uncertain RL tasks.

----

## [1342] Symbolic Regression Enhanced Decision Trees for Classification Tasks

**Authors**: *Kei Sen Fong, Mehul Motani*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29091](https://doi.org/10.1609/aaai.v38i11.29091)

**Abstract**:

We introduce a conceptually simple yet effective method to create small, compact decision trees - by using splits found via Symbolic Regression (SR). Traditional decision tree (DT) algorithms partition a dataset on axis-parallel splits. When the true boundaries are not along the feature axes, DT is likely to have a complicated structure and a dense decision boundary. In this paper, we introduce SR-Enhanced DT (SREDT) - a method which utilizes SR to increase the richness of the class of possible DT splits. We evaluate SREDT on both synthetic and real-world datasets. Despite its simplicity, our method produces surprisingly small trees that outperform both DT and oblique DT (ODT) on supervised classification tasks in terms of accuracy and F-score. We show empirically that SREDTs decrease inference time (compared to DT and ODT) and argue that they allow us to obtain more explainable descriptions of the decision process. SREDT also performs competitively against state-of-the-art tabular classification methods, including tree ensembles and deep models. Finally, we introduce a local search mechanism to improve SREDT and evaluate it on 56 PMLB datasets. This mechanism shows improved performance on 77.2% of the datasets, outperforming DT and ODT. In terms of F-Score, local SREDT outperforms DT and ODT in 82.5% and 73.7% of the datasets respectively and in terms of inference time, local SREDT requires 25.8% and 26.6% less inference time than DT and ODT respectively.

----

## [1343] Fast Machine Unlearning without Retraining through Selective Synaptic Dampening

**Authors**: *Jack Foster, Stefan Schoepf, Alexandra Brintrup*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29092](https://doi.org/10.1609/aaai.v38i11.29092)

**Abstract**:

Machine unlearning, the ability for a machine learning model to forget, is becoming increasingly important to comply with data privacy regulations, as well as to remove harmful, manipulated, or outdated information. The key challenge lies in forgetting specific information while protecting model performance on the remaining data.  While current state-of-the-art methods perform well, they typically require some level of retraining over the retained data, in order to protect or restore model performance. This adds computational overhead and mandates that the training data remain available and accessible, which may not be feasible. In contrast, other methods employ a retrain-free paradigm, however, these approaches are prohibitively computationally expensive and do not perform on par with their retrain-based counterparts. We present Selective Synaptic Dampening (SSD), a novel two-step, post hoc, retrain-free approach to machine unlearning which is fast, performant, and does not require long-term storage of the training data. First, SSD uses the Fisher information matrix of the training and forgetting data to select parameters that are disproportionately important to the forget set. Second, SSD induces forgetting by dampening these parameters proportional to their relative importance to the forget set with respect to the wider training data. We evaluate our method against several existing unlearning methods in a range of experiments using ResNet18 and Vision Transformer. Results show that the performance of SSD is competitive with retrain-based post hoc methods, demonstrating the viability of retrain-free post hoc unlearning approaches.

----

## [1344] Combinatorial Stochastic-Greedy Bandit

**Authors**: *Fares Fourati, Christopher John Quinn, Mohamed-Slim Alouini, Vaneet Aggarwal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29093](https://doi.org/10.1609/aaai.v38i11.29093)

**Abstract**:

We propose a novel combinatorial stochastic-greedy bandit (SGB) algorithm for combinatorial multi-armed bandit problems when no extra information other than the joint reward of the selected set of n arms at each time step t in [T] is observed. SGB adopts an optimized stochastic-explore-then-commit approach and is specifically designed for scenarios with a large set of base arms. Unlike existing methods that explore the entire set of unselected base arms during each selection step, our SGB algorithm samples only an optimized proportion of unselected arms and selects actions from this subset. We prove that our algorithm achieves a (1-1/e)-regret bound of O(n^(1/3) k^(2/3) T^(2/3) log(T)^(2/3)) for monotone stochastic submodular rewards, which outperforms the state-of-the-art in terms of the cardinality constraint k. Furthermore, we empirically evaluate the performance of our algorithm in the context of online constrained social influence maximization. Our results demonstrate that our proposed approach consistently outperforms the other algorithms, increasing the performance gap as k grows.

----

## [1345] REGLO: Provable Neural Network Repair for Global Robustness Properties

**Authors**: *Feisi Fu, Zhilu Wang, Weichao Zhou, Yixuan Wang, Jiameng Fan, Chao Huang, Qi Zhu, Xin Chen, Wenchao Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29094](https://doi.org/10.1609/aaai.v38i11.29094)

**Abstract**:

We present REGLO, a novel methodology for repairing pretrained neural networks to satisfy global robustness and individual fairness properties. A neural network is said to be globally robust with respect to a given input region if and only if all the input points in the region are locally robust. This notion of global robustness also captures the notion of individual fairness as a special case. We prove that any counterexample to a global robustness property must exhibit a corresponding large gradient. For ReLU networks, this result allows us to efficiently identify the linear regions that violate a given global robustness property. By formulating and solving a suitable robust convex optimization problem, REGLO then computes a minimal weight change that will provably repair these violating linear regions.

----

## [1346] QLABGrad: A Hyperparameter-Free and Convergence-Guaranteed Scheme for Deep Learning

**Authors**: *Minghan Fu, Fang-Xiang Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29095](https://doi.org/10.1609/aaai.v38i11.29095)

**Abstract**:

The learning rate is a critical hyperparameter for deep learning
tasks since it determines the extent to which the model
parameters are adjusted during the learning course. However,
the choice of learning rates typically depends on empirical
judgment, which may not result in satisfactory outcomes
without intensive try-and-error experiments. In this
study, we propose a novel learning rate adaptation scheme
called QLABGrad. Without any user-specified hyperparameter,
QLABGrad automatically determines the learning rate by
optimizing the quadratic loss approximation-based (QLAB)
function for a given gradient descent direction, where only
one extra forward propagation is required. We theoretically
prove the convergence of QLABGrad under the smooth Lipschitz
condition on the loss function. Experiment results on
multiple architectures, including MLP, CNN, and ResNet, on
MNIST, CIFAR10, and ImageNet datasets, demonstrate that
QLABGrad outperforms widely adopted schemes for deep
learning.

----

## [1347] DTL: Disentangled Transfer Learning for Visual Recognition

**Authors**: *Minghao Fu, Ke Zhu, Jianxin Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29096](https://doi.org/10.1609/aaai.v38i11.29096)

**Abstract**:

When pre-trained models become rapidly larger, the cost of fine-tuning on downstream tasks steadily increases, too. To economically fine-tune these models, parameter-efficient transfer learning (PETL) is proposed, which only tunes a tiny subset of trainable parameters to efficiently learn quality representations. However, current PETL methods are facing the dilemma that during training the GPU memory footprint is not effectively reduced as trainable parameters. PETL will likely fail, too, if the full fine-tuning encounters the out-of-GPU-memory issue. This phenomenon happens because trainable parameters from these methods are generally entangled with the backbone, such that a lot of intermediate states have to be stored in GPU memory for gradient propagation. To alleviate this problem, we introduce Disentangled Transfer Learning (DTL), which disentangles the trainable parameters from the backbone using a lightweight Compact Side Network (CSN). By progressively extracting task-specific information with a few low-rank linear mappings and appropriately adding the information back to the backbone, CSN effectively realizes knowledge transfer in various downstream tasks. We conducted extensive experiments to validate the effectiveness of our method. The proposed method not only reduces a large amount of GPU memory usage and trainable parameters, but also outperforms existing PETL methods by a significant margin in accuracy, achieving new state-of-the-art on several standard benchmarks.

----

## [1348] Noise-Aware Image Captioning with Progressively Exploring Mismatched Words

**Authors**: *Zhongtian Fu, Kefei Song, Luping Zhou, Yang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29097](https://doi.org/10.1609/aaai.v38i11.29097)

**Abstract**:

Image captioning aims to automatically generate captions for images by learning a cross-modal generator from vision to language. The large amount of image-text pairs required for training is usually sourced from the internet due to the manual cost, which brings the noise with mismatched relevance that affects the learning process. Unlike traditional noisy label learning, the key challenge in processing noisy image-text pairs is to finely identify the mismatched words to make the most use of trustworthy information in the text, rather than coarsely weighing the entire examples. To tackle this challenge, we propose a Noise-aware Image Captioning method (NIC) to adaptively mitigate the erroneous guidance from noise by progressively exploring mismatched words. Specifically, NIC first identifies mismatched words by quantifying word-label reliability from two aspects: 1) inter-modal representativeness, which measures the significance of the current word by assessing cross-modal correlation via prediction certainty; 2) intra-modal informativeness, which amplifies the effect of current prediction by combining the quality of subsequent word generation. During optimization, NIC constructs the pseudo-word-labels considering the reliability of the origin word-labels and model convergence to periodically coordinate mismatched words. As a result, NIC can effectively exploit both clean and noisy image-text pairs to learn a more robust mapping function. Extensive experiments conducted on the MS-COCO and Conceptual Caption datasets validate the effectiveness of our method in various noisy scenarios.

----

## [1349] Learning Small Decision Trees with Few Outliers: A Parameterized Perspective

**Authors**: *Harmender Gahlawat, Meirav Zehavi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29098](https://doi.org/10.1609/aaai.v38i11.29098)

**Abstract**:

Decision trees is a fundamental tool in machine learning for representing, classifying, and generalizing data. It is desirable to construct ``small'' decision trees, by minimizing either the size (s) or the depth (d) of the decision tree (DT). Recently, the parameterized complexity of Decision Tree Learning has attracted a lot of attention.
We consider a generalization of Decision Tree Learning where given a classification instance E and an integer t, the task is to find a ``small'' DT that disagrees with E in at most t examples. We consider two problems: DTSO and DTDO, where  the goal is to construct a DT minimizing s and d, respectively. We first establish that both DTSO and DTDO are W[1]-hard when parameterized by s+y and d+y, respectively, where y is the maximum number of features in which two differently labeled examples can differ. We complement this result by showing that these problems become FPT if we include the parameter t. We also consider the kernelization complexity of these problems and establish several positive and negative results for both DTSO and DTDO.

----

## [1350] Online Sensitivity Optimization in Differentially Private Learning

**Authors**: *Filippo Galli, Catuscia Palamidessi, Tommaso Cucinotta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29099](https://doi.org/10.1609/aaai.v38i11.29099)

**Abstract**:

Training differentially private machine learning models requires constraining an individual's contribution to the optimization process. This is achieved by clipping the 2-norm of their gradient at a predetermined threshold prior to averaging and batch sanitization. This selection adversely influences optimization in two opposing ways: it either exacerbates the bias due to excessive clipping at lower values, or augments sanitization noise at higher values. The choice significantly hinges on factors such as the dataset, model architecture, and even varies within the same optimization, demanding meticulous tuning usually accomplished through a grid search. In order to circumvent the privacy expenses incurred in hyperparameter tuning, we present a novel approach to dynamically optimize the clipping threshold. We treat this threshold as an additional learnable parameter, establishing a clean relationship between the threshold and the cost function. This allows us to optimize the former with gradient descent, with minimal repercussions on the overall privacy analysis. Our method is thoroughly assessed against alternative fixed and adaptive strategies across diverse datasets, tasks, model dimensions, and privacy levels. Our results indicate that it performs comparably or better in the evaluated scenarios, given the same privacy requirements.

----

## [1351] Compressing Image-to-Image Translation GANs Using Local Density Structures on Their Learned Manifold

**Authors**: *Alireza Ganjdanesh, Shangqian Gao, Hirad Alipanah, Heng Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29100](https://doi.org/10.1609/aaai.v38i11.29100)

**Abstract**:

Generative Adversarial Networks (GANs) have shown remarkable success in modeling complex data distributions for image-to-image translation. Still, their high computational demands prohibit their deployment in practical scenarios like edge devices. Existing GAN compression methods mainly rely on knowledge distillation or convolutional classifiers' pruning techniques. Thus, they neglect the critical characteristic of GANs: their local density structure over their learned manifold. Accordingly, we approach GAN compression from a new perspective by explicitly encouraging the pruned model to preserve the density structure of the original parameter-heavy model on its learned manifold. We facilitate this objective for the pruned model by partitioning the learned manifold of the original generator into local neighborhoods around its generated samples. Then, we propose a novel pruning objective to regularize the pruned model to preserve the local density structure over each neighborhood, resembling the kernel density estimation method. Also, we develop a collaborative pruning scheme in which the discriminator and generator are pruned by two pruning agents. We design the agents to capture interactions between the generator and discriminator by exchanging their peer's feedback when determining corresponding models' architectures. Thanks to such a design, our pruning method can efficiently find performant sub-networks and can maintain the balance between the generator and discriminator more effectively compared to baselines during pruning, thereby showing more stable pruning dynamics. Our experiments on image translation GAN models, Pix2Pix and CycleGAN, with various benchmark datasets and architectures demonstrate our method's effectiveness.

----

## [1352] ACT: Empowering Decision Transformer with Dynamic Programming via Advantage Conditioning

**Authors**: *Chenxiao Gao, Chenyang Wu, Mingjun Cao, Rui Kong, Zongzhang Zhang, Yang Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29101](https://doi.org/10.1609/aaai.v38i11.29101)

**Abstract**:

Decision Transformer (DT), which employs expressive sequence modeling techniques to perform action generation, has emerged as a promising approach to offline policy optimization. However, DT generates actions conditioned on a desired future return, which is known to bear some weaknesses such as the susceptibility to environmental stochasticity. To overcome DT's weaknesses, we propose to empower DT with dynamic programming. Our method comprises three steps. First, we employ in-sample value iteration to obtain approximated value functions, which involves dynamic programming over the MDP structure. Second, we evaluate action quality in context with estimated advantages. We introduce two types of advantage estimators, IAE and GAE, which are suitable for different tasks. Third, we train an Advantage-Conditioned Transformer (ACT) to generate actions conditioned on the estimated advantages. Finally, during testing, ACT generates actions conditioned on a desired advantage. Our evaluation results validate that, by leveraging the power of dynamic programming, ACT demonstrates effective trajectory stitching and robust action generation in spite of the environmental stochasticity, outperforming baseline methods across various benchmarks. Additionally, we conduct an in-depth analysis of ACT's various design choices through ablation studies. Our code is available at https://github.com/LAMDA-RL/ACT.

----

## [1353] Get a Head Start: On-Demand Pedagogical Policy Selection in Intelligent Tutoring

**Authors**: *Ge Gao, Xi Yang, Min Chi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29102](https://doi.org/10.1609/aaai.v38i11.29102)

**Abstract**:

Reinforcement learning (RL) is broadly employed in human-involved systems to enhance human outcomes. Off-policy evaluation (OPE) has been pivotal for RL in those realms since online policy learning and evaluation can be high-stake. Intelligent tutoring has raised tremendous attentions as highly challenging when applying OPE to human-involved systems, due to that students' subgroups can favor different pedagogical policies and the costly procedure that policies have to be induced fully offline and then directly deployed to the upcoming semester. In this work, we formulate on-demand pedagogical policy selection (ODPS) to tackle the challenges for OPE in intelligent tutoring. We propose a pipeline, EduPlanner, as a concrete solution for ODPS. Our pipeline results in an theoretically unbiased estimator, and enables efficient and customized policy selection by identifying subgroups over both historical data and on-arrival initial logs. We evaluate our approach on the Probability ITS that has been used in real classrooms for over eight years. Our study shows significant improvement on learning outcomes of students with EduPlanner, especially for the ones associated with low-performing subgroups.

----

## [1354] Rethinking Causal Relationships Learning in Graph Neural Networks

**Authors**: *Hang Gao, Chengyu Yao, Jiangmeng Li, Lingyu Si, Yifan Jin, Fengge Wu, Changwen Zheng, Huaping Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29103](https://doi.org/10.1609/aaai.v38i11.29103)

**Abstract**:

Graph Neural Networks (GNNs) demonstrate their significance by effectively modeling complex interrelationships within graph-structured data. To enhance the credibility and robustness of GNNs, it becomes exceptionally crucial to bolster their ability to capture causal relationships. However, despite recent advancements that have indeed strengthened GNNs from a causal learning perspective, conducting an in-depth analysis specifically targeting the causal modeling prowess of GNNs remains an unresolved issue. In order to comprehensively analyze various GNN models from a causal learning perspective, we constructed an artificially synthesized dataset with known and controllable causal relationships between data and labels. The rationality of the generated data is further ensured through theoretical foundations. Drawing insights from analyses conducted using our dataset, we introduce a lightweight and highly adaptable GNN module designed to strengthen GNNs' causal learning capabilities across a diverse range of tasks. Through a series of experiments conducted on both synthetic datasets and other real-world datasets, we empirically validate the effectiveness of the proposed module. The codes are available at https://github.com/yaoyao-yaoyao-cell/CRCG.

----

## [1355] AVSegFormer: Audio-Visual Segmentation with Transformer

**Authors**: *Shengyi Gao, Zhe Chen, Guo Chen, Wenhai Wang, Tong Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29104](https://doi.org/10.1609/aaai.v38i11.29104)

**Abstract**:

Audio-visual segmentation (AVS) aims to locate and segment the sounding objects in a given video, which demands audio-driven pixel-level scene understanding. The existing methods cannot fully process the fine-grained correlations between audio and visual cues across various situations dynamically. They also face challenges in adapting to complex scenarios, such as evolving audio, the coexistence of multiple objects, and more. In this paper, we propose AVSegFormer, a novel framework for AVS that leverages the transformer architecture. Specifically, It comprises a dense audio-visual mixer, which can dynamically adjust interested visual features, and a sparse audio-visual decoder, which implicitly separates audio sources and automatically matches optimal visual features. Combining both components provides a more robust bidirectional conditional multi-modal representation, improving the segmentation performance in different scenarios. Extensive experiments demonstrate that AVSegFormer achieves state-of-the-art results on the AVS benchmark. The code is available at https://github.com/vvvb-github/AVSegFormer.

----

## [1356] Eliciting Kemeny Rankings

**Authors**: *Anne-Marie George, Christos Dimitrakakis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29105](https://doi.org/10.1609/aaai.v38i11.29105)

**Abstract**:

We formulate the problem of eliciting agents' preferences with the goal of finding a Kemeny ranking as a Dueling Bandits problem.
Here the bandits' arms correspond to alternatives that need to be ranked and the feedback corresponds to a pairwise comparison between alternatives by a randomly sampled agent.
We consider both sampling with and without replacement, i.e., the possibility to ask the same agent about some comparison multiple times or not.

We find approximation bounds for Kemeny rankings dependant on confidence intervals over estimated winning probabilities of arms. 
Based on these we state algorithms to find Probably Approximately Correct (PAC) solutions and elaborate on their sample complexity for sampling with or without replacement.
Furthermore, if all agents' preferences are strict rankings over the alternatives, we provide means to prune confidence intervals and thereby guide a more efficient elicitation. We formulate several adaptive sampling methods that use look-aheads to estimate how much confidence intervals (and thus approximation guarantees) might be tightened. 
All described methods are compared on synthetic data.

----

## [1357] Interactive Hyperparameter Optimization in Multi-Objective Problems via Preference Learning

**Authors**: *Joseph Giovanelli, Alexander Tornede, Tanja Tornede, Marius Lindauer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29106](https://doi.org/10.1609/aaai.v38i11.29106)

**Abstract**:

Hyperparameter optimization (HPO) is important to leverage the full potential of machine learning (ML). 
In practice, users are often interested in multi-objective (MO) problems, i.e., optimizing potentially conflicting objectives, like accuracy and energy consumption.
To tackle this, the vast majority of MO-ML algorithms return a Pareto front of non-dominated machine learning models to the user.
Optimizing the hyperparameters of such algorithms is non-trivial as evaluating a hyperparameter configuration entails evaluating the quality of the resulting Pareto front. 
In literature, there are known indicators that assess the quality of a Pareto front (e.g., hypervolume, R2) by quantifying different properties (e.g., volume, proximity to a reference point). However, choosing the indicator that leads to the desired Pareto front might be a hard task for a user. In this paper, we propose a human-centered interactive HPO approach tailored towards multi-objective ML leveraging preference learning to extract desiderata from users that guide the optimization.
Instead of relying on the user guessing the most suitable indicator for their needs, our approach automatically learns an appropriate indicator.
Concretely, we leverage pairwise comparisons of distinct Pareto fronts to learn such an appropriate quality indicator.
Then, we optimize the hyperparameters of the underlying MO-ML algorithm towards this learned indicator using a state-of-the-art HPO approach.
In an experimental study targeting the environmental impact of ML, we demonstrate that our approach leads to substantially better Pareto fronts compared to optimizing based on a wrong indicator pre-selected by the user, and performs comparable in the case of an advanced user knowing which indicator to pick.

----

## [1358] Layer Compression of Deep Networks with Straight Flows

**Authors**: *Chengyue Gong, Xiaocong Du, Bhargav Bhushanam, Lemeng Wu, Xingchao Liu, Dhruv Choudhary, Arun Kejariwal, Qiang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29107](https://doi.org/10.1609/aaai.v38i11.29107)

**Abstract**:

Very deep neural networks lead to significantly better performance on various real tasks. However, it usually causes slow inference and is hard to be deployed on real-world devices. How to reduce the number of layers to save memory and to accelerate the inference is an eye-catching topic. 
   In this work, we introduce an intermediate objective, a continuous-time network, before distilling deep networks into shallow networks.
   First, we distill a given deep network into a continuous-time neural flow model, which can be discretized with an ODE solver and the inference requires passing through the network multiple times.
   By forcing the flow transport trajectory to be straight lines, we find that it is easier to compress the infinite step model into a one-step neural flow model, which only requires passing through the flow model once.
   Secondly, we refine the one-step flow model together with the final head layer with knowledge distillation and finally, we can replace the given deep network with this one-step flow network. 
   Empirically, we demonstrate that our method outperforms direct distillation and other baselines on different model architectures (e.g. ResNet, ViT) on image classification and semantic segmentation tasks. 
   We also manifest that our distilled model naturally serves as an early-exit dynamic inference model.

----

## [1359] Fast and Controllable Post-training Sparsity: Learning Optimal Sparsity Allocation with Global Constraint in Minutes

**Authors**: *Ruihao Gong, Yang Yong, Zining Wang, Jinyang Guo, Xiuying Wei, Yuqing Ma, Xianglong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29108](https://doi.org/10.1609/aaai.v38i11.29108)

**Abstract**:

Neural network sparsity has attracted many research interests due to its similarity to biological schemes and high energy efficiency. However, existing methods depend on long-time training or fine-tuning, which prevents large-scale applications. Recently, some works focusing on post-training sparsity (PTS) have emerged. They get rid of the high training cost but usually suffer from distinct accuracy degradation due to neglect of the reasonable sparsity rate at each layer. Previous methods for finding sparsity rates mainly focus on the training-aware scenario, which usually fails to converge stably under the PTS setting with limited data and much less training cost. In this paper, we propose a fast and controllable post-training sparsity (FCPTS) framework. By incorporating a differentiable bridge function and a controllable optimization objective, our method allows for rapid and accurate sparsity allocation learning in minutes, with the added assurance of convergence to a predetermined global sparsity rate. Equipped with these techniques, we can surpass the state-of-the-art methods by a large margin, e.g., over 30\% improvement for ResNet-50 on ImageNet under the sparsity rate of 80\%. Our plug-and-play code and supplementary materials are open-sourced at https://github.com/ModelTC/FCPTS.

----

## [1360] DeepSaDe: Learning Neural Networks That Guarantee Domain Constraint Satisfaction

**Authors**: *Kshitij Goyal, Sebastijan Dumancic, Hendrik Blockeel*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29109](https://doi.org/10.1609/aaai.v38i11.29109)

**Abstract**:

As machine learning models, specifically neural networks, are becoming increasingly popular, there are concerns regarding their trustworthiness, specially in safety-critical applications, e.g. actions of an autonomous vehicle must be safe. There are approaches that can train neural networks where such domain requirements are enforced as constraints, but they either cannot guarantee that the constraint will be satisfied by all possible predictions (even on unseen data) or they are limited in the type of constraints that can be enforced. In this paper, we present an approach to train neural networks which can enforce a wide variety of constraints and guarantee that the constraint is satisfied by all possible predictions. The approach builds on earlier work where learning linear models is formulated as a constraint satisfaction problem (CSP). To make this idea applicable to neural networks, two crucial new elements are added: constraint propagation over the network layers, and weight updates based on a mix of gradient descent and CSP solving. Evaluation on various machine learning tasks demonstrates that our approach is flexible enough to enforce a wide variety of domain constraints and is able to guarantee them in neural networks.

----

## [1361] Taming the Sigmoid Bottleneck: Provably Argmaxable Sparse Multi-Label Classification

**Authors**: *Andreas Grivas, Antonio Vergari, Adam Lopez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29110](https://doi.org/10.1609/aaai.v38i11.29110)

**Abstract**:

Sigmoid output layers are widely used in multi-label classification (MLC) tasks, in which multiple labels can be assigned to any input. In many practical MLC tasks, the number of possible labels is in the thousands, often exceeding the number of input features and resulting in a low-rank output layer. In multi-class classification, it is known that such a low-rank output layer is a bottleneck that can result in unargmaxable classes: classes which cannot be predicted for any input. 
In this paper, we show that for MLC tasks, the analogous sigmoid bottleneck results in exponentially many unargmaxable label combinations. We explain how to detect these unargmaxable outputs and demonstrate their presence in three widely used MLC datasets. We then show that they can be prevented in practice by introducing a Discrete Fourier Transform (DFT) output layer, which guarantees that all sparse label combinations with up to k active labels are argmaxable. Our DFT layer trains faster and is more parameter efficient, matching the F1@k score of a sigmoid layer while using up to 50% fewer trainable parameters. Our code is publicly available at https://github.com/andreasgrv/sigmoid-bottleneck.

----

## [1362] Summarizing Stream Data for Memory-Constrained Online Continual Learning

**Authors**: *Jianyang Gu, Kai Wang, Wei Jiang, Yang You*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29111](https://doi.org/10.1609/aaai.v38i11.29111)

**Abstract**:

Replay-based methods have proved their effectiveness on online continual learning by rehearsing past samples from an auxiliary memory. With many efforts made on improving training schemes based on the memory, however, the information carried by each sample in the memory remains under-investigated. Under circumstances with restricted storage space, the informativeness of the memory becomes critical for effective replay. Although some works design specific strategies to select representative samples, by only employing a small number of original images, the storage space is still not well utilized. To this end, we propose to Summarize the knowledge from the Stream Data (SSD) into more informative samples by distilling the training characteristics of real images. Through maintaining the consistency of training gradients and relationship to the past tasks, the summarized samples are more representative for the stream data compared to the original images. Extensive experiments are conducted on multiple online continual learning benchmarks to support that the proposed SSD method significantly enhances the replay effects. We demonstrate that with limited extra computational overhead, SSD provides more than 3% accuracy boost for sequential CIFAR-100 under extremely restricted memory buffer. Code in https://github.com/vimar-gu/SSD.

----

## [1363] G-Adapter: Towards Structure-Aware Parameter-Efficient Transfer Learning for Graph Transformer Networks

**Authors**: *Anchun Gui, Jinqiang Ye, Han Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29112](https://doi.org/10.1609/aaai.v38i11.29112)

**Abstract**:

It has become a popular paradigm to transfer the knowledge of large-scale pre-trained models to various downstream tasks via fine-tuning the entire model parameters. However, with the growth of model scale and the rising number of downstream tasks, this paradigm inevitably meets the challenges in terms of computation consumption and memory footprint issues. Recently, Parameter-Efficient Fine-Tuning (PEFT) (e.g., Adapter, LoRA, BitFit) shows a promising paradigm to alleviate these concerns by updating only a portion of parameters. Despite these PEFTs having demonstrated satisfactory performance in natural language processing, it remains under-explored for the question: whether these techniques could be transferred to graph-based tasks with Graph Transformer Networks (GTNs)? Therefore, in this paper, we fill this gap by providing extensive benchmarks with traditional PEFTs on a range of graph-based downstream tasks. Our empirical study shows that it is sub-optimal to directly transfer existing PEFTs to graph-based tasks due to the issue of feature distribution shift. To address this issue, we propose a novel structure-aware PEFT approach, named G-Adapter, which leverages graph convolution operation to introduce graph structure information (e.g., graph adjacency matrix) as an inductive bias to guide the updating process. Further, we propose Bregman proximal point optimization to alleviate feature distribution shift by preventing the model from aggressive update. Extensive experiments demonstrate that G-Adapter obtains state-of-the-art performance compared to counterparts on nine graph benchmark datasets based on diverse pre-trained GTNs, and delivers tremendous memory footprint efficiency compared to the conventional paradigm.

----

## [1364] FedCSL: A Scalable and Accurate Approach to Federated Causal Structure Learning

**Authors**: *Xianjie Guo, Kui Yu, Lin Liu, Jiuyong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29113](https://doi.org/10.1609/aaai.v38i11.29113)

**Abstract**:

As an emerging research direction, federated causal structure learning (CSL) aims at learning causal relationships from decentralized data across multiple clients while preserving data privacy. Existing federated CSL algorithms suffer from scalability and accuracy issues, since they require computationally expensive CSL algorithms to be executed at each client. Furthermore, in real-world scenarios, the number of samples held by each client varies significantly, and existing methods still assign equal weights to the learned structural information from each client, which severely harms the learning accuracy of those methods. To address these two limitations, we propose FedCSL, a scalable and accurate method for federated CSL. Specifically, FedCSL consists of two novel strategies: (1) a federated local-to-global learning strategy that enables FedCSL to scale to high-dimensional data for tackling the scalability issue, and (2) a novel weighted aggregation strategy that does not rely on any complex encryption techniques while preserving data privacy for tackling the accuracy issue. Extensive experiments on benchmark datasets, high-dimensional synthetic datasets and a real-world dataset verify the efficacy of the proposed FedCSL method. The source code is available at https://github.com/Xianjie-Guo/FedCSL.

----

## [1365] Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks

**Authors**: *Yufei Guo, Yuanpei Chen, Xiaode Liu, Weihang Peng, Yuhan Zhang, Xuhui Huang, Zhe Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29114](https://doi.org/10.1609/aaai.v38i11.29114)

**Abstract**:

The Spiking Neural Network (SNN), as one of the biologically inspired neural network infrastructures, has drawn increasing attention recently. It adopts binary spike activations to transmit information, thus the multiplications of activations and weights can be substituted by additions, which brings high energy efficiency.  However, in the paper, we theoretically and experimentally prove that the binary spike activation map cannot carry enough information, thus causing information loss and resulting in accuracy decreasing. To handle the problem, we propose a ternary spike neuron to transmit information. The ternary spike neuron can also enjoy the event-driven and multiplication-free operation advantages of the binary spike neuron but will boost the information capacity. Furthermore, we also embed a trainable factor in the ternary spike neuron to learn the suitable spike amplitude, thus our SNN will adopt different spike amplitudes along layers, which can better suit the phenomenon that the membrane potential distributions are different along layers. To retain the efficiency of the vanilla ternary spike, the trainable ternary spike SNN will be converted to a standard one again via a re-parameterization technique in the inference. Extensive experiments with several popular network structures over static and dynamic datasets show that the ternary spike can consistently outperform state-of-the-art methods. Our code is open-sourced at https://github.com/yfguo91/Ternary-Spike.

----

## [1366] From Past to Future: Rethinking Eligibility Traces

**Authors**: *Dhawal Gupta, Scott M. Jordan, Shreyas Chaudhari, Bo Liu, Philip S. Thomas, Bruno Castro da Silva*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29115](https://doi.org/10.1609/aaai.v38i11.29115)

**Abstract**:

In this paper, we introduce a fresh perspective on the challenges of credit assignment and policy evaluation. First, we delve into the nuances of eligibility traces and explore instances where their updates may result in unexpected credit assignment to preceding states. From this investigation emerges the concept of a novel value function, which we refer to as the ????????????? ????? ????????. Unlike traditional state value functions, bidirectional value functions account for both future expected returns (rewards anticipated from the current state onward) and past expected returns (cumulative rewards from the episode's start to the present). We derive principled update equations to learn this value function and, through experimentation, demonstrate its efficacy in enhancing the process of policy evaluation. In particular, our results indicate that the proposed learning approach can, in certain challenging contexts, perform policy evaluation more rapidly than TD(λ)–a method that learns forward value functions, v^π, ????????. Overall, our findings present a new perspective on eligibility traces and potential advantages associated with the novel value function it inspires, especially for policy evaluation.

----

## [1367] Domain-Aware Fine-Tuning: Enhancing Neural Network Adaptability

**Authors**: *Seokhyeon Ha, Sunbeom Jeong, Jungwoo Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29116](https://doi.org/10.1609/aaai.v38i11.29116)

**Abstract**:

Fine-tuning pre-trained neural network models has become a widely adopted approach across various domains. However, it can lead to the distortion of pre-trained feature extractors that already possess strong generalization capabilities. Mitigating feature distortion during adaptation to new target domains is crucial. Recent studies have shown promising results in handling feature distortion by aligning the head layer on in-distribution datasets before performing fine-tuning. Nonetheless, a significant limitation arises from the treatment of batch normalization layers during fine-tuning, leading to suboptimal performance. In this paper, we propose Domain-Aware Fine-Tuning (DAFT), a novel approach that incorporates batch normalization conversion and the integration of linear probing and fine-tuning. Our batch normalization conversion method effectively mitigates feature distortion by reducing modifications to the neural network during fine-tuning. Additionally, we introduce the integration of linear probing and fine-tuning to optimize the head layer with gradual adaptation of the feature extractor. By leveraging batch normalization layers and integrating linear probing and fine-tuning, our DAFT significantly mitigates feature distortion and achieves improved model performance on both in-distribution and out-of-distribution datasets. Extensive experiments demonstrate that our method outperforms other baseline methods, demonstrating its effectiveness in not only improving performance but also mitigating feature distortion.

----

## [1368] Forced Exploration in Bandit Problems

**Authors**: *Qi Han, Li Zhu, Fei Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29117](https://doi.org/10.1609/aaai.v38i11.29117)

**Abstract**:

The multi-armed bandit(MAB)  is a classical sequential decision problem. Most work requires assumptions about the reward distribution (e.g.,  bounded), while practitioners may have difficulty obtaining information about these distributions to design models for their problems, especially in non-stationary MAB problems. This paper aims to  design a multi-armed bandit algorithm that can be implemented without using information about the reward distribution  while still achieving substantial regret upper bounds. To this end, we propose a novel algorithm alternating between greedy rule and forced exploration. Our method can be applied to Gaussian, Bernoulli and other subgaussian distributions, and its implementation does not require additional information. We employ a unified analysis method for different forced exploration strategies and provide problem-dependent regret upper bounds for stationary and piecewise-stationary settings. Furthermore, we compare our algorithm with popular bandit algorithms on different reward distributions.

----

## [1369] Complexity of Neural Network Training and ETR: Extensions with Effectively Continuous Functions

**Authors**: *Teemu Hankala, Miika Hannula, Juha Kontinen, Jonni Virtema*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29118](https://doi.org/10.1609/aaai.v38i11.29118)

**Abstract**:

The training problem of neural networks (NNs) is known to be ER-complete with respect to ReLU and linear activation functions. We show that the training problem for NNs equipped with arbitrary activation functions is polynomial-time bireducible to the existential theory of the reals extended with the corresponding activation functions. For effectively continuous activation functions (e.g., the sigmoid function), we obtain an inclusion to low levels of the arithmetical hierarchy. Consequently, the sigmoid activation function leads to the existential theory of the reals with the exponential function, and hence the decidability of training NNs using the sigmoid activation function is equivalent to the decidability of the existential theory of the reals with the exponential function, a long-standing open problem. In contrast, we obtain that the training problem is undecidable if sinusoidal activation functions are considered.

----

## [1370] Composite Active Learning: Towards Multi-Domain Active Learning with Theoretical Guarantees

**Authors**: *Guang-Yuan Hao, Hengguan Huang, Haotian Wang, Jie Gao, Hao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29119](https://doi.org/10.1609/aaai.v38i11.29119)

**Abstract**:

Active learning (AL) aims to improve model performance within a fixed labeling budget by choosing the most informative data points to label. Existing AL focuses on the single-domain setting, where all data come from the same domain (e.g., the same dataset). However, many real-world tasks often involve multiple domains. For example, in visual recognition, it is often desirable to train an image classifier that works across different environments (e.g., different backgrounds), where images from each environment constitute one domain. Such a multi-domain AL setting is challenging for prior methods because they (1) ignore the similarity among different domains when assigning labeling budget and (2) fail to handle distribution shift of data across different domains. In this paper, we propose the first general method, dubbed composite active learning (CAL), for multi-domain AL. Our approach explicitly considers the domain-level and instance-level information in the problem; CAL first assigns domain-level budgets according to domain-level importance, which is estimated by optimizing an upper error bound that we develop; with the domain-level budgets, CAL then leverages a certain instance-level query strategy to select samples to label from each domain. Our theoretical analysis shows that our method achieves a better error bound compared to current AL methods. Our empirical results demonstrate that our approach significantly outperforms the state-of-the-art AL methods on both synthetic and real-world multi-domain datasets. Code is available at https://github.com/Wang-ML-Lab/multi-domain-active-learning.

----

## [1371] Double-Layer Hybrid-Label Identification Feature Selection for Multi-View Multi-Label Learning

**Authors**: *Pingting Hao, Kunpeng Liu, Wanfu Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29120](https://doi.org/10.1609/aaai.v38i11.29120)

**Abstract**:

Multi-view multi-label feature selection aims to select informative features where the data are collected from multiple sources with multiple interdependent class labels. For fully exploiting multi-view information, most prior works mainly focus on the common part in the ideal circumstance. However, the inconsistent part hidden in each view, including noises and specific elements, may affect the quality of mapping between labels and feature representations. Meanwhile, ignoring the specific part might lead to a suboptimal result, as each label is supposed to possess specific characteristics of its own. To deal with the double problems in multi-view multi-label feature selection, we propose a unified loss function which is a totally splitting structure for observed labels as hybrid labels that is, common labels, view-to-all specific labels and noisy labels, and the view-to-all specific labels further splits into several specific labels of each view. The proposed method simultaneously considers the consistency and complementarity of different views. Through exploring the feature weights of hybrid labels, the mapping relationships between labels and features can be established sequentially based on their attributes. Additionally, the interrelatedness among hybrid labels is also investigated and injected into the function. Specific to the specific labels of each view, we construct the novel regularization paradigm incorporating logic operations. Finally, the convergence of the result is proved after applying the multiplicative update rules. Experiments on six datasets demonstrate the effectiveness and superiority of our method compared with the state-of-the-art methods.

----

## [1372] Multiagent Gumbel MuZero: Efficient Planning in Combinatorial Action Spaces

**Authors**: *Xiaotian Hao, Jianye Hao, Chenjun Xiao, Kai Li, Dong Li, Yan Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29121](https://doi.org/10.1609/aaai.v38i11.29121)

**Abstract**:

AlphaZero and MuZero have achieved state-of-the-art (SOTA) performance in a wide range of domains, including board games and robotics, with discrete and continuous action spaces. However, to obtain an improved policy, they often require an excessively large number of simulations, especially for domains with large action spaces. As the simulation budget decreases, their performance drops significantly. In addition, many important real-world applications have combinatorial (or exponential) action spaces, making it infeasible to search directly over all possible actions. In this paper, we extend AlphaZero and MuZero to learn and plan in more complex multiagent (MA) Markov decision processes, where the action spaces increase exponentially with the number of agents. Our new algorithms, MA Gumbel AlphaZero and MA Gumbel MuZero, respectively without and with model learning, achieve superior performance on cooperative multiagent control problems, while reducing the number of environmental interactions by up to an order of magnitude compared to model-free approaches. In particular, we significantly improve prior performance when planning with much fewer simulation budgets. The code and appendix are available at https://github.com/tjuHaoXiaotian/MA-MuZero.

----

## [1373] Calibrated One Round Federated Learning with Bayesian Inference in the Predictive Space

**Authors**: *Mohsin Hasan, Guojun Zhang, Kaiyang Guo, Xi Chen, Pascal Poupart*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29122](https://doi.org/10.1609/aaai.v38i11.29122)

**Abstract**:

Federated Learning (FL) involves training a model over a dataset distributed among clients, with the constraint that each client’s dataset is localized and possibly heterogeneous. In FL, small and noisy datasets are common, highlighting the need for well-calibrated models that represent the uncertainty of predictions. The closest FL techniques to achieving such goals are the Bayesian FL methods which collect parameter samples from local posteriors, and aggregate them to approximate the global posterior. To improve scalability for larger models, one common Bayesian approach is to approximate the global predictive posterior by multiplying local predictive posteriors. In this work, we demonstrate that this method gives systematically overconfident predictions, and we remedy this by proposing  β-Predictive Bayes, a Bayesian FL algorithm that interpolates between a mixture and product of the predictive posteriors, using a tunable parameter  β. This parameter is tuned to improve the global ensemble’s calibration, before it is distilled to a single model. Our method is evaluated on a variety of regression and classification datasets to demonstrate its superiority in calibration to other baselines, even as data heterogeneity increases. Code available at  https://github.com/hasanmohsin/betaPredBayesFL. Our paper's full version is at https://arxiv.org/abs/2312.09817.

----

## [1374] Selective Deep Autoencoder for Unsupervised Feature Selection

**Authors**: *Wael Hassanieh, Abdallah Chehade*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29123](https://doi.org/10.1609/aaai.v38i11.29123)

**Abstract**:

In light of the advances in big data, high-dimensional datasets are often encountered. Incorporating them into data-driven models can enhance performance; however, this comes at the cost of high computation and the risk of overfitting, particularly due to abundant redundant features. Identifying an informative subset of the features helps in reducing the dimensionality and enhancing model interpretability. In this paper, we propose a novel framework for unsupervised feature selection, called Selective Deep Auto-Encoder (SDAE). It aims to reduce the number of features used in unlabeled datasets without compromising the quality of information obtained. It achieves this by selecting sufficient features - from the original feature set - capable of representing the entire feature space and reconstructing them. Architecturally, it leverages the use of highly nonlinear latent representations in deep Autoencoders and intrinsically learns, in an unsupervised fashion, the relevant and globally representative subset of features through a customized Selective Layer. Extensive experimental results on three high-dimensional public datasets have shown promising feature selection performance by SDAE in comparison to other existing state-of-the-art unsupervised feature selection methods.

----

## [1375] Fairness under Covariate Shift: Improving Fairness-Accuracy Tradeoff with Few Unlabeled Test Samples

**Authors**: *Shreyas Havaldar, Jatin Chauhan, Karthikeyan Shanmugam, Jay Nandy, Aravindan Raghuveer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29124](https://doi.org/10.1609/aaai.v38i11.29124)

**Abstract**:

Covariate shift in the test data is a common practical phenomena that can significantly downgrade both the accuracy and the fairness performance of the model. Ensuring fairness across different sensitive groups under covariate shift is of paramount importance due to societal implications like criminal justice. We operate in the unsupervised regime where only a small set of unlabeled test samples along with a labeled training set is available. Towards improving fairness under this highly challenging yet realistic scenario, we make three contributions. First is a novel composite weighted entropy based objective for prediction accuracy which is optimized along with a representation matching loss for fairness. We experimentally verify that optimizing with our loss formulation outperforms a number of state-of-the-art baselines in the pareto sense with respect to the fairness-accuracy tradeoff on several standard datasets. Our second contribution is a new setting we term Asymmetric Covariate Shift that, to the best of our knowledge, has not been studied before. Asymmetric covariate shift occurs when distribution of covariates of one group shifts significantly compared to the other groups and this happens when a dominant group is over-represented. While this setting is extremely challenging for current baselines, We show that our proposed method significantly outperforms them. Our third contribution is theoretical, where we show that our weighted entropy term along with prediction loss on the training set approximates test loss under covariate shift. Empirically and through formal sample complexity bounds, we show that this approximation to the unseen test loss does not depend on importance sampling variance which affects many other baselines.

----

## [1376] A New Mechanism for Eliminating Implicit Conflict in Graph Contrastive Learning

**Authors**: *Dongxiao He, Jitao Zhao, Cuiying Huo, Yongqi Huang, Yuxiao Huang, Zhiyong Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29125](https://doi.org/10.1609/aaai.v38i11.29125)

**Abstract**:

Graph contrastive learning (GCL) has attracted considerable attention because it can self-supervisedly extract low-dimensional representation of graph data. InfoNCE-based loss function is widely used in graph contrastive learning, which pulls the representations of positive pairs close to each other and pulls the representations of negative pairs away from each other. Recent works mainly focus on designing new augmentation methods or sampling strategies. However, we argue that the widely used InfoNCE-based methods may contain an implicit conflict which seriously confuses models when learning from negative pairs. This conflict is engendered by the encoder's message-passing mechanism and the InfoNCE loss function. As a result, the learned representations between negative samples cannot be far away from each other, compromising the model performance. To our best knowledge, this is the first time to report and analysis this conflict of GCL. To address this problem, we propose a simple but effective method called Partial ignored Graph Contrastive Learning (PiGCL). Specifically, PiGCL first dynamically captures the conflicts during training by detecting the gradient of representation similarities. It then enables the loss function to ignore the conflict, allowing the encoder to adaptively learn the ignored information without self-supervised samples. Extensive experiments demonstrate the effectiveness of our method.

----

## [1377] Improving Distinguishability of Class for Graph Neural Networks

**Authors**: *Dongxiao He, Shuwei Liu, Meng Ge, Zhizhi Yu, Guangquan Xu, Zhiyong Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29126](https://doi.org/10.1609/aaai.v38i11.29126)

**Abstract**:

Graph Neural Networks (GNNs) have received widespread attention and applications due to their excellent performance in graph representation learning. Most existing GNNs can only aggregate 1-hop neighbors in a GNN layer, so they usually stack multiple GNN layers to obtain more information from larger neighborhoods. However, many studies have shown that model performance experiences a significant degradation with the increase of GNN layers. In this paper, we first introduce the concept of distinguishability of class to indirectly evaluate the learned node representations, and verify the positive correlation between distinguishability of class and model performance. Then, we propose a Graph Neural Network guided by Distinguishability of class (Disc-GNN) to monitor the representation learning, so as to learn better node representations and improve model performance. Specifically, we first perform inter-layer filtering and initial compensation based on Local Distinguishability of Class (LDC) in each layer, so that the learned node representations have the ability to distinguish different classes. Furthermore, we add a regularization term based on Global Distinguishability of Class (GDC) to achieve global optimization of model performance. Extensive experiments on six real-world datasets have shown that the competitive performance of Disc-GNN to the state-of-the-art methods on node classification and node clustering tasks.

----

## [1378] Decoupling Meta-Reinforcement Learning with Gaussian Task Contexts and Skills

**Authors**: *Hongcai He, Anjie Zhu, Shuang Liang, Feiyu Chen, Jie Shao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29127](https://doi.org/10.1609/aaai.v38i11.29127)

**Abstract**:

Offline meta-reinforcement learning (meta-RL) methods, which adapt to unseen target tasks with prior experience, are essential in robot control tasks. Current methods typically utilize task contexts and skills as prior experience, where task contexts are related to the information within each task and skills represent a set of temporally extended actions for solving subtasks. However, these methods still suffer from limited performance when adapting to unseen target tasks, mainly because the learned prior experience lacks generalization, i.e., they are unable to extract effective prior experience from meta-training tasks by exploration and learning of continuous latent spaces. We propose a framework called decoupled meta-reinforcement learning (DCMRL), which (1) contrastively restricts the learning of task contexts through pulling in similar task contexts within the same task and pushing away different task contexts of different tasks, and (2) utilizes a Gaussian quantization variational autoencoder (GQ-VAE) for clustering the Gaussian distributions of the task contexts and skills respectively, and decoupling the exploration and learning processes of their spaces. These cluster centers which serve as representative and discrete distributions of task context and skill are stored in task context codebook and skill codebook, respectively. DCMRL can acquire generalizable prior experience and achieve effective adaptation to unseen target tasks during the meta-testing phase. Experiments in the navigation and robot manipulation continuous control tasks show that DCMRL is more effective than previous meta-RL methods with more generalizable prior experience.

----

## [1379] IS-DARTS: Stabilizing DARTS through Precise Measurement on Candidate Importance

**Authors**: *Hongyi He, Longjun Liu, Haonan Zhang, Nanning Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29128](https://doi.org/10.1609/aaai.v38i11.29128)

**Abstract**:

Among existing Neural Architecture Search methods, DARTS is known for its efficiency and simplicity. This approach applies continuous relaxation of network representation to construct a weight-sharing supernet and enables the identification of excellent subnets in just a few GPU days. However, performance collapse in DARTS results in deteriorating architectures filled with parameter-free operations and remains a great challenge to the robustness. To resolve this problem, we reveal that the fundamental reason is the biased estimation of the candidate importance in the search space through theoretical and experimental analysis, and more precisely select operations via information-based measurements. Furthermore, we demonstrate that the excessive concern over the supernet and inefficient utilization of data in bi-level optimization also account for suboptimal results. We adopt a more realistic objective focusing on the performance of subnets and simplify it with the help of the informationbased measurements. Finally, we explain theoretically why progressively shrinking the width of the supernet is necessary and reduce the approximation error of optimal weights in DARTS. Our proposed method, named IS-DARTS, comprehensively improves DARTS and resolves the aforementioned problems. Extensive experiments on NAS-Bench-201 and DARTS-based search space demonstrate the effectiveness of IS-DARTS.

----

## [1380] Not All Tasks Are Equally Difficult: Multi-Task Deep Reinforcement Learning with Dynamic Depth Routing

**Authors**: *Jinmin He, Kai Li, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29129](https://doi.org/10.1609/aaai.v38i11.29129)

**Abstract**:

Multi-task reinforcement learning endeavors to accomplish a set of different tasks with a single policy. To enhance data efficiency by sharing parameters across multiple tasks, a common practice segments the network into distinct modules and trains a routing network to recombine these modules into task-specific policies. However, existing routing approaches employ a fixed number of modules for all tasks, neglecting that tasks with varying difficulties commonly require varying amounts of knowledge. This work presents a Dynamic Depth Routing (D2R) framework, which learns strategic skipping of certain intermediate modules, thereby flexibly choosing different numbers of modules for each task. Under this framework, we further introduce a ResRouting method to address the issue of disparate routing paths between behavior and target policies during off-policy training. In addition, we design an automatic route-balancing mechanism to encourage continued routing exploration for unmastered tasks without disturbing the routing of mastered ones. We conduct extensive experiments on various robotics manipulation tasks in the Meta-World benchmark, where D2R achieves state-of-the-art performance with significantly improved learning efficiency.

----

## [1381] Enhancing Semi-supervised Domain Adaptation via Effective Target Labeling

**Authors**: *Jiujun He, Bin Liu, Guosheng Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29130](https://doi.org/10.1609/aaai.v38i11.29130)

**Abstract**:

Existing semi-supervised domain adaptation (SSDA) models have exhibited impressive performance on the target domain by effectively utilizing few labeled target samples per class (e.g., 3 samples per class). To guarantee an equal number of labeled target samples for each class, however, they require domain experts to manually recognize a considerable amount of the unlabeled target data. Moreover, as the target samples are not equally informative for shaping the decision boundaries of the learning models, it is crucial to select the most informative target samples for labeling, which is, however, impossible for human selectors. As a remedy, we propose an EFfective Target Labeling (EFTL) framework that harnesses active learning and pseudo-labeling strategies to automatically select some informative target samples to annotate. Concretely, we introduce a novel sample query strategy, called non-maximal degree node suppression (NDNS), that iteratively performs maximal degree node query and non-maximal degree node removal to select representative and diverse target samples for labeling. To learn target-specific characteristics, we propose a novel pseudo-labeling strategy that attempts to label low-confidence target samples accurately via clustering consistency (CC), and then inject information of the model uncertainty into our query process. CC enhances the utilization of the annotation budget and increases the number of “labeled” target samples while requiring no additional manual effort. Our proposed EFTL framework can be easily coupled with existing SSDA models, showing significant improvements on three benchmarks

----

## [1382] Generative Calibration of Inaccurate Annotation for Label Distribution Learning

**Authors**: *Liang He, Yunan Lu, Weiwei Li, Xiuyi Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29131](https://doi.org/10.1609/aaai.v38i11.29131)

**Abstract**:

Label distribution learning (LDL) is an effective learning paradigm for handling label ambiguity. When applying LDL, it typically requires datasets annotated with label distributions. However, obtaining supervised data for LDL is a challenging task. Due to the randomness of label annotation, the annotator can produce inaccurate annotation results for the instance, affecting the accuracy and generalization ability of the LDL model. To address this problem, we propose a generative approach to calibrate the inaccurate annotation for LDL using variational inference techniques. Specifically, we assume that instances with similar features share latent similar label distributions. The feature vectors and label distributions are generated by Gaussian mixture and Dirichlet mixture, respectively. The relationship between them is established through a shared categorical variable, which effectively utilizes the label distribution of instances with similar features, and achieves a more accurate label distribution through the generative approach. Furthermore, we use a confusion matrix to model the factors that contribute to the inaccuracy during the annotation process, which captures the relationship between label distributions and inaccurate label distributions. Finally, the label distribution is used to calibrate the available information in the noisy dataset to obtain the ground-truth label distribution.

----

## [1383] Exploring Channel-Aware Typical Features for Out-of-Distribution Detection

**Authors**: *Rundong He, Yue Yuan, Zhongyi Han, Fan Wang, Wan Su, Yilong Yin, Tongliang Liu, Yongshun Gong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29132](https://doi.org/10.1609/aaai.v38i11.29132)

**Abstract**:

Detecting out-of-distribution (OOD) data is essential to ensure the reliability of machine learning models when deployed in real-world scenarios. Different from most previous test-time OOD detection methods that focus on designing OOD scores, we delve into the challenges in OOD detection from the perspective of typicality and regard the feature’s high-probability region as the feature’s typical set. However, the existing typical-feature-based OOD detection method implies an assumption: the proportion of typical feature sets for each channel is fixed. According to our experimental analysis, each channel contributes differently to OOD detection. Adopting a fixed proportion for all channels results in several channels losing too many typical features or incorporating too many abnormal features, resulting in low performance. Therefore, exploring the channel-aware typical features is crucial to better-separating ID and OOD data. Driven by this insight, we propose expLoring channel-Aware tyPical featureS (LAPS). Firstly, LAPS obtains the channel-aware typical set by calibrating the channel-level typical set with the global typical set from the mean and standard deviation. Then, LAPS rectifies the features into channel-aware typical sets to obtain channel-aware typical features. Finally, LAPS leverages the channel-aware typical features to calculate the energy score for OOD detection. Theoretical and visual analyses verify that LAPS achieves a better bias-variance trade-off. Experiments verify the effectiveness and generalization of LAPS under different architectures and OOD scores.

----

## [1384] Learning Only When It Matters: Cost-Aware Long-Tailed Classification

**Authors**: *Yu-Cheng He, Yao-Xiang Ding, Han-Jia Ye, Zhi-Hua Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29133](https://doi.org/10.1609/aaai.v38i11.29133)

**Abstract**:

Most current long-tailed classification approaches assume the cost-agnostic scenario, where the training distribution of classes is long-tailed while the testing distribution of classes is balanced. Meanwhile, the misclassification costs of all instances are the same. On the other hand, in many real-world applications, it is more proper to assume that the training and testing distributions of classes are the same, while the misclassification cost of tail-class instances is varied. In this work, we model such a scenario as cost-aware long-tailed classification, in which the identification of high-cost tail instances and focusing learning on them thereafter is essential. In consequence, we propose the learning strategy of augmenting new instances based on adaptive region partition in the feature space. We conduct theoretical analysis to show that under the assumption
that the feature-space distance and the misclassification cost are correlated, the identification of high-cost tail instances can be realized by building region partitions with a low variance of risk within each region. The resulting AugARP approach could significantly outperform baseline approaches on both benchmark datasets and real-world product sales datasets.

----

## [1385] SoundCount: Sound Counting from Raw Audio with Dyadic Decomposition Neural Network

**Authors**: *Yuhang He, Zhuangzhuang Dai, Niki Trigoni, Long Chen, Andrew Markham*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29134](https://doi.org/10.1609/aaai.v38i11.29134)

**Abstract**:

In this paper, we study an underexplored, yet important and challenging problem: counting the number of distinct sounds in raw audio characterized by a high degree of polyphonicity. We do so by systematically proposing a novel end-to-end trainable neural network~(which we call DyDecNet, consisting of a dyadic decomposition front-end and backbone network), and quantifying the difficulty level of counting depending on sound polyphonicity. The dyadic decomposition front-end progressively decomposes the raw waveform dyadically along the frequency axis to obtain time-frequency representation in multi-stage, coarse-to-fine manner. Each intermediate waveform convolved by a parent filter is further processed by a pair of child filters that evenly split the parent filter's carried frequency response, with the higher-half child filter encoding the detail and lower-half child filter encoding the approximation. We further introduce an energy gain normalization to normalize sound loudness variance and spectrum overlap, and apply it to each intermediate parent waveform before feeding it to the two child filters. To better quantify sound counting difficulty level, we further design three polyphony-aware metrics: polyphony ratio, max polyphony and mean polyphony. We test DyDecNet on various datasets to show its superiority, and we further show dyadic decomposition network can be used as a general front-end to tackle other acoustic tasks.

----

## [1386] Training-Free Quantum Architecture Search

**Authors**: *Zhimin He, Maijie Deng, Shenggen Zheng, Lvzhou Li, Haozhen Situ*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29135](https://doi.org/10.1609/aaai.v38i11.29135)

**Abstract**:

Variational quantum algorithm (VQA) derives advantages from its error resilience and high flexibility in quantum resource requirements, rendering it broadly applicable in the noisy intermediate-scale quantum era. As the performance of VQA highly relies on the structure of the parameterized quantum circuit, it is worthwhile to propose quantum architecture search (QAS) algorithms to automatically search for high-performance circuits. Nevertheless, existing QAS methods are time-consuming, requiring circuit training to assess circuit performance. This study pioneers training-free QAS by utilizing two training-free proxies to rank quantum circuits, in place of the expensive circuit training employed in conventional QAS. Taking into account the precision and computational overhead of the path-based and expressibility-based proxies, we devise a two-stage progressive training-free QAS (TF-QAS). Initially, directed acyclic graphs (DAGs) are employed for circuit representation, and a zero-cost proxy based on the number of paths in the DAG is designed to filter out a substantial portion of unpromising circuits. Subsequently, an expressibility-based proxy, finely reflecting circuit performance, is employed to identify high-performance circuits from the remaining candidates. These proxies evaluate circuit performance without circuit training, resulting in a remarkable reduction in computational cost compared to current training-based QAS methods. Simulations on three VQE tasks demonstrate that TF-QAS achieves a substantial enhancement of sampling efficiency ranging from 5 to 57 times compared to state-of-the-art QAS, while also being 6 to 17 times faster.

----

## [1387] Imitate the Good and Avoid the Bad: An Incremental Approach to Safe Reinforcement Learning

**Authors**: *Huy Hoang, Tien Mai, Pradeep Varakantham*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29136](https://doi.org/10.1609/aaai.v38i11.29136)

**Abstract**:

A popular framework for enforcing safe actions in Reinforcement Learning (RL) is Constrained RL, where trajectory based constraints on expected cost (or other cost measures) are employed to enforce safety and more importantly these constraints are enforced while maximizing expected reward. Most recent approaches for solving Constrained RL convert the trajectory based cost constraint into a surrogate problem that can be solved using minor modifications to RL methods. A key drawback with such approaches is an over or underestimation of the cost constraint at each state.  Therefore, we provide an approach that does not modify the trajectory based cost constraint and instead imitates "good" trajectories and avoids "bad" trajectories generated from incrementally improving policies. We employ an oracle that utilizes a reward threshold (which is varied with learning) and the overall cost constraint to label trajectories as "good" or "bad". A key advantage of our approach is that we are able to work from any starting policy or set of trajectories and improve on it. In an exhaustive set of experiments, we demonstrate that our approach is able to outperform top benchmark approaches for solving Constrained RL problems, with respect to expected cost, CVaR cost, or even unknown cost constraints.

----

## [1388] Few-Shot Learning via Repurposing Ensemble of Black-Box Models

**Authors**: *Minh Hoang, Trong Nghia Hoang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29137](https://doi.org/10.1609/aaai.v38i11.29137)

**Abstract**:

This paper investigates the problem of exploiting existing solution models of previous tasks to address a related target task with limited training data. Existing approaches addressing this problem often require access to the internal parameterization of the existing solution models and possibly their training data, which is not possible in many practical settings. To relax this requirement, We approach this problem from a new perspective of black-box re-purposing, which augments the target inputs and leverages their corresponding outputs generated by existing black-box APIs into a feature ensemble. We hypothesize that such feature ensemble can be learned to incorporate and encode relevant black-box knowledge into the feature representation of target data, which will compensate for their scarcity. This hypothesis is confirmed via the reported successes of our proposed black-box ensemble in solving multiple few-shot learning tasks derived from various benchmark datasets. All reported results show consistently that the set of heterogeneous black-box solutions of previous tasks can indeed be reused and combined effectively to solve a reasonably related target task without requiring access to a large training dataset. This is the first step towards enabling new possibilities to further supplement existing techniques in transfer or meta learning with black-box knowledge.

----

## [1389] Transitivity-Preserving Graph Representation Learning for Bridging Local Connectivity and Role-Based Similarity

**Authors**: *Van Thuy Hoang, O-Joun Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29138](https://doi.org/10.1609/aaai.v38i11.29138)

**Abstract**:

Graph representation learning (GRL) methods, such as graph neural networks and graph transformer models, have been successfully used to analyze graph-structured data, mainly focusing on node classification and link prediction tasks. However, the existing studies mostly only consider local connectivity while ignoring long-range connectivity and the roles of nodes. In this paper, we propose Unified Graph Transformer Networks (UGT) that effectively integrate local and global structural information into fixed-length vector representations. First, UGT learns local structure by identifying the local sub-structures and aggregating features of the k-hop neighborhoods of each node. Second, we construct virtual edges, bridging distant nodes with structural similarity to capture the long-range dependencies. Third, UGT learns unified representations through self-attention, encoding structural distance and p-step transition probability between node pairs. Furthermore, we propose a self-supervised learning task that effectively learns transition probability to fuse local and global structural features, which could then be transferred to other downstream tasks. Experimental results on real-world benchmark datasets over various downstream tasks showed that UGT significantly outperformed baselines that consist of state-of-the-art models. In addition, UGT reaches the third-order Weisfeiler-Lehman power to distinguish non-isomorphic graph pairs.

----

## [1390] Colored Noise in PPO: Improved Exploration and Performance through Correlated Action Sampling

**Authors**: *Jakob J. Hollenstein, Georg Martius, Justus H. Piater*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29139](https://doi.org/10.1609/aaai.v38i11.29139)

**Abstract**:

Proximal Policy Optimization (PPO), a popular on-policy deep reinforcement learning method, employs a stochastic policy for exploration. In this paper, we propose a colored noise-based stochastic policy variant of PPO. Previous research highlighted the importance of temporal correlation in action noise for effective exploration in off-policy reinforcement learning. Building on this, we investigate whether correlated noise can also enhance exploration in on-policy methods like PPO. We discovered that correlated noise for action selection improves learning performance and outperforms the currently popular uncorrelated white noise approach in on-policy methods. Unlike off-policy learning, where pink noise was found to be highly effective, we found that a colored noise, intermediate between white and pink, performed best for on-policy learning in PPO. We examined the impact of varying the amount of data collected for each update by modifying the number of parallel simulation environments for data collection and observed that with a larger number of parallel environments, more strongly correlated noise is beneficial. Due to the significant impact and ease of implementation, we recommend switching to correlated noise as the default noise source in PPO.

----

## [1391] Foreseeing Reconstruction Quality of Gradient Inversion: An Optimization Perspective

**Authors**: *Hyeong Gwon Hong, Yooshin Cho, Hanbyel Cho, Jaesung Ahn, Junmo Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29140](https://doi.org/10.1609/aaai.v38i11.29140)

**Abstract**:

Gradient inversion attacks can leak data privacy when clients share weight updates with the server in federated learning (FL). Existing studies mainly use L2 or cosine distance as the loss function for gradient matching in the attack. Our empirical investigation shows that the vulnerability ranking varies with the loss function used. Gradient norm, which is commonly used as a vulnerability proxy for gradient inversion attack, cannot explain this as it remains constant regardless of the loss function for gradient matching. In this paper, we propose a loss-aware vulnerability proxy (LAVP) for the first time. LAVP refers to either the maximum or minimum eigenvalue of the Hessian with respect to gradient matching loss at ground truth. This suggestion is based on our theoretical findings regarding the local optimization of the gradient inversion in proximity to the ground truth, which corresponds to the worst case attack scenario. We demonstrate the effectiveness of LAVP on various architectures and datasets, showing its consistent superiority over the gradient norm in capturing sample vulnerabilities. The performance of each proxy is measured in terms of Spearman's rank correlation with respect to several similarity scores. This work will contribute to enhancing FL security against any potential loss functions beyond L2 or cosine distance in the future.

----

## [1392] Complete Neural Networks for Complete Euclidean Graphs

**Authors**: *Snir Hordan, Tal Amir, Steven J. Gortler, Nadav Dym*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29141](https://doi.org/10.1609/aaai.v38i11.29141)

**Abstract**:

Neural networks for point clouds, which respect their natural invariance to permutation and rigid motion, have enjoyed recent success in modeling geometric phenomena, from molecular dynamics to recommender systems. Yet, to date, no architecture with polynomial complexity is known to be complete, that is, able to distinguish between any pair of non-isomorphic point clouds. We fill this theoretical gap by showing that point clouds can be completely determined, up to permutation and rigid motion, by applying the 3-WL graph isomorphism test to the point cloud's centralized Gram matrix. Moreover, we formulate an Euclidean variant of the 2-WL test and show that it is also sufficient to achieve completeness. We then show how our complete Euclidean WL tests can be simulated by an Euclidean graph neural network of moderate size and demonstrate their separation capability on highly symmetrical point clouds.

----

## [1393] Structured Probabilistic Coding

**Authors**: *Dou Hu, Lingwei Wei, Yaxin Liu, Wei Zhou, Songlin Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29142](https://doi.org/10.1609/aaai.v38i11.29142)

**Abstract**:

This paper presents a new supervised representation learning framework, namely structured probabilistic coding (SPC), to learn compact and informative representations from input related to the target task. SPC is an encoder-only probabilistic coding technology with a structured regularization from the target space. It can enhance the generalization ability of pre-trained language models for better language understanding. Specifically, our probabilistic coding simultaneously performs information encoding and task prediction in one module to more fully utilize the effective information from input data. It uses variational inference in the output space to reduce randomness and uncertainty. Besides, to better control the learning process of probabilistic representations, a structured regularization is proposed to promote uniformity across classes in the latent space. With the regularization term, SPC can preserve the Gaussian structure of the latent code and achieve better coverage of the hidden space with class uniformly. Experimental results on 12 natural language understanding tasks demonstrate that our SPC effectively improves the performance of pre-trained language models for classification and regression. Extensive experiments show that SPC can enhance the generalization capability, robustness to label noise, and clustering quality of output representations.

----

## [1394] A Sequentially Fair Mechanism for Multiple Sensitive Attributes

**Authors**: *François Hu, Philipp Ratz, Arthur Charpentier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29143](https://doi.org/10.1609/aaai.v38i11.29143)

**Abstract**:

In the standard use case of Algorithmic Fairness, the goal is to eliminate the relationship between a sensitive variable and a corresponding score. Throughout recent years, the scientific community has developed a host of definitions and tools to solve this task, which work well in many practical applications. However, the applicability and effectivity of these tools and definitions becomes less straightfoward in the case of multiple sensitive attributes. To tackle this issue, we propose a sequential framework, which allows to progressively achieve fairness across a set of sensitive features. We accomplish this by leveraging multi-marginal Wasserstein barycenters, which extends the standard notion of Strong Demographic Parity to the case with multiple sensitive characteristics. This method also provides a closed-form solution for the optimal, sequentially fair predictor, permitting a clear interpretation of inter-sensitive feature correlations. Our approach seamlessly extends to approximate fairness, enveloping a framework accommodating the trade-off between risk and unfairness. This extension permits a targeted prioritization of fairness improvements for a specific attribute within a set of sensitive attributes, allowing for a case specific adaptation. A data-driven estimation procedure for the derived solution is developed, and comprehensive numerical experiments are conducted on both synthetic and real datasets. Our empirical findings decisively underscore the practical efficacy of our post-processing approach in fostering fair decision-making.

----

## [1395] Relax Image-Specific Prompt Requirement in SAM: A Single Generic Prompt for Segmenting Camouflaged Objects

**Authors**: *Jian Hu, Jiayi Lin, Shaogang Gong, Weitong Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29144](https://doi.org/10.1609/aaai.v38i11.29144)

**Abstract**:

Camouflaged object detection (COD) approaches heavily rely on pixel-level annotated datasets.  Weakly-supervised COD (WSCOD) approaches use sparse annotations like scribbles or points to reduce annotation efforts, but this can lead to decreased accuracy.  The Segment Anything Model (SAM) shows remarkable segmentation ability with sparse prompts like points. However, manual prompt is not always feasible, as it may not be accessible in real-world application. Additionally, it only provides localization information instead of semantic one, which can intrinsically cause ambiguity in interpreting targets. In this work, we aim to eliminate the need for manual prompt. The key idea is to employ Cross-modal Chains of Thought Prompting (CCTP) to reason visual prompts using the semantic information given by a generic text prompt. To that end, we introduce a test-time instance-wise adaptation mechanism called Generalizable SAM (GenSAM) to automatically generate and optimize visual prompts from the generic task prompt for WSCOD. In particular, CCTP maps a single generic text prompt onto image-specific consensus foreground and background heatmaps using vision-language models,  acquiring reliable visual prompts. Moreover, to test-time adapt the visual prompts, we further propose Progressive Mask Generation (PMG) to iteratively reweight the input image, guiding the model to focus on the targeted region in a coarse-to-fine manner. Crucially, all network parameters are fixed, avoiding the need for additional training. Experiments on three benchmarks demonstrate that GenSAM outperforms point supervision approaches and achieves comparable results to scribble supervision ones, solely relying on general task descriptions. Our codes is in https://github.com/jyLin8100/GenSAM.

----

## [1396] Sequential Fusion Based Multi-Granularity Consistency for Space-Time Transformer Tracking

**Authors**: *Kun Hu, Wenjing Yang, Wanrong Huang, Xianchen Zhou, Mingyu Cao, Jing Ren, Huibin Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29145](https://doi.org/10.1609/aaai.v38i11.29145)

**Abstract**:

Regarded as a template-matching task for a long time, visual object tracking has witnessed significant progress in space-wise exploration. However, since tracking is performed on videos with substantial time-wise information, it is important to simultaneously mine the temporal contexts which have not yet been deeply explored.  Previous supervised works mostly consider template reform as the breakthrough point, but they are often limited by additional computational burdens or the quality of chosen templates. To address this issue, we propose a Space-Time Consistent Transformer Tracker (STCFormer), which uses a sequential fusion framework with multi-granularity consistency constraints to learn spatiotemporal context information. We design a sequential fusion framework that recombines template and search images based on tracking results from chronological frames, fusing updated tracking states in training. To further overcome the over-reliance on the fixed template without increasing computational complexity, we design three space-time consistent constraints: Label Consistency Loss (LCL) for label-level consistency, Attention Consistency Loss (ACL) for patch-level ROI consistency, and Semantic Consistency Loss (SCL) for feature-level semantic consistency. Specifically, in ACL and SCL, the label information is used to constrain the attention and feature consistency of the target and the background, respectively, to avoid mutual interference. Extensive experiments have shown that our STCFormer outperforms many of the best-performing trackers on several popular benchmarks.

----

## [1397] FedMut: Generalized Federated Learning via Stochastic Mutation

**Authors**: *Ming Hu, Yue Cao, Anran Li, Zhiming Li, Chengwei Liu, Tianlin Li, Mingsong Chen, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29146](https://doi.org/10.1609/aaai.v38i11.29146)

**Abstract**:

Although Federated Learning (FL) enables collaborative model training without sharing the raw data of clients, it encounters low-performance problems caused by various heterogeneous scenarios. Due to the limitation of dispatching the same global model to clients for local training, traditional Federated Average (FedAvg)-based FL models face the problem of easily getting stuck into a sharp solution, which results in training a low-performance global model. To address this problem, this paper presents a novel FL approach named FedMut, which mutates the global model according to the gradient change to generate several intermediate models for the next round of training. Each intermediate model will be dispatched to a client for local training. Eventually, the global model converges into a flat area within the range of mutated models and has a well-generalization compared with the global model trained by FedAvg. Experimental results on well-known datasets demonstrate the effectiveness of our FedMut approach in various data heterogeneity scenarios.

----

## [1398] PrefAce: Face-Centric Pretraining with Self-Structure Aware Distillation

**Authors**: *Siyuan Hu, Zheng Wang, Peng Hu, Xi Peng, Jie Wu, Hongyuan Zhu, Yew Soon Ong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29147](https://doi.org/10.1609/aaai.v38i11.29147)

**Abstract**:

Video-based facial analysis is important for autonomous agents to understand human expressions and sentiments. However, limited labeled data is available to learn effective facial representations. This paper proposes a novel self-supervised face-centric pretraining framework, called PrefAce, which learns transferable video facial representation without labels. The self-supervised learning is performed with an effective landmark-guided global-local tube distillation. Meanwhile, a novel instance-wise update FaceFeat Cache is built to enforce more discriminative and diverse representations for downstream tasks. Extensive experiments demonstrate that the proposed framework learns universal instance-aware facial representations with fine-grained landmark details from videos. The point is that it can transfer across various facial analysis tasks, e.g., Facial Attribute Recognition (FAR), Facial Expression Recognition (FER), DeepFake Detection (DFD), and Lip Synchronization (LS). Our framework also outperforms the state-of-the-art on various downstream tasks, even in low data regimes. Code is available at https://github.com/siyuan-h/PrefAce.

----

## [1399] PA2D-MORL: Pareto Ascent Directional Decomposition Based Multi-Objective Reinforcement Learning

**Authors**: *Tianmeng Hu, Biao Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i11.29148](https://doi.org/10.1609/aaai.v38i11.29148)

**Abstract**:

Multi-objective reinforcement learning (MORL) provides an effective solution for decision-making problems involving conflicting objectives. However, achieving high-quality approximations to the Pareto policy set remains challenging, especially in complex tasks with continuous or high-dimensional state-action space. In this paper, we propose the Pareto Ascent Directional Decomposition based Multi-Objective Reinforcement Learning (PA2D-MORL) method, which constructs an efficient scheme for multi-objective problem decomposition and policy improvement, leading to a superior approximation of Pareto policy set. The proposed method leverages Pareto ascent direction to select the scalarization weights and computes the multi-objective policy gradient, which determines the policy optimization direction and ensures joint improvement on all objectives. Meanwhile, multiple policies are selectively optimized under an evolutionary framework to approximate the Pareto frontier from different directions. Additionally, a Pareto adaptive fine-tuning approach is applied to enhance the density and spread of the Pareto frontier approximation. Experiments on various multi-objective robot control tasks show that the proposed method clearly outperforms the current state-of-the-art algorithm in terms of both quality and stability of the outcomes.

----



[Go to the previous page](AAAI-2024-list06.md)

[Go to the next page](AAAI-2024-list08.md)

[Go to the catalog section](README.md)