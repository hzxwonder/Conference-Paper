## [800] HyperDynamics: Meta-Learning Object and Agent Dynamics with Hypernetworks

**Authors**: *Zhou Xian, Shamit Lal, Hsiao-Yu Tung, Emmanouil Antonios Platanios, Katerina Fragkiadaki*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pHXfe1cOmA](https://openreview.net/forum?id=pHXfe1cOmA)

**Abstract**:

We propose HyperDynamics, a dynamics meta-learning framework that conditions on an agent’s interactions with the environment and optionally its visual observations, and generates the parameters of neural dynamics models based on inferred properties of the dynamical system. Physical and visual properties of the environment that are not part of the low-dimensional state yet affect its temporal dynamics are inferred from the interaction history and visual observations, and are implicitly captured in the generated parameters. We test HyperDynamics on a set of object pushing and locomotion tasks. It outperforms existing dynamics models in the literature that adapt to environment variations by learning dynamics over high dimensional visual observations, capturing the interactions of the agent in recurrent state representations, or using gradient-based meta-optimization. We also show our method matches the performance of an ensemble of separately trained experts, while also being able to generalize well to unseen environment variations at test time. We attribute its good performance to the multiplicative interactions between the inferred system properties—captured in the generated parameters—and the low-dimensional state representation of the dynamical system.

----

## [801] Improving Zero-Shot Voice Style Transfer via Disentangled Representation Learning

**Authors**: *Siyang Yuan, Pengyu Cheng, Ruiyi Zhang, Weituo Hao, Zhe Gan, Lawrence Carin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TgSVWXw22FQ](https://openreview.net/forum?id=TgSVWXw22FQ)

**Abstract**:

Voice style transfer, also called voice conversion, seeks to modify one speaker's voice to generate speech as if it came from another (target) speaker. Previous works have made progress on voice conversion with parallel training data and pre-known speakers. However, zero-shot voice style transfer, which learns from non-parallel data and generates voices for previously unseen speakers, remains a challenging problem. In this paper we propose a novel zero-shot voice transfer method via disentangled representation learning. The proposed method first encodes speaker-related style and voice content of each input voice into separate low-dimensional embedding spaces, and then transfers to a new voice by combining the source content embedding and target style embedding through a decoder. With information-theoretic guidance, the style and content embedding spaces are representative and (ideally) independent of each other. On real-world datasets, our method outperforms other baselines and obtains state-of-the-art results in terms of transfer accuracy and voice naturalness.

----

## [802] GraPPa: Grammar-Augmented Pre-Training for Table Semantic Parsing

**Authors**: *Tao Yu, Chien-Sheng Wu, Xi Victoria Lin, Bailin Wang, Yi Chern Tan, Xinyi Yang, Dragomir R. Radev, Richard Socher, Caiming Xiong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kyaIeYj4zZ](https://openreview.net/forum?id=kyaIeYj4zZ)

**Abstract**:

We present GraPPa, an effective pre-training approach for table semantic parsing that learns a compositional inductive bias in the joint representations of textual and tabular data. We construct synthetic question-SQL pairs over high-quality tables via a synchronous context-free grammar (SCFG). We pre-train our model on the synthetic data to inject important structural properties commonly found in semantic parsing into the pre-training language model. To maintain the model's ability to represent real-world data, we also include masked language modeling (MLM) on several existing table-related datasets to regularize our pre-training process.  Our proposed pre-training strategy is much data-efficient. When incorporated with strong base semantic parsers, GraPPa achieves new state-of-the-art results on four popular fully supervised and weakly supervised table semantic parsing tasks.

----

## [803] Estimating Lipschitz constants of monotone deep equilibrium models

**Authors**: *Chirag Pabbaraju, Ezra Winston, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VcB4QkSfyO](https://openreview.net/forum?id=VcB4QkSfyO)

**Abstract**:

Several methods have been proposed in recent years to provide bounds on the Lipschitz constants of deep networks, which can be used to provide robustness guarantees, generalization bounds, and characterize the smoothness of decision boundaries. However, existing bounds get substantially weaker with increasing depth of the network, which makes it unclear how to apply such bounds to recently proposed models such as the deep equilibrium (DEQ) model, which can be viewed as representing an infinitely-deep network. In this paper, we show that monotone DEQs, a recently-proposed subclass of DEQs, have Lipschitz constants that can be bounded as a simple function of the strong monotonicity parameter of the network. We derive simple-yet-tight bounds on both the input-output mapping and the weight-output mapping defined by these networks, and demonstrate that they are small relative to those for comparable standard DNNs. We show that one can use these bounds to design monotone DEQ models, even with e.g. multi-scale convolutional structure, that still have constraints on the Lipschitz constant. We also highlight how to use these bounds to develop PAC-Bayes generalization bounds that do not depend on any depth of the network, and which avoid the exponential depth-dependence of comparable DNN bounds.

----

## [804] Estimating informativeness of samples with Smooth Unique Information

**Authors**: *Hrayr Harutyunyan, Alessandro Achille, Giovanni Paolini, Orchid Majumder, Avinash Ravichandran, Rahul Bhotika, Stefano Soatto*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kEnBH98BGs5](https://openreview.net/forum?id=kEnBH98BGs5)

**Abstract**:

We define a notion of information that an individual sample provides to the training of a neural network, and we specialize it to measure both how much a sample informs the final weights and how much it informs the function computed by the weights. Though related, we show that these quantities have a  qualitatively different behavior. We give efficient approximations of these quantities using a linearized network and demonstrate empirically that the approximation is accurate for real-world architectures, such as pre-trained ResNets. We apply these measures to several problems, such as dataset summarization, analysis of under-sampled classes, comparison of informativeness of different data sources, and detection of adversarial and corrupted examples. Our work generalizes existing frameworks, but enjoys better computational properties for heavily over-parametrized models, which makes it possible to apply it to real-world networks.

----

## [805] NBDT: Neural-Backed Decision Tree

**Authors**: *Alvin Wan, Lisa Dunlap, Daniel Ho, Jihan Yin, Scott Lee, Suzanne Petryk, Sarah Adel Bargal, Joseph E. Gonzalez*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mCLVeEpplNE](https://openreview.net/forum?id=mCLVeEpplNE)

**Abstract**:

Machine learning applications such as finance and medicine demand accurate and justifiable predictions, barring most deep learning methods from use. In response, previous work combines decision trees with deep learning, yielding models that (1) sacrifice interpretability for accuracy or (2) sacrifice accuracy for interpretability. We forgo this dilemma by jointly improving accuracy and interpretability using Neural-Backed Decision Trees (NBDTs). NBDTs replace a neural network's final linear layer with a differentiable sequence of decisions and a surrogate loss. This forces the model to learn high-level concepts and lessens reliance on highly-uncertain decisions, yielding (1) accuracy: NBDTs match or outperform modern neural networks on CIFAR, ImageNet and better generalize to unseen classes by up to 16%. Furthermore, our surrogate loss improves the original model's accuracy by up to 2%. NBDTs also afford (2) interpretability: improving human trustby clearly identifying model mistakes and assisting in dataset debugging. Code and pretrained NBDTs are at https://github.com/alvinwan/neural-backed-decision-trees.

----

## [806] Accurate Learning of Graph Representations with Graph Multiset Pooling

**Authors**: *Jinheon Baek, Minki Kang, Sung Ju Hwang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JHcqXGaqiGn](https://openreview.net/forum?id=JHcqXGaqiGn)

**Abstract**:

Graph neural networks have been widely used on modeling graph data, achieving impressive results on node classification and link prediction tasks. Yet, obtaining an accurate representation for a graph further requires a pooling function that maps a set of node representations into a compact form. A simple sum or average over all node representations considers all node features equally without consideration of their task relevance, and any structural dependencies among them. Recently proposed hierarchical graph pooling methods, on the other hand, may yield the same representation for two different graphs that are distinguished by the Weisfeiler-Lehman test, as they suboptimally preserve information from the node features. To tackle these limitations of existing graph pooling methods, we first formulate the graph pooling problem as a multiset encoding problem with auxiliary information about the graph structure, and propose a Graph Multiset Transformer (GMT) which is a multi-head attention based global pooling layer that captures the interaction between nodes according to their structural dependencies. We show that GMT satisfies both injectiveness and permutation invariance, such that it is at most as powerful as the Weisfeiler-Lehman graph isomorphism test. Moreover, our methods can be easily extended to the previous node clustering approaches for hierarchical graph pooling. Our experimental results show that GMT significantly outperforms state-of-the-art graph pooling methods on graph classification benchmarks with high memory and time efficiency, and obtains even larger performance gain on graph reconstruction and generation tasks.

----

## [807] Byzantine-Resilient Non-Convex Stochastic Gradient Descent

**Authors**: *Zeyuan Allen-Zhu, Faeze Ebrahimianghazani, Jerry Li, Dan Alistarh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PbEHqvFtcS](https://openreview.net/forum?id=PbEHqvFtcS)

**Abstract**:

We study adversary-resilient stochastic distributed optimization, in which $m$ machines can independently compute stochastic gradients, and cooperate to jointly optimize over their local objective functions. However, an $\alpha$-fraction of the machines are Byzantine, in that they may behave in arbitrary, adversarial ways. We consider a variant of this procedure in the challenging non-convex case. Our main result is a new algorithm SafeguardSGD, which can provably escape saddle points and find approximate local minima of the non-convex objective. The algorithm is based on a new concentration filtering technique, and its sample and time complexity bounds match the best known theoretical bounds in the stochastic, distributed setting when no Byzantine machines are present. 

Our algorithm is very practical: it improves upon the performance of all prior methods when training deep neural networks, it is relatively lightweight, and it is the first method to withstand two recently-proposed Byzantine attacks.

----

## [808] MetaNorm: Learning to Normalize Few-Shot Batches Across Domains

**Authors**: *Ying-Jun Du, Xiantong Zhen, Ling Shao, Cees G. M. Snoek*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9z_dNsC4B5t](https://openreview.net/forum?id=9z_dNsC4B5t)

**Abstract**:

Batch normalization plays a crucial role when training deep neural networks. However, batch statistics become unstable with small batch sizes and are unreliable in the presence of distribution shifts. We propose MetaNorm, a simple yet effective meta-learning normalization. It tackles the aforementioned issues in a unified way by leveraging the meta-learning setting and learns to infer adaptive statistics for batch normalization. MetaNorm is generic, flexible and model-agnostic, making it a simple plug-and-play module that is seamlessly embedded into existing meta-learning approaches. It can be efficiently implemented by lightweight hypernetworks with low computational cost. We verify its effectiveness by extensive evaluation on representative tasks suffering from the small batch and domain shift problems: few-shot learning and domain generalization. We further introduce an even more challenging setting: few-shot domain generalization. Results demonstrate that MetaNorm consistently achieves better, or at least competitive, accuracy compared to existing batch normalization methods.

----

## [809] Large Batch Simulation for Deep Reinforcement Learning

**Authors**: *Brennan Shacklett, Erik Wijmans, Aleksei Petrenko, Manolis Savva, Dhruv Batra, Vladlen Koltun, Kayvon Fatahalian*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=cP5IcoAkfKa](https://openreview.net/forum?id=cP5IcoAkfKa)

**Abstract**:

We accelerate deep reinforcement learning-based training in visually complex 3D environments by two orders of magnitude over prior work, realizing end-to-end training speeds of over 19,000 frames of experience per second on a single GPU and up to 72,000 frames per second on a single eight-GPU machine. The key idea of our approach is to design a 3D renderer and embodied navigation simulator around the principle of “batch simulation”: accepting and executing large batches of requests simultaneously.  Beyond exposing large amounts of work at once, batch simulation allows implementations to amortize in-memory storage of scene assets, rendering work, data loading, and synchronization costs across many simulation requests, dramatically improving the number of simulated agents per GPU and overall simulation throughput.  To balance DNN inference and training costs with faster simulation, we also build a computationally efficient policy DNN that maintains high task performance, and modify training algorithms to maintain sample efficiency when training with large mini-batches. By combining batch simulation and DNN performance optimizations, we demonstrate that PointGoal navigation agents can be trained in complex 3D environments on a single GPU in 1.5 days to 97% of the accuracy of agents trained on a prior state-of-the-art system using a 64-GPU cluster over three days.  We provide open-source reference implementations of our batch 3D renderer and simulator to facilitate incorporation of these ideas into RL systems.

----

## [810] Personalized Federated Learning with First Order Model Optimization

**Authors**: *Michael Zhang, Karan Sapra, Sanja Fidler, Serena Yeung, José M. Álvarez*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ehJqJQk9cw](https://openreview.net/forum?id=ehJqJQk9cw)

**Abstract**:

While federated learning traditionally aims to train a single global model across decentralized local datasets, one model may not always be ideal for all participating clients. Here we propose an alternative, where each client only federates with other relevant clients to obtain a stronger model per client-specific objectives. To achieve this personalization, rather than computing a single model average with constant weights for the entire federation as in traditional FL, we efficiently calculate optimal weighted model combinations for each client, based on figuring out how much a client can benefit from another's model. We do not assume knowledge of any underlying data distributions or client similarities, and allow each client to optimize for arbitrary target distributions of interest, enabling greater flexibility for personalization. We evaluate and characterize our method on a variety of federated settings, datasets, and degrees of local data heterogeneity. Our method outperforms existing alternatives, while also enabling new features for personalized FL such as transfer outside of local data distributions.

----

## [811] Combining Physics and Machine Learning for Network Flow Estimation

**Authors**: *Arlei Lopes da Silva, Furkan Kocayusufoglu, Saber Jafarpour, Francesco Bullo, Ananthram Swami, Ambuj K. Singh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=l0V53bErniB](https://openreview.net/forum?id=l0V53bErniB)

**Abstract**:

The flow estimation problem consists of predicting missing edge flows in a network (e.g., traffic, power, and water) based on partial observations. These missing flows depend both on the underlying \textit{physics} (edge features and a flow conservation law) as well as the observed edge flows. This paper introduces an optimization framework for computing missing edge flows and solves the problem using bilevel optimization and deep learning. More specifically, we learn regularizers that depend on edge features (e.g., number of lanes in a road, the resistance of a power line) using neural networks. Empirical results show that our method accurately predicts missing flows, outperforming the best baseline, and is able to capture relevant physical properties in traffic and power networks.

----

## [812] Knowledge Distillation as Semiparametric Inference

**Authors**: *Tri Dao, Govinda M. Kamath, Vasilis Syrgkanis, Lester Mackey*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=m4UCf24r0Y](https://openreview.net/forum?id=m4UCf24r0Y)

**Abstract**:

A popular approach to model compression is to train an inexpensive student model to mimic the class probabilities of a highly accurate but cumbersome teacher model. Surprisingly, this two-step knowledge distillation process often leads to higher accuracy than training the student directly on labeled data. To explain and enhance this phenomenon, we cast knowledge distillation as a semiparametric inference problem with the optimal student model as the target, the unknown Bayes class probabilities as nuisance, and the teacher probabilities as a plug-in nuisance estimate.  By adapting modern semiparametric tools, we derive new guarantees for the prediction error of standard distillation and develop two enhancements—cross-fitting and loss correction—to mitigate the impact of teacher overfitting and underfitting on student performance. We validate our findings empirically on both tabular and image data and observe consistent improvements from our knowledge distillation enhancements.

----

## [813] Learning Value Functions in Deep Policy Gradients using Residual Variance

**Authors**: *Yannis Flet-Berliac, Reda Ouhamma, Odalric-Ambrym Maillard, Philippe Preux*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NX1He-aFO_F](https://openreview.net/forum?id=NX1He-aFO_F)

**Abstract**:

Policy gradient algorithms have proven to be successful in diverse decision making and control tasks. However, these methods suffer from high sample complexity and instability issues. In this paper, we address these challenges by providing a different approach for training the critic in the actor-critic framework. Our work builds on recent studies indicating that traditional actor-critic algorithms do not succeed in fitting the true value function, calling for the need to identify a better objective for the critic. In our method, the critic uses a new state-value (resp. state-action-value) function approximation that learns the value of the states (resp. state-action pairs) relative to their mean value rather than the absolute value as in conventional actor-critic. We prove the theoretical consistency of the new gradient estimator and observe dramatic empirical improvement across a variety of continuous control tasks and algorithms. Furthermore, we validate our method in tasks with sparse rewards, where we provide experimental evidence and theoretical insights.

----

## [814] Randomized Ensembled Double Q-Learning: Learning Fast Without a Model

**Authors**: *Xinyue Chen, Che Wang, Zijian Zhou, Keith W. Ross*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=AY8zfZm0tDd](https://openreview.net/forum?id=AY8zfZm0tDd)

**Abstract**:

Using a high Update-To-Data (UTD) ratio, model-based methods have recently achieved much higher sample efficiency than previous model-free methods for continuous-action DRL benchmarks. In this paper, we introduce a simple model-free algorithm, Randomized Ensembled Double Q-Learning (REDQ), and show that its performance is just as good as, if not better than, a state-of-the-art model-based algorithm for the MuJoCo benchmark. Moreover, REDQ can achieve this performance using fewer parameters than the model-based method, and with less wall-clock run time. REDQ has three carefully integrated ingredients which allow it to achieve its high performance: (i) a UTD ratio $\gg 1$; (ii) an ensemble of Q functions; (iii) in-target minimization across a random subset of Q functions from the ensemble. Through carefully designed experiments, we provide a detailed analysis of REDQ and related model-free algorithms. To our knowledge, REDQ is the first successful model-free DRL algorithm for continuous-action spaces using a UTD ratio $\gg 1$.

----

## [815] Decentralized Attribution of Generative Models

**Authors**: *Changhoon Kim, Yi Ren, Yezhou Yang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_kxlwvhOodK](https://openreview.net/forum?id=_kxlwvhOodK)

**Abstract**:

Growing applications of generative models have led to new threats such as malicious personation and digital copyright infringement. 
One solution to these threats is model attribution, i.e., the identification of user-end models where the contents under question are generated.
Existing studies showed empirical feasibility of attribution through a centralized classifier trained on all existing user-end models. 
However, this approach is not scalable in a reality where the number of models ever grows. Neither does it provide an attributability guarantee.
To this end, this paper studies decentralized attribution, which relies on binary classifiers associated with each user-end model. 
Each binary classifier is parameterized by a user-specific key and distinguishes its associated model distribution from the authentic data distribution. 
We develop sufficient conditions of the keys that guarantee an attributability lower bound.
Our method is validated on MNIST, CelebA, and FFHQ datasets. We also examine the trade-off between generation quality and robustness of attribution against adversarial post-processes.

----

## [816] Attentional Constellation Nets for Few-Shot Learning

**Authors**: *Weijian Xu, Yifan Xu, Huaijin Wang, Zhuowen Tu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vujTf_I8Kmc](https://openreview.net/forum?id=vujTf_I8Kmc)

**Abstract**:

The success of deep convolutional neural networks builds on top of the learning of effective convolution operations, capturing a hierarchy of structured features via filtering, activation, and pooling. However, the explicit structured features, e.g. object parts, are not expressive in the existing CNN frameworks. In this paper, we tackle the few-shot learning problem and make an effort to enhance structured features by expanding CNNs with a constellation model, which performs cell feature clustering and encoding with a dense part representation; the relationships among the cell features are further modeled by an attention mechanism. With the additional constellation branch to increase the awareness of object parts, our method is able to attain the advantages of the CNNs while making the overall internal representations more robust in the few-shot learning setting. Our approach attains a significant improvement over the existing methods in few-shot learning on the CIFAR-FS, FC100, and mini-ImageNet benchmarks.

----

## [817] Adapting to Reward Progressivity via Spectral Reinforcement Learning

**Authors**: *Michael Dann, John Thangarajah*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dyjPVUc2KB](https://openreview.net/forum?id=dyjPVUc2KB)

**Abstract**:

In this paper we consider reinforcement learning tasks with progressive rewards; that is, tasks where the rewards tend to increase in magnitude over time. We hypothesise that this property may be problematic for value-based deep reinforcement learning agents, particularly if the agent must first succeed in relatively unrewarding regions of the task in order to reach more rewarding regions. To address this issue, we propose Spectral DQN, which decomposes the reward into frequencies such that the high frequencies only activate when large rewards are found. This allows the training loss to be balanced so that it gives more even weighting across small and large reward regions. In two domains with extreme reward progressivity, where standard value-based methods struggle significantly, Spectral DQN is able to make much farther progress. Moreover, when evaluated on a set of six standard Atari games that do not overtly favour the approach, Spectral DQN remains more than competitive: While it underperforms one of the benchmarks in a single game, it comfortably surpasses the benchmarks in three games. These results demonstrate that the approach is not overfit to its target problem, and suggest that Spectral DQN may have advantages beyond addressing reward progressivity.

----

## [818] TropEx: An Algorithm for Extracting Linear Terms in Deep Neural Networks

**Authors**: *Martin Trimmel, Henning Petzka, Cristian Sminchisescu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=IqtonxWI0V3](https://openreview.net/forum?id=IqtonxWI0V3)

**Abstract**:

Deep neural networks with rectified linear (ReLU) activations are piecewise linear functions, where hyperplanes partition the input space into an astronomically high number of linear regions. Previous work focused on counting linear regions to measure the network's expressive power and on analyzing geometric properties of the hyperplane configurations. In contrast, we aim to understand the impact of the linear terms on network performance, by examining the information encoded in their coefficients. To this end, we derive TropEx, a nontrivial tropical algebra-inspired algorithm to systematically extract linear terms based on data. Applied to convolutional and fully-connected networks, our algorithm uncovers significant differences in how the different networks utilize linear regions for generalization. This underlines the importance of systematic linear term exploration, to better understand generalization in neural networks trained with complex data sets.

----

## [819] Reset-Free Lifelong Learning with Skill-Space Planning

**Authors**: *Kevin Lu, Aditya Grover, Pieter Abbeel, Igor Mordatch*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HIGSa_3kOx3](https://openreview.net/forum?id=HIGSa_3kOx3)

**Abstract**:

The objective of \textit{lifelong} reinforcement learning (RL) is to optimize agents which can continuously adapt and interact in changing environments. However, current RL approaches fail drastically when environments are non-stationary and interactions are non-episodic. We propose \textit{Lifelong Skill Planning} (LiSP), an algorithmic framework for lifelong RL based on planning in an abstract space of higher-order skills. We learn the skills in an unsupervised manner using intrinsic rewards and plan over the learned skills using a learned dynamics model. Moreover, our framework permits skill discovery even from offline data, thereby reducing the need for excessive real-world interactions. We demonstrate empirically that LiSP successfully enables long-horizon planning and learns agents that can avoid catastrophic failures even in challenging non-stationary and non-episodic environments derived from gridworld and MuJoCo benchmarks.

----

## [820] Robust Learning of Fixed-Structure Bayesian Networks in Nearly-Linear Time

**Authors**: *Yu Cheng, Honghao Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=euDnVs0Ynts](https://openreview.net/forum?id=euDnVs0Ynts)

**Abstract**:

We study the problem of learning Bayesian networks where an $\epsilon$-fraction of the samples are adversarially corrupted.  We focus on the fully-observable case where the underlying graph structure is known.  In this work, we present the first nearly-linear time algorithm for this problem with a dimension-independent error guarantee.  Previous robust algorithms with comparable error guarantees are slower by at least a factor of $(d/\epsilon)$, where $d$ is the number of variables in the Bayesian network and $\epsilon$ is the fraction of corrupted samples.

Our algorithm and analysis are considerably simpler than those in previous work.  We achieve this by establishing a direct connection between robust learning of Bayesian networks and robust mean estimation.  As a subroutine in our algorithm, we develop a robust mean estimation algorithm whose runtime is nearly-linear in the number of nonzeros in the input samples, which may be of independent interest.

----

## [821] Teaching Temporal Logics to Neural Networks

**Authors**: *Christopher Hahn, Frederik Schmitt, Jens U. Kreber, Markus Norman Rabe, Bernd Finkbeiner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dOcQK-f4byz](https://openreview.net/forum?id=dOcQK-f4byz)

**Abstract**:

We study two fundamental questions in neuro-symbolic computing: can deep learning tackle challenging problems in logics end-to-end, and can neural networks learn the semantics of logics. In this work we focus on linear-time temporal logic (LTL), as it is widely used in verification. We train a Transformer on the problem to directly predict a solution, i.e. a trace, to a given LTL formula. The training data is generated with classical solvers, which, however, only provide one of many possible solutions to each formula. We demonstrate that it is sufficient to train on those particular solutions to formulas, and that Transformers can predict solutions even to formulas from benchmarks from the literature on which the classical solver timed out. Transformers also generalize to the semantics of the logics: while they often deviate from the solutions found by the classical solvers, they still predict correct solutions to most formulas.

----

## [822] Spatially Structured Recurrent Modules

**Authors**: *Nasim Rahaman, Anirudh Goyal, Muhammad Waleed Gondal, Manuel Wuthrich, Stefan Bauer, Yash Sharma, Yoshua Bengio, Bernhard Schölkopf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5l9zj5G7vDY](https://openreview.net/forum?id=5l9zj5G7vDY)

**Abstract**:

Capturing the structure of a data-generating process by means of appropriate inductive biases can help in learning models that generalise well and are robust to changes in the input distribution. While methods that harness spatial and temporal structures find broad application, recent work has demonstrated the potential of models that leverage sparse and modular structure using an ensemble of sparingly interacting modules. In this work, we take a step towards dynamic models that are capable of simultaneously exploiting both modular and spatiotemporal structures. To this end, we model the dynamical system as a collection of autonomous but sparsely interacting sub-systems that interact according to a learned topology which is informed by the spatial structure of the underlying system. This gives rise to a class of models that are well suited for capturing the dynamics of systems that only offer local views into their state, along with corresponding spatial locations of those views. On the tasks of video prediction from cropped frames and multi-agent world modelling from partial observations in the challenging Starcraft2 domain, we find our models to be more robust to the number of available views and better capable of generalisation to novel tasks without additional training than strong baselines that perform equally well or better on the training distribution.

----

## [823] Bayesian Few-Shot Classification with One-vs-Each Pólya-Gamma Augmented Gaussian Processes

**Authors**: *Jake Snell, Richard S. Zemel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lgNx56yZh8a](https://openreview.net/forum?id=lgNx56yZh8a)

**Abstract**:

Few-shot classification (FSC), the task of adapting a classifier to unseen classes given a small labeled dataset, is an important step on the path toward human-like machine learning. Bayesian methods are well-suited to tackling the fundamental issue of overfitting in the few-shot scenario because they allow practitioners to specify prior beliefs and update those beliefs in light of observed data. Contemporary approaches to Bayesian few-shot classification maintain a posterior distribution over model parameters, which is slow and requires storage that scales with model size. Instead, we propose a Gaussian process classifier based on a novel combination of Pólya-Gamma augmentation and the one-vs-each softmax approximation that allows us to efficiently marginalize over functions rather than model parameters. We demonstrate improved accuracy and uncertainty quantification on both standard few-shot classification benchmarks and few-shot domain transfer tasks.

----

## [824] Parameter-Based Value Functions

**Authors**: *Francesco Faccio, Louis Kirsch, Jürgen Schmidhuber*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tV6oBfuyLTQ](https://openreview.net/forum?id=tV6oBfuyLTQ)

**Abstract**:

Traditional off-policy actor-critic Reinforcement Learning (RL) algorithms learn value functions of a single target policy. However, when value functions are updated to track the learned policy, they forget potentially useful information about old policies. We introduce a class of value functions called Parameter-Based Value Functions (PBVFs) whose inputs include the policy parameters. They can generalize across different policies. PBVFs can evaluate the performance of any policy given a state, a state-action pair, or a distribution over the RL agent's initial states. First we show how PBVFs yield novel off-policy policy gradient theorems. Then we derive off-policy actor-critic algorithms based on PBVFs trained by Monte Carlo or Temporal Difference methods. We show how learned PBVFs can zero-shot learn new policies that outperform any policy seen during training. Finally our algorithms are evaluated on a selection of discrete and continuous control tasks using shallow policies and deep neural networks. Their performance is comparable to state-of-the-art methods.

----

## [825] Hyperbolic Neural Networks++

**Authors**: *Ryohei Shimizu, Yusuke Mukuta, Tatsuya Harada*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ec85b0tUwbA](https://openreview.net/forum?id=Ec85b0tUwbA)

**Abstract**:

Hyperbolic spaces, which have the capacity to embed tree structures without distortion owing to their exponential volume growth, have recently been applied to machine learning to better capture the hierarchical nature of data. In this study, we generalize the fundamental components of neural networks in a single hyperbolic geometry model, namely, the Poincaré ball model. This novel methodology constructs a multinomial logistic regression, fully-connected layers, convolutional layers, and attention mechanisms under a unified mathematical interpretation, without increasing the parameters. Experiments show the superior parameter efficiency of our methods compared to conventional hyperbolic components, and stability and outperformance over their Euclidean counterparts.

----

## [826] Neural Jump Ordinary Differential Equations: Consistent Continuous-Time Prediction and Filtering

**Authors**: *Calypso Herrera, Florian Krach, Josef Teichmann*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JFKR3WqwyXR](https://openreview.net/forum?id=JFKR3WqwyXR)

**Abstract**:

Combinations of neural ODEs with recurrent neural networks (RNN), like GRU-ODE-Bayes or ODE-RNN are well suited to model irregularly observed time series. While those models outperform existing discrete-time approaches, no theoretical guarantees for their predictive capabilities are available. Assuming that the irregularly-sampled time series data originates from a continuous stochastic process, the $L^2$-optimal online prediction is the conditional expectation given the currently available information. We introduce the Neural Jump ODE (NJ-ODE) that provides a data-driven approach to learn, continuously in time, the conditional expectation of a stochastic process. Our approach models the conditional expectation between two observations with a neural ODE and jumps whenever a new observation is made. We define a novel training framework, which allows us to prove theoretical guarantees for the first time. In particular, we show that the output of our model converges to the $L^2$-optimal prediction. This can be interpreted as solution to a special filtering problem. We provide experiments showing that the theoretical results also hold empirically. Moreover, we experimentally show that our model outperforms the baselines in more complex learning tasks and give comparisons on real-world datasets.

----

## [827] Seq2Tens: An Efficient Representation of Sequences by Low-Rank Tensor Projections

**Authors**: *Csaba Tóth, Patric Bonnier, Harald Oberhauser*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dx4b7lm8jMM](https://openreview.net/forum?id=dx4b7lm8jMM)

**Abstract**:

Sequential data such as time series, video, or text can be challenging to analyse as the ordered structure gives rise to complex dependencies. At the heart of this is non-commutativity, in the sense that reordering the elements of a sequence can completely change its meaning. We use a classical mathematical object -- the free algebra -- to capture this non-commutativity. To address the innate computational complexity of this algebra, we use compositions of low-rank tensor projections. This yields modular and scalable building blocks that give state-of-the-art performance on standard benchmarks such as multivariate time series classification, mortality prediction and generative models for video.

----

## [828] FOCAL: Efficient Fully-Offline Meta-Reinforcement Learning via Distance Metric Learning and Behavior Regularization

**Authors**: *Lanqing Li, Rui Yang, Dijun Luo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8cpHIfgY4Dj](https://openreview.net/forum?id=8cpHIfgY4Dj)

**Abstract**:

We study the offline meta-reinforcement learning (OMRL) problem, a paradigm which enables reinforcement learning (RL) algorithms to quickly adapt to unseen tasks without any interactions with the environments, making RL truly practical in many real-world applications. This problem is still not fully understood, for which two major challenges need to be addressed. First, offline RL usually suffers from bootstrapping errors of out-of-distribution state-actions which leads to divergence of value functions. Second, meta-RL requires efficient and robust task inference learned jointly with control policy. In this work, we enforce behavior regularization on learned policy as a general approach to offline RL, combined with a deterministic context encoder for efficient task inference. We propose a novel negative-power distance metric on bounded context embedding space, whose gradients propagation is detached from the Bellman backup. We provide analysis and insight showing that some simple design choices can yield substantial improvements over recent approaches involving meta-RL and distance metric learning. To the best of our knowledge, our method is the first model-free and end-to-end OMRL algorithm, which is computationally efficient and demonstrated to outperform prior algorithms on several meta-RL benchmarks.

----

## [829] On the Curse of Memory in Recurrent Neural Networks: Approximation and Optimization Analysis

**Authors**: *Zhong Li, Jiequn Han, Weinan E, Qianxiao Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8Sqhl-nF50](https://openreview.net/forum?id=8Sqhl-nF50)

**Abstract**:

We study the approximation properties and optimization dynamics of recurrent neural networks (RNNs) when applied to learn input-output relationships in temporal data. We consider the simple but representative setting of using continuous-time linear RNNs to learn from data generated by linear relationships. Mathematically, the latter can be understood as a sequence of linear functionals.  We prove a universal approximation theorem of such linear functionals and characterize the approximation rate.  Moreover, we perform a fine-grained dynamical analysis of training linear RNNs by gradient methods.  A unifying theme uncovered is the non-trivial effect of memory, a notion that can be made precise in our framework, on both approximation and optimization: when there is long-term memory in the target, it takes a large number of neurons to approximate it. Moreover, the training process will suffer from slow downs.  In particular, both of these effects become exponentially more pronounced with increasing memory - a phenomenon we call the “curse of memory”.  These analyses represent a basic step towards a concrete mathematical understanding of new phenomenons that may arise in learning temporal relationships using recurrent architectures.

----

## [830] Generating Adversarial Computer Programs using Optimized Obfuscations

**Authors**: *Shashank Srikant, Sijia Liu, Tamara Mitrovska, Shiyu Chang, Quanfu Fan, Gaoyuan Zhang, Una-May O'Reilly*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PH5PH9ZO_4](https://openreview.net/forum?id=PH5PH9ZO_4)

**Abstract**:

Machine learning (ML) models that learn and predict properties of computer programs are increasingly being adopted and deployed. 
These models have demonstrated success in applications such as auto-completing code, summarizing large programs, and detecting bugs and malware in programs. 
In this work, we investigate principled ways to adversarially perturb a computer program to fool such learned models, and thus determine their adversarial robustness. We use program obfuscations, which have conventionally been used to avoid attempts at reverse engineering programs, as adversarial perturbations. These perturbations modify programs in ways that do not alter their functionality but can be crafted to deceive an ML model when making a decision. We provide a general formulation for an adversarial program that allows applying multiple obfuscation transformations to a program in any language. We develop first-order optimization algorithms to  efficiently determine two key aspects -- which parts of the program to transform, and what transformations to use. We show that it is important to optimize both these aspects to generate the best adversarially perturbed program. Due to the discrete nature of this problem, we also propose using randomized smoothing to improve the attack loss landscape to ease optimization. 
We evaluate our work on Python and Java programs on the problem of program summarization. 
We show that our best attack proposal achieves a $52\%$ improvement over a state-of-the-art attack generation approach for programs trained on a \textsc{seq2seq} model.
We further show that our formulation is better at training models that are robust to adversarial attacks.

----

## [831] BOIL: Towards Representation Change for Few-shot Learning

**Authors**: *Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, Se-Young Yun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=umIdUL8rMH](https://openreview.net/forum?id=umIdUL8rMH)

**Abstract**:

Model Agnostic Meta-Learning (MAML) is one of the most representative of gradient-based meta-learning algorithms. MAML learns new tasks with a few data samples using inner updates from a meta-initialization point and learns the meta-initialization parameters with outer updates. It has recently been hypothesized that representation reuse, which makes little change in efficient representations, is the dominant factor in the performance of the meta-initialized model through MAML in contrast to representation change, which causes a significant change in representations. In this study, we investigate the necessity of representation change for the ultimate goal of few-shot learning, which is solving domain-agnostic tasks. To this aim, we propose a novel meta-learning algorithm, called BOIL (Body Only update in Inner Loop), which updates only the body (extractor) of the model and freezes the head (classifier) during inner loop updates. BOIL leverages representation change rather than representation reuse. A frozen head cannot achieve better results than even a random guessing classifier at the initial point of new tasks, and feature vectors (representations) have to move quickly to their corresponding frozen head vectors. We visualize this property using cosine similarity, CKA, and empirical results without the head. Although the inner loop updates purely hinge on representation change, BOIL empirically shows significant performance improvement over MAML, particularly on cross-domain tasks. The results imply that representation change in gradient-based meta-learning approaches is a critical component.

----

## [832] Interpreting and Boosting Dropout from a Game-Theoretic View

**Authors**: *Hao Zhang, Sen Li, Yinchao Ma, Mingjie Li, Yichen Xie, Quanshi Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Jacdvfjicf7](https://openreview.net/forum?id=Jacdvfjicf7)

**Abstract**:

This paper aims to understand and improve the utility of the dropout operation from the perspective of game-theoretical interactions. We prove that dropout can suppress the strength of interactions between input variables of deep neural networks (DNNs). The theoretical proof is also verified by various experiments. Furthermore, we find that such interactions were strongly related to the over-fitting problem in deep learning. So, the utility of dropout can be regarded as decreasing interactions to alleviating the significance of over-fitting. Based on this understanding, we propose the interaction loss to further improve the utility of dropout. Experimental results on various DNNs and datasets have shown that the interaction loss can effectively improve the utility of dropout and boost the performance of DNNs.

----

## [833] Representation Learning via Invariant Causal Mechanisms

**Authors**: *Jovana Mitrovic, Brian McWilliams, Jacob C. Walker, Lars Holger Buesing, Charles Blundell*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9p2ekP904Rs](https://openreview.net/forum?id=9p2ekP904Rs)

**Abstract**:

Self-supervised learning has emerged as a strategy to reduce the reliance on costly supervised signal by pretraining representations only using unlabeled data. These methods combine heuristic proxy classification tasks with data augmentations and have achieved significant success, but our theoretical understanding of this success remains limited. In this paper we analyze self-supervised representation learning using a causal framework.  We show how data augmentations can be more effectively utilized through explicit invariance constraints on the proxy classifiers employed during pretraining. Based on this, we propose a novel self-supervised objective, Representation Learning via Invariant Causal Mechanisms (ReLIC), that enforces invariant prediction of proxy targets across augmentations through an invariance regularizer which yields improved generalization guarantees. Further, using causality we generalize contrastive learning, a particular kind of self-supervised method,  and provide an alternative theoretical explanation for the  success  of  these  methods. Empirically, ReLIC significantly outperforms competing methods in terms of robustness and out-of-distribution generalization on ImageNet, while also significantly outperforming these methods on Atari achieving above human-level performance on 51 out of 57 games.

----

## [834] Fooling a Complete Neural Network Verifier

**Authors**: *Dániel Zombori, Balázs Bánhelyi, Tibor Csendes, István Megyeri, Márk Jelasity*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4IwieFS44l](https://openreview.net/forum?id=4IwieFS44l)

**Abstract**:

The efficient and accurate characterization of the robustness of neural networks to input perturbation is an important open problem. Many approaches exist including heuristic and exact (or complete) methods. Complete methods are expensive but their mathematical formulation guarantees that they provide exact robustness metrics. However, this guarantee is valid only if we assume that the verified network applies arbitrary-precision arithmetic and the verifier is reliable. In practice, however, both the networks and the verifiers apply limited-precision floating point arithmetic. In this paper, we show that numerical roundoff errors can be exploited to craft adversarial networks, in which the actual robustness and the robustness computed by a state-of-the-art complete verifier radically differ. We also show that such adversarial networks can be used to insert a backdoor into any network in such a way that the backdoor is completely missed by the verifier. The attack is easy to detect in its naive form but, as we show, the adversarial network can be transformed to make its detection less trivial. We offer a simple defense against our particular attack based on adding a very small perturbation to the network weights. However, our conjecture is that other numerical attacks are possible, and exact verification has to take into account all the details of the computation executed by the verified networks, which makes the problem significantly harder.

----

## [835] CPR: Classifier-Projection Regularization for Continual Learning

**Authors**: *Sungmin Cha, Hsiang Hsu, Taebaek Hwang, Flávio P. Calmon, Taesup Moon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=F2v4aqEL6ze](https://openreview.net/forum?id=F2v4aqEL6ze)

**Abstract**:

We propose a general, yet simple patch that can be applied to existing regularization-based continual learning methods called classifier-projection regularization (CPR). Inspired by both recent results on neural networks with wide local minima and information theory, CPR adds an additional regularization term that maximizes the entropy of a classifier's output probability. We demonstrate that this additional term can be interpreted as a projection of the conditional probability given by a classifier's output to the uniform distribution. By applying the Pythagorean theorem for KL divergence, we then prove that this projection may (in theory) improve the performance of continual learning methods. In our extensive experimental results, we apply CPR to several state-of-the-art regularization-based continual learning methods and benchmark performance on popular image recognition datasets. Our results demonstrate that CPR indeed promotes a wide local minima and significantly improves both accuracy and plasticity while simultaneously mitigating the catastrophic forgetting of baseline continual learning methods. The codes and scripts for this work are available at https://github.com/csm9493/CPR_CL.

----

## [836] CO2: Consistent Contrast for Unsupervised Visual Representation Learning

**Authors**: *Chen Wei, Huiyu Wang, Wei Shen, Alan L. Yuille*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=U4XLJhqwNF1](https://openreview.net/forum?id=U4XLJhqwNF1)

**Abstract**:

Contrastive learning has recently been a core for unsupervised visual representation learning. Without human annotation, the common practice is to perform an instance discrimination task: Given a query image crop, label crops from the same image as positives, and crops from other randomly sampled images as negatives. An important limitation of this label assignment is that it can not reflect the heterogeneous similarity of the query crop to crops from other images, but regarding them as equally negative. To address this issue, inspired by consistency regularization in semi-supervised learning, we propose Consistent Contrast (CO2), which introduces a consistency term into unsupervised contrastive learning framework. The consistency term takes the similarity of the query crop to crops from other images as unlabeled, and the corresponding similarity of a positive crop as a pseudo label. It then encourages consistency between these two similarities. Empirically, CO2 improves Momentum Contrast (MoCo) by 2.9% top-1 accuracy on ImageNet linear protocol, 3.8% and 1.1% top-5 accuracy on 1% and 10% labeled semi-supervised settings. It also transfers to image classification, object detection, and semantic segmentation on PASCAL VOC. This shows that CO2 learns better visual representations for downstream tasks.

----

## [837] GAN2GAN: Generative Noise Learning for Blind Denoising with Single Noisy Images

**Authors**: *Sungmin Cha, Taeeon Park, Byeongjoon Kim, Jongduk Baek, Taesup Moon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=SHvF5xaueVn](https://openreview.net/forum?id=SHvF5xaueVn)

**Abstract**:

We tackle a challenging blind image denoising problem, in which only single distinct noisy images are available for training a denoiser, and no information about noise is known, except for it being zero-mean, additive, and independent of the clean image. In such a setting, which often occurs in practice, it is not possible to train a denoiser with the standard discriminative training or with the recently developed Noise2Noise (N2N) training; the former requires the underlying clean image for the given noisy image, and the latter requires two independently realized noisy image pair for a clean image. To that end, we propose GAN2GAN (Generated-Artificial-Noise to Generated-Artificial-Noise) method that first learns a generative model that can 1) simulate the noise in the given noisy images and 2) generate a rough, noisy estimates of the clean images, then 3) iteratively trains a denoiser with subsequently synthesized noisy image pairs (as in N2N), obtained from the generative model. In results, we show the denoiser trained with our GAN2GAN achieves an impressive denoising performance on both synthetic and real-world datasets for the blind denoising setting; it almost approaches the performance of the standard discriminatively-trained or N2N-trained models that have more information than ours, and it significantly outperforms the recent baseline for the same setting, \textit{e.g.}, Noise2Void, and a more conventional yet strong one, BM3D. The official code of our method is available at https://github.com/csm9493/GAN2GAN.

----

## [838] Learning Subgoal Representations with Slow Dynamics

**Authors**: *Siyuan Li, Lulu Zheng, Jianhao Wang, Chongjie Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wxRwhSdORKG](https://openreview.net/forum?id=wxRwhSdORKG)

**Abstract**:

In goal-conditioned Hierarchical Reinforcement Learning (HRL), a high-level policy periodically sets subgoals for a low-level policy, and the low-level policy is trained to reach those subgoals. A proper subgoal representation function, which abstracts a state space to a latent subgoal space, is crucial for effective goal-conditioned HRL, since different low-level behaviors are induced by reaching subgoals in the compressed representation space. Observing that the high-level agent operates at an abstract temporal scale, we propose a slowness objective to effectively learn the subgoal representation (i.e., the high-level action space). We provide a theoretical grounding for the slowness objective. That is, selecting slow features as the subgoal space can achieve efficient hierarchical exploration. As a result of better exploration ability, our approach significantly outperforms state-of-the-art HRL and exploration methods on a number of benchmark continuous-control tasks. Thanks to the generality of the proposed subgoal representation learning method, empirical results also demonstrate that the learned representation and corresponding low-level policies can be transferred between distinct tasks.

----

## [839] Bowtie Networks: Generative Modeling for Joint Few-Shot Recognition and Novel-View Synthesis

**Authors**: *Zhipeng Bao, Yu-Xiong Wang, Martial Hebert*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ESG-DMKQKsD](https://openreview.net/forum?id=ESG-DMKQKsD)

**Abstract**:

We propose a novel task of joint few-shot recognition and novel-view synthesis: given only one or few images of a novel object from arbitrary views with only category annotation, we aim to simultaneously learn an object classifier and generate images of that type of object from new viewpoints. While existing work copes with two or more tasks mainly by multi-task learning of shareable feature representations, we take a different perspective. We focus on the interaction and cooperation between a generative model and a discriminative model, in a way that facilitates knowledge to flow across tasks in complementary directions. To this end, we propose bowtie networks that jointly learn 3D geometric and semantic representations with a feedback loop. Experimental evaluation on challenging fine-grained recognition datasets demonstrates that our synthesized images are realistic from multiple viewpoints and significantly improve recognition performance as ways of data augmentation, especially in the low-data regime.

----

## [840] Taming GANs with Lookahead-Minmax

**Authors**: *Tatjana Chavdarova, Matteo Pagliardini, Sebastian U. Stich, François Fleuret, Martin Jaggi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZW0yXJyNmoG](https://openreview.net/forum?id=ZW0yXJyNmoG)

**Abstract**:

Generative Adversarial Networks are notoriously challenging to train. The underlying minmax optimization is highly susceptible to the variance of the stochastic gradient and the rotational component of the associated game vector field. To tackle these challenges, we propose the Lookahead algorithm for minmax optimization, originally developed for single objective minimization only. The backtracking step of our Lookahead–minmax naturally handles the rotational game dynamics, a property which was identified to be key for enabling gradient ascent descent methods to converge on challenging examples often analyzed in the literature. Moreover, it implicitly handles high variance without using large mini-batches, known to be essential for reaching state of the art performance. Experimental results on MNIST, SVHN, CIFAR-10, and ImageNet demonstrate a clear advantage of combining Lookahead–minmax with Adam or extragradient, in terms of performance and improved stability, for negligible memory and computational cost. Using 30-fold fewer parameters and 16-fold smaller minibatches we outperform the reported performance of the class-dependent BigGAN on CIFAR-10 by obtaining FID of 12.19 without using the class labels, bringing state-of-the-art GAN training within reach of common computational resources.

----

## [841] Certify or Predict: Boosting Certified Robustness with Compositional Architectures

**Authors**: *Mark Niklas Müller, Mislav Balunovic, Martin T. Vechev*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=USCNapootw](https://openreview.net/forum?id=USCNapootw)

**Abstract**:

A core challenge with existing certified defense mechanisms is that while they improve certified robustness, they also tend to drastically decrease natural accuracy, making it difficult to use these methods in practice. In this work, we propose a new architecture which addresses this challenge and enables one to boost the certified robustness of any state-of-the-art deep network, while controlling the overall accuracy loss, without requiring retraining. The key idea is to combine this model with a (smaller) certified network where at inference time, an adaptive selection mechanism decides on the network to process the input sample. The approach is compositional: one can combine any pair of state-of-the-art (e.g., EfficientNet or ResNet) and certified networks, without restriction. The resulting architecture enables much higher natural accuracy than previously possible with certified defenses alone, while substantially boosting the certified robustness of deep networks. We demonstrate the effectiveness of this adaptive approach on a variety of datasets and architectures.  For instance, on CIFAR-10 with an $\ell_\infty$ perturbation of 2/255, we are the first to obtain a high natural accuracy (90.1%) with non-trivial certified robustness (27.5%). Notably, prior state-of-the-art methods incur a substantial drop in accuracy for a similar certified robustness.

----

## [842] New Bounds For Distributed Mean Estimation and Variance Reduction

**Authors**: *Peter Davies, Vijaykrishna Gurunanthan, Niusha Moshrefi, Saleh Ashkboos, Dan Alistarh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=t86MwoUCCNe](https://openreview.net/forum?id=t86MwoUCCNe)

**Abstract**:

We consider the problem of distributed mean estimation (DME), in which $n$ machines are each given a local $d$-dimensional vector $\mathbf x_v \in \mathbb R^d$, and must cooperate to estimate the mean of their inputs $\mathbf \mu = \frac 1n\sum_{v = 1}^n \mathbf x_v$, while minimizing total communication cost. DME is a fundamental construct in distributed machine learning, and there has been considerable work on variants of this problem, especially in the context of distributed variance reduction for stochastic gradients in parallel SGD. Previous work typically assumes an upper bound on the norm of the input vectors, and achieves an error bound in terms of this norm. However, in many real applications, the input vectors are concentrated around the correct output $\mathbf \mu$, but $\mathbf \mu$ itself has large norm. In such cases, previous output error bounds perform poorly. 
            In this paper, we show that output error bounds need not depend on input norm. We provide a method of quantization which allows distributed mean estimation to be performed with solution quality dependent only on the distance between inputs, not on input norm, and show an analogous result for distributed variance reduction. The technique is based on a new connection with lattice theory. We also provide lower bounds showing that the communication to error trade-off of our algorithms is asymptotically optimal. As the lattices achieving optimal bounds under $\ell_2$-norm can be computationally impractical, we also present an extension which leverages  easy-to-use cubic lattices, and is loose only up to a logarithmic factor in $d$. We show experimentally that our method yields practical improvements for common applications, relative to prior approaches.

----

## [843] Interpretable Neural Architecture Search via Bayesian Optimisation with Weisfeiler-Lehman Kernels

**Authors**: *Bin Xin Ru, Xingchen Wan, Xiaowen Dong, Michael A. Osborne*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=j9Rv7qdXjd](https://openreview.net/forum?id=j9Rv7qdXjd)

**Abstract**:

Current neural architecture search (NAS) strategies focus only on finding a single, good, architecture. They offer little insight into why a specific network is performing well, or how we should modify the architecture if we want further improvements. We propose a Bayesian optimisation (BO) approach for NAS that combines the Weisfeiler-Lehman graph kernel with a Gaussian process surrogate. Our method not only optimises the architecture in a highly data-efficient manner, but also affords interpretability by discovering useful network features and their corresponding impact on the network performance. Moreover, our method is capable of capturing the topological structures of the architectures and is scalable to large graphs, thus making the high-dimensional and graph-like search spaces amenable to BO. We demonstrate empirically that our surrogate model is capable of identifying useful motifs which can guide the generation of new architectures. We finally show that our method outperforms existing NAS approaches to achieve the state of the art on both closed- and open-domain search spaces.

----

## [844] A Discriminative Gaussian Mixture Model with Sparsity

**Authors**: *Hideaki Hayashi, Seiichi Uchida*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-_Zp7r2-cGK](https://openreview.net/forum?id=-_Zp7r2-cGK)

**Abstract**:

In probabilistic classification, a discriminative model based on the softmax function has a potential limitation in that it assumes unimodality for each class in the feature space. The mixture model can address this issue, although it leads to an increase in the number of parameters. We propose a sparse classifier based on a discriminative GMM, referred to as a sparse discriminative Gaussian mixture (SDGM). In the SDGM, a GMM-based discriminative model is trained via sparse Bayesian learning. Using this sparse learning framework, we can simultaneously remove redundant Gaussian components and reduce the number of parameters used in the remaining components during learning; this learning method reduces the model complexity, thereby improving the generalization capability. Furthermore, the SDGM can be embedded into neural networks (NNs), such as convolutional NNs, and can be trained in an end-to-end manner. Experimental results demonstrated that the proposed method outperformed the existing softmax-based discriminative models.

----

## [845] Communication in Multi-Agent Reinforcement Learning: Intention Sharing

**Authors**: *Woojun Kim, Jongeui Park, Youngchul Sung*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qpsl2dR9twy](https://openreview.net/forum?id=qpsl2dR9twy)

**Abstract**:

Communication is one of the core components for learning coordinated behavior in multi-agent systems.
In this paper, we propose a new communication scheme named  Intention Sharing (IS) for multi-agent reinforcement learning in order to enhance the coordination among agents. In the proposed IS scheme, each agent generates an imagined trajectory by modeling the environment dynamics and other agents' actions. The imagined trajectory is the simulated future trajectory of each agent based on the learned model of the environment dynamics and other agents and represents each agent's future action plan. Each agent compresses this imagined trajectory capturing its future action plan to generate its intention message for communication by applying an attention mechanism to learn the relative importance of the components in the imagined trajectory based on the received message from other agents. Numeral results show that the proposed IS scheme outperforms other communication schemes in multi-agent reinforcement learning.

----

## [846] Is Attention Better Than Matrix Decomposition?

**Authors**: *Zhengyang Geng, Meng-Hao Guo, Hongxu Chen, Xia Li, Ke Wei, Zhouchen Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1FvkSpWosOl](https://openreview.net/forum?id=1FvkSpWosOl)

**Abstract**:

As an essential ingredient of modern deep learning, attention mechanism, especially self-attention, plays a vital role in the global correlation discovery. However, is hand-crafted attention irreplaceable when modeling the global context? Our intriguing finding is that self-attention is not better than the matrix decomposition~(MD) model developed 20 years ago regarding the performance and computational cost for encoding the long-distance dependencies. We model the global context issue as a low-rank completion problem and show that its optimization algorithms can help design global information blocks. This paper then proposes a series of Hamburgers, in which we employ the optimization algorithms for solving MDs to factorize the input representations into sub-matrices and reconstruct a low-rank embedding. Hamburgers with different MDs can perform favorably against the popular global context module self-attention when carefully coping with gradients back-propagated through MDs. Comprehensive experiments are conducted in the vision tasks where it is crucial to learn the global context, including semantic segmentation and image generation, demonstrating significant improvements over self-attention and its variants. Code is available at https://github.com/Gsunshine/Enjoy-Hamburger.

----

## [847] Fast and Complete: Enabling Complete Neural Network Verification with Rapid and Massively Parallel Incomplete Verifiers

**Authors**: *Kaidi Xu, Huan Zhang, Shiqi Wang, Yihan Wang, Suman Jana, Xue Lin, Cho-Jui Hsieh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=nVZtXBI6LNn](https://openreview.net/forum?id=nVZtXBI6LNn)

**Abstract**:

Formal verification of neural networks (NNs) is a challenging and important problem. Existing efficient complete solvers typically require the branch-and-bound (BaB) process, which splits the problem domain into sub-domains and solves each sub-domain using faster but weaker incomplete verifiers, such as Linear Programming (LP) on linearly relaxed sub-domains.  In this paper, we propose to use the backward mode linear relaxation based perturbation analysis (LiRPA) to replace LP during the BaB process, which can be efficiently implemented on the typical machine learning accelerators such as GPUs and TPUs.  However, unlike LP, LiRPA when applied naively can produce much weaker bounds and even cannot check certain conflicts of sub-domains during splitting, making the entire procedure incomplete after BaB. To address these challenges, we apply a fast gradient based bound tightening procedure combined with batch splits and the design of minimal usage of LP bound procedure, enabling us to effectively use LiRPA on the accelerator hardware for the challenging complete NN verification problem and significantly outperform LP-based approaches. On a single GPU, we demonstrate an order of magnitude speedup compared to existing LP-based approaches.

----

## [848] A Geometric Analysis of Deep Generative Image Models and Its Applications

**Authors**: *Binxu Wang, Carlos R. Ponce*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=GH7QRzUDdXG](https://openreview.net/forum?id=GH7QRzUDdXG)

**Abstract**:

Generative adversarial networks (GANs) have emerged as a powerful unsupervised method to model the statistical patterns of real-world data sets, such as natural images. These networks are trained to map random inputs in their latent space to new samples representative of the learned data. However, the structure of the latent space is hard to intuit due to its high dimensionality and the non-linearity of the generator, which limits the usefulness of the models. Understanding the latent space requires a way to identify input codes for existing real-world images (inversion), and a way to identify directions with known image transformations (interpretability). Here, we use a geometric framework to address both issues simultaneously. We develop an architecture-agnostic method to compute the Riemannian metric of the image manifold created by GANs. The eigen-decomposition of the metric isolates axes that account for different levels of image variability. An empirical analysis of several pretrained GANs shows that image variation around each position is concentrated along surprisingly few major axes (the space is highly anisotropic) and the directions that create this large variation are similar at different positions in the space (the space is homogeneous). We show that many of the top eigenvectors correspond to interpretable transforms in the image space, with a substantial part of eigenspace corresponding to minor transforms which could be compressed out. This geometric understanding unifies key previous results related to GAN interpretability. We show that the use of this metric allows for more efficient optimization in the latent space (e.g. GAN inversion) and facilitates unsupervised discovery of interpretable axes. Our results illustrate that defining the geometry of the GAN image manifold can serve as a general framework for understanding GANs.

----

## [849] Solving Compositional Reinforcement Learning Problems via Task Reduction

**Authors**: *Yunfei Li, Yilin Wu, Huazhe Xu, Xiaolong Wang, Yi Wu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9SS69KwomAM](https://openreview.net/forum?id=9SS69KwomAM)

**Abstract**:

We propose a novel learning paradigm, Self-Imitation via Reduction (SIR), for solving compositional reinforcement learning problems. SIR is based on two core ideas: task reduction and self-imitation. Task reduction tackles a hard-to-solve task by actively reducing it to an easier task whose solution is known by the RL agent. Once the original hard task is successfully solved by task reduction, the agent naturally obtains a self-generated solution trajectory to imitate. By continuously collecting and imitating such demonstrations, the agent is able to progressively expand the solved subspace in the entire task space. Experiment results show that SIR can significantly accelerate and improve learning on a variety of challenging sparse-reward continuous-control problems with compositional structures. Code and videos are available at https://sites.google.com/view/sir-compositional.

----

## [850] ARMOURED: Adversarially Robust MOdels using Unlabeled data by REgularizing Diversity

**Authors**: *Kangkang Lu, Cuong Manh Nguyen, Xun Xu, Kiran Chari, Yu Jing Goh, Chuan-Sheng Foo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JoCR4h9O3Ew](https://openreview.net/forum?id=JoCR4h9O3Ew)

**Abstract**:

Adversarial attacks pose a major challenge for modern deep neural networks. Recent advancements show that adversarially robust generalization requires a large amount of labeled data for training. If annotation becomes a burden, can unlabeled data help bridge the gap? In this paper, we propose ARMOURED, an adversarially robust training method based on semi-supervised learning that consists of two components. The first component applies multi-view learning to simultaneously optimize multiple independent networks and utilizes unlabeled data to enforce labeling consistency. The second component reduces adversarial transferability among the networks via diversity regularizers inspired by determinantal point processes and entropy maximization. Experimental results show that under small perturbation budgets, ARMOURED is robust against strong adaptive adversaries. Notably, ARMOURED does not rely on generating adversarial samples during training. When used in combination with adversarial training, ARMOURED yields competitive performance with the state-of-the-art adversarially-robust benchmarks on SVHN and outperforms them on CIFAR-10, while offering higher clean accuracy.

----

## [851] Acting in Delayed Environments with Non-Stationary Markov Policies

**Authors**: *Esther Derman, Gal Dalal, Shie Mannor*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=j1RMMKeP2gR](https://openreview.net/forum?id=j1RMMKeP2gR)

**Abstract**:

The standard Markov Decision Process (MDP) formulation hinges on the assumption that an action is executed immediately after it was chosen. However, assuming it is often unrealistic and can lead to catastrophic failures in applications such as robotic manipulation, cloud computing, and finance. We introduce a framework for learning and planning in MDPs where the decision-maker commits actions that are executed with a delay of $m$ steps. The brute-force state augmentation baseline where the state is concatenated to the last $m$ committed actions suffers from an exponential complexity in $m$, as we show for policy iteration. We then prove that with execution delay, deterministic Markov policies in the original state-space are sufficient for attaining maximal reward, but need to be non-stationary. As for stationary Markov policies, we show they are sub-optimal in general. Consequently, we devise a non-stationary Q-learning style model-based algorithm that solves delayed execution tasks without resorting to state-augmentation. Experiments on tabular, physical, and Atari domains reveal that it converges quickly to high performance even for substantial delays, while standard approaches that either ignore the delay or rely on state-augmentation struggle or fail due to divergence. The code is available at \url{https://github.com/galdl/rl_delay_basic.git}.

----

## [852] Overfitting for Fun and Profit: Instance-Adaptive Data Compression

**Authors**: *Ties van Rozendaal, Iris A. M. Huijben, Taco Cohen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=oFp8Mx_V5FL](https://openreview.net/forum?id=oFp8Mx_V5FL)

**Abstract**:

Neural data compression has been shown to outperform classical methods in terms of $RD$ performance, with results still improving rapidly.
At a high level, neural compression is based on an autoencoder that tries to reconstruct the input instance from a (quantized) latent representation, coupled with a prior that is used to losslessly compress these latents.
Due to limitations on model capacity and imperfect optimization and generalization, such models will suboptimally compress test data in general.
However, one of the great strengths of learned compression is that if the test-time data distribution is known and relatively low-entropy (e.g. a camera watching a static scene, a dash cam in an autonomous car, etc.), the model can easily be finetuned or adapted to this distribution, leading to improved $RD$ performance.
In this paper we take this concept to the extreme, adapting the full model to a single video, and sending model updates (quantized and compressed using a parameter-space prior) along with the latent representation. Unlike previous work, we finetune not only the encoder/latents but the entire model, and - during finetuning - take into account both the effect of model quantization and the additional costs incurred by sending the model updates. We evaluate an image compression model on I-frames (sampled at 2 fps) from videos of the Xiph dataset, and demonstrate that full-model adaptation improves $RD$ performance by ~1 dB, with respect to encoder-only finetuning.

----

## [853] Learnable Embedding sizes for Recommender Systems

**Authors**: *Siyi Liu, Chen Gao, Yihong Chen, Depeng Jin, Yong Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=vQzcqQWIS0q](https://openreview.net/forum?id=vQzcqQWIS0q)

**Abstract**:

The embedding-based representation learning is commonly used in deep learning recommendation models to map the raw sparse features to dense vectors. The traditional embedding manner that assigns a uniform size to all features has two issues. First, the numerous features inevitably lead to a gigantic embedding table that causes a high memory usage cost. Second, it is likely to cause the over-fitting problem for those features that do not require too large representation capacity. Existing works that try to address the problem always cause a significant drop in recommendation performance or suffers from the limitation of unaffordable training time cost. In this paper, we proposed a novel approach, named PEP (short for Plug-in Embedding Pruning), to reduce the size of the embedding table while avoiding the drop of recommendation accuracy. PEP prunes embedding parameter where the pruning threshold(s) can be adaptively learned from data. Therefore we can automatically obtain a mixed-dimension embedding-scheme by pruning redundant parameters for each feature. PEP is a general framework that can plug in various base recommendation models. Extensive experiments demonstrate it can efficiently cut down embedding parameters and boost the base model's performance. Specifically, it achieves strong recommendation performance while reducing 97-99% parameters. As for the computation cost, PEP only brings an additional 20-30% time cost compare with base models.

----

## [854] Generative Scene Graph Networks

**Authors**: *Fei Deng, Zhuo Zhi, Donghun Lee, Sungjin Ahn*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RmcPm9m3tnk](https://openreview.net/forum?id=RmcPm9m3tnk)

**Abstract**:

Human perception excels at building compositional hierarchies of parts and objects from unlabeled scenes that help systematic generalization. Yet most work on generative scene modeling either ignores the part-whole relationship or assumes access to predefined part labels. In this paper, we propose Generative Scene Graph Networks (GSGNs), the first deep generative model that learns to discover the primitive parts and infer the part-whole relationship jointly from multi-object scenes without supervision and in an end-to-end trainable way. We formulate GSGN as a variational autoencoder in which the latent representation is a tree-structured probabilistic scene graph. The leaf nodes in the latent tree correspond to primitive parts, and the edges represent the symbolic pose variables required for recursively composing the parts into whole objects and then the full scene. This allows novel objects and scenes to be generated both by sampling from the prior and by manual configuration of the pose variables, as we do with graphics engines. We evaluate GSGN on datasets of scenes containing multiple compositional objects, including a challenging Compositional CLEVR dataset that we have developed. We show that GSGN is able to infer the latent scene graph, generalize out of the training regime, and improve data efficiency in downstream tasks.

----

## [855] Deconstructing the Regularization of BatchNorm

**Authors**: *Yann N. Dauphin, Ekin Dogus Cubuk*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=d-XzF81Wg1](https://openreview.net/forum?id=d-XzF81Wg1)

**Abstract**:

Batch normalization (BatchNorm) has become a standard technique in deep learning. Its popularity is in no small part due to its often positive effect on generalization. Despite this success, the regularization effect of the technique is still poorly understood. This study aims to decompose BatchNorm into separate mechanisms that are much simpler. We identify three effects of BatchNorm and assess their impact directly with ablations and interventions. Our experiments show that preventing explosive growth at the final layer at initialization and during training can recover a large part of BatchNorm's generalization boost. This regularization mechanism can lift accuracy by $2.9\%$ for Resnet-50 on Imagenet without BatchNorm. We show it is linked to other methods like Dropout and recent initializations like Fixup. Surprisingly, this simple mechanism matches the improvement of $0.9\%$ of the more complex Dropout regularization for the state-of-the-art Efficientnet-B8 model on Imagenet. This demonstrates the underrated effectiveness of simple regularizations and sheds light on directions to further improve generalization for deep nets.

----

## [856] PolarNet: Learning to Optimize Polar Keypoints for Keypoint Based Object Detection

**Authors**: *Xiongwei Wu, Doyen Sahoo, Steven C. H. Hoi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=TYXs_y84xRj](https://openreview.net/forum?id=TYXs_y84xRj)

**Abstract**:

A variety of anchor-free object detectors have been actively proposed as possible alternatives to the mainstream anchor-based detectors that often rely on complicated design of anchor boxes. Despite achieving promising performance on par with anchor-based detectors, the existing anchor-free detectors such as FCOS or CenterNet predict objects based on standard Cartesian coordinates, which often yield poor quality keypoints. Further, the feature representation is also scale-sensitive. In this paper, we propose a new anchor-free keypoint based detector ``PolarNet", where keypoints are represented as a set of Polar coordinates instead of Cartesian coordinates. The ``PolarNet" detector learns offsets pointing to the corners of objects in order to learn high quality keypoints. Additionally, PolarNet uses features of corner points to localize objects, making the localization scale-insensitive. Finally in our experiments, we show that PolarNet, an anchor-free detector, outperforms the existing anchor-free detectors, and it is able to achieve highly competitive result on COCO test-dev benchmark ($47.8\%$ and $50.3\%$ AP under the single-model single-scale and multi-scale testing) which is on par with the state-of-the-art two-stage anchor-based object detectors. The code and the models are available at https://github.com/XiongweiWu/PolarNetV1

----

## [857] Simple Spectral Graph Convolution

**Authors**: *Hao Zhu, Piotr Koniusz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=CYO5T-YjWZV](https://openreview.net/forum?id=CYO5T-YjWZV)

**Abstract**:

Graph Convolutional Networks (GCNs) are leading methods for learning graph representations. However, without specially designed architectures, the performance of GCNs degrades quickly with increased depth. As the aggregated neighborhood size and neural network depth are two completely orthogonal aspects of graph representation, several methods focus on summarizing the neighborhood by aggregating K-hop neighborhoods of nodes while using shallow neural networks. However, these methods still encounter oversmoothing, and suffer from high computation and storage costs. In this paper, we use a modified Markov Diffusion Kernel to derive a variant of GCN called Simple Spectral Graph Convolution (SSGC). Our spectral analysis shows that our simple spectral graph convolution used in SSGC is a trade-off of low- and high-pass filter bands which capture the global and local contexts of each node. We provide two theoretical claims which demonstrate that we can aggregate over a sequence of increasingly larger neighborhoods compared to competitors while limiting severe oversmoothing.  Our experimental evaluations show that SSGC with a linear learner is competitive in text and node classification tasks. Moreover, SSGC is comparable to other state-of-the-art methods for node clustering and community prediction tasks.

----

## [858] Explainable Subgraph Reasoning for Forecasting on Temporal Knowledge Graphs

**Authors**: *Zhen Han, Peng Chen, Yunpu Ma, Volker Tresp*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pGIHq1m7PU](https://openreview.net/forum?id=pGIHq1m7PU)

**Abstract**:

Modeling time-evolving knowledge graphs (KGs) has recently gained increasing interest. Here, graph representation learning has become the dominant paradigm for link prediction on temporal KGs. However, the embedding-based approaches largely operate in a black-box fashion, lacking the ability to interpret their predictions. This paper provides a link forecasting framework that reasons over query-relevant subgraphs of temporal KGs and jointly models the structural dependencies and the temporal dynamics. Especially, we propose a temporal relational attention mechanism and a novel reverse representation update scheme to guide the extraction of an enclosing subgraph around the query. The subgraph is expanded by an iterative sampling of temporal neighbors and by attention propagation. Our approach provides human-understandable evidence explaining the forecast. We evaluate our model on four benchmark temporal knowledge graphs for the link forecasting task. While being more explainable, our model obtains a relative improvement of up to 20 $\%$ on Hits@1 compared to the previous best temporal KG forecasting method. We also conduct a survey with 53 respondents, and the results show that the evidence extracted by the model for link forecasting is aligned with human understanding.

----

## [859] Evaluation of Neural Architectures trained with square Loss vs Cross-Entropy in Classification Tasks

**Authors**: *Like Hui, Mikhail Belkin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hsFN92eQEla](https://openreview.net/forum?id=hsFN92eQEla)

**Abstract**:

Modern neural architectures for classification tasks are trained using the cross-entropy loss, which is widely believed to be empirically superior to the square loss. In this work we provide evidence indicating that this belief may not be well-founded. 
We explore several major neural architectures and a range of standard benchmark datasets for NLP, automatic speech recognition (ASR) and computer vision tasks to show that these architectures, with the same hyper-parameter settings as reported in the literature, perform comparably or better when trained with the square loss, even after equalizing computational resources.
Indeed, we observe that the square loss produces better results in the dominant majority of NLP and ASR experiments. Cross-entropy appears to have a slight edge on computer vision tasks.

We argue that there is little compelling empirical or theoretical evidence indicating a clear-cut advantage to the cross-entropy loss. Indeed, in our experiments, performance on nearly all non-vision tasks  can be improved, sometimes significantly, by switching to the square loss. Furthermore, training with square loss appears to be less sensitive to the randomness in initialization. We posit that
training using the square loss for classification needs to be a part of best practices of modern deep learning on equal footing with cross-entropy.

----



[Go to the previous page](ICLR-2021-list04.md)

[Go to the catalog section](README.md)