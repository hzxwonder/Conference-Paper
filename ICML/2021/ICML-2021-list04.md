## [600] Uncovering the Connections Between Adversarial Transferability and Knowledge Transferability

**Authors**: *Kaizhao Liang, Jacky Y. Zhang, Boxin Wang, Zhuolin Yang, Sanmi Koyejo, Bo Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liang21b.html](http://proceedings.mlr.press/v139/liang21b.html)

**Abstract**:

Knowledge transferability, or transfer learning, has been widely adopted to allow a pre-trained model in the source domain to be effectively adapted to downstream tasks in the target domain. It is thus important to explore and understand the factors affecting knowledge transferability. In this paper, as the first work, we analyze and demonstrate the connections between knowledge transferability and another important phenomenon–adversarial transferability, \emph{i.e.}, adversarial examples generated against one model can be transferred to attack other models. Our theoretical studies show that adversarial transferability indicates knowledge transferability, and vice versa. Moreover, based on the theoretical insights, we propose two practical adversarial transferability metrics to characterize this process, serving as bidirectional indicators between adversarial and knowledge transferability. We conduct extensive experiments for different scenarios on diverse datasets, showing a positive correlation between adversarial transferability and knowledge transferability. Our findings will shed light on future research about effective knowledge transfer learning and adversarial transferability analyses.

----

## [601] Parallel Droplet Control in MEDA Biochips using Multi-Agent Reinforcement Learning

**Authors**: *Tung-Che Liang, Jin Zhou, Yun-Sheng Chan, Tsung-Yi Ho, Krishnendu Chakrabarty, Cy Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liang21c.html](http://proceedings.mlr.press/v139/liang21c.html)

**Abstract**:

Microfluidic biochips are being utilized for clinical diagnostics, including COVID-19 testing, because of they provide sample-to-result turnaround at low cost. Recently, microelectrode-dot-array (MEDA) biochips have been proposed to advance microfluidics technology. A MEDA biochip manipulates droplets of nano/picoliter volumes to automatically execute biochemical protocols. During bioassay execution, droplets are transported in parallel to achieve high-throughput outcomes. However, a major concern associated with the use of MEDA biochips is microelectrode degradation over time. Recent work has shown that formulating droplet transportation as a reinforcement-learning (RL) problem enables the training of policies to capture the underlying health conditions of microelectrodes and ensure reliable fluidic operations. However, the above RL-based approach suffers from two key limitations: 1) it cannot be used for concurrent transportation of multiple droplets; 2) it requires the availability of CCD cameras for monitoring droplet movement. To overcome these problems, we present a multi-agent reinforcement learning (MARL) droplet-routing solution that can be used for various sizes of MEDA biochips with integrated sensors, and we demonstrate the reliable execution of a serial-dilution bioassay with the MARL droplet router on a fabricated MEDA biochip. To facilitate further research, we also present a simulation environment based on the PettingZoo Gym Interface for MARL-guided droplet-routing problems on MEDA biochips.

----

## [602] Information Obfuscation of Graph Neural Networks

**Authors**: *Peiyuan Liao, Han Zhao, Keyulu Xu, Tommi S. Jaakkola, Geoffrey J. Gordon, Stefanie Jegelka, Ruslan Salakhutdinov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liao21a.html](http://proceedings.mlr.press/v139/liao21a.html)

**Abstract**:

While the advent of Graph Neural Networks (GNNs) has greatly improved node and graph representation learning in many applications, the neighborhood aggregation scheme exposes additional vulnerabilities to adversaries seeking to extract node-level information about sensitive attributes. In this paper, we study the problem of protecting sensitive attributes by information obfuscation when learning with graph structured data. We propose a framework to locally filter out pre-determined sensitive attributes via adversarial training with the total variation and the Wasserstein distance. Our method creates a strong defense against inference attacks, while only suffering small loss in task performance. Theoretically, we analyze the effectiveness of our framework against a worst-case adversary, and characterize an inherent trade-off between maximizing predictive accuracy and minimizing information leakage. Experiments across multiple datasets from recommender systems, knowledge graphs and quantum chemistry demonstrate that the proposed approach provides a robust defense across various graph structures and tasks, while producing competitive GNN encoders for downstream tasks.

----

## [603] Guided Exploration with Proximal Policy Optimization using a Single Demonstration

**Authors**: *Gabriele Libardi, Gianni De Fabritiis, Sebastian Dittert*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/libardi21a.html](http://proceedings.mlr.press/v139/libardi21a.html)

**Abstract**:

Solving sparse reward tasks through exploration is one of the major challenges in deep reinforcement learning, especially in three-dimensional, partially-observable environments. Critically, the algorithm proposed in this article is capable of using a single human demonstration to solve hard-exploration problems. We train an agent on a combination of demonstrations and own experience to solve problems with variable initial conditions and we integrate it with proximal policy optimization (PPO). The agent is also able to increase its performance and to tackle harder problems by replaying its own past trajectories prioritizing them based on the obtained reward and the maximum value of the trajectory. We finally compare variations of this algorithm to different imitation learning algorithms on a set of hard-exploration tasks in the Animal-AI Olympics environment. To the best of our knowledge, learning a task in a three-dimensional environment with comparable difficulty has never been considered before using only one human demonstration.

----

## [604] Debiasing a First-order Heuristic for Approximate Bi-level Optimization

**Authors**: *Valerii Likhosherstov, Xingyou Song, Krzysztof Choromanski, Jared Quincy Davis, Adrian Weller*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/likhosherstov21a.html](http://proceedings.mlr.press/v139/likhosherstov21a.html)

**Abstract**:

Approximate bi-level optimization (ABLO) consists of (outer-level) optimization problems, involving numerical (inner-level) optimization loops. While ABLO has many applications across deep learning, it suffers from time and memory complexity proportional to the length $r$ of its inner optimization loop. To address this complexity, an earlier first-order method (FOM) was proposed as a heuristic which omits second derivative terms, yielding significant speed gains and requiring only constant memory. Despite FOM’s popularity, there is a lack of theoretical understanding of its convergence properties. We contribute by theoretically characterizing FOM’s gradient bias under mild assumptions. We further demonstrate a rich family of examples where FOM-based SGD does not converge to a stationary point of the ABLO objective. We address this concern by proposing an unbiased FOM (UFOM) enjoying constant memory complexity as a function of $r$. We characterize the introduced time-variance tradeoff, demonstrate convergence bounds, and find an optimal UFOM for a given ABLO problem. Finally, we propose an efficient adaptive UFOM scheme.

----

## [605] Making transport more robust and interpretable by moving data through a small number of anchor points

**Authors**: *Chi-Heng Lin, Mehdi Azabou, Eva L. Dyer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lin21a.html](http://proceedings.mlr.press/v139/lin21a.html)

**Abstract**:

Optimal transport (OT) is a widely used technique for distribution alignment, with applications throughout the machine learning, graphics, and vision communities. Without any additional structural assumptions on transport, however, OT can be fragile to outliers or noise, especially in high dimensions. Here, we introduce Latent Optimal Transport (LOT), a new approach for OT that simultaneously learns low-dimensional structure in data while leveraging this structure to solve the alignment task. The idea behind our approach is to learn two sets of “anchors” that constrain the flow of transport between a source and target distribution. In both theoretical and empirical studies, we show that LOT regularizes the rank of transport and makes it more robust to outliers and the sampling density. We show that by allowing the source and target to have different anchors, and using LOT to align the latent spaces between anchors, the resulting transport plan has better structural interpretability and highlights connections between both the individual data points and the local geometry of the datasets.

----

## [606] Straight to the Gradient: Learning to Use Novel Tokens for Neural Text Generation

**Authors**: *Xiang Lin, Simeng Han, Shafiq R. Joty*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lin21b.html](http://proceedings.mlr.press/v139/lin21b.html)

**Abstract**:

Advanced large-scale neural language models have led to significant success in many language generation tasks. However, the most commonly used training objective, Maximum Likelihood Estimation (MLE), has been shown problematic, where the trained model prefers using dull and repetitive phrases. In this work, we introduce ScaleGrad, a modification straight to the gradient of the loss function, to remedy the degeneration issue of the standard MLE objective. By directly maneuvering the gradient information, ScaleGrad makes the model learn to use novel tokens. Empirical results show the effectiveness of our method not only in open-ended generation, but also in directed generation tasks. With the simplicity in architecture, our method can serve as a general training objective that is applicable to most of the neural text generation tasks.

----

## [607] Quasi-global Momentum: Accelerating Decentralized Deep Learning on Heterogeneous Data

**Authors**: *Tao Lin, Sai Praneeth Karimireddy, Sebastian U. Stich, Martin Jaggi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lin21c.html](http://proceedings.mlr.press/v139/lin21c.html)

**Abstract**:

Decentralized training of deep learning models is a key element for enabling data privacy and on-device learning over networks. In realistic learning scenarios, the presence of heterogeneity across different clients’ local datasets poses an optimization challenge and may severely deteriorate the generalization performance. In this paper, we investigate and identify the limitation of several decentralized optimization algorithms for different degrees of data heterogeneity. We propose a novel momentum-based method to mitigate this decentralized training difficulty. We show in extensive empirical experiments on various CV/NLP datasets (CIFAR-10, ImageNet, and AG News) and several network topologies (Ring and Social Network) that our method is much more robust to the heterogeneity of clients’ data than other existing methods, by a significant improvement in test performance (1%-20%).

----

## [608] Generative Causal Explanations for Graph Neural Networks

**Authors**: *Wanyu Lin, Hao Lan, Baochun Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lin21d.html](http://proceedings.mlr.press/v139/lin21d.html)

**Abstract**:

This paper presents {\em Gem}, a model-agnostic approach for providing interpretable explanations for any GNNs on various graph learning tasks. Specifically, we formulate the problem of providing explanations for the decisions of GNNs as a causal learning task. Then we train a causal explanation model equipped with a loss function based on Granger causality. Different from existing explainers for GNNs, {\em Gem} explains GNNs on graph-structured data from a causal perspective. It has better generalization ability as it has no requirements on the internal structure of the GNNs or prior knowledge on the graph learning tasks. In addition, {\em Gem}, once trained, can be used to explain the target GNN very quickly. Our theoretical analysis shows that several recent explainers fall into a unified framework of {\em additive feature attribution methods}. Experimental results on synthetic and real-world datasets show that {\em Gem} achieves a relative increase of the explanation accuracy by up to $30%$ and speeds up the explanation process by up to $110\times$ as compared to its state-of-the-art alternatives.

----

## [609] Tractable structured natural-gradient descent using local parameterizations

**Authors**: *Wu Lin, Frank Nielsen, Mohammad Emtiyaz Khan, Mark Schmidt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lin21e.html](http://proceedings.mlr.press/v139/lin21e.html)

**Abstract**:

Natural-gradient descent (NGD) on structured parameter spaces (e.g., low-rank covariances) is computationally challenging due to difficult Fisher-matrix computations. We address this issue by using \emph{local-parameter coordinates} to obtain a flexible and efficient NGD method that works well for a wide-variety of structured parameterizations. We show four applications where our method (1) generalizes the exponential natural evolutionary strategy, (2) recovers existing Newton-like algorithms, (3) yields new structured second-order algorithms, and (4) gives new algorithms to learn covariances of Gaussian and Wishart-based distributions. We show results on a range of problems from deep learning, variational inference, and evolution strategies. Our work opens a new direction for scalable structured geometric methods.

----

## [610] Active Learning of Continuous-time Bayesian Networks through Interventions

**Authors**: *Dominik Linzner, Heinz Koeppl*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/linzner21a.html](http://proceedings.mlr.press/v139/linzner21a.html)

**Abstract**:

We consider the problem of learning structures and parameters of Continuous-time Bayesian Networks (CTBNs) from time-course data under minimal experimental resources. In practice, the cost of generating experimental data poses a bottleneck, especially in the natural and social sciences. A popular approach to overcome this is Bayesian optimal experimental design (BOED). However, BOED becomes infeasible in high-dimensional settings, as it involves integration over all possible experimental outcomes. We propose a novel criterion for experimental design based on a variational approximation of the expected information gain. We show that for CTBNs, a semi-analytical expression for this criterion can be calculated for structure and parameter learning. By doing so, we can replace sampling over experimental outcomes by solving the CTBNs master-equation, for which scalable approximations exist. This alleviates the computational burden of sampling possible experimental outcomes in high-dimensions. We employ this framework to recommend interventional sequences. In this context, we extend the CTBN model to conditional CTBNs to incorporate interventions. We demonstrate the performance of our criterion on synthetic and real-world data.

----

## [611] Phase Transitions, Distance Functions, and Implicit Neural Representations

**Authors**: *Yaron Lipman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lipman21a.html](http://proceedings.mlr.press/v139/lipman21a.html)

**Abstract**:

Representing surfaces as zero level sets of neural networks recently emerged as a powerful modeling paradigm, named Implicit Neural Representations (INRs), serving numerous downstream applications in geometric deep learning and 3D vision. Training INRs previously required choosing between occupancy and distance function representation and different losses with unknown limit behavior and/or bias. In this paper we draw inspiration from the theory of phase transitions of fluids and suggest a loss for training INRs that learns a density function that converges to a proper occupancy function, while its log transform converges to a distance function. Furthermore, we analyze the limit minimizer of this loss showing it satisfies the reconstruction constraints and has minimal surface perimeter, a desirable inductive bias for surface reconstruction. Training INRs with this new loss leads to state-of-the-art reconstructions on a standard benchmark.

----

## [612] The Earth Mover's Pinball Loss: Quantiles for Histogram-Valued Regression

**Authors**: *Florian List*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/list21a.html](http://proceedings.mlr.press/v139/list21a.html)

**Abstract**:

Although ubiquitous in the sciences, histogram data have not received much attention by the Deep Learning community. Whilst regression and classification tasks for scalar and vector data are routinely solved by neural networks, a principled approach for estimating histogram labels as a function of an input vector or image is lacking in the literature. We present a dedicated method for Deep Learning-based histogram regression, which incorporates cross-bin information and yields distributions over possible histograms, expressed by $\tau$-quantiles of the cumulative histogram in each bin. The crux of our approach is a new loss function obtained by applying the pinball loss to the cumulative histogram, which for 1D histograms reduces to the Earth Mover’s distance (EMD) in the special case of the median ($\tau = 0.5$), and generalizes it to arbitrary quantiles. We validate our method with an illustrative toy example, a football-related task, and an astrophysical computer vision problem. We show that with our loss function, the accuracy of the predicted median histograms is very similar to the standard EMD case (and higher than for per-bin loss functions such as cross-entropy), while the predictions become much more informative at almost no additional computational cost.

----

## [613] Understanding Instance-Level Label Noise: Disparate Impacts and Treatments

**Authors**: *Yang Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21a.html](http://proceedings.mlr.press/v139/liu21a.html)

**Abstract**:

This paper aims to provide understandings for the effect of an over-parameterized model, e.g. a deep neural network, memorizing instance-dependent noisy labels. We first quantify the harms caused by memorizing noisy instances, and show the disparate impacts of noisy labels for sample instances with different representation frequencies. We then analyze how several popular solutions for learning with noisy labels mitigate this harm at the instance level. Our analysis reveals that existing approaches lead to disparate treatments when handling noisy instances. While higher-frequency instances often enjoy a high probability of an improvement by applying these solutions, lower-frequency instances do not. Our analysis reveals new understandings for when these approaches work, and provides theoretical justifications for previously reported empirical observations. This observation requires us to rethink the distribution of label noise across instances and calls for different treatments for instances in different regimes.

----

## [614] APS: Active Pretraining with Successor Features

**Authors**: *Hao Liu, Pieter Abbeel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21b.html](http://proceedings.mlr.press/v139/liu21b.html)

**Abstract**:

We introduce a new unsupervised pretraining objective for reinforcement learning. During the unsupervised reward-free pretraining phase, the agent maximizes mutual information between tasks and states induced by the policy. Our key contribution is a novel lower bound of this intractable quantity. We show that by reinterpreting and combining variational successor features \citep{Hansen2020Fast} with nonparametric entropy maximization \citep{liu2021behavior}, the intractable mutual information can be efficiently optimized. The proposed method Active Pretraining with Successor Feature (APS) explores the environment via nonparametric entropy maximization, and the explored data can be efficiently leveraged to learn behavior by variational successor features. APS addresses the limitations of existing mutual information maximization based and entropy maximization based unsupervised RL, and combines the best of both worlds. When evaluated on the Atari 100k data-efficiency benchmark, our approach significantly outperforms previous methods combining unsupervised pretraining with task-specific finetuning.

----

## [615] Learning by Turning: Neural Architecture Aware Optimisation

**Authors**: *Yang Liu, Jeremy Bernstein, Markus Meister, Yisong Yue*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21c.html](http://proceedings.mlr.press/v139/liu21c.html)

**Abstract**:

Descent methods for deep networks are notoriously capricious: they require careful tuning of step size, momentum and weight decay, and which method will work best on a new benchmark is a priori unclear. To address this problem, this paper conducts a combined study of neural architecture and optimisation, leading to a new optimiser called Nero: the neuronal rotator. Nero trains reliably without momentum or weight decay, works in situations where Adam and SGD fail, and requires little to no learning rate tuning. Also, Nero’s memory footprint is   square root that of Adam or LAMB. Nero combines two ideas: (1) projected gradient descent over the space of balanced networks; (2) neuron-specific updates, where the step size sets the angle through which each neuron’s hyperplane turns. The paper concludes by discussing how this geometric connection between architecture and optimisation may impact theories of generalisation in deep learning.

----

## [616] Dynamic Game Theoretic Neural Optimizer

**Authors**: *Guan-Horng Liu, Tianrong Chen, Evangelos A. Theodorou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21d.html](http://proceedings.mlr.press/v139/liu21d.html)

**Abstract**:

The connection between training deep neural networks (DNNs) and optimal control theory (OCT) has attracted considerable attention as a principled tool of algorithmic design. Despite few attempts being made, they have been limited to architectures where the layer propagation resembles a Markovian dynamical system. This casts doubts on their flexibility to modern networks that heavily rely on non-Markovian dependencies between layers (e.g. skip connections in residual networks). In this work, we propose a novel dynamic game perspective by viewing each layer as a player in a dynamic game characterized by the DNN itself. Through this lens, different classes of optimizers can be seen as matching different types of Nash equilibria, depending on the implicit information structure of each (p)layer. The resulting method, called Dynamic Game Theoretic Neural Optimizer (DGNOpt), not only generalizes OCT-inspired optimizers to richer network class; it also motivates a new training principle by solving a multi-player cooperative game. DGNOpt shows convergence improvements over existing methods on image classification datasets with residual and inception networks. Our work marries strengths from both OCT and game theory, paving ways to new algorithmic opportunities from robust optimal control and bandit-based optimization.

----

## [617] Besov Function Approximation and Binary Classification on Low-Dimensional Manifolds Using Convolutional Residual Networks

**Authors**: *Hao Liu, Minshuo Chen, Tuo Zhao, Wenjing Liao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21e.html](http://proceedings.mlr.press/v139/liu21e.html)

**Abstract**:

Most of existing statistical theories on deep neural networks have sample complexities cursed by the data dimension and therefore cannot well explain the empirical success of deep learning on high-dimensional data. To bridge this gap, we propose to exploit the low-dimensional structures of the real world datasets and establish theoretical guarantees of convolutional residual networks (ConvResNet) in terms of function approximation and statistical recovery for binary classification problem. Specifically, given the data lying on a $d$-dimensional manifold isometrically embedded in $\mathbb{R}^D$, we prove that if the network architecture is properly chosen, ConvResNets can (1) approximate {\it Besov functions} on manifolds with arbitrary accuracy, and (2) learn a classifier by minimizing the empirical logistic risk, which gives an {\it excess risk} in the order of $n^{-\frac{s}{2s+2(s\vee d)}}$, where $s$ is a smoothness parameter. This implies that the sample complexity depends on the intrinsic dimension $d$, instead of the data dimension $D$. Our results demonstrate that ConvResNets are adaptive to low-dimensional structures of data sets.

----

## [618] Just Train Twice: Improving Group Robustness without Training Group Information

**Authors**: *Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21f.html](http://proceedings.mlr.press/v139/liu21f.html)

**Abstract**:

Standard training via empirical risk minimization (ERM) can produce models that achieve low error on average but high error on minority groups, especially in the presence of spurious correlations between the input and label. Prior approaches to this problem, like group distributionally robust optimization (group DRO), generally require group annotations for every training point. On the other hand, approaches that do not use group annotations generally do not improve minority performance. For example, we find that joint DRO, which dynamically upweights examples with high training loss, tends to optimize for examples that are irrelevant to the specific groups we seek to do well on. In this paper, we propose a simple two-stage approach, JTT, that achieves comparable performance to group DRO while only requiring group annotations on a significantly smaller validation set. JTT first attempts to identify informative training examples, which are often minority examples, by training an initial ERM classifier and selecting the examples with high training loss. Then, it trains a final classifier by upsampling the selected examples. Crucially, unlike joint DRO, JTT does not iteratively upsample examples that have high loss under the final classifier. On four image classification and natural language processing tasks with spurious correlations, we show that JTT closes 85% of the gap in accuracy on the worst group between ERM and group DRO.

----

## [619] Event Outlier Detection in Continuous Time

**Authors**: *Siqi Liu, Milos Hauskrecht*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21g.html](http://proceedings.mlr.press/v139/liu21g.html)

**Abstract**:

Continuous-time event sequences represent discrete events occurring in continuous time. Such sequences arise frequently in real-life. Usually we expect the sequences to follow some regular pattern over time. However, sometimes these patterns may be interrupted by unexpected absence or occurrences of events. Identification of these unexpected cases can be very important as they may point to abnormal situations that need human attention. In this work, we study and develop methods for detecting outliers in continuous-time event sequences, including unexpected absence and unexpected occurrences of events. Since the patterns that event sequences tend to follow may change in different contexts, we develop outlier detection methods based on point processes that can take context information into account. Our methods are based on Bayesian decision theory and hypothesis testing with theoretical guarantees. To test the performance of the methods, we conduct experiments on both synthetic data and real-world clinical data and show the effectiveness of the proposed methods.

----

## [620] Heterogeneous Risk Minimization

**Authors**: *Jiashuo Liu, Zheyuan Hu, Peng Cui, Bo Li, Zheyan Shen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21h.html](http://proceedings.mlr.press/v139/liu21h.html)

**Abstract**:

Machine learning algorithms with empirical risk minimization usually suffer from poor generalization performance due to the greedy exploitation of correlations among the training data, which are not stable under distributional shifts. Recently, some invariant learning methods for out-of-distribution (OOD) generalization have been proposed by leveraging multiple training environments to find invariant relationships. However, modern datasets are frequently assembled by merging data from multiple sources without explicit source labels. The resultant unobserved heterogeneity renders many invariant learning methods inapplicable. In this paper, we propose Heterogeneous Risk Minimization (HRM) framework to achieve joint learning of latent heterogeneity among the data and invariant relationship, which leads to stable prediction despite distributional shifts. We theoretically characterize the roles of the environment labels in invariant learning and justify our newly proposed HRM framework. Extensive experimental results validate the effectiveness of our HRM framework.

----

## [621] Stochastic Iterative Graph Matching

**Authors**: *Linfeng Liu, Michael C. Hughes, Soha Hassoun, Liping Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21i.html](http://proceedings.mlr.press/v139/liu21i.html)

**Abstract**:

Recent works apply Graph Neural Networks (GNNs) to graph matching tasks and show promising results. Considering that model outputs are complex matchings, we devise several techniques to improve the learning of GNNs and obtain a new model, Stochastic Iterative Graph MAtching (SIGMA). Our model predicts a distribution of matchings, instead of a single matching, for a graph pair so the model can explore several probable matchings. We further introduce a novel multi-step matching procedure, which learns how to refine a graph pair’s matching results incrementally. The model also includes dummy nodes so that the model does not have to find matchings for nodes without correspondence. We fit this model to data via scalable stochastic optimization. We conduct extensive experiments across synthetic graph datasets as well as biochemistry and computer vision applications. Across all tasks, our results show that SIGMA can produce significantly improved graph matching results compared to state-of-the-art models. Ablation studies verify that each of our components (stochastic training, iterative matching, and dummy nodes) offers noticeable improvement.

----

## [622] Cooperative Exploration for Multi-Agent Deep Reinforcement Learning

**Authors**: *Iou-Jen Liu, Unnat Jain, Raymond A. Yeh, Alexander G. Schwing*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21j.html](http://proceedings.mlr.press/v139/liu21j.html)

**Abstract**:

Exploration is critical for good results in deep reinforcement learning and has attracted much attention. However, existing multi-agent deep reinforcement learning algorithms still use mostly noise-based techniques. Very recently, exploration methods that consider cooperation among multiple agents have been developed. However, existing methods suffer from a common challenge: agents struggle to identify states that are worth exploring, and hardly coordinate exploration efforts toward those states. To address this shortcoming, in this paper, we propose cooperative multi-agent exploration (CMAE): agents share a common goal while exploring. The goal is selected from multiple projected state spaces by a normalized entropy-based technique. Then, agents are trained to reach the goal in a coordinated manner. We demonstrate that CMAE consistently outperforms baselines on various tasks, including a sparse-reward version of multiple-particle environment (MPE) and the Starcraft multi-agent challenge (SMAC).

----

## [623] Elastic Graph Neural Networks

**Authors**: *Xiaorui Liu, Wei Jin, Yao Ma, Yaxin Li, Hua Liu, Yiqi Wang, Ming Yan, Jiliang Tang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21k.html](http://proceedings.mlr.press/v139/liu21k.html)

**Abstract**:

While many existing graph neural networks (GNNs) have been proven to perform $\ell_2$-based graph smoothing that enforces smoothness globally, in this work we aim to further enhance the local smoothness adaptivity of GNNs via $\ell_1$-based graph smoothing. As a result, we introduce a family of GNNs (Elastic GNNs) based on $\ell_1$ and $\ell_2$-based graph smoothing. In particular, we propose a novel and general message passing scheme into GNNs. This message passing algorithm is not only friendly to back-propagation training but also achieves the desired smoothing properties with a theoretical convergence guarantee. Experiments on semi-supervised learning tasks demonstrate that the proposed Elastic GNNs obtain better adaptivity on benchmark datasets and are significantly robust to graph adversarial attacks. The implementation of Elastic GNNs is available at \url{https://github.com/lxiaorui/ElasticGNN}.

----

## [624] One Pass Late Fusion Multi-view Clustering

**Authors**: *Xinwang Liu, Li Liu, Qing Liao, Siwei Wang, Yi Zhang, Wenxuan Tu, Chang Tang, Jiyuan Liu, En Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21l.html](http://proceedings.mlr.press/v139/liu21l.html)

**Abstract**:

Existing late fusion multi-view clustering (LFMVC) optimally integrates a group of pre-specified base partition matrices to learn a consensus one. It is then taken as the input of the widely used k-means to generate the cluster labels. As observed, the learning of the consensus partition matrix and the generation of cluster labels are separately done. These two procedures lack necessary negotiation and can not best serve for each other, which may adversely affect the clustering performance. To address this issue, we propose to unify the aforementioned two learning procedures into a single optimization, in which the consensus partition matrix can better serve for the generation of cluster labels, and the latter is able to guide the learning of the former. To optimize the resultant optimization problem, we develop a four-step alternate algorithm with proved convergence. We theoretically analyze the clustering generalization error of the proposed algorithm on unseen data. Comprehensive experiments on multiple benchmark datasets demonstrate the superiority of our algorithm in terms of both clustering accuracy and computational efficiency. It is expected that the simplicity and effectiveness of our algorithm will make it a good option to be considered for practical multi-view clustering applications.

----

## [625] Coach-Player Multi-agent Reinforcement Learning for Dynamic Team Composition

**Authors**: *Bo Liu, Qiang Liu, Peter Stone, Animesh Garg, Yuke Zhu, Anima Anandkumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21m.html](http://proceedings.mlr.press/v139/liu21m.html)

**Abstract**:

In real-world multi-agent systems, agents with different capabilities may join or leave without altering the team’s overarching goals. Coordinating teams with such dynamic composition is challenging: the optimal team strategy varies with the composition. We propose COPA, a coach-player framework to tackle this problem. We assume the coach has a global view of the environment and coordinates the players, who only have partial views, by distributing individual strategies. Specifically, we 1) adopt the attention mechanism for both the coach and the players; 2) propose a variational objective to regularize learning; and 3) design an adaptive communication method to let the coach decide when to communicate with the players. We validate our methods on a resource collection task, a rescue game, and the StarCraft micromanagement tasks. We demonstrate zero-shot generalization to new team compositions. Our method achieves comparable or better performance than the setting where all players have a full view of the environment. Moreover, we see that the performance remains high even when the coach communicates as little as 13% of the time using the adaptive communication strategy.

----

## [626] From Local to Global Norm Emergence: Dissolving Self-reinforcing Substructures with Incremental Social Instruments

**Authors**: *Yiwei Liu, Jiamou Liu, Kaibin Wan, Zhan Qin, Zijian Zhang, Bakhadyr Khoussainov, Liehuang Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21n.html](http://proceedings.mlr.press/v139/liu21n.html)

**Abstract**:

Norm emergence is a process where agents in a multi-agent system establish self-enforcing conformity through repeated interactions. When such interactions are confined to a social topology, several self-reinforcing substructures (SRS) may emerge within the population. This prevents a formation of a global norm. We propose incremental social instruments (ISI) to dissolve these SRSs by creating ties between agents. Establishing ties requires some effort and cost. Hence, it is worth to design methods that build a small number of ties yet dissolve the SRSs. By using the notion of information entropy, we propose an indicator called the BA-ratio that measures the current SRSs. We find that by building ties with minimal BA-ratio, our ISI is effective in facilitating the global norm emergence. We explain this through our experiments and theoretical results. Furthermore, we propose the small-degree principle in minimising the BA-ratio that helps us to design efficient ISI algorithms for finding the optimal ties. Experiments on both synthetic and real-world network topologies demonstrate that our adaptive ISI is efficient at dissolving SRS.

----

## [627] A Value-Function-based Interior-point Method for Non-convex Bi-level Optimization

**Authors**: *Risheng Liu, Xuan Liu, Xiaoming Yuan, Shangzhi Zeng, Jin Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21o.html](http://proceedings.mlr.press/v139/liu21o.html)

**Abstract**:

Bi-level optimization model is able to capture a wide range of complex learning tasks with practical interest. Due to the witnessed efficiency in solving bi-level programs, gradient-based methods have gained popularity in the machine learning community. In this work, we propose a new gradient-based solution scheme, namely, the Bi-level Value-Function-based Interior-point Method (BVFIM). Following the main idea of the log-barrier interior-point scheme, we penalize the regularized value function of the lower level problem into the upper level objective. By further solving a sequence of differentiable unconstrained approximation problems, we consequently derive a sequential programming scheme. The numerical advantage of our scheme relies on the fact that, when gradient methods are applied to solve the approximation problem, we successfully avoid computing any expensive Hessian-vector or Jacobian-vector product. We prove the convergence without requiring any convexity assumption on either the upper level or the lower level objective. Experiments demonstrate the efficiency of the proposed BVFIM on non-convex bi-level problems.

----

## [628] Selfish Sparse RNN Training

**Authors**: *Shiwei Liu, Decebal Constantin Mocanu, Yulong Pei, Mykola Pechenizkiy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21p.html](http://proceedings.mlr.press/v139/liu21p.html)

**Abstract**:

Sparse neural networks have been widely applied to reduce the computational demands of training and deploying over-parameterized deep neural networks. For inference acceleration, methods that discover a sparse network from a pre-trained dense network (dense-to-sparse training) work effectively. Recently, dynamic sparse training (DST) has been proposed to train sparse neural networks without pre-training a dense model (sparse-to-sparse training), so that the training process can also be accelerated. However, previous sparse-to-sparse methods mainly focus on Multilayer Perceptron Networks (MLPs) and Convolutional Neural Networks (CNNs), failing to match the performance of dense-to-sparse methods in the Recurrent Neural Networks (RNNs) setting. In this paper, we propose an approach to train intrinsically sparse RNNs with a fixed parameter count in one single run, without compromising performance. During training, we allow RNN layers to have a non-uniform redistribution across cell gates for better regularization. Further, we propose SNT-ASGD, a novel variant of the averaged stochastic gradient optimizer, which significantly improves the performance of all sparse training methods for RNNs. Using these strategies, we achieve state-of-the-art sparse training results, better than the dense-to-sparse methods, with various types of RNNs on Penn TreeBank and Wikitext-2 datasets. Our codes are available at https://github.com/Shiweiliuiiiiiii/Selfish-RNN.

----

## [629] Temporal Difference Learning as Gradient Splitting

**Authors**: *Rui Liu, Alex Olshevsky*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21q.html](http://proceedings.mlr.press/v139/liu21q.html)

**Abstract**:

Temporal difference learning with linear function approximation is a popular method to obtain a low-dimensional approximation of the value function of a policy in a Markov Decision Process. We provide an interpretation of this method in terms of a splitting of the gradient of an appropriately chosen function. As a consequence of this interpretation, convergence proofs for gradient descent can be applied almost verbatim to temporal difference learning. Beyond giving a fuller explanation of why temporal difference works, this interpretation also yields improved convergence times. We consider the setting with $1/\sqrt{T}$ step-size, where previous comparable finite-time convergence time bounds for temporal difference learning had the multiplicative factor $1/(1-\gamma)$ in front of the bound, with $\gamma$ being the discount factor. We show that a minor variation on TD learning which estimates the mean of the value function separately has a convergence time where $1/(1-\gamma)$ only multiplies an asymptotically negligible term.

----

## [630] On Robust Mean Estimation under Coordinate-level Corruption

**Authors**: *Zifan Liu, Jongho Park, Theodoros Rekatsinas, Christos Tzamos*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21r.html](http://proceedings.mlr.press/v139/liu21r.html)

**Abstract**:

We study the problem of robust mean estimation and introduce a novel Hamming distance-based measure of distribution shift for coordinate-level corruptions. We show that this measure yields adversary models that capture more realistic corruptions than those used in prior works, and present an information-theoretic analysis of robust mean estimation in these settings. We show that for structured distributions, methods that leverage the structure yield information theoretically more accurate mean estimation. We also focus on practical algorithms for robust mean estimation and study when data cleaning-inspired approaches that first fix corruptions in the input data and then perform robust mean estimation can match the information theoretic bounds of our analysis. We finally demonstrate experimentally that this two-step approach outperforms structure-agnostic robust estimation and provides accurate mean estimation even for high-magnitude corruption.

----

## [631] Decoupling Exploration and Exploitation for Meta-Reinforcement Learning without Sacrifices

**Authors**: *Evan Zheran Liu, Aditi Raghunathan, Percy Liang, Chelsea Finn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21s.html](http://proceedings.mlr.press/v139/liu21s.html)

**Abstract**:

The goal of meta-reinforcement learning (meta-RL) is to build agents that can quickly learn new tasks by leveraging prior experience on related tasks. Learning a new task often requires both exploring to gather task-relevant information and exploiting this information to solve the task. In principle, optimal exploration and exploitation can be learned end-to-end by simply maximizing task performance. However, such meta-RL approaches struggle with local optima due to a chicken-and-egg problem: learning to explore requires good exploitation to gauge the exploration’s utility, but learning to exploit requires information gathered via exploration. Optimizing separate objectives for exploration and exploitation can avoid this problem, but prior meta-RL exploration objectives yield suboptimal policies that gather information irrelevant to the task. We alleviate both concerns by constructing an exploitation objective that automatically identifies task-relevant information and an exploration objective to recover only this information. This avoids local optima in end-to-end training, without sacrificing optimal exploration. Empirically, DREAM substantially outperforms existing approaches on complex meta-RL problems, such as sparse-reward 3D visual navigation. Videos of DREAM: https://ezliu.github.io/dream/

----

## [632] How Do Adam and Training Strategies Help BNNs Optimization

**Authors**: *Zechun Liu, Zhiqiang Shen, Shichao Li, Koen Helwegen, Dong Huang, Kwang-Ting Cheng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21t.html](http://proceedings.mlr.press/v139/liu21t.html)

**Abstract**:

The best performing Binary Neural Networks (BNNs) are usually attained using Adam optimization and its multi-step training variants. However, to the best of our knowledge, few studies explore the fundamental reasons why Adam is superior to other optimizers like SGD for BNN optimization or provide analytical explanations that support specific training strategies. To address this, in this paper we first investigate the trajectories of gradients and weights in BNNs during the training process. We show the regularization effect of second-order momentum in Adam is crucial to revitalize the weights that are dead due to the activation saturation in BNNs. We find that Adam, through its adaptive learning rate strategy, is better equipped to handle the rugged loss surface of BNNs and reaches a better optimum with higher generalization ability. Furthermore, we inspect the intriguing role of the real-valued weights in binary networks, and reveal the effect of weight decay on the stability and sluggishness of BNN optimization. Through extensive experiments and analysis, we derive a simple training scheme, building on existing Adam-based optimization, which achieves 70.5% top-1 accuracy on the ImageNet dataset using the same architecture as the state-of-the-art ReActNet while achieving 1.1% higher accuracy. Code and models are available at https://github.com/liuzechun/AdamBNN.

----

## [633] SagaNet: A Small Sample Gated Network for Pediatric Cancer Diagnosis

**Authors**: *Yuhan Liu, Shiliang Sun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21u.html](http://proceedings.mlr.press/v139/liu21u.html)

**Abstract**:

The scarcity of available samples and the high annotation cost of medical data cause a bottleneck in many digital diagnosis tasks based on deep learning. This problem is especially severe in pediatric tumor tasks, due to the small population base of children and high sample diversity caused by the high metastasis rate of related tumors. Targeted research on pediatric tumors is urgently needed but lacks sufficient attention. In this work, we propose a novel model to solve the diagnosis task of small round blue cell tumors (SRBCTs). To solve the problem of high noise and high diversity in the small sample scenario, the model is constrained to pay attention to the valid areas in the pathological image with a masking mechanism, and a length-aware loss is proposed to improve the tolerance to feature diversity. We evaluate this framework on a challenging small sample SRBCTs dataset, whose classification is difficult even for professional pathologists. The proposed model shows the best performance compared with state-of-the-art deep models and generalization on another pathological dataset, which illustrates the potentiality of deep learning applications in difficult small sample medical tasks.

----

## [634] Learning Deep Neural Networks under Agnostic Corrupted Supervision

**Authors**: *Boyang Liu, Mengying Sun, Ding Wang, Pang-Ning Tan, Jiayu Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21v.html](http://proceedings.mlr.press/v139/liu21v.html)

**Abstract**:

Training deep neural network models in the presence of corrupted supervision is challenging as the corrupted data points may significantly impact generalization performance. To alleviate this problem, we present an efficient robust algorithm that achieves strong guarantees without any assumption on the type of corruption and provides a unified framework for both classification and regression problems. Unlike many existing approaches that quantify the quality of the data points (e.g., based on their individual loss values), and filter them accordingly, the proposed algorithm focuses on controlling the collective impact of data points on the average gradient. Even when a corrupted data point failed to be excluded by our algorithm, the data point will have a very limited impact on the overall loss, as compared with state-of-the-art filtering methods based on loss values. Extensive experiments on multiple benchmark datasets have demonstrated the robustness of our algorithm under different types of corruption. Our code is available at \url{https://github.com/illidanlab/PRL}.

----

## [635] Leveraging Public Data for Practical Private Query Release

**Authors**: *Terrance Liu, Giuseppe Vietri, Thomas Steinke, Jonathan R. Ullman, Zhiwei Steven Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21w.html](http://proceedings.mlr.press/v139/liu21w.html)

**Abstract**:

In many statistical problems, incorporating priors can significantly improve performance. However, the use of prior knowledge in differentially private query release has remained underexplored, despite such priors commonly being available in the form of public datasets, such as previous US Census releases. With the goal of releasing statistics about a private dataset, we present PMW^Pub, which—unlike existing baselines—leverages public data drawn from a related distribution as prior information. We provide a theoretical analysis and an empirical evaluation on the American Community Survey (ACS) and ADULT datasets, which shows that our method outperforms state-of-the-art methods. Furthermore, PMW^Pub scales well to high-dimensional data domains, where running many existing methods would be computationally infeasible.

----

## [636] Watermarking Deep Neural Networks with Greedy Residuals

**Authors**: *Hanwen Liu, Zhenyu Weng, Yuesheng Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21x.html](http://proceedings.mlr.press/v139/liu21x.html)

**Abstract**:

Deep neural networks (DNNs) are considered as intellectual property of their corresponding owners and thus are in urgent need of ownership protection, due to the massive amount of time and resources invested in designing, tuning and training them. In this paper, we propose a novel watermark-based ownership protection method by using the residuals of important parameters. Different from other watermark-based ownership protection methods that rely on some specific neural network architectures and during verification require external data source, namely ownership indicators, our method does not explicitly use ownership indicators for verification to defeat various attacks against DNN watermarks. Specifically, we greedily select a few and important model parameters for embedding so that the impairment caused by the changed parameters can be reduced and the robustness against different attacks can be improved as the selected parameters can well preserve the model information. Also, without the external data sources for verification, the adversary can hardly cast doubts on ownership verification by forging counterfeit watermarks. The extensive experiments show that our method outperforms previous state-of-the-art methods in five tasks.

----

## [637] Do We Actually Need Dense Over-Parameterization? In-Time Over-Parameterization in Sparse Training

**Authors**: *Shiwei Liu, Lu Yin, Decebal Constantin Mocanu, Mykola Pechenizkiy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21y.html](http://proceedings.mlr.press/v139/liu21y.html)

**Abstract**:

In this paper, we introduce a new perspective on training deep neural networks capable of state-of-the-art performance without the need for the expensive over-parameterization by proposing the concept of In-Time Over-Parameterization (ITOP) in sparse training. By starting from a random sparse network and continuously exploring sparse connectivities during training, we can perform an Over-Parameterization over the course of training, closing the gap in the expressibility between sparse training and dense training. We further use ITOP to understand the underlying mechanism of Dynamic Sparse Training (DST) and discover that the benefits of DST come from its ability to consider across time all possible parameters when searching for the optimal sparse connectivity. As long as sufficient parameters have been reliably explored, DST can outperform the dense neural network by a large margin. We present a series of experiments to support our conjecture and achieve the state-of-the-art sparse training performance with ResNet-50 on ImageNet. More impressively, ITOP achieves dominant performance over the overparameterization-based sparse methods at extreme sparsities. When trained with ResNet-34 on CIFAR-100, ITOP can match the performance of the dense model at an extreme sparsity 98%.

----

## [638] A Sharp Analysis of Model-based Reinforcement Learning with Self-Play

**Authors**: *Qinghua Liu, Tiancheng Yu, Yu Bai, Chi Jin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21z.html](http://proceedings.mlr.press/v139/liu21z.html)

**Abstract**:

Model-based algorithms—algorithms that explore the environment through building and utilizing an estimated model—are widely used in reinforcement learning practice and theoretically shown to achieve optimal sample efficiency for single-agent reinforcement learning in Markov Decision Processes (MDPs). However, for multi-agent reinforcement learning in Markov games, the current best known sample complexity for model-based algorithms is rather suboptimal and compares unfavorably against recent model-free approaches. In this paper, we present a sharp analysis of model-based self-play algorithms for multi-agent Markov games. We design an algorithm \emph{Optimistic Nash Value Iteration} (Nash-VI) for two-player zero-sum Markov games that is able to output an $\epsilon$-approximate Nash policy in $\tilde{\mathcal{O}}(H^3SAB/\epsilon^2)$ episodes of game playing, where $S$ is the number of states, $A,B$ are the number of actions for the two players respectively, and $H$ is the horizon length. This significantly improves over the best known model-based guarantee of $\tilde{\mathcal{O}}(H^4S^2AB/\epsilon^2)$, and is the first that matches the information-theoretic lower bound $\Omega(H^3S(A+B)/\epsilon^2)$ except for a $\min\{A,B\}$ factor. In addition, our guarantee compares favorably against the best known model-free algorithm if $\min\{A,B\}=o(H^3)$, and outputs a single Markov policy while existing sample-efficient model-free algorithms output a nested mixture of Markov policies that is in general non-Markov and rather inconvenient to store and execute. We further adapt our analysis to designing a provably efficient task-agnostic algorithm for zero-sum Markov games, and designing the first line of provably sample-efficient algorithms for multi-player general-sum Markov games.

----

## [639] Lottery Ticket Preserves Weight Correlation: Is It Desirable or Not?

**Authors**: *Ning Liu, Geng Yuan, Zhengping Che, Xuan Shen, Xiaolong Ma, Qing Jin, Jian Ren, Jian Tang, Sijia Liu, Yanzhi Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21aa.html](http://proceedings.mlr.press/v139/liu21aa.html)

**Abstract**:

In deep model compression, the recent finding "Lottery Ticket Hypothesis" (LTH) pointed out that there could exist a winning ticket (i.e., a properly pruned sub-network together with original weight initialization) that can achieve competitive performance than the original dense network. However, it is not easy to observe such winning property in many scenarios, where for example, a relatively large learning rate is used even if it benefits training the original dense model. In this work, we investigate the underlying condition and rationale behind the winning property, and find that the underlying reason is largely attributed to the correlation between initialized weights and final-trained weights when the learning rate is not sufficiently large. Thus, the existence of winning property is correlated with an insufficient DNN pretraining, and is unlikely to occur for a well-trained DNN. To overcome this limitation, we propose the "pruning & fine-tuning" method that consistently outperforms lottery ticket sparse training under the same pruning algorithm and the same total training epochs. Extensive experiments over multiple deep models (VGG, ResNet, MobileNet-v2) on different datasets have been conducted to justify our proposals.

----

## [640] Group Fisher Pruning for Practical Network Compression

**Authors**: *Liyang Liu, Shilong Zhang, Zhanghui Kuang, Aojun Zhou, Jing-Hao Xue, Xinjiang Wang, Yimin Chen, Wenming Yang, Qingmin Liao, Wayne Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21ab.html](http://proceedings.mlr.press/v139/liu21ab.html)

**Abstract**:

Network compression has been widely studied since it is able to reduce the memory and computation cost during inference. However, previous methods seldom deal with complicated structures like residual connections, group/depth-wise convolution and feature pyramid network, where channels of multiple layers are coupled and need to be pruned simultaneously. In this paper, we present a general channel pruning approach that can be applied to various complicated structures. Particularly, we propose a layer grouping algorithm to find coupled channels automatically. Then we derive a unified metric based on Fisher information to evaluate the importance of a single channel and coupled channels. Moreover, we find that inference speedup on GPUs is more correlated with the reduction of memory rather than FLOPs, and thus we employ the memory reduction of each channel to normalize the importance. Our method can be used to prune any structures including those with coupled channels. We conduct extensive experiments on various backbones, including the classic ResNet and ResNeXt, mobile-friendly MobileNetV2, and the NAS-based RegNet, both on image classification and object detection which is under-explored. Experimental results validate that our method can effectively prune sophisticated networks, boosting inference speed without sacrificing accuracy.

----

## [641] Infinite-Dimensional Optimization for Zero-Sum Games via Variational Transport

**Authors**: *Lewis Liu, Yufeng Zhang, Zhuoran Yang, Reza Babanezhad, Zhaoran Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21ac.html](http://proceedings.mlr.press/v139/liu21ac.html)

**Abstract**:

Game optimization has been extensively studied when decision variables lie in a finite-dimensional space, of which solutions correspond to pure strategies at the Nash equilibrium (NE), and the gradient descent-ascent (GDA) method works widely in practice. In this paper, we consider infinite-dimensional zero-sum games by a min-max distributional optimization problem over a space of probability measures defined on a continuous variable set, which is inspired by finding a mixed NE for finite-dimensional zero-sum games. We then aim to answer the following question: \textit{Will GDA-type algorithms still be provably efficient when extended to infinite-dimensional zero-sum games?} To answer this question, we propose a particle-based variational transport algorithm based on GDA in the functional spaces. Specifically, the algorithm performs multi-step functional gradient descent-ascent in the Wasserstein space via pushing two sets of particles in the variable space. By characterizing the gradient estimation error from variational form maximization and the convergence behavior of each player with different objective landscapes, we prove rigorously that the generalized GDA algorithm converges to the NE or the value of the game efficiently for a class of games under the Polyak-Ł{ojasiewicz} (PL) condition. To conclude, we provide complete statistical and convergence guarantees for solving an infinite-dimensional zero-sum game via a provably efficient particle-based method. Additionally, our work provides the first thorough statistical analysis for the particle-based algorithm to learn an objective functional with a variational form using universal approximators (\textit{i.e.}, neural networks (NNs)), which is of independent interest.

----

## [642] Noise and Fluctuation of Finite Learning Rate Stochastic Gradient Descent

**Authors**: *Kangqiao Liu, Ziyin Liu, Masahito Ueda*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21ad.html](http://proceedings.mlr.press/v139/liu21ad.html)

**Abstract**:

In the vanishing learning rate regime, stochastic gradient descent (SGD) is now relatively well understood. In this work, we propose to study the basic properties of SGD and its variants in the non-vanishing learning rate regime. The focus is on deriving exactly solvable results and discussing their implications. The main contributions of this work are to derive the stationary distribution for discrete-time SGD in a quadratic loss function with and without momentum; in particular, one implication of our result is that the fluctuation caused by discrete-time dynamics takes a distorted shape and is dramatically larger than a continuous-time theory could predict. Examples of applications of the proposed theory considered in this work include the approximation error of variants of SGD, the effect of minibatch noise, the optimal Bayesian inference, the escape rate from a sharp minimum, and the stationary covariance of a few second-order methods including damped Newton’s method, natural gradient descent, and Adam.

----

## [643] Multi-layered Network Exploration via Random Walks: From Offline Optimization to Online Learning

**Authors**: *Xutong Liu, Jinhang Zuo, Xiaowei Chen, Wei Chen, John C. S. Lui*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liu21ae.html](http://proceedings.mlr.press/v139/liu21ae.html)

**Abstract**:

Multi-layered network exploration (MuLaNE) problem is an important problem abstracted from many applications. In MuLaNE, there are multiple network layers where each node has an importance weight and each layer is explored by a random walk. The MuLaNE task is to allocate total random walk budget $B$ into each network layer so that the total weights of the unique nodes visited by random walks are maximized. We systematically study this problem from offline optimization to online learning. For the offline optimization setting where the network structure and node weights are known, we provide greedy based constant-ratio approximation algorithms for overlapping networks, and greedy or dynamic-programming based optimal solutions for non-overlapping networks. For the online learning setting, neither the network structure nor the node weights are known initially. We adapt the combinatorial multi-armed bandit framework and design algorithms to learn random walk related parameters and node weights while optimizing the budget allocation in multiple rounds, and prove that they achieve logarithmic regret bounds. Finally, we conduct experiments on a real-world social network dataset to validate our theoretical results.

----

## [644] Relative Positional Encoding for Transformers with Linear Complexity

**Authors**: *Antoine Liutkus, Ondrej Cífka, Shih-Lun Wu, Umut Simsekli, Yi-Hsuan Yang, Gaël Richard*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/liutkus21a.html](http://proceedings.mlr.press/v139/liutkus21a.html)

**Abstract**:

Recent advances in Transformer models allow for unprecedented sequence lengths, due to linear space and time complexity. In the meantime, relative positional encoding (RPE) was proposed as beneficial for classical Transformers and consists in exploiting lags instead of absolute positions for inference. Still, RPE is not available for the recent linear-variants of the Transformer, because it requires the explicit computation of the attention matrix, which is precisely what is avoided by such methods. In this paper, we bridge this gap and present Stochastic Positional Encoding as a way to generate PE that can be used as a replacement to the classical additive (sinusoidal) PE and provably behaves like RPE. The main theoretical contribution is to make a connection between positional encoding and cross-covariance structures of correlated Gaussian processes. We illustrate the performance of our approach on the Long-Range Arena benchmark and on music generation.

----

## [645] Joint Online Learning and Decision-making via Dual Mirror Descent

**Authors**: *Alfonso Lobos, Paul Grigas, Zheng Wen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lobos21a.html](http://proceedings.mlr.press/v139/lobos21a.html)

**Abstract**:

We consider an online revenue maximization problem over a finite time horizon subject to lower and upper bounds on cost. At each period, an agent receives a context vector sampled i.i.d. from an unknown distribution and needs to make a decision adaptively. The revenue and cost functions depend on the context vector as well as some fixed but possibly unknown parameter vector to be learned. We propose a novel offline benchmark and a new algorithm that mixes an online dual mirror descent scheme with a generic parameter learning process. When the parameter vector is known, we demonstrate an $O(\sqrt{T})$ regret result as well an $O(\sqrt{T})$ bound on the possible constraint violations. When the parameter is not known and must be learned, we demonstrate that the regret and constraint violations are the sums of the previous $O(\sqrt{T})$ terms plus terms that directly depend on the convergence of the learning process.

----

## [646] Symmetric Spaces for Graph Embeddings: A Finsler-Riemannian Approach

**Authors**: *Federico López, Beatrice Pozzetti, Steve Trettel, Michael Strube, Anna Wienhard*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lopez21a.html](http://proceedings.mlr.press/v139/lopez21a.html)

**Abstract**:

Learning faithful graph representations as sets of vertex embeddings has become a fundamental intermediary step in a wide range of machine learning applications. We propose the systematic use of symmetric spaces in representation learning, a class encompassing many of the previously used embedding targets. This enables us to introduce a new method, the use of Finsler metrics integrated in a Riemannian optimization scheme, that better adapts to dissimilar structures in the graph. We develop a tool to analyze the embeddings and infer structural properties of the data sets. For implementation, we choose Siegel spaces, a versatile family of symmetric spaces. Our approach outperforms competitive baselines for graph reconstruction tasks on various synthetic and real-world datasets. We further demonstrate its applicability on two downstream tasks, recommender systems and node classification.

----

## [647] HEMET: A Homomorphic-Encryption-Friendly Privacy-Preserving Mobile Neural Network Architecture

**Authors**: *Qian Lou, Lei Jiang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lou21a.html](http://proceedings.mlr.press/v139/lou21a.html)

**Abstract**:

Recently Homomorphic Encryption (HE) is used to implement Privacy-Preserving Neural Networks (PPNNs) that perform inferences directly on encrypted data without decryption. Prior PPNNs adopt mobile network architectures such as SqueezeNet for smaller computing overhead, but we find naïvely using mobile network architectures for a PPNN does not necessarily achieve shorter inference latency. Despite having less parameters, a mobile network architecture typically introduces more layers and increases the HE multiplicative depth of a PPNN, thereby prolonging its inference latency. In this paper, we propose a \textbf{HE}-friendly privacy-preserving \textbf{M}obile neural n\textbf{ET}work architecture, \textbf{HEMET}. Experimental results show that, compared to state-of-the-art (SOTA) PPNNs, HEMET reduces the inference latency by $59.3%\sim 61.2%$, and improves the inference accuracy by $0.4 % \sim 0.5%$.

----

## [648] Optimal Complexity in Decentralized Training

**Authors**: *Yucheng Lu, Christopher De Sa*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lu21a.html](http://proceedings.mlr.press/v139/lu21a.html)

**Abstract**:

Decentralization is a promising method of scaling up parallel machine learning systems. In this paper, we provide a tight lower bound on the iteration complexity for such methods in a stochastic non-convex setting. Our lower bound reveals a theoretical gap in known convergence rates of many existing decentralized training algorithms, such as D-PSGD. We prove by construction this lower bound is tight and achievable. Motivated by our insights, we further propose DeTAG, a practical gossip-style decentralized algorithm that achieves the lower bound with only a logarithm gap. Empirically, we compare DeTAG with other decentralized algorithms on image classification tasks, and we show DeTAG enjoys faster convergence compared to baselines, especially on unshuffled data and in sparse networks.

----

## [649] DANCE: Enhancing saliency maps using decoys

**Authors**: *Yang Young Lu, Wenbo Guo, Xinyu Xing, William Stafford Noble*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lu21b.html](http://proceedings.mlr.press/v139/lu21b.html)

**Abstract**:

Saliency methods can make deep neural network predictions more interpretable by identifying a set of critical features in an input sample, such as pixels that contribute most strongly to a prediction made by an image classifier. Unfortunately, recent evidence suggests that many saliency methods poorly perform, especially in situations where gradients are saturated, inputs contain adversarial perturbations, or predictions rely upon inter-feature dependence. To address these issues, we propose a framework, DANCE, which improves the robustness of saliency methods by following a two-step procedure. First, we introduce a perturbation mechanism that subtly varies the input sample without changing its intermediate representations. Using this approach, we can gather a corpus of perturbed ("decoy") data samples while ensuring that the perturbed and original input samples follow similar distributions. Second, we compute saliency maps for the decoy samples and propose a new method to aggregate saliency maps. With this design, we offset influence of gradient saturation. From a theoretical perspective, we show that the aggregated saliency map not only captures inter-feature dependence but, more importantly, is robust against previously described adversarial perturbation methods. Our empirical results suggest that, both qualitatively and quantitatively, DANCE outperforms existing methods in a variety of application domains.

----

## [650] Binary Classification from Multiple Unlabeled Datasets via Surrogate Set Classification

**Authors**: *Nan Lu, Shida Lei, Gang Niu, Issei Sato, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lu21c.html](http://proceedings.mlr.press/v139/lu21c.html)

**Abstract**:

To cope with high annotation costs, training a classifier only from weakly supervised data has attracted a great deal of attention these days. Among various approaches, strengthening supervision from completely unsupervised classification is a promising direction, which typically employs class priors as the only supervision and trains a binary classifier from unlabeled (U) datasets. While existing risk-consistent methods are theoretically grounded with high flexibility, they can learn only from two U sets. In this paper, we propose a new approach for binary classification from $m$ U-sets for $m\ge2$. Our key idea is to consider an auxiliary classification task called surrogate set classification (SSC), which is aimed at predicting from which U set each observed sample is drawn. SSC can be solved by a standard (multi-class) classification method, and we use the SSC solution to obtain the final binary classifier through a certain linear-fractional transformation. We built our method in a flexible and efficient end-to-end deep learning framework and prove it to be classifier-consistent. Through experiments, we demonstrate the superiority of our proposed method over state-of-the-art methods.

----

## [651] Variance Reduced Training with Stratified Sampling for Forecasting Models

**Authors**: *Yucheng Lu, Youngsuk Park, Lifan Chen, Yuyang Wang, Christopher De Sa, Dean Foster*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lu21d.html](http://proceedings.mlr.press/v139/lu21d.html)

**Abstract**:

In large-scale time series forecasting, one often encounters the situation where the temporal patterns of time series, while drifting over time, differ from one another in the same dataset. In this paper, we provably show under such heterogeneity, training a forecasting model with commonly used stochastic optimizers (e.g. SGD) potentially suffers large variance on gradient estimation, and thus incurs long-time training. We show that this issue can be efficiently alleviated via stratification, which allows the optimizer to sample from pre-grouped time series strata. For better trading-off gradient variance and computation complexity, we further propose SCott (Stochastic Stratified Control Variate Gradient Descent), a variance reduced SGD-style optimizer that utilizes stratified sampling via control variate. In theory, we provide the convergence guarantee of SCott on smooth non-convex objectives. Empirically, we evaluate SCott and other baseline optimizers on both synthetic and real-world time series forecasting problems, and demonstrate SCott converges faster with respect to both iterations and wall clock time.

----

## [652] ACE: Explaining cluster from an adversarial perspective

**Authors**: *Yang Young Lu, Timothy C. Yu, Giancarlo Bonora, William Stafford Noble*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lu21e.html](http://proceedings.mlr.press/v139/lu21e.html)

**Abstract**:

A common workflow in single-cell RNA-seq analysis is to project the data to a latent space, cluster the cells in that space, and identify sets of marker genes that explain the differences among the discovered clusters. A primary drawback to this three-step procedure is that each step is carried out independently, thereby neglecting the effects of the nonlinear embedding and inter-gene dependencies on the selection of marker genes. Here we propose an integrated deep learning framework, Adversarial Clustering Explanation (ACE), that bundles all three steps into a single workflow. The method thus moves away from the notion of "marker genes" to instead identify a panel of explanatory genes. This panel may include genes that are not only enriched but also depleted relative to other cell types, as well as genes that exhibit differences between closely related cell types. Empirically, we demonstrate that ACE is able to identify gene panels that are both highly discriminative and nonredundant, and we demonstrate the applicability of ACE to an image recognition task.

----

## [653] On Monotonic Linear Interpolation of Neural Network Parameters

**Authors**: *James Lucas, Juhan Bae, Michael R. Zhang, Stanislav Fort, Richard S. Zemel, Roger B. Grosse*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lucas21a.html](http://proceedings.mlr.press/v139/lucas21a.html)

**Abstract**:

Linear interpolation between initial neural network parameters and converged parameters after training with stochastic gradient descent (SGD) typically leads to a monotonic decrease in the training objective. This Monotonic Linear Interpolation (MLI) property, first observed by Goodfellow et al. 2014, persists in spite of the non-convex objectives and highly non-linear training dynamics of neural networks. Extending this work, we evaluate several hypotheses for this property that, to our knowledge, have not yet been explored. Using tools from differential geometry, we draw connections between the interpolated paths in function space and the monotonicity of the network — providing sufficient conditions for the MLI property under mean squared error. While the MLI property holds under various settings (e.g., network architectures and learning problems), we show in practice that networks violating the MLI property can be produced systematically, by encouraging the weights to move far from initialization. The MLI property raises important questions about the loss landscape geometry of neural networks and highlights the need to further study their global properties.

----

## [654] Improving Breadth-Wise Backpropagation in Graph Neural Networks Helps Learning Long-Range Dependencies

**Authors**: *Denis Lukovnikov, Asja Fischer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lukovnikov21a.html](http://proceedings.mlr.press/v139/lukovnikov21a.html)

**Abstract**:

In this work, we focus on the ability of graph neural networks (GNNs) to learn long-range patterns in graphs with edge features. Learning patterns that involve longer paths in the graph, requires using deeper GNNs. However, GNNs suffer from a drop in performance with increasing network depth. To improve the performance of deeper GNNs, previous works have investigated normalization techniques and various types of skip connections. While they are designed to improve depth-wise backpropagation between the representations of the same node in successive layers, they do not improve breadth-wise backpropagation between representations of neighbouring nodes. To analyse the consequences, we design synthetic datasets serving as a testbed for the ability of GNNs to learn long-range patterns. Our analysis shows that several commonly used GNN variants with only depth-wise skip connections indeed have problems learning long-range patterns. They are clearly outperformed by an attention-based GNN architecture that we propose for improving both depth- and breadth-wise backpropagation. We also verify that the presented architecture is competitive on real-world data.

----

## [655] GraphDF: A Discrete Flow Model for Molecular Graph Generation

**Authors**: *Youzhi Luo, Keqiang Yan, Shuiwang Ji*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/luo21a.html](http://proceedings.mlr.press/v139/luo21a.html)

**Abstract**:

We consider the problem of molecular graph generation using deep models. While graphs are discrete, most existing methods use continuous latent variables, resulting in inaccurate modeling of discrete graph structures. In this work, we propose GraphDF, a novel discrete latent variable model for molecular graph generation based on normalizing flow methods. GraphDF uses invertible modulo shift transforms to map discrete latent variables to graph nodes and edges. We show that the use of discrete latent variables reduces computational costs and eliminates the negative effect of dequantization. Comprehensive experimental results show that GraphDF outperforms prior methods on random generation, property optimization, and constrained optimization tasks.

----

## [656] Trajectory Diversity for Zero-Shot Coordination

**Authors**: *Andrei Lupu, Brandon Cui, Hengyuan Hu, Jakob N. Foerster*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lupu21a.html](http://proceedings.mlr.press/v139/lupu21a.html)

**Abstract**:

We study the problem of zero-shot coordination (ZSC), where agents must independently produce strategies for a collaborative game that are compatible with novel partners not seen during training. Our first contribution is to consider the need for diversity in generating such agents. Because self-play (SP) agents control their own trajectory distribution during training, each policy typically only performs well on this exact distribution. As a result, they achieve low scores in ZSC, since playing with another agent is likely to put them in situations they have not encountered during training. To address this issue, we train a common best response (BR) to a population of agents, which we regulate to be diverse. To this end, we introduce \textit{Trajectory Diversity} (TrajeDi) – a differentiable objective for generating diverse reinforcement learning policies. We derive TrajeDi as a generalization of the Jensen-Shannon divergence between policies and motivate it experimentally in two simple settings. We then focus on the collaborative card game Hanabi, demonstrating the scalability of our method and improving upon the cross-play scores of both independently trained SP agents and BRs to unregularized populations.

----

## [657] HyperHyperNetwork for the Design of Antenna Arrays

**Authors**: *Shahar Lutati, Lior Wolf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lutati21a.html](http://proceedings.mlr.press/v139/lutati21a.html)

**Abstract**:

We present deep learning methods for the design of arrays and single instances of small antennas. Each design instance is conditioned on a target radiation pattern and is required to conform to specific spatial dimensions and to include, as part of its metallic structure, a set of predetermined locations. The solution, in the case of a single antenna, is based on a composite neural network that combines a simulation network, a hypernetwork, and a refinement network. In the design of the antenna array, we add an additional design level and employ a hypernetwork within a hypernetwork. The learning objective is based on measuring the similarity of the obtained radiation pattern to the desired one. Our experiments demonstrate that our approach is able to design novel antennas and antenna arrays that are compliant with the design requirements, considerably better than the baseline methods. We compare the solutions obtained by our method to existing designs and demonstrate a high level of overlap. When designing the antenna array of a cellular phone, the obtained solution displays improved properties over the existing one.

----

## [658] Value Iteration in Continuous Actions, States and Time

**Authors**: *Michael Lutter, Shie Mannor, Jan Peters, Dieter Fox, Animesh Garg*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/lutter21a.html](http://proceedings.mlr.press/v139/lutter21a.html)

**Abstract**:

Classical value iteration approaches are not applicable to environments with continuous states and actions. For such environments the states and actions must be discretized, which leads to an exponential increase in computational complexity. In this paper, we propose continuous fitted value iteration (cFVI). This algorithm enables dynamic programming for continuous states and actions with a known dynamics model. Exploiting the continuous time formulation, the optimal policy can be derived for non-linear control-affine dynamics. This closed-form solution enables the efficient extension of value iteration to continuous environments. We show in non-linear control experiments that the dynamic programming solution obtains the same quantitative performance as deep reinforcement learning methods in simulation but excels when transferred to the physical system.The policy obtained by cFVI is more robust to changes in the dynamics despite using only a deterministic model and without explicitly incorporating robustness in the optimization

----

## [659] Meta-Cal: Well-controlled Post-hoc Calibration by Ranking

**Authors**: *Xingchen Ma, Matthew B. Blaschko*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ma21a.html](http://proceedings.mlr.press/v139/ma21a.html)

**Abstract**:

In many applications, it is desirable that a classifier not only makes accurate predictions, but also outputs calibrated posterior probabilities. However, many existing classifiers, especially deep neural network classifiers, tend to be uncalibrated. Post-hoc calibration is a technique to recalibrate a model by learning a calibration map. Existing approaches mostly focus on constructing calibration maps with low calibration errors, however, this quality is inadequate for a calibrator being useful. In this paper, we introduce two constraints that are worth consideration in designing a calibration map for post-hoc calibration. Then we present Meta-Cal, which is built from a base calibrator and a ranking model. Under some mild assumptions, two high-probability bounds are given with respect to these constraints. Empirical results on CIFAR-10, CIFAR-100 and ImageNet and a range of popular network architectures show our proposed method significantly outperforms the current state of the art for post-hoc multi-class classification calibration.

----

## [660] Neural-Pull: Learning Signed Distance Function from Point clouds by Learning to Pull Space onto Surface

**Authors**: *Baorui Ma, Zhizhong Han, Yu-Shen Liu, Matthias Zwicker*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ma21b.html](http://proceedings.mlr.press/v139/ma21b.html)

**Abstract**:

Reconstructing continuous surfaces from 3D point clouds is a fundamental operation in 3D geometry processing. Several recent state-of-the-art methods address this problem using neural networks to learn signed distance functions (SDFs). In this paper, we introduce Neural-Pull, a new approach that is simple and leads to high quality SDFs. Specifically, we train a neural network to pull query 3D locations to their closest points on the surface using the predicted signed distance values and the gradient at the query locations, both of which are computed by the network itself. The pulling operation moves each query location with a stride given by the distance predicted by the network. Based on the sign of the distance, this may move the query location along or against the direction of the gradient of the SDF. This is a differentiable operation that allows us to update the signed distance value and the gradient simultaneously during training. Our outperforming results under widely used benchmarks demonstrate that we can learn SDFs more accurately and flexibly for surface reconstruction and single image reconstruction than the state-of-the-art methods. Our code and data are available at https://github.com/mabaorui/NeuralPull.

----

## [661] Learning Stochastic Behaviour from Aggregate Data

**Authors**: *Shaojun Ma, Shu Liu, Hongyuan Zha, Haomin Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ma21c.html](http://proceedings.mlr.press/v139/ma21c.html)

**Abstract**:

Learning nonlinear dynamics from aggregate data is a challenging problem because the full trajectory of each individual is not available, namely, the individual observed at one time may not be observed at the next time point, or the identity of individual is unavailable. This is in sharp contrast to learning dynamics with full trajectory data, on which the majority of existing methods are based. We propose a novel method using the weak form of Fokker Planck Equation (FPE) — a partial differential equation — to describe the density evolution of data in a sampled form, which is then combined with Wasserstein generative adversarial network (WGAN) in the training process. In such a sample-based framework we are able to learn the nonlinear dynamics from aggregate data without explicitly solving the partial differential equation (PDE) FPE. We demonstrate our approach in the context of a series of synthetic and real-world data sets.

----

## [662] Local Algorithms for Finding Densely Connected Clusters

**Authors**: *Peter Macgregor, He Sun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/macgregor21a.html](http://proceedings.mlr.press/v139/macgregor21a.html)

**Abstract**:

Local graph clustering is an important algorithmic technique for analysing massive graphs, and has been widely applied in many research fields of data science. While the objective of most (local) graph clustering algorithms is to find a vertex set of low conductance, there has been a sequence of recent studies that highlight the importance of the inter-connection between clusters when analysing real-world datasets. Following this line of research, in this work we study local algorithms for finding a pair of vertex sets defined with respect to their inter-connection and their relationship with the rest of the graph. The key to our analysis is a new reduction technique that relates the structure of multiple sets to a single vertex set in the reduced graph. Among many potential applications, we show that our algorithms successfully recover densely connected clusters in the Interstate Disputes Dataset and the US Migration Dataset.

----

## [663] Learning to Generate Noise for Multi-Attack Robustness

**Authors**: *Divyam Madaan, Jinwoo Shin, Sung Ju Hwang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/madaan21a.html](http://proceedings.mlr.press/v139/madaan21a.html)

**Abstract**:

Adversarial learning has emerged as one of the successful techniques to circumvent the susceptibility of existing methods against adversarial perturbations. However, the majority of existing defense methods are tailored to defend against a single category of adversarial perturbation (e.g. $\ell_\infty$-attack). In safety-critical applications, this makes these methods extraneous as the attacker can adopt diverse adversaries to deceive the system. Moreover, training on multiple perturbations simultaneously significantly increases the computational overhead during training. To address these challenges, we propose a novel meta-learning framework that explicitly learns to generate noise to improve the model’s robustness against multiple types of attacks. Its key component is \emph{Meta Noise Generator (MNG)} that outputs optimal noise to stochastically perturb a given sample, such that it helps lower the error on diverse adversarial perturbations. By utilizing samples generated by MNG, we train a model by enforcing the label consistency across multiple perturbations. We validate the robustness of models trained by our scheme on various datasets and against a wide variety of perturbations, demonstrating that it significantly outperforms the baselines across multiple perturbations with a marginal computational cost.

----

## [664] Learning Interaction Kernels for Agent Systems on Riemannian Manifolds

**Authors**: *Mauro Maggioni, Jason Miller, Hongda Qiu, Ming Zhong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/maggioni21a.html](http://proceedings.mlr.press/v139/maggioni21a.html)

**Abstract**:

Interacting agent and particle systems are extensively used to model complex phenomena in science and engineering. We consider the problem of learning interaction kernels in these dynamical systems constrained to evolve on Riemannian manifolds from given trajectory data. The models we consider are based on interaction kernels depending on pairwise Riemannian distances between agents, with agents interacting locally along the direction of the shortest geodesic connecting them. We show that our estimators converge at a rate that is independent of the dimension of the state space, and derive bounds on the trajectory estimation error, on the manifold, between the observed and estimated dynamics. We demonstrate the performance of our estimator on two classical first order interacting systems: Opinion Dynamics and a Predator-Swarm system, with each system constrained on two prototypical manifolds, the $2$-dimensional sphere and the Poincaré disk model of hyperbolic space.

----

## [665] Tesseract: Tensorised Actors for Multi-Agent Reinforcement Learning

**Authors**: *Anuj Mahajan, Mikayel Samvelyan, Lei Mao, Viktor Makoviychuk, Animesh Garg, Jean Kossaifi, Shimon Whiteson, Yuke Zhu, Animashree Anandkumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mahajan21a.html](http://proceedings.mlr.press/v139/mahajan21a.html)

**Abstract**:

Reinforcement Learning in large action spaces is a challenging problem. This is especially true for cooperative multi-agent reinforcement learning (MARL), which often requires tractable learning while respecting various constraints like communication budget and information about other agents. In this work, we focus on the fundamental hurdle affecting both value-based and policy-gradient approaches: an exponential blowup of the action space with the number of agents. For value-based methods, it poses challenges in accurately representing the optimal value function for value-based methods, thus inducing suboptimality. For policy gradient methods, it renders the critic ineffective and exacerbates the problem of the lagging critic. We show that from a learning theory perspective, both problems can be addressed by accurately representing the associated action-value function with a low-complexity hypothesis class. This requires accurately modelling the agent interactions in a sample efficient way. To this end, we propose a novel tensorised formulation of the Bellman equation. This gives rise to our method Tesseract, which utilises the view of Q-function seen as a tensor where the modes correspond to action spaces of different agents. Algorithms derived from Tesseract decompose the Q-tensor across the agents and utilise low-rank tensor approximations to model the agent interactions relevant to the task. We provide PAC analysis for Tesseract based algorithms and highlight their relevance to the class of rich observation MDPs. Empirical results in different domains confirm the gains in sample efficiency using Tesseract as supported by the theory.

----

## [666] Domain Generalization using Causal Matching

**Authors**: *Divyat Mahajan, Shruti Tople, Amit Sharma*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mahajan21b.html](http://proceedings.mlr.press/v139/mahajan21b.html)

**Abstract**:

In the domain generalization literature, a common objective is to learn representations independent of the domain after conditioning on the class label. We show that this objective is not sufficient: there exist counter-examples where a model fails to generalize to unseen domains even after satisfying class-conditional domain invariance. We formalize this observation through a structural causal model and show the importance of modeling within-class variations for generalization. Specifically, classes contain objects that characterize specific causal features, and domains can be interpreted as interventions on these objects that change non-causal features. We highlight an alternative condition: inputs across domains should have the same representation if they are derived from the same object. Based on this objective, we propose matching-based algorithms when base objects are observed (e.g., through data augmentation) and approximate the objective when objects are not observed (MatchDG). Our simple matching-based algorithms are competitive to prior work on out-of-domain accuracy for rotated MNIST, Fashion-MNIST, PACS, and Chest-Xray datasets. Our method MatchDG also recovers ground-truth object matches: on MNIST and Fashion-MNIST, top-10 matches from MatchDG have over 50% overlap with ground-truth matches.

----

## [667] Stability and Convergence of Stochastic Gradient Clipping: Beyond Lipschitz Continuity and Smoothness

**Authors**: *Vien V. Mai, Mikael Johansson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mai21a.html](http://proceedings.mlr.press/v139/mai21a.html)

**Abstract**:

Stochastic gradient algorithms are often unstable when applied to functions that do not have Lipschitz-continuous and/or bounded gradients. Gradient clipping is a simple and effective technique to stabilize the training process for problems that are prone to the exploding gradient problem. Despite its widespread popularity, the convergence properties of the gradient clipping heuristic are poorly understood, especially for stochastic problems. This paper establishes both qualitative and quantitative convergence results of the clipped stochastic (sub)gradient method (SGD) for non-smooth convex functions with rapidly growing subgradients. Our analyses show that clipping enhances the stability of SGD and that the clipped SGD algorithm enjoys finite convergence rates in many cases. We also study the convergence of a clipped method with momentum, which includes clipped SGD as a special case, for weakly convex problems under standard assumptions. With a novel Lyapunov analysis, we show that the proposed method achieves the best-known rate for the considered class of problems, demonstrating the effectiveness of clipped methods also in this regime. Numerical results confirm our theoretical developments.

----

## [668] Nonparametric Hamiltonian Monte Carlo

**Authors**: *Carol Mak, Fabian Zaiser, Luke Ong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mak21a.html](http://proceedings.mlr.press/v139/mak21a.html)

**Abstract**:

Probabilistic programming uses programs to express generative models whose posterior probability is then computed by built-in inference engines. A challenging goal is to develop general purpose inference algorithms that work out-of-the-box for arbitrary programs in a universal probabilistic programming language (PPL). The densities defined by such programs, which may use stochastic branching and recursion, are (in general) nonparametric, in the sense that they correspond to models on an infinite-dimensional parameter space. However standard inference algorithms, such as the Hamiltonian Monte Carlo (HMC) algorithm, target distributions with a fixed number of parameters. This paper introduces the Nonparametric Hamiltonian Monte Carlo (NP-HMC) algorithm which generalises HMC to nonparametric models. Inputs to NP-HMC are a new class of measurable functions called “tree representable”, which serve as a language-independent representation of the density functions of probabilistic programs in a universal PPL. We provide a correctness proof of NP-HMC, and empirically demonstrate significant performance improvements over existing approaches on several nonparametric examples.

----

## [669] Exploiting structured data for learning contagious diseases under incomplete testing

**Authors**: *Maggie Makar, Lauren West, David Hooper, Eric Horvitz, Erica Shenoy, John V. Guttag*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/makar21a.html](http://proceedings.mlr.press/v139/makar21a.html)

**Abstract**:

One of the ways that machine learning algorithms can help control the spread of an infectious disease is by building models that predict who is likely to become infected making them good candidates for preemptive interventions. In this work we ask: can we build reliable infection prediction models when the observed data is collected under limited, and biased testing that prioritizes testing symptomatic individuals? Our analysis suggests that when the infection is highly transmissible, incomplete testing might be sufficient to achieve good out-of-sample prediction error. Guided by this insight, we develop an algorithm that predicts infections, and show that it outperforms baselines on simulated data. We apply our model to data from a large hospital to predict Clostridioides difficile infections; a communicable disease that is characterized by both symptomatically infected and asymptomatic (i.e., untested) carriers. Using a proxy instead of the unobserved untested-infected state, we show that our model outperforms benchmarks in predicting infections.

----

## [670] Near-Optimal Algorithms for Explainable k-Medians and k-Means

**Authors**: *Konstantin Makarychev, Liren Shan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/makarychev21a.html](http://proceedings.mlr.press/v139/makarychev21a.html)

**Abstract**:

We consider the problem of explainable $k$-medians and $k$-means introduced by Dasgupta, Frost, Moshkovitz, and Rashtchian (ICML 2020). In this problem, our goal is to find a \emph{threshold decision tree} that partitions data into $k$ clusters and minimizes the $k$-medians or $k$-means objective. The obtained clustering is easy to interpret because every decision node of a threshold tree splits data based on a single feature into two groups. We propose a new algorithm for this problem which is $\tilde O(\log k)$ competitive with $k$-medians with $\ell_1$ norm and $\tilde O(k)$ competitive with $k$-means. This is an improvement over the previous guarantees of $O(k)$ and $O(k^2)$ by Dasgupta et al (2020). We also provide a new algorithm which is $O(\log^{\nicefrac{3}{2}} k)$ competitive for $k$-medians with $\ell_2$ norm. Our first algorithm is near-optimal: Dasgupta et al (2020) showed a lower bound of $\Omega(\log k)$ for $k$-medians; in this work, we prove a lower bound of $\tilde\Omega(k)$ for $k$-means. We also provide a lower bound of $\Omega(\log k)$ for $k$-medians with $\ell_2$ norm.

----

## [671] KO codes: inventing nonlinear encoding and decoding for reliable wireless communication via deep-learning

**Authors**: *Ashok Vardhan Makkuva, Xiyang Liu, Mohammad Vahid Jamali, Hessam Mahdavifar, Sewoong Oh, Pramod Viswanath*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/makkuva21a.html](http://proceedings.mlr.press/v139/makkuva21a.html)

**Abstract**:

Landmark codes underpin reliable physical layer communication, e.g., Reed-Muller, BCH, Convolution, Turbo, LDPC, and Polar codes: each is a linear code and represents a mathematical breakthrough. The impact on humanity is huge: each of these codes has been used in global wireless communication standards (satellite, WiFi, cellular). Reliability of communication over the classical additive white Gaussian noise (AWGN) channel enables benchmarking and ranking of the different codes. In this paper, we construct KO codes, a computationally efficient family of deep-learning driven (encoder, decoder) pairs that outperform the state-of-the-art reliability performance on the standardized AWGN channel. KO codes beat state-of-the-art Reed-Muller and Polar codes, under the low-complexity successive cancellation decoding, in the challenging short-to-medium block length regime on the AWGN channel. We show that the gains of KO codes are primarily due to the nonlinear mapping of information bits directly to transmit symbols (bypassing modulation) and yet possess an efficient, high-performance decoder. The key technical innovation that renders this possible is design of a novel family of neural architectures inspired by the computation tree of the {\bf K}ronecker {\bf O}peration (KO) central to Reed-Muller and Polar codes. These architectures pave way for the discovery of a much richer class of hitherto unexplored nonlinear algebraic structures.

----

## [672] Quantifying the Benefit of Using Differentiable Learning over Tangent Kernels

**Authors**: *Eran Malach, Pritish Kamath, Emmanuel Abbe, Nathan Srebro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/malach21a.html](http://proceedings.mlr.press/v139/malach21a.html)

**Abstract**:

We study the relative power of learning with gradient descent on differentiable models, such as neural networks, versus using the corresponding tangent kernels. We show that under certain conditions, gradient descent achieves small error only if a related tangent kernel method achieves a non-trivial advantage over random guessing (a.k.a. weak learning), though this advantage might be very small even when gradient descent can achieve arbitrarily high accuracy. Complementing this, we show that without these conditions, gradient descent can in fact learn with small error even when no kernel method, in particular using the tangent kernel, can achieve a non-trivial advantage over random guessing.

----

## [673] Inverse Constrained Reinforcement Learning

**Authors**: *Shehryar Malik, Usman Anwar, Alireza Aghasi, Ali Ahmed*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/malik21a.html](http://proceedings.mlr.press/v139/malik21a.html)

**Abstract**:

In real world settings, numerous constraints are present which are hard to specify mathematically. However, for the real world deployment of reinforcement learning (RL), it is critical that RL agents are aware of these constraints, so that they can act safely. In this work, we consider the problem of learning constraints from demonstrations of a constraint-abiding agent’s behavior. We experimentally validate our approach and show that our framework can successfully learn the most likely constraints that the agent respects. We further show that these learned constraints are \textit{transferable} to new agents that may have different morphologies and/or reward functions. Previous works in this regard have either mainly been restricted to tabular (discrete) settings, specific types of constraints or assume the environment’s transition dynamics. In contrast, our framework is able to learn arbitrary \textit{Markovian} constraints in high-dimensions in a completely model-free setting. The code is available at: \url{https://github.com/shehryar-malik/icrl}.

----

## [674] A Sampling-Based Method for Tensor Ring Decomposition

**Authors**: *Osman Asif Malik, Stephen Becker*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/malik21b.html](http://proceedings.mlr.press/v139/malik21b.html)

**Abstract**:

We propose a sampling-based method for computing the tensor ring (TR) decomposition of a data tensor. The method uses leverage score sampled alternating least squares to fit the TR cores in an iterative fashion. By taking advantage of the special structure of TR tensors, we can efficiently estimate the leverage scores and attain a method which has complexity sublinear in the number of input tensor entries. We provide high-probability relative-error guarantees for the sampled least squares problems. We compare our proposal to existing methods in experiments on both synthetic and real data. Our method achieves substantial speedup—sometimes two or three orders of magnitude—over competing methods, while maintaining good accuracy. We also provide an example of how our method can be used for rapid feature extraction.

----

## [675] Sample Efficient Reinforcement Learning In Continuous State Spaces: A Perspective Beyond Linearity

**Authors**: *Dhruv Malik, Aldo Pacchiano, Vishwak Srinivasan, Yuanzhi Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/malik21c.html](http://proceedings.mlr.press/v139/malik21c.html)

**Abstract**:

Reinforcement learning (RL) is empirically successful in complex nonlinear Markov decision processes (MDPs) with continuous state spaces. By contrast, the majority of theoretical RL literature requires the MDP to satisfy some form of linear structure, in order to guarantee sample efficient RL. Such efforts typically assume the transition dynamics or value function of the MDP are described by linear functions of the state features. To resolve this discrepancy between theory and practice, we introduce the Effective Planning Window (EPW) condition, a structural condition on MDPs that makes no linearity assumptions. We demonstrate that the EPW condition permits sample efficient RL, by providing an algorithm which provably solves MDPs satisfying this condition. Our algorithm requires minimal assumptions on the policy class, which can include multi-layer neural networks with nonlinear activation functions. Notably, the EPW condition is directly motivated by popular gaming benchmarks, and we show that many classic Atari games satisfy this condition. We additionally show the necessity of conditions like EPW, by demonstrating that simple MDPs with slight nonlinearities cannot be solved sample efficiently.

----

## [676] Beyond the Pareto Efficient Frontier: Constraint Active Search for Multiobjective Experimental Design

**Authors**: *Gustavo Malkomes, Bolong Cheng, Eric Hans Lee, Mike Mccourt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/malkomes21a.html](http://proceedings.mlr.press/v139/malkomes21a.html)

**Abstract**:

Many problems in engineering design and simulation require balancing competing objectives under the presence of uncertainty. Sample-efficient multiobjective optimization methods focus on the objective function values in metric space and ignore the sampling behavior of the design configurations in parameter space. Consequently, they may provide little actionable insight on how to choose designs in the presence of metric uncertainty or limited precision when implementing a chosen design. We propose a new formulation that accounts for the importance of the parameter space and is thus more suitable for multiobjective design problems; instead of searching for the Pareto-efficient frontier, we solicit the desired minimum performance thresholds on all objectives to define regions of satisfaction. We introduce an active search algorithm called Expected Coverage Improvement (ECI) to efficiently discover the region of satisfaction and simultaneously sample diverse acceptable configurations. We demonstrate our algorithm on several design and simulation domains: mechanical design, additive manufacturing, medical monitoring, and plasma physics.

----

## [677] Consistent Nonparametric Methods for Network Assisted Covariate Estimation

**Authors**: *Xueyu Mao, Deepayan Chakrabarti, Purnamrita Sarkar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mao21a.html](http://proceedings.mlr.press/v139/mao21a.html)

**Abstract**:

Networks with node covariates are commonplace: for example, people in a social network have interests, or product preferences, etc. If we know the covariates for some nodes, can we infer them for the remaining nodes? In this paper we propose a new similarity measure between two nodes based on the patterns of their 2-hop neighborhoods. We show that a simple algorithm (CN-VEC) like nearest neighbor regression with this metric is consistent for a wide range of models when the degree grows faster than $n^{1/3}$ up-to logarithmic factors, where $n$ is the number of nodes. For "low-rank" latent variable models, the natural contender will be to estimate the latent variables using SVD and use them for non-parametric regression. While we show consistency of this method under less stringent sparsity conditions, our experimental results suggest that the simple local CN-VEC method either outperforms the global SVD-RBF method, or has comparable performance for low rank models. We also present simulated and real data experiments to show the effectiveness of our algorithms compared to the state of the art.

----

## [678] Near-Optimal Model-Free Reinforcement Learning in Non-Stationary Episodic MDPs

**Authors**: *Weichao Mao, Kaiqing Zhang, Ruihao Zhu, David Simchi-Levi, Tamer Basar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mao21b.html](http://proceedings.mlr.press/v139/mao21b.html)

**Abstract**:

We consider model-free reinforcement learning (RL) in non-stationary Markov decision processes. Both the reward functions and the state transition functions are allowed to vary arbitrarily over time as long as their cumulative variations do not exceed certain variation budgets. We propose Restarted Q-Learning with Upper Confidence Bounds (RestartQ-UCB), the first model-free algorithm for non-stationary RL, and show that it outperforms existing solutions in terms of dynamic regret. Specifically, RestartQ-UCB with Freedman-type bonus terms achieves a dynamic regret bound of $\widetilde{O}(S^{\frac{1}{3}} A^{\frac{1}{3}} \Delta^{\frac{1}{3}} H T^{\frac{2}{3}})$, where $S$ and $A$ are the numbers of states and actions, respectively, $\Delta>0$ is the variation budget, $H$ is the number of time steps per episode, and $T$ is the total number of time steps. We further show that our algorithm is \emph{nearly optimal} by establishing an information-theoretical lower bound of $\Omega(S^{\frac{1}{3}} A^{\frac{1}{3}} \Delta^{\frac{1}{3}} H^{\frac{2}{3}} T^{\frac{2}{3}})$, the first lower bound in non-stationary RL. Numerical experiments validate the advantages of RestartQ-UCB in terms of both cumulative rewards and computational efficiency. We further demonstrate the power of our results in the context of multi-agent RL, where non-stationarity is a key challenge.

----

## [679] Adaptive Sampling for Best Policy Identification in Markov Decision Processes

**Authors**: *Aymen Al Marjani, Alexandre Proutière*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/marjani21a.html](http://proceedings.mlr.press/v139/marjani21a.html)

**Abstract**:

We investigate the problem of best-policy identification in discounted Markov Decision Processes (MDPs) when the learner has access to a generative model. The objective is to devise a learning algorithm returning the best policy as early as possible. We first derive a problem-specific lower bound of the sample complexity satisfied by any learning algorithm. This lower bound corresponds to an optimal sample allocation that solves a non-convex program, and hence, is hard to exploit in the design of efficient algorithms. We then provide a simple and tight upper bound of the sample complexity lower bound, whose corresponding nearly-optimal sample allocation becomes explicit. The upper bound depends on specific functionals of the MDP such as the sub-optimality gaps and the variance of the next-state value function, and thus really captures the hardness of the MDP. Finally, we devise KLB-TS (KL Ball Track-and-Stop), an algorithm tracking this nearly-optimal allocation, and provide asymptotic guarantees for its sample complexity (both almost surely and in expectation). The advantages of KLB-TS against state-of-the-art algorithms are discussed and illustrated numerically.

----

## [680] Explanations for Monotonic Classifiers

**Authors**: *João Marques-Silva, Thomas Gerspacher, Martin C. Cooper, Alexey Ignatiev, Nina Narodytska*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/marques-silva21a.html](http://proceedings.mlr.press/v139/marques-silva21a.html)

**Abstract**:

In many classification tasks there is a requirement of monotonicity. Concretely, if all else remains constant, increasing (resp. decreasing) the value of one or more features must not decrease (resp. increase) the value of the prediction. Despite comprehensive efforts on learning monotonic classifiers, dedicated approaches for explaining monotonic classifiers are scarce and classifier-specific. This paper describes novel algorithms for the computation of one formal explanation of a (black-box) monotonic classifier. These novel algorithms are polynomial (indeed linear) in the run time complexity of the classifier. Furthermore, the paper presents a practically efficient model-agnostic algorithm for enumerating formal explanations.

----

## [681] Multi-Agent Training beyond Zero-Sum with Correlated Equilibrium Meta-Solvers

**Authors**: *Luke Marris, Paul Muller, Marc Lanctot, Karl Tuyls, Thore Graepel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/marris21a.html](http://proceedings.mlr.press/v139/marris21a.html)

**Abstract**:

Two-player, constant-sum games are well studied in the literature, but there has been limited progress outside of this setting. We propose Joint Policy-Space Response Oracles (JPSRO), an algorithm for training agents in n-player, general-sum extensive form games, which provably converges to an equilibrium. We further suggest correlated equilibria (CE) as promising meta-solvers, and propose a novel solution concept Maximum Gini Correlated Equilibrium (MGCE), a principled and computationally efficient family of solutions for solving the correlated equilibrium selection problem. We conduct several experiments using CE meta-solvers for JPSRO and demonstrate convergence on n-player, general-sum games.

----

## [682] Blind Pareto Fairness and Subgroup Robustness

**Authors**: *Natalia Martínez, Martín Bertrán, Afroditi Papadaki, Miguel R. D. Rodrigues, Guillermo Sapiro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/martinez21a.html](http://proceedings.mlr.press/v139/martinez21a.html)

**Abstract**:

Much of the work in the field of group fairness addresses disparities between predefined groups based on protected features such as gender, age, and race, which need to be available at train, and often also at test, time. These approaches are static and retrospective, since algorithms designed to protect groups identified a priori cannot anticipate and protect the needs of different at-risk groups in the future. In this work we analyze the space of solutions for worst-case fairness beyond demographics, and propose Blind Pareto Fairness (BPF), a method that leverages no-regret dynamics to recover a fair minimax classifier that reduces worst-case risk of any potential subgroup of sufficient size, and guarantees that the remaining population receives the best possible level of service. BPF addresses fairness beyond demographics, that is, it does not rely on predefined notions of at-risk groups, neither at train nor at test time. Our experimental results show that the proposed framework improves worst-case risk in multiple standard datasets, while simultaneously providing better levels of service for the remaining population. The code is available at github.com/natalialmg/BlindParetoFairness

----

## [683] Necessary and sufficient conditions for causal feature selection in time series with latent common causes

**Authors**: *Atalanti-Anastasia Mastakouri, Bernhard Schölkopf, Dominik Janzing*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mastakouri21a.html](http://proceedings.mlr.press/v139/mastakouri21a.html)

**Abstract**:

We study the identification of direct and indirect causes on time series with latent variables, and provide a constrained-based causal feature selection method, which we prove that is both sound and complete under some graph constraints. Our theory and estimation algorithm require only two conditional independence tests for each observed candidate time series to determine whether or not it is a cause of an observed target time series. Furthermore, our selection of the conditioning set is such that it improves signal to noise ratio. We apply our method on real data, and on a wide range of simulated experiments, which yield very low false positive and relatively low false negative rates.

----

## [684] Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction

**Authors**: *Afsaneh Mastouri, Yuchen Zhu, Limor Gultchin, Anna Korba, Ricardo Silva, Matt J. Kusner, Arthur Gretton, Krikamol Muandet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mastouri21a.html](http://proceedings.mlr.press/v139/mastouri21a.html)

**Abstract**:

We address the problem of causal effect estima-tion in the presence of unobserved confounding,but where proxies for the latent confounder(s) areobserved. We propose two kernel-based meth-ods for nonlinear causal effect estimation in thissetting: (a) a two-stage regression approach, and(b) a maximum moment restriction approach. Wefocus on the proximal causal learning setting, butour methods can be used to solve a wider classof inverse problems characterised by a Fredholmintegral equation. In particular, we provide a uni-fying view of two-stage and moment restrictionapproaches for solving this problem in a nonlin-ear setting. We provide consistency guaranteesfor each algorithm, and demonstrate that these ap-proaches achieve competitive results on syntheticdata and data simulating a real-world task. In par-ticular, our approach outperforms earlier methodsthat are not suited to leveraging proxy variables.

----

## [685] Robust Unsupervised Learning via L-statistic Minimization

**Authors**: *Andreas Maurer, Daniela Angela Parletta, Andrea Paudice, Massimiliano Pontil*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/maurer21a.html](http://proceedings.mlr.press/v139/maurer21a.html)

**Abstract**:

Designing learning algorithms that are resistant to perturbations of the underlying data distribution is a problem of wide practical and theoretical importance. We present a general approach to this problem focusing on unsupervised learning. The key assumption is that the perturbing distribution is characterized by larger losses relative to a given class of admissible models. This is exploited by a general descent algorithm which minimizes an $L$-statistic criterion over the model class, weighting small losses more. Our analysis characterizes the robustness of the method in terms of bounds on the reconstruction error relative to the underlying unperturbed distribution. As a byproduct, we prove uniform convergence bounds with respect to the proposed criterion for several popular models in unsupervised learning, a result which may be of independent interest. Numerical experiments with \textsc{kmeans} clustering and principal subspace analysis demonstrate the effectiveness of our approach.

----

## [686] Adversarial Multi Class Learning under Weak Supervision with Performance Guarantees

**Authors**: *Alessio Mazzetto, Cyrus Cousins, Dylan Sam, Stephen H. Bach, Eli Upfal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mazzetto21a.html](http://proceedings.mlr.press/v139/mazzetto21a.html)

**Abstract**:

We develop a rigorous approach for using a set of arbitrarily correlated weak supervision sources in order to solve a multiclass classification task when only a very small set of labeled data is available. Our learning algorithm provably converges to a model that has minimum empirical risk with respect to an adversarial choice over feasible labelings for a set of unlabeled data, where the feasibility of a labeling is computed through constraints defined by rigorously estimated statistics of the weak supervision sources. We show theoretical guarantees for this approach that depend on the information provided by the weak supervision sources. Notably, this method does not require the weak supervision sources to have the same labeling space as the multiclass classification task. We demonstrate the effectiveness of our approach with experiments on various image classification tasks.

----

## [687] Fundamental Tradeoffs in Distributionally Adversarial Training

**Authors**: *Mohammad Mehrabi, Adel Javanmard, Ryan A. Rossi, Anup B. Rao, Tung Mai*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mehrabi21a.html](http://proceedings.mlr.press/v139/mehrabi21a.html)

**Abstract**:

Adversarial training is among the most effective techniques to improve robustness of models against adversarial perturbations. However, the full effect of this approach on models is not well understood. For example, while adversarial training can reduce the adversarial risk (prediction error against an adversary), it sometimes increase standard risk (generalization error when there is no adversary). In this paper, we focus on \emph{distribution perturbing} adversary framework wherein the adversary can change the test distribution within a neighborhood of the training data distribution. The neighborhood is defined via Wasserstein distance between distributions and the radius of the neighborhood is a measure of adversary’s manipulative power. We study the tradeoff between standard risk and adversarial risk and derive the Pareto-optimal tradeoff, achievable over specific classes of models, in the infinite data limit with features dimension kept fixed. We consider three learning settings: 1) Regression with the class of linear models; 2) Binary classification under the Gaussian mixtures data model, with the class of linear classifiers; 3) Regression with the class of random features model (which can be equivalently represented as two-layer neural network with random first-layer weights). We show that a tradeoff between standard and adversarial risk is manifested in all three settings. We further characterize the Pareto-optimal tradeoff curves and discuss how a variety of factors, such as features correlation, adversary’s power or the width of two-layer neural network would affect this tradeoff.

----

## [688] Leveraging Non-uniformity in First-order Non-convex Optimization

**Authors**: *Jincheng Mei, Yue Gao, Bo Dai, Csaba Szepesvári, Dale Schuurmans*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mei21a.html](http://proceedings.mlr.press/v139/mei21a.html)

**Abstract**:

Classical global convergence results for first-order methods rely on uniform smoothness and the Ł{}ojasiewicz inequality. Motivated by properties of objective functions that arise in machine learning, we propose a non-uniform refinement of these notions, leading to \emph{Non-uniform Smoothness} (NS) and \emph{Non-uniform Ł{}ojasiewicz inequality} (NŁ{}). The new definitions inspire new geometry-aware first-order methods that are able to converge to global optimality faster than the classical $\Omega(1/t^2)$ lower bounds. To illustrate the power of these geometry-aware methods and their corresponding non-uniform analysis, we consider two important problems in machine learning: policy gradient optimization in reinforcement learning (PG), and generalized linear model training in supervised learning (GLM). For PG, we find that normalizing the gradient ascent method can accelerate convergence to $O(e^{- c \cdot t})$ (where $c > 0$) while incurring less overhead than existing algorithms. For GLM, we show that geometry-aware normalized gradient descent can also achieve a linear convergence rate, which significantly improves the best known results. We additionally show that the proposed geometry-aware gradient descent methods escape landscape plateaus faster than standard gradient descent. Experimental results are used to illustrate and complement the theoretical findings.

----

## [689] Controlling Graph Dynamics with Reinforcement Learning and Graph Neural Networks

**Authors**: *Eli A. Meirom, Haggai Maron, Shie Mannor, Gal Chechik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/meirom21a.html](http://proceedings.mlr.press/v139/meirom21a.html)

**Abstract**:

We consider the problem of controlling a partially-observed dynamic process on a graph by a limited number of interventions. This problem naturally arises in contexts such as scheduling virus tests to curb an epidemic; targeted marketing in order to promote a product; and manually inspecting posts to detect fake news spreading on social networks. We formulate this setup as a sequential decision problem over a temporal graph process. In face of an exponential state space, combinatorial action space and partial observability, we design a novel tractable scheme to control dynamical processes on temporal graphs. We successfully apply our approach to two popular problems that fall into our framework: prioritizing which nodes should be tested in order to curb the spread of an epidemic, and influence maximization on a graph.

----

## [690] A theory of high dimensional regression with arbitrary correlations between input features and target functions: sample complexity, multiple descent curves and a hierarchy of phase transitions

**Authors**: *Gabriel Mel, Surya Ganguli*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mel21a.html](http://proceedings.mlr.press/v139/mel21a.html)

**Abstract**:

The performance of neural networks depends on precise relationships between four distinct ingredients: the architecture, the loss function, the statistical structure of inputs, and the ground truth target function. Much theoretical work has focused on understanding the role of the first two ingredients under highly simplified models of random uncorrelated data and target functions. In contrast, performance likely relies on a conspiracy between the statistical structure of the input distribution and the structure of the function to be learned. To understand this better we revisit ridge regression in high dimensions, which corresponds to an exceedingly simple architecture and loss function, but we analyze its performance under arbitrary correlations between input features and the target function. We find a rich mathematical structure that includes: (1) a dramatic reduction in sample complexity when the target function aligns with data anisotropy; (2) the existence of multiple descent curves; (3) a sequence of phase transitions in the performance, loss landscape, and optimal regularization as a function of the amount of data that explains the first two effects.

----

## [691] Neural Architecture Search without Training

**Authors**: *Joe Mellor, Jack Turner, Amos J. Storkey, Elliot J. Crowley*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mellor21a.html](http://proceedings.mlr.press/v139/mellor21a.html)

**Abstract**:

The time and effort involved in hand-designing deep neural networks is immense. This has prompted the development of Neural Architecture Search (NAS) techniques to automate this design. However, NAS algorithms tend to be slow and expensive; they need to train vast numbers of candidate networks to inform the search process. This could be alleviated if we could partially predict a network’s trained accuracy from its initial state. In this work, we examine the overlap of activations between datapoints in untrained networks and motivate how this can give a measure which is usefully indicative of a network’s trained performance. We incorporate this measure into a simple algorithm that allows us to search for powerful networks without any training in a matter of seconds on a single GPU, and verify its effectiveness on NAS-Bench-101, NAS-Bench-201, NATS-Bench, and Network Design Spaces. Our approach can be readily combined with more expensive search methods; we examine a simple adaptation of regularised evolutionary search. Code for reproducing our experiments is available at https://github.com/BayesWatch/nas-without-training.

----

## [692] Fast active learning for pure exploration in reinforcement learning

**Authors**: *Pierre Ménard, Omar Darwiche Domingues, Anders Jonsson, Emilie Kaufmann, Edouard Leurent, Michal Valko*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/menard21a.html](http://proceedings.mlr.press/v139/menard21a.html)

**Abstract**:

Realistic environments often provide agents with very limited feedback. When the environment is initially unknown, the feedback, in the beginning, can be completely absent, and the agents may first choose to devote all their effort on \emph{exploring efficiently.} The exploration remains a challenge while it has been addressed with many hand-tuned heuristics with different levels of generality on one side, and a few theoretically-backed exploration strategies on the other. Many of them are incarnated by \emph{intrinsic motivation} and in particular \emph{explorations bonuses}. A common choice is to use $1/\sqrt{n}$ bonus, where $n$ is a number of times this particular state-action pair was visited. We show that, surprisingly, for a pure-exploration objective of \emph{reward-free exploration}, bonuses that scale with $1/n$ bring faster learning rates, improving the known upper bounds with respect to the dependence on the horizon $H$. Furthermore, we show that with an improved analysis of the stopping time, we can improve by a factor $H$ the sample complexity in the \emph{best-policy identification} setting, which is another pure-exploration objective, where the environment provides rewards but the agent is not penalized for its behavior during the exploration phase.

----

## [693] UCB Momentum Q-learning: Correcting the bias without forgetting

**Authors**: *Pierre Ménard, Omar Darwiche Domingues, Xuedong Shang, Michal Valko*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/menard21b.html](http://proceedings.mlr.press/v139/menard21b.html)

**Abstract**:

We propose UCBMQ, Upper Confidence Bound Momentum Q-learning, a new algorithm for reinforcement learning in tabular and possibly stage-dependent, episodic Markov decision process. UCBMQ is based on Q-learning where we add a momentum term and rely on the principle of optimism in face of uncertainty to deal with exploration. Our new technical ingredient of UCBMQ is the use of momentum to correct the bias that Q-learning suffers while, \emph{at the same time}, limiting the impact it has on the second-order term of the regret. For UCBMQ, we are able to guarantee a regret of at most $\tilde{O}(\sqrt{H^3SAT}+ H^4 S A)$ where $H$ is the length of an episode, $S$ the number of states, $A$ the number of actions, $T$ the number of episodes and ignoring terms in poly$\log(SAHT)$. Notably, UCBMQ is the first algorithm that simultaneously matches the lower bound of $\Omega(\sqrt{H^3SAT})$ for large enough $T$ and has a second-order term (with respect to $T$) that scales \emph{only linearly} with the number of states $S$.

----

## [694] An Integer Linear Programming Framework for Mining Constraints from Data

**Authors**: *Tao Meng, Kai-Wei Chang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/meng21a.html](http://proceedings.mlr.press/v139/meng21a.html)

**Abstract**:

Structured output prediction problems (e.g., sequential tagging, hierarchical multi-class classification) often involve constraints over the output space. These constraints interact with the learned models to filter infeasible solutions and facilitate in building an accountable system. However, despite constraints are useful, they are often based on hand-crafted rules. This raises a question – can we mine constraints and rules from data based on a learning algorithm? In this paper, we present a general framework for mining constraints from data. In particular, we consider the inference in structured output prediction as an integer linear programming (ILP) problem. Then, given the coefficients of the objective function and the corresponding solution, we mine the underlying constraints by estimating the outer and inner polytopes of the feasible set. We verify the proposed constraint mining algorithm in various synthetic and real-world applications and demonstrate that the proposed approach successfully identifies the feasible set at scale. In particular, we show that our approach can learn to solve 9x9 Sudoku puzzles and minimal spanning tree problems from examples without providing the underlying rules. Our algorithm can also integrate with a neural network model to learn the hierarchical label structure of a multi-label classification task. Besides, we provide theoretical analysis about the tightness of the polytopes and the reliability of the mined constraints.

----

## [695] A statistical perspective on distillation

**Authors**: *Aditya Krishna Menon, Ankit Singh Rawat, Sashank J. Reddi, Seungyeon Kim, Sanjiv Kumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/menon21a.html](http://proceedings.mlr.press/v139/menon21a.html)

**Abstract**:

Knowledge distillation is a technique for improving a “student” model by replacing its one-hot training labels with a label distribution obtained from a “teacher” model. Despite its broad success, several basic questions — e.g., Why does distillation help? Why do more accurate teachers not necessarily distill better? — have received limited formal study. In this paper, we present a statistical perspective on distillation which provides an answer to these questions. Our core observation is that a “Bayes teacher” providing the true class-probabilities can lower the variance of the student objective, and thus improve performance. We then establish a bias-variance tradeoff that quantifies the value of teachers that approximate the Bayes class-probabilities. This provides a formal criterion as to what constitutes a “good” teacher, namely, the quality of its probability estimates. Finally, we illustrate how our statistical perspective facilitates novel applications of distillation to bipartite ranking and multiclass retrieval.

----

## [696] Learn2Hop: Learned Optimization on Rough Landscapes

**Authors**: *Amil Merchant, Luke Metz, Samuel S. Schoenholz, Ekin D. Cubuk*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/merchant21a.html](http://proceedings.mlr.press/v139/merchant21a.html)

**Abstract**:

Optimization of non-convex loss surfaces containing many local minima remains a critical problem in a variety of domains, including operations research, informatics, and material design. Yet, current techniques either require extremely high iteration counts or a large number of random restarts for good performance. In this work, we propose adapting recent developments in meta-learning to these many-minima problems by learning the optimization algorithm for various loss landscapes. We focus on problems from atomic structural optimization—finding low energy configurations of many-atom systems—including widely studied models such as bimetallic clusters and disordered silicon. We find that our optimizer learns a hopping behavior which enables efficient exploration and improves the rate of low energy minima discovery. Finally, our learned optimizers show promising generalization with efficiency gains on never before seen tasks (e.g. new elements or compositions). Code is available at https://learn2hop.page.link/github.

----

## [697] Counterfactual Credit Assignment in Model-Free Reinforcement Learning

**Authors**: *Thomas Mesnard, Theophane Weber, Fabio Viola, Shantanu Thakoor, Alaa Saade, Anna Harutyunyan, Will Dabney, Thomas S. Stepleton, Nicolas Heess, Arthur Guez, Eric Moulines, Marcus Hutter, Lars Buesing, Rémi Munos*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mesnard21a.html](http://proceedings.mlr.press/v139/mesnard21a.html)

**Abstract**:

Credit assignment in reinforcement learning is the problem of measuring an action’s influence on future rewards. In particular, this requires separating skill from luck, i.e. disentangling the effect of an action on rewards from that of external factors and subsequent actions. To achieve this, we adapt the notion of counterfactuals from causality theory to a model-free RL setup. The key idea is to condition value functions on future events, by learning to extract relevant information from a trajectory. We formulate a family of policy gradient algorithms that use these future-conditional value functions as baselines or critics, and show that they are provably low variance. To avoid the potential bias from conditioning on future information, we constrain the hindsight information to not contain information about the agent’s actions. We demonstrate the efficacy and validity of our algorithm on a number of illustrative and challenging problems.

----

## [698] Provably Efficient Learning of Transferable Rewards

**Authors**: *Alberto Maria Metelli, Giorgia Ramponi, Alessandro Concetti, Marcello Restelli*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/metelli21a.html](http://proceedings.mlr.press/v139/metelli21a.html)

**Abstract**:

The reward function is widely accepted as a succinct, robust, and transferable representation of a task. Typical approaches, at the basis of Inverse Reinforcement Learning (IRL), leverage on expert demonstrations to recover a reward function. In this paper, we study the theoretical properties of the class of reward functions that are compatible with the expert’s behavior. We analyze how the limited knowledge of the expert’s policy and of the environment affects the reward reconstruction phase. Then, we examine how the error propagates to the learned policy’s performance when transferring the reward function to a different environment. We employ these findings to devise a provably efficient active sampling approach, aware of the need for transferring the reward function, that can be paired with a large variety of IRL algorithms. Finally, we provide numerical simulations on benchmark environments.

----

## [699] Mixed Nash Equilibria in the Adversarial Examples Game

**Authors**: *Laurent Meunier, Meyer Scetbon, Rafael Pinot, Jamal Atif, Yann Chevaleyre*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/meunier21a.html](http://proceedings.mlr.press/v139/meunier21a.html)

**Abstract**:

This paper tackles the problem of adversarial examples from a game theoretic point of view. We study the open question of the existence of mixed Nash equilibria in the zero-sum game formed by the attacker and the classifier. While previous works usually allow only one player to use randomized strategies, we show the necessity of considering randomization for both the classifier and the attacker. We demonstrate that this game has no duality gap, meaning that it always admits approximate Nash equilibria. We also provide the first optimization algorithms to learn a mixture of classifiers that approximately realizes the value of this game, \emph{i.e.} procedures to build an optimally robust randomized classifier.

----

## [700] Learning in Nonzero-Sum Stochastic Games with Potentials

**Authors**: *David Henry Mguni, Yutong Wu, Yali Du, Yaodong Yang, Ziyi Wang, Minne Li, Ying Wen, Joel Jennings, Jun Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mguni21a.html](http://proceedings.mlr.press/v139/mguni21a.html)

**Abstract**:

Multi-agent reinforcement learning (MARL) has become effective in tackling discrete cooperative game scenarios. However, MARL has yet to penetrate settings beyond those modelled by team and zero-sum games, confining it to a small subset of multi-agent systems. In this paper, we introduce a new generation of MARL learners that can handle \textit{nonzero-sum} payoff structures and continuous settings. In particular, we study the MARL problem in a class of games known as stochastic potential games (SPGs) with continuous state-action spaces. Unlike cooperative games, in which all agents share a common reward, SPGs are capable of modelling real-world scenarios where agents seek to fulfil their individual goals. We prove theoretically our learning method, $\ourmethod$, enables independent agents to learn Nash equilibrium strategies in \textit{polynomial time}. We demonstrate our framework tackles previously unsolvable tasks such as \textit{Coordination Navigation} and \textit{large selfish routing games} and that it outperforms the state of the art MARL baselines such as MADDPG and COMIX in such scenarios.

----

## [701] EfficientTTS: An Efficient and High-Quality Text-to-Speech Architecture

**Authors**: *Chenfeng Miao, Shuang Liang, Zhengchen Liu, Minchuan Chen, Jun Ma, Shaojun Wang, Jing Xiao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/miao21a.html](http://proceedings.mlr.press/v139/miao21a.html)

**Abstract**:

In this work, we address the Text-to-Speech (TTS) task by proposing a non-autoregressive architecture called EfficientTTS. Unlike the dominant non-autoregressive TTS models, which are trained with the need of external aligners, EfficientTTS optimizes all its parameters with a stable, end-to-end training procedure, allowing for synthesizing high quality speech in a fast and efficient manner. EfficientTTS is motivated by a new monotonic alignment modeling approach, which specifies monotonic constraints to the sequence alignment with almost no increase of computation. By combining EfficientTTS with different feed-forward network structures, we develop a family of TTS models, including both text-to-melspectrogram and text-to-waveform networks. We experimentally show that the proposed models significantly outperform counterpart models such as Tacotron 2 and Glow-TTS in terms of speech quality, training efficiency and synthesis speed, while still producing the speeches of strong robustness and great diversity. In addition, we demonstrate that proposed approach can be easily extended to autoregressive models such as Tacotron 2.

----

## [702] Outside the Echo Chamber: Optimizing the Performative Risk

**Authors**: *John Miller, Juan C. Perdomo, Tijana Zrnic*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/miller21a.html](http://proceedings.mlr.press/v139/miller21a.html)

**Abstract**:

In performative prediction, predictions guide decision-making and hence can influence the distribution of future data. To date, work on performative prediction has focused on finding performatively stable models, which are the fixed points of repeated retraining. However, stable solutions can be far from optimal when evaluated in terms of the performative risk, the loss experienced by the decision maker when deploying a model. In this paper, we shift attention beyond performative stability and focus on optimizing the performative risk directly. We identify a natural set of properties of the loss function and model-induced distribution shift under which the performative risk is convex, a property which does not follow from convexity of the loss alone. Furthermore, we develop algorithms that leverage our structural assumptions to optimize the performative risk with better sample efficiency than generic methods for derivative-free convex optimization.

----

## [703] Accuracy on the Line: on the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization

**Authors**: *John Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, Ludwig Schmidt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/miller21b.html](http://proceedings.mlr.press/v139/miller21b.html)

**Abstract**:

For machine learning systems to be reliable, we must understand their performance in unseen, out- of-distribution environments. In this paper, we empirically show that out-of-distribution performance is strongly correlated with in-distribution performance for a wide range of models and distribution shifts. Specifically, we demonstrate strong correlations between in-distribution and out-of- distribution performance on variants of CIFAR- 10 & ImageNet, a synthetic pose estimation task derived from YCB objects, FMoW-WILDS satellite imagery classification, and wildlife classification in iWildCam-WILDS. The correlation holds across model architectures, hyperparameters, training set size, and training duration, and is more precise than what is expected from existing domain adaptation theory. To complete the picture, we also investigate cases where the correlation is weaker, for instance some synthetic distribution shifts from CIFAR-10-C and the tissue classification dataset Camelyon17-WILDS. Finally, we provide a candidate theory based on a Gaussian data model that shows how changes in the data covariance arising from distribution shift can affect the observed correlations.

----

## [704] Signatured Deep Fictitious Play for Mean Field Games with Common Noise

**Authors**: *Ming Min, Ruimeng Hu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/min21a.html](http://proceedings.mlr.press/v139/min21a.html)

**Abstract**:

Existing deep learning methods for solving mean-field games (MFGs) with common noise fix the sampling common noise paths and then solve the corresponding MFGs. This leads to a nested loop structure with millions of simulations of common noise paths in order to produce accurate solutions, which results in prohibitive computational cost and limits the applications to a large extent. In this paper, based on the rough path theory, we propose a novel single-loop algorithm, named signatured deep fictitious play (Sig-DFP), by which we can work with the unfixed common noise setup to avoid the nested loop structure and reduce the computational complexity significantly. The proposed algorithm can accurately capture the effect of common uncertainty changes on mean-field equilibria without further training of neural networks, as previously needed in the existing machine learning algorithms. The efficiency is supported by three applications, including linear-quadratic MFGs, mean-field portfolio game, and mean-field game of optimal consumption and investment. Overall, we provide a new point of view from the rough path theory to solve MFGs with common noise with significantly improved efficiency and an extensive range of applications. In addition, we report the first deep learning work to deal with extended MFGs (a mean-field interaction via both the states and controls) with common noise.

----

## [705] Meta-StyleSpeech : Multi-Speaker Adaptive Text-to-Speech Generation

**Authors**: *Dongchan Min, Dong Bok Lee, Eunho Yang, Sung Ju Hwang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/min21b.html](http://proceedings.mlr.press/v139/min21b.html)

**Abstract**:

With rapid progress in neural text-to-speech (TTS) models, personalized speech generation is now in high demand for many applications. For practical applicability, a TTS model should generate high-quality speech with only a few audio samples from the given speaker, that are also short in length. However, existing methods either require to fine-tune the model or achieve low adaptation quality without fine-tuning. In this work, we propose StyleSpeech, a new TTS model which not only synthesizes high-quality speech but also effectively adapts to new speakers. Specifically, we propose Style-Adaptive Layer Normalization (SALN) which aligns gain and bias of the text input according to the style extracted from a reference speech audio. With SALN, our model effectively synthesizes speech in the style of the target speaker even from a single speech audio. Furthermore, to enhance StyleSpeech’s adaptation to speech from new speakers, we extend it to Meta-StyleSpeech by introducing two discriminators trained with style prototypes, and performing episodic training. The experimental results show that our models generate high-quality speech which accurately follows the speaker’s voice with single short-duration (1-3 sec) speech audio, significantly outperforming baselines.

----

## [706] On the Explicit Role of Initialization on the Convergence and Implicit Bias of Overparametrized Linear Networks

**Authors**: *Hancheng Min, Salma Tarmoun, René Vidal, Enrique Mallada*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/min21c.html](http://proceedings.mlr.press/v139/min21c.html)

**Abstract**:

Neural networks trained via gradient descent with random initialization and without any regularization enjoy good generalization performance in practice despite being highly overparametrized. A promising direction to explain this phenomenon is to study how initialization and overparametrization affect convergence and implicit bias of training algorithms. In this paper, we present a novel analysis of single-hidden-layer linear networks trained under gradient flow, which connects initialization, optimization, and overparametrization. Firstly, we show that the squared loss converges exponentially to its optimum at a rate that depends on the level of imbalance of the initialization. Secondly, we show that proper initialization constrains the dynamics of the network parameters to lie within an invariant set. In turn, minimizing the loss over this set leads to the min-norm solution. Finally, we show that large hidden layer width, together with (properly scaled) random initialization, ensures proximity to such an invariant set during training, allowing us to derive a novel non-asymptotic upper-bound on the distance between the trained network and the min-norm solution.

----

## [707] An Identifiable Double VAE For Disentangled Representations

**Authors**: *Graziano Mita, Maurizio Filippone, Pietro Michiardi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mita21a.html](http://proceedings.mlr.press/v139/mita21a.html)

**Abstract**:

A large part of the literature on learning disentangled representations focuses on variational autoencoders (VAEs). Recent developments demonstrate that disentanglement cannot be obtained in a fully unsupervised setting without inductive biases on models and data. However, Khemakhem et al., AISTATS, 2020 suggest that employing a particular form of factorized prior, conditionally dependent on auxiliary variables complementing input observations, can be one such bias, resulting in an identifiable model with guarantees on disentanglement. Working along this line, we propose a novel VAE-based generative model with theoretical guarantees on identifiability. We obtain our conditional prior over the latents by learning an optimal representation, which imposes an additional strength on their regularization. We also extend our method to semi-supervised settings. Experimental results indicate superior performance with respect to state-of-the-art approaches, according to several established metrics proposed in the literature on disentanglement.

----

## [708] Offline Meta-Reinforcement Learning with Advantage Weighting

**Authors**: *Eric Mitchell, Rafael Rafailov, Xue Bin Peng, Sergey Levine, Chelsea Finn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mitchell21a.html](http://proceedings.mlr.press/v139/mitchell21a.html)

**Abstract**:

This paper introduces the offline meta-reinforcement learning (offline meta-RL) problem setting and proposes an algorithm that performs well in this setting. Offline meta-RL is analogous to the widely successful supervised learning strategy of pre-training a model on a large batch of fixed, pre-collected data (possibly from various tasks) and fine-tuning the model to a new task with relatively little data. That is, in offline meta-RL, we meta-train on fixed, pre-collected data from several tasks and adapt to a new task with a very small amount (less than 5 trajectories) of data from the new task. By nature of being offline, algorithms for offline meta-RL can utilize the largest possible pool of training data available and eliminate potentially unsafe or costly data collection during meta-training. This setting inherits the challenges of offline RL, but it differs significantly because offline RL does not generally consider a) transfer to new tasks or b) limited data from the test task, both of which we face in offline meta-RL. Targeting the offline meta-RL setting, we propose Meta-Actor Critic with Advantage Weighting (MACAW). MACAW is an optimization-based meta-learning algorithm that uses simple, supervised regression objectives for both the inner and outer loop of meta-training. On offline variants of common meta-RL benchmarks, we empirically find that this approach enables fully offline meta-reinforcement learning and achieves notable gains over prior methods.

----

## [709] The Power of Log-Sum-Exp: Sequential Density Ratio Matrix Estimation for Speed-Accuracy Optimization

**Authors**: *Taiki Miyagawa, Akinori F. Ebihara*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/miyagawa21a.html](http://proceedings.mlr.press/v139/miyagawa21a.html)

**Abstract**:

We propose a model for multiclass classification of time series to make a prediction as early and as accurate as possible. The matrix sequential probability ratio test (MSPRT) is known to be asymptotically optimal for this setting, but contains a critical assumption that hinders broad real-world applications; the MSPRT requires the underlying probability density. To address this problem, we propose to solve density ratio matrix estimation (DRME), a novel type of density ratio estimation that consists of estimating matrices of multiple density ratios with constraints and thus is more challenging than the conventional density ratio estimation. We propose a log-sum-exp-type loss function (LSEL) for solving DRME and prove the following: (i) the LSEL provides the true density ratio matrix as the sample size of the training set increases (consistency); (ii) it assigns larger gradients to harder classes (hard class weighting effect); and (iii) it provides discriminative scores even on class-imbalanced datasets (guess-aversion). Our overall architecture for early classification, MSPRT-TANDEM, statistically significantly outperforms baseline models on four datasets including action recognition, especially in the early stage of sequential observations. Our code and datasets are publicly available.

----

## [710] PODS: Policy Optimization via Differentiable Simulation

**Authors**: *Miguel Zamora, Momchil Peychev, Sehoon Ha, Martin T. Vechev, Stelian Coros*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mora21a.html](http://proceedings.mlr.press/v139/mora21a.html)

**Abstract**:

Current reinforcement learning (RL) methods use simulation models as simple black-box oracles. In this paper, with the goal of improving the performance exhibited by RL algorithms, we explore a systematic way of leveraging the additional information provided by an emerging class of differentiable simulators. Building on concepts established by Deterministic Policy Gradients (DPG) methods, the neural network policies learned with our approach represent deterministic actions. In a departure from standard methodologies, however, learning these policies does not hinge on approximations of the value function that must be learned concurrently in an actor-critic fashion. Instead, we exploit differentiable simulators to directly compute the analytic gradient of a policy’s value function with respect to the actions it outputs. This, in turn, allows us to efficiently perform locally optimal policy improvement iterations. Compared against other state-of-the-art RL methods, we show that with minimal hyper-parameter tuning our approach consistently leads to better asymptotic behavior across a set of payload manipulation tasks that demand a high degree of accuracy and precision.

----

## [711] Efficient Deviation Types and Learning for Hindsight Rationality in Extensive-Form Games

**Authors**: *Dustin Morrill, Ryan D'Orazio, Marc Lanctot, James R. Wright, Michael Bowling, Amy R. Greenwald*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/morrill21a.html](http://proceedings.mlr.press/v139/morrill21a.html)

**Abstract**:

Hindsight rationality is an approach to playing general-sum games that prescribes no-regret learning dynamics for individual agents with respect to a set of deviations, and further describes jointly rational behavior among multiple agents with mediated equilibria. To develop hindsight rational learning in sequential decision-making settings, we formalize behavioral deviations as a general class of deviations that respect the structure of extensive-form games. Integrating the idea of time selection into counterfactual regret minimization (CFR), we introduce the extensive-form regret minimization (EFR) algorithm that achieves hindsight rationality for any given set of behavioral deviations with computation that scales closely with the complexity of the set. We identify behavioral deviation subsets, the partial sequence deviation types, that subsume previously studied types and lead to efficient EFR instances in games with moderate lengths. In addition, we present a thorough empirical analysis of EFR instantiated with different deviation types in benchmark games, where we find that stronger types typically induce better performance.

----

## [712] Neural Rough Differential Equations for Long Time Series

**Authors**: *James Morrill, Cristopher Salvi, Patrick Kidger, James Foster*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/morrill21b.html](http://proceedings.mlr.press/v139/morrill21b.html)

**Abstract**:

Neural controlled differential equations (CDEs) are the continuous-time analogue of recurrent neural networks, as Neural ODEs are to residual networks, and offer a memory-efficient continuous-time way to model functions of potentially irregular time series. Existing methods for computing the forward pass of a Neural CDE involve embedding the incoming time series into path space, often via interpolation, and using evaluations of this path to drive the hidden state. Here, we use rough path theory to extend this formulation. Instead of directly embedding into path space, we instead represent the input signal over small time intervals through its \textit{log-signature}, which are statistics describing how the signal drives a CDE. This is the approach for solving \textit{rough differential equations} (RDEs), and correspondingly we describe our main contribution as the introduction of Neural RDEs. This extension has a purpose: by generalising the Neural CDE approach to a broader class of driving signals, we demonstrate particular advantages for tackling long time series. In this regime, we demonstrate efficacy on problems of length up to 17k observations and observe significant training speed-ups, improvements in model performance, and reduced memory requirements compared to existing approaches.

----

## [713] Connecting Interpretability and Robustness in Decision Trees through Separation

**Authors**: *Michal Moshkovitz, Yao-Yuan Yang, Kamalika Chaudhuri*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/moshkovitz21a.html](http://proceedings.mlr.press/v139/moshkovitz21a.html)

**Abstract**:

Recent research has recognized interpretability and robustness as essential properties of trustworthy classification. Curiously, a connection between robustness and interpretability was empirically observed, but the theoretical reasoning behind it remained elusive. In this paper, we rigorously investigate this connection. Specifically, we focus on interpretation using decision trees and robustness to l_{\infty}-perturbation. Previous works defined the notion of r-separation as a sufficient condition for robustness. We prove upper and lower bounds on the tree size in case the data is r-separated. We then show that a tighter bound on the size is possible when the data is linearly separated. We provide the first algorithm with provable guarantees both on robustness, interpretability, and accuracy in the context of decision trees. Experiments confirm that our algorithm yields classifiers that are both interpretable and robust and have high accuracy.

----

## [714] Outlier-Robust Optimal Transport

**Authors**: *Debarghya Mukherjee, Aritra Guha, Justin M. Solomon, Yuekai Sun, Mikhail Yurochkin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mukherjee21a.html](http://proceedings.mlr.press/v139/mukherjee21a.html)

**Abstract**:

Optimal transport (OT) measures distances between distributions in a way that depends on the geometry of the sample space. In light of recent advances in computational OT, OT distances are widely used as loss functions in machine learning. Despite their prevalence and advantages, OT loss functions can be extremely sensitive to outliers. In fact, a single adversarially-picked outlier can increase the standard $W_2$-distance arbitrarily. To address this issue, we propose an outlier-robust formulation of OT. Our formulation is convex but challenging to scale at a first glance. Our main contribution is deriving an \emph{equivalent} formulation based on cost truncation that is easy to incorporate into modern algorithms for computational OT. We demonstrate the benefits of our formulation in mean estimation problems under the Huber contamination model in simulations and outlier detection tasks on real data.

----

## [715] Oblivious Sketching for Logistic Regression

**Authors**: *Alexander Munteanu, Simon Omlor, David P. Woodruff*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/munteanu21a.html](http://proceedings.mlr.press/v139/munteanu21a.html)

**Abstract**:

What guarantees are possible for solving logistic regression in one pass over a data stream? To answer this question, we present the first data oblivious sketch for logistic regression. Our sketch can be computed in input sparsity time over a turnstile data stream and reduces the size of a $d$-dimensional data set from $n$ to only $\operatorname{poly}(\mu d\log n)$ weighted points, where $\mu$ is a useful parameter which captures the complexity of compressing the data. Solving (weighted) logistic regression on the sketch gives an $O(\log n)$-approximation to the original problem on the full data set. We also show how to obtain an $O(1)$-approximation with slight modifications. Our sketches are fast, simple, easy to implement, and our experiments demonstrate their practicality.

----

## [716] Bias-Variance Reduced Local SGD for Less Heterogeneous Federated Learning

**Authors**: *Tomoya Murata, Taiji Suzuki*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/murata21a.html](http://proceedings.mlr.press/v139/murata21a.html)

**Abstract**:

Recently, local SGD has got much attention and been extensively studied in the distributed learning community to overcome the communication bottleneck problem. However, the superiority of local SGD to minibatch SGD only holds in quite limited situations. In this paper, we study a new local algorithm called Bias-Variance Reduced Local SGD (BVR-L-SGD) for nonconvex distributed optimization. Algorithmically, our proposed bias and variance reduced local gradient estimator fully utilizes small second-order heterogeneity of local objectives and suggests randomly picking up one of the local models instead of taking the average of them when workers are synchronized. Theoretically, under small heterogeneity of local objectives, we show that BVR-L-SGD achieves better communication complexity than both the previous non-local and local methods under mild conditions, and particularly BVR-L-SGD is the first method that breaks the barrier of communication complexity $\Theta(1/\varepsilon)$ for general nonconvex smooth objectives when the heterogeneity is small and the local computation budget is large. Numerical results are given to verify the theoretical findings and give empirical evidence of the superiority of our method.

----

## [717] Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold

**Authors**: *Kieran A. Murphy, Carlos Esteves, Varun Jampani, Srikumar Ramalingam, Ameesh Makadia*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/murphy21a.html](http://proceedings.mlr.press/v139/murphy21a.html)

**Abstract**:

In the deep learning era, the vast majority of methods to predict pose from a single image are trained to classify or regress to a single given ground truth pose per image. Such methods have two main shortcomings, i) they cannot represent uncertainty about the predictions, and ii) they cannot handle symmetric objects, where multiple (potentially infinite) poses may be correct. Only recently these shortcomings have been addressed, but current approaches as limited in that they cannot express the full rich space of distributions on the rotation manifold. To this end, we introduce a method to estimate arbitrary, non-parametric distributions on SO(3). Our key idea is to represent the distributions implicitly, with a neural network that estimates the probability density, given the input image and a candidate pose. At inference time, grid sampling or gradient ascent can be used to find the most likely pose, but it is also possible to evaluate the density at any pose, enabling reasoning about symmetries and uncertainty. This is the most general way of representing distributions on manifolds, and to demonstrate its expressive power we introduce a new dataset containing symmetric and nearly-symmetric objects. Our method also shows advantages on the popular object pose estimation benchmarks ModelNet10-SO(3) and T-LESS. Code, data, and visualizations may be found at implicit-pdf.github.io.

----

## [718] No-regret Algorithms for Capturing Events in Poisson Point Processes

**Authors**: *Mojmir Mutny, Andreas Krause*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/mutny21a.html](http://proceedings.mlr.press/v139/mutny21a.html)

**Abstract**:

Inhomogeneous Poisson point processes are widely used models of event occurrences. We address \emph{adaptive sensing of Poisson Point processes}, namely, maximizing the number of captured events subject to sensing costs. We encode prior assumptions on the rate function by modeling it as a member of a known \emph{reproducing kernel Hilbert space} (RKHS). By partitioning the domain into separate small regions, and using heteroscedastic linear regression, we propose a tractable estimator of Poisson process rates for two feedback models: \emph{count-record}, where exact locations of events are observed, and \emph{histogram} feedback, where only counts of events are observed. We derive provably accurate anytime confidence estimates for our estimators for sequentially acquired Poisson count data. Using these, we formulate algorithms based on optimism that provably incur sublinear count-regret. We demonstrate the practicality of the method on problems from crime modeling, revenue maximization as well as environmental monitoring.

----

## [719] Online Limited Memory Neural-Linear Bandits with Likelihood Matching

**Authors**: *Ofir Nabati, Tom Zahavy, Shie Mannor*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nabati21a.html](http://proceedings.mlr.press/v139/nabati21a.html)

**Abstract**:

We study neural-linear bandits for solving problems where {\em both} exploration and representation learning play an important role. Neural-linear bandits harnesses the representation power of Deep Neural Networks (DNNs) and combines it with efficient exploration mechanisms by leveraging uncertainty estimation of the model, designed for linear contextual bandits on top of the last hidden layer. In order to mitigate the problem of representation change during the process, new uncertainty estimations are computed using stored data from an unlimited buffer. Nevertheless, when the amount of stored data is limited, a phenomenon called catastrophic forgetting emerges. To alleviate this, we propose a likelihood matching algorithm that is resilient to catastrophic forgetting and is completely online. We applied our algorithm, Limited Memory Neural-Linear with Likelihood Matching (NeuralLinear-LiM2) on a variety of datasets and observed that our algorithm achieves comparable performance to the unlimited memory approach while exhibits resilience to catastrophic forgetting.

----

## [720] Quantitative Understanding of VAE as a Non-linearly Scaled Isometric Embedding

**Authors**: *Akira Nakagawa, Keizo Kato, Taiji Suzuki*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nakagawa21a.html](http://proceedings.mlr.press/v139/nakagawa21a.html)

**Abstract**:

Variational autoencoder (VAE) estimates the posterior parameters (mean and variance) of latent variables corresponding to each input data. While it is used for many tasks, the transparency of the model is still an underlying issue. This paper provides a quantitative understanding of VAE property through the differential geometric and information-theoretic interpretations of VAE. According to the Rate-distortion theory, the optimal transform coding is achieved by using an orthonormal transform with PCA basis where the transform space is isometric to the input. Considering the analogy of transform coding to VAE, we clarify theoretically and experimentally that VAE can be mapped to an implicit isometric embedding with a scale factor derived from the posterior parameter. As a result, we can estimate the data probabilities in the input space from the prior, loss metrics, and corresponding posterior parameters, and further, the quantitative importance of each latent variable can be evaluated like the eigenvalue of PCA.

----

## [721] GMAC: A Distributional Perspective on Actor-Critic Framework

**Authors**: *Daniel Wontae Nam, Younghoon Kim, Chan Y. Park*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nam21a.html](http://proceedings.mlr.press/v139/nam21a.html)

**Abstract**:

In this paper, we devise a distributional framework on actor-critic as a solution to distributional instability, action type restriction, and conflation between samples and statistics. We propose a new method that minimizes the Cram{é}r distance with the multi-step Bellman target distribution generated from a novel Sample-Replacement algorithm denoted SR(\lambda), which learns the correct value distribution under multiple Bellman operations. Parameterizing a value distribution with Gaussian Mixture Model further improves the efficiency and the performance of the method, which we name GMAC. We empirically show that GMAC captures the correct representation of value distributions and improves the performance of a conventional actor-critic method with low computational cost, in both discrete and continuous action spaces using Arcade Learning Environment (ALE) and PyBullet environment.

----

## [722] Memory-Efficient Pipeline-Parallel DNN Training

**Authors**: *Deepak Narayanan, Amar Phanishayee, Kaiyu Shi, Xie Chen, Matei Zaharia*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/narayanan21a.html](http://proceedings.mlr.press/v139/narayanan21a.html)

**Abstract**:

Many state-of-the-art ML results have been obtained by scaling up the number of parameters in existing models. However, parameters and activations for such large models often do not fit in the memory of a single accelerator device; this means that it is necessary to distribute training of large models over multiple accelerators. In this work, we propose PipeDream-2BW, a system that supports memory-efficient pipeline parallelism. PipeDream-2BW uses a novel pipelining and weight gradient coalescing strategy, combined with the double buffering of weights, to ensure high throughput, low memory footprint, and weight update semantics similar to data parallelism. In addition, PipeDream-2BW automatically partitions the model over the available hardware resources, while respecting hardware constraints such as memory capacities of accelerators and interconnect topologies. PipeDream-2BW can accelerate the training of large GPT and BERT language models by up to 20x with similar final model accuracy.

----

## [723] Randomized Dimensionality Reduction for Facility Location and Single-Linkage Clustering

**Authors**: *Shyam Narayanan, Sandeep Silwal, Piotr Indyk, Or Zamir*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/narayanan21b.html](http://proceedings.mlr.press/v139/narayanan21b.html)

**Abstract**:

Random dimensionality reduction is a versatile tool for speeding up algorithms for high-dimensional problems. We study its application to two clustering problems: the facility location problem, and the single-linkage hierarchical clustering problem, which is equivalent to computing the minimum spanning tree. We show that if we project the input pointset $X$ onto a random $d = O(d_X)$-dimensional subspace (where $d_X$ is the doubling dimension of $X$), then the optimum facility location cost in the projected space approximates the original cost up to a constant factor. We show an analogous statement for minimum spanning tree, but with the dimension $d$ having an extra $\log \log n$ term and the approximation factor being arbitrarily close to $1$. Furthermore, we extend these results to approximating {\em solutions} instead of just their {\em costs}. Lastly, we provide experimental results to validate the quality of solutions and the speedup due to the dimensionality reduction. Unlike several previous papers studying this approach in the context of $k$-means and $k$-medians, our dimension bound does not depend on the number of clusters but only on the intrinsic dimensionality of $X$.

----

## [724] Generating images with sparse representations

**Authors**: *Charlie Nash, Jacob Menick, Sander Dieleman, Peter W. Battaglia*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nash21a.html](http://proceedings.mlr.press/v139/nash21a.html)

**Abstract**:

The high dimensionality of images presents architecture and sampling-efficiency challenges for likelihood-based generative models. Previous approaches such as VQ-VAE use deep autoencoders to obtain compact representations, which are more practical as inputs for likelihood-based models. We present an alternative approach, inspired by common image compression methods like JPEG, and convert images to quantized discrete cosine transform (DCT) blocks, which are represented sparsely as a sequence of DCT channel, spatial location, and DCT coefficient triples. We propose a Transformer-based autoregressive architecture, which is trained to sequentially predict the conditional distribution of the next element in such sequences, and which scales effectively to high resolution images. On a range of image datasets, we demonstrate that our approach can generate high quality, diverse images, with sample metric scores competitive with state of the art methods. We additionally show that simple modifications to our method yield effective image colorization and super-resolution models.

----

## [725] Geometric convergence of elliptical slice sampling

**Authors**: *Viacheslav Natarovskii, Daniel Rudolf, Björn Sprungk*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/natarovskii21a.html](http://proceedings.mlr.press/v139/natarovskii21a.html)

**Abstract**:

For Bayesian learning, given likelihood function and Gaussian prior, the elliptical slice sampler, introduced by Murray, Adams and MacKay 2010, provides a tool for the construction of a Markov chain for approximate sampling of the underlying posterior distribution. Besides of its wide applicability and simplicity its main feature is that no tuning is necessary. Under weak regularity assumptions on the posterior density we show that the corresponding Markov chain is geometrically ergodic and therefore yield qualitative convergence guarantees. We illustrate our result for Gaussian posteriors as they appear in Gaussian process regression in a fully Gaussian scenario, which for example is exhibited in Gaussian process regression, as well as in a setting of a multi-modal distribution. Remarkably, our numerical experiments indicate a dimension-independent performance of elliptical slice sampling even in situations where our ergodicity result does not apply.

----

## [726] HardCoRe-NAS: Hard Constrained diffeRentiable Neural Architecture Search

**Authors**: *Niv Nayman, Yonathan Aflalo, Asaf Noy, Lihi Zelnik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nayman21a.html](http://proceedings.mlr.press/v139/nayman21a.html)

**Abstract**:

Realistic use of neural networks often requires adhering to multiple constraints on latency, energy and memory among others. A popular approach to find fitting networks is through constrained Neural Architecture Search (NAS), however, previous methods enforce the constraint only softly. Therefore, the resulting networks do not exactly adhere to the resource constraint and their accuracy is harmed. In this work we resolve this by introducing Hard Constrained diffeRentiable NAS (HardCoRe-NAS), that is based on an accurate formulation of the expected resource requirement and a scalable search method that satisfies the hard constraint throughout the search. Our experiments show that HardCoRe-NAS generates state-of-the-art architectures, surpassing other NAS methods, while strictly satisfying the hard resource constraints without any tuning required.

----

## [727] Emergent Social Learning via Multi-agent Reinforcement Learning

**Authors**: *Kamal Ndousse, Douglas Eck, Sergey Levine, Natasha Jaques*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ndousse21a.html](http://proceedings.mlr.press/v139/ndousse21a.html)

**Abstract**:

Social learning is a key component of human and animal intelligence. By taking cues from the behavior of experts in their environment, social learners can acquire sophisticated behavior and rapidly adapt to new circumstances. This paper investigates whether independent reinforcement learning (RL) agents in a multi-agent environment can learn to use social learning to improve their performance. We find that in most circumstances, vanilla model-free RL agents do not use social learning. We analyze the reasons for this deficiency, and show that by imposing constraints on the training environment and introducing a model-based auxiliary loss we are able to obtain generalized social learning policies which enable agents to: i) discover complex skills that are not learned from single-agent training, and ii) adapt online to novel environments by taking cues from experts present in the new environment. In contrast, agents trained with model-free RL or imitation learning generalize poorly and do not succeed in the transfer tasks. By mixing multi-agent and solo training, we can obtain agents that use social learning to gain skills that they can deploy when alone, even out-performing agents trained alone from the start.

----

## [728] Bayesian Algorithm Execution: Estimating Computable Properties of Black-box Functions Using Mutual Information

**Authors**: *Willie Neiswanger, Ke Alexander Wang, Stefano Ermon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/neiswanger21a.html](http://proceedings.mlr.press/v139/neiswanger21a.html)

**Abstract**:

In many real world problems, we want to infer some property of an expensive black-box function f, given a budget of T function evaluations. One example is budget constrained global optimization of f, for which Bayesian optimization is a popular method. Other properties of interest include local optima, level sets, integrals, or graph-structured information induced by f. Often, we can find an algorithm A to compute the desired property, but it may require far more than T queries to execute. Given such an A, and a prior distribution over f, we refer to the problem of inferring the output of A using T evaluations as Bayesian Algorithm Execution (BAX). To tackle this problem, we present a procedure, InfoBAX, that sequentially chooses queries that maximize mutual information with respect to the algorithm’s output. Applying this to Dijkstra’s algorithm, for instance, we infer shortest paths in synthetic and real-world graphs with black-box edge costs. Using evolution strategies, we yield variants of Bayesian optimization that target local, rather than global, optima. On these problems, InfoBAX uses up to 500 times fewer queries to f than required by the original algorithm. Our method is closely connected to other Bayesian optimal experimental design procedures such as entropy search methods and optimal sensor placement using Gaussian processes.

----

## [729] Continuous Coordination As a Realistic Scenario for Lifelong Learning

**Authors**: *Hadi Nekoei, Akilesh Badrinaaraayanan, Aaron C. Courville, Sarath Chandar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nekoei21a.html](http://proceedings.mlr.press/v139/nekoei21a.html)

**Abstract**:

Current deep reinforcement learning (RL) algorithms are still highly task-specific and lack the ability to generalize to new environments. Lifelong learning (LLL), however, aims at solving multiple tasks sequentially by efficiently transferring and using knowledge between tasks. Despite a surge of interest in lifelong RL in recent years, the lack of a realistic testbed makes robust evaluation of LLL algorithms difficult. Multi-agent RL (MARL), on the other hand, can be seen as a natural scenario for lifelong RL due to its inherent non-stationarity, since the agents’ policies change over time. In this work, we introduce a multi-agent lifelong learning testbed that supports both zero-shot and few-shot settings. Our setup is based on Hanabi {—} a partially-observable, fully cooperative multi-agent game that has been shown to be challenging for zero-shot coordination. Its large strategy space makes it a desirable environment for lifelong RL tasks. We evaluate several recent MARL methods, and benchmark state-of-the-art LLL algorithms in limited memory and computation regimes to shed light on their strengths and weaknesses. This continual learning paradigm also provides us with a pragmatic way of going beyond centralized training which is the most commonly used training protocol in MARL. We empirically show that the agents trained in our setup are able to coordinate well with unseen agents, without any additional assumptions made by previous works. The code and all pre-trained models are available at https://github.com/chandar-lab/Lifelong-Hanabi.

----

## [730] Policy Caches with Successor Features

**Authors**: *Mark W. Nemecek, Ron Parr*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nemecek21a.html](http://proceedings.mlr.press/v139/nemecek21a.html)

**Abstract**:

Transfer in reinforcement learning is based on the idea that it is possible to use what is learned in one task to improve the learning process in another task. For transfer between tasks which share transition dynamics but differ in reward function, successor features have been shown to be a useful representation which allows for efficient computation of action-value functions for previously-learned policies in new tasks. These functions induce policies in the new tasks, so an agent may not need to learn a new policy for each new task it encounters, especially if it is allowed some amount of suboptimality in those tasks. We present new bounds for the performance of optimal policies in a new task, as well as an approach to use these bounds to decide, when presented with a new task, whether to use cached policies or learn a new policy.

----

## [731] Causality-aware counterfactual confounding adjustment as an alternative to linear residualization in anticausal prediction tasks based on linear learners

**Authors**: *Elias Chaibub Neto*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/neto21a.html](http://proceedings.mlr.press/v139/neto21a.html)

**Abstract**:

Linear residualization is a common practice for confounding adjustment in machine learning applications. Recently, causality-aware predictive modeling has been proposed as an alternative causality-inspired approach for adjusting for confounders. In this paper, we compare the linear residualization approach against the causality-aware confounding adjustment in anticausal prediction tasks. Our comparisons include both the settings where the training and test sets come from the same distributions, as well as, when the training and test sets are shifted due to selection biases. In the absence of dataset shifts, we show that the causality-aware approach tends to (asymptotically) outperform the residualization adjustment in terms of predictive performance in linear learners. Importantly, our results still holds even when the true model generating the data is not linear. We illustrate our results in both regression and classification tasks. Furthermore, in the presence of dataset shifts in the joint distribution of the confounders and outcome variables, we show that the causality-aware approach is more stable than linear residualization.

----

## [732] Incentivizing Compliance with Algorithmic Instruments

**Authors**: *Dung Daniel T. Ngo, Logan Stapleton, Vasilis Syrgkanis, Steven Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ngo21a.html](http://proceedings.mlr.press/v139/ngo21a.html)

**Abstract**:

Randomized experiments can be susceptible to selection bias due to potential non-compliance by the participants. While much of the existing work has studied compliance as a static behavior, we propose a game-theoretic model to study compliance as dynamic behavior that may change over time. In rounds, a social planner interacts with a sequence of heterogeneous agents who arrive with their unobserved private type that determines both their prior preferences across the actions (e.g., control and treatment) and their baseline rewards without taking any treatment. The planner provides each agent with a randomized recommendation that may alter their beliefs and their action selection. We develop a novel recommendation mechanism that views the planner’s recommendation as a form of instrumental variable (IV) that only affects an agents’ action selection, but not the observed rewards. We construct such IVs by carefully mapping the history –the interactions between the planner and the previous agents– to a random recommendation. Even though the initial agents may be completely non-compliant, our mechanism can incentivize compliance over time, thereby enabling the estimation of the treatment effect of each treatment, and minimizing the cumulative regret of the planner whose goal is to identify the optimal treatment.

----

## [733] On the Proof of Global Convergence of Gradient Descent for Deep ReLU Networks with Linear Widths

**Authors**: *Quynh Nguyen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21a.html](http://proceedings.mlr.press/v139/nguyen21a.html)

**Abstract**:

We give a simple proof for the global convergence of gradient descent in training deep ReLU networks with the standard square loss, and show some of its improvements over the state-of-the-art. In particular, while prior works require all the hidden layers to be wide with width at least $\Omega(N^8)$ ($N$ being the number of training samples), we require a single wide layer of linear, quadratic or cubic width depending on the type of initialization. Unlike many recent proofs based on the Neural Tangent Kernel (NTK), our proof need not track the evolution of the entire NTK matrix, or more generally, any quantities related to the changes of activation patterns during training. Instead, we only need to track the evolution of the output at the last hidden layer, which can be done much more easily thanks to the Lipschitz property of ReLU. Some highlights of our setting: (i) all the layers are trained with standard gradient descent, (ii) the network has standard parameterization as opposed to the NTK one, and (iii) the network has a single wide layer as opposed to having all wide hidden layers as in most of NTK-related results.

----

## [734] Value-at-Risk Optimization with Gaussian Processes

**Authors**: *Quoc Phong Nguyen, Zhongxiang Dai, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21b.html](http://proceedings.mlr.press/v139/nguyen21b.html)

**Abstract**:

Value-at-risk (VaR) is an established measure to assess risks in critical real-world applications with random environmental factors. This paper presents a novel VaR upper confidence bound (V-UCB) algorithm for maximizing the VaR of a black-box objective function with the first no-regret guarantee. To realize this, we first derive a confidence bound of VaR and then prove the existence of values of the environmental random variable (to be selected to achieve no regret) such that the confidence bound of VaR lies within that of the objective function evaluated at such values. Our V-UCB algorithm empirically demonstrates state-of-the-art performance in optimizing synthetic benchmark functions, a portfolio optimization problem, and a simulated robot task.

----

## [735] Cross-model Back-translated Distillation for Unsupervised Machine Translation

**Authors**: *Xuan-Phi Nguyen, Shafiq R. Joty, Thanh-Tung Nguyen, Kui Wu, Ai Ti Aw*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21c.html](http://proceedings.mlr.press/v139/nguyen21c.html)

**Abstract**:

Recent unsupervised machine translation (UMT) systems usually employ three main principles: initialization, language modeling and iterative back-translation, though they may apply them differently. Crucially, iterative back-translation and denoising auto-encoding for language modeling provide data diversity to train the UMT systems. However, the gains from these diversification processes has seemed to plateau. We introduce a novel component to the standard UMT framework called Cross-model Back-translated Distillation (CBD), that is aimed to induce another level of data diversification that existing principles lack. CBD is applicable to all previous UMT approaches. In our experiments, CBD achieves the state of the art in the WMT’14 English-French, WMT’16 English-German and English-Romanian bilingual unsupervised translation tasks, with 38.2, 30.1, and 36.3 BLEU respectively. It also yields 1.5–3.3 BLEU improvements in IWSLT English-French and English-German tasks. Through extensive experimental analyses, we show that CBD is effective because it embraces data diversity while other similar variants do not.

----

## [736] Optimal Transport Kernels for Sequential and Parallel Neural Architecture Search

**Authors**: *Vu Nguyen, Tam Le, Makoto Yamada, Michael A. Osborne*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21d.html](http://proceedings.mlr.press/v139/nguyen21d.html)

**Abstract**:

Neural architecture search (NAS) automates the design of deep neural networks. One of the main challenges in searching complex and non-continuous architectures is to compare the similarity of networks that the conventional Euclidean metric may fail to capture. Optimal transport (OT) is resilient to such complex structure by considering the minimal cost for transporting a network into another. However, the OT is generally not negative definite which may limit its ability to build the positive-definite kernels required in many kernel-dependent frameworks. Building upon tree-Wasserstein (TW), which is a negative definite variant of OT, we develop a novel discrepancy for neural architectures, and demonstrate it within a Gaussian process surrogate model for the sequential NAS settings. Furthermore, we derive a novel parallel NAS, using quality k-determinantal point process on the GP posterior, to select diverse and high-performing architectures from a discrete set of candidates. Empirically, we demonstrate that our TW-based approaches outperform other baselines in both sequential and parallel NAS.

----

## [737] Interactive Learning from Activity Description

**Authors**: *Khanh Nguyen, Dipendra Misra, Robert E. Schapire, Miroslav Dudík, Patrick Shafto*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21e.html](http://proceedings.mlr.press/v139/nguyen21e.html)

**Abstract**:

We present a novel interactive learning protocol that enables training request-fulfilling agents by verbally describing their activities. Unlike imitation learning (IL), our protocol allows the teaching agent to provide feedback in a language that is most appropriate for them. Compared with reward in reinforcement learning (RL), the description feedback is richer and allows for improved sample complexity. We develop a probabilistic framework and an algorithm that practically implements our protocol. Empirical results in two challenging request-fulfilling problems demonstrate the strengths of our approach: compared with RL baselines, it is more sample-efficient; compared with IL baselines, it achieves competitive success rates without requiring the teaching agent to be able to demonstrate the desired behavior using the learning agent’s actions. Apart from empirical evaluation, we also provide theoretical guarantees for our algorithm under certain assumptions about the teacher and the environment.

----

## [738] Nonmyopic Multifidelity Acitve Search

**Authors**: *Quan Nguyen, Arghavan Modiri, Roman Garnett*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21f.html](http://proceedings.mlr.press/v139/nguyen21f.html)

**Abstract**:

Active search is a learning paradigm where we seek to identify as many members of a rare, valuable class as possible given a labeling budget. Previous work on active search has assumed access to a faithful (and expensive) oracle reporting experimental results. However, some settings offer access to cheaper surrogates such as computational simulation that may aid in the search. We propose a model of multifidelity active search, as well as a novel, computationally efficient policy for this setting that is motivated by state-of-the-art classical policies. Our policy is nonmyopic and budget aware, allowing for a dynamic tradeoff between exploration and exploitation. We evaluate the performance of our solution on real-world datasets and demonstrate significantly better performance than natural benchmarks.

----

## [739] Tight Bounds on the Smallest Eigenvalue of the Neural Tangent Kernel for Deep ReLU Networks

**Authors**: *Quynh Nguyen, Marco Mondelli, Guido F. Montúfar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21g.html](http://proceedings.mlr.press/v139/nguyen21g.html)

**Abstract**:

A recent line of work has analyzed the theoretical properties of deep neural networks via the Neural Tangent Kernel (NTK). In particular, the smallest eigenvalue of the NTK has been related to the memorization capacity, the global convergence of gradient descent algorithms and the generalization of deep nets. However, existing results either provide bounds in the two-layer setting or assume that the spectrum of the NTK matrices is bounded away from 0 for multi-layer networks. In this paper, we provide tight bounds on the smallest eigenvalue of NTK matrices for deep ReLU nets, both in the limiting case of infinite widths and for finite widths. In the finite-width setting, the network architectures we consider are fairly general: we require the existence of a wide layer with roughly order of $N$ neurons, $N$ being the number of data samples; and the scaling of the remaining layer widths is arbitrary (up to logarithmic factors). To obtain our results, we analyze various quantities of independent interest: we give lower bounds on the smallest singular value of hidden feature matrices, and upper bounds on the Lipschitz constant of input-output feature maps.

----

## [740] Temporal Predictive Coding For Model-Based Planning In Latent Space

**Authors**: *Tung D. Nguyen, Rui Shu, Tuan Pham, Hung Bui, Stefano Ermon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21h.html](http://proceedings.mlr.press/v139/nguyen21h.html)

**Abstract**:

High-dimensional observations are a major challenge in the application of model-based reinforcement learning (MBRL) to real-world environments. To handle high-dimensional sensory inputs, existing approaches use representation learning to map high-dimensional observations into a lower-dimensional latent space that is more amenable to dynamics estimation and planning. In this work, we present an information-theoretic approach that employs temporal predictive coding to encode elements in the environment that can be predicted across time. Since this approach focuses on encoding temporally-predictable information, we implicitly prioritize the encoding of task-relevant components over nuisance information within the environment that are provably task-irrelevant. By learning this representation in conjunction with a recurrent state space model, we can then perform planning in latent space. We evaluate our model on a challenging modification of standard DMControl tasks where the background is replaced with natural videos that contain complex but irrelevant information to the planning task. Our experiments show that our model is superior to existing methods in the challenging complex-background setting while remaining competitive with current state-of-the-art models in the standard setting.

----

## [741] Differentially Private Densest Subgraph Detection

**Authors**: *Dung Nguyen, Anil Vullikanti*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nguyen21i.html](http://proceedings.mlr.press/v139/nguyen21i.html)

**Abstract**:

Densest subgraph detection is a fundamental graph mining problem, with a large number of applications. There has been a lot of work on efficient algorithms for finding the densest subgraph in massive networks. However, in many domains, the network is private, and returning a densest subgraph can reveal information about the network. Differential privacy is a powerful framework to handle such settings. We study the densest subgraph problem in the edge privacy model, in which the edges of the graph are private. We present the first sequential and parallel differentially private algorithms for this problem. We show that our algorithms have an additive approximation guarantee. We evaluate our algorithms on a large number of real-world networks, and observe a good privacy-accuracy tradeoff when the network has high density.

----

## [742] Data Augmentation for Meta-Learning

**Authors**: *Renkun Ni, Micah Goldblum, Amr Sharaf, Kezhi Kong, Tom Goldstein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ni21a.html](http://proceedings.mlr.press/v139/ni21a.html)

**Abstract**:

Conventional image classifiers are trained by randomly sampling mini-batches of images. To achieve state-of-the-art performance, practitioners use sophisticated data augmentation schemes to expand the amount of training data available for sampling. In contrast, meta-learning algorithms sample support data, query data, and tasks on each training step. In this complex sampling scenario, data augmentation can be used not only to expand the number of images available per class, but also to generate entirely new classes/tasks. We systematically dissect the meta-learning pipeline and investigate the distinct ways in which data augmentation can be integrated at both the image and class levels. Our proposed meta-specific data augmentation significantly improves the performance of meta-learners on few-shot classification benchmarks.

----

## [743] Improved Denoising Diffusion Probabilistic Models

**Authors**: *Alexander Quinn Nichol, Prafulla Dhariwal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nichol21a.html](http://proceedings.mlr.press/v139/nichol21a.html)

**Abstract**:

Denoising diffusion probabilistic models (DDPM) are a class of generative models which have recently been shown to produce excellent samples. We show that with a few simple modifications, DDPMs can also achieve competitive log-likelihoods while maintaining high sample quality. Additionally, we find that learning variances of the reverse diffusion process allows sampling with an order of magnitude fewer forward passes with a negligible difference in sample quality, which is important for the practical deployment of these models. We additionally use precision and recall to compare how well DDPMs and GANs cover the target distribution. Finally, we show that the sample quality and likelihood of these models scale smoothly with model capacity and training compute, making them easily scalable. We release our code and pre-trained models at https://github.com/openai/improved-diffusion.

----

## [744] Smooth p-Wasserstein Distance: Structure, Empirical Approximation, and Statistical Applications

**Authors**: *Sloan Nietert, Ziv Goldfeld, Kengo Kato*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nietert21a.html](http://proceedings.mlr.press/v139/nietert21a.html)

**Abstract**:

Discrepancy measures between probability distributions, often termed statistical distances, are ubiquitous in probability theory, statistics and machine learning. To combat the curse of dimensionality when estimating these distances from data, recent work has proposed smoothing out local irregularities in the measured distributions via convolution with a Gaussian kernel. Motivated by the scalability of this framework to high dimensions, we investigate the structural and statistical behavior of the Gaussian-smoothed $p$-Wasserstein distance $\mathsf{W}_p^{(\sigma)}$, for arbitrary $p\geq 1$. After establishing basic metric and topological properties of $\mathsf{W}_p^{(\sigma)}$, we explore the asymptotic statistical properties of $\mathsf{W}_p^{(\sigma)}(\hat{\mu}_n,\mu)$, where $\hat{\mu}_n$ is the empirical distribution of $n$ independent observations from $\mu$. We prove that $\mathsf{W}_p^{(\sigma)}$ enjoys a parametric empirical convergence rate of $n^{-1/2}$, which contrasts the $n^{-1/d}$ rate for unsmoothed $\Wp$ when $d \geq 3$. Our proof relies on controlling $\mathsf{W}_p^{(\sigma)}$ by a $p$th-order smooth Sobolev distance $\mathsf{d}_p^{(\sigma)}$ and deriving the limit distribution of $\sqrt{n}\,\mathsf{d}_p^{(\sigma)}(\hat{\mu}_n,\mu)$ for all dimensions $d$. As applications, we provide asymptotic guarantees for two-sample testing and minimum distance estimation using $\mathsf{W}_p^{(\sigma)}$, with experiments for $p=2$ using a maximum mean discrepancy formulation of $\mathsf{d}_2^{(\sigma)}$.

----

## [745] AdaXpert: Adapting Neural Architecture for Growing Data

**Authors**: *Shuaicheng Niu, Jiaxiang Wu, Guanghui Xu, Yifan Zhang, Yong Guo, Peilin Zhao, Peng Wang, Mingkui Tan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/niu21a.html](http://proceedings.mlr.press/v139/niu21a.html)

**Abstract**:

In real-world applications, data often come in a growing manner, where the data volume and the number of classes may increase dynamically. This will bring a critical challenge for learning: given the increasing data volume or the number of classes, one has to instantaneously adjust the neural model capacity to obtain promising performance. Existing methods either ignore the growing nature of data or seek to independently search an optimal architecture for a given dataset, and thus are incapable of promptly adjusting the architectures for the changed data. To address this, we present a neural architecture adaptation method, namely Adaptation eXpert (AdaXpert), to efficiently adjust previous architectures on the growing data. Specifically, we introduce an architecture adjuster to generate a suitable architecture for each data snapshot, based on the previous architecture and the different extent between current and previous data distributions. Furthermore, we propose an adaptation condition to determine the necessity of adjustment, thereby avoiding unnecessary and time-consuming adjustments. Extensive experiments on two growth scenarios (increasing data volume and number of classes) demonstrate the effectiveness of the proposed method.

----

## [746] Asynchronous Decentralized Optimization With Implicit Stochastic Variance Reduction

**Authors**: *Kenta Niwa, Guoqiang Zhang, W. Bastiaan Kleijn, Noboru Harada, Hiroshi Sawada, Akinori Fujino*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/niwa21a.html](http://proceedings.mlr.press/v139/niwa21a.html)

**Abstract**:

A novel asynchronous decentralized optimization method that follows Stochastic Variance Reduction (SVR) is proposed. Average consensus algorithms, such as Decentralized Stochastic Gradient Descent (DSGD), facilitate distributed training of machine learning models. However, the gradient will drift within the local nodes due to statistical heterogeneity of the subsets of data residing on the nodes and long communication intervals. To overcome the drift problem, (i) Gradient Tracking-SVR (GT-SVR) integrates SVR into DSGD and (ii) Edge-Consensus Learning (ECL) solves a model constrained minimization problem using a primal-dual formalism. In this paper, we reformulate the update procedure of ECL such that it implicitly includes the gradient modification of SVR by optimally selecting a constraint-strength control parameter. Through convergence analysis and experiments, we confirmed that the proposed ECL with Implicit SVR (ECL-ISVR) is stable and approximately reaches the reference performance obtained with computation on a single-node using full data set.

----

## [747] WGAN with an Infinitely Wide Generator Has No Spurious Stationary Points

**Authors**: *Albert No, Taeho Yoon, Sehyun Kwon, Ernest K. Ryu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/no21a.html](http://proceedings.mlr.press/v139/no21a.html)

**Abstract**:

Generative adversarial networks (GAN) are a widely used class of deep generative models, but their minimax training dynamics are not understood very well. In this work, we show that GANs with a 2-layer infinite-width generator and a 2-layer finite-width discriminator trained with stochastic gradient ascent-descent have no spurious stationary points. We then show that when the width of the generator is finite but wide, there are no spurious stationary points within a ball whose radius becomes arbitrarily large (to cover the entire parameter space) as the width goes to infinity.

----

## [748] The Impact of Record Linkage on Learning from Feature Partitioned Data

**Authors**: *Richard Nock, Stephen Hardy, Wilko Henecka, Hamish Ivey-Law, Jakub Nabaglo, Giorgio Patrini, Guillaume Smith, Brian Thorne*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nock21a.html](http://proceedings.mlr.press/v139/nock21a.html)

**Abstract**:

There has been recently a significant boost to machine learning with distributed data, in particular with the success of federated learning. A common and very challenging setting is that of vertical or feature partitioned data, when multiple data providers hold different features about common entities. In general, training needs to be preceded by record linkage (RL), a step that finds the correspondence between the observations of the datasets. RL is prone to mistakes in the real world. Despite the importance of the problem, there has been so far no formal assessment of the way in which RL errors impact learning models. Work in the area either use heuristics or assume that the optimal RL is known in advance. In this paper, we provide the first assessment of the problem for supervised learning. For wide sets of losses, we provide technical conditions under which the classifier learned after noisy RL converges (with the data size) to the best classifier that would be learned from mistake-free RL. This yields new insights on the way the pipeline RL + ML operates, from the role of large margin classification on dampening the impact of RL mistakes to clues on how to further optimize RL as a preprocessing step to ML. Experiments on a large UCI benchmark validate those formal observations.

----

## [749] Accuracy, Interpretability, and Differential Privacy via Explainable Boosting

**Authors**: *Harsha Nori, Rich Caruana, Zhiqi Bu, Judy Hanwen Shen, Janardhan Kulkarni*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nori21a.html](http://proceedings.mlr.press/v139/nori21a.html)

**Abstract**:

We show that adding differential privacy to Explainable Boosting Machines (EBMs), a recent method for training interpretable ML models, yields state-of-the-art accuracy while protecting privacy. Our experiments on multiple classification and regression datasets show that DP-EBM models suffer surprisingly little accuracy loss even with strong differential privacy guarantees. In addition to high accuracy, two other benefits of applying DP to EBMs are: a) trained models provide exact global and local interpretability, which is often important in settings where differential privacy is needed; and b) the models can be edited after training without loss of privacy to correct errors which DP noise may have introduced.

----

## [750] Posterior Value Functions: Hindsight Baselines for Policy Gradient Methods

**Authors**: *Chris Nota, Philip S. Thomas, Bruno C. da Silva*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/nota21a.html](http://proceedings.mlr.press/v139/nota21a.html)

**Abstract**:

Hindsight allows reinforcement learning agents to leverage new observations to make inferences about earlier states and transitions. In this paper, we exploit the idea of hindsight and introduce posterior value functions. Posterior value functions are computed by inferring the posterior distribution over hidden components of the state in previous timesteps and can be used to construct novel unbiased baselines for policy gradient methods. Importantly, we prove that these baselines reduce (and never increase) the variance of policy gradient estimators compared to traditional state value functions. While the posterior value function is motivated by partial observability, we extend these results to arbitrary stochastic MDPs by showing that hindsight-capable agents can model stochasticity in the environment as a special case of partial observability. Finally, we introduce a pair of methods for learning posterior value functions and prove their convergence.

----

## [751] Global inducing point variational posteriors for Bayesian neural networks and deep Gaussian processes

**Authors**: *Sebastian W. Ober, Laurence Aitchison*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ober21a.html](http://proceedings.mlr.press/v139/ober21a.html)

**Abstract**:

We consider the optimal approximate posterior over the top-layer weights in a Bayesian neural network for regression, and show that it exhibits strong dependencies on the lower-layer weights. We adapt this result to develop a correlated approximate posterior over the weights at all layers in a Bayesian neural network. We extend this approach to deep Gaussian processes, unifying inference in the two model classes. Our approximate posterior uses learned "global” inducing points, which are defined only at the input layer and propagated through the network to obtain inducing inputs at subsequent layers. By contrast, standard, "local”, inducing point methods from the deep Gaussian process literature optimise a separate set of inducing inputs at every layer, and thus do not model correlations across layers. Our method gives state-of-the-art performance for a variational Bayesian method, without data augmentation or tempering, on CIFAR-10 of 86.7%, which is comparable to SGMCMC without tempering but with data augmentation (88% in Wenzel et al. 2020).

----

## [752] Regularizing towards Causal Invariance: Linear Models with Proxies

**Authors**: *Michael Oberst, Nikolaj Thams, Jonas Peters, David A. Sontag*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/oberst21a.html](http://proceedings.mlr.press/v139/oberst21a.html)

**Abstract**:

We propose a method for learning linear models whose predictive performance is robust to causal interventions on unobserved variables, when noisy proxies of those variables are available. Our approach takes the form of a regularization term that trades off between in-distribution performance and robustness to interventions. Under the assumption of a linear structural causal model, we show that a single proxy can be used to create estimators that are prediction optimal under interventions of bounded strength. This strength depends on the magnitude of the measurement noise in the proxy, which is, in general, not identifiable. In the case of two proxy variables, we propose a modified estimator that is prediction optimal under interventions up to a known strength. We further show how to extend these estimators to scenarios where additional information about the "test time" intervention is available during training. We evaluate our theoretical findings in synthetic experiments and using real data of hourly pollution levels across several cities in China.

----

## [753] Sparsity-Agnostic Lasso Bandit

**Authors**: *Min-hwan Oh, Garud Iyengar, Assaf Zeevi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/oh21a.html](http://proceedings.mlr.press/v139/oh21a.html)

**Abstract**:

We consider a stochastic contextual bandit problem where the dimension $d$ of the feature vectors is potentially large, however, only a sparse subset of features of cardinality $s_0 \ll d$ affect the reward function. Essentially all existing algorithms for sparse bandits require a priori knowledge of the value of the sparsity index $s_0$. This knowledge is almost never available in practice, and misspecification of this parameter can lead to severe deterioration in the performance of existing methods. The main contribution of this paper is to propose an algorithm that does not require prior knowledge of the sparsity index $s_0$ and establish tight regret bounds on its performance under mild conditions. We also comprehensively evaluate our proposed algorithm numerically and show that it consistently outperforms existing methods, even when the correct sparsity index is revealed to them but is kept hidden from our algorithm.

----

## [754] Autoencoder Image Interpolation by Shaping the Latent Space

**Authors**: *Alon Oring, Zohar Yakhini, Yacov Hel-Or*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/oring21a.html](http://proceedings.mlr.press/v139/oring21a.html)

**Abstract**:

One of the fascinating properties of deep learning is the ability of the network to reveal the underlying factors characterizing elements in datasets of different types. Autoencoders represent an effective approach for computing these factors. Autoencoders have been studied in the context of enabling interpolation between data points by decoding convex combinations of latent vectors. However, this interpolation often leads to artifacts or produces unrealistic results during reconstruction. We argue that these incongruities are due to the structure of the latent space and to the fact that such naively interpolated latent vectors deviate from the data manifold. In this paper, we propose a regularization technique that shapes the latent representation to follow a manifold that is consistent with the training images and that forces the manifold to be smooth and locally convex. This regularization not only enables faithful interpolation between data points, as we show herein but can also be used as a general regularization technique to avoid overfitting or to produce new samples for data augmentation.

----

## [755] Generalization Guarantees for Neural Architecture Search with Train-Validation Split

**Authors**: *Samet Oymak, Mingchen Li, Mahdi Soltanolkotabi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/oymak21a.html](http://proceedings.mlr.press/v139/oymak21a.html)

**Abstract**:

Neural Architecture Search (NAS) is a popular method for automatically designing optimized deep-learning architectures. NAS methods commonly use bilevel optimization where one optimizes the weights over the training data (lower-level problem) and hyperparameters - such as the architecture - over the validation data (upper-level problem). This paper explores the statistical aspects of such problems with train-validation splits. In practice, the lower-level problem is often overparameterized and can easily achieve zero loss. Thus, a-priori, it seems impossible to distinguish the right hyperparameters based on training loss alone which motivates a better understanding of train-validation split. To this aim, we first show that refined properties of the validation loss such as risk and hyper-gradients are indicative of those of the true test loss and help prevent overfitting with a near-minimal validation sample size. Importantly, this is established for continuous search spaces which are relevant for differentiable search schemes. We then establish generalization bounds for NAS problems with an emphasis on an activation search problem and gradient-based methods. Finally, we show rigorous connections between NAS and low-rank matrix learning which leads to algorithmic insights where the solution of the upper problem can be accurately learned via spectral methods to achieve near-minimal risk.

----

## [756] Vector Quantized Models for Planning

**Authors**: *Sherjil Ozair, Yazhe Li, Ali Razavi, Ioannis Antonoglou, Aäron van den Oord, Oriol Vinyals*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ozair21a.html](http://proceedings.mlr.press/v139/ozair21a.html)

**Abstract**:

Recent developments in the field of model-based RL have proven successful in a range of environments, especially ones where planning is essential. However, such successes have been limited to deterministic fully-observed environments. We present a new approach that handles stochastic and partially-observable environments. Our key insight is to use discrete autoencoders to capture the multiple possible effects of an action in a stochastic environment. We use a stochastic variant of Monte Carlo tree search to plan over both the agent’s actions and the discrete latent variables representing the environment’s response. Our approach significantly outperforms an offline version of MuZero on a stochastic interpretation of chess where the opponent is considered part of the environment. We also show that our approach scales to DeepMind Lab, a first-person 3D environment with large visual observations and partial observability.

----

## [757] Training Adversarially Robust Sparse Networks via Bayesian Connectivity Sampling

**Authors**: *Ozan Özdenizci, Robert Legenstein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ozdenizci21a.html](http://proceedings.mlr.press/v139/ozdenizci21a.html)

**Abstract**:

Deep neural networks have been shown to be susceptible to adversarial attacks. This lack of adversarial robustness is even more pronounced when models are compressed in order to meet hardware limitations. Hence, if adversarial robustness is an issue, training of sparsely connected networks necessitates considering adversarially robust sparse learning. Motivated by the efficient and stable computational function of the brain in the presence of a highly dynamic synaptic connectivity structure, we propose an intrinsically sparse rewiring approach to train neural networks with state-of-the-art robust learning objectives under high sparsity. Importantly, in contrast to previously proposed pruning techniques, our approach satisfies global connectivity constraints throughout robust optimization, i.e., it does not require dense pre-training followed by pruning. Based on a Bayesian posterior sampling principle, a network rewiring process simultaneously learns the sparse connectivity structure and the robustness-accuracy trade-off based on the adversarial learning objective. Although our networks are sparsely connected throughout the whole training process, our experimental benchmark evaluations show that their performance is superior to recently proposed robustness-aware network pruning methods which start from densely connected networks.

----

## [758] Opening the Blackbox: Accelerating Neural Differential Equations by Regularizing Internal Solver Heuristics

**Authors**: *Avik Pal, Yingbo Ma, Viral B. Shah, Christopher Vincent Rackauckas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/pal21a.html](http://proceedings.mlr.press/v139/pal21a.html)

**Abstract**:

Democratization of machine learning requires architectures that automatically adapt to new problems. Neural Differential Equations (NDEs) have emerged as a popular modeling framework by removing the need for ML practitioners to choose the number of layers in a recurrent model. While we can control the computational cost by choosing the number of layers in standard architectures, in NDEs the number of neural network evaluations for a forward pass can depend on the number of steps of the adaptive ODE solver. But, can we force the NDE to learn the version with the least steps while not increasing the training cost? Current strategies to overcome slow prediction require high order automatic differentiation, leading to significantly higher training time. We describe a novel regularization method that uses the internal cost heuristics of adaptive differential equation solvers combined with discrete adjoint sensitivities to guide the training process towards learning NDEs that are easier to solve. This approach opens up the blackbox numerical analysis behind the differential equation solver’s algorithm and directly uses its local error estimates and stiffness heuristics as cheap and accurate cost estimates. We incorporate our method without any change in the underlying NDE framework and show that our method extends beyond Ordinary Differential Equations to accommodate Neural Stochastic Differential Equations. We demonstrate how our approach can halve the prediction time and, unlike other methods which can increase the training time by an order of magnitude, we demonstrate similar reduction in training times. Together this showcases how the knowledge embedded within state-of-the-art equation solvers can be used to enhance machine learning.

----

## [759] RNN with Particle Flow for Probabilistic Spatio-temporal Forecasting

**Authors**: *Soumyasundar Pal, Liheng Ma, Yingxue Zhang, Mark Coates*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/pal21b.html](http://proceedings.mlr.press/v139/pal21b.html)

**Abstract**:

Spatio-temporal forecasting has numerous applications in analyzing wireless, traffic, and financial networks. Many classical statistical models often fall short in handling the complexity and high non-linearity present in time-series data. Recent advances in deep learning allow for better modelling of spatial and temporal dependencies. While most of these models focus on obtaining accurate point forecasts, they do not characterize the prediction uncertainty. In this work, we consider the time-series data as a random realization from a nonlinear state-space model and target Bayesian inference of the hidden states for probabilistic forecasting. We use particle flow as the tool for approximating the posterior distribution of the states, as it is shown to be highly effective in complex, high-dimensional settings. Thorough experimentation on several real world time-series datasets demonstrates that our approach provides better characterization of uncertainty while maintaining comparable accuracy to the state-of-the-art point forecasting methods.

----

## [760] Inference for Network Regression Models with Community Structure

**Authors**: *Mengjie Pan, Tyler H. McCormick, Bailey K. Fosdick*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/pan21a.html](http://proceedings.mlr.press/v139/pan21a.html)

**Abstract**:

Network regression models, where the outcome comprises the valued edge in a network and the predictors are actor or dyad-level covariates, are used extensively in the social and biological sciences. Valid inference relies on accurately modeling the residual dependencies among the relations. Frequently homogeneity assumptions are placed on the errors which are commonly incorrect and ignore critical natural clustering of the actors. In this work, we present a novel regression modeling framework that models the errors as resulting from a community-based dependence structure and exploits the subsequent exchangeability properties of the error distribution to obtain parsimonious standard errors for regression parameters.

----

## [761] Latent Space Energy-Based Model of Symbol-Vector Coupling for Text Generation and Classification

**Authors**: *Bo Pang, Ying Nian Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/pang21a.html](http://proceedings.mlr.press/v139/pang21a.html)

**Abstract**:

We propose a latent space energy-based prior model for text generation and classification. The model stands on a generator network that generates the text sequence based on a continuous latent vector. The energy term of the prior model couples a continuous latent vector and a symbolic one-hot vector, so that discrete category can be inferred from the observed example based on the continuous latent vector. Such a latent space coupling naturally enables incorporation of information bottleneck regularization to encourage the continuous latent vector to extract information from the observed example that is informative of the underlying category. In our learning method, the symbol-vector coupling, the generator network and the inference network are learned jointly. Our model can be learned in an unsupervised setting where no category labels are provided. It can also be learned in semi-supervised setting where category labels are provided for a subset of training examples. Our experiments demonstrate that the proposed model learns well-structured and meaningful latent space, which (1) guides the generator to generate text with high quality, diversity, and interpretability, and (2) effectively classifies text.

----

## [762] Leveraging Good Representations in Linear Contextual Bandits

**Authors**: *Matteo Papini, Andrea Tirinzoni, Marcello Restelli, Alessandro Lazaric, Matteo Pirotta*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/papini21a.html](http://proceedings.mlr.press/v139/papini21a.html)

**Abstract**:

The linear contextual bandit literature is mostly focused on the design of efficient learning algorithms for a given representation. However, a contextual bandit problem may admit multiple linear representations, each one with different characteristics that directly impact the regret of the learning algorithm. In particular, recent works showed that there exist “good” representations for which constant problem-dependent regret can be achieved. In this paper, we first provide a systematic analysis of the different definitions of “good” representations proposed in the literature. We then propose a novel selection algorithm able to adapt to the best representation in a set of $M$ candidates. We show that the regret is indeed never worse than the regret obtained by running \textsc{LinUCB} on best representation (up to a $\ln M$ factor). As a result, our algorithm achieves constant regret if a “good” representation is available in the set. Furthermore, we show the algorithm may still achieve constant regret by implicitly constructing a “good” representation, even when none of the initial representations is “good”. Finally, we validate our theoretical findings in a number of standard contextual bandit problems.

----

## [763] Wasserstein Distributional Normalization For Robust Distributional Certification of Noisy Labeled Data

**Authors**: *Sung Woo Park, Junseok Kwon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/park21a.html](http://proceedings.mlr.press/v139/park21a.html)

**Abstract**:

We propose a novel Wasserstein distributional normalization method that can classify noisy labeled data accurately. Recently, noisy labels have been successfully handled based on small-loss criteria, but have not been clearly understood from the theoretical point of view. In this paper, we address this problem by adopting distributionally robust optimization (DRO). In particular, we present a theoretical investigation of the distributional relationship between uncertain and certain samples based on the small-loss criteria. Our method takes advantage of this relationship to exploit useful information from uncertain samples. To this end, we normalize uncertain samples into the robustly certified region by introducing the non-parametric Ornstein-Ulenbeck type of Wasserstein gradient flows called Wasserstein distributional normalization, which is cheap and fast to implement. We verify that network confidence and distributional certification are fundamentally correlated and show the concentration inequality when the network escapes from over-parameterization. Experimental results demonstrate that our non-parametric classification method outperforms other parametric baselines on the Clothing1M and CIFAR-10/100 datasets when the data have diverse noisy labels.

----

## [764] Unsupervised Representation Learning via Neural Activation Coding

**Authors**: *Yookoon Park, Sangho Lee, Gunhee Kim, David M. Blei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/park21b.html](http://proceedings.mlr.press/v139/park21b.html)

**Abstract**:

We present neural activation coding (NAC) as a novel approach for learning deep representations from unlabeled data for downstream applications. We argue that the deep encoder should maximize its nonlinear expressivity on the data for downstream predictors to take full advantage of its representation power. To this end, NAC maximizes the mutual information between activation patterns of the encoder and the data over a noisy communication channel. We show that learning for a noise-robust activation code increases the number of distinct linear regions of ReLU encoders, hence the maximum nonlinear expressivity. More interestingly, NAC learns both continuous and discrete representations of data, which we respectively evaluate on two downstream tasks: (i) linear classification on CIFAR-10 and ImageNet-1K and (ii) nearest neighbor retrieval on CIFAR-10 and FLICKR-25K. Empirical results show that NAC attains better or comparable performance on both tasks over recent baselines including SimCLR and DistillHash. In addition, NAC pretraining provides significant benefits to the training of deep generative models. Our code is available at https://github.com/yookoon/nac.

----

## [765] Conditional Distributional Treatment Effect with Kernel Conditional Mean Embeddings and U-Statistic Regression

**Authors**: *Junhyung Park, Uri Shalit, Bernhard Schölkopf, Krikamol Muandet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/park21c.html](http://proceedings.mlr.press/v139/park21c.html)

**Abstract**:

We propose to analyse the conditional distributional treatment effect (CoDiTE), which, in contrast to the more common conditional average treatment effect (CATE), is designed to encode a treatment’s distributional aspects beyond the mean. We first introduce a formal definition of the CoDiTE associated with a distance function between probability measures. Then we discuss the CoDiTE associated with the maximum mean discrepancy via kernel conditional mean embeddings, which, coupled with a hypothesis test, tells us whether there is any conditional distributional effect of the treatment. Finally, we investigate what kind of conditional distributional effect the treatment has, both in an exploratory manner via the conditional witness function, and in a quantitative manner via U-statistic regression, generalising the CATE to higher-order moments. Experiments on synthetic, semi-synthetic and real datasets demonstrate the merits of our approach.

----

## [766] Generative Adversarial Networks for Markovian Temporal Dynamics: Stochastic Continuous Data Generation

**Authors**: *Sung Woo Park, Dong Wook Shu, Junseok Kwon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/park21d.html](http://proceedings.mlr.press/v139/park21d.html)

**Abstract**:

In this paper, we present a novel generative adversarial network (GAN) that can describe Markovian temporal dynamics. To generate stochastic sequential data, we introduce a novel stochastic differential equation-based conditional generator and spatial-temporal constrained discriminator networks. To stabilize the learning dynamics of the min-max type of the GAN objective function, we propose well-posed constraint terms for both networks. We also propose a novel conditional Markov Wasserstein distance to induce a pathwise Wasserstein distance. The experimental results demonstrate that our method outperforms state-of-the-art methods using several different types of data.

----

## [767] Optimal Counterfactual Explanations in Tree Ensembles

**Authors**: *Axel Parmentier, Thibaut Vidal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/parmentier21a.html](http://proceedings.mlr.press/v139/parmentier21a.html)

**Abstract**:

Counterfactual explanations are usually generated through heuristics that are sensitive to the search’s initial conditions. The absence of guarantees of performance and robustness hinders trustworthiness. In this paper, we take a disciplined approach towards counterfactual explanations for tree ensembles. We advocate for a model-based search aiming at "optimal" explanations and propose efficient mixed-integer programming approaches. We show that isolation forests can be modeled within our framework to focus the search on plausible explanations with a low outlier score. We provide comprehensive coverage of additional constraints that model important objectives, heterogeneous data types, structural constraints on the feature space, along with resource and actionability restrictions. Our experimental analyses demonstrate that the proposed search approach requires a computational effort that is orders of magnitude smaller than previous mathematical programming algorithms. It scales up to large data sets and tree ensembles, where it provides, within seconds, systematic explanations grounded on well-defined models solved to optimality.

----

## [768] PHEW : Constructing Sparse Networks that Learn Fast and Generalize Well without Training Data

**Authors**: *Shreyas Malakarjun Patil, Constantine Dovrolis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/patil21a.html](http://proceedings.mlr.press/v139/patil21a.html)

**Abstract**:

Methods that sparsify a network at initialization are important in practice because they greatly improve the efficiency of both learning and inference. Our work is based on a recently proposed decomposition of the Neural Tangent Kernel (NTK) that has decoupled the dynamics of the training process into a data-dependent component and an architecture-dependent kernel {–} the latter referred to as Path Kernel. That work has shown how to design sparse neural networks for faster convergence, without any training data, using the Synflow-L2 algorithm. We first show that even though Synflow-L2 is optimal in terms of convergence, for a given network density, it results in sub-networks with “bottleneck” (narrow) layers {–} leading to poor performance as compared to other data-agnostic methods that use the same number of parameters. Then we propose a new method to construct sparse networks, without any training data, referred to as Paths with Higher-Edge Weights (PHEW). PHEW is a probabilistic network formation method based on biased random walks that only depends on the initial weights. It has similar path kernel properties as Synflow-L2 but it generates much wider layers, resulting in better generalization and performance. PHEW achieves significant improvements over the data-independent SynFlow and SynFlow-L2 methods at a wide range of network densities.

----

## [769] CombOptNet: Fit the Right NP-Hard Problem by Learning Integer Programming Constraints

**Authors**: *Anselm Paulus, Michal Rolínek, Vít Musil, Brandon Amos, Georg Martius*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/paulus21a.html](http://proceedings.mlr.press/v139/paulus21a.html)

**Abstract**:

Bridging logical and algorithmic reasoning with modern machine learning techniques is a fundamental challenge with potentially transformative impact. On the algorithmic side, many NP-hard problems can be expressed as integer programs, in which the constraints play the role of their ’combinatorial specification’. In this work, we aim to integrate integer programming solvers into neural network architectures as layers capable of learning both the cost terms and the constraints. The resulting end-to-end trainable architectures jointly extract features from raw data and solve a suitable (learned) combinatorial problem with state-of-the-art integer programming solvers. We demonstrate the potential of such layers with an extensive performance analysis on synthetic data and with a demonstration on a competitive computer vision keypoint matching benchmark.

----

## [770] Ensemble Bootstrapping for Q-Learning

**Authors**: *Oren Peer, Chen Tessler, Nadav Merlis, Ron Meir*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/peer21a.html](http://proceedings.mlr.press/v139/peer21a.html)

**Abstract**:

Q-learning (QL), a common reinforcement learning algorithm, suffers from over-estimation bias due to the maximization term in the optimal Bellman operator. This bias may lead to sub-optimal behavior. Double-Q-learning tackles this issue by utilizing two estimators, yet results in an under-estimation bias. Similar to over-estimation in Q-learning, in certain scenarios, the under-estimation bias may degrade performance. In this work, we introduce a new bias-reduced algorithm called Ensemble Bootstrapped Q-Learning (EBQL), a natural extension of Double-Q-learning to ensembles. We analyze our method both theoretically and empirically. Theoretically, we prove that EBQL-like updates yield lower MSE when estimating the maximal mean of a set of independent random variables. Empirically, we show that there exist domains where both over and under-estimation result in sub-optimal performance. Finally, We demonstrate the superior performance of a deep RL variant of EBQL over other deep QL algorithms for a suite of ATARI games.

----

## [771] Homomorphic Sensing: Sparsity and Noise

**Authors**: *Liangzu Peng, Boshi Wang, Manolis C. Tsakiris*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/peng21a.html](http://proceedings.mlr.press/v139/peng21a.html)

**Abstract**:

\emph{Unlabeled sensing} is a recent problem encompassing many data science and engineering applications and typically formulated as solving linear equations whose right-hand side vector has undergone an unknown permutation. It was generalized to the \emph{homomorphic sensing} problem by replacing the unknown permutation with an unknown linear map from a given finite set of linear maps. In this paper we present tighter and simpler conditions for the homomorphic sensing problem to admit a unique solution. We show that this solution is locally stable under noise, while under a sparsity assumption it remains unique under less demanding conditions. Sparsity in the context of unlabeled sensing leads to the problem of \textit{unlabeled compressed sensing}, and a consequence of our general theory is the existence under mild conditions of a unique sparsest solution. On the algorithmic level, we solve unlabeled compressed sensing by an iterative algorithm validated by synthetic data experiments. Finally, under the unifying homomorphic sensing framework we connect unlabeled sensing to other important practical problems.

----

## [772] How could Neural Networks understand Programs?

**Authors**: *Dinglan Peng, Shuxin Zheng, Yatao Li, Guolin Ke, Di He, Tie-Yan Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/peng21b.html](http://proceedings.mlr.press/v139/peng21b.html)

**Abstract**:

Semantic understanding of programs is a fundamental problem for programming language processing (PLP). Recent works that learn representations of code based on pre-training techniques in NLP have pushed the frontiers in this direction. However, the semantics of PL and NL have essential differences. These being ignored, we believe it is difficult to build a model to better understand programs, by either directly applying off-the-shelf NLP pre-training techniques to the source code, or adding features to the model by the heuristic. In fact, the semantics of a program can be rigorously defined by formal semantics in PL theory. For example, the operational semantics, describes the meaning of a valid program as updating the environment (i.e., the memory address-value function) through fundamental operations, such as memory I/O and conditional branching. Inspired by this, we propose a novel program semantics learning paradigm, that the model should learn from information composed of (1) the representations which align well with the fundamental operations in operational semantics, and (2) the information of environment transition, which is indispensable for program understanding. To validate our proposal, we present a hierarchical Transformer-based pre-training model called OSCAR to better facilitate the understanding of programs. OSCAR learns from intermediate representation (IR) and an encoded representation derived from static analysis, which are used for representing the fundamental operations and approximating the environment transitions respectively. OSCAR empirically shows the outstanding capability of program semantics understanding on many practical software engineering tasks. Code and models are released at: \url{https://github.com/pdlan/OSCAR}.

----

## [773] Privacy-Preserving Video Classification with Convolutional Neural Networks

**Authors**: *Sikha Pentyala, Rafael Dowsley, Martine De Cock*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/pentyala21a.html](http://proceedings.mlr.press/v139/pentyala21a.html)

**Abstract**:

Many video classification applications require access to personal data, thereby posing an invasive security risk to the users’ privacy. We propose a privacy-preserving implementation of single-frame method based video classification with convolutional neural networks that allows a party to infer a label from a video without necessitating the video owner to disclose their video to other entities in an unencrypted manner. Similarly, our approach removes the requirement of the classifier owner from revealing their model parameters to outside entities in plaintext. To this end, we combine existing Secure Multi-Party Computation (MPC) protocols for private image classification with our novel MPC protocols for oblivious single-frame selection and secure label aggregation across frames. The result is an end-to-end privacy-preserving video classification pipeline. We evaluate our proposed solution in an application for private human emotion recognition. Our results across a variety of security settings, spanning honest and dishonest majority configurations of the computing parties, and for both passive and active adversaries, demonstrate that videos can be classified with state-of-the-art accuracy, and without leaking sensitive user information.

----

## [774] Rissanen Data Analysis: Examining Dataset Characteristics via Description Length

**Authors**: *Ethan Perez, Douwe Kiela, Kyunghyun Cho*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/perez21a.html](http://proceedings.mlr.press/v139/perez21a.html)

**Abstract**:

We introduce a method to determine if a certain capability helps to achieve an accurate model of given data. We view labels as being generated from the inputs by a program composed of subroutines with different capabilities, and we posit that a subroutine is useful if and only if the minimal program that invokes it is shorter than the one that does not. Since minimum program length is uncomputable, we instead estimate the labels’ minimum description length (MDL) as a proxy, giving us a theoretically-grounded method for analyzing dataset characteristics. We call the method Rissanen Data Analysis (RDA) after the father of MDL, and we showcase its applicability on a wide variety of settings in NLP, ranging from evaluating the utility of generating subquestions before answering a question, to analyzing the value of rationales and explanations, to investigating the importance of different parts of speech, and uncovering dataset gender bias.

----

## [775] Modelling Behavioural Diversity for Learning in Open-Ended Games

**Authors**: *Nicolas Perez Nieves, Yaodong Yang, Oliver Slumbers, David Henry Mguni, Ying Wen, Jun Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/perez-nieves21a.html](http://proceedings.mlr.press/v139/perez-nieves21a.html)

**Abstract**:

Promoting behavioural diversity is critical for solving games with non-transitive dynamics where strategic cycles exist, and there is no consistent winner (e.g., Rock-Paper-Scissors). Yet, there is a lack of rigorous treatment for defining diversity and constructing diversity-aware learning dynamics. In this work, we offer a geometric interpretation of behavioural diversity in games and introduce a novel diversity metric based on \emph{determinantal point processes} (DPP). By incorporating the diversity metric into best-response dynamics, we develop \emph{diverse fictitious play} and \emph{diverse policy-space response oracle} for solving normal-form games and open-ended games. We prove the uniqueness of the diverse best response and the convergence of our algorithms on two-player games. Importantly, we show that maximising the DPP-based diversity metric guarantees to enlarge the \emph{gamescape} – convex polytopes spanned by agents’ mixtures of strategies. To validate our diversity-aware solvers, we test on tens of games that show strong non-transitivity. Results suggest that our methods achieve at least the same, and in most games, lower exploitability than PSRO solvers by finding effective and diverse strategies.

----

## [776] From Poincaré Recurrence to Convergence in Imperfect Information Games: Finding Equilibrium via Regularization

**Authors**: *Julien Pérolat, Rémi Munos, Jean-Baptiste Lespiau, Shayegan Omidshafiei, Mark Rowland, Pedro A. Ortega, Neil Burch, Thomas W. Anthony, David Balduzzi, Bart De Vylder, Georgios Piliouras, Marc Lanctot, Karl Tuyls*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/perolat21a.html](http://proceedings.mlr.press/v139/perolat21a.html)

**Abstract**:

In this paper we investigate the Follow the Regularized Leader dynamics in sequential imperfect information games (IIG). We generalize existing results of Poincar{é} recurrence from normal-form games to zero-sum two-player imperfect information games and other sequential game settings. We then investigate how adapting the reward (by adding a regularization term) of the game can give strong convergence guarantees in monotone games. We continue by showing how this reward adaptation technique can be leveraged to build algorithms that converge exactly to the Nash equilibrium. Finally, we show how these insights can be directly used to build state-of-the-art model-free algorithms for zero-sum two-player Imperfect Information Games (IIG).

----

## [777] Spectral Smoothing Unveils Phase Transitions in Hierarchical Variational Autoencoders

**Authors**: *Adeel Pervez, Efstratios Gavves*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/pervez21a.html](http://proceedings.mlr.press/v139/pervez21a.html)

**Abstract**:

Variational autoencoders with deep hierarchies of stochastic layers have been known to suffer from the problem of posterior collapse, where the top layers fall back to the prior and become independent of input. We suggest that the hierarchical VAE objective explicitly includes the variance of the function parameterizing the mean and variance of the latent Gaussian distribution which itself is often a high variance function. Building on this we generalize VAE neural networks by incorporating a smoothing parameter motivated by Gaussian analysis to reduce higher frequency components and consequently the variance in parameterizing functions and show that this can help to solve the problem of posterior collapse. We further show that under such smoothing the VAE loss exhibits a phase transition, where the top layer KL divergence sharply drops to zero at a critical value of the smoothing parameter that is similar for the same model across datasets. We validate the phenomenon across model configurations and datasets.

----

## [778] Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision

**Authors**: *Felix Petersen, Christian Borgelt, Hilde Kuehne, Oliver Deussen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/petersen21a.html](http://proceedings.mlr.press/v139/petersen21a.html)

**Abstract**:

Sorting and ranking supervision is a method for training neural networks end-to-end based on ordering constraints. That is, the ground truth order of sets of samples is known, while their absolute values remain unsupervised. For that, we propose differentiable sorting networks by relaxing their pairwise conditional swap operations. To address the problems of vanishing gradients and extensive blurring that arise with larger numbers of layers, we propose mapping activations to regions with moderate gradients. We consider odd-even as well as bitonic sorting networks, which outperform existing relaxations of the sorting operation. We show that bitonic sorting networks can achieve stable training on large input sets of up to 1024 elements.

----

## [779] Megaverse: Simulating Embodied Agents at One Million Experiences per Second

**Authors**: *Aleksei Petrenko, Erik Wijmans, Brennan Shacklett, Vladlen Koltun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/petrenko21a.html](http://proceedings.mlr.press/v139/petrenko21a.html)

**Abstract**:

We present Megaverse, a new 3D simulation platform for reinforcement learning and embodied AI research. The efficient design of our engine enables physics-based simulation with high-dimensional egocentric observations at more than 1,000,000 actions per second on a single 8-GPU node. Megaverse is up to 70x faster than DeepMind Lab in fully-shaded 3D scenes with interactive objects. We achieve this high simulation performance by leveraging batched simulation, thereby taking full advantage of the massive parallelism of modern GPUs. We use Megaverse to build a new benchmark that consists of several single-agent and multi-agent tasks covering a variety of cognitive challenges. We evaluate model-free RL on this benchmark to provide baselines and facilitate future research.

----

## [780] Towards Practical Mean Bounds for Small Samples

**Authors**: *My Phan, Philip S. Thomas, Erik G. Learned-Miller*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/phan21a.html](http://proceedings.mlr.press/v139/phan21a.html)

**Abstract**:

Historically, to bound the mean for small sample sizes, practitioners have had to choose between using methods with unrealistic assumptions about the unknown distribution (e.g., Gaussianity) and methods like Hoeffding’s inequality that use weaker assumptions but produce much looser (wider) intervals. In 1969, \citet{Anderson1969} proposed a mean confidence interval strictly better than or equal to Hoeffding’s whose only assumption is that the distribution’s support is contained in an interval $[a,b]$. For the first time since then, we present a new family of bounds that compares favorably to Anderson’s. We prove that each bound in the family has {\em guaranteed coverage}, i.e., it holds with probability at least $1-\alpha$ for all distributions on an interval $[a,b]$. Furthermore, one of the bounds is tighter than or equal to Anderson’s for all samples. In simulations, we show that for many distributions, the gain over Anderson’s bound is substantial.

----

## [781] DG-LMC: A Turn-key and Scalable Synchronous Distributed MCMC Algorithm via Langevin Monte Carlo within Gibbs

**Authors**: *Vincent Plassier, Maxime Vono, Alain Durmus, Eric Moulines*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/plassier21a.html](http://proceedings.mlr.press/v139/plassier21a.html)

**Abstract**:

Performing reliable Bayesian inference on a big data scale is becoming a keystone in the modern era of machine learning. A workhorse class of methods to achieve this task are Markov chain Monte Carlo (MCMC) algorithms and their design to handle distributed datasets has been the subject of many works. However, existing methods are not completely either reliable or computationally efficient. In this paper, we propose to fill this gap in the case where the dataset is partitioned and stored on computing nodes within a cluster under a master/slaves architecture. We derive a user-friendly centralised distributed MCMC algorithm with provable scaling in high-dimensional settings. We illustrate the relevance of the proposed methodology on both synthetic and real data experiments.

----

## [782] GeomCA: Geometric Evaluation of Data Representations

**Authors**: *Petra Poklukar, Anastasiia Varava, Danica Kragic*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/poklukar21a.html](http://proceedings.mlr.press/v139/poklukar21a.html)

**Abstract**:

Evaluating the quality of learned representations without relying on a downstream task remains one of the challenges in representation learning. In this work, we present Geometric Component Analysis (GeomCA) algorithm that evaluates representation spaces based on their geometric and topological properties. GeomCA can be applied to representations of any dimension, independently of the model that generated them. We demonstrate its applicability by analyzing representations obtained from a variety of scenarios, such as contrastive learning models, generative models and supervised learning models.

----

## [783] Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech

**Authors**: *Vadim Popov, Ivan Vovk, Vladimir Gogoryan, Tasnima Sadekova, Mikhail A. Kudinov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/popov21a.html](http://proceedings.mlr.press/v139/popov21a.html)

**Abstract**:

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

----

## [784] Bias-Free Scalable Gaussian Processes via Randomized Truncations

**Authors**: *Andres Potapczynski, Luhuan Wu, Dan Biderman, Geoff Pleiss, John P. Cunningham*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/potapczynski21a.html](http://proceedings.mlr.press/v139/potapczynski21a.html)

**Abstract**:

Scalable Gaussian Process methods are computationally attractive, yet introduce modeling biases that require rigorous study. This paper analyzes two common techniques: early truncated conjugate gradients (CG) and random Fourier features (RFF). We find that both methods introduce a systematic bias on the learned hyperparameters: CG tends to underfit while RFF tends to overfit. We address these issues using randomized truncation estimators that eliminate bias in exchange for increased variance. In the case of RFF, we show that the bias-to-variance conversion is indeed a trade-off: the additional variance proves detrimental to optimization. However, in the case of CG, our unbiased learning procedure meaningfully outperforms its biased counterpart with minimal additional computation. Our code is available at https://github.com/ cunningham-lab/RTGPS.

----

## [785] Dense for the Price of Sparse: Improved Performance of Sparsely Initialized Networks via a Subspace Offset

**Authors**: *Ilan Price, Jared Tanner*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/price21a.html](http://proceedings.mlr.press/v139/price21a.html)

**Abstract**:

That neural networks may be pruned to high sparsities and retain high accuracy is well established. Recent research efforts focus on pruning immediately after initialization so as to allow the computational savings afforded by sparsity to extend to the training process. In this work, we introduce a new ‘DCT plus Sparse’ layer architecture, which maintains information propagation and trainability even with as little as 0.01% trainable parameters remaining. We show that standard training of networks built with these layers, and pruned at initialization, achieves state-of-the-art accuracy for extreme sparsities on a variety of benchmark network architectures and datasets. Moreover, these results are achieved using only simple heuristics to determine the locations of the trainable parameters in the network, and thus without having to initially store or compute with the full, unpruned network, as is required by competing prune-at-initialization algorithms. Switching from standard sparse layers to DCT plus Sparse layers does not increase the storage footprint of a network and incurs only a small additional computational overhead.

----

## [786] BANG: Bridging Autoregressive and Non-autoregressive Generation with Large Scale Pretraining

**Authors**: *Weizhen Qi, Yeyun Gong, Jian Jiao, Yu Yan, Weizhu Chen, Dayiheng Liu, Kewen Tang, Houqiang Li, Jiusheng Chen, Ruofei Zhang, Ming Zhou, Nan Duan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qi21a.html](http://proceedings.mlr.press/v139/qi21a.html)

**Abstract**:

In this paper, we propose BANG, a new pretraining model to Bridge the gap between Autoregressive (AR) and Non-autoregressive (NAR) Generation. AR and NAR generation can be uniformly regarded as to what extent previous tokens can be attended, and BANG bridges AR and NAR generation through designing a novel model structure for large-scale pre-training. A pretrained BANG model can simultaneously support AR, NAR, and semi-NAR generation to meet different requirements. Experiments on question generation (SQuAD 1.1), summarization (XSum), and dialogue generation (PersonaChat) show that BANG improves NAR and semi-NAR performance significantly as well as attaining comparable performance with strong AR pretrained models. Compared with the semi-NAR strong baselines, BANG achieves absolute improvements of 14.01 and 5.24 in the overall scores of SQuAD 1.1 and XSum, respectively. In addition, BANG achieves absolute improvements of 10.73, 6.39, and 5.90 in the overall scores of SQuAD, XSUM, and PersonaChat compared with the NAR strong baselines, respectively. Our code will be made publicly available.

----

## [787] A Probabilistic Approach to Neural Network Pruning

**Authors**: *Xin Qian, Diego Klabjan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qian21a.html](http://proceedings.mlr.press/v139/qian21a.html)

**Abstract**:

Neural network pruning techniques reduce the number of parameters without compromising predicting ability of a network. Many algorithms have been developed for pruning both over-parameterized fully-connected networks (FCN) and convolutional neural networks (CNN), but analytical studies of capabilities and compression ratios of such pruned sub-networks are lacking. We theoretically study the performance of two pruning techniques (random and magnitude-based) on FCN and CNN. Given a target network, we provide a universal approach to bound the gap between a pruned and the target network in a probabilistic sense, which is the first study of this nature. The results establish that there exist pruned networks with expressive power within any specified bound from the target network and with a significant compression ratio.

----

## [788] Global Prosody Style Transfer Without Text Transcriptions

**Authors**: *Kaizhi Qian, Yang Zhang, Shiyu Chang, Jinjun Xiong, Chuang Gan, David D. Cox, Mark Hasegawa-Johnson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qian21b.html](http://proceedings.mlr.press/v139/qian21b.html)

**Abstract**:

Prosody plays an important role in characterizing the style of a speaker or an emotion, but most non-parallel voice or emotion style transfer algorithms do not convert any prosody information. Two major components of prosody are pitch and rhythm. Disentangling the prosody information, particularly the rhythm component, from the speech is challenging because it involves breaking the synchrony between the input speech and the disentangled speech representation. As a result, most existing prosody style transfer algorithms would need to rely on some form of text transcriptions to identify the content information, which confines their application to high-resource languages only. Recently, SpeechSplit has made sizeable progress towards unsupervised prosody style transfer, but it is unable to extract high-level global prosody style in an unsupervised manner. In this paper, we propose AutoPST, which can disentangle global prosody style from speech without relying on any text transcriptions. AutoPST is an Autoencoder-based Prosody Style Transfer framework with a thorough rhythm removal module guided by the self-expressive representation learning. Experiments on different style transfer tasks show that AutoPST can effectively convert prosody that correctly reflects the styles of the target domains.

----

## [789] Efficient Differentiable Simulation of Articulated Bodies

**Authors**: *Yi-Ling Qiao, Junbang Liang, Vladlen Koltun, Ming C. Lin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qiao21a.html](http://proceedings.mlr.press/v139/qiao21a.html)

**Abstract**:

We present a method for efficient differentiable simulation of articulated bodies. This enables integration of articulated body dynamics into deep learning frameworks, and gradient-based optimization of neural networks that operate on articulated bodies. We derive the gradients of the contact solver using spatial algebra and the adjoint method. Our approach is an order of magnitude faster than autodiff tools. By only saving the initial states throughout the simulation process, our method reduces memory requirements by two orders of magnitude. We demonstrate the utility of efficient differentiable dynamics for articulated bodies in a variety of applications. We show that reinforcement learning with articulated systems can be accelerated using gradients provided by our method. In applications to control and inverse problems, gradient-based optimization enabled by our work accelerates convergence by more than an order of magnitude.

----

## [790] Oneshot Differentially Private Top-k Selection

**Authors**: *Gang Qiao, Weijie J. Su, Li Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qiao21b.html](http://proceedings.mlr.press/v139/qiao21b.html)

**Abstract**:

Being able to efficiently and accurately select the top-$k$ elements with differential privacy is an integral component of various private data analysis tasks. In this paper, we present the oneshot Laplace mechanism, which generalizes the well-known Report Noisy Max \cite{dwork2014algorithmic} mechanism to reporting noisy top-$k$ elements. We show that the oneshot Laplace mechanism with a noise level of $\widetilde{O}(\sqrt{k}/\eps)$ is approximately differentially private. Compared to the previous peeling approach of running Report Noisy Max $k$ times, the oneshot Laplace mechanism only adds noises and computes the top $k$ elements once, hence much more efficient for large $k$. In addition, our proof of privacy relies on a novel coupling technique that bypasses the composition theorems so without the linear dependence on $k$ which is inherent to various composition theorems. Finally, we present a novel application of efficient top-$k$ selection in the classical problem of ranking from pairwise comparisons.

----

## [791] Density Constrained Reinforcement Learning

**Authors**: *Zengyi Qin, Yuxiao Chen, Chuchu Fan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qin21a.html](http://proceedings.mlr.press/v139/qin21a.html)

**Abstract**:

We study constrained reinforcement learning (CRL) from a novel perspective by setting constraints directly on state density functions, rather than the value functions considered by previous works. State density has a clear physical and mathematical interpretation, and is able to express a wide variety of constraints such as resource limits and safety requirements. Density constraints can also avoid the time-consuming process of designing and tuning cost functions required by value function-based constraints to encode system specifications. We leverage the duality between density functions and Q functions to develop an effective algorithm to solve the density constrained RL problem optimally and the constrains are guaranteed to be satisfied. We prove that the proposed algorithm converges to a near-optimal solution with a bounded error even when the policy update is imperfect. We use a set of comprehensive experiments to demonstrate the advantages of our approach over state-of-the-art CRL methods, with a wide range of density constrained tasks as well as standard CRL benchmarks such as Safety-Gym.

----

## [792] Budgeted Heterogeneous Treatment Effect Estimation

**Authors**: *Tian Qin, Tian-Zuo Wang, Zhi-Hua Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qin21b.html](http://proceedings.mlr.press/v139/qin21b.html)

**Abstract**:

Heterogeneous treatment effect (HTE) estimation is receiving increasing interest due to its important applications in fields such as healthcare, economics, and education. Current HTE estimation methods generally assume the existence of abundant observational data, though the acquisition of such data can be costly. In some real scenarios, it is easy to access the pre-treatment covariates and treatment assignments, but expensive to obtain the factual outcomes. To make HTE estimation more practical, in this paper, we examine the problem of estimating HTEs with a budget constraint on observational data, aiming to obtain accurate HTE estimates with limited costs. By deriving an informative generalization bound and connecting to active learning, we propose an effective and efficient method which is validated both theoretically and empirically.

----

## [793] Neural Transformation Learning for Deep Anomaly Detection Beyond Images

**Authors**: *Chen Qiu, Timo Pfrommer, Marius Kloft, Stephan Mandt, Maja Rudolph*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qiu21a.html](http://proceedings.mlr.press/v139/qiu21a.html)

**Abstract**:

Data transformations (e.g. rotations, reflections, and cropping) play an important role in self-supervised learning. Typically, images are transformed into different views, and neural networks trained on tasks involving these views produce useful feature representations for downstream tasks, including anomaly detection. However, for anomaly detection beyond image data, it is often unclear which transformations to use. Here we present a simple end-to-end procedure for anomaly detection with learnable transformations. The key idea is to embed the transformed data into a semantic space such that the transformed data still resemble their untransformed form, while different transformations are easily distinguishable. Extensive experiments on time series show that our proposed method outperforms existing approaches in the one-vs.-rest setting and is competitive in the more challenging n-vs.-rest anomaly-detection task. On medical and cyber-security tabular data, our method learns domain-specific transformations and detects anomalies more accurately than previous work.

----

## [794] Provably Efficient Fictitious Play Policy Optimization for Zero-Sum Markov Games with Structured Transitions

**Authors**: *Shuang Qiu, Xiaohan Wei, Jieping Ye, Zhaoran Wang, Zhuoran Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qiu21b.html](http://proceedings.mlr.press/v139/qiu21b.html)

**Abstract**:

While single-agent policy optimization in a fixed environment has attracted a lot of research attention recently in the reinforcement learning community, much less is known theoretically when there are multiple agents playing in a potentially competitive environment. We take steps forward by proposing and analyzing new fictitious play policy optimization algorithms for two-player zero-sum Markov games with structured but unknown transitions. We consider two classes of transition structures: factored independent transition and single-controller transition. For both scenarios, we prove tight $\widetilde{\mathcal{O}}(\sqrt{T})$ regret bounds after $T$ steps in a two-agent competitive game scenario. The regret of each player is measured against a potentially adversarial opponent who can choose a single best policy in hindsight after observing the full policy sequence. Our algorithms feature a combination of Upper Confidence Bound (UCB)-type optimism and fictitious play under the scope of simultaneous policy optimization in a non-stationary environment. When both players adopt the proposed algorithms, their overall optimality gap is $\widetilde{\mathcal{O}}(\sqrt{T})$.

----

## [795] Optimization Planning for 3D ConvNets

**Authors**: *Zhaofan Qiu, Ting Yao, Chong-Wah Ngo, Tao Mei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qiu21c.html](http://proceedings.mlr.press/v139/qiu21c.html)

**Abstract**:

It is not trivial to optimally learn a 3D Convolutional Neural Networks (3D ConvNets) due to high complexity and various options of the training scheme. The most common hand-tuning process starts from learning 3D ConvNets using short video clips and then is followed by learning long-term temporal dependency using lengthy clips, while gradually decaying the learning rate from high to low as training progresses. The fact that such process comes along with several heuristic settings motivates the study to seek an optimal "path" to automate the entire training. In this paper, we decompose the path into a series of training "states" and specify the hyper-parameters, e.g., learning rate and the length of input clips, in each state. The estimation of the knee point on the performance-epoch curve triggers the transition from one state to another. We perform dynamic programming over all the candidate states to plan the optimal permutation of states, i.e., optimization path. Furthermore, we devise a new 3D ConvNets with a unique design of dual-head classifier to improve spatial and temporal discrimination. Extensive experiments on seven public video recognition benchmarks demonstrate the advantages of our proposal. With the optimization planning, our 3D ConvNets achieves superior results when comparing to the state-of-the-art recognition methods. More remarkably, we obtain the top-1 accuracy of 80.5% and 82.7% on Kinetics-400 and Kinetics-600 datasets, respectively.

----

## [796] On Reward-Free RL with Kernel and Neural Function Approximations: Single-Agent MDP and Markov Game

**Authors**: *Shuang Qiu, Jieping Ye, Zhaoran Wang, Zhuoran Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/qiu21d.html](http://proceedings.mlr.press/v139/qiu21d.html)

**Abstract**:

To achieve sample efficiency in reinforcement learning (RL), it necessitates to efficiently explore the underlying environment. Under the offline setting, addressing the exploration challenge lies in collecting an offline dataset with sufficient coverage. Motivated by such a challenge, we study the reward-free RL problem, where an agent aims to thoroughly explore the environment without any pre-specified reward function. Then, given any extrinsic reward, the agent computes the optimal policy via offline RL with data collected in the exploration stage. Moreover, we tackle this problem under the context of function approximation, leveraging powerful function approximators. Specifically, we propose to explore via an optimistic variant of the value-iteration algorithm incorporating kernel and neural function approximations, where we adopt the associated exploration bonus as the exploration reward. Moreover, we design exploration and planning algorithms for both single-agent MDPs and zero-sum Markov games and prove that our methods can achieve $\widetilde{\mathcal{O}}(1 /\varepsilon^2)$ sample complexity for generating a $\varepsilon$-suboptimal policy or $\varepsilon$-approximate Nash equilibrium when given an arbitrary extrinsic reward. To the best of our knowledge, we establish the first provably efficient reward-free RL algorithm with kernel and neural function approximators.

----

## [797] Learning Transferable Visual Models From Natural Language Supervision

**Authors**: *Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/radford21a.html](http://proceedings.mlr.press/v139/radford21a.html)

**Abstract**:

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.

----

## [798] A General Framework For Detecting Anomalous Inputs to DNN Classifiers

**Authors**: *Jayaram Raghuram, Varun Chandrasekaran, Somesh Jha, Suman Banerjee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/raghuram21a.html](http://proceedings.mlr.press/v139/raghuram21a.html)

**Abstract**:

Detecting anomalous inputs, such as adversarial and out-of-distribution (OOD) inputs, is critical for classifiers (including deep neural networks or DNNs) deployed in real-world applications. While prior works have proposed various methods to detect such anomalous samples using information from the internal layer representations of a DNN, there is a lack of consensus on a principled approach for the different components of such a detection method. As a result, often heuristic and one-off methods are applied for different aspects of this problem. We propose an unsupervised anomaly detection framework based on the internal DNN layer representations in the form of a meta-algorithm with configurable components. We proceed to propose specific instantiations for each component of the meta-algorithm based on ideas grounded in statistical testing and anomaly detection. We evaluate the proposed methods on well-known image classification datasets with strong adversarial attacks and OOD inputs, including an adaptive attack that uses the internal layer representations of the DNN (often not considered in prior work). Comparisons with five recently-proposed competing detection methods demonstrates the effectiveness of our method in detecting adversarial and OOD inputs.

----

## [799] Towards Open Ad Hoc Teamwork Using Graph-based Policy Learning

**Authors**: *Arrasy Rahman, Niklas Höpner, Filippos Christianos, Stefano V. Albrecht*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rahman21a.html](http://proceedings.mlr.press/v139/rahman21a.html)

**Abstract**:

Ad hoc teamwork is the challenging problem of designing an autonomous agent which can adapt quickly to collaborate with teammates without prior coordination mechanisms, including joint training. Prior work in this area has focused on closed teams in which the number of agents is fixed. In this work, we consider open teams by allowing agents with different fixed policies to enter and leave the environment without prior notification. Our solution builds on graph neural networks to learn agent models and joint-action value models under varying team compositions. We contribute a novel action-value computation that integrates the agent model and joint-action value model to produce action-value estimates. We empirically demonstrate that our approach successfully models the effects other agents have on the learner, leading to policies that robustly adapt to dynamic team compositions and significantly outperform several alternative methods.

----



[Go to the previous page](ICML-2021-list03.md)

[Go to the next page](ICML-2021-list05.md)

[Go to the catalog section](README.md)