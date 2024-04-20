## [400] Grounding Spatio-Temporal Language with Transformers

**Authors**: *Tristan Karch, Laetitia Teodorescu, Katja Hofmann, Clément Moulin-Frier, Pierre-Yves Oudeyer*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/29daf9442f3c0b60642b14c081b4a556-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/29daf9442f3c0b60642b14c081b4a556-Abstract.html)

**Abstract**:

Language is an interface to the outside world. In order for embodied agents to use it, language must be grounded in other, sensorimotor modalities. While there is an extended literature studying how machines can learn grounded language, the topic of how to learn spatio-temporal linguistic concepts is still largely uncharted. To make progress in this direction, we here introduce a novel spatio-temporal language grounding task where the goal is to learn the meaning of spatio-temporal descriptions of behavioral traces of an embodied agent. This is achieved by training a truth function that predicts if a description matches a given history of observations. The descriptions involve time-extended predicates in past and present tense as well as spatio-temporal references to objects in the scene. To study the role of architectural biases in this task, we train several models including multimodal Transformer architectures; the latter implement different attention computations between words and objects across space and time. We test models on two classes of generalization: 1) generalization to new sentences, 2) generalization to grammar primitives. We observe that maintaining object identity in the attention computation of our Transformers is instrumental to achieving good performance on generalization overall, and that summarizing object traces in a single token has little influence on performance. We then discuss how this opens new perspectives for language-guided autonomous embodied agents.

----

## [401] Learning where to learn: Gradient sparsity in meta and continual learning

**Authors**: *Johannes von Oswald, Dominic Zhao, Seijin Kobayashi, Simon Schug, Massimo Caccia, Nicolas Zucchet, João Sacramento*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2a10665525774fa2501c2c8c4985ce61-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2a10665525774fa2501c2c8c4985ce61-Abstract.html)

**Abstract**:

Finding neural network weights that generalize well from small datasets is difficult. A promising approach is to learn a weight initialization such that a small number of weight changes results in low generalization error. We show that this form of meta-learning can be improved by letting the learning algorithm decide which weights to change, i.e., by learning where to learn. We find that patterned sparsity emerges from this process, with the pattern of sparsity varying on a problem-by-problem basis. This selective sparsity results in better generalization and less interference in a range of few-shot and continual learning problems. Moreover, we find that sparse learning also emerges in a more expressive model where learning rates are meta-learned. Our results shed light on an ongoing debate on whether meta-learning can discover adaptable features and suggest that learning by sparse gradient descent is a powerful inductive bias for meta-learning systems.

----

## [402] Domain Invariant Representation Learning with Domain Density Transformations

**Authors**: *A. Tuan Nguyen, Toan Tran, Yarin Gal, Atilim Gunes Baydin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2a2717956118b4d223ceca17ce3865e2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2a2717956118b4d223ceca17ce3865e2-Abstract.html)

**Abstract**:

Domain generalization refers to the problem where we aim to train a model on data from a set of source domains so that the model can generalize to unseen target domains. Naively training a model on the aggregate set of data (pooled from all source domains) has been shown to perform suboptimally, since the information learned by that model might be domain-specific and generalize imperfectly to target domains. To tackle this problem, a predominant domain generalization approach is to  learn some domain-invariant information for the prediction task, aiming at a good generalization across domains. In this paper, we propose a theoretically grounded method to learn a domain-invariant representation by enforcing the representation network to be invariant under all transformation functions among domains. We next introduce the use of generative adversarial networks to learn such domain transformations in a possible implementation of our method in practice. We demonstrate the effectiveness of our method on several widely used datasets for the domain generalization problem, on all of which we achieve competitive results with state-of-the-art models.

----

## [403] PlayVirtual: Augmenting Cycle-Consistent Virtual Trajectories for Reinforcement Learning

**Authors**: *Tao Yu, Cuiling Lan, Wenjun Zeng, Mingxiao Feng, Zhizheng Zhang, Zhibo Chen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2a38a4a9316c49e5a833517c45d31070-Abstract.html)

**Abstract**:

Learning good feature representations is important for deep reinforcement learning (RL). However, with limited experience, RL often suffers from data inefficiency for training. For un-experienced or less-experienced trajectories (i.e., state-action sequences), the lack of data limits the use of them for better feature learning. In this work, we propose a novel method, dubbed PlayVirtual, which augments cycle-consistent virtual trajectories to enhance the data efficiency for RL feature representation learning. Specifically, PlayVirtual predicts future states in a latent space based on the current state and action by a dynamics model and then predicts the previous states by a backward dynamics model, which forms a trajectory cycle. Based on this, we augment the actions to generate a large amount of virtual state-action trajectories. Being free of groudtruth state supervision, we enforce a trajectory to meet the cycle consistency constraint, which can significantly enhance the data efficiency. We validate the effectiveness of our designs on the Atari and DeepMind Control Suite benchmarks. Our method achieves the state-of-the-art performance on both benchmarks. Our code is available at https://github.com/microsoft/Playvirtual.

----

## [404] Efficient Equivariant Network

**Authors**: *Lingshen He, Yuxuan Chen, Zhengyang Shen, Yiming Dong, Yisen Wang, Zhouchen Lin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2a79ea27c279e471f4d180b08d62b00a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2a79ea27c279e471f4d180b08d62b00a-Abstract.html)

**Abstract**:

Convolutional neural networks (CNNs) have dominated the field of Computer Vision and achieved great success due to their built-in translation equivariance. Group equivariant CNNs (G-CNNs) that incorporate more equivariance can significantly improve the performance of conventional CNNs. However, G-CNNs are faced with two major challenges: \emph{spatial-agnostic problem} and \emph{expensive computational cost}. In this work, we propose a general framework of previous equivariant models, which includes G-CNNs and equivariant self-attention layers as special cases. Under this framework, we explicitly decompose the feature aggregation operation into a kernel generator and an encoder, and decouple the spatial and extra geometric dimensions in the computation. Therefore, our filters are essentially dynamic rather than being spatial-agnostic. We further show that our \emph{E}quivariant model is parameter \emph{E}fficient and computation \emph{E}fficient by complexity analysis, and also data \emph{E}fficient by experiments, so we call our model $E^4$-Net. Extensive experiments verify that our model can significantly improve previous works with smaller model size.Especially, under the setting of training on $1/5$ data of CIFAR10, our model improves G-CNNs by $5\%+$ accuracy,while using only $56\%$ parameters and $68\%$ FLOPs.

----

## [405] Unifying Gradient Estimators for Meta-Reinforcement Learning via Off-Policy Evaluation

**Authors**: *Yunhao Tang, Tadashi Kozuno, Mark Rowland, Rémi Munos, Michal Valko*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2a8009525763356ad5e3bb48b7475b4d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2a8009525763356ad5e3bb48b7475b4d-Abstract.html)

**Abstract**:

Model-agnostic meta-reinforcement learning requires estimating the Hessian matrix of value functions. This is challenging from an implementation perspective, as repeatedly differentiating policy gradient estimates may lead to biased Hessian estimates. In this work, we provide a unifying framework for estimating higher-order derivatives of value functions, based on off-policy evaluation. Our framework interprets a number of prior approaches as special cases and elucidates the bias and variance trade-off of Hessian estimates. This framework also opens the door to a new family of estimates, which can be easily implemented with auto-differentiation libraries, and lead to performance gains in practice.

----

## [406] Even your Teacher Needs Guidance: Ground-Truth Targets Dampen Regularization Imposed by Self-Distillation

**Authors**: *Kenneth Borup, Lars Nørvang Andersen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2adcefe38fbcd3dcd45908fbab1bf628-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2adcefe38fbcd3dcd45908fbab1bf628-Abstract.html)

**Abstract**:

Knowledge distillation is classically a procedure where a neural network is trained on the output of another network along with the original targets in order to transfer knowledge between the architectures. The special case of self-distillation, where the network architectures are identical, has been observed to improve generalization accuracy. In this paper, we consider an iterative variant of self-distillation in a kernel regression setting, in which successive steps incorporate both model outputs and the ground-truth targets. This allows us to provide the first theoretical results on the importance of using the weighted ground-truth targets in self-distillation. Our focus is on fitting nonlinear functions to training data with a weighted mean square error objective function suitable for distillation, subject to $\ell_2$ regularization of the model parameters. We show that any such function obtained with self-distillation can be calculated directly as a function of the initial fit, and that infinite distillation steps yields the same optimization problem as the original with amplified regularization. Furthermore, we provide a closed form solution for the optimal choice of weighting parameter at each step, and show how to efficiently estimate this weighting parameter for deep learning and significantly reduce the computational requirements compared to a grid search.

----

## [407] Compressing Neural Networks: Towards Determining the Optimal Layer-wise Decomposition

**Authors**: *Lucas Liebenwein, Alaa Maalouf, Dan Feldman, Daniela Rus*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2adcfc3929e7c03fac3100d3ad51da26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2adcfc3929e7c03fac3100d3ad51da26-Abstract.html)

**Abstract**:

We present a novel global compression framework for deep neural networks that automatically analyzes each layer to identify the optimal per-layer compression ratio, while simultaneously achieving the desired overall compression. Our algorithm hinges on the idea of compressing each convolutional (or fully-connected) layer by slicing its channels into multiple groups and decomposing each group via low-rank decomposition. At the core of our algorithm is the derivation of layer-wise error bounds from the Eckart–Young–Mirsky theorem. We then leverage these bounds to frame the compression problem as an optimization problem where we wish to minimize the maximum compression error across layers and propose an efficient algorithm towards a solution. Our experiments indicate that our method outperforms existing low-rank compression approaches across a wide range of networks and data sets. We believe that our results open up new avenues for future research into the global performance-size trade-offs of modern neural networks.

----

## [408] Equilibrium and non-Equilibrium regimes in the learning of Restricted Boltzmann Machines

**Authors**: *Aurélien Decelle, Cyril Furtlehner, Beatriz Seoane*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2aedcba61ca55ceb62d785c6b7f10a83-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2aedcba61ca55ceb62d785c6b7f10a83-Abstract.html)

**Abstract**:

Training Restricted Boltzmann Machines (RBMs) has been challenging for a long time due to the difficulty of computing precisely the log-likelihood gradient. Over the past decades, many works have proposed more or less successful recipes but without studying systematically the crucial quantity of the problem: the mixing time i.e. the number of MCMC iterations needed to sample completely new configurations from a model. In this work, we show that this mixing time plays a crucial role in the behavior and stability of the trained model, and that RBMs operate in two well-defined distinct regimes, namely equilibrium and out-of-equilibrium, depending on the interplay between this mixing time of the model and the number of MCMC steps, $k$, used to approximate the gradient.  We further show empirically that this mixing time increases along the learning, which often implies a transition from one regime to another as soon as $k$ becomes smaller than this time.In particular, we show that using the popular $k$ (persistent) contrastive divergence approaches, with $k$ small, the dynamics of the fitted model are extremely slow and often dominated by strong out-of-equilibrium effects. On the contrary, RBMs trained in equilibrium display much faster dynamics, and a smooth convergence to dataset-like configurations during the sampling.Finally, we discuss how to exploit in practice both regimes depending on the task one aims to fulfill: (i) short $k$s can be used to generate convincing samples in short learning times, (ii) large $k$ (or increasingly large) must be used to learn the correct equilibrium distribution of the RBM. Finally, the existence of these two operational regimes seems to be a general property of energy based models trained via likelihood maximization.

----

## [409] Imitation with Neural Density Models

**Authors**: *Kuno Kim, Akshat Jindal, Yang Song, Jiaming Song, Yanan Sui, Stefano Ermon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b0aa0d9e30ea3a55fc271ced8364536-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b0aa0d9e30ea3a55fc271ced8364536-Abstract.html)

**Abstract**:

We propose a new framework for Imitation Learning (IL) via density estimation of the expert's occupancy measure followed by Maximum Occupancy Entropy Reinforcement Learning (RL) using the density as a reward. Our approach maximizes a non-adversarial model-free RL objective that provably lower bounds reverse Kullbackâ€“Leibler divergence between occupancy measures of the expert and imitator. We present a practical IL algorithm, Neural Density Imitation (NDI), which obtains state-of-the-art demonstration efficiency on benchmark control tasks.

----

## [410] Accurate Point Cloud Registration with Robust Optimal Transport

**Authors**: *Zhengyang Shen, Jean Feydy, Peirong Liu, Ariel Hernán Curiale, Rubén San José Estépar, Raúl San José Estépar, Marc Niethammer*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html)

**Abstract**:

This work investigates the use of robust optimal transport (OT) for shape matching. Specifically, we show that recent OT solvers improve both optimization-based and deep learning methods for point cloud registration, boosting accuracy at an affordable computational cost. This manuscript starts with a practical overview of modern OT theory. We then provide solutions to the main difficulties in using this framework for shape matching. Finally, we showcase the performance of transport-enhanced registration models on a wide range of challenging tasks: rigid registration for partial shapes; scene flow estimation on the Kitti dataset; and nonparametric registration of lung vascular trees between inspiration and expiration. Our OT-based methods achieve state-of-the-art results on Kitti and for the challenging lung registration task, both in terms of accuracy and scalability. We also release PVT1010, a new public dataset of 1,010 pairs of lung vascular trees with densely sampled points. This dataset provides a challenging use case for point cloud registration algorithms with highly complex shapes and deformations. Our work demonstrates that robust OT enables fast pre-alignment and fine-tuning for a wide range of registration models, thereby providing a new key method for the computer vision toolbox. Our code and dataset are available online at: https://github.com/uncbiag/robot.

----

## [411] Simple steps are all you need: Frank-Wolfe and generalized self-concordant functions

**Authors**: *Alejandro Carderera, Mathieu Besançon, Sebastian Pokutta*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b323d6eb28422cef49b266557dd31ad-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b323d6eb28422cef49b266557dd31ad-Abstract.html)

**Abstract**:

Generalized self-concordance is a key property present in the objective function of many important learning problems.  We establish the convergence rate of a simple Frank-Wolfe variant that uses the open-loop step size strategy $\gamma_t = 2/(t+2)$, obtaining a $\mathcal{O}(1/t)$ convergence rate for this class of functions in terms of primal gap and Frank-Wolfe gap, where $t$ is the iteration count. This avoids the use of second-order information or the need to estimate local smoothness parameters of previous work. We also show improved convergence rates for various common cases, e.g., when the feasible region under consideration is uniformly convex or polyhedral.

----

## [412] Automatic Data Augmentation for Generalization in Reinforcement Learning

**Authors**: *Roberta Raileanu, Maxwell Goldstein, Denis Yarats, Ilya Kostrikov, Rob Fergus*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b38c2df6a49b97f706ec9148ce48d86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b38c2df6a49b97f706ec9148ce48d86-Abstract.html)

**Abstract**:

Deep reinforcement learning (RL) agents often fail to generalize beyond their training environments. To alleviate this problem, recent work has proposed the use of data augmentation. However, different tasks tend to benefit from different types of augmentations and selecting the right one typically requires expert knowledge. In this paper, we introduce three approaches for automatically finding an effective augmentation for any RL task. These are combined with two novel regularization terms for the policy and value function, required to make the use of data augmentation theoretically sound for actor-critic algorithms. Our method achieves a new state-of-the-art on the Procgen benchmark and outperforms popular RL algorithms on DeepMind Control tasks with distractors. In addition, our agent learns policies and representations which are more robust to changes in the environment that are irrelevant for solving the task, such as the background.

----

## [413] Blending Anti-Aliasing into Vision Transformer

**Authors**: *Shengju Qian, Hao Shao, Yi Zhu, Mu Li, Jiaya Jia*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b3bf3eee2475e03885a110e9acaab61-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b3bf3eee2475e03885a110e9acaab61-Abstract.html)

**Abstract**:

The transformer architectures, based on self-attention mechanism and convolution-free design, recently found superior performance and booming applications in computer vision. However, the discontinuous patch-wise tokenization process implicitly introduces jagged artifacts into attention maps, arising the traditional problem of aliasing for vision transformers. Aliasing effect occurs when discrete patterns are used to produce high frequency or continuous information, resulting in the indistinguishable distortions. Recent researches have found that modern convolution networks still suffer from this phenomenon. In this work, we analyze the uncharted problem of aliasing in vision transformer and explore to incorporate anti-aliasing properties. Specifically, we propose a plug-and-play Aliasing-Reduction Module (ARM) to alleviate the aforementioned issue. We investigate the effectiveness and generalization of the proposed method across multiple tasks and various vision transformer families. This lightweight design consistently attains a clear boost over several famous structures. Furthermore, our module also improves data efficiency and robustness of vision transformers.

----

## [414] A Trainable Spectral-Spatial Sparse Coding Model for Hyperspectral Image Restoration

**Authors**: *Théo Bodrito, Alexandre Zouaoui, Jocelyn Chanussot, Julien Mairal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b515e2bdd63b7f034269ad747c93a42-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b515e2bdd63b7f034269ad747c93a42-Abstract.html)

**Abstract**:

Hyperspectral imaging offers new perspectives for diverse applications, ranging from the monitoring of the environment using airborne or satellite remote sensing, precision farming, food safety, planetary exploration, or astrophysics. Unfortunately, the spectral diversity of information comes at the expense of various sources of degradation,  and the lack of accurate ground-truth "clean" hyperspectral signals acquired on the spot makes restoration tasks challenging.  In particular, training deep neural networks for restoration is difficult, in contrast to traditional RGB imaging problems where deep models tend to shine. In this paper, we advocate instead for a hybrid approach based on sparse coding principles that retain the interpretability of classical techniques encoding domain knowledge with handcrafted image priors, while allowing to train model parameters end-to-end without massive amounts of data. We show on various denoising benchmarks that our method is computationally efficient and  significantly outperforms the state of the art.

----

## [415] Posterior Collapse and Latent Variable Non-identifiability

**Authors**: *Yixin Wang, David M. Blei, John P. Cunningham*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b6921f2c64dee16ba21ebf17f3c2c92-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b6921f2c64dee16ba21ebf17f3c2c92-Abstract.html)

**Abstract**:

Variational autoencoders model high-dimensional data by positinglow-dimensional latent variables that are mapped through a flexibledistribution parametrized by a neural network. Unfortunately,variational autoencoders often suffer from posterior collapse: theposterior of the latent variables is equal to its prior, rendering thevariational autoencoder useless as a means to produce meaningfulrepresentations. Existing approaches to posterior collapse oftenattribute it to the use of neural networks or optimization issues dueto variational approximation. In this paper, we consider posteriorcollapse as a problem of latent variable non-identifiability. We provethat the posterior collapses if and only if the latent variables arenon-identifiable in the generative model. This fact implies thatposterior collapse is not a phenomenon specific to the use of flexibledistributions or approximate inference. Rather, it can occur inclassical probabilistic models even with exact inference, which wealso demonstrate. Based on these results, we propose a class oflatent-identifiable variational autoencoders, deep generative modelswhich enforce identifiability without sacrificing flexibility. Thismodel class resolves the problem of latent variablenon-identifiability by leveraging bijective Brenier maps andparameterizing them with input convex neural networks, without specialvariational inference objectives or optimization tricks. Acrosssynthetic and real datasets, latent-identifiable variationalautoencoders outperform existing methods in mitigating posteriorcollapse and providing meaningful representations of the data.

----

## [416] The Benefits of Implicit Regularization from SGD in Least Squares Problems

**Authors**: *Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, Dean P. Foster, Sham M. Kakade*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b6bb5354a56ce256116b6b307a1ea10-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b6bb5354a56ce256116b6b307a1ea10-Abstract.html)

**Abstract**:

Stochastic gradient descent (SGD) exhibits strong algorithmic regularization effects in practice, which has been hypothesized to play an important role in the generalization of modern machine learning approaches. In this work, we seek to understand these issues in the simpler setting of linear regression (including both underparameterized and overparameterized regimes), where our goal is to make sharp instance-based comparisons of the implicit regularization afforded by (unregularized) average SGD with the explicit regularization of ridge regression. For a broad class of least squares problem instances (that are natural in high-dimensional settings), we show: (1) for every problem instance and for every ridge parameter, (unregularized) SGD, when provided with \emph{logarithmically} more samples than that provided to the ridge algorithm, generalizes no worse than the ridge solution (provided SGD uses a tuned constant stepsize); (2) conversely, there exist instances (in this wide problem class) where optimally-tuned ridge regression requires \emph{quadratically} more samples than SGD in order to have the same generalization performance. Taken together, our results show that, up to the logarithmic factors, the generalization performance of SGD is always no worse than that of ridge regression in a wide range of overparameterized problems, and, in fact, could be much better for some problem instances. More generally, our results show how algorithmic regularization has important consequences even in simpler (overparameterized) convex settings.

----

## [417] Generalization of Model-Agnostic Meta-Learning Algorithms: Recurring and Unseen Tasks

**Authors**: *Alireza Fallah, Aryan Mokhtari, Asuman E. Ozdaglar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2b763288faedb7707c0748abe015ab6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2b763288faedb7707c0748abe015ab6c-Abstract.html)

**Abstract**:

In this paper, we study the generalization properties of Model-Agnostic Meta-Learning (MAML) algorithms for supervised learning problems. We focus on the setting in which we train the MAML model over $m$ tasks, each with $n$ data points, and characterize its generalization error from two points of view: First, we assume the new task at test time is one of the training tasks, and we show that, for strongly convex objective functions, the expected excess population loss is bounded by $\mathcal{O}(1/mn)$. Second, we consider the MAML algorithm's generalization to an unseen task and show that the resulting generalization error depends on the total variation distance between the underlying distributions of the new task and the tasks observed during the training process. Our proof techniques rely on the connections between algorithmic stability and generalization bounds of algorithms. In particular, we propose a new definition of stability for meta-learning algorithms, which allows us to capture the role of both the number of tasks $m$ and number of samples per task $n$ on the generalization error of MAML.

----

## [418] Factored Policy Gradients: Leveraging Structure for Efficient Learning in MOMDPs

**Authors**: *Thomas Spooner, Nelson Vadori, Sumitra Ganesh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2ba8698b79439589fdd2b0f7218d8b07-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2ba8698b79439589fdd2b0f7218d8b07-Abstract.html)

**Abstract**:

Policy gradient methods can solve complex tasks but often fail when the dimensionality of the action-space or objective multiplicity grow very large. This occurs, in part, because the variance on score-based gradient estimators scales quadratically. In this paper, we address this problem through a factor baseline which exploits independence structure encoded in a novel action-target influence network. Factored policy gradients (FPGs), which follow, provide a common framework for analysing key state-of-the-art algorithms, are shown to generalise traditional policy gradients, and yield a principled way of incorporating prior knowledge of a problem domain's generative processes. We provide an analysis of the proposed estimator and identify the conditions under which variance is reduced. The algorithmic aspects of FPGs are discussed, including optimal policy factorisation, as characterised by minimum biclique coverings, and the implications for the bias variance trade-off of incorrectly specifying the network. Finally, we demonstrate the performance advantages of our algorithm on large-scale bandit and traffic intersection problems,  providing a novel contribution to the latter in the form of a spatial approximation.

----

## [419] MarioNette: Self-Supervised Sprite Learning

**Authors**: *Dmitriy Smirnov, Michaël Gharbi, Matthew Fisher, Vitor Guizilini, Alexei A. Efros, Justin M. Solomon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2bcab9d935d219641434683dd9d18a03-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2bcab9d935d219641434683dd9d18a03-Abstract.html)

**Abstract**:

Artists and video game designers often construct 2D animations using libraries of sprites---textured patches of objects and characters. We propose a deep learning approach that decomposes sprite-based video animations into a disentangled representation of recurring graphic elements in a self-supervised manner. By jointly learning a dictionary of possibly transparent patches and training a network that places them onto a canvas, we deconstruct sprite-based content into a sparse, consistent, and explicit representation that can be easily used in downstream tasks, like editing or analysis. Our framework offers a promising approach for discovering recurring visual patterns in image collections without supervision.

----

## [420] RLlib Flow: Distributed Reinforcement Learning is a Dataflow Problem

**Authors**: *Eric Liang, Zhanghao Wu, Michael Luo, Sven Mika, Joseph E. Gonzalez, Ion Stoica*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2bce32ed409f5ebcee2a7b417ad9beed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2bce32ed409f5ebcee2a7b417ad9beed-Abstract.html)

**Abstract**:

Researchers and practitioners in the field of reinforcement learning (RL) frequently leverage parallel computation, which has led to a plethora of new algorithms and systems in the last few years. In this paper, we re-examine the challenges posed by distributed RL and try to view it through the lens of an old idea: distributed dataflow. We show that viewing RL as a dataflow problem leads to highly composable and performant implementations. We propose RLlib Flow, a hybrid actor-dataflow programming model for distributed RL, and validate its practicality by porting the full suite of algorithms in RLlib, a widely adopted distributed RL library. Concretely, RLlib Flow provides 2-9$\times$ code savings in real production code and enables the composition of multi-agent algorithms not possible by end users before. The open-source code is available as part of RLlib at https://github.com/ray-project/ray/tree/master/rllib.

----

## [421] Improve Agents without Retraining: Parallel Tree Search with Off-Policy Correction

**Authors**: *Gal Dalal, Assaf Hallak, Steven Dalton, Iuri Frosio, Shie Mannor, Gal Chechik*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2bd235c31c97855b7ef2dc8b414779af-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2bd235c31c97855b7ef2dc8b414779af-Abstract.html)

**Abstract**:

Tree Search (TS) is crucial to some of the most influential successes in reinforcement learning. Here, we tackle two major challenges with TS that limit its usability: \textit{distribution shift} and \textit{scalability}. We first discover and analyze a counter-intuitive phenomenon: action selection through TS and a pre-trained value function often leads to lower performance compared to the original pre-trained agent, even when having access to the exact state and reward in future steps. We show this is due to a distribution shift to areas where value estimates are highly inaccurate and analyze this effect using Extreme Value theory. To overcome this problem, we introduce a novel off-policy correction term that accounts for the mismatch between the pre-trained value and its corresponding TS policy by penalizing under-sampled trajectories. We prove that our correction eliminates the above mismatch and bound the probability of sub-optimal action selection. Our correction significantly improves pre-trained Rainbow agents without any further training, often more than doubling their scores on Atari games. Next, we address the scalability issue given by the computational complexity of exhaustive TS that scales exponentially with the tree depth. We introduce Batch-BFS: a GPU breadth-first search that advances all nodes in each depth of the tree simultaneously. Batch-BFS reduces runtime by two orders of magnitude and, beyond inference, enables also training with TS of depths that were not feasible before. We train DQN agents from scratch using TS and show improvement in several Atari games compared to both the original DQN and the more advanced Rainbow. We will share the code upon publication.

----

## [422] Redesigning the Transformer Architecture with Insights from Multi-particle Dynamical Systems

**Authors**: *Subhabrata Dutta, Tanya Gautam, Soumen Chakrabarti, Tanmoy Chakraborty*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2bd388f731f26312bfc0fe30da009595-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2bd388f731f26312bfc0fe30da009595-Abstract.html)

**Abstract**:

The Transformer and its variants have been proven to be efficient sequence learners in many different domains. Despite their staggering success, a critical issue has been the enormous number of parameters that must be trained (ranging from $10^7$ to $10^{11}$) along with the quadratic complexity of dot-product attention. In this work, we investigate the problem of approximating the two central components of the Transformer --- multi-head self-attention and point-wise feed-forward transformation, with reduced parameter space and computational complexity. We build upon recent developments in analyzing deep neural networks as numerical solvers of ordinary differential equations. Taking advantage of an analogy between Transformer stages and the evolution of a dynamical system of multiple interacting particles, we formulate a temporal evolution scheme, \name, to bypass costly dot-product attention over multiple stacked layers.  We perform exhaustive experiments with \name\ on well-known encoder-decoder as well as encoder-only tasks. We observe that the degree of approximation (or inversely, the degree of parameter reduction) has different effects on the performance, depending on the task. While in the encoder-decoder regime, \name\ delivers performances comparable to the original Transformer, in encoder-only tasks it consistently outperforms Transformer along with several subsequent variants.

----

## [423] Exploring Architectural Ingredients of Adversarially Robust Deep Neural Networks

**Authors**: *Hanxun Huang, Yisen Wang, Sarah M. Erfani, Quanquan Gu, James Bailey, Xingjun Ma*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2bd7f907b7f5b6bbd91822c0c7b835f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2bd7f907b7f5b6bbd91822c0c7b835f6-Abstract.html)

**Abstract**:

Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks. A range of defense methods have been proposed to train adversarially robust DNNs, among which adversarial training has demonstrated promising results. However, despite preliminary understandings developed for adversarial training, it is still not clear, from the architectural perspective, what configurations can lead to more robust DNNs. In this paper, we address this gap via a comprehensive investigation on the impact of network width and depth on the robustness of adversarially trained DNNs. Specifically, we make the following key observations: 1) more parameters (higher model capacity) does not necessarily help adversarial robustness; 2) reducing capacity at the last stage (the last group of blocks) of the network can actually improve adversarial robustness; and 3) under the same parameter budget, there exists an optimal architectural configuration for adversarial robustness. We also provide a theoretical analysis explaning why such network configuration can help robustness. These architectural insights can help design adversarially robust DNNs.

----

## [424] Center Smoothing: Certified Robustness for Networks with Structured Outputs

**Authors**: *Aounon Kumar, Tom Goldstein*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2be8328f41144106f7144802f2367487-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2be8328f41144106f7144802f2367487-Abstract.html)

**Abstract**:

The study of provable adversarial robustness has mostly been limited to classification tasks and models with one-dimensional real-valued outputs. We extend the scope of certifiable robustness to problems with more general and structured outputs like sets, images, language, etc. We model the output space as a metric space under a distance/similarity function, such as intersection-over-union, perceptual similarity, total variation distance, etc. Such models are used in many machine learning problems like image segmentation, object detection, generative models, image/audio-to-text systems, etc. Based on a robustness technique called randomized smoothing, our center smoothing procedure can produce models with the guarantee that the change in the output, as measured by the distance metric, remains small for any norm-bounded adversarial perturbation of the input. We apply our method to create certifiably robust models with disparate output spaces -- from sets to images -- and show that it yields meaningful certificates without significantly degrading the performance of the base model.

----

## [425] Breaking the Linear Iteration Cost Barrier for Some Well-known Conditional Gradient Methods Using MaxIP Data-structures

**Authors**: *Zhaozhuo Xu, Zhao Song, Anshumali Shrivastava*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2c27a260f16ad3098393cc529f391f4a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2c27a260f16ad3098393cc529f391f4a-Abstract.html)

**Abstract**:

Conditional gradient methods (CGM) are widely used in modern machine learning. CGM's overall running time usually consists of two parts: the number of iterations and the cost of each iteration. Most efforts focus on reducing the number of iterations as a means to reduce the overall running time. In this work, we focus on improving the per iteration cost of CGM. The bottleneck step in most CGM is maximum inner product search (MaxIP), which requires a linear scan over the parameters.  In practice, approximate MaxIP data-structures are found to be helpful heuristics. However, theoretically, nothing is known about the combination of approximate MaxIP data-structures and CGM. In this work, we answer this question positively by providing a formal framework to combine the locality sensitive hashing type approximate MaxIP data-structures with CGM algorithms.  As a result, we show the first algorithm, where the cost per iteration is sublinear in the number of parameters, for many fundamental optimization algorithms, e.g., Frank-Wolfe, Herding algorithm, and policy gradient.

----

## [426] Neural Regression, Representational Similarity, Model Zoology & Neural Taskonomy at Scale in Rodent Visual Cortex

**Authors**: *Colin Conwell, David Mayo, Andrei Barbu, Michael A. Buice, George Alvarez, Boris Katz*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2c29d89cc56cdb191c60db2f0bae796b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2c29d89cc56cdb191c60db2f0bae796b-Abstract.html)

**Abstract**:

How well do deep neural networks fare as models of mouse visual cortex? A majority of research to date suggests results far more mixed than those produced in the modeling of primate visual cortex. Here, we perform a large-scale benchmarking of dozens of deep neural network models in mouse visual cortex with both representational similarity analysis and neural regression. Using the Allen Brain Observatory's 2-photon calcium-imaging dataset of activity in over 6,000 reliable rodent visual cortical neurons recorded in response to natural scenes, we replicate previous findings and resolve previous discrepancies, ultimately demonstrating that modern neural networks can in fact be used to explain activity in the mouse visual cortex to a more reasonable degree than previously suggested. Using our benchmark as an atlas, we offer preliminary answers to overarching questions about levels of analysis (e.g. do models that better predict the representations of individual neurons also predict representational similarity across neural populations?); questions about the properties of models that best predict the visual system overall (e.g. is convolution or category-supervision necessary to better predict neural activity?); and questions about the mapping between biological and artificial representations (e.g. does the information processing hierarchy in deep nets match the anatomical hierarchy of mouse visual cortex?). Along the way, we catalogue a number of models (including vision transformers, MLP-Mixers, normalization free networks, Taskonomy encoders and self-supervised models) outside the traditional circuit of convolutional object recognition. Taken together, our results provide a reference point for future ventures in the deep neural network modeling of mouse visual cortex, hinting at novel combinations of mapping method, architecture, and task to more fully characterize the computational motifs of visual representation in a species so central to neuroscience, but with a perceptual physiology and ecology markedly different from the ones we study in primates.

----

## [427] A Topological Perspective on Causal Inference

**Authors**: *Duligur Ibeling, Thomas Icard*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2c463dfdde588f3bfc60d53118c10d6b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2c463dfdde588f3bfc60d53118c10d6b-Abstract.html)

**Abstract**:

This paper presents a topological learning-theoretic perspective on causal inference by introducing a series of topologies defined on general spaces of structural causal models (SCMs). As an illustration of the framework we prove a topological causal hierarchy theorem, showing that substantive assumption-free causal inference is possible only in a meager set of SCMs. Thanks to a known correspondence between open sets in the weak topology and statistically verifiable hypotheses, our results show that inductive assumptions sufficient to license valid causal inferences are statistically unverifiable in principle. Similar to no-free-lunch theorems for statistical inference, the present results clarify the inevitability of substantial assumptions for causal inference. An additional benefit of our topological approach is that it easily accommodates SCMs with infinitely many variables. We finally suggest that our framework may be helpful for the positive project of exploring and assessing alternative causal-inductive assumptions.

----

## [428] Parameter Inference with Bifurcation Diagrams

**Authors**: *Gregory Szép, Neil Dalchau, Attila Csikász-Nagy*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2c6ae45a3e88aee548c0714fad7f8269-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2c6ae45a3e88aee548c0714fad7f8269-Abstract.html)

**Abstract**:

Estimation of parameters in differential equation models can be achieved by applying learning algorithms to quantitative time-series data. However, sometimes it is only possible to measure qualitative changes of a system in response to a controlled condition. In dynamical systems theory, such change points are known as bifurcations and lie on a function of the controlled condition called the bifurcation diagram. In this work, we propose a gradient-based approach for inferring the parameters of differential equations that produce a user-specified bifurcation diagram. The cost function contains an error term that is minimal when the model bifurcations match the specified targets and a bifurcation measure which has gradients that push optimisers towards bifurcating parameter regimes. The gradients can be computed without the need to differentiate through the operations of the solver that was used to compute the diagram. We demonstrate parameter inference with minimal models which explore the space of saddle-node and pitchfork diagrams and the genetic toggle switch from synthetic biology. Furthermore, the cost landscape allows us to organise models in terms of topological and geometric equivalence.

----

## [429] Scalable Thompson Sampling using Sparse Gaussian Process Models

**Authors**: *Sattar Vakili, Henry B. Moss, Artem Artemev, Vincent Dutordoir, Victor Picheny*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2c7f9ccb5a39073e24babc3a4cb45e60-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2c7f9ccb5a39073e24babc3a4cb45e60-Abstract.html)

**Abstract**:

Thompson Sampling (TS) from Gaussian Process (GP) models is a powerful tool for the optimization of black-box functions. Although TS enjoys strong theoretical guarantees and convincing empirical performance, it incurs a large computational overhead that scales polynomially with the optimization budget. Recently, scalable TS methods based on sparse GP models have been proposed to increase the scope of TS, enabling its application to problems that are sufficiently multi-modal, noisy or combinatorial to require more than a few hundred evaluations to be solved. However, the approximation error introduced by sparse GPs invalidates all existing regret bounds. In this work, we perform a theoretical and empirical analysis of scalable TS. We provide theoretical guarantees and show that the drastic reduction in computational complexity of scalable TS can be enjoyed without loss in the regret performance over the standard TS. These conceptual claims are validated for practical implementations of scalable TS on synthetic benchmarks and as part of a real-world high-throughput molecular design task.

----

## [430] Robust Counterfactual Explanations on Graph Neural Networks

**Authors**: *Mohit Bajaj, Lingyang Chu, Zi Yu Xue, Jian Pei, Lanjun Wang, Peter Cho-Ho Lam, Yong Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2c8c3a57383c63caef6724343eb62257-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2c8c3a57383c63caef6724343eb62257-Abstract.html)

**Abstract**:

Massive deployment of Graph Neural Networks (GNNs) in high-stake applications generates a strong demand for explanations that are robust to noise and align well with human intuition. Most existing methods generate explanations by identifying a subgraph of an input graph that has a strong correlation with the prediction. These explanations are not robust to noise because independently optimizing the correlation for a single input can easily overfit noise. Moreover, they are not counterfactual because removing an identified subgraph from an input graph does not necessarily change the prediction result. In this paper, we propose a novel method to generate robust counterfactual explanations on GNNs by explicitly modelling the common decision logic of GNNs on similar input graphs. Our explanations are naturally robust to noise because they are produced from the common decision boundaries of a GNN that govern the predictions of many similar input graphs. The explanations are also counterfactual because removing the set of edges identified by an explanation from the input graph changes the prediction significantly. Exhaustive experiments on many public datasets demonstrate the superior performance of our method.

----

## [431] Similarity and Matching of Neural Network Representations

**Authors**: *Adrián Csiszárik, Péter Korösi-Szabó, Ákos K. Matszangosz, Gergely Papp, Dániel Varga*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2cb274e6ce940f47beb8011d8ecb1462-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2cb274e6ce940f47beb8011d8ecb1462-Abstract.html)

**Abstract**:

We employ a toolset --- dubbed Dr. Frankenstein --- to analyse the similarity of representations in deep neural networks. With this toolset we aim to match the activations on given layers of two trained neural networks by joining them with a stitching layer. We demonstrate that the inner representations emerging in deep convolutional neural networks with the same architecture but different initialisations can be matched with a surprisingly high degree of accuracy even with a single, affine stitching layer. We choose the stitching layer from several possible classes of linear transformations and investigate their performance and properties. The task of matching representations is closely related to notions of similarity. Using this toolset we also provide a novel viewpoint on the current line of research regarding similarity indices of neural network representations: the perspective of the performance on a task.

----

## [432] DOCTOR: A Simple Method for Detecting Misclassification Errors

**Authors**: *Federica Granese, Marco Romanelli, Daniele Gorla, Catuscia Palamidessi, Pablo Piantanida*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2cb6b10338a7fc4117a80da24b582060-Abstract.html)

**Abstract**:

Deep neural networks (DNNs) have shown to perform very well on large scale object recognition problems and lead to widespread use for real-world applications, including situations where DNN are implemented as “black boxes”.  A promising approach to secure their use is to accept decisions that are likely to be correct while discarding the others.  In this work, we propose DOCTOR, a simple method that aims to identify whether the prediction of a DNN classifier should (or should not) be trusted so that, consequently, it would be possible to accept it or to reject it. Two scenarios are investigated: Totally Black Box (TBB) where only the soft-predictions are available and Partially Black Box (PBB) where gradient-propagation to perform input pre-processing is allowed. Empirically, we show that DOCTOR outperforms all state-of-the-art methods on various well-known images and sentiment analysis datasets. In particular, we observe a reduction of up to 4% of the false rejection rate (FRR) in the PBB scenario. DOCTOR can be applied to any pre-trained model, it does not require prior information about the underlying dataset and is as simple as the simplest available methods in the literature.

----

## [433] Contrastive Laplacian Eigenmaps

**Authors**: *Hao Zhu, Ke Sun, Peter Koniusz*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d1b2a5ff364606ff041650887723470-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d1b2a5ff364606ff041650887723470-Abstract.html)

**Abstract**:

Graph contrastive learning attracts/disperses node representations for similar/dissimilar node pairs under some notion of similarity. It may be combined with a low-dimensional embedding of nodes to preserve intrinsic and structural properties of a graph. In this paper, we extend the celebrated Laplacian Eigenmaps with contrastive learning, and call them COntrastive Laplacian EigenmapS (COLES). Starting from a GAN-inspired contrastive formulation, we show that the Jensen-Shannon divergence underlying many contrastive graph embedding models fails under disjoint positive and negative distributions, which may naturally emerge during sampling in the contrastive setting. In contrast, we demonstrate analytically that COLES essentially minimizes a surrogate of Wasserstein distance, which is known to cope well under disjoint distributions. Moreover, we show that the loss of COLES belongs to the family of so-called block-contrastive losses, previously shown to be superior compared to pair-wise losses typically used by contrastive methods. We show on popular benchmarks/backbones that COLES offers favourable accuracy/scalability compared to DeepWalk, GCN, Graph2Gauss, DGI and GRACE baselines.

----

## [434] Machine learning structure preserving brackets for forecasting irreversible processes

**Authors**: *Kookjin Lee, Nathaniel Trask, Panos Stinis*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d1bcedd27b586d2a9562a0f8e076b41-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d1bcedd27b586d2a9562a0f8e076b41-Abstract.html)

**Abstract**:

Forecasting of time-series data requires imposition of inductive biases to obtain predictive extrapolation, and recent works have imposed Hamiltonian/Lagrangian form to preserve structure for systems with \emph{reversible} dynamics. In this work we present a novel parameterization of dissipative brackets from metriplectic dynamical systems appropriate for learning \emph{irreversible} dynamics with unknown a priori model form. The process learns generalized Casimirs for energy and entropy guaranteed to be conserved and nondecreasing, respectively. Furthermore, for the case of added thermal noise, we guarantee exact preservation of a fluctuation-dissipation theorem, ensuring thermodynamic consistency. We provide benchmarks for dissipative systems demonstrating learned dynamics are more robust and generalize better than either "black-box" or penalty-based approaches.

----

## [435] On the Variance of the Fisher Information for Deep Learning

**Authors**: *Alexander Soen, Ke Sun*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d290e496d16c9dcaa9b4ded5cac10cc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d290e496d16c9dcaa9b4ded5cac10cc-Abstract.html)

**Abstract**:

In the realm of deep learning, the Fisher information matrix (FIM) gives novel insights and useful tools to characterize the loss landscape, perform second-order optimization, and build geometric learning theories. The exact FIM is either unavailable in closed form or too expensive to compute. In practice, it is almost always estimated based on empirical samples.  We investigate two such estimators based on two equivalent representations of the FIM --- both unbiased and consistent. Their estimation quality is naturally gauged by their variance given in closed form. We analyze how the parametric structure of a deep neural network can affect the variance. The meaning of this variance measure and its upper bounds are then discussed in the context of deep learning.

----

## [436] A$^2$-Net: Learning Attribute-Aware Hash Codes for Large-Scale Fine-Grained Image Retrieval

**Authors**: *Xiu-Shen Wei, Yang Shen, Xuhao Sun, Han-Jia Ye, Jian Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d3acd3e240c61820625fff66a19938f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d3acd3e240c61820625fff66a19938f-Abstract.html)

**Abstract**:

Our work focuses on tackling large-scale fine-grained image retrieval as ranking the images depicting the concept of interests (i.e., the same sub-category labels) highest based on the fine-grained details in the query. It is desirable to alleviate the challenges of both fine-grained nature of small inter-class variations with large intra-class variations and explosive growth of fine-grained data for such a practical task. In this paper, we propose an Attribute-Aware hashing Network (A$^2$-Net) for generating attribute-aware hash codes to not only make the retrieval process efficient, but also establish explicit correspondences between hash codes and visual attributes. Specifically, based on the captured visual representations by attention, we develop an encoder-decoder structure network of a reconstruction task to unsupervisedly distill high-level attribute-specific vectors from the appearance-specific visual representations without attribute annotations. A$^2$-Net is also equipped with a feature decorrelation constraint upon these attribute vectors to enhance their representation abilities. Finally, the required hash codes are generated by the attribute vectors driven by preserving original similarities. Qualitative experiments on five benchmark fine-grained datasets show our superiority over competing methods. More importantly, quantitative results demonstrate the obtained hash codes can strongly correspond to certain kinds of crucial properties of fine-grained objects.

----

## [437] Shape Registration in the Time of Transformers

**Authors**: *Giovanni Trappolini, Luca Cosmo, Luca Moschella, Riccardo Marin, Simone Melzi, Emanuele Rodolà*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d3d9d5373f378108cdbd30a3c52bd3e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d3d9d5373f378108cdbd30a3c52bd3e-Abstract.html)

**Abstract**:

In this paper, we propose a transformer-based procedure for the efficient registration of non-rigid 3D point clouds. The proposed approach is data-driven and adopts for the first time the transformers architecture in the registration task. Our method is general and applies to different settings. Given a fixed template with some desired properties (e.g. skinning weights or other animation cues), we can register raw acquired data to it, thereby transferring all the template properties to the input geometry. Alternatively, given a pair of shapes, our method can register the first onto the second (or vice-versa), obtaining a high-quality dense correspondence between the two.In both contexts, the quality of our results enables us to target real applications such as texture transfer and shape interpolation.Furthermore, we also show that including an estimation of the underlying density of the surface eases the learning process. By exploiting the potential of this architecture, we can train our model requiring only a sparse set of ground truth correspondences ($10\sim20\%$ of the total points). The proposed model and the analysis that we perform pave the way for future exploration of transformer-based architectures for registration and matching applications. Qualitative and quantitative evaluations demonstrate that our pipeline outperforms state-of-the-art methods for deformable and unordered 3D data registration on different datasets and scenarios.

----

## [438] Brick-by-Brick: Combinatorial Construction with Deep Reinforcement Learning

**Authors**: *Hyunsoo Chung, Jungtaek Kim, Boris Knyazev, Jinhwi Lee, Graham W. Taylor, Jaesik Park, Minsu Cho*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d4027d6df9c0256b8d4474ce88f8c88-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d4027d6df9c0256b8d4474ce88f8c88-Abstract.html)

**Abstract**:

Discovering a solution in a combinatorial space is prevalent in many real-world problems but it is also challenging due to diverse complex constraints and the vast number of possible combinations. To address such a problem, we introduce a novel formulation, combinatorial construction, which requires a building agent to assemble unit primitives (i.e., LEGO bricks) sequentially -- every connection between two bricks must follow a fixed rule, while no bricks mutually overlap. To construct a target object, we provide incomplete knowledge about the desired target (i.e., 2D images) instead of exact and explicit volumetric information to the agent. This problem requires a comprehensive understanding of partial information and long-term planning to append a brick sequentially, which leads us to employ reinforcement learning. The approach has to consider a variable-sized action space where a large number of invalid actions, which would cause overlap between bricks, exist. To resolve these issues, our model, dubbed Brick-by-Brick, adopts an action validity prediction network that efficiently filters invalid actions for an actor-critic network. We demonstrate that the proposed method successfully learns to construct an unseen object conditioned on a single image or multiple views of a target object.

----

## [439] Dissecting the Diffusion Process in Linear Graph Convolutional Networks

**Authors**: *Yifei Wang, Yisen Wang, Jiansheng Yang, Zhouchen Lin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d95666e2649fcfc6e3af75e09f5adb9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d95666e2649fcfc6e3af75e09f5adb9-Abstract.html)

**Abstract**:

Graph Convolutional Networks (GCNs) have attracted more and more attentions in recent years. A typical GCN layer consists of a linear feature propagation step and a nonlinear transformation step. Recent works show that a linear GCN can achieve comparable performance to the original non-linear GCN while being much more computationally efficient. In this paper, we dissect the feature propagation steps of linear GCNs from a perspective of continuous graph diffusion, and analyze why linear GCNs fail to benefit from more propagation steps. Following that, we propose Decoupled Graph Convolution (DGC) that decouples the terminal time and the feature propagation steps, making it more flexible and capable of exploiting a very large number of feature propagation steps. Experiments demonstrate that our proposed DGC improves linear GCNs by a large margin and makes them competitive with many modern variants of non-linear GCNs.

----

## [440] Dynamic Grained Encoder for Vision Transformers

**Authors**: *Lin Song, Songyang Zhang, Songtao Liu, Zeming Li, Xuming He, Hongbin Sun, Jian Sun, Nanning Zheng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2d969e2cee8cfa07ce7ca0bb13c7a36d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2d969e2cee8cfa07ce7ca0bb13c7a36d-Abstract.html)

**Abstract**:

Transformers, the de-facto standard for language modeling, have been recently applied for vision tasks. This paper introduces sparse queries for vision transformers to exploit the intrinsic spatial redundancy of natural images and save computational costs. Specifically, we propose a Dynamic Grained Encoder for vision transformers, which can adaptively assign a suitable number of queries to each spatial region. Thus it achieves a fine-grained representation in discriminative regions while keeping high efficiency. Besides, the dynamic grained encoder is compatible with most vision transformer frameworks. Without bells and whistles, our encoder allows the state-of-the-art vision transformers to reduce computational complexity by 40%-60% while maintaining comparable performance on image classification. Extensive experiments on object detection and segmentation further demonstrate the generalizability of our approach. Code is available at https://github.com/StevenGrove/vtpack.

----

## [441] Understanding Negative Samples in Instance Discriminative Self-supervised Representation Learning

**Authors**: *Kento Nozawa, Issei Sato*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2dace78f80bc92e6d7493423d729448e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2dace78f80bc92e6d7493423d729448e-Abstract.html)

**Abstract**:

Instance discriminative self-supervised representation learning has been attracted attention thanks to its unsupervised nature and informative feature representation for downstream tasks. In practice, it commonly uses a larger number of negative samples than the number of supervised classes. However, there is an inconsistency in the existing analysis; theoretically, a large number of negative samples degrade classification performance on a downstream supervised task, while empirically, they improve the performance. We provide a novel framework to analyze this empirical result regarding negative samples using the coupon collector's problem. Our bound can implicitly incorporate the supervised loss of the downstream task in the self-supervised loss by increasing the number of negative samples. We confirm that our proposed analysis holds on real-world benchmark datasets.

----

## [442] On UMAP's True Loss Function

**Authors**: *Sebastian Damrich, Fred A. Hamprecht*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)

**Abstract**:

UMAP has supplanted $t$-SNE as state-of-the-art for visualizing high-dimensional datasets in many disciplines, but the reason for its success is not well understood. In this work, we investigate UMAP's sampling based optimization scheme in detail. We derive UMAP's true loss function in closed form and find that it differs from the published one in a dataset size dependent way. As a consequence, we show that UMAP does not aim to reproduce its theoretically motivated high-dimensional UMAP similarities. Instead, it tries to reproduce  similarities that only encode the $k$ nearest neighbor graph, thereby challenging the previous understanding of UMAP's effectiveness. Alternatively, we consider the implicit balancing of attraction and repulsion due to the negative sampling to be key to UMAP's success. We corroborate our theoretical findings on toy and single cell RNA sequencing data.

----

## [443] Fast Pure Exploration via Frank-Wolfe

**Authors**: *Po-An Wang, Ruo-Chun Tzeng, Alexandre Proutière*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2dffbc474aa176b6dc957938c15d0c8b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2dffbc474aa176b6dc957938c15d0c8b-Abstract.html)

**Abstract**:

We study the problem of active pure exploration with fixed confidence in generic stochastic bandit environments. The goal of the learner is to answer a query about the environment with a given level of certainty while minimizing her sampling budget. For this problem, instance-specific lower bounds on the expected sample complexity reveal the optimal proportions of arm draws an Oracle algorithm would apply. These proportions solve an optimization problem whose tractability strongly depends on the structural properties of the environment, but may be instrumental in the design of efficient learning algorithms. We devise Frank-Wolfe-based Sampling (FWS), a simple algorithm whose sample complexity matches the lower bounds for a wide class of pure exploration problems. The algorithm is computationally efficient as, to learn and track the optimal proportion of arm draws, it relies on a single iteration of Frank-Wolfe algorithm applied to the lower-bound optimization problem. We apply FWS to various pure exploration tasks, including best arm identification in unstructured, thresholded, linear, and Lipschitz bandits. Despite its simplicity, FWS is competitive compared to state-of-art algorithms.

----

## [444] iFlow: Numerically Invertible Flows for Efficient Lossless Compression via a Uniform Coder

**Authors**: *Shifeng Zhang, Ning Kang, Tom Ryder, Zhenguo Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e3d2c4f33a7a1f58bc6c81cacd21e9c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e3d2c4f33a7a1f58bc6c81cacd21e9c-Abstract.html)

**Abstract**:

It was estimated that the world produced $59 ZB$ ($5.9 \times 10^{13} GB$) of data in 2020, resulting in the enormous costs of both data storage and transmission. Fortunately, recent advances in deep generative models have spearheaded a new class of so-called "neural compression" algorithms, which significantly outperform traditional codecs in terms of compression ratio. Unfortunately, the application of neural compression garners little commercial interest due to its limited bandwidth; therefore, developing highly efficient frameworks is of critical practical importance. In this paper, we discuss lossless compression using normalizing flows which have demonstrated a great capacity for achieving high compression ratios. As such, we introduce iFlow, a new method for achieving efficient lossless compression. We first propose Modular Scale Transform (MST) and a novel family of numerically invertible flow transformations based on MST. Then we introduce the Uniform Base Conversion System (UBCS), a fast uniform-distribution codec incorporated into iFlow, enabling efficient compression. iFlow achieves state-of-the-art compression ratios and is $5 \times$ quicker than other high-performance schemes. Furthermore, the techniques presented in this paper can be used to accelerate coding time for a broad class of flow-based algorithms.

----

## [445] History Aware Multimodal Transformer for Vision-and-Language Navigation

**Authors**: *Shizhe Chen, Pierre-Louis Guhur, Cordelia Schmid, Ivan Laptev*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e5c2cb8d13e8fba78d95211440ba326-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e5c2cb8d13e8fba78d95211440ba326-Abstract.html)

**Abstract**:

Vision-and-language navigation (VLN) aims to build autonomous visual agents that follow instructions and navigate in real scenes. To remember previously visited locations and actions taken, most approaches to VLN implement memory using recurrent states. Instead, we introduce a History Aware Multimodal Transformer (HAMT) to incorporate a long-horizon history into multimodal decision making. HAMT efficiently encodes all the past panoramic observations via a hierarchical vision transformer (ViT), which first encodes individual images with ViT, then models spatial relation between images in a panoramic observation and finally takes into account temporal relation between panoramas in the history. It, then, jointly combines text, history and current observation to predict the next action. We first train HAMT end-to-end using several proxy tasks including single step action prediction and spatial relation prediction, and then use reinforcement learning to further improve the navigation policy. HAMT achieves new state of the art on a broad range of VLN tasks, including VLN with fine-grained instructions (R2R, RxR), high-level instructions (R2R-Last, REVERIE), dialogs (CVDN) as well as long-horizon VLN (R4R, R2R-Back). We demonstrate HAMT to be particularly effective for navigation tasks with longer trajectories.

----

## [446] Meta Two-Sample Testing: Learning Kernels for Testing with Limited Data

**Authors**: *Feng Liu, Wenkai Xu, Jie Lu, Danica J. Sutherland*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e6d9c6052e99fcdfa61d9b9da273ca2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e6d9c6052e99fcdfa61d9b9da273ca2-Abstract.html)

**Abstract**:

Modern kernel-based two-sample tests have shown great success in distinguishing complex, high-dimensional distributions by learning appropriate kernels (or, as a special case, classifiers). Previous work, however, has assumed that many samples are observed from both of the distributions being distinguished. In realistic scenarios with very limited numbers of data samples, it can be challenging to identify a kernel powerful enough to distinguish complex distributions. We address this issue by introducing the problem of meta two-sample testing (M2ST), which aims to exploit (abundant) auxiliary data on related tasks to find an algorithm that can quickly identify a powerful test on new target tasks. We propose two specific algorithms for this task: a generic scheme which improves over baselines, and a more tailored approach which performs even better. We provide both theoretical justification and empirical evidence that our proposed meta-testing schemes outperform learning kernel-based tests directly from scarce observations, and identify when such schemes will be successful.

----

## [447] Process for Adapting Language Models to Society (PALMS) with Values-Targeted Datasets

**Authors**: *Irene Solaiman, Christy Dennison*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e855f9489df0712b4bd8ea9e2848c5a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e855f9489df0712b4bd8ea9e2848c5a-Abstract.html)

**Abstract**:

Language models can generate harmful and biased outputs and exhibit undesirable behavior according to a given cultural context. We propose a Process for Adapting Language Models to Society (PALMS) with Values-Targeted Datasets, an iterative process to significantly change model behavior by crafting and fine-tuning on a dataset that reflects a predetermined set of target values. We evaluate our process using three metrics: quantitative metrics with human evaluations that score output adherence to a target value, toxicity scoring on outputs; and qualitative metrics analyzing the most common word associated with a given social category. Through each iteration, we add additional training dataset examples based on observed shortcomings from evaluations. PALMS performs significantly better on all metrics compared to baseline and control models for a broad range of GPT-3 language model sizes without compromising capability integrity. We find that the effectiveness of PALMS increases with model size. We show that significantly adjusting language model behavior is feasible with a small, hand-curated dataset.

----

## [448] The Lazy Online Subgradient Algorithm is Universal on Strongly Convex Domains

**Authors**: *Daron Anderson, Douglas J. Leith*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e907f44e0a9616314cf3d964d4e3c93-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e907f44e0a9616314cf3d964d4e3c93-Abstract.html)

**Abstract**:

We study Online Lazy Gradient Descent for optimisation on a strongly convex domain. The algorithm is known to achieve  $O(\sqrt N)$ regret against adversarial opponents; here we show it is universal in the sense that it also achieves $O(\log N)$ expected regret against i.i.d opponents. This improves upon the more complex meta-algorithm of Huang et al \cite{FTLBall} that only gets $O(\sqrt {N \log N})$ and $ O(\log N)$ bounds. In addition  we show that, unlike  for the simplex, order bounds for pseudo-regret and expected regret are equivalent for strongly convex domains.

----

## [449] Computer-Aided Design as Language

**Authors**: *Yaroslav Ganin, Sergey Bartunov, Yujia Li, Ethan Keller, Stefano Saliceti*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e92962c0b6996add9517e4242ea9bdc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e92962c0b6996add9517e4242ea9bdc-Abstract.html)

**Abstract**:

Computer-Aided Design (CAD) applications are used in manufacturing to model everything from coffee mugs to sports cars. These programs are complex and require years of training and experience to master. A component of all CAD models particularly difficult to make are the highly structured 2D sketches that lie at the heart of every 3D construction. In this work, we propose a machine learning model capable of automatically generating such sketches. Through this, we pave the way for developing intelligent tools that would help engineers create better designs with less effort. The core of our method is a combination of a general-purpose language modeling technique alongside an off-the-shelf data serialization protocol. Additionally, we explore several extensions allowing us to gain finer control over the generation process. We show that our approach has enough flexibility to accommodate the complexity of the domain and performs well for both unconditional synthesis and image-to-sketch translation.

----

## [450] COHESIV: Contrastive Object and Hand Embedding Segmentation In Video

**Authors**: *Dandan Shan, Richard E. L. Higgins, David F. Fouhey*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e976ab88a42d723d9f2ee6027b707f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e976ab88a42d723d9f2ee6027b707f5-Abstract.html)

**Abstract**:

In this paper we learn to segment hands and hand-held objects from motion. Our system takes a single RGB image and hand location as input to segment the hand and hand-held object. For learning, we generate responsibility maps that show how well a hand's motion explains other pixels' motion in video. We use these responsibility maps as pseudo-labels to train a weakly-supervised neural network using an attention-based similarity loss and contrastive loss. Our system outperforms alternate methods, achieving good performance on the 100DOH, EPIC-KITCHENS, and HO3D datasets.

----

## [451] ByPE-VAE: Bayesian Pseudocoresets Exemplar VAE

**Authors**: *Qingzhong Ai, Lirong He, Shiyu Liu, Zenglin Xu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2e9f978b222a956ba6bdf427efbd9ab3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2e9f978b222a956ba6bdf427efbd9ab3-Abstract.html)

**Abstract**:

Recent studies show that advanced priors play a major role in deep generative models. Exemplar VAE, as a variant of VAE with an exemplar-based prior, has achieved impressive results. However, due to the nature of model design, an exemplar-based model usually requires vast amounts of data to participate in training, which leads to huge computational complexity. To address this issue, we propose Bayesian Pseudocoresets Exemplar VAE (ByPE-VAE), a new variant of VAE with a prior based on Bayesian pseudocoreset. The proposed prior is conditioned on a small-scale pseudocoreset rather than the whole dataset for reducing the computational cost and avoiding overfitting. Simultaneously, we obtain the optimal pseudocoreset via a stochastic optimization algorithm during VAE training aiming to minimize the Kullback-Leibler divergence between the prior based on the pseudocoreset and that based on the whole dataset. Experimental results show that ByPE-VAE can achieve competitive improvements over the state-of-the-art VAEs in the tasks of density estimation, representation learning, and generative data augmentation. Particularly, on a basic VAE architecture, ByPE-VAE is up to 3 times faster than Exemplar VAE while almost holding the performance. Code is available at \url{https://github.com/Aiqz/ByPE-VAE}.

----

## [452] Recovery Analysis for Plug-and-Play Priors using the Restricted Eigenvalue Condition

**Authors**: *Jiaming Liu, M. Salman Asif, Brendt Wohlberg, Ulugbek Kamilov*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2ea1202aed1e0ce30d41be4919b0cc99-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2ea1202aed1e0ce30d41be4919b0cc99-Abstract.html)

**Abstract**:

The plug-and-play priors (PnP) and regularization by denoising (RED) methods have become widely used for solving inverse problems by leveraging pre-trained deep denoisers as image priors.  While the empirical imaging performance and the theoretical convergence properties of these algorithms have been widely investigated, their recovery properties have not previously been theoretically analyzed.  We address this gap by showing how to establish theoretical recovery guarantees for PnP/RED by assuming that the solution of these methods lies near the fixed-points of a deep neural network. We also present numerical results comparing the recovery performance of PnP/RED in compressive sensing against that of recent compressive sensing algorithms based on generative models. Our numerical results suggest that PnP with a pre-trained artifact removal network provides significantly better results compared to the existing state-of-the-art methods.

----

## [453] Group Equivariant Subsampling

**Authors**: *Jin Xu, Hyunjik Kim, Thomas Rainforth, Yee Whye Teh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2ea6241cf767c279cf1e80a790df1885-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2ea6241cf767c279cf1e80a790df1885-Abstract.html)

**Abstract**:

Subsampling is used in convolutional neural networks (CNNs) in the form of pooling or strided convolutions, to reduce the spatial dimensions of feature maps and to allow the receptive fields to grow exponentially with depth. However, it is known that such subsampling operations are not translation equivariant, unlike convolutions that are translation equivariant. Here, we first introduce translation equivariant subsampling/upsampling layers that can be used to construct exact translation equivariant CNNs. We then generalise these layers beyond translations to general groups, thus proposing group equivariant subsampling/upsampling. We use these layers to construct group equivariant autoencoders (GAEs) that allow us to learn low-dimensional equivariant representations. We empirically verify on images that the representations are indeed equivariant to input translations and rotations, and thus generalise well to unseen positions and orientations. We further use GAEs in models that learn object-centric representations on multi-object datasets, and show improved data efficiency and decomposition compared to non-equivariant baselines.

----

## [454] Data Sharing and Compression for Cooperative Networked Control

**Authors**: *Jiangnan Cheng, Marco Pavone, Sachin Katti, Sandeep Chinchali, Ao Tang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2eb5657d37f474e4c4cf01e4882b8962-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2eb5657d37f474e4c4cf01e4882b8962-Abstract.html)

**Abstract**:

Sharing forecasts of network timeseries data, such as cellular or electricity load patterns, can improve independent control applications ranging from traffic scheduling to power generation. Typically, forecasts are designed without knowledge of a downstream controller's task objective, and thus simply optimize for mean prediction error. However, such task-agnostic representations are often too large to stream over a communication network and do not emphasize salient temporal features for cooperative control. This paper presents a solution to learn succinct, highly-compressed forecasts that are co-designed with a modular controller's task objective. Our simulations with real cellular, Internet-of-Things (IoT), and electricity load data show we can improve a model predictive controller's performance by at least 25% while transmitting 80% less data than the competing method. Further, we present theoretical compression results for a networked variant of the classical linear quadratic regulator (LQR) control problem.

----

## [455] Hyperbolic Procrustes Analysis Using Riemannian Geometry

**Authors**: *Ya-Wei Eileen Lin, Yuval Kluger, Ronen Talmon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2ed80f6311c1825feb854d78fa969d34-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2ed80f6311c1825feb854d78fa969d34-Abstract.html)

**Abstract**:

Label-free alignment between datasets collected at different times, locations, or by different instruments is a fundamental scientific task. Hyperbolic spaces have recently provided a fruitful foundation for the development of informative representations of hierarchical data. Here, we take a purely geometric approach for label-free alignment of hierarchical datasets and introduce hyperbolic Procrustes analysis (HPA). HPA consists of new implementations of the three prototypical Procrustes analysis components: translation, scaling, and rotation, based on the Riemannian geometry of the Lorentz model of hyperbolic space. We analyze the proposed components, highlighting their useful properties for alignment. The efficacy of HPA, its theoretical properties, stability and computational efficiency are demonstrated in simulations. In addition, we showcase its performance on three batch correction tasks involving gene expression and mass cytometry data. Specifically, we demonstrate high-quality unsupervised batch effect removal from data acquired at different sites and with different technologies that outperforms recent methods for label-free alignment in hyperbolic spaces.

----

## [456] No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data

**Authors**: *Mi Luo, Fei Chen, Dapeng Hu, Yifan Zhang, Jian Liang, Jiashi Feng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html)

**Abstract**:

A central challenge in training classification models in the real-world federated system is learning with non-IID data. To cope with this, most of the existing works involve enforcing regularization in local optimization or improving the model aggregation scheme at the server. Other works also share public datasets or synthesized samples to supplement the training of under-represented classes or introduce a certain level of personalization. Though effective, they lack a deep understanding of how the data heterogeneity affects each layer of a deep classification model. In this paper, we bridge this gap by performing an experimental analysis of the representations learned by different layers. Our observations are surprising: (1) there exists a greater bias in the classifier than other layers, and (2) the classification performance can be significantly improved by post-calibrating the classifier after federated training. Motivated by the above findings, we propose a novel and simple algorithm called Classifier Calibration with Virtual Representations (CCVR), which adjusts the classifier using virtual representations sampled from an approximated gaussian mixture model. Experimental results demonstrate that CCVR achieves state-of-the-art performance on popular federated learning benchmarks including CIFAR-10, CIFAR-100, and CINIC-10. We hope that our simple yet effective method can shed some light on the future research of federated learning with non-IID data.

----

## [457] Preconditioned Gradient Descent for Over-Parameterized Nonconvex Matrix Factorization

**Authors**: *Jialun Zhang, Salar Fattahi, Richard Y. Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2f2cd5c753d3cee48e47dbb5bbaed331-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2f2cd5c753d3cee48e47dbb5bbaed331-Abstract.html)

**Abstract**:

In practical instances of nonconvex matrix factorization, the rank of the true solution $r^{\star}$ is often unknown, so the rank $r$of the model can be over-specified as $r>r^{\star}$. This over-parameterized regime of matrix factorization significantly slows down the convergence of local search algorithms, from a linear rate with $r=r^{\star}$ to a sublinear rate when $r>r^{\star}$. We propose an inexpensive preconditioner for the matrix sensing variant of nonconvex matrix factorization that restores the convergence rate of gradient descent back to linear, even in the over-parameterized case, while also making it agnostic to possible ill-conditioning in the ground truth. Classical gradient descent in a neighborhood of the solution slows down due to the need for the model matrix factor to become singular. Our key result is that this singularity can be corrected by $\ell_{2}$ regularization with a specific range of values for the damping parameter. In fact, a good damping parameter can be inexpensively estimated from the current iterate. The resulting algorithm, which we call preconditioned gradient descent or PrecGD, is stable under noise, and converges linearly to an information theoretically optimal error bound. Our numerical experiments find that PrecGD works equally well in restoring the linear convergence of other variants of nonconvex matrix factorization in the over-parameterized regime.

----

## [458] Improving Contrastive Learning on Imbalanced Data via Open-World Sampling

**Authors**: *Ziyu Jiang, Tianlong Chen, Ting Chen, Zhangyang Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2f37d10131f2a483a8dd005b3d14b0d9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2f37d10131f2a483a8dd005b3d14b0d9-Abstract.html)

**Abstract**:

Contrastive learning approaches have achieved great success in learning visual representations with few labels of the target classes. That implies a tantalizing possibility of scaling them up beyond a curated “seed" benchmark, to incorporating more unlabeled images from the internet-scale external sources to enhance its performance. However, in practice, larger amount of unlabeled data will require more computing resources due to the bigger model size and longer training needed. Moreover, open-world unlabeled data usually follows an implicit long-tail class or attribute distribution, many of which also do not belong to the target classes. Blindly leveraging all unlabeled data hence can lead to the data imbalance as well as distraction issues. This motivates us to seek a principled approach to strategically select unlabeled data from an external source, in order to learn generalizable, balanced and diverse representations for relevant classes. In this work, we present an open-world unlabeled data sampling framework called Model-Aware K-center (MAK), which follows three simple principles: (1) tailness, which encourages sampling of examples from tail classes, by sorting the empirical contrastive loss expectation (ECLE) of samples over random data augmentations; (2) proximity, which rejects the out-of-distribution outliers that may distract training; and (3) diversity, which ensures diversity in the set of sampled examples. Empirically, using ImageNet-100-LT (without labels) as the seed dataset and two “noisy” external data sources, we demonstrate that MAK can consistently improve both the overall representation quality and the class balancedness of the learned features, as evaluated via linear classifier evaluation on full-shot and few-shot settings. Thecode is available at: https://github.com/VITA-Group/MAK.

----

## [459] Searching for Efficient Transformers for Language Modeling

**Authors**: *David R. So, Wojciech Manke, Hanxiao Liu, Zihang Dai, Noam Shazeer, Quoc V. Le*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2f3c6a4cd8af177f6456e7e51a916ff3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2f3c6a4cd8af177f6456e7e51a916ff3-Abstract.html)

**Abstract**:

Large Transformer models have been central to recent advances in natural language processing. The training and inference costs of these models, however, have grown rapidly and become prohibitively expensive. Here we aim to reduce the costs of Transformers by searching for a more efficient variant. Compared to previous approaches, our search is performed at a lower level, over the primitives that define a Transformer TensorFlow program. We identify an architecture, named Primer, that has a smaller training cost than the original Transformer and other variants for auto-regressive language modeling. Primer’s improvements can be mostly attributed to two simple modifications: squaring ReLU activations and adding a depthwise convolution layer after each Q, K, and V projection in self-attention.Experiments show Primer’s gains over Transformer increase as compute scale grows and follow a power law with respect to quality at optimal model sizes. We also verify empirically that Primer can be dropped into different codebases to significantly speed up training without additional tuning. For example, at a 500M parameter size, Primer improves the original T5 architecture on C4 auto-regressive language modeling, reducing the training cost by 4X. Furthermore, the reduced training cost means Primer needs much less compute to reach a target one-shot performance. For instance, in a 1.9B parameter configuration similar to GPT-3 XL, Primer uses 1/3 of the training compute to achieve the same one-shot performance as Transformer. We open source our models and several comparisons in T5 to help with reproducibility.

----

## [460] Scaling Ensemble Distribution Distillation to Many Classes with Proxy Targets

**Authors**: *Max Ryabinin, Andrey Malinin, Mark J. F. Gales*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2f4ccb0f7a84f335affb418aee08a6df-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2f4ccb0f7a84f335affb418aee08a6df-Abstract.html)

**Abstract**:

Ensembles of machine learning models yield improved system performance as well as robust and interpretable uncertainty estimates; however, their inference costs can be prohibitively high. Ensemble Distribution Distillation (EnD$^2$) is an approach that allows a single model to efficiently capture both the predictive performance and uncertainty estimates of an ensemble. For classification, this is achieved by training a Dirichlet distribution over the ensemble members' output distributions via the maximum likelihood criterion. Although theoretically principled, this work shows that the criterion exhibits poor convergence when applied to large-scale tasks where the number of classes is very high. Specifically, we show that for the Dirichlet log-likelihood criterion classes with low probability induce larger gradients than high-probability classes. Hence during training the model focuses on the distribution of the ensemble tail-class probabilities rather than the probability of the correct and closely related classes. We propose a new training objective which minimizes the reverse KL-divergence to a \emph{Proxy-Dirichlet} target derived from the ensemble. This loss resolves the gradient issues of EnD$^2$, as we demonstrate both theoretically and empirically on the ImageNet, LibriSpeech, and WMT17 En-De datasets containing 1000, 5000, and 40,000 classes, respectively.

----

## [461] Multi-Person 3D Motion Prediction with Multi-Range Transformers

**Authors**: *Jiashun Wang, Huazhe Xu, Medhini Narasimhan, Xiaolong Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/2fd5d41ec6cfab47e32164d5624269b1-Abstract.html)

**Abstract**:

We propose a novel framework for multi-person 3D motion trajectory prediction. Our key observation is that a human's action and behaviors may highly depend on the other persons around. Thus, instead of predicting each human pose trajectory in isolation, we introduce a Multi-Range Transformers model which contains of a local-range encoder for individual motion and a global-range encoder for social interactions. The Transformer decoder then performs prediction for each person by taking a corresponding pose as a query which attends to both local and global-range encoder features. Our model not only outperforms state-of-the-art methods on long-term 3D motion prediction, but also generates diverse social interactions. More interestingly, our model can even predict 15-person motion simultaneously by automatically dividing the persons into different interaction groups.  Project page with code is available at https://jiashunwang.github.io/MRT/.

----

## [462] STEM: A Stochastic Two-Sided Momentum Algorithm Achieving Near-Optimal Sample and Communication Complexities for Federated Learning

**Authors**: *Prashant Khanduri, Pranay Sharma, Haibo Yang, Mingyi Hong, Jia Liu, Ketan Rajawat, Pramod K. Varshney*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3016a447172f3045b65f5fc83e04b554-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3016a447172f3045b65f5fc83e04b554-Abstract.html)

**Abstract**:

Federated Learning (FL) refers to the paradigm where multiple worker nodes (WNs) build a joint model by using local data. Despite extensive research, for a generic non-convex FL problem, it is not clear, how to choose the WNs' and the server's update directions, the minibatch sizes, and the local update frequency, so that the WNs use the minimum number of samples and communication rounds to achieve the desired solution. This work addresses the above question and considers a class of stochastic algorithms where the WNs perform a few local updates before communication. We show that when both the WN's and the server's directions are chosen based on certain stochastic momentum estimator, the algorithm requires $\tilde{\mathcal{O}}(\epsilon^{-3/2})$ samples and $\tilde{\mathcal{O}}(\epsilon^{-1})$ communication rounds to compute an $\epsilon$-stationary solution. To the best of our knowledge, this is the first FL algorithm that achieves such {\it near-optimal} sample and communication complexities simultaneously.  Further, we show that there is a trade-off curve between local update frequencies and local minibatch sizes, on which the above sample and communication complexities can be maintained. {Finally,   we show that for the classical FedAvg (a.k.a. Local SGD, which is a momentum-less special case of the STEM), a similar trade-off curve exists, albeit with worse sample and communication complexities. Our insights on this trade-off provides guidelines for choosing the four important design elements for FL algorithms, the update frequency, directions, and minibatch sizes to achieve the best performance.}

----

## [463] Bubblewrap: Online tiling and real-time flow prediction on neural manifolds

**Authors**: *Anne Draelos, Pranjal Gupta, Na Young Jun, Chaichontat Sriworarat, John M. Pearson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/307eb8ee16198da891c521eca21464c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/307eb8ee16198da891c521eca21464c1-Abstract.html)

**Abstract**:

While most classic studies of function in experimental neuroscience have focused on the coding properties of individual neurons, recent developments in recording technologies have resulted in an increasing emphasis on the dynamics of neural populations. This has given rise to a wide variety of models for analyzing population activity in relation to experimental variables, but direct testing of many neural population hypotheses requires intervening in the system based on current neural state, necessitating models capable of inferring neural state online. Existing approaches, primarily based on dynamical systems, require strong parametric assumptions that are easily violated in the noise-dominated regime and do not scale well to the thousands of data channels in modern experiments. To address this problem, we propose a method that combines fast, stable dimensionality reduction with a soft tiling of the resulting neural manifold, allowing dynamics to be approximated as a probability flow between tiles. This method can be fit efficiently using online expectation maximization, scales to tens of thousands of tiles, and outperforms existing methods when dynamics are noise-dominated or feature multi-modal transition probabilities. The resulting model can be trained at kiloHertz data rates, produces accurate approximations of neural dynamics within minutes, and generates predictions on submillisecond time scales. It retains predictive performance throughout many time steps into the future and is fast enough to serve as a component of closed-loop causal experiments.

----

## [464] The Semi-Random Satisfaction of Voting Axioms

**Authors**: *Lirong Xia*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3083202a936b7d0ef8b680d7ae73fa1a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3083202a936b7d0ef8b680d7ae73fa1a-Abstract.html)

**Abstract**:

We initiate the work towards a comprehensive picture of the worst average-case satisfaction of  voting axioms in semi-random models, to provide a finer and more realistic foundation for comparing voting rules. We adopt the semi-random model and formulation in [Xia 2020],  where an adversary chooses arbitrarily correlated ``ground truth'' preferences for the agents, on top of which random noises are added. We  focus on characterizing the semi-random satisfaction of two well-studied voting axioms:  Condorcet criterion and participation. We prove that  for any fixed number of alternatives, when the number of voters $n$ is sufficiently large, the semi-random satisfaction of the Condorcet criterion under a wide range of voting rules is $1$, $1-\exp(-\Theta(n))$, $\Theta(n^{-0.5})$, $ \exp(-\Theta(n))$, or being $\Theta(1)$ and $1-\Theta(1)$ at the same time; and the semi-random satisfaction of participation is  $1-\Theta(n^{-0.5})$.   Our results  address  open questions by Berg and Lepelley in 1994, and also  confirm the following high-level message: the Condorcet criterion is a bigger concern than participation under realistic models.

----

## [465] Deep Marching Tetrahedra: a Hybrid Representation for High-Resolution 3D Shape Synthesis

**Authors**: *Tianchang Shen, Jun Gao, Kangxue Yin, Ming-Yu Liu, Sanja Fidler*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/30a237d18c50f563cba4531f1db44acf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/30a237d18c50f563cba4531f1db44acf-Abstract.html)

**Abstract**:

We introduce DMTet, a deep 3D conditional generative model that can synthesize high-resolution 3D shapes using simple user guides such as coarse voxels. It marries the merits of implicit and explicit 3D representations by leveraging a novel hybrid 3D representation. Compared to the current implicit approaches, which are trained to regress the signed distance values, DMTet directly optimizes for the reconstructed surface, which enables us to synthesize finer geometric details with fewer artifacts. Unlike deep 3D generative models that directly generate explicit representations such as meshes, our model can synthesize shapes with arbitrary topology. The core of DMTet includes a deformable tetrahedral grid that encodes a discretized signed distance function and a differentiable marching tetrahedra layer that converts the implicit signed distance representation to the explicit surface mesh representation. This combination allows joint optimization of the surface geometry and topology as well as generation of the hierarchy of subdivisions using reconstruction and adversarial losses defined explicitly on the surface mesh. Our approach significantly outperforms existing work on conditional shape synthesis from coarse voxel inputs, trained on a dataset of complex 3D animal shapes. Project page: https://nv-tlabs.github.io/DMTet/.

----

## [466] Learning to Combine Per-Example Solutions for Neural Program Synthesis

**Authors**: *Disha Shrivastava, Hugo Larochelle, Daniel Tarlow*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/30d411fdc0e6daf092a74354094359bb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/30d411fdc0e6daf092a74354094359bb-Abstract.html)

**Abstract**:

The goal of program synthesis from examples is to find a computer program that is consistent with a given set of input-output examples. Most learning-based approaches try to find a program that satisfies all examples at once. Our work, by contrast, considers an approach that breaks the problem into two stages: (a) find programs that satisfy only one example, and (b) leverage these per-example solutions to yield a program that satisfies all examples. We introduce the Cross Aggregator neural network module based on a multi-head attention mechanism that learns to combine the cues present in these per-example solutions to synthesize a global solution. Evaluation across programs of different lengths and under two different experimental settings reveal that when given the same time budget, our technique significantly improves the success rate over PCCoder [Zohar et. al 2018] and other ablation baselines.

----

## [467] On Success and Simplicity: A Second Look at Transferable Targeted Attacks

**Authors**: *Zhengyu Zhao, Zhuoran Liu, Martha A. Larson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/30d454f09b771b9f65e3eaf6e00fa7bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/30d454f09b771b9f65e3eaf6e00fa7bd-Abstract.html)

**Abstract**:

Achieving transferability of targeted attacks is reputed to be remarkably difficult. The current state of the art has resorted to resource-intensive solutions that necessitate training model(s) for each target class with additional data. In our investigation, we find, however, that simple transferable attacks which require neither model training nor additional data can achieve surprisingly strong targeted transferability. This insight has been overlooked until now, mainly because the widespread practice of attacking with only few iterations has largely limited the attack convergence to optimal targeted transferability. In particular, we, for the first time, identify that a very simple logit loss can largely surpass the commonly adopted cross-entropy loss, and yield even better results than the resource-intensive state of the art. Our analysis spans a variety of transfer scenarios, especially including three new, realistic scenarios: an ensemble transfer scenario with little model similarity, a worse-case scenario with low-ranked target classes, and also a real-world attack on the Google Cloud Vision API. Results in these new transfer scenarios demonstrate that the commonly adopted, easy scenarios cannot fully reveal the actual strength of different attacks and may cause misleading comparative results. We also show the usefulness of the simple logit loss for generating targeted universal adversarial perturbations in a data-free manner. Overall, the aim of our analysis is to inspire a more meaningful evaluation on targeted transferability. Code is available at https://github.com/ZhengyuZhao/Targeted-Tansfer.

----

## [468] Provably efficient, succinct, and precise explanations

**Authors**: *Guy Blanc, Jane Lange, Li-Yang Tan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/30d4e6422cd65c7913bc9ce62e078b79-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/30d4e6422cd65c7913bc9ce62e078b79-Abstract.html)

**Abstract**:

We consider the problem of explaining the predictions of an arbitrary blackbox model $f$: given query access to $f$ and an instance $x$, output a small set of $x$'s features that in conjunction essentially determines $f(x)$. We design an efficient algorithm with provable guarantees on the succinctness and precision of the explanations that it returns. Prior algorithms were either efficient but lacked such guarantees, or achieved such guarantees but were inefficient.   We obtain our algorithm via a connection to the problem of {\sl implicitly} learning decision trees.  The implicit nature of this learning task allows for efficient algorithms even when the complexity of~$f$ necessitates an intractably large surrogate decision tree.  We solve the implicit learning problem by bringing together techniques from learning theory, local computation algorithms, and complexity theory.   Our approach of “explaining by implicit learning” shares elements of two previously disparate methods for post-hoc explanations, global and local explanations, and we make the case that it enjoys advantages of both.

----

## [469] Refined Learning Bounds for Kernel and Approximate $k$-Means

**Authors**: *Yong Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/30f8f6b940d1073d8b6a5eebc46dd6e5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/30f8f6b940d1073d8b6a5eebc46dd6e5-Abstract.html)

**Abstract**:

Kernel $k$-means is one of the most popular approaches to clustering and its theoretical properties have been investigated for decades. However, the existing state-of-the-art risk bounds are of order $\mathcal{O}(k/\sqrt{n})$, which do not match with the stated lower bound $\Omega(\sqrt{k/n})$ in terms of $k$, where $k$ is the number of clusters and $n$ is the size of the training set. In this paper, we study the statistical properties of kernel $k$-means and Nystr\"{o}m-based kernel $k$-means, and obtain optimal clustering risk bounds, which improve the existing risk bounds. Particularly, based on a refined upper bound of Rademacher complexity [21], we first derive an optimal risk bound of rate $\mathcal{O}(\sqrt{k/n})$ for empirical risk minimizer (ERM), and further extend it to general cases beyond ERM. Then, we analyze the statistical effect of computational approximations of Nystr\"{o}m kernel $k$-means, and prove that it achieves the same statistical accuracy as the original kernel $k$-means considering only $\Omega(\sqrt{nk})$ Nystr\"{o}m landmark points. We further relax the restriction of landmark points from $\Omega(\sqrt{nk})$ to $\Omega(\sqrt{n})$ under a mild condition. Finally, we validate the theoretical findings via numerical experiments.

----

## [470] Learning Causal Semantic Representation for Out-of-Distribution Prediction

**Authors**: *Chang Liu, Xinwei Sun, Jindong Wang, Haoyue Tang, Tao Li, Tao Qin, Wei Chen, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/310614fca8fb8e5491295336298c340f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/310614fca8fb8e5491295336298c340f-Abstract.html)

**Abstract**:

Conventional supervised learning methods, especially deep ones, are found to be sensitive to out-of-distribution (OOD) examples, largely because the learned representation mixes the semantic factor with the variation factor due to their domain-specific correlation, while only the semantic factor causes the output. To address the problem, we propose a Causal Semantic Generative model (CSG) based on a causal reasoning so that the two factors are modeled separately, and develop methods for OOD prediction from a single training domain, which is common and challenging. The methods are based on the causal invariance principle, with a novel design in variational Bayes for both efficient learning and easy prediction. Theoretically, we prove that under certain conditions, CSG can identify the semantic factor by fitting training data, and this semantic-identification guarantees the boundedness of OOD generalization error and the success of adaptation. Empirical study shows improved OOD performance over prevailing baselines.

----

## [471] A first-order primal-dual method with adaptivity to local smoothness

**Authors**: *Maria-Luiza Vladarean, Yura Malitsky, Volkan Cevher*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/310b60949d2b6096903d7e8a539b20f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/310b60949d2b6096903d7e8a539b20f5-Abstract.html)

**Abstract**:

We consider the problem of finding a saddle point for the convex-concave objective $\min_x \max_y f(x) + \langle Ax, y\rangle - g^*(y)$, where $f$ is a convex function with locally Lipschitz gradient and $g$ is convex and possibly non-smooth. We propose an adaptive version of the Condat-VÅ© algorithm, which alternates between primal gradient steps and dual proximal steps. The method achieves stepsize adaptivity through a simple rule involving $\|A\|$ and the norm of recently computed gradients of $f$. Under standard assumptions, we prove an $\mathcal{O}(k^{-1})$ ergodic convergence rate. Furthermore, when $f$ is also locally strongly convex and $A$ has full row rank we show that our method converges with a linear rate. Numerical experiments are provided for illustrating the practical performance of the algorithm.

----

## [472] A Theory-Driven Self-Labeling Refinement Method for Contrastive Representation Learning

**Authors**: *Pan Zhou, Caiming Xiong, Xiaotong Yuan, Steven Chu-Hong Hoi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/310ce61c90f3a46e340ee8257bc70e93-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/310ce61c90f3a46e340ee8257bc70e93-Abstract.html)

**Abstract**:

For an image  query, unsupervised contrastive learning  labels crops of  the same image as positives,  and other image crops as  negatives. Although intuitive, such a native label assignment strategy cannot reveal the underlying semantic similarity between a  query and  its positives and negatives, and impairs performance,  since some negatives are  semantically similar to  the query or even share the same semantic class as the query.  In this work, we first  prove that for  contrastive learning,  inaccurate label assignment heavily  impairs its generalization for semantic instance discrimination, while accurate labels  benefit its generalization.  Inspired by this theory, we  propose   a novel self-labeling refinement approach for contrastive learning. It improves the label quality via two complementary  modules:  (i)  self-labeling refinery (SLR) to  generate accurate labels and (ii)  momentum mixup (MM)  to enhance similarity between query and its positive. SLR uses a positive of a query to estimate  semantic similarity between  a query and its positive and negatives, and  combines estimated similarity with  vanilla label assignment in contrastive learning to  iteratively generate  more accurate and informative soft labels. We theoretically show that our SLR can exactly recover the true semantic  labels of  label-corrupted  data, and  supervises   networks to achieve zero prediction  error on classification tasks.  MM randomly  combines   queries and  positives to increase  semantic similarity between the generated virtual queries and their positives so as to improves label accuracy.  Experimental results on CIFAR10,  ImageNet, VOC and COCO show the effectiveness of our method.

----

## [473] Adversarial Robustness with Semi-Infinite Constrained Learning

**Authors**: *Alexander Robey, Luiz F. O. Chamon, George J. Pappas, Hamed Hassani, Alejandro Ribeiro*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/312ecfdfa8b239e076b114498ce21905-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/312ecfdfa8b239e076b114498ce21905-Abstract.html)

**Abstract**:

Despite strong performance in numerous applications, the fragility of deep learning to input perturbations has raised serious questions about its use in safety-critical domains.  While adversarial training can mitigate this issue in practice, state-of-the-art methods are increasingly application-dependent, heuristic in nature, and suffer from fundamental trade-offs between nominal performance and robustness. Moreover, the problem of finding worst-case perturbations is non-convex and underparameterized, both of which engender a non-favorable optimization landscape. Thus, there is a gap between the theory and practice of robust learning, particularly with respect to when and why adversarial training works.  In this paper, we take a constrained learning approach to address these questions and to provide a theoretical foundation for robust learning. In particular, we leverage semi-infinite optimization and non-convex duality theory to show that adversarial training is equivalent to a statistical problem over perturbation distributions. Notably, we show that a myriad of previous robust training techniques can be recovered for particular, sub-optimal choices of these distributions. Using these insights, we then propose a hybrid Langevin Markov Chain Monte Carlo approach for which several common algorithms (e.g., PGD) are special cases. Finally, we show that our approach can mitigate the trade-off between nominal and robust performance, yielding state-of-the-art results on MNIST and CIFAR-10.  Our code is available at: https://github.com/arobey1/advbench.

----

## [474] Conformal Time-series Forecasting

**Authors**: *Kamile Stankeviciute, Ahmed M. Alaa, Mihaela van der Schaar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html)

**Abstract**:

Current approaches for multi-horizon time series forecasting using recurrent neural networks (RNNs) focus on issuing point estimates, which is insufficient for decision-making in critical application domains where an uncertainty estimate is also required. Existing approaches for uncertainty quantification in RNN-based time-series forecasts are limited as they may require significant alterations to the underlying model architecture, may be computationally complex, may be difficult to calibrate, may incur high sample complexity, and may not provide theoretical guarantees on frequentist coverage. In this paper, we extend the inductive conformal prediction framework to the time-series forecasting setup, and propose a lightweight algorithm to address all of the above limitations, providing uncertainty estimates with theoretical guarantees for any multi-horizon forecast predictor and any dataset with minimal exchangeability assumptions. We demonstrate the effectiveness of our approach by comparing it with existing benchmarks on a variety of synthetic and real-world datasets.

----

## [475] A 3D Generative Model for Structure-Based Drug Design

**Authors**: *Shitong Luo, Jiaqi Guan, Jianzhu Ma, Jian Peng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/314450613369e0ee72d0da7f6fee773c-Abstract.html)

**Abstract**:

We study a fundamental problem in structure-based drug design --- generating molecules that bind to specific protein binding sites. While we have witnessed the great success of deep generative models in drug design, the existing methods are mostly string-based or graph-based. They are limited by the lack of spatial information and thus unable to be applied to structure-based design tasks. Particularly, such models have no or little knowledge of how molecules interact with their target proteins exactly in 3D space. In this paper, we propose a 3D generative model that generates molecules given a designated 3D protein binding site. Specifically, given a binding site as the 3D context, our model estimates the probability density of atom's occurrences in 3D space --- positions that are more likely to have atoms will be assigned higher probability. To generate 3D molecules, we propose an auto-regressive sampling scheme --- atoms are sampled sequentially from the learned distribution until there is no room for new atoms. Combined with this sampling scheme, our model can generate valid and diverse molecules, which could be applicable to various structure-based molecular design tasks such as molecule sampling and linker design. Experimental results demonstrate that molecules sampled from our model exhibit high binding affinity to specific targets and good drug properties such as drug-likeness even if the model is not explicitly optimized for them.

----

## [476] Bootstrapping the Error of Oja's Algorithm

**Authors**: *Robert Lunde, Purnamrita Sarkar, Rachel A. Ward*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3152e3b1e52e2cb123363787d5f76c95-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3152e3b1e52e2cb123363787d5f76c95-Abstract.html)

**Abstract**:

We consider the problem of quantifying uncertainty for the estimation error of the leading eigenvector from Oja's algorithm for streaming principal component analysis, where the data are generated IID from some unknown distribution.  By combining classical tools from the U-statistics literature with recent results on high-dimensional central limit theorems for quadratic forms of random vectors and concentration of matrix products, we establish a weighted $\chi^2$ approximation result for the $\sin^2$ error between the population eigenvector and the output of Ojaâ€™s algorithm. Since estimating the covariance matrix associated with the approximating distribution requires knowledge of unknown model parameters, we propose a multiplier bootstrap algorithm that may be updated in an online manner.  We establish conditions under which the bootstrap distribution is close to the corresponding sampling distribution with high probability, thereby establishing the bootstrap as a consistent inferential method in an appropriate asymptotic regime.

----

## [477] Landscape analysis of an improved power method for tensor decomposition

**Authors**: *Joe Kileel, Timo Klock, João M. Pereira*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31784d9fc1fa0d25d04eae50ac9bf787-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31784d9fc1fa0d25d04eae50ac9bf787-Abstract.html)

**Abstract**:

In this work, we consider the optimization formulation for symmetric tensor decomposition recently introduced in the Subspace Power Method (SPM) of Kileel and Pereira.  Unlike popular alternative functionals for tensor decomposition, the SPM objective function has the desirable properties that its maximal value is known in advance, and its global optima are exactly the rank-1 components of the tensor when the input is sufficiently low-rank.  We analyze the non-convex optimization landscape associated with the SPM objective.  Our analysis accounts for working with noisy tensors.  We derive quantitative bounds such that any second-order critical point with SPM objective value exceeding the bound must equal a tensor component in the noiseless case, and must approximate a tensor component in the noisy case. For decomposing tensors of size $D^{\times m}$, we obtain a near-global guarantee up to rank $\widetilde{o}(D^{\lfloor m/2 \rfloor})$ under a random tensor model, and a global guarantee up to rank $\mathcal{O}(D)$ assuming deterministic frame conditions.  This implies that SPM with suitable initialization is a provable, efficient, robust algorithm for low-rank symmetric tensor decomposition.  We conclude with numerics that show a practical preferability for using the SPM functional over a more established counterpart.

----

## [478] Curriculum Offline Imitating Learning

**Authors**: *Minghuan Liu, Hanye Zhao, Zhengyu Yang, Jian Shen, Weinan Zhang, Li Zhao, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31839b036f63806cba3f47b93af8ccb5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31839b036f63806cba3f47b93af8ccb5-Abstract.html)

**Abstract**:

Offline reinforcement learning (RL) tasks require the agent to learn from a pre-collected dataset with no further interactions with the environment. Despite the potential to surpass the behavioral policies, RL-based methods are generally impractical due to the training instability and bootstrapping the extrapolation errors, which always require careful hyperparameter tuning via online evaluation. In contrast, offline imitation learning (IL) has no such issues since it learns the policy directly without estimating the value function by bootstrapping. However, IL is usually limited in the capability of the behavioral policy and tends to learn a mediocre behavior from the dataset collected by the mixture of policies. In this paper, we aim to take advantage of IL but mitigate such a drawback. Observing that behavior cloning is able to imitate neighboring policies with less data, we propose \textit{Curriculum Offline Imitation Learning (COIL)}, which utilizes an experience picking strategy to make the agent imitate from adaptive neighboring policies with a higher return, and improves the current policy along curriculum stages. On continuous control benchmarks, we compare COIL against both imitation-based methods and RL-based methods, showing that COIL not only avoids just learning a mediocre behavior on mixed datasets but is also even competitive with state-of-the-art offline RL methods.

----

## [479] Robust Pose Estimation in Crowded Scenes with Direct Pose-Level Inference

**Authors**: *Dongkai Wang, Shiliang Zhang, Gang Hua*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31857b449c407203749ae32dd0e7d64a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31857b449c407203749ae32dd0e7d64a-Abstract.html)

**Abstract**:

Multi-person pose estimation in crowded scenes is challenging because overlapping and occlusions make it difficult to detect person bounding boxes and infer pose cues from individual keypoints. To address those issues, this paper proposes a direct pose-level inference strategy that is free of bounding box detection and keypoint grouping. Instead of inferring individual keypoints, the Pose-level Inference Network (PINet) directly infers the complete pose cues for a person from his/her visible body parts. PINet first applies the Part-based Pose Generation (PPG) to infer multiple coarse poses for each person from his/her body parts. Those coarse poses are refined by the Pose Refinement module through incorporating pose priors, and finally are fused in the Pose Fusion module. PINet relies on discriminative body parts to differentiate overlapped persons, and applies visual body cues to infer the global pose cues.  Experiments on several crowded scenes pose estimation benchmarks demonstrate the superiority of PINet. For instance, it achieves 59.8% AP on the OCHuman dataset, outperforming the recent works by a large margin.

----

## [480] Ising Model Selection Using $\ell_{1}$-Regularized Linear Regression: A Statistical Mechanics Analysis

**Authors**: *Xiangming Meng, Tomoyuki Obuchi, Yoshiyuki Kabashima*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31917677a66c6eddd3ab1f68b0679e2f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31917677a66c6eddd3ab1f68b0679e2f-Abstract.html)

**Abstract**:

We theoretically analyze the typical learning performance of $\ell_{1}$-regularized linear regression ($\ell_1$-LinR) for Ising model selection using the replica method from statistical mechanics. For typical random regular graphs in the paramagnetic phase, an accurate estimate of the typical sample complexity of $\ell_1$-LinR is obtained.   Remarkably, despite the model misspecification, $\ell_1$-LinR is model selection consistent with the same order of sample complexity as $\ell_{1}$-regularized logistic regression ($\ell_1$-LogR), i.e., $M=\mathcal{O}\left(\log N\right)$,  where $N$ is the number of variables of the Ising model. Moreover, we provide an efficient method to accurately predict the non-asymptotic behavior of $\ell_1$-LinR for moderate $M, N$, such as precision and recall. Simulations show a fairly good agreement between theoretical predictions and experimental results, even for graphs with many loops, which supports our findings. Although this paper mainly focuses on $\ell_1$-LinR, our method is readily applicable for precisely characterizing the typical learning performances of a wide class of  $\ell_{1}$-regularized $M$-estimators including $\ell_1$-LogR and interaction screening.

----

## [481] Conformal Prediction using Conditional Histograms

**Authors**: *Matteo Sesia, Yaniv Romano*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31b3b31a1c2f8a370206f111127c0dbd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31b3b31a1c2f8a370206f111127c0dbd-Abstract.html)

**Abstract**:

This paper develops a conformal method to compute prediction intervals for non-parametric regression that can automatically adapt to skewed data. Leveraging black-box machine learning algorithms to estimate the conditional distribution of the outcome using histograms, it translates their output into the shortest prediction intervals with approximate conditional coverage. The resulting prediction intervals provably have marginal coverage in finite samples, while asymptotically achieving conditional coverage and optimal length if the black-box model is consistent. Numerical experiments with simulated and real data demonstrate improved performance compared to state-of-the-art alternatives, including conformalized quantile regression and other distributional conformal prediction approaches.

----

## [482] Contrastive Graph Poisson Networks: Semi-Supervised Learning with Extremely Limited Labels

**Authors**: *Sheng Wan, Yibing Zhan, Liu Liu, Baosheng Yu, Shirui Pan, Chen Gong*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html)

**Abstract**:

Graph Neural Networks (GNNs) have achieved remarkable performance in the task of semi-supervised node classification. However, most existing GNN models require sufficient labeled data for effective network training. Their performance can be seriously degraded when labels are extremely limited. To address this issue, we propose a new framework termed Contrastive Graph Poisson Networks (CGPN) for node classification under extremely limited labeled data. Specifically, our CGPN derives from variational inference; integrates a newly designed Graph Poisson Network (GPN) to effectively propagate the limited labels to the entire graph and a normal GNN, such as Graph Attention Network, that flexibly guides the propagation of GPN; applies a contrastive objective to further exploit the supervision information from the learning process of GPN and GNN models. Essentially, our CGPN can enhance the learning performance of GNNs under extremely limited labels by contrastively propagating the limited labels to the entire graph. We conducted extensive experiments on different types of datasets to demonstrate the superiority of CGPN.

----

## [483] Collaborative Uncertainty in Multi-Agent Trajectory Forecasting

**Authors**: *Bohan Tang, Yiqi Zhong, Ulrich Neumann, Gang Wang, Siheng Chen, Ya Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/31ca0ca71184bbdb3de7b20a51e88e90-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/31ca0ca71184bbdb3de7b20a51e88e90-Abstract.html)

**Abstract**:

Uncertainty modeling is critical in trajectory-forecasting systems for both interpretation and safety reasons. To better predict the future trajectories of multiple agents, recent works have introduced interaction modules to capture interactions among agents. This approach leads to correlations among the predicted trajectories. However, the uncertainty brought by such correlations is neglected. To fill this gap, we propose a novel concept, collaborative uncertainty (CU), which models the uncertainty resulting from the interaction module. We build a general CU-based framework to make a prediction model learn the future trajectory and the corresponding uncertainty. The CU-based framework is integrated as a plugin module to current state-of-the-art (SOTA) systems and deployed in two special cases based on multivariate Gaussian and Laplace distributions. In each case, we conduct extensive experiments on two synthetic datasets and two public, large-scale benchmarks of trajectory forecasting. The results are promising: 1) The results of synthetic datasets show that CU-based framework allows the model to nicely rebuild the ground-truth distribution. 2) The results of trajectory forecasting benchmarks demonstrate that the CU-based framework steadily helps SOTA systems improve their performances. Specially, the proposed CU-based framework helps VectorNet improve by 57 cm regarding Final Displacement Error on nuScenes dataset. 3) The visualization results of CU illustrate that the value of CU is highly related to the amount of the interactive information among agents.

----

## [484] Network-to-Network Regularization: Enforcing Occam's Razor to Improve Generalization

**Authors**: *Rohan Ghosh, Mehul Motani*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/321cf86b4c9f5ddd04881a44067c2a5a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/321cf86b4c9f5ddd04881a44067c2a5a-Abstract.html)

**Abstract**:

What makes a classifier have the ability to generalize? There have been a lot of important attempts to address this question, but a clear answer is still elusive. Proponents of complexity theory find that the complexity of the classifier's function space is key to deciding generalization, whereas other recent work reveals that classifiers which extract invariant feature representations are likely to generalize better. Recent theoretical and empirical studies, however, have shown that even within a classifier's function space, there can be significant differences in the ability to generalize. Specifically, empirical studies have shown that among functions which have a good training data fit, functions with lower Kolmogorov complexity (KC) are likely to generalize better, while the opposite is true for functions of higher KC. Motivated by these findings, we propose, in this work, a novel measure of complexity called Kolmogorov Growth (KG), which we use to derive new generalization error bounds that only depend on the final choice of the classification function. Guided by the bounds, we propose a novel way of regularizing neural networks by constraining the network trajectory to remain in the low KG zone during training. Minimizing KG while learning is akin to applying the Occam's razor to neural networks. The proposed approach, called network-to-network regularization, leads to clear improvements in the generalization ability of classifiers. We verify this for three popular image datasets (MNIST, CIFAR-10, CIFAR-100) across varying training data sizes. Empirical studies find that conventional training of neural networks, unlike network-to-network regularization, leads to networks of high KG and lower test accuracies. Furthermore, we present the benefits of N2N regularization in the scenario where the training data labels are noisy. Using N2N regularization, we achieve competitive performance on MNIST, CIFAR-10 and CIFAR-100 datasets with corrupted training labels, significantly improving network performance compared to standard cross-entropy baselines in most cases. These findings illustrate the many benefits obtained from imposing a function complexity prior like Kolmogorov Growth during the training process.

----

## [485] Generalized and Discriminative Few-Shot Object Detection via SVD-Dictionary Enhancement

**Authors**: *Aming Wu, Suqi Zhao, Cheng Deng, Wei Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/325995af77a0e8b06d1204a171010b3a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/325995af77a0e8b06d1204a171010b3a-Abstract.html)

**Abstract**:

Few-shot object detection (FSOD) aims to detect new objects based on few annotated samples. To alleviate the impact of few samples, enhancing the generalization and discrimination abilities of detectors on new objects plays an important role. In this paper, we explore employing Singular Value Decomposition (SVD) to boost both the generalization and discrimination abilities. In specific, we propose a novel method, namely, SVD-Dictionary enhancement, to build two separated spaces based on the sorted singular values. Concretely, the eigenvectors corresponding to larger singular values are used to build the generalization space in which localization is performed, as these eigenvectors generally suppress certain variations (e.g., the variation of styles) and contain intrinsical characteristics of objects. Meanwhile, since the eigenvectors corresponding to relatively smaller singular values may contain richer category-related information, we can utilize them to build the discrimination space in which classification is performed. Dictionary learning is further leveraged to capture high-level discriminative information from the discrimination space, which is beneficial for improving detection accuracy. In the experiments, we separately verify the effectiveness of our method on PASCAL VOC and COCO benchmarks. Particularly, for the 2-shot case in VOC split1, our method significantly outperforms the baseline by 6.2\%. Moreover, visualization analysis shows that our method is instrumental in doing FSOD.

----

## [486] Conditioning Sparse Variational Gaussian Processes for Online Decision-making

**Authors**: *Wesley J. Maddox, Samuel Stanton, Andrew Gordon Wilson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html)

**Abstract**:

With a principled representation of uncertainty and closed form posterior updates, Gaussian processes (GPs) are a natural choice for online decision making. However, Gaussian processes typically require at least $\mathcal{O}(n^2)$ computations for $n$ training points, limiting their general applicability. Stochastic variational Gaussian processes (SVGPs) can provide scalable inference for a dataset of fixed size, but are difficult to efficiently condition on new data. We propose online variational conditioning (OVC), a procedure for efficiently conditioning SVGPs in an online setting that does not require re-training through the evidence lower bound with the addition of new data. OVC enables the pairing of SVGPs with advanced look-ahead acquisition functions for black-box optimization, even with non-Gaussian likelihoods. We show OVC provides compelling performance in a range of applications including active learning of malaria incidence, and reinforcement learning on MuJoCo simulated robotic control tasks.

----

## [487] Spherical Motion Dynamics: Learning Dynamics of Normalized Neural Network using SGD and Weight Decay

**Authors**: *Ruosi Wan, Zhanxing Zhu, Xiangyu Zhang, Jian Sun*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/326a8c055c0d04f5b06544665d8bb3ea-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/326a8c055c0d04f5b06544665d8bb3ea-Abstract.html)

**Abstract**:

In this paper, we comprehensively reveal the learning dynamics of normalized neural network using Stochastic Gradient Descent (with momentum) and Weight Decay (WD), named as Spherical Motion Dynamics (SMD). Most related works focus on studying behavior of effective learning rate" inequilibrium" state, i.e. assuming weight norm remains unchanged. However, their discussion on why this equilibrium can be reached is either absent or less convincing. Our work directly explores the cause of equilibrium, as a special state of SMD. Specifically, 1) we introduce the assumptions that can lead to equilibrium state in SMD, and prove equilibrium can be reached in a linear rate regime under given assumptions; 2) we propose ``angular update" as a substitute for effective learning rate to depict the state of SMD, and derive the theoretical value of angular update in equilibrium state; 3) we verify our assumptions and theoretical results on various large-scale computer vision tasks including ImageNet and MSCOCO with standard settings. Experiment results show our theoretical findings agree well with empirical observations. We also show that the behavior of angular update in SMD can produce interesting effect to the optimization of neural network in practice.

----

## [488] Imitating Deep Learning Dynamics via Locally Elastic Stochastic Differential Equations

**Authors**: *Jiayao Zhang, Hua Wang, Weijie J. Su*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/327af0f71f7acdfd882774225f04775f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/327af0f71f7acdfd882774225f04775f-Abstract.html)

**Abstract**:

Understanding the training dynamics of deep learning models is perhaps a necessary step toward demystifying the effectiveness of these models. In particular, how do training data from different classes gradually become separable in their feature spaces when training neural networks using stochastic gradient descent? In this paper, we model the evolution of features during deep learning training using a set of stochastic differential equations (SDEs) that each corresponding to a training sample. As a crucial ingredient in our modeling strategy, each SDE contains a drift term that reflects the impact of backpropagation at an input on the features of all samples. Our main finding uncovers a sharp phase transition phenomenon regarding the intra-class impact: if the SDEs are locally elastic in the sense that the impact is more significant on samples from the same class as the input, the features of training data become linearly separable---meaning vanishing training loss; otherwise, the features are not separable, no matter how long the training time is. In the presence of local elasticity, moreover, an analysis of our SDEs shows the emergence of a simple geometric structure called neural collapse of the features. Taken together, our results shed light on the decisive role of local elasticity underlying the training dynamics of neural networks. We corroborate our theoretical analysis with experiments on a synthesized dataset of geometric shapes as well as on CIFAR-10.

----

## [489] Probabilistic Forecasting: A Level-Set Approach

**Authors**: *Hilaf Hasson, Bernie Wang, Tim Januschowski, Jan Gasthaus*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32b127307a606effdcc8e51f60a45922-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32b127307a606effdcc8e51f60a45922-Abstract.html)

**Abstract**:

Large-scale time series panels have become ubiquitous over the last years in areas such as retail, operational metrics, IoT, and medical domain (to name only a few). This has resulted in a need for forecasting techniques that effectively leverage all available data by learning across all time series in each panel. Among the desirable properties of forecasting techniques, being able to generate probabilistic predictions ranks among the top. In this paper, we therefore present Level Set Forecaster (LSF), a simple yet effective general approach to transform a point estimator into a probabilistic one. By recognizing the connection of our algorithm to random forests (RFs) and quantile regression forests (QRFs), we are able to prove consistency guarantees of our approach under mild assumptions on the underlying point estimator. As a byproduct, we prove the first consistency results for QRFs under the CART-splitting criterion. Empirical experiments show that our approach, equipped with tree-based models as the point estimator, rivals state-of-the-art deep learning models in terms of forecasting accuracy.

----

## [490] Roto-translated Local Coordinate Frames For Interacting Dynamical Systems

**Authors**: *Miltiadis Kofinas, Naveen Shankar Nagaraja, Efstratios Gavves*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32b991e5d77ad140559ffb95522992d0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32b991e5d77ad140559ffb95522992d0-Abstract.html)

**Abstract**:

Modelling interactions is critical in learning complex dynamical systems, namely systems of interacting objects with highly non-linear and time-dependent behaviour. A large class of such systems can be formalized as $\textit{geometric graphs}$, $\textit{i.e.}$ graphs with nodes positioned in the Euclidean space given an $\textit{arbitrarily}$ chosen global coordinate system, for instance vehicles in a traffic scene. Notwithstanding the arbitrary global coordinate system, the governing dynamics of the respective dynamical systems are invariant to rotations and translations, also known as $\textit{Galilean invariance}$. As ignoring these invariances leads to worse generalization, in this work we propose local coordinate systems per node-object to induce roto-translation invariance to the geometric graph of the interacting dynamical system. Further, the local coordinate systems allow for a natural definition of anisotropic filtering in graph neural networks. Experiments in traffic scenes, 3D motion capture, and colliding particles demonstrate the proposed approach comfortably outperforms the recent state-of-the-art.

----

## [491] ParK: Sound and Efficient Kernel Ridge Regression by Feature Space Partitions

**Authors**: *Luigi Carratino, Stefano Vigogna, Daniele Calandriello, Lorenzo Rosasco*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32b9e74c8f60958158eba8d1fa372971-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32b9e74c8f60958158eba8d1fa372971-Abstract.html)

**Abstract**:

We introduce ParK, a new large-scale solver for kernel ridge regression. Our approach combines partitioning with random projections and iterative optimization to reduce space and time complexity while provably maintaining the same statistical accuracy. In particular, constructing suitable partitions directly in the feature space rather than in the input space, we promote orthogonality between the local estimators, thus ensuring that key quantities such as local effective dimension and bias remain under control. We characterize the statistical-computational tradeoff of our model, and demonstrate the effectiveness of our method by numerical experiments on large-scale datasets.

----

## [492] Scaling Gaussian Processes with Derivative Information Using Variational Inference

**Authors**: *Misha Padidar, Xinran Zhu, Leo Huang, Jacob R. Gardner, David Bindel*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32bbf7b2bc4ed14eb1e9c2580056a989-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32bbf7b2bc4ed14eb1e9c2580056a989-Abstract.html)

**Abstract**:

Gaussian processes with derivative information are useful in many settings where derivative information is available, including numerous Bayesian optimization and regression tasks that arise in the natural sciences. Incorporating derivative observations, however, comes with a dominating $O(N^3D^3)$ computational cost when training on $N$ points in $D$ input dimensions. This is intractable for even moderately sized problems. While recent work has addressed this intractability in the low-$D$ setting, the high-$N$, high-$D$ setting is still unexplored and of great value, particularly as machine learning problems increasingly become high dimensional. In this paper, we introduce methods to achieve fully scalable Gaussian process regression with derivatives using variational inference. Analogous to the use of inducing values to sparsify the labels of a training set, we introduce the concept of inducing directional derivatives to sparsify the partial derivative information of the training set. This enables us to construct a variational posterior that incorporates derivative information but whose size depends neither on the full dataset size $N$ nor the full dimensionality $D$. We demonstrate the full scalability of our approach on a variety of tasks, ranging from a high dimensional Stellarator fusion regression task to training graph convolutional neural networks on PubMed using Bayesian optimization. Surprisingly, we additionally find that our approach can improve regression performance even in settings where only label data is available.

----

## [493] On the Representation of Solutions to Elliptic PDEs in Barron Spaces

**Authors**: *Ziang Chen, Jianfeng Lu, Yulong Lu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32cfdce9631d8c7906e8e9d6e68b514b-Abstract.html)

**Abstract**:

Numerical solutions to high-dimensional partial differential equations (PDEs) based on neural networks have seen exciting developments. This paper derives complexity estimates of the solutions of $d$-dimensional second-order elliptic PDEs in the Barron space, that is a set of functions admitting the integral of certain parametric ridge function against a probability measure on the parameters. We prove under some appropriate assumptions that if the coefficients and the source term of the elliptic PDE lie in Barron spaces, then the solution of the PDE is $\epsilon$-close with respect to the $H^1$ norm to a Barron function. Moreover, we prove dimension-explicit bounds for the Barron norm of this approximate solution, depending at most polynomially on the dimension $d$ of the PDE. As a direct consequence of the complexity estimates, the solution of the PDE can be approximated on any bounded domain by a two-layer neural network with respect to the $H^1$ norm with a dimension-explicit convergence rate.

----

## [494] A/B Testing for Recommender Systems in a Two-sided Marketplace

**Authors**: *Preetam Nandy, Divya Venugopalan, Chun Lo, Shaunak Chatterjee*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32e19424b63cc63077a4031b87fb1010-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32e19424b63cc63077a4031b87fb1010-Abstract.html)

**Abstract**:

Two-sided marketplaces are standard business models of many online platforms (e.g., Amazon, Facebook, LinkedIn), wherein the platforms have consumers, buyers or content viewers on one side and producers, sellers or content-creators on the other. Consumer side measurement of the impact of a treatment variant can be done via simple online A/B testing. Producer side measurement is more challenging because the producer experience depends on the treatment assignment of the consumers. Existing approaches for producer side measurement are either based on graph cluster-based randomization or on certain treatment propagation assumptions. The former approach results in low-powered experiments as the producer-consumer network density increases and the latter approach lacks a strict notion of error control. In this paper, we propose (i) a quantification of the quality of a producer side experiment design, and (ii) a new experiment design mechanism that generates high-quality experiments based on this quantification. Our approach, called UniCoRn (Unifying Counterfactual Rankings), provides explicit control over the quality of the experiment and its computation cost. Further, we prove that our experiment design is optimal to the proposed design quality measure. Our approach is agnostic to the density of the producer-consumer network and does not rely on any treatment propagation assumption. Moreover, unlike the existing approaches, we do not need to know the underlying network in advance, making this widely applicable to the industrial setting where the underlying network is unknown and challenging to predict a priori due to its dynamic nature. We use simulations to validate our approach and compare it against existing methods. We also deployed UniCoRn in an edge recommendation application that serves tens of millions of members and billions of edge recommendations daily.

----

## [495] Retiring Adult: New Datasets for Fair Machine Learning

**Authors**: *Frances Ding, Moritz Hardt, John Miller, Ludwig Schmidt*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html)

**Abstract**:

Although the fairness community has recognized the importance of data, researchers in the area primarily rely on UCI Adult when it comes to tabular data. Derived from a 1994 US Census survey, this dataset has appeared in hundreds of research papers where it served as the basis for the development and comparison of many algorithmic fairness interventions. We reconstruct a superset of the UCI Adult data from available US Census sources and reveal idiosyncrasies of the UCI Adult dataset that limit its external validity. Our primary contribution is a suite of new datasets derived from US Census surveys that extend the existing data ecosystem for research on fair machine learning. We create prediction tasks relating to income, employment, health, transportation, and housing. The data span multiple years and all states of the United States, allowing researchers to study temporal shift and geographic variation. We highlight a broad initial sweep of new empirical insights relating to trade-offs between fairness criteria, performance of algorithmic interventions, and the role of distribution shift based on our new datasets. Our findings inform ongoing debates, challenge some existing narratives, and point to future research directions.

----

## [496] Cardinality constrained submodular maximization for random streams

**Authors**: *Paul Liu, Aviad Rubinstein, Jan Vondrák, Junyao Zhao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/333222170ab9edca4785c39f55221fe7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/333222170ab9edca4785c39f55221fe7-Abstract.html)

**Abstract**:

We consider the problem of maximizing submodular functions in single-pass streaming and secretaries-with-shortlists models, both with random arrival order.For cardinality constrained monotone functions, Agrawal, Shadravan, and Stein~\cite{SMC19} gave a single-pass $(1-1/e-\varepsilon)$-approximation algorithm using only linear memory, but their exponential dependence on $\varepsilon$ makes it impractical even for $\varepsilon=0.1$.We simplify both the algorithm and the analysis, obtaining an exponential improvement in the $\varepsilon$-dependence (in particular, $O(k/\varepsilon)$ memory).Extending these techniques, we also give a simple $(1/e-\varepsilon)$-approximation for non-monotone functions in $O(k/\varepsilon)$ memory. For the monotone case, we also give a corresponding unconditional hardness barrier of $1-1/e+\varepsilon$ for single-pass algorithms in randomly ordered streams, even assuming unlimited computation. Finally, we show that the algorithms are simple to implement and work well on real world datasets.

----

## [497] Self-Instantiated Recurrent Units with Dynamic Soft Recursion

**Authors**: *Aston Zhang, Yi Tay, Yikang Shen, Alvin Chan, Shuai Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3341f6f048384ec73a7ba2e77d2db48b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3341f6f048384ec73a7ba2e77d2db48b-Abstract.html)

**Abstract**:

While standard recurrent neural networks explicitly impose a chain structure on different forms of data, they do not have an explicit bias towards recursive self-instantiation where the extent of recursion is dynamic.  Given diverse and even growing data modalities (e.g., logic, algorithmic input and output, music, code, images, and language) that can be expressed in sequences and may benefit from more architectural flexibility, we propose the self-instantiated recurrent unit (Self-IRU) with a novel inductive bias towards dynamic soft recursion. On one hand, theSelf-IRU is characterized by recursive self-instantiation via its gating functions, i.e., gating mechanisms of the Self-IRU are controlled by instances of the Self-IRU itself, which are repeatedly invoked in a recursive fashion. On the other hand, the extent of the Self-IRU recursion is controlled by gates whose values are between 0 and 1 and may vary across the temporal dimension of sequences,  enabling dynamic soft recursion depth at each time step. The architectural flexibility and effectiveness of our proposed approach are demonstrated across multiple data modalities. For example, the Self-IRU achieves state-of-the-art performance on the logical inference dataset [Bowman et al., 2014] even when comparing with competitive models that have access to ground-truth syntactic information.

----

## [498] Sparse Uncertainty Representation in Deep Learning with Inducing Weights

**Authors**: *Hippolyt Ritter, Martin Kukla, Cheng Zhang, Yingzhen Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/334467d41d5cf21e234465a1530ba647-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/334467d41d5cf21e234465a1530ba647-Abstract.html)

**Abstract**:

Bayesian Neural Networks and deep ensembles represent two modern paradigms of uncertainty quantification in deep learning. Yet these approaches struggle to scale mainly due to memory inefficiency, requiring parameter storage several times that of their deterministic counterparts. To address this, we augment each weight matrix with a small inducing weight matrix, projecting the uncertainty quantification into a lower dimensional space. We further extend Matheronâ€™s conditional Gaussian sampling rule to enable fast weight sampling, which enables our inference method to maintain reasonable run-time as compared with ensembles. Importantly, our approach achieves competitive performance to the state-of-the-art in prediction and uncertainty estimation tasks with fully connected neural networks and ResNets, while reducing the parameter size to $\leq 24.3\%$ of that of a single neural network.

----

## [499] Scalable Inference of Sparsely-changing Gaussian Markov Random Fields

**Authors**: *Salar Fattahi, Andrés Gómez*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/33853141e0873909be88f5c3e6144cc6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/33853141e0873909be88f5c3e6144cc6-Abstract.html)

**Abstract**:

We study the problem of inferring time-varying Gaussian Markov random fields, where the underlying graphical model is both sparse and changes {sparsely} over time. Most of the existing methods for the inference of time-varying Markov random fields (MRFs) rely on the \textit{regularized maximum likelihood estimation} (MLE), that typically suffer from weak statistical guarantees and high computational time. Instead, we introduce a new class of constrained optimization problems for the inference of sparsely-changing Gaussian MRFs (GMRFs). The proposed optimization problem is formulated based on the exact $\ell_0$ regularization, and can be solved in near-linear time and memory. Moreover, we show that the proposed estimator enjoys a provably small estimation error. We derive sharp statistical guarantees in the high-dimensional regime, showing that such problems can be learned with as few as one sample per time period. Our proposed method is extremely efficient in practice: it can accurately estimate sparsely-changing GMRFs with more than 500 million variables in less than one hour.

----

## [500] Grad2Task: Improved Few-shot Text Classification Using Gradients for Task Representation

**Authors**: *Jixuan Wang, Kuan-Chieh Wang, Frank Rudzicz, Michael Brudno*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/33a854e247155d590883b93bca53848a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/33a854e247155d590883b93bca53848a-Abstract.html)

**Abstract**:

Large pretrained language models (LMs) like BERT have improved performance in many disparate natural language processing (NLP) tasks. However, fine tuning such models requires a large number of training examples for each target task. Simultaneously, many realistic NLP problems are "few shot", without a sufficiently large training set. In this work, we propose a novel conditional neural process-based approach for few-shot text classification that learns to transfer from other diverse tasks with rich annotation. Our key idea is to represent each task using gradient information from a base model and to train an adaptation network that modulates a text classifier conditioned on the task representation. While previous task-aware few-shot learners represent tasks by input encoding, our novel task representation is more powerful, as the gradient captures input-output relationships of a task. Experimental results show that our approach outperforms traditional fine-tuning, sequential transfer learning, and state-of-the-art meta learning approaches on a collection of diverse few-shot tasks. We further conducted analysis and ablations to justify our design choices.

----

## [501] Learnability of Linear Thresholds from Label Proportions

**Authors**: *Rishi Saket*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/33b3214d792caf311e1f00fd22b392c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/33b3214d792caf311e1f00fd22b392c5-Abstract.html)

**Abstract**:

We study the problem of properly learning linear threshold functions (LTFs) in the learning from label proportions (LLP) framework. In this, the learning is on a collection of bags of feature-vectors with only the proportion of labels available for each bag. First, we provide an algorithm that, given a collection of such bags each of size at most two whose label proportions are consistent with (i.e., the bags are satisfied by) an unknown LTF, efficiently produces an LTF that satisfies at least $(2/5)$-fraction of the bags. If all the bags are non-monochromatic (i.e., bags of size two with differently labeled feature-vectors) the algorithm satisfies at least $(1/2)$-fraction of them. For the special case of OR over the $d$-dimensional boolean vectors, we give an algorithm which computes an LTF achieving an additional $\Omega(1/d)$ in accuracy for the two cases.Our main result provides evidence that these algorithmic bounds cannot be significantly improved, even for learning monotone ORs using LTFs. We prove that it is NP-hard, given a collection of non-monochromatic bags which are all satisfied by some monotone OR, to compute any function of constantly many LTFs that satisfies  $(1/2 + \varepsilon)$-fraction of the bags for any constant $\varepsilon > 0$. This bound is tight for the non-monochromatic bags case.The above is in contrast to the usual supervised learning setup (i.e., unit-sized bags) in which LTFs are efficiently learnable to arbitrary accuracy using linear programming, and even a trivial algorithm (any LTF or its complement) achieves an accuracy of $1/2$. These techniques however, fail in the LLP setting. Indeed, we show that the LLP learning of LTFs (even for the special case of monotone ORs) using LTFs dramatically increases in complexity as soon as bags of size two are allowed.Our work gives the first inapproximability for LLP learning LTFs, and a strong complexity separation between LLP  and traditional supervised learning.

----

## [502] A variational approximate posterior for the deep Wishart process

**Authors**: *Sebastian W. Ober, Laurence Aitchison*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/33ceb07bf4eeb3da587e268d663aba1a-Abstract.html)

**Abstract**:

Recent work introduced deep kernel processes as an entirely kernel-based alternative to NNs (Aitchison et al. 2020). Deep kernel processes flexibly learn good top-layer representations by alternately sampling the kernel from a distribution over positive semi-definite matrices and performing nonlinear transformations. A particular deep kernel process, the deep Wishart process (DWP), is of particular interest because its prior can be made equivalent to deep Gaussian process (DGP) priors for kernels that can be expressed entirely in terms of Gram matrices. However, inference in DWPs has not yet been possible due to the lack of sufficiently flexible distributions over positive semi-definite matrices. Here, we give a novel approach to obtaining flexible distributions over positive semi-definite matrices by generalising the Bartlett decomposition of the Wishart probability density. We use this new distribution to develop an approximate posterior for the DWP that includes dependency across layers. We develop a doubly-stochastic inducing-point inference scheme for the DWP and show experimentally that inference in the DWP can improve performance over doing inference in a DGP with the equivalent prior.

----

## [503] Neural Pseudo-Label Optimism for the Bank Loan Problem

**Authors**: *Aldo Pacchiano, Shaun Singh, Edward Chou, Alexander C. Berg, Jakob N. Foerster*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/33d6548e48d4318ceb0e3916a79afc84-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/33d6548e48d4318ceb0e3916a79afc84-Abstract.html)

**Abstract**:

We study a class of classification problems best exemplified by the \emph{bank loan} problem, where a lender decides whether or not to issue a loan. The lender only observes whether a customer will repay a loan if the loan is issued to begin with, and thus modeled decisions affect what data is available to the lender for future decisions. As a result, it is possible for the lender's algorithm to ``get stuck'' with a self-fulfilling model. This model never corrects its false negatives, since it never sees the true label for rejected data, thus accumulating infinite regret. In the case of linear models, this issue can be addressed by adding optimism directly into the model predictions. However, there are few methods that extend to the function approximation case using Deep Neural Networks. We present Pseudo-Label Optimism (PLOT), a conceptually and computationally simple method for this setting applicable to DNNs. \PLOT{} adds an optimistic label to the subset of decision points the current model is deciding on, trains the model on all data so far (including these points along with their optimistic labels), and finally uses the resulting \emph{optimistic} model for decision making. \PLOT{} achieves competitive performance on a set of three challenging benchmark problems, requiring minimal hyperparameter tuning. We also show that \PLOT{} satisfies a logarithmic regret guarantee, under a Lipschitz and logistic mean label model, and under a separability condition on the data.

----

## [504] Visualizing the Emergence of Intermediate Visual Patterns in DNNs

**Authors**: *Mingjie Li, Shaobo Wang, Quanshi Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/33ebd5b07dc7e407752fe773eed20635-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/33ebd5b07dc7e407752fe773eed20635-Abstract.html)

**Abstract**:

This paper proposes a method to visualize the discrimination power of intermediate-layer visual patterns encoded by a DNN. Specifically, we visualize (1) how the DNN gradually learns regional visual patterns in each intermediate layer during the training process, and (2) the effects of the DNN using non-discriminative patterns in low layers to construct disciminative patterns in middle/high layers through the forward propagation. Based on our visualization method, we can quantify knowledge points (i.e. the number of discriminative visual patterns) learned by the DNN to evaluate the representation capacity of the DNN. Furthermore, this method also provides new insights into signal-processing behaviors of existing deep-learning techniques, such as adversarial attacks and knowledge distillation.

----

## [505] Learning 3D Dense Correspondence via Canonical Point Autoencoder

**Authors**: *An-Chieh Cheng, Xueting Li, Min Sun, Ming-Hsuan Yang, Sifei Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3413ce14d52b87557e87e2c1518c2cbe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3413ce14d52b87557e87e2c1518c2cbe-Abstract.html)

**Abstract**:

We propose a canonical point autoencoder (CPAE) that predicts dense correspondences between 3D shapes of the same category. The autoencoder performs two key functions: (a) encoding an arbitrarily ordered point cloud to a canonical primitive, e.g., a sphere, and (b) decoding the primitive back to the original input instance shape. As being placed in the bottleneck, this primitive plays a key role to map all the unordered point clouds on the canonical surface, and to be reconstructed in an ordered fashion. Once trained, points from different shape instances that are mapped to the same locations on the primitive surface are determined to be a pair of correspondence. Our method does not require any form of annotation or self-supervised part segmentation network and can handle unaligned input point clouds within a certain rotation range. Experimental results on 3D semantic keypoint transfer and part segmentation transfer show that our model performs favorably against state-of-the-art correspondence learning methods.

----

## [506] Speech-T: Transducer for Text to Speech and Beyond

**Authors**: *Jiawei Chen, Xu Tan, Yichong Leng, Jin Xu, Guihua Wen, Tao Qin, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/344ef5151be171062f42f03e69663ecf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/344ef5151be171062f42f03e69663ecf-Abstract.html)

**Abstract**:

Neural Transducer (e.g., RNN-T) has been widely used in automatic speech recognition (ASR) due to its capabilities of efficiently modeling monotonic alignments between input and output sequences and naturally supporting streaming inputs. Considering that monotonic alignments are also critical to text to speech (TTS) synthesis and streaming TTS is also an important application scenario, in this work, we explore the possibility of applying Transducer to TTS and more. However, it is challenging because it is difficult to trade off the emission (continuous mel-spectrogram prediction) probability and transition (ASR Transducer predicts blank token to indicate transition to next input) probability when calculating the output probability lattice in Transducer, and it is not easy to learn the alignments between text and speech through the output probability lattice. We propose SpeechTransducer (Speech-T for short), a Transformer based Transducer model that 1) uses a new forward algorithm to separate the transition prediction from the continuous mel-spectrogram prediction when calculating the output probability lattice, and uses a diagonal constraint in the probability lattice to help the alignment learning; 2) supports both full-sentence or streaming TTS by adjusting the look-ahead context; and 3) further supports both TTS and ASR together for the first time, which enjoys several advantages including fewer parameters as well as streaming synthesis and recognition in a single model. Experiments on LJSpeech datasets demonstrate that Speech-T 1) is more robust than the attention based autoregressive TTS model due to its inherent monotonic alignments between text and speech; 2) naturally supports streaming TTS with good voice quality; and 3) enjoys the benefit of joint modeling TTS and ASR in a single network.

----

## [507] Multi-modal Dependency Tree for Video Captioning

**Authors**: *Wentian Zhao, Xinxiao Wu, Jiebo Luo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3473decccb0509fb264818a7512a8b9b-Abstract.html)

**Abstract**:

Generating fluent and relevant language to describe visual content is critical for the video captioning task. Many existing methods generate captions using sequence models that predict words in a left-to-right order. In this paper, we investigate a graph-structured model for caption generation by explicitly modeling the hierarchical structure in the sentences to further improve the fluency and relevance of sentences. To this end, we propose a novel video captioning method that generates a sentence by first constructing a multi-modal dependency tree and then traversing the constructed tree, where the syntactic structure and semantic relationship in the sentence are represented by the tree topology. To take full advantage of the information from both vision and language, both the visual and textual representation features are encoded into each tree node. Different from existing dependency parsing methods that generate uni-modal dependency trees for language understanding, our method construct s multi-modal dependency trees for language generation of images and videos. We also propose a tree-structured reinforcement learning algorithm to effectively optimize the captioning model where a novel reward is designed by evaluating the semantic consistency between the generated sub-tree and the ground-truth tree. Extensive experiments on several video captioning datasets demonstrate the effectiveness of the proposed method.

----

## [508] Greedy and Random Quasi-Newton Methods with Faster Explicit Superlinear Convergence

**Authors**: *Dachao Lin, Haishan Ye, Zhihua Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/347665597cbfaef834886adbb848011f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/347665597cbfaef834886adbb848011f-Abstract.html)

**Abstract**:

In this paper, we follow Rodomanov and Nesterovâ€™s work to study quasi-Newton methods. We focus on the common SR1 and BFGS quasi-Newton methods to establish better explicit (local) superlinear convergence rates. First, based on the greedy quasi-Newton update which greedily selects the direction to maximize a certain measure of progress, we improve the convergence rate to a condition-number-free superlinear convergence rate. Second, based on the random quasi-Newton update that selects the direction randomly from a spherically symmetric distribution, we show the same superlinear convergence rate established as above. Our analysis is closely related to the approximation of a given Hessian matrix, unconstrained quadratic objective, as well as the general strongly convex, smooth, and strongly self-concordant functions.

----

## [509] Neural Tangent Kernel Maximum Mean Discrepancy

**Authors**: *Xiuyuan Cheng, Yao Xie*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/348a38cd25abeab0e440f37510e9b1fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/348a38cd25abeab0e440f37510e9b1fa-Abstract.html)

**Abstract**:

We present a novel neural network Maximum Mean Discrepancy (MMD) statistic by identifying a new connection between neural tangent kernel (NTK) and MMD. This connection enables us to develop a computationally efficient and memory-efficient approach to compute the MMD statistic and perform NTK based two-sample tests towards addressing the long-standing challenge of memory and computational complexity of the MMD statistic, which is essential for online implementation to assimilating new samples. Theoretically, such a connection allows us to understand the NTK test statistic properties, such as the Type-I error and testing power for performing the two-sample test, by adapting existing theories for kernel MMD. Numerical experiments on synthetic and real-world datasets validate the theory and demonstrate the effectiveness of the proposed NTK-MMD statistic.

----

## [510] Subgraph Federated Learning with Missing Neighbor Generation

**Authors**: *Ke Zhang, Carl Yang, Xiaoxiao Li, Lichao Sun, Siu-Ming Yiu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/34adeb8e3242824038aa65460a47c29e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/34adeb8e3242824038aa65460a47c29e-Abstract.html)

**Abstract**:

Graphs have been widely used in data mining and machine learning due to their unique representation of real-world objects and their interactions. As graphs are getting bigger and bigger nowadays, it is common to see their subgraphs separately collected and stored in multiple local systems. Therefore, it is natural to consider the subgraph federated learning setting, where each local system holds a small subgraph that may be biased from the distribution of the whole graph. Hence, the subgraph federated learning aims to collaboratively train a powerful and generalizable graph mining model without directly sharing their graph data. In this work, towards the novel yet realistic setting of subgraph federated learning, we propose two major techniques: (1) FedSage, which trains a GraphSage model based on FedAvg to integrate node features, link structures, and task labels on multiple local subgraphs; (2) FedSage+, which trains a missing neighbor generator along FedSage to deal with missing links across local subgraphs. Empirical results on four real-world graph datasets with synthesized subgraph federated learning settings demonstrate the effectiveness and efficiency of our proposed techniques. At the same time, consistent theoretical implications are made towards their generalization ability on the global graphs.

----

## [511] Bellman-consistent Pessimism for Offline Reinforcement Learning

**Authors**: *Tengyang Xie, Ching-An Cheng, Nan Jiang, Paul Mineiro, Alekh Agarwal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/34f98c7c5d7063181da890ea8d25265a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/34f98c7c5d7063181da890ea8d25265a-Abstract.html)

**Abstract**:

The use of pessimism, when reasoning about datasets lacking exhaustive exploration has recently gained prominence in offline reinforcement learning. Despite the robustness it adds to the algorithm, overly pessimistic reasoning can be equally damaging in precluding the discovery of good policies, which is an issue for the popular bonus-based pessimism. In this paper, we introduce the notion of Bellman-consistent pessimism for general function approximation: instead of calculating a point-wise lower bound for the value function, we implement pessimism at the initial state over the set of functions consistent with the Bellman equations. Our theoretical guarantees only require Bellman closedness as standard in the exploratory setting, in which case bonus-based pessimism fails to provide guarantees.  Even in the special case of linear function approximation where stronger expressivity assumptions hold, our result improves upon a recent bonus-based approach by $\mathcal O(d)$ in its sample complexity (when the action space is finite). Remarkably, our algorithms automatically adapt to the best bias-variance tradeoff in the hindsight, whereas most prior approaches require tuning extra hyperparameters a priori.

----

## [512] Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks

**Authors**: *Avi Schwarzschild, Eitan Borgnia, Arjun Gupta, Furong Huang, Uzi Vishkin, Micah Goldblum, Tom Goldstein*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3501672ebc68a5524629080e3ef60aef-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3501672ebc68a5524629080e3ef60aef-Abstract.html)

**Abstract**:

Deep neural networks are powerful machines for visual pattern recognition, but reasoning tasks that are easy for humans may still be difficult for neural models. Humans possess the ability to extrapolate reasoning strategies learned on simple problems to solve harder examples, often by thinking for longer. For example, a person who has learned to solve small mazes can easily extend the very same search techniques to solve much larger mazes by spending more time.  In computers, this behavior is often achieved through the use of algorithms, which scale to arbitrarily hard problem instances at the cost of more computation. In contrast, the sequential computing budget of feed-forward neural networks is limited by their depth, and networks trained on simple problems have no way of extending their reasoning to accommodate harder problems. In this work, we show that recurrent networks trained to solve simple problems with few recurrent steps can indeed solve much more complex problems simply by performing additional recurrences during inference. We demonstrate this algorithmic behavior of recurrent networks on prefix sum computation, mazes, and chess.  In all three domains, networks trained on simple problem instances are able to extend their reasoning abilities at test time simply by "thinking for longer."

----

## [513] Sub-Linear Memory: How to Make Performers SLiM

**Authors**: *Valerii Likhosherstov, Krzysztof Marcin Choromanski, Jared Quincy Davis, Xingyou Song, Adrian Weller*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/35309226eb45ec366ca86a4329a2b7c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/35309226eb45ec366ca86a4329a2b7c3-Abstract.html)

**Abstract**:

Transformer architectures have become very popular yet the original implementation requires  $O(L^2)$ in serial time and memory as functions of input length $L$. Recent works proposed various linear self-attention mechanisms, scaling only as $O(L)$ for serial computation. We conduct a thorough complexity analysis of Performers, a class which includes most recent linear Transformer mechanisms. We note a remarkable computational flexibility: the gradient computation can be performed with no approximations using sublinear memory as a function of $L$ (in addition to negligible storage for the input sequence), at a cost of greater time complexity in the parallel setting. In the extreme case, a Performer consumes only $O(1)$ memory, and still requires $O(L)$ time. Due to complete backward-compatibility, this discovered time-memory tradeoff can be used for fine-tuning on low-memory devices in a decentralized fashion without any server computations.

----

## [514] Efficient Learning of Discrete-Continuous Computation Graphs

**Authors**: *David Friede, Mathias Niepert*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3556a3018cce3076e27dbbf9645b44d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3556a3018cce3076e27dbbf9645b44d5-Abstract.html)

**Abstract**:

Numerous models for supervised and reinforcement learning benefit from combinations of discrete and continuous model components. End-to-end learnable discrete-continuous models are compositional, tend to generalize better, and are more interpretable. A popular approach to building discrete-continuous computation graphs is that of integrating discrete probability distributions into neural networks using stochastic softmax tricks. Prior work has mainly focused on computation graphs with a single discrete component on each of the graph's execution paths. We analyze the behavior of more complex stochastic computations graphs with multiple sequential discrete components. We show that it is challenging to optimize the parameters of these models, mainly due to small  gradients and local minima. We then propose two new strategies to overcome these challenges. First, we show that increasing the scale parameter of the Gumbel noise perturbations during training improves the learning behavior. Second, we propose dropout residual connections specifically tailored to stochastic, discrete-continuous computation graphs. With an extensive set of experiments, we show that we can train complex discrete-continuous models which one cannot train with standard stochastic softmax tricks. We also show that complex discrete-stochastic models generalize better than their continuous counterparts on several benchmark datasets.

----

## [515] VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using Vector Quantization

**Authors**: *Mucong Ding, Kezhi Kong, Jingling Li, Chen Zhu, John Dickerson, Furong Huang, Tom Goldstein*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3569df159ec477451530c4455b2a9e86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3569df159ec477451530c4455b2a9e86-Abstract.html)

**Abstract**:

Most state-of-the-art Graph Neural Networks (GNNs) can be defined as a form of graph convolution which can be realized by message passing between direct neighbors or beyond. To scale such GNNs to large graphs, various neighbor-, layer-, or subgraph-sampling techniques are proposed to alleviate the "neighbor explosion" problem by considering only a small subset of messages passed to the nodes in a mini-batch. However, sampling-based methods are difficult to apply to GNNs that utilize many-hops-away or global context each layer, show unstable performance for different tasks and datasets, and do not speed up model inference. We propose a principled and fundamentally different approach, VQ-GNN, a universal framework to scale up any convolution-based GNNs using Vector Quantization (VQ) without compromising the performance. In contrast to sampling-based techniques, our approach can effectively preserve all the messages passed to a mini-batch of nodes by learning and updating a small number of quantized reference vectors of global node representations, using VQ within each GNN layer. Our framework avoids the "neighbor explosion" problem of GNNs using quantized representations combined with a low-rank version of the graph convolution matrix. We show that such a compact low-rank version of the gigantic convolution matrix is sufficient both theoretically and experimentally. In company with VQ, we design a novel approximated message passing algorithm and a nontrivial back-propagation rule for our framework. Experiments on various types of GNN backbones demonstrate the scalability and competitive performance of our framework on large-graph node classification and link prediction benchmarks.

----

## [516] Overcoming Catastrophic Forgetting in Incremental Few-Shot Learning by Finding Flat Minima

**Authors**: *Guangyuan Shi, Jiaxin Chen, Wenlong Zhang, Li-Ming Zhan, Xiao-Ming Wu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/357cfba15668cc2e1e73111e09d54383-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/357cfba15668cc2e1e73111e09d54383-Abstract.html)

**Abstract**:

This paper considers incremental few-shot learning, which requires a model to continually recognize new categories with only a few examples provided. Our study shows that existing methods severely suffer from catastrophic forgetting, a well-known problem in incremental learning, which is aggravated due to data scarcity and imbalance in the few-shot setting. Our analysis further suggests that to prevent catastrophic forgetting, actions need to be taken in the primitive stage -- the training of base classes instead of later few-shot learning sessions. Therefore, we propose to search for flat local minima of the base training objective function and then fine-tune the model parameters within the flat region on new tasks. In this way, the model can efficiently learn new classes while preserving the old ones. Comprehensive experimental results demonstrate that our approach outperforms all prior state-of-the-art methods and is very close to the approximate upper bound. The source code is available at https://github.com/moukamisama/F2M.

----

## [517] Functional Neural Networks for Parametric Image Restoration Problems

**Authors**: *Fangzhou Luo, Xiaolin Wu, Yanhui Guo*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/35936504a37d53e03abdfbc7318d9ec7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/35936504a37d53e03abdfbc7318d9ec7-Abstract.html)

**Abstract**:

Almost every single image restoration problem has a closely related parameter, such as the scale factor in super-resolution, the noise level in image denoising, and the quality factor in JPEG deblocking. Although recent studies on image restoration problems have achieved great success due to the development of deep neural networks, they handle the parameter involved in an unsophisticated way. Most previous researchers either treat problems with different parameter levels as independent tasks, and train a specific model for each parameter level; or simply ignore the parameter, and train a single model for all parameter levels. The two popular approaches have their own shortcomings. The former is inefficient in computing and the latter is ineffective in performance. In this work, we propose a novel system called functional neural network (FuncNet) to solve a parametric image restoration problem with a single model. Unlike a plain neural network, the smallest conceptual element of our FuncNet is no longer a floating-point variable, but a function of the parameter of the problem. This feature makes it both efficient and effective for a parametric problem. We apply FuncNet to super-resolution, image denoising, and JPEG deblocking. The experimental results show the superiority of our FuncNet on all three parametric image restoration tasks over the state of the arts.

----

## [518] Intrinsic Dimension, Persistent Homology and Generalization in Neural Networks

**Authors**: *Tolga Birdal, Aaron Lou, Leonidas J. Guibas, Umut Simsekli*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/35a12c43227f217207d4e06ffefe39d3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/35a12c43227f217207d4e06ffefe39d3-Abstract.html)

**Abstract**:

Disobeying the classical wisdom of statistical learning theory, modern deep neural networks generalize well even though they typically contain millions of parameters. Recently, it has been shown that the trajectories of iterative optimization algorithms can possess \emph{fractal structures}, and their generalization error can be formally linked to the complexity of such fractals. This complexity is measured by the fractal's \emph{intrinsic dimension}, a quantity usually much smaller than the number of parameters in the network. Even though this perspective provides an explanation for why overparametrized networks would not overfit, computing the intrinsic dimension (\eg, for monitoring generalization during training) is a notoriously difficult task,  where existing methods typically fail even in moderate ambient dimensions. In this study, we consider this problem from the lens of topological data analysis (TDA) and develop a generic computational tool that is built on rigorous mathematical foundations. By making a novel connection between learning theory and TDA, we first illustrate that the generalization error can be equivalently bounded in terms of a notion called the 'persistent homology dimension' (PHD), where, compared with prior work, our approach does not require any additional geometrical or statistical assumptions on the training dynamics. Then, by utilizing recently established theoretical results and TDA tools, we develop an efficient algorithm to estimate PHD in the scale of modern deep neural networks and further provide visualization tools to help understand generalization in deep learning. Our experiments show that the proposed approach can efficiently compute a network's intrinsic dimension in a variety of settings, which is predictive of the generalization error.

----

## [519] GemNet: Universal Directional Graph Neural Networks for Molecules

**Authors**: *Johannes Gasteiger, Florian Becker, Stephan Günnemann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html)

**Abstract**:

Effectively predicting molecular interactions has the potential to accelerate molecular dynamics by multiple orders of magnitude and thus revolutionize chemical simulations. Graph neural networks (GNNs) have recently shown great successes for this task, overtaking classical methods based on fixed molecular kernels. However, they still appear very limited from a theoretical perspective, since regular GNNs cannot distinguish certain types of graphs. In this work we close this gap between theory and practice. We show that GNNs with directed edge embeddings and two-hop message passing are indeed universal approximators for predictions that are invariant to translation, and equivariant to permutation and rotation. We then leverage these insights and multiple structural improvements to propose the geometric message passing neural network (GemNet). We demonstrate the benefits of the proposed changes in multiple ablation studies. GemNet outperforms previous models on the COLL, MD17, and OC20 datasets by 34%, 41%, and 20%, respectively, and performs especially well on the most challenging molecules. Our implementation is available online.

----

## [520] Loss function based second-order Jensen inequality and its application to particle variational inference

**Authors**: *Futoshi Futami, Tomoharu Iwata, Naonori Ueda, Issei Sato, Masashi Sugiyama*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/36165c62f7b7df72863d470d73302627-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/36165c62f7b7df72863d470d73302627-Abstract.html)

**Abstract**:

Bayesian model averaging, obtained as the expectation of a likelihood function by a posterior distribution, has been widely used for prediction, evaluation of uncertainty, and model selection. Various approaches have been developed to efficiently capture the information in the posterior distribution; one such approach is the optimization of a set of models simultaneously with interaction to ensure the diversity of the individual models in the same way as ensemble learning. A representative approach is particle variational inference (PVI), which uses an ensemble of models as an empirical approximation for the posterior distribution. PVI iteratively updates each model with a repulsion force to ensure the diversity of the optimized models. However, despite its promising performance, a theoretical understanding of this repulsion and its association with the generalization ability remains unclear. In this paper, we tackle this problem in light of PAC-Bayesian analysis. First, we provide a new second-order Jensen inequality, which has the repulsion term based on the loss function. Thanks to the repulsion term, it is tighter than the standard Jensen inequality. Then, we derive a novel generalization error bound and show that it can be reduced by enhancing the diversity of models. Finally, we derive a new PVI that optimizes the generalization error bound directly. Numerical experiments demonstrate that the performance of the proposed PVI compares favorably with existing methods in the experiment.

----

## [521] Detecting and Adapting to Irregular Distribution Shifts in Bayesian Online Learning

**Authors**: *Aodong Li, Alex Boyd, Padhraic Smyth, Stephan Mandt*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/362387494f6be6613daea643a7706a42-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/362387494f6be6613daea643a7706a42-Abstract.html)

**Abstract**:

We consider the problem of online learning in the presence of distribution shifts that occur at an unknown rate and of unknown intensity. We derive a new Bayesian online inference approach to simultaneously infer these distribution shifts and adapt the model to the detected changes by integrating ideas from change point detection, switching dynamical systems, and Bayesian online learning. Using a binary ‘change variable,’ we construct an informative prior such that--if a change is detected--the model partially erases the information of past model updates by tempering to facilitate adaptation to the new data distribution. Furthermore, the approach uses beam search to track multiple change-point hypotheses and selects the most probable one in hindsight. Our proposed method is model-agnostic, applicable in both supervised and unsupervised learning settings, suitable for an environment of concept drifts or covariate drifts, and yields improvements over state-of-the-art Bayesian online learning approaches.

----

## [522] Asynchronous Decentralized SGD with Quantized and Local Updates

**Authors**: *Giorgi Nadiradze, Amirmojtaba Sabour, Peter Davies, Shigang Li, Dan Alistarh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/362c99307cdc3f2d8b410652386a9dd1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/362c99307cdc3f2d8b410652386a9dd1-Abstract.html)

**Abstract**:

Decentralized optimization is emerging as a viable alternative for scalable distributed machine learning, but also introduces new challenges in terms of synchronization costs.  To this end, several communication-reduction techniques, such as non-blocking communication, quantization, and local steps, have been explored in the decentralized setting. Due to the complexity of analyzing optimization in such a relaxed setting, this line of work often assumes \emph{global} communication rounds, which require additional synchronization. In this paper, we consider decentralized optimization in the simpler, but harder to analyze, \emph{asynchronous gossip} model, in which communication occurs in discrete, randomly chosen pairings among nodes. Perhaps surprisingly, we show that a variant of SGD called \emph{SwarmSGD} still converges in this setting, even if \emph{non-blocking communication}, \emph{quantization}, and \emph{local steps} are all applied \emph{in conjunction}, and even if the node data distributions and underlying graph topology are both \emph{heterogenous}. Our analysis is based on a new connection with multi-dimensional load-balancing processes. We implement this algorithm and deploy it in a super-computing environment, showing that it can outperform previous decentralized methods in terms of end-to-end training time, and that it can even rival carefully-tuned large-batch SGD for certain tasks.

----

## [523] Stochastic Shortest Path: Minimax, Parameter-Free and Towards Horizon-Free Regret

**Authors**: *Jean Tarbouriech, Runlong Zhou, Simon S. Du, Matteo Pirotta, Michal Valko, Alessandro Lazaric*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/367147f1755502d9bc6189f8e2c3005d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/367147f1755502d9bc6189f8e2c3005d-Abstract.html)

**Abstract**:

We study the problem of learning in the stochastic shortest path (SSP) setting, where an agent seeks to minimize the expected cost accumulated before reaching a goal state. We design a novel model-based algorithm EB-SSP that carefully skews the empirical transitions and perturbs the empirical costs with an exploration bonus to induce an optimistic SSP problem whose associated value iteration scheme is guaranteed to converge. We prove that EB-SSP achieves the minimax regret rate $\widetilde{O}(B_{\star} \sqrt{S A K})$, where $K$ is the number of episodes, $S$ is the number of states, $A$ is the number of actions and $B_{\star}$ bounds the expected cumulative cost of the optimal policy from any state, thus closing the gap with the lower bound. Interestingly, EB-SSP obtains this result while being parameter-free, i.e., it does not require any prior knowledge of $B_{\star}$, nor of $T_{\star}$, which bounds the expected time-to-goal of the optimal policy from any state. Furthermore, we illustrate various cases (e.g., positive costs, or general costs when an order-accurate estimate of $T_{\star}$ is available) where the regret only contains a logarithmic dependence on $T_{\star}$, thus yielding the first (nearly) horizon-free regret bound beyond the finite-horizon MDP setting.

----

## [524] Nested Counterfactual Identification from Arbitrary Surrogate Experiments

**Authors**: *Juan D. Correa, Sanghack Lee, Elias Bareinboim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/36bedb6eb7152f39b16328448942822b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/36bedb6eb7152f39b16328448942822b-Abstract.html)

**Abstract**:

The Ladder of Causation describes three qualitatively different types of activities an agent may be interested in engaging in, namely, seeing (observational), doing (interventional), and imagining (counterfactual) (Pearl and Mackenzie, 2018). The inferential challenge imposed by the causal hierarchy is that data is collected by an agent observing or intervening in a system (layers 1 and 2), while its goal may be to understand what would have happened had it taken a different course of action, contrary to what factually ended up happening (layer 3). While there exists a solid understanding of the conditions under which cross-layer inferences are allowed from observations to interventions, the results are somewhat scarcer when targeting counterfactual quantities. In this paper, we study the identification of nested counterfactuals from an arbitrary combination of observations and experiments. Specifically, building on a more explicit definition of nested counterfactuals, we prove the counterfactual unnesting theorem (CUT), which allows one to map arbitrary nested counterfactuals to unnested ones. For instance, applications in mediation and fairness analysis usually evoke notions of direct, indirect, and spurious effects, which naturally require nesting. Second, we introduce a sufficient and necessary graphical condition for counterfactual identification from an arbitrary combination of observational and experimental distributions. Lastly, we develop an efficient and complete algorithm for identifying nested counterfactuals; failure of the algorithm returning an expression for a query implies it is not identifiable.

----

## [525] Sim and Real: Better Together

**Authors**: *Shirli Di-Castro Shashua, Dotan Di Castro, Shie Mannor*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/36f4d832825380f102846560a5104c90-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/36f4d832825380f102846560a5104c90-Abstract.html)

**Abstract**:

Simulation is used extensively in autonomous systems, particularly in robotic manipulation. By far, the most common approach is to train a controller in simulation, and then use it as an initial starting point for the real system. We demonstrate how to learn simultaneously from both simulation and interaction with the real environment. We propose an algorithm for balancing the large number of samples from the high throughput but less accurate simulation and the low-throughput, high-fidelity and costly samples from the real environment. We achieve that by maintaining a replay buffer for each environment the agent interacts with. We analyze such multi-environment interaction theoretically, and provide convergence properties, through a novel theoretical replay buffer analysis.  We demonstrate the efficacy of our method on a sim-to-real environment.

----

## [526] Trustworthy Multimodal Regression with Mixture of Normal-inverse Gamma Distributions

**Authors**: *Huan Ma, Zongbo Han, Changqing Zhang, Huazhu Fu, Joey Tianyi Zhou, Qinghua Hu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/371bce7dc83817b7893bcdeed13799b5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/371bce7dc83817b7893bcdeed13799b5-Abstract.html)

**Abstract**:

Multimodal regression is a fundamental task, which integrates the information from different sources to improve the performance of follow-up applications. However, existing methods mainly focus on improving the performance and often ignore the confidence of prediction for diverse situations. In this study, we are devoted to trustworthy multimodal regression which is critical in cost-sensitive domains. To this end, we introduce a novel Mixture of Normal-Inverse Gamma distributions (MoNIG) algorithm, which efficiently estimates uncertainty in principle for adaptive integration of different modalities and produces a trustworthy regression result. Our model can be dynamically aware of uncertainty for each modality, and also robust for corrupted modalities. Furthermore, the proposed MoNIG ensures explicitly representation of (modality-specific/global) epistemic and aleatoric uncertainties, respectively. Experimental results on both synthetic and different real-world data demonstrate the effectiveness and trustworthiness of our method on various multimodal regression tasks (e.g., temperature prediction for superconductivity, relative location prediction for CT slices, and multimodal sentiment analysis).

----

## [527] An Empirical Study of Adder Neural Networks for Object Detection

**Authors**: *Xinghao Chen, Chang Xu, Minjing Dong, Chunjing Xu, Yunhe Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/37693cfc748049e45d87b8c7d8b9aacd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/37693cfc748049e45d87b8c7d8b9aacd-Abstract.html)

**Abstract**:

Adder neural networks (AdderNets) have shown impressive performance on image classification with only addition operations, which are more energy efficient than traditional convolutional neural networks built with multiplications. Compared with classification, there is a strong demand on reducing the energy consumption of modern object detectors via AdderNets for real-world applications such as autonomous driving and face detection. In this paper, we present an empirical study of AdderNets for object detection. We first reveal that the batch normalization statistics in the pre-trained adder backbone should not be frozen, since the relatively large feature variance of AdderNets. Moreover, we insert more shortcut connections in the neck part and design a new feature fusion architecture for avoiding the sparse features of adder layers. We present extensive ablation studies to explore several design choices of adder detectors. Comparisons with state-of-the-arts are conducted on COCO and PASCAL VOC benchmarks. Specifically, the proposed Adder FCOS achieves a 37.8% AP on the COCO val set, demonstrating comparable performance to that of the convolutional counterpart with an about $1.4\times$ energy reduction.

----

## [528] Does Knowledge Distillation Really Work?

**Authors**: *Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/376c6b9ff3bedbbea56751a84fffc10c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/376c6b9ff3bedbbea56751a84fffc10c-Abstract.html)

**Abstract**:

Knowledge distillation is a popular technique for training a small student network to emulate a larger teacher model, such as an ensemble of networks. We show that while knowledge distillation can improve student generalization, it does not typically work as it is commonly understood: there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the student, even in cases when the student has the capacity to perfectly match the teacher. We identify difficulties in optimization as a key reason for why the student is unable to match the teacher. We also show how the details of the dataset used for distillation play a role in how closely the student matches the teacher --- and that more closely matching the teacher paradoxically does not always lead to better student generalization.

----

## [529] Teachable Reinforcement Learning via Advice Distillation

**Authors**: *Olivia Watkins, Abhishek Gupta, Trevor Darrell, Pieter Abbeel, Jacob Andreas*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/37cfff3c04f95b22bcf166df586cd7a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/37cfff3c04f95b22bcf166df586cd7a9-Abstract.html)

**Abstract**:

Training automated agents to complete complex tasks in interactive environments is challenging: reinforcement learning requires careful hand-engineering of reward functions, imitation learning requires specialized infrastructure and access to a human expert, and learning from intermediate forms of supervision (like binary preferences) is time-consuming and extracts little information from each human intervention. Can we overcome these challenges by building agents that learn from rich, interactive feedback instead? We propose a new supervision paradigm for interactive learning based on "teachable" decision-making systems that learn from structured advice provided by an external teacher. We begin by formalizing a class of human-in-the-loop decision making problems in which multiple forms of teacher-provided advice are available to a learner. We then describe a simple learning algorithm for these problems that first learns to interpret advice, then learns from advice to complete tasks even in the absence of human supervision. In puzzle-solving, navigation, and locomotion domains, we show that agents that learn from advice can acquire new skills with significantly less human supervision than standard reinforcement learning algorithms and often less than imitation learning.

----

## [530] Antipodes of Label Differential Privacy: PATE and ALIBI

**Authors**: *Mani Malek Esmaeili, Ilya Mironov, Karthik Prasad, Igor Shilov, Florian Tramèr*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/37ecd27608480aa3569a511a638ca74f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/37ecd27608480aa3569a511a638ca74f-Abstract.html)

**Abstract**:

We consider the privacy-preserving machine learning (ML) setting where the trained model must satisfy differential privacy (DP) with respect to the labels of the training examples. We propose two novel approaches based on, respectively, the Laplace mechanism and the PATE framework, and demonstrate their effectiveness on standard benchmarks.While recent work by Ghazi et al. proposed Label DP schemes based on a randomized response mechanism, we argue that additive Laplace noise coupled with Bayesian inference (ALIBI) is a better fit for typical ML tasks. Moreover, we show how to achieve very strong privacy levels in some regimes, with our adaptation of the PATE framework that builds on recent advances in semi-supervised learning.We complement theoretical analysis of our algorithms' privacy guarantees with empirical evaluation of their memorization properties. Our evaluation suggests that comparing different algorithms according to their provable DP guarantees can be misleading and favor a less private algorithm with a tighter analysis.Code for implementation of algorithms and memorization attacks is available from https://github.com/facebookresearch/labeldpantipodes.

----

## [531] Visual Search Asymmetry: Deep Nets and Humans Share Similar Inherent Biases

**Authors**: *Shashi Kant Gupta, Mengmi Zhang, Chia-Chien Wu, Jeremy M. Wolfe, Gabriel Kreiman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/37f0e884fbad9667e38940169d0a3c95-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/37f0e884fbad9667e38940169d0a3c95-Abstract.html)

**Abstract**:

Visual search is a ubiquitous and often challenging daily task, exemplified by looking for the car keys at home or a friend in a crowd. An intriguing property of some classical search tasks is an asymmetry such that finding a target A among distractors B can be easier than finding B among A. To elucidate the mechanisms responsible for asymmetry in visual search, we propose a computational model that takes a target and a search image as inputs and produces a sequence of eye movements until the target is found. The model integrates eccentricity-dependent visual recognition with target-dependent top-down cues. We compared the model against human behavior in six paradigmatic search tasks that show asymmetry in humans. Without prior exposure to the stimuli or task-specific training, the model provides a plausible mechanism for search asymmetry. We hypothesized that the polarity of search asymmetry arises from experience with the natural environment. We tested this hypothesis by training the model on augmented versions of ImageNet where the biases of natural images were either removed or reversed. The polarity of search asymmetry disappeared or was altered depending on the training protocol. This study highlights how classical perceptual properties can emerge in neural network models, without the need for task-specific training, but rather as a consequence of the statistical properties of the developmental diet fed to the model. All source code and data are publicly available at https://github.com/kreimanlab/VisualSearchAsymmetry.

----

## [532] On the Universality of Graph Neural Networks on Large Random Graphs

**Authors**: *Nicolas Keriven, Alberto Bietti, Samuel Vaiter*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/38181d991caac98be8fb2ecb8bd0f166-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/38181d991caac98be8fb2ecb8bd0f166-Abstract.html)

**Abstract**:

We study the approximation power of Graph Neural Networks (GNNs) on latent position random graphs. In the large graph limit, GNNs are known to converge to certain ``continuous'' models known as c-GNNs, which directly enables a study of their approximation power on random graph models. In the absence of input node features however, just as GNNs are limited by the Weisfeiler-Lehman isomorphism test, c-GNNs will be severely limited on simple random graph models. For instance, they will fail to distinguish the communities of a well-separated Stochastic Block Model (SBM) with constant degree function. Thus, we consider recently proposed architectures that augment GNNs with unique node identifiers, referred to as Structural GNNs here (SGNNs). We study the convergence of SGNNs to their continuous counterpart (c-SGNNs) in the large random graph limit, under new conditions on the node identifiers. We then show that c-SGNNs are strictly more powerful than c-GNNs in the continuous limit, and prove their universality on several random graph models of interest, including most SBMs and a large class of random geometric graphs. Our results cover both permutation-invariant and permutation-equivariant architectures.

----

## [533] Inverse Reinforcement Learning in a Continuous State Space with Formal Guarantees

**Authors**: *Gregory Dexter, Kevin Bello, Jean Honorio*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/384babc3e7faa44cf1ca671b74499c3b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/384babc3e7faa44cf1ca671b74499c3b-Abstract.html)

**Abstract**:

Inverse Reinforcement Learning (IRL) is the problem of finding a reward function which describes observed/known expert behavior.  The IRL setting is remarkably useful for automated control, in situations where the reward function is difficult to specify manually or as a means to extract agent preference. In this work, we provide a new IRL algorithm for the continuous state space setting with unknown transition dynamics by modeling the system using a basis of orthonormal functions. Moreover, we provide a proof of correctness and formal guarantees on the sample and time complexity of our algorithm.  Finally, we present synthetic experiments to corroborate our theoretical guarantees.

----

## [534] Adversarial Attacks on Graph Classifiers via Bayesian Optimisation

**Authors**: *Xingchen Wan, Henry Kenlay, Robin Ru, Arno Blaas, Michael A. Osborne, Xiaowen Dong*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/38811c5285e34e2e3319ab7d9f2cfa5b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/38811c5285e34e2e3319ab7d9f2cfa5b-Abstract.html)

**Abstract**:

Graph neural networks, a popular class of models effective in a wide range of graph-based learning tasks, have been shown to be vulnerable to adversarial attacks. While the majority of the literature focuses on such vulnerability in node-level classification tasks, little effort has been dedicated to analysing adversarial attacks on graph-level classification, an important problem with numerous real-life applications such as biochemistry and social network analysis. The few existing methods often require unrealistic setups, such as access to internal information of the victim models, or an impractically-large number of queries. We present a novel Bayesian optimisation-based attack method for graph classification models. Our method is black-box, query-efficient and parsimonious with respect to the perturbation applied. We empirically validate the effectiveness and flexibility of the proposed method on a wide range of graph classification tasks involving varying graph properties, constraints and modes of attack. Finally, we analyse common interpretable patterns behind the adversarial samples produced, which may shed further light on the adversarial robustness of graph classification models.

----

## [535] Regulating algorithmic filtering on social media

**Authors**: *Sarah Huiyi Cen, Devavrat Shah*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/38b4f06e27fd4f6fdcceabc6f5c068ea-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/38b4f06e27fd4f6fdcceabc6f5c068ea-Abstract.html)

**Abstract**:

By filtering the content that users see, social media platforms have the ability to influence users' perceptions and decisions, from their dining choices to their voting preferences. This influence has drawn scrutiny, with many calling for regulations on filtering algorithms, but designing and enforcing regulations remains challenging. In this work, we examine three questions. First, given a regulation, how would one design an audit to enforce it? Second, does the audit impose a performance cost on the platform? Third, how does the audit affect the content that the platform is incentivized to filter? In response to these questions, we propose a method such that, given a regulation, an auditor can test whether that regulation is met with only black-box access to the filtering algorithm. We then turn to the platform's perspective. The platform's goal is to maximize an objective function while meeting regulation. We find that there are conditions under which the regulation does not place a high performance cost on the platform and, notably, that content diversity can play a key role in aligning the interests of the platform and regulators.

----

## [536] argmax centroid

**Authors**: *Chengyue Gong, Mao Ye, Qiang Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/38eb982ee635354d3febf457beeee736-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/38eb982ee635354d3febf457beeee736-Abstract.html)

**Abstract**:

We propose a general method to construct centroid approximation for the distribution of maximum points of a random function (a.k.a. argmax distribution), which finds broad applications in machine learning. Our method optimizes a set of centroid points to compactly approximate the argmax distribution with a simple objective function, without explicitly drawing exact samples from the argmax distribution. Theoretically, the argmax centroid method can be shown to minimize a surrogate of Wasserstein distance between the ground-truth argmax distribution and the centroid approximation under proper conditions. We demonstrate the applicability and effectiveness of our method on a variety of real-world multi-task learning applications, including few-shot image classification, personalized dialogue systems and multi-target domain adaptation.

----

## [537] Contrastive Learning of Global and Local Video Representations

**Authors**: *Shuang Ma, Zhaoyang Zeng, Daniel McDuff, Yale Song*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/38ef4b66cb25e92abe4d594acb841471-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/38ef4b66cb25e92abe4d594acb841471-Abstract.html)

**Abstract**:

Contrastive learning has delivered impressive results for various tasks in the self-supervised regime. However, existing approaches optimize for learning representations specific to downstream scenarios, i.e., global representations suitable for tasks such as classification or local representations for tasks such as detection and localization. While they produce satisfactory results in the intended downstream scenarios, they often fail to generalize to tasks that they were not originally designed for. In this work, we propose to learn video representations that generalize to both the tasks which require global semantic information (e.g., classification) and the tasks that require local fine-grained spatio-temporal information (e.g., localization). We achieve this by optimizing two contrastive objectives that together encourage our model to learn global-local visual information given audio signals. We show that the two objectives mutually improve the generalizability of the learned global-local representations, significantly outperforming their disjointly learned counterparts. We demonstrate our approach on various tasks including action/sound classification, lipreading, deepfake detection, event and sound localization.

----

## [538] BooVI: Provably Efficient Bootstrapped Value Iteration

**Authors**: *Boyi Liu, Qi Cai, Zhuoran Yang, Zhaoran Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/39144da5a6180c47885443c83547ec14-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/39144da5a6180c47885443c83547ec14-Abstract.html)

**Abstract**:

Despite the tremendous success of reinforcement learning (RL) with function approximation, efficient exploration remains a significant challenge, both practically and theoretically. In particular, existing theoretically grounded RL algorithms based on upper confidence bounds (UCBs), such as optimistic least-squares value iteration (LSVI), are often incompatible with practically powerful function approximators, such as neural networks. In this paper, we develop a variant of \underline{boo}tstrapped LS\underline{VI}, namely BooVI, which bridges such a gap between practice and theory. Practically, BooVI drives exploration through (re)sampling, making it compatible with general function approximators. Theoretically, BooVI inherits the worst-case $\tilde{O}(\sqrt{d^3 H^3 T})$-regret of optimistic LSVI in the episodic linear setting. Here $d$ is the feature dimension, $H$ is the episode horizon, and $T$ is the total number of steps.

----

## [539] Do Wider Neural Networks Really Help Adversarial Robustness?

**Authors**: *Boxi Wu, Jinghui Chen, Deng Cai, Xiaofei He, Quanquan Gu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3937230de3c8041e4da6ac3246a888e8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3937230de3c8041e4da6ac3246a888e8-Abstract.html)

**Abstract**:

Adversarial training is a powerful type of defense against adversarial examples. Previous empirical results suggest that adversarial training requires wider networks for better performances. However, it remains elusive how does neural network width affect model robustness. In this paper, we carefully examine the relationship between network width and model robustness. Specifically, we show that the model robustness is closely related to the tradeoff between natural accuracy and perturbation stability, which is controlled by the robust regularization parameter λ. With the same λ, wider networks can achieve better natural accuracy but worse perturbation stability, leading to a potentially worse overall model robustness. To understand the origin of this phenomenon, we further relate the perturbation stability with the network's local Lipschitzness. By leveraging recent results on neural tangent kernels, we theoretically show that wider networks tend to have worse perturbation stability. Our analyses suggest that: 1) the common strategy of first fine-tuning λ on small networks and then directly use it for wide model training could lead to deteriorated model robustness; 2) one needs to properly enlarge λ to unleash the robustness potential of wider models fully. Finally, we propose a new Width Adjusted Regularization (WAR) method that adaptively enlarges λ on wide models and significantly saves the tuning time.

----

## [540] Exploring the Limits of Out-of-Distribution Detection

**Authors**: *Stanislav Fort, Jie Ren, Balaji Lakshminarayanan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3941c4358616274ac2436eacf67fae05-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3941c4358616274ac2436eacf67fae05-Abstract.html)

**Abstract**:

Near out-of-distribution detection (OOD) is a major challenge for deep neural networks. We demonstrate that large-scale pre-trained transformers can significantly improve the state-of-the-art (SOTA) on a range of near OOD tasks across different data modalities. For instance, on CIFAR-100 vs CIFAR-10 OOD detection, we improve the AUROC from 85% (current SOTA) to more than 96% using Vision Transformers pre-trained on ImageNet21k. On a challenging genomics OOD detection benchmark, we improve the AUROC from 66% to 77% using transformer and unsupervised pre-training.  To further improve performance, we explore the few-shot outlier exposure setting where a few examples from outlier classes may be available; we show that  pre-trained transformers are particularly well-suited for outlier exposure, and that the AUROC of OOD detection on CIFAR-100 vs CIFAR-10  can be improved to 98.7% with just 1 image per OOD class, and 99.46% with 10 images per OOD class. For multi-modal image-text pre-trained transformers such as CLIP, we explore a new way of using just the names of outlier classes as a sole source of information without any accompanying images, and show that this outperforms previous SOTA on standard OOD benchmark tasks.

----

## [541] ABC: Auxiliary Balanced Classifier for Class-imbalanced Semi-supervised Learning

**Authors**: *Hyuck Lee, Seungjae Shin, Heeyoung Kim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3953630da28e5181cffca1278517e3cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3953630da28e5181cffca1278517e3cf-Abstract.html)

**Abstract**:

Existing semi-supervised learning (SSL) algorithms typically assume class-balanced datasets, although the class distributions of many real world datasets are imbalanced. In general, classifiers trained on a class-imbalanced dataset are biased toward the majority classes. This issue becomes more problematic for SSL algorithms because they utilize the biased prediction of unlabeled data for training. However, traditional class-imbalanced learning techniques, which are designed for labeled data, cannot be readily combined with SSL algorithms. We propose a scalable class-imbalanced SSL algorithm that can effectively use unlabeled data, while mitigating class imbalance by introducing an auxiliary balanced classifier (ABC) of a single layer, which is attached to a representation layer of an existing SSL algorithm. The ABC is trained with a class-balanced loss of a minibatch, while using high-quality representations learned from all data points in the minibatch using the backbone SSL algorithm to avoid overfitting and information loss. Moreover, we use consistency regularization, a recent SSL technique for utilizing unlabeled data in a modified way, to train the ABC to be balanced among the classes by selecting unlabeled data with the same probability for each class. The proposed algorithm achieves state-of-the-art performance in various class-imbalanced SSL experiments using four benchmark datasets.

----

## [542] BCD Nets: Scalable Variational Approaches for Bayesian Causal Discovery

**Authors**: *Chris Cundy, Aditya Grover, Stefano Ermon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/39799c18791e8d7eb29704fc5bc04ac8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/39799c18791e8d7eb29704fc5bc04ac8-Abstract.html)

**Abstract**:

A structural equation model (SEM) is an effective framework to reason over causal relationships represented via a directed acyclic graph (DAG).Recent advances have enabled effective maximum-likelihood point estimation of DAGs from observational data. However, a point estimate may not accurately capture the uncertainty in inferring the underlying graph in practical scenarios, wherein the true DAG is non-identifiable and/or the observed dataset is limited.We propose Bayesian Causal Discovery Nets (BCD Nets), a variational inference framework for estimating a distribution over DAGs characterizing a linear-Gaussian SEM.Developing a full Bayesian posterior over DAGs is challenging due to the the discrete and combinatorial nature of graphs.We analyse key design choices for scalable VI over DAGs, such as 1) the parametrization of DAGs via an expressive variational family, 2) a continuous relaxation that enables low-variance stochastic optimization, and 3) suitable priors over the latent variables.We provide a series of experiments on real and synthetic data showing that BCD Nets outperform maximum-likelihood methods on standard causal discovery metrics such as structural Hamming distance in low data regimes.

----

## [543] Discovering Dynamic Salient Regions for Spatio-Temporal Graph Neural Networks

**Authors**: *Iulia Duta, Andrei Liviu Nicolicioiu, Marius Leordeanu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/398410ece9d7343091093a2a7f8ee381-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/398410ece9d7343091093a2a7f8ee381-Abstract.html)

**Abstract**:

Graph Neural Networks are perfectly suited to capture latent interactions between various entities in the spatio-temporal domain (e.g. videos). However, when an explicit structure is not available, it is not obvious what atomic elements should be represented as nodes. Current works generally use pre-trained object detectors or fixed, predefined regions to extract graph nodes. Improving upon this, our proposed model learns nodes that dynamically attach to well-delimited salient regions, which are relevant for a higher-level task, without using any object-level supervision. Constructing these localized, adaptive nodes gives our model inductive bias towards object-centric representations and we show that it discovers regions that are well correlated with objects in the video. In extensive ablation studies and experiments on two challenging datasets, we show superior performance to previous graph neural networks models for video classification.

----

## [544] Information-constrained optimization: can adaptive processing of gradients help?

**Authors**: *Jayadev Acharya, Clément L. Canonne, Prathamesh Mayekar, Himanshu Tyagi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/398475c83b47075e8897a083e97eb9f0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/398475c83b47075e8897a083e97eb9f0-Abstract.html)

**Abstract**:

We revisit first-order optimization under local information constraints such as local privacy, gradient quantization, and computational constraints limiting access to a few coordinates of the gradient. In this setting, the optimization algorithm is not allowed to directly access the complete output of the gradient oracle, but only gets limited information about it subject to the local information constraints.   We study the role of adaptivity in processing the gradient output to obtain this limited information from it, and obtain tight or nearly tight bounds for both convex and strongly convex optimization when adaptive gradient processing is allowed.

----

## [545] Towards Calibrated Model for Long-Tailed Visual Recognition from Prior Perspective

**Authors**: *Zhengzhuo Xu, Zenghao Chai, Chun Yuan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/39ae2ed11b14a4ccb41d35e9d1ba5d11-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/39ae2ed11b14a4ccb41d35e9d1ba5d11-Abstract.html)

**Abstract**:

Real-world data universally confronts a severe class-imbalance problem and exhibits a long-tailed distribution, i.e., most labels are associated with limited instances. The na√Øve models supervised by such datasets would prefer dominant labels, encounter a serious generalization challenge and become poorly calibrated. We propose two novel methods from the prior perspective to alleviate this dilemma. First, we deduce a balance-oriented data augmentation named Uniform Mixup (UniMix) to promote mixup in long-tailed scenarios, which adopts advanced mixing factor and sampler in favor of the minority. Second, motivated by the Bayesian theory, we figure out the Bayes Bias (Bayias), an inherent bias caused by the inconsistency of prior, and compensate it as a modification on standard cross-entropy loss. We further prove that both the proposed methods ensure the classification calibration theoretically and empirically. Extensive experiments verify that our strategies contribute to a better-calibrated model, and their combination achieves state-of-the-art performance on CIFAR-LT, ImageNet-LT, and iNaturalist 2018.

----

## [546] Learning to Draw: Emergent Communication through Sketching

**Authors**: *Daniela Mihai, Jonathon S. Hare*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/39d0a8908fbe6c18039ea8227f827023-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/39d0a8908fbe6c18039ea8227f827023-Abstract.html)

**Abstract**:

Evidence that visual communication preceded written language and provided a basis for it goes back to prehistory, in forms such as cave and rock paintings depicting traces of our distant ancestors. Emergent communication research has sought to explore how agents can learn to communicate in order to collaboratively solve tasks. Existing research has focused on language, with a learned communication channel transmitting sequences of discrete tokens between the agents. In this work, we explore a visual communication channel between agents that are allowed to draw with simple strokes. Our agents are parameterised by deep neural networks, and the drawing procedure is differentiable, allowing for end-to-end training. In the framework of a referential communication game, we demonstrate that agents can not only successfully learn to communicate by drawing, but with appropriate inductive biases, can do so in a fashion that humans can interpret. We hope to encourage future research to consider visual communication as a more flexible and directly interpretable alternative of training collaborative agents.

----

## [547] Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks

**Authors**: *Jesse J. Hagenaars, Federico Paredes-Vallés, Guido de Croon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/39d4b545fb02556829aab1db805021c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/39d4b545fb02556829aab1db805021c3-Abstract.html)

**Abstract**:

The field of neuromorphic computing promises extremely low-power and low-latency sensing and processing. Challenges in transferring learning algorithms from traditional artificial neural networks (ANNs) to spiking neural networks (SNNs) have so far prevented their application to large-scale, complex regression tasks. Furthermore, realizing a truly asynchronous and fully neuromorphic pipeline that maximally attains the abovementioned benefits involves rethinking the way in which this pipeline takes in and accumulates information. In the case of perception, spikes would be passed as-is and one-by-one between an event camera and an SNN, meaning all temporal integration of information must happen inside the network. In this article, we tackle these two problems. We focus on the complex task of learning to estimate optical flow from event-based camera inputs in a self-supervised manner, and modify the state-of-the-art ANN training pipeline to encode minimal temporal information in its inputs. Moreover, we reformulate the self-supervised loss function for event-based optical flow to improve its convexity. We perform experiments with various types of recurrent ANNs and SNNs using the proposed pipeline. Concerning SNNs, we investigate the effects of elements such as parameter initialization and optimization, surrogate gradient shape, and adaptive neuronal mechanisms. We find that initialization and surrogate gradient width play a crucial part in enabling learning with sparse inputs, while the inclusion of adaptivity and learnable neuronal parameters can improve performance. We show that the performance of the proposed ANNs and SNNs are on par with that of the current state-of-the-art ANNs trained in a self-supervised manner.

----

## [548] On the Value of Infinite Gradients in Variational Autoencoder Models

**Authors**: *Bin Dai, Wenliang Li, David P. Wipf*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3a15c7d0bbe60300a39f76f8a5ba6896-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3a15c7d0bbe60300a39f76f8a5ba6896-Abstract.html)

**Abstract**:

A number of recent studies of continuous variational autoencoder (VAE) models have noted, either directly or indirectly, the tendency of various parameter gradients to drift towards infinity during training.  Because such gradients could potentially contribute to numerical instabilities, and are often framed as a problematic phenomena to be avoided, it may be tempting to shift to alternative energy functions that guarantee bounded gradients.  But it remains an open question: What might the unintended consequences of such a restriction be?  To address this issue, we examine how unbounded gradients relate to the regularization of a broad class of autoencoder-based architectures, including VAE models, as applied to data lying on or near a low-dimensional manifold (e.g., natural images).  Our main finding is that, if the ultimate goal is to simultaneously avoid over-regularization (high reconstruction errors, sometimes referred to as posterior collapse) and under-regularization (excessive latent dimensions are not pruned from the model), then an autoencoder-based energy function with infinite gradients around optimal representations is provably required per a certain technical sense which we carefully detail.  Given that both over- and under-regularization can directly lead to poor generated sample quality or suboptimal feature selection, this result suggests that heuristic modifications to or constraints on the VAE energy function may at times be ill-advised, and large gradients should be accommodated to the extent possible.

----

## [549] Online Robust Reinforcement Learning with Model Uncertainty

**Authors**: *Yue Wang, Shaofeng Zou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3a4496776767aaa99f9804d0905fe584-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3a4496776767aaa99f9804d0905fe584-Abstract.html)

**Abstract**:

Robust reinforcement learning (RL) is to find a policy that optimizes the worst-case performance over an uncertainty set of MDPs. In this paper, we focus on model-free robust RL, where the uncertainty set is defined to be centering at a misspecified MDP that generates samples, and is assumed to be unknown. We develop a sample-based approach to estimate the unknown uncertainty set, and design robust Q-learning algorithm (tabular case) and robust TDC algorithm (function approximation setting), which can be implemented in an online and incremental fashion.  For the robust Q-learning algorithm, we prove that it converges to the optimal robust Q function, and for the robust TDC algorithm, we prove that it converges asymptotically to some stationary points. Unlike the results in [Roy et al., 2017], our algorithms do not need any additional conditions on the discount factor to guarantee the convergence. We further characterize the finite-time error bounds of the two algorithms, and show that both the robust Q-learning and robust TDC algorithms converge as fast as their vanilla counterparts (within a constant factor). Our numerical experiments further demonstrate the robustness of our algorithms. Our approach can be readily extended to robustify many other algorithms, e.g., TD, SARSA, and other GTD algorithms.

----

## [550] Neural View Synthesis and Matching for Semi-Supervised Few-Shot Learning of 3D Pose

**Authors**: *Angtian Wang, Shenxiao Mei, Alan L. Yuille, Adam Kortylewski*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3a61ed715ee66c48bacf237fa7bb5289-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3a61ed715ee66c48bacf237fa7bb5289-Abstract.html)

**Abstract**:

We study the problem of learning to estimate the 3D object pose from a few labelled examples and a collection of unlabelled data. Our main contribution is a learning framework, neural view synthesis and matching, that can transfer the 3D pose annotation from the labelled to unlabelled images reliably, despite unseen 3D views and nuisance variations such as the object shape, texture, illumination or scene context. In our approach, objects are represented as 3D cuboid meshes composed of feature vectors at each mesh vertex. The model is initialized from a few labelled images and is subsequently used to synthesize feature representations of unseen 3D views. The synthesized views are matched with the feature representations of unlabelled images to generate pseudo-labels of the 3D pose. The pseudo-labelled data is, in turn, used to train the feature extractor such that the features at each mesh vertex are more invariant across varying 3D views of the object. Our model is trained in an EM-type manner alternating between increasing the 3D pose invariance of the feature extractor and annotating unlabelled data through neural view synthesis and matching. We demonstrate the effectiveness of the proposed semi-supervised learning framework for 3D pose estimation on the PASCAL3D+ and KITTI datasets. We find that our approach outperforms all baselines by a wide margin, particularly in an extreme few-shot setting where only 7 annotated images are given. Remarkably, we observe that our model also achieves an exceptional robustness in out-of-distribution scenarios that involve partial occlusion.

----

## [551] Sharp Impossibility Results for Hyper-graph Testing

**Authors**: *Jiashun Jin, Zheng Tracy Ke, Jiajun Liang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3b24156ad560a696116454056bc88ab4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3b24156ad560a696116454056bc88ab4-Abstract.html)

**Abstract**:

In a broad Degree-Corrected Mixed-Membership (DCMM) setting, we test whether a non-uniform hypergraph has only one community or has multiple communities. Since both the null and alternative hypotheses have many unknown parameters, the challenge is, given an alternative, how to identify the null that is hardest to separate from the alternative. We approach this by proposing a degree matching strategy where the main idea is leveraging the theory for tensor scaling to create a least favorable pair of hypotheses. We present a  result on standard  minimax lower bound theory and a result on Region of Impossibility (which is more informative than the minimax lower bound). We show that our lower bounds are tight by introducing a new test that attains the lower bound up to a logarithmic factor. We also discuss the case where the hypergraphs may have mixed-memberships.

----

## [552] Evaluating Gradient Inversion Attacks and Defenses in Federated Learning

**Authors**: *Yangsibo Huang, Samyak Gupta, Zhao Song, Kai Li, Sanjeev Arora*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3b3fff6463464959dcd1b68d0320f781-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3b3fff6463464959dcd1b68d0320f781-Abstract.html)

**Abstract**:

Gradient inversion attack (or input recovery from gradient) is an emerging threat to the security and privacy preservation of Federated learning, whereby malicious eavesdroppers or participants in the protocol can recover (partially) the clients' private data. This paper evaluates existing attacks and defenses. We find that some attacks make strong assumptions about the setup. Relaxing such assumptions can substantially weaken these attacks. We then evaluate the benefits of three proposed defense mechanisms against gradient inversion attacks. We show the trade-offs of privacy leakage and data utility of these defense methods, and find that combining them in an appropriate manner makes the attack less effective, even under the original strong assumptions. We also estimate the computation cost of end-to-end recovery of a single image under each evaluated defense. Our findings suggest that the state-of-the-art attacks can currently be defended against with minor data utility loss, as summarized in a list of potential strategies.

----

## [553] Faster Non-asymptotic Convergence for Double Q-learning

**Authors**: *Lin Zhao, Huaqing Xiong, Yingbin Liang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3b712de48137572f3849aabd5666a4e3-Abstract.html)

**Abstract**:

Double Q-learning (Hasselt, 2010) has gained significant success in practice due to its effectiveness in overcoming the overestimation issue of Q-learning. However, the theoretical understanding of double Q-learning is rather limited. The only existing finite-time analysis was recently established in (Xiong et al. 2020), where the polynomial learning rate adopted in the analysis typically yields a slower convergence rate. This paper tackles the more challenging case of a constant learning rate, and develops new analytical tools that improve the existing convergence rate by orders of magnitude. Specifically, we show that synchronous double Q-learning attains an $\epsilon$-accurate global optimum with a time complexity of $\tilde{\Omega}\left(\frac{\ln D}{(1-\gamma)^7\epsilon^2} \right)$, and the asynchronous algorithm achieves a time complexity of $\tilde{\Omega}\left(\frac{L}{(1-\gamma)^7\epsilon^2} \right)$, where $D$ is the cardinality of the state-action space, $\gamma$ is the discount factor, and $L$ is a parameter related to the sampling strategy for asynchronous double Q-learning. These results improve the existing convergence rate by the order of magnitude in terms of its dependence on all major parameters $(\epsilon,1-\gamma, D, L)$.  This paper presents a substantial step toward the full understanding of the fast convergence of double-Q learning.

----

## [554] Towards Tight Communication Lower Bounds for Distributed Optimisation

**Authors**: *Janne H. Korhonen, Dan Alistarh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3b92d18aa7a6176dd37d372bc2f1eb71-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3b92d18aa7a6176dd37d372bc2f1eb71-Abstract.html)

**Abstract**:

We consider a standard distributed optimisation setting where $N$ machines, each holding a $d$-dimensional function $f_i$, aim to jointly minimise the sum of the functions $\sum_{i = 1}^N f_i (x)$.  This problem arises naturally in  large-scale distributed optimisation, where a standard solution is to apply variants of (stochastic) gradient descent. We focus on the communication complexity of this problem: our main result provides the first fully unconditional bounds on total number of bits which need to be sent and received by the $N$ machines to solve this problem under point-to-point communication, within a given error-tolerance. Specifically, we show that $\Omega( Nd \log d / N\varepsilon)$ total bits need to be communicated between the machines to find an additive $\epsilon$-approximation to the minimum of $\sum_{i = 1}^N f_i (x)$. The result holds for both deterministic and randomised algorithms, and, importantly, requires no assumptions on the algorithm structure. The lower bound is tight under certain restrictions on parameter values, and is matched within constant factors for quadratic objectives by a new variant of quantised gradient descent, which we describe and analyse. Our results bring over tools from communication complexity to distributed optimisation, which has potential for further  applications.

----

## [555] Fast Multi-Resolution Transformer Fine-tuning for Extreme Multi-label Text Classification

**Authors**: *Jiong Zhang, Wei-Cheng Chang, Hsiang-Fu Yu, Inderjit S. Dhillon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3bbca1d243b01b47c2bf42b29a8b265c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3bbca1d243b01b47c2bf42b29a8b265c-Abstract.html)

**Abstract**:

Extreme multi-label text classification~(XMC) seeks to find relevant labels from an extreme large label collection for a given text input. Many real-world applications can be formulated as XMC problems, such as recommendation systems, document tagging and semantic search. Recently, transformer based XMC methods, such as X-Transformer and LightXML, have shown significant improvement over other XMC methods. Despite leveraging pre-trained transformer models for text representation, the fine-tuning procedure of transformer models on large label space still has lengthy computational time even with powerful GPUs. In this paper, we propose a novel recursive approach, XR-Transformer to accelerate the procedure through recursively fine-tuning transformer models on a series of multi-resolution objectives related to the original XMC objective function. Empirical results show that XR-Transformer takes significantly less training time compared to other transformer-based XMC models while yielding better state-of-the-art results. In particular, on the public Amazon-3M dataset with 3 million labels, XR-Transformer is not only 20x faster than X-Transformer but also improves the Precision@1 from 51% to 54%.

----

## [556] HRFormer: High-Resolution Vision Transformer for Dense Predict

**Authors**: *Yuhui Yuan, Rao Fu, Lang Huang, Weihong Lin, Chao Zhang, Xilin Chen, Jingdong Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html)

**Abstract**:

We present a High-Resolution Transformer (HRFormer) that learns high-resolution representations for dense prediction tasks, in contrast to the original Vision Transformer that produces low-resolution representations and has high memory and computational cost. We take advantage of the multi-resolution parallel design introduced in high-resolution convolutional networks (HRNet [45]), along with local-window self-attention that performs self-attention over small non-overlapping image windows [21], for improving the memory and computation efficiency. In addition, we introduce a convolution into the FFN to exchange information across the disconnected image windows. We demonstrate the effectiveness of the HighResolution Transformer on both human pose estimation and semantic segmentation tasks, e.g., HRFormer outperforms Swin transformer [27] by 1.3 AP on COCO pose estimation with 50% fewer parameters and 30% fewer FLOPs. Code is available at: https://github.com/HRNet/HRFormer

----

## [557] Manifold Topology Divergence: a Framework for Comparing Data Manifolds

**Authors**: *Serguei Barannikov, Ilya Trofimov, Grigorii Sotnikov, Ekaterina Trimbach, Alexander Korotin, Alexander Filippov, Evgeny Burnaev*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3bc31a430954d8326605fc690ed22f4d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3bc31a430954d8326605fc690ed22f4d-Abstract.html)

**Abstract**:

We propose a framework for comparing data manifolds, aimed, in particular, towards the evaluation of deep generative models. We describe a novel tool, Cross-Barcode(P,Q), that, given a pair of distributions in a high-dimensional space, tracks multiscale topology spacial discrepancies between manifolds on which the distributions are concentrated. Based on the Cross-Barcode, we introduce the Manifold Topology Divergence score (MTop-Divergence) and apply it to assess the performance of deep generative models in various domains: images, 3D-shapes, time-series, and on different datasets: MNIST, Fashion MNIST, SVHN, CIFAR10, FFHQ, market stock data, ShapeNet. We demonstrate that the MTop-Divergence accurately detects various degrees of mode-dropping, intra-mode collapse, mode invention, and image disturbance. Our algorithm scales well (essentially linearly) with the increase of the dimension of the ambient high-dimensional space. It is one of the first TDA-based methodologies that can be applied universally to datasets of different sizes and dimensions, including the ones on which the most recent GANs in the visual domain are trained. The proposed method is domain agnostic and does not rely on pre-trained networks.

----

## [558] Weak-shot Fine-grained Classification via Similarity Transfer

**Authors**: *Junjie Chen, Li Niu, Liu Liu, Liqing Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3bd4017318837e92a66298c7855f4427-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3bd4017318837e92a66298c7855f4427-Abstract.html)

**Abstract**:

Recognizing fine-grained categories remains a challenging task, due to the subtle distinctions among different subordinate categories, which results in the need of abundant annotated samples. To alleviate the data-hungry problem, we consider the problem of learning novel categories from web data with the support of a clean set of base categories, which is referred to as weak-shot learning. In this setting, we propose a method called SimTrans to transfer pairwise semantic similarity from base categories to novel categories. Specifically, we firstly train a similarity net on clean data, and then leverage the transferred similarity to denoise web training data using two simple yet effective strategies. In addition, we apply adversarial loss on similarity net to enhance the transferability of similarity. Comprehensive experiments demonstrate the effectiveness of our weak-shot setting and our SimTrans method.

----

## [559] Shape your Space: A Gaussian Mixture Regularization Approach to Deterministic Autoencoders

**Authors**: *Amrutha Saseendran, Kathrin Skubch, Stefan Falkner, Margret Keuper*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3c057cb2b41f22c0e740974d7a428918-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3c057cb2b41f22c0e740974d7a428918-Abstract.html)

**Abstract**:

Variational Autoencoders (VAEs) are powerful probabilistic models to learn representations of complex data distributions. One important limitation of VAEs is the strong prior assumption that latent representations learned by the model follow a simple uni-modal Gaussian distribution. Further, the variational training procedure poses considerable practical challenges. Recently proposed regularized autoencoders offer a deterministic autoencoding framework, that simplifies the original VAE objective and is significantly easier to train. Since these models only provide weak control over the learned latent distribution, they require an ex-post density estimation step to generate samples comparable to those of VAEs. In this paper, we propose a simple and end-to-end trainable deterministic autoencoding framework, that efficiently shapes the latent space of the model during training and utilizes the capacity of expressive multi-modal latent distributions. The proposed training procedure provides direct evidence if the latent distribution adequately captures complex aspects of the encoded data. We show in experiments the expressiveness and sample quality of our model in various challenging continuous and discrete domains. An implementation is available at https://github.com/boschresearch/GMM_DAE.

----

## [560] An Even More Optimal Stochastic Optimization Algorithm: Minibatching and Interpolation Learning

**Authors**: *Blake E. Woodworth, Nathan Srebro*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3c63ec7be1b6c49e6c308397023fd8cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3c63ec7be1b6c49e6c308397023fd8cd-Abstract.html)

**Abstract**:

We present and analyze an algorithm for optimizing smooth and convex or strongly convex objectives using minibatch stochastic gradient estimates. The algorithm is optimal with respect to its dependence on both the minibatch size and minimum expected loss simultaneously. This improves over the optimal method of Lan, which is insensitive to the minimum expected loss; over the optimistic acceleration of Cotter et al., which has suboptimal dependence on the minibatch size; and over the algorithm of Liu and Belkin, which is limited to least squares problems and is also similarly suboptimal.  Applied to interpolation learning, the improvement over Cotter et al.~and Liu and Belkin translates to a linear, rather than square-root, parallelization speedup.

----

## [561] Indexed Minimum Empirical Divergence for Unimodal Bandits

**Authors**: *Hassan Saber, Pierre Ménard, Odalric-Ambrym Maillard*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3c88c1db16b9523b4dcdcd572aa1e16a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3c88c1db16b9523b4dcdcd572aa1e16a-Abstract.html)

**Abstract**:

We consider a stochastic multi-armed bandit problem specified by a set of one-dimensional family exponential distributions endowed with a unimodal structure. The unimodal structure is of practical relevance for several applications. We introduce IMED-UB, an algorithm that exploits provably optimally the unimodal-structure, by adapting to this setting the Indexed Minimum Empirical Divergence (IMED) algorithm introduced by Honda and Takemura (2015).  Owing to our proof technique, we are able to provide a concise finite-time analysis of the IMED-UB algorithm, that is simple and yet yields asymptotic optimality. We finally provide numerical experiments showing that IMED-UB competes favorably with the recently introduced state-of-the-art algorithms.

----

## [562] SOAT: A Scene- and Object-Aware Transformer for Vision-and-Language Navigation

**Authors**: *Abhinav Moudgil, Arjun Majumdar, Harsh Agrawal, Stefan Lee, Dhruv Batra*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3c8a49145944fed2bbcaade178a426c4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3c8a49145944fed2bbcaade178a426c4-Abstract.html)

**Abstract**:

Natural language instructions for visual navigation often use scene descriptions (e.g., bedroom) and object references (e.g., green chairs) to provide a breadcrumb trail to a goal location. This work presents a transformer-based vision-and-language navigation (VLN) agent that uses two different visual encoders -- a scene classification network and an object detector -- which produce features that match these two distinct types of visual cues. In our method, scene features contribute high-level contextual information that supports object-level processing. With this design, our model is able to use vision-and-language pretraining (i.e., learning the alignment between images and text from large-scale web data) to substantially improve performance on the Room-to-Room (R2R) and Room-Across-Room (RxR) benchmarks. Specifically, our approach leads to improvements of 1.8% absolute in SPL on R2R and 3.7% absolute in SR on RxR. Our analysis reveals even larger gains for navigation instructions that contain six or more object references, which further suggests that our approach is better able to use object features and align them to references in the instructions.

----

## [563] A Normative and Biologically Plausible Algorithm for Independent Component Analysis

**Authors**: *Yanis Bahroun, Dmitri B. Chklovskii, Anirvan M. Sengupta*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3ce3bd7d63a2c9c81983cc8e9bd02ae5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3ce3bd7d63a2c9c81983cc8e9bd02ae5-Abstract.html)

**Abstract**:

The brain effortlessly solves blind source separation (BSS) problems, but the algorithm it uses remains elusive. In signal processing, linear BSS problems are often solved by Independent Component Analysis (ICA). To serve as a model of a biological circuit, the ICA neural network (NN) must satisfy at least the following requirements: 1. The algorithm must operate in the online setting where data samples are streamed one at a time, and the NN computes the sources on the fly without storing any significant fraction of the data in memory. 2. The synaptic weight update is local, i.e., it depends only on the biophysical variables present in the vicinity of a synapse. Here, we propose a novel objective function for ICA from which we derive a biologically plausible NN, including both the neural architecture and the synaptic learning rules. Interestingly, our algorithm relies on modulating synaptic plasticity by the total activity of the output neurons. In the brain, this could be accomplished by neuromodulators, extracellular calcium, local field potential, or nitric oxide.

----

## [564] Regret Bounds for Gaussian-Process Optimization in Large Domains

**Authors**: *Manuel Wüthrich, Bernhard Schölkopf, Andreas Krause*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3cec07e9ba5f5bb252d13f5f431e4bbb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3cec07e9ba5f5bb252d13f5f431e4bbb-Abstract.html)

**Abstract**:

The goal of this paper is to characterize Gaussian-Process optimization in the setting where the function domain is large relative to the number of admissible function evaluations, i.e., where it is impossible to find the global optimum. We provide upper bounds on the suboptimality (Bayesian simple regret) of the solution found by optimization strategies that are closely related to the widely used expected improvement (EI) and upper confidence bound (UCB) algorithms. These regret bounds illuminate the relationship between the number of evaluations, the domain size (i.e. cardinality of finite domains / Lipschitz constant of the covariance function in continuous domains), and the optimality of the retrieved function value.In particular, we show that even when the number of evaluations is far too small to find the global optimum, we can find nontrivial function values (e.g. values that achieve a certain ratio with the optimal value).

----

## [565] Deeply Shared Filter Bases for Parameter-Efficient Convolutional Neural Networks

**Authors**: *Woochul Kang, Daeyeon Kim*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3cf2559725a9fdfa602ec8c887440f32-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3cf2559725a9fdfa602ec8c887440f32-Abstract.html)

**Abstract**:

Modern convolutional neural networks (CNNs) have massive identical convolution blocks, and, hence, recursive sharing of parameters across these blocks has been proposed to reduce the amount of parameters.  However, naive sharing of parameters poses many challenges such as limited representational power and the vanishing/exploding gradients problem of recursively shared parameters. In this paper, we present a recursive convolution block design and training method, in which a recursively shareable part, or a filter basis, is separated and learned while effectively avoiding the vanishing/exploding gradients problem during training. We show that the unwieldy vanishing/exploding gradients problem can be controlled by enforcing the elements of the filter basis orthonormal, and empirically demonstrate that the proposed orthogonality regularization improves the flow of gradients during training. Experimental results on image classification and object detection show that our approach, unlike previous parameter-sharing approaches, does not trade performance to save parameters and consistently outperforms over parameterized counterpart networks. This superior performance demonstrates that the proposed recursive convolution block design and the orthogonality regularization not only prevent performance degradation, but also consistently improve the representation capability while a significant amount of parameters are recursively shared.

----

## [566] On Optimal Robustness to Adversarial Corruption in Online Decision Problems

**Authors**: *Shinji Ito*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d191ef6e236bd1b9bdb9ff4743c47fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d191ef6e236bd1b9bdb9ff4743c47fe-Abstract.html)

**Abstract**:

This paper considers two fundamental sequential decision-making problems: the problem of prediction with expert advice and the multi-armed bandit problem.  We focus on stochastic regimes in which an adversary may corrupt losses, and we investigate what level of robustness can be achieved against adversarial corruption.  The main contribution of this paper is to show that optimal robustness can be expressed by a square-root dependency on the amount of corruption.  More precisely, we show that two classes of algorithms, anytime Hedge with decreasing learning rate and algorithms with second-order regret bounds, achieve $O( \frac{\log N}{\Delta} + \sqrt{ \frac{C \log N }{\Delta} } )$-regret, where $N, \Delta$, and $C$ represent the number of experts, the gap parameter, and the corruption level, respectively.  We further provide a matching lower bound, which means that this regret bound is tight up to a constant factor. For the multi-armed bandit problem, we also provide a nearly-tight lower bound up to a logarithmic factor.

----

## [567] Directed Spectrum Measures Improve Latent Network Models Of Neural Populations

**Authors**: *Neil Gallagher, Kafui Dzirasa, David E. Carlson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d36c07721a0a5a96436d6c536a132ec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d36c07721a0a5a96436d6c536a132ec-Abstract.html)

**Abstract**:

Systems neuroscience aims to understand how networks of neurons distributed throughout the brain mediate computational tasks. One popular approach to identify those networks is to first calculate measures of neural activity (e.g. power spectra) from multiple brain regions, and then apply a linear factor model to those measures. Critically, despite the established role of directed communication between brain regions in neural computation, measures of directed communication have been rarely utilized in network estimation because they are incompatible with the implicit assumptions of the linear factor model approach. Here, we develop a novel spectral measure of directed communication called the Directed Spectrum (DS). We prove that it is compatible with the implicit assumptions of linear factor models, and we provide a method to estimate the DS. We demonstrate that latent linear factor models of DS measures better capture underlying brain networks in both simulated and real neural recording data compared to available alternatives. Thus, linear factor models of the Directed Spectrum offer neuroscientists a simple and effective way to explicitly model directed communication in networks of neural populations.

----

## [568] Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble

**Authors**: *Gaon An, Seungyong Moon, Jang-Hyun Kim, Hyun Oh Song*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d3d286a8d153a4a58156d0e02d8570c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d3d286a8d153a4a58156d0e02d8570c-Abstract.html)

**Abstract**:

Offline reinforcement learning (offline RL), which aims to find an optimal policy from a previously collected static dataset, bears algorithmic difficulties due to function approximation errors from out-of-distribution (OOD) data points. To this end, offline RL algorithms adopt either a constraint or a penalty term that explicitly guides the policy to stay close to the given dataset. However, prior methods typically require accurate estimation of the behavior policy or sampling from OOD data points, which themselves can be a non-trivial problem. Moreover, these methods under-utilize the generalization ability of deep neural networks and often fall into suboptimal solutions too close to the given dataset. In this work, we propose an uncertainty-based offline RL method that takes into account the confidence of the Q-value prediction and does not require any estimation or sampling of the data distribution. We show that the clipped Q-learning, a technique widely used in online RL, can be leveraged to successfully penalize OOD data points with high prediction uncertainties. Surprisingly, we find that it is possible to substantially outperform existing offline RL methods on various tasks by simply increasing the number of Q-networks along with the clipped Q-learning. Based on this observation, we propose an ensemble-diversified actor-critic algorithm that reduces the number of required ensemble networks down to a tenth compared to the naive ensemble while achieving state-of-the-art performance on most of the D4RL benchmarks considered.

----

## [569] Distribution-free inference for regression: discrete, continuous, and in between

**Authors**: *Yonghoon Lee, Rina Barber*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d4893419e57449fb290647149f738d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d4893419e57449fb290647149f738d4-Abstract.html)

**Abstract**:

In data analysis problems where we are not able to rely on distributional assumptions, what types of inference guarantees can still be obtained? Many popular methods, such as holdout methods, cross-validation methods, and conformal prediction, are able to provide distribution-free guarantees for predictive inference, but the problem of providing inference for the underlying regression function (for example, inference on the conditional mean $\mathbb{E}[Y|X]$) is more challenging. In the setting where the features $X$ are continuously distributed, recent work has established that any confidence interval for $\mathbb{E}[Y|X]$ must have non-vanishing width, even as sample size tends to infinity. At the other extreme, if $X$ takes only a small number of possible values, then inference on $\mathbb{E}[Y|X]$ is trivial to achieve. In this work, we study the problem in settings in between these two extremes. We find that there are several distinct regimes in between the finite setting and the continuous setting, where vanishing-width confidence intervals are achievable if and only if the effective support size of the distribution of $X$ is smaller than the square of the sample size.

----

## [570] Statistical Inference with M-Estimators on Adaptively Collected Data

**Authors**: *Kelly W. Zhang, Lucas Janson, Susan A. Murphy*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d7d9461075eb7c37fbbfcad1d7042c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d7d9461075eb7c37fbbfcad1d7042c1-Abstract.html)

**Abstract**:

Bandit algorithms are increasingly used in real-world sequential decision-making problems. Associated with this is an increased desire to be able to use the resulting datasets to answer scientific questions like: Did one type of ad lead to more purchases? In which contexts is a mobile health intervention effective? However, classical statistical approaches fail to provide valid confidence intervals when used with data collected with bandit algorithms. Alternative methods have recently been developed for simple models (e.g., comparison of means). Yet there is a lack of general methods for  conducting statistical inference using more complex models on data collected with (contextual) bandit algorithms; for example, current methods cannot be used for valid inference on parameters in a logistic regression model for a binary reward. In this  work, we develop theory justifying the use of M-estimators---which  includes estimators based on empirical risk minimization as well as maximum likelihood---on data collected with adaptive algorithms, including (contextual) bandit algorithms. Specifically, we show that M-estimators, modified with particular adaptive weights, can be used  to construct asymptotically valid confidence regions for a variety of inferential targets.

----

## [571] NeuroLKH: Combining Deep Learning Model with Lin-Kernighan-Helsgaun Heuristic for Solving the Traveling Salesman Problem

**Authors**: *Liang Xin, Wen Song, Zhiguang Cao, Jie Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d863b367aa379f71c7afc0c9cdca41d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d863b367aa379f71c7afc0c9cdca41d-Abstract.html)

**Abstract**:

We present NeuroLKH, a novel algorithm that combines deep learning with the strong traditional heuristic Lin-Kernighan-Helsgaun (LKH) for solving Traveling Salesman Problem. Specifically, we train a Sparse Graph Network (SGN) with supervised learning for edge scores and unsupervised learning for node penalties, both of which are critical for improving the performance of LKH. Based on the output of SGN, NeuroLKH creates the edge candidate set and transforms edge distances to guide the searching process of LKH. Extensive experiments firmly demonstrate that, by training one model on a wide range of problem sizes, NeuroLKH significantly outperforms LKH and generalizes well to much larger sizes. Also, we show that NeuroLKH can be applied to other routing problems such as Capacitated Vehicle Routing Problem (CVRP), Pickup and Delivery Problem (PDP), and CVRP with Time Windows (CVRPTW).

----

## [572] LSH-SMILE: Locality Sensitive Hashing Accelerated Simulation and Learning

**Authors**: *Chonghao Sima, Yexiang Xue*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3d98b79ac6c8d1cef43d7bf1dadf8647-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3d98b79ac6c8d1cef43d7bf1dadf8647-Abstract.html)

**Abstract**:

The advancement of deep neural networks over the last decade has enabled progress in scientific knowledge discovery in the form of learning Partial Differential Equations (PDEs) directly from experiment data. Nevertheless, forward simulation and backward learning of large-scale dynamic systems require handling billions of mutually interacting elements, the scale of which overwhelms current computing architectures. We propose Locality Sensitive Hashing Accelerated Simulation and Learning (LSH-SMILE), a unified framework to scale up both forward simulation and backward learning of physics systems. LSH-SMILE takes advantage of (i) the locality of PDE updates, (ii) similar temporal dynamics shared by multiple elements. LSH-SMILE hashes elements with similar dynamics into a single hash bucket and handles their updates at once. This allows LSH-SMILE to scale with respect to the number of non-empty hash buckets, a drastic improvement over conventional approaches. Theoretically, we prove a novel bound on the errors introduced by LSH-SMILE. Experimentally, we demonstrate that LSH-SMILE simulates physics systems at comparable quality with exact approaches, but with way less time and space complexity. Such savings also translate to better learning performance due to LSH-SMILE's ability to propagate gradients over a long duration.

----

## [573] Meta-learning with an Adaptive Task Scheduler

**Authors**: *Huaxiu Yao, Yu Wang, Ying Wei, Peilin Zhao, Mehrdad Mahdavi, Defu Lian, Chelsea Finn*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3dc4876f3f08201c7c76cb71fa1da439-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3dc4876f3f08201c7c76cb71fa1da439-Abstract.html)

**Abstract**:

To benefit the learning of a new task, meta-learning has been proposed to transfer a well-generalized meta-model learned from various meta-training tasks. Existing meta-learning algorithms randomly sample meta-training tasks with a uniform probability, under the assumption that tasks are of equal importance. However, it is likely that tasks are detrimental with noise or imbalanced given a limited number of meta-training tasks. To prevent the meta-model from being corrupted by such detrimental tasks or dominated by tasks in the majority, in this paper, we propose an adaptive task scheduler (ATS) for the meta-training process. In ATS, for the first time, we design a neural scheduler to decide which meta-training tasks to use next by predicting the probability being sampled for each candidate task, and train the scheduler to optimize the generalization capacity of the meta-model to unseen tasks. We identify two meta-model-related factors as the input of the neural scheduler, which characterize the difficulty of a candidate task to the meta-model. Theoretically, we show that a scheduler taking the two factors into account improves the meta-training loss and also the optimization landscape. Under the setting of meta-learning with noise and limited budgets, ATS improves the performance on both miniImageNet and a real-world drug discovery benchmark by up to 13% and 18%, respectively, compared to state-of-the-art task schedulers.

----

## [574] Neural Active Learning with Performance Guarantees

**Authors**: *Zhilei Wang, Pranjal Awasthi, Christoph Dann, Ayush Sekhari, Claudio Gentile*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3dcaf04c357c577a857f3ffadc555f9b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3dcaf04c357c577a857f3ffadc555f9b-Abstract.html)

**Abstract**:

We investigate the problem of active learning in the streaming setting in non-parametric regimes, where the labels are stochastically generated from a class of functions on which we make no assumptions whatsoever. We rely on recently proposed Neural Tangent Kernel (NTK) approximation tools to construct a suitable neural embedding that determines the feature space the algorithm operates on and the learned model computed atop. Since the shape of the label requesting threshold is tightly related to the complexity of the function to be learned, which is a-priori unknown, we also derive a version of the algorithm which is agnostic to any prior knowledge. This algorithm relies on a regret balancing scheme to solve the resulting online model selection problem, and is computationally efficient. We prove joint guarantees on the cumulative regret and number of requested labels which depend on the complexity of the labeling function at hand. In the linear case, these guarantees recover known minimax results of the generalization error as a function of the label complexity in a standard statistical learning setting.

----

## [575] A Gradient Method for Multilevel Optimization

**Authors**: *Ryo Sato, Mirai Tanaka, Akiko Takeda*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)

**Abstract**:

Although application examples of multilevel optimization have already been discussed since the 1990s, the development of solution methods was almost limited to bilevel cases due to the difficulty of the problem. In recent years, in machine learning, Franceschi et al. have proposed a method for solving bilevel optimization problems by replacing their lower-level problems with the $T$ steepest descent update equations with some prechosen iteration number $T$. In this paper, we have developed a gradient-based algorithm for multilevel optimization with $n$ levels based on their idea and proved that our reformulation asymptotically converges to the original multilevel problem. As far as we know, this is one of the first algorithms with some theoretical guarantee for multilevel optimization. Numerical experiments show that a trilevel hyperparameter learning model considering data poisoning produces more stable prediction results than an existing bilevel hyperparameter learning model in noisy data settings.

----

## [576] Edge Representation Learning with Hypergraphs

**Authors**: *Jaehyeong Jo, Jinheon Baek, Seul Lee, Dongki Kim, Minki Kang, Sung Ju Hwang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3def184ad8f4755ff269862ea77393dd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3def184ad8f4755ff269862ea77393dd-Abstract.html)

**Abstract**:

Graph neural networks have recently achieved remarkable success in representing graph-structured data, with rapid progress in both the node embedding and graph pooling methods. Yet, they mostly focus on capturing information from the nodes considering their connectivity, and not much work has been done in representing the edges, which are essential components of a graph. However, for tasks such as graph reconstruction and generation, as well as graph classification tasks for which the edges are important for discrimination, accurately representing edges of a given graph is crucial to the success of the graph representation learning. To this end, we propose a novel edge representation learning framework based on Dual Hypergraph Transformation (DHT), which transforms the edges of a graph into the nodes of a hypergraph. This dual hypergraph construction allows us to apply message-passing techniques for node representations to edges. After obtaining edge representations from the hypergraphs, we then cluster or drop edges to obtain holistic graph-level edge representations. We validate our edge representation learning method with hypergraphs on diverse graph datasets for graph representation and generation performance, on which our method largely outperforms existing graph representation learning methods. Moreover, our edge representation learning and pooling method also largely outperforms state-of-the-art graph pooling methods on graph classification, not only because of its accurate edge representation learning, but also due to its lossless compression of the nodes and removal of irrelevant edges for effective message-passing.

----

## [577] One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval

**Authors**: *Akari Asai, Xinyan Yu, Jungo Kasai, Hanna Hajishirzi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3df07fdae1ab273a967aaa1d355b8bb6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3df07fdae1ab273a967aaa1d355b8bb6-Abstract.html)

**Abstract**:

We present Cross-lingual Open-Retrieval Answer Generation (CORA), the first unified many-to-many question answering (QA) model that can answer questions across many languages, even for ones without language-specific annotated data or knowledge sources.We introduce a new dense passage retrieval algorithm that is trained to retrieve documents across languages for a question.Combined with a  multilingual autoregressive generation model, CORA answers directly in the target language without any translation or in-language retrieval modules as used in prior work. We propose an iterative training method that automatically extends annotated data available only in high-resource languages to low-resource ones. Our results show that CORA substantially outperforms the previous state of the art on multilingual open QA benchmarks across 26 languages, 9 of which are unseen during training. Our analyses show the significance of cross-lingual retrieval and generation in many languages, particularly under low-resource settings.

----

## [578] LEADS: Learning Dynamical Systems that Generalize Across Environments

**Authors**: *Yuan Yin, Ibrahim Ayed, Emmanuel de Bézenac, Nicolas Baskiotis, Patrick Gallinari*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3df1d4b96d8976ff5986393e8767f5b2-Abstract.html)

**Abstract**:

When modeling dynamical systems from real-world data samples, the distribution of data often changes according to the environment in which they are captured, and the dynamics of the system itself vary from one environment to another. Generalizing across environments thus challenges the conventional frameworks. The classical settings suggest either considering data as i.i.d and learning a single model to cover all situations or learning environment-specific models. Both are sub-optimal: the former disregards the discrepancies between environments leading to biased solutions, while the latter does not exploit their potential commonalities and is prone to scarcity problems. We propose LEADS, a novel framework that leverages the commonalities and discrepancies among known environments to improve model generalization. This is achieved with a tailored training formulation aiming at capturing common dynamics within a shared model while additional terms capture environment-specific dynamics. We ground our approach in theory, exhibiting a decrease in sample complexity w.r.t classical alternatives.  We show how theory and practice coincides on the simplified case of linear dynamics. Moreover, we instantiate this framework for neural networks and evaluate it experimentally on representative families of nonlinear dynamics. We show that this new setting can exploit knowledge extracted from environment-dependent data and improves generalization for both known and novel environments.

----

## [579] Storchastic: A Framework for General Stochastic Automatic Differentiation

**Authors**: *Emile van Krieken, Jakub M. Tomczak, Annette ten Teije*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3dfe2f633108d604df160cd1b01710db-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3dfe2f633108d604df160cd1b01710db-Abstract.html)

**Abstract**:

Modelers use automatic differentiation (AD) of computation graphs to implement complex Deep Learning models without defining gradient computations. Stochastic AD extends AD to stochastic computation graphs with sampling steps, which arise when modelers handle the intractable expectations common in Reinforcement Learning and Variational Inference. However, current methods for stochastic AD are limited: They are either only applicable to continuous random variables and differentiable functions, or can only use simple but high variance score-function estimators. To overcome these limitations, we introduce Storchastic, a new framework for AD of stochastic computation graphs. Storchastic allows the modeler to choose from a wide variety of gradient estimation methods at each sampling step, to optimally reduce the variance of the gradient estimates. Furthermore, Storchastic is provably unbiased for estimation of any-order gradients, and generalizes variance reduction techniques to higher-order gradient estimates. Finally, we implement Storchastic as a PyTorch library at github.com/HEmile/storchastic.

----

## [580] Concentration inequalities under sub-Gaussian and sub-exponential conditions

**Authors**: *Andreas Maurer, Massimiliano Pontil*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3e33b970f21d2fc65096871ea0d2c6e4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3e33b970f21d2fc65096871ea0d2c6e4-Abstract.html)

**Abstract**:

We prove analogues of the popular bounded difference inequality (also called McDiarmid's inequality) for functions of independent random variables under sub-gaussian and sub-exponential conditions. Applied to vector-valued concentration and the method of Rademacher complexities these inequalities allow an easy extension of uniform convergence results for PCA and linear regression to the case potentially unbounded input- and output variables.

----

## [581] Variance-Aware Off-Policy Evaluation with Linear Function Approximation

**Authors**: *Yifei Min, Tianhao Wang, Dongruo Zhou, Quanquan Gu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3e6260b81898beacda3d16db379ed329-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3e6260b81898beacda3d16db379ed329-Abstract.html)

**Abstract**:

We study the off-policy evaluation (OPE) problem in reinforcement learning with linear function approximation, which aims to estimate the value function of a target policy based on the offline data collected by a behavior policy. We propose to incorporate the variance information of the value function to improve the sample efficiency of OPE. More specifically, for time-inhomogeneous episodic linear Markov decision processes (MDPs), we propose an algorithm, \texttt{VA-OPE}, which uses the estimated variance of the value function to reweight the Bellman residual in Fitted Q-Iteration. We show that our algorithm achieves a tighter error bound than the best-known result. We also provide a fine-grained characterization of the distribution shift between the behavior policy and the target policy. Extensive numerical experiments corroborate our theory.

----

## [582] A Provably Efficient Sample Collection Strategy for Reinforcement Learning

**Authors**: *Jean Tarbouriech, Matteo Pirotta, Michal Valko, Alessandro Lazaric*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3e98410c45ea98addec555019bbae8eb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3e98410c45ea98addec555019bbae8eb-Abstract.html)

**Abstract**:

One of the challenges in online reinforcement learning (RL) is that the agent needs to trade off the exploration of the environment and the exploitation of the samples to optimize its behavior. Whether we optimize for regret, sample complexity, state-space coverage or model estimation, we need to strike a different exploration-exploitation trade-off. In this paper, we propose to tackle the exploration-exploitation problem following a decoupled approach composed of: 1) An "objective-specific" algorithm that (adaptively) prescribes how many samples to collect at which states, as if it has access to a generative model (i.e., a simulator of the environment); 2) An "objective-agnostic" sample collection exploration strategy responsible for generating the prescribed samples as fast as possible. Building on recent methods for exploration in the stochastic shortest path problem, we first provide an algorithm that, given as input the number of samples $b(s,a)$ needed in each state-action pair, requires $\widetilde{O}(B D + D^{3/2} S^2 A)$ time steps to collect the $B=\sum_{s,a} b(s,a)$ desired samples, in any unknown communicating MDP with $S$ states, $A$ actions and diameter $D$. Then we show how this general-purpose exploration algorithm can be paired with "objective-specific" strategies that prescribe the sample requirements to tackle a variety of settings — e.g., model estimation, sparse reward discovery, goal-free cost-free exploration in communicating MDPs — for which we obtain improved or novel sample complexity guarantees.

----

## [583] Improved Regret Bounds for Tracking Experts with Memory

**Authors**: *James Robinson, Mark Herbster*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3e9f7c16bd1cdea78f8e2eea72dfdfbe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3e9f7c16bd1cdea78f8e2eea72dfdfbe-Abstract.html)

**Abstract**:

We address the problem of sequential prediction with expert advice in a non-stationary environment with long-term memory guarantees in the sense of Bousquet and Warmuth [4]. We give a linear-time algorithm that improves on the best known regret bound [27]. This algorithm incorporates a relative entropy projection step. This projection is advantageous over previous weight-sharing approaches in that weight updates may come with implicit costs as in for example portfolio optimization. We give an algorithm to compute this projection step in linear time, which may be of independent interest.

----

## [584] Robustness of Graph Neural Networks at Scale

**Authors**: *Simon Geisler, Tobias Schmidt, Hakan Sirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html)

**Abstract**:

Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.

----

## [585] Random Noise Defense Against Query-Based Black-Box Attacks

**Authors**: *Zeyu Qin, Yanbo Fan, Hongyuan Zha, Baoyuan Wu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3eb414bf1c2a66a09c185d60553417b8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3eb414bf1c2a66a09c185d60553417b8-Abstract.html)

**Abstract**:

The query-based black-box attacks have raised serious threats to machine learning models in many real applications. In this work, we study a lightweight defense method, dubbed Random Noise Defense (RND), which adds proper Gaussian noise to each query. We conduct the theoretical analysis about the effectiveness of RND against query-based black-box attacks and the corresponding adaptive attacks. Our theoretical results reveal that the defense performance of RND is determined by the magnitude ratio between the noise induced by RND and the noise added by the attackers for gradient estimation or local search.  The large magnitude ratio leads to the stronger defense performance of RND, and it's also critical for mitigating adaptive attacks. Based on our analysis, we further propose to combine RND with a plausible Gaussian augmentation Fine-tuning (RND-GF). It enables RND to add larger noise to each query while maintaining the clean accuracy to obtain a better trade-off between clean accuracy and defense performance. Additionally, RND can be flexibly combined with the existing defense methods to further boost the adversarial robustness, such as adversarial training (AT). Extensive experiments on CIFAR-10 and ImageNet verify our theoretical findings and the effectiveness of RND and RND-GF.

----

## [586] SADGA: Structure-Aware Dual Graph Aggregation Network for Text-to-SQL

**Authors**: *Ruichu Cai, Jinjie Yuan, Boyan Xu, Zhifeng Hao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3f1656d9668dffcf8119e3ecff873558-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3f1656d9668dffcf8119e3ecff873558-Abstract.html)

**Abstract**:

The Text-to-SQL task, aiming to translate the natural language of the questions into SQL queries, has drawn much attention recently.  One of the most challenging problems of Text-to-SQL is how to generalize the trained model to the unseen database schemas, also known as the cross-domain Text-to-SQL task. The key lies in the generalizability of (i) the encoding method to model the question and the database schema and (ii) the question-schema linking method to learn the mapping between words in the question and tables/columns in the database schema. Focusing on the above two key issues, we propose a \emph{Structure-Aware Dual Graph Aggregation Network} (SADGA) for cross-domain Text-to-SQL. In SADGA, we adopt the graph structure to provide a unified encoding model for both the natural language question and database schema. Based on the proposed unified modeling, we further devise a structure-aware aggregation method to learn the mapping between the question-graph and schema-graph. The structure-aware aggregation method is featured with \emph{Global Graph Linking}, \emph{Local Graph Linking} and \emph{Dual-Graph Aggregation Mechanism}. We not only study the performance of our proposal empirically but also achieved 3rd place on the challenging Text-to-SQL benchmark Spider at the time of writing.

----

## [587] Near-Optimal Offline Reinforcement Learning via Double Variance Reduction

**Authors**: *Ming Yin, Yu Bai, Yu-Xiang Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3f24bb08a5741e4197af64e1f93a5029-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3f24bb08a5741e4197af64e1f93a5029-Abstract.html)

**Abstract**:

We consider the problem of offline reinforcement learning (RL)  --- a well-motivated setting of RL that aims at policy optimization using only historical data. Despite its wide applicability, theoretical understandings of offline RL, such as its optimal sample complexity, remain largely open even in basic settings such as \emph{tabular} Markov Decision Processes (MDPs).  In this paper, we propose \emph{Off-Policy Double Variance Reduction} (OPDVR), a new variance reduction-based algorithm for offline RL. Our main result shows that OPDVR provably identifies an $\epsilon$-optimal policy with $\widetilde{O}(H^2/d_m\epsilon^2)$ episodes of offline data in the finite-horizon \emph{stationary transition} setting, where $H$ is the horizon length and $d_m$ is the minimal marginal state-action distribution induced by the behavior policy. This improves over the best-known upper bound by a factor of $H$. Moreover, we establish an information-theoretic lower bound of $\Omega(H^2/d_m\epsilon^2)$ which certifies that OPDVR is optimal up to logarithmic factors.  Lastly, we show that OPDVR also achieves rate-optimal sample complexity under alternative settings such as the finite-horizon MDPs with non-stationary transitions and the infinite horizon MDPs with discounted rewards.

----

## [588] Joint Modeling of Visual Objects and Relations for Scene Graph Generation

**Authors**: *Minghao Xu, Meng Qu, Bingbing Ni, Jian Tang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3f67fd97162d20e6fe27748b5b372509-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3f67fd97162d20e6fe27748b5b372509-Abstract.html)

**Abstract**:

An in-depth scene understanding usually requires recognizing all the objects and their relations in an image, encoded as a scene graph. Most existing approaches for scene graph generation first independently recognize each object and then predict their relations independently. Though these approaches are very efficient, they ignore the dependency between different objects as well as between their relations. In this paper, we propose a principled approach to jointly predict the entire scene graph by fully capturing the dependency between different objects and between their relations. Specifically, we establish a unified conditional random field (CRF) to model the joint distribution of all the objects and their relations in a scene graph. We carefully design the potential functions to enable relational reasoning among different objects according to knowledge graph embedding methods. We further propose an efficient and effective algorithm for inference based on mean-field variational inference, in which we first provide a warm initialization by independently predicting the objects and their relations according to the current model, followed by a few iterations of relational reasoning. Experimental results on both the relationship retrieval and zero-shot relationship retrieval tasks prove the efficiency and efficacy of our proposed approach.

----

## [589] Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

**Authors**: *Kazuki Irie, Imanol Schlag, Róbert Csordás, Jürgen Schmidhuber*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3f9e3767ef3b10a0de4c256d7ef9805d-Abstract.html)

**Abstract**:

Transformers with linearised attention (''linear Transformers'') have demonstrated the practical scalability and effectiveness of outer product-based Fast Weight Programmers (FWPs) from the '90s. However, the original FWP formulation is more general than the one of linear Transformers: a slow neural network (NN) continually reprograms the weights of a fast NN with arbitrary architecture. In existing linear Transformers, both NNs are feedforward and consist of a single layer. Here we explore new variations by adding recurrence to the slow and fast nets. We evaluate our novel recurrent FWPs (RFWPs) on two synthetic algorithmic tasks (code execution and sequential ListOps), Wikitext-103 language models, and on the Atari 2600 2D game environment. Our models exhibit properties of Transformers and RNNs. In the reinforcement learning setting, we report large improvements over LSTM in several Atari games. Our code is public.

----

## [590] Reinforced Few-Shot Acquisition Function Learning for Bayesian Optimization

**Authors**: *Bing-Jing Hsieh, Ping-Chun Hsieh, Xi Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3fab5890d8113d0b5a4178201dc842ad-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3fab5890d8113d0b5a4178201dc842ad-Abstract.html)

**Abstract**:

Bayesian optimization (BO) conventionally relies on handcrafted acquisition functions (AFs) to sequentially determine the sample points. However, it has been widely observed in practice that the best-performing AF in terms of regret can vary significantly under different types of black-box functions. It has remained a challenge to design one AF that can attain the best performance over a wide variety of black-box functions. This paper aims to attack this challenge through the perspective of reinforced few-shot AF learning (FSAF). Specifically, we first connect the notion of AFs with Q-functions and view a deep Q-network (DQN) as a surrogate differentiable AF. While it serves as a natural idea to combine DQN and an existing few-shot learning method, we identify that such a direct combination does not perform well due to severe overfitting, which is particularly critical in BO due to the need of a versatile sampling policy. To address this, we present a Bayesian variant of DQN with the following three features: (i) It learns a distribution of Q-networks as AFs based on the Kullback-Leibler regularization framework. This inherently provides the uncertainty required in sampling for BO and mitigates overfitting. (ii) For the prior of the Bayesian DQN, we propose to use a demo policy induced by an off-the-shelf AF for better training stability. (iii) On the meta-level, we leverage the meta-loss of Bayesian model-agnostic meta-learning, which serves as a natural companion to the proposed FSAF. Moreover, with the proper design of the Q-networks, FSAF is general-purpose in that it is agnostic to the dimension and the cardinality of the input domain. Through extensive experiments, we demonstrate that the FSAF achieves comparable or better regrets than the state-of-the-art benchmarks on a wide variety of synthetic and real-world test functions.

----

## [591] Forster Decomposition and Learning Halfspaces with Noise

**Authors**: *Ilias Diakonikolas, Daniel Kane, Christos Tzamos*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3ff4cea152080fd7d692a8286a587a67-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3ff4cea152080fd7d692a8286a587a67-Abstract.html)

**Abstract**:

A Forster transform is an operation that turns a multivariate distribution into one with good anti-concentration properties. While a Forster transform does not always exist, we show that any distribution can be efficiently decomposed as a disjoint mixture of few distributions for which a Forster transform exists and can be computed efficiently. As the main application of this result, we obtain the first polynomial-time algorithm for distribution-independent PAC learning of halfspaces in the Massart noise model with strongly polynomial sample complexity, i.e., independent of the bit complexity of the examples. Previous algorithms for this learning problem incurred sample complexity scaling polynomially with the bit complexity, even though such a dependence is not information-theoretically necessary.

----

## [592] Cortico-cerebellar networks as decoupling neural interfaces

**Authors**: *Joseph Pemberton, Ellen Boven, Richard Apps, Rui Ponte Costa*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/3ffebb08d23c609875d7177ee769a3e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/3ffebb08d23c609875d7177ee769a3e9-Abstract.html)

**Abstract**:

The brain solves the credit assignment problem remarkably well. For credit to be assigned across neural networks they must, in principle, wait for specific neural computations to finish. How the brain deals with this inherent locking problem has remained unclear. Deep learning methods suffer from similar locking constraints both on the forward and feedback phase. Recently, decoupled neural interfaces (DNIs) were introduced as a solution to the forward and feedback locking problems in deep networks.Here we propose that a specialised brain region, the cerebellum, helps the cerebral cortex solve similar locking problems akin to DNIs. To demonstrate the potential of this framework we introduce a systems-level model in which a recurrent cortical network receives online temporal feedback predictions from a cerebellar module. We test this cortico-cerebellar recurrent neural network (ccRNN) model on a number of sensorimotor (line and digit drawing) and cognitive tasks (pattern recognition and caption generation) that have been shown to be cerebellar-dependent. In all tasks, we observe that ccRNNs facilitates learning while reducing ataxia-like behaviours, consistent with classical experimental observations. Moreover, our model also explains recent behavioural and neuronal observations while making several testable predictions across multiple levels.Overall, our work offers a novel perspective on the cerebellum as a brain-wide decoupling machine for efficient credit assignment and opens a new avenue between deep learning and neuroscience.

----

## [593] To The Point: Correspondence-driven monocular 3D category reconstruction

**Authors**: *Filippos Kokkinos, Iasonas Kokkinos*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/40008b9a5380fcacce3976bf7c08af5b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/40008b9a5380fcacce3976bf7c08af5b-Abstract.html)

**Abstract**:

We present To The Point (TTP), a method for reconstructing 3D objects from a single image using 2D to 3D correspondences given only foreground masks, a category specific template and optionally sparse keypoints for supervision.  We recover a 3D shape from a 2D image by first regressing the 2D positions corresponding to the 3D template vertices and then jointly estimating a rigid camera transform and non-rigid template deformation that optimally explain the 2D positions through the 3D shape projection. By relying on correspondences we use a simple per-sample optimization problem to replace CNN-based regression of camera pose and non-rigid deformation and thereby obtain substantially more accurate 3D reconstructions. We treat this optimization as a differentiable layer and train the whole system in an end-to-end manner using geometry-driven losses. We report systematic quantitative improvements on multiple categories and provide qualitative results comprising diverse shape, poses and texture prediction examples.

----

## [594] Proper Value Equivalence

**Authors**: *Christopher Grimm, André Barreto, Gregory Farquhar, David Silver, Satinder Singh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)

**Abstract**:

One of the main challenges in model-based reinforcement learning (RL) is to decide which aspects of the environment should be modeled. The value-equivalence (VE) principle proposes a simple answer to this question: a model should capture the aspects of the environment that are relevant for value-based planning. Technically, VE distinguishes models based on a set of policies and a set of functions: a model is said to be VE to the environment if the Bellman operators it induces for the policies yield the correct result when applied to the functions. As the number of policies and functions increase, the set of VE models shrinks, eventually collapsing to a single point corresponding to a perfect model. A fundamental question underlying the VE principle is thus how to select the smallest sets of policies and functions that are sufficient for planning. In this paper we take an important step towards answering this question. We start by generalizing the concept of VE to order-$k$ counterparts defined with respect to $k$ applications of the Bellman operator. This leads to a family of VE classes that increase in size as $k \rightarrow \infty$. In the limit, all functions become value functions, and we have a special instantiation of VE which we call proper VE or simply PVE. Unlike VE, the PVE class may contain multiple models even in the limit when all value functions are used. Crucially, all these models are sufficient for planning, meaning that they will yield an optimal policy despite the fact that they may ignore many aspects of the environment. We construct a loss function for learning PVE models and argue that popular algorithms such as MuZero can be understood as minimizing an upper bound for this loss. We leverage this connection to propose a modification to MuZero and show that it can lead to improved performance in practice.

----

## [595] Challenges and Opportunities in High Dimensional Variational Inference

**Authors**: *Akash Kumar Dhaka, Alejandro Catalina, Manushi Welandawe, Michael Riis Andersen, Jonathan H. Huggins, Aki Vehtari*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/404dcc91b2aeaa7caa47487d1483e48a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/404dcc91b2aeaa7caa47487d1483e48a-Abstract.html)

**Abstract**:

Current black-box variational inference (BBVI) methods require the user to make numerous design choices – such as the selection of variational objective and approximating family – yet there is little principled guidance on how to do so. We develop a conceptual framework and set of experimental tools to understand the effects of these choices, which we leverage to propose best practices for maximizing posterior approximation accuracy. Our approach is based on studying the pre-asymptotic tail behavior of the density ratios between the joint distribution and the variational approximation, then exploiting insights and tools from the importance sampling literature. Our framework and supporting experiments help to distinguish between the behavior of BBVI methods for approximating low-dimensional versus moderate-to-high-dimensional posteriors. In the latter case, we show that mass-covering variational objectives are difficult to optimize and do not improve accuracy, but flexible variational families can improve accuracy and the effectiveness of importance sampling – at the cost of additional optimization challenges. Therefore, for moderate-to-high-dimensional posteriors we recommend using the (mode-seeking) exclusive KL divergence since it is the easiest to optimize, and improving the variational family or using model parameter transformations to make the posterior and optimal variational approximation more similar. On the other hand, in low-dimensional settings, we show that heavy-tailed variational families and mass-covering divergences are effective and can increase the chances that the approximation can be improved by importance sampling.

----

## [596] On the Expressivity of Markov Reward

**Authors**: *David Abel, Will Dabney, Anna Harutyunyan, Mark K. Ho, Michael L. Littman, Doina Precup, Satinder Singh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/4079016d940210b4ae9ae7d41c4a2065-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4079016d940210b4ae9ae7d41c4a2065-Abstract.html)

**Abstract**:

Reward is the driving force for reinforcement-learning agents. This paper is dedicated to understanding the expressivity of reward as a way to capture tasks that we would want an agent to perform. We frame this study around three new abstract notions of “task” that might be desirable: (1) a set of acceptable behaviors, (2) a partial ordering over behaviors, or (3) a partial ordering over trajectories. Our main results prove that while reward can express many of these tasks, there exist instances of each task type that no Markov reward function can capture. We then provide a set of polynomial-time algorithms that construct a Markov reward function that allows an agent to optimize tasks of each of these three types, and correctly determine when no such reward function exists. We conclude with an empirical study that corroborates and illustrates our theoretical findings.

----

## [597] One More Step Towards Reality: Cooperative Bandits with Imperfect Communication

**Authors**: *Udari Madhushani, Abhimanyu Dubey, Naomi Ehrich Leonard, Alex Pentland*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/40cb228987243c91b2dd0b7c9c4a0856-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/40cb228987243c91b2dd0b7c9c4a0856-Abstract.html)

**Abstract**:

The cooperative bandit problem is increasingly becoming relevant due to its applications in large-scale decision-making. However, most research for this problem focuses exclusively on the setting with perfect communication, whereas in most real-world distributed settings, communication is often over stochastic networks, with arbitrary corruptions and delays. In this paper, we study cooperative bandit learning under three typical real-world communication scenarios, namely, (a) message-passing over stochastic time-varying networks, (b) instantaneous reward-sharing over a network with random delays, and (c) message-passing with adversarially corrupted rewards, including byzantine communication. For each of these environments, we propose decentralized algorithms that achieve competitive performance, along with near-optimal guarantees on the incurred group regret as well. Furthermore, in the setting with perfect  communication, we present an improved delayed-update algorithm that outperforms the existing state-of-the-art on various network topologies. Finally, we present tight network-dependent minimax lower bounds on the group regret. Our proposed algorithms are straightforward to implement and obtain competitive empirical performance.

----

## [598] Multi-Agent Reinforcement Learning in Stochastic Networked Systems

**Authors**: *Yiheng Lin, Guannan Qu, Longbo Huang, Adam Wierman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/412604be30f701b1b1e3124c252065e6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/412604be30f701b1b1e3124c252065e6-Abstract.html)

**Abstract**:

We study multi-agent reinforcement learning (MARL) in a stochastic network of agents. The objective is to find localized policies that maximize the (discounted) global reward. In general, scalability is a challenge in this setting because the size of the global state/action space can be exponential in the number of agents. Scalable algorithms are only known in cases where dependencies are static, fixed and local, e.g., between neighbors in a fixed, time-invariant underlying graph. In this work, we propose a Scalable Actor Critic framework that applies in settings where the dependencies can be non-local and stochastic, and provide a finite-time error bound that shows how the convergence rate depends on the speed of information spread in the network.  Additionally, as a byproduct of our analysis, we obtain novel finite-time convergence results for a general stochastic approximation scheme and for temporal difference learning with state aggregation, which apply beyond the setting of MARL in networked systems.

----

## [599] Neural Scene Flow Prior

**Authors**: *Xueqian Li, Jhony Kaesemodel Pontes, Simon Lucey*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/41263b9a46f6f8f22668476661614478-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/41263b9a46f6f8f22668476661614478-Abstract.html)

**Abstract**:

Before the deep learning revolution, many perception algorithms were based on runtime optimization in conjunction with a strong prior/regularization penalty. A prime example of this in computer vision is optical and scene flow. Supervised learning has largely displaced the need for explicit regularization. Instead, they rely on large amounts of labeled data to capture prior statistics, which are not always readily available for many problems. Although optimization is employed to learn the neural network, at runtime, the weights of this network are frozen. As a result, these learning solutions are domain-specific and do not generalize well to other statistically different scenarios. This paper revisits the scene flow problem that relies predominantly on runtime optimization and strong regularization. A central innovation here is the inclusion of a neural scene flow prior, which utilizes the architecture of neural networks as a new type of implicit regularizer. Unlike learning-based scene flow methods, optimization occurs at runtime, and our approach needs no offline datasets---making it ideal for deployment in new environments such as autonomous driving. We show that an architecture based exclusively on multilayer perceptrons (MLPs) can be used as a scene flow prior.  Our method attains competitive---if not better---results on scene flow benchmarks. Also, our neural prior's implicit and continuous scene flow representation allows us to estimate dense long-term correspondences across a sequence of point clouds. The dense motion information is represented by scene flow fields where points can be propagated through time by integrating motion vectors. We demonstrate such a capability by accumulating a sequence of lidar point clouds.

----



[Go to the previous page](NIPS-2021-list02.md)

[Go to the next page](NIPS-2021-list04.md)

[Go to the catalog section](README.md)