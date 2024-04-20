## [800] Decoupling Value and Policy for Generalization in Reinforcement Learning

**Authors**: *Roberta Raileanu, Rob Fergus*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/raileanu21a.html](http://proceedings.mlr.press/v139/raileanu21a.html)

**Abstract**:

Standard deep reinforcement learning algorithms use a shared representation for the policy and value function, especially when training directly from images. However, we argue that more information is needed to accurately estimate the value function than to learn the optimal policy. Consequently, the use of a shared representation for the policy and value function can lead to overfitting. To alleviate this problem, we propose two approaches which are combined to create IDAAC: Invariant Decoupled Advantage Actor-Critic. First, IDAAC decouples the optimization of the policy and value function, using separate networks to model them. Second, it introduces an auxiliary loss which encourages the representation to be invariant to task-irrelevant properties of the environment. IDAAC shows good generalization to unseen environments, achieving a new state-of-the-art on the Procgen benchmark and outperforming popular methods on DeepMind Control tasks with distractors. Our implementation is available at https://github.com/rraileanu/idaac.

----

## [801] Hierarchical Clustering of Data Streams: Scalable Algorithms and Approximation Guarantees

**Authors**: *Anand Rajagopalan, Fabio Vitale, Danny Vainstein, Gui Citovsky, Cecilia M. Procopiuc, Claudio Gentile*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rajagopalan21a.html](http://proceedings.mlr.press/v139/rajagopalan21a.html)

**Abstract**:

We investigate the problem of hierarchically clustering data streams containing metric data in R^d. We introduce a desirable invariance property for such algorithms, describe a general family of hyperplane-based methods enjoying this property, and analyze two scalable instances of this general family against recently popularized similarity/dissimilarity-based metrics for hierarchical clustering. We prove a number of new results related to the approximation ratios of these algorithms, improving in various ways over the literature on this subject. Finally, since our algorithms are principled but also very practical, we carry out an experimental comparison on both synthetic and real-world datasets showing competitive results against known baselines.

----

## [802] Differentially Private Sliced Wasserstein Distance

**Authors**: *Alain Rakotomamonjy, Liva Ralaivola*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rakotomamonjy21a.html](http://proceedings.mlr.press/v139/rakotomamonjy21a.html)

**Abstract**:

Developing machine learning methods that are privacy preserving is today a central topic of research, with huge practical impacts. Among the numerous ways to address privacy-preserving learning, we here take the perspective of computing the divergences between distributions under the Differential Privacy (DP) framework — being able to compute divergences between distributions is pivotal for many machine learning problems, such as learning generative models or domain adaptation problems. Instead of resorting to the popular gradient-based sanitization method for DP, we tackle the problem at its roots by focusing on the Sliced Wasserstein Distance and seamlessly making it differentially private. Our main contribution is as follows: we analyze the property of adding a Gaussian perturbation to the intrinsic randomized mechanism of the Sliced Wasserstein Distance, and we establish the sensitivity of the resulting differentially private mechanism. One of our important findings is that this DP mechanism transforms the Sliced Wasserstein distance into another distance, that we call the Smoothed Sliced Wasserstein Distance. This new differentially private distribution distance can be plugged into generative models and domain adaptation algorithms in a transparent way, and we empirically show that it yields highly competitive performance compared with gradient-based DP approaches from the literature, with almost no loss in accuracy for the domain adaptation problems that we consider.

----

## [803] Zero-Shot Text-to-Image Generation

**Authors**: *Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ramesh21a.html](http://proceedings.mlr.press/v139/ramesh21a.html)

**Abstract**:

Text-to-image generation has traditionally focused on finding better modeling assumptions for training on a fixed dataset. These assumptions might involve complex architectures, auxiliary losses, or side information such as object part labels or segmentation masks supplied during training. We describe a simple approach for this task based on a transformer that autoregressively models the text and image tokens as a single stream of data. With sufficient data and scale, our approach is competitive with previous domain-specific models when evaluated in a zero-shot fashion.

----

## [804] End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series

**Authors**: *Syama Sundar Rangapuram, Lucien D. Werner, Konstantinos Benidis, Pedro Mercado, Jan Gasthaus, Tim Januschowski*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rangapuram21a.html](http://proceedings.mlr.press/v139/rangapuram21a.html)

**Abstract**:

This paper presents a novel approach for hierarchical time series forecasting that produces coherent, probabilistic forecasts without requiring any explicit post-processing reconciliation. Unlike the state-of-the-art, the proposed method simultaneously learns from all time series in the hierarchy and incorporates the reconciliation step into a single trainable model. This is achieved by applying the reparameterization trick and casting reconciliation as an optimization problem with a closed-form solution. These model features make end-to-end learning of hierarchical forecasts possible, while accomplishing the challenging task of generating forecasts that are both probabilistic and coherent. Importantly, our approach also accommodates general aggregation constraints including grouped and temporal hierarchies. An extensive empirical evaluation on real-world hierarchical datasets demonstrates the advantages of the proposed approach over the state-of-the-art.

----

## [805] MSA Transformer

**Authors**: *Roshan Rao, Jason Liu, Robert Verkuil, Joshua Meier, John F. Canny, Pieter Abbeel, Tom Sercu, Alexander Rives*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rao21a.html](http://proceedings.mlr.press/v139/rao21a.html)

**Abstract**:

Unsupervised protein language models trained across millions of diverse sequences learn structure and function of proteins. Protein language models studied to date have been trained to perform inference from individual sequences. The longstanding approach in computational biology has been to make inferences from a family of evolutionarily related sequences by fitting a model to each family independently. In this work we combine the two paradigms. We introduce a protein language model which takes as input a set of sequences in the form of a multiple sequence alignment. The model interleaves row and column attention across the input sequences and is trained with a variant of the masked language modeling objective across many protein families. The performance of the model surpasses current state-of-the-art unsupervised structure learning methods by a wide margin, with far greater parameter efficiency than prior state-of-the-art protein language models.

----

## [806] Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting

**Authors**: *Kashif Rasul, Calvin Seward, Ingmar Schuster, Roland Vollgraf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rasul21a.html](http://proceedings.mlr.press/v139/rasul21a.html)

**Abstract**:

In this work, we propose TimeGrad, an autoregressive model for multivariate probabilistic time series forecasting which samples from the data distribution at each time step by estimating its gradient. To this end, we use diffusion probabilistic models, a class of latent variable models closely connected to score matching and energy-based methods. Our model learns gradients by optimizing a variational bound on the data likelihood and at inference time converts white noise into a sample of the distribution of interest through a Markov chain using Langevin sampling. We demonstrate experimentally that the proposed autoregressive denoising diffusion model is the new state-of-the-art multivariate probabilistic forecasting method on real-world data sets with thousands of correlated dimensions. We hope that this method is a useful tool for practitioners and lays the foundation for future research in this area.

----

## [807] Generative Particle Variational Inference via Estimation of Functional Gradients

**Authors**: *Neale Ratzlaff, Qinxun Bai, Fuxin Li, Wei Xu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ratzlaff21a.html](http://proceedings.mlr.press/v139/ratzlaff21a.html)

**Abstract**:

Recently, particle-based variational inference (ParVI) methods have gained interest because they can avoid arbitrary parametric assumptions that are common in variational inference. However, many ParVI approaches do not allow arbitrary sampling from the posterior, and the few that do allow such sampling suffer from suboptimality. This work proposes a new method for learning to approximately sample from the posterior distribution. We construct a neural sampler that is trained with the functional gradient of the KL-divergence between the empirical sampling distribution and the target distribution, assuming the gradient resides within a reproducing kernel Hilbert space. Our generative ParVI (GPVI) approach maintains the asymptotic performance of ParVI methods while offering the flexibility of a generative sampler. Through carefully constructed experiments, we show that GPVI outperforms previous generative ParVI methods such as amortized SVGD, and is competitive with ParVI as well as gold-standard approaches like Hamiltonian Monte Carlo for fitting both exactly known and intractable target distributions.

----

## [808] Enhancing Robustness of Neural Networks through Fourier Stabilization

**Authors**: *Netanel Raviv, Aidan Kelley, Minzhe Guo, Yevgeniy Vorobeychik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/raviv21a.html](http://proceedings.mlr.press/v139/raviv21a.html)

**Abstract**:

Despite the considerable success of neural networks in security settings such as malware detection, such models have proved vulnerable to evasion attacks, in which attackers make slight changes to inputs (e.g., malware) to bypass detection. We propose a novel approach, Fourier stabilization, for designing evasion-robust neural networks with binary inputs. This approach, which is complementary to other forms of defense, replaces the weights of individual neurons with robust analogs derived using Fourier analytic tools. The choice of which neurons to stabilize in a neural network is then a combinatorial optimization problem, and we propose several methods for approximately solving it. We provide a formal bound on the per-neuron drop in accuracy due to Fourier stabilization, and experimentally demonstrate the effectiveness of the proposed approach in boosting robustness of neural networks in several detection settings. Moreover, we show that our approach effectively composes with adversarial training.

----

## [809] Disentangling Sampling and Labeling Bias for Learning in Large-output Spaces

**Authors**: *Ankit Singh Rawat, Aditya Krishna Menon, Wittawat Jitkrittum, Sadeep Jayasumana, Felix X. Yu, Sashank J. Reddi, Sanjiv Kumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rawat21a.html](http://proceedings.mlr.press/v139/rawat21a.html)

**Abstract**:

Negative sampling schemes enable efficient training given a large number of classes, by offering a means to approximate a computationally expensive loss function that takes all labels into account. In this paper, we present a new connection between these schemes and loss modification techniques for countering label imbalance. We show that different negative sampling schemes implicitly trade-off performance on dominant versus rare labels. Further, we provide a unified means to explicitly tackle both sampling bias, arising from working with a subset of all labels, and labeling bias, which is inherent to the data due to label imbalance. We empirically verify our findings on long-tail classification and retrieval benchmarks.

----

## [810] Cross-domain Imitation from Observations

**Authors**: *Dripta S. Raychaudhuri, Sujoy Paul, Jeroen van Baar, Amit K. Roy-Chowdhury*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/raychaudhuri21a.html](http://proceedings.mlr.press/v139/raychaudhuri21a.html)

**Abstract**:

Imitation learning seeks to circumvent the difficulty in designing proper reward functions for training agents by utilizing expert behavior. With environments modeled as Markov Decision Processes (MDP), most of the existing imitation algorithms are contingent on the availability of expert demonstrations in the same MDP as the one in which a new imitation policy is to be learned. In this paper, we study the problem of how to imitate tasks when discrepancies exist between the expert and agent MDP. These discrepancies across domains could include differing dynamics, viewpoint, or morphology; we present a novel framework to learn correspondences across such domains. Importantly, in contrast to prior works, we use unpaired and unaligned trajectories containing only states in the expert domain, to learn this correspondence. We utilize a cycle-consistency constraint on both the state space and a domain agnostic latent space to do this. In addition, we enforce consistency on the temporal position of states via a normalized position estimator function, to align the trajectories across the two domains. Once this correspondence is found, we can directly transfer the demonstrations on one domain to the other and use it for imitation. Experiments across a wide variety of challenging domains demonstrate the efficacy of our approach.

----

## [811] Implicit Regularization in Tensor Factorization

**Authors**: *Noam Razin, Asaf Maman, Nadav Cohen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/razin21a.html](http://proceedings.mlr.press/v139/razin21a.html)

**Abstract**:

Recent efforts to unravel the mystery of implicit regularization in deep learning have led to a theoretical focus on matrix factorization — matrix completion via linear neural network. As a step further towards practical deep learning, we provide the first theoretical analysis of implicit regularization in tensor factorization — tensor completion via certain type of non-linear neural network. We circumvent the notorious difficulty of tensor problems by adopting a dynamical systems perspective, and characterizing the evolution induced by gradient descent. The characterization suggests a form of greedy low tensor rank search, which we rigorously prove under certain conditions, and empirically demonstrate under others. Motivated by tensor rank capturing the implicit regularization of a non-linear neural network, we empirically explore it as a measure of complexity, and find that it captures the essence of datasets on which neural networks generalize. This leads us to believe that tensor rank may pave way to explaining both implicit regularization in deep learning, and the properties of real-world data translating this implicit regularization to generalization.

----

## [812] Align, then memorise: the dynamics of learning with feedback alignment

**Authors**: *Maria Refinetti, Stéphane d'Ascoli, Ruben Ohana, Sebastian Goldt*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/refinetti21a.html](http://proceedings.mlr.press/v139/refinetti21a.html)

**Abstract**:

Direct Feedback Alignment (DFA) is emerging as an efficient and biologically plausible alternative to backpropagation for training deep neural networks. Despite relying on random feedback weights for the backward pass, DFA successfully trains state-of-the-art models such as Transformers. On the other hand, it notoriously fails to train convolutional networks. An understanding of the inner workings of DFA to explain these diverging results remains elusive. Here, we propose a theory of feedback alignment algorithms. We first show that learning in shallow networks proceeds in two steps: an alignment phase, where the model adapts its weights to align the approximate gradient with the true gradient of the loss function, is followed by a memorisation phase, where the model focuses on fitting the data. This two-step process has a degeneracy breaking effect: out of all the low-loss solutions in the landscape, a net-work trained with DFA naturally converges to the solution which maximises gradient alignment. We also identify a key quantity underlying alignment in deep linear networks: the conditioning of the alignment matrices. The latter enables a detailed understanding of the impact of data structure on alignment, and suggests a simple explanation for the well-known failure of DFA to train convolutional neural networks. Numerical experiments on MNIST and CIFAR10 clearly demonstrate degeneracy breaking in deep non-linear networks and show that the align-then-memorize process occurs sequentially from the bottom layers of the network to the top.

----

## [813] Classifying high-dimensional Gaussian mixtures: Where kernel methods fail and neural networks succeed

**Authors**: *Maria Refinetti, Sebastian Goldt, Florent Krzakala, Lenka Zdeborová*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/refinetti21b.html](http://proceedings.mlr.press/v139/refinetti21b.html)

**Abstract**:

A recent series of theoretical works showed that the dynamics of neural networks with a certain initialisation are well-captured by kernel methods. Concurrent empirical work demonstrated that kernel methods can come close to the performance of neural networks on some image classification tasks. These results raise the question of whether neural networks only learn successfully if kernels also learn successfully, despite being the more expressive function class. Here, we show that two-layer neural networks with *only a few neurons* achieve near-optimal performance on high-dimensional Gaussian mixture classification while lazy training approaches such as random features and kernel methods do not. Our analysis is based on the derivation of a set of ordinary differential equations that exactly track the dynamics of the network and thus allow to extract the asymptotic performance of the network as a function of regularisation or signal-to-noise ratio. We also show how over-parametrising the neural network leads to faster convergence, but does not improve its final performance.

----

## [814] Sharf: Shape-conditioned Radiance Fields from a Single View

**Authors**: *Konstantinos Rematas, Ricardo Martin-Brualla, Vittorio Ferrari*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rematas21a.html](http://proceedings.mlr.press/v139/rematas21a.html)

**Abstract**:

We present a method for estimating neural scenes representations of objects given only a single image. The core of our method is the estimation of a geometric scaffold for the object and its use as a guide for the reconstruction of the underlying radiance field. Our formulation is based on a generative process that first maps a latent code to a voxelized shape, and then renders it to an image, with the object appearance being controlled by a second latent code. During inference, we optimize both the latent codes and the networks to fit a test image of a new object. The explicit disentanglement of shape and appearance allows our model to be fine-tuned given a single image. We can then render new views in a geometrically consistent manner and they represent faithfully the input object. Additionally, our method is able to generalize to images outside of the training domain (more realistic renderings and even real photographs). Finally, the inferred geometric scaffold is itself an accurate estimate of the object’s 3D shape. We demonstrate in several experiments the effectiveness of our approach in both synthetic and real images.

----

## [815] LEGO: Latent Execution-Guided Reasoning for Multi-Hop Question Answering on Knowledge Graphs

**Authors**: *Hongyu Ren, Hanjun Dai, Bo Dai, Xinyun Chen, Michihiro Yasunaga, Haitian Sun, Dale Schuurmans, Jure Leskovec, Denny Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ren21a.html](http://proceedings.mlr.press/v139/ren21a.html)

**Abstract**:

Answering complex natural language questions on knowledge graphs (KGQA) is a challenging task. It requires reasoning with the input natural language questions as well as a massive, incomplete heterogeneous KG. Prior methods obtain an abstract structured query graph/tree from the input question and traverse the KG for answers following the query tree. However, they inherently cannot deal with missing links in the KG. Here we present LEGO, a Latent Execution-Guided reasOning framework to handle this challenge in KGQA. LEGO works in an iterative way, which alternates between (1) a Query Synthesizer, which synthesizes a reasoning action and grows the query tree step-by-step, and (2) a Latent Space Executor that executes the reasoning action in the latent embedding space to combat against the missing information in KG. To learn the synthesizer without step-wise supervision, we design a generic latent execution guided bottom-up search procedure to find good execution traces efficiently in the vast query space. Experimental results on several KGQA benchmarks demonstrate the effectiveness of our framework compared with previous state of the art.

----

## [816] Interpreting and Disentangling Feature Components of Various Complexity from DNNs

**Authors**: *Jie Ren, Mingjie Li, Zexu Liu, Quanshi Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ren21b.html](http://proceedings.mlr.press/v139/ren21b.html)

**Abstract**:

This paper aims to define, visualize, and analyze the feature complexity that is learned by a DNN. We propose a generic definition for the feature complexity. Given the feature of a certain layer in the DNN, our method decomposes and visualizes feature components of different complexity orders from the feature. The feature decomposition enables us to evaluate the reliability, the effectiveness, and the significance of over-fitting of these feature components. Furthermore, such analysis helps to improve the performance of DNNs. As a generic method, the feature complexity also provides new insights into existing deep-learning techniques, such as network compression and knowledge distillation.

----

## [817] Integrated Defense for Resilient Graph Matching

**Authors**: *Jiaxiang Ren, Zijie Zhang, Jiayin Jin, Xin Zhao, Sixing Wu, Yang Zhou, Yelong Shen, Tianshi Che, Ruoming Jin, Dejing Dou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ren21c.html](http://proceedings.mlr.press/v139/ren21c.html)

**Abstract**:

A recent study has shown that graph matching models are vulnerable to adversarial manipulation of their input which is intended to cause a mismatching. Nevertheless, there is still a lack of a comprehensive solution for further enhancing the robustness of graph matching against adversarial attacks. In this paper, we identify and study two types of unique topology attacks in graph matching: inter-graph dispersion and intra-graph assembly attacks. We propose an integrated defense model, IDRGM, for resilient graph matching with two novel defense techniques to defend against the above two attacks simultaneously. A detection technique of inscribed simplexes in the hyperspheres consisting of multiple matched nodes is proposed to tackle inter-graph dispersion attacks, in which the distances among the matched nodes in multiple graphs are maximized to form regular simplexes. A node separation method based on phase-type distribution and maximum likelihood estimation is developed to estimate the distribution of perturbed graphs and separate the nodes within the same graphs over a wide space, for defending intra-graph assembly attacks, such that the interference from the similar neighbors of the perturbed nodes is significantly reduced. We evaluate the robustness of our IDRGM model on real datasets against state-of-the-art algorithms.

----

## [818] Solving high-dimensional parabolic PDEs using the tensor train format

**Authors**: *Lorenz Richter, Leon Sallandt, Nikolas Nüsken*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/richter21a.html](http://proceedings.mlr.press/v139/richter21a.html)

**Abstract**:

High-dimensional partial differential equations (PDEs) are ubiquitous in economics, science and engineering. However, their numerical treatment poses formidable challenges since traditional grid-based methods tend to be frustrated by the curse of dimensionality. In this paper, we argue that tensor trains provide an appealing approximation framework for parabolic PDEs: the combination of reformulations in terms of backward stochastic differential equations and regression-type methods in the tensor format holds the promise of leveraging latent low-rank structures enabling both compression and efficient computation. Following this paradigm, we develop novel iterative schemes, involving either explicit and fast or implicit and accurate updates. We demonstrate in a number of examples that our methods achieve a favorable trade-off between accuracy and computational efficiency in comparison with state-of-the-art neural network based approaches.

----

## [819] Best Arm Identification in Graphical Bilinear Bandits

**Authors**: *Geovani Rizk, Albert Thomas, Igor Colin, Rida Laraki, Yann Chevaleyre*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rizk21a.html](http://proceedings.mlr.press/v139/rizk21a.html)

**Abstract**:

We introduce a new graphical bilinear bandit problem where a learner (or a \emph{central entity}) allocates arms to the nodes of a graph and observes for each edge a noisy bilinear reward representing the interaction between the two end nodes. We study the best arm identification problem in which the learner wants to find the graph allocation maximizing the sum of the bilinear rewards. By efficiently exploiting the geometry of this bandit problem, we propose a \emph{decentralized} allocation strategy based on random sampling with theoretical guarantees. In particular, we characterize the influence of the graph structure (e.g. star, complete or circle) on the convergence rate and propose empirical experiments that confirm this dependency.

----

## [820] Principled Simplicial Neural Networks for Trajectory Prediction

**Authors**: *T. Mitchell Roddenberry, Nicholas Glaze, Santiago Segarra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/roddenberry21a.html](http://proceedings.mlr.press/v139/roddenberry21a.html)

**Abstract**:

We consider the construction of neural network architectures for data on simplicial complexes. In studying maps on the chain complex of a simplicial complex, we define three desirable properties of a simplicial neural network architecture: namely, permutation equivariance, orientation equivariance, and simplicial awareness. The first two properties respectively account for the fact that the node indexing and the simplex orientations in a simplicial complex are arbitrary. The last property encodes the desirable feature that the output of the neural network depends on the entire simplicial complex and not on a subset of its dimensions. Based on these properties, we propose a simple convolutional architecture, rooted in tools from algebraic topology, for the problem of trajectory prediction, and show that it obeys all three of these properties when an odd, nonlinear activation function is used. We then demonstrate the effectiveness of this architecture in extrapolating trajectories on synthetic and real datasets, with particular emphasis on the gains in generalizability to unseen trajectories.

----

## [821] On Linear Identifiability of Learned Representations

**Authors**: *Geoffrey Roeder, Luke Metz, Durk Kingma*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/roeder21a.html](http://proceedings.mlr.press/v139/roeder21a.html)

**Abstract**:

Identifiability is a desirable property of a statistical model: it implies that the true model parameters may be estimated to any desired precision, given sufficient computational resources and data. We study identifiability in the context of representation learning: discovering nonlinear data representations that are optimal with respect to some downstream task. When parameterized as deep neural networks, such representation functions lack identifiability in parameter space, because they are over-parameterized by design. In this paper, building on recent advances in nonlinear Independent Components Analysis, we aim to rehabilitate identifiability by showing that a large family of discriminative models are in fact identifiable in function space, up to a linear indeterminacy. Many models for representation learning in a wide variety of domains have been identifiable in this sense, including text, images and audio, state-of-the-art at time of publication. We derive sufficient conditions for linear identifiability and provide empirical support for the result on both simulated and real-world data.

----

## [822] Representation Matters: Assessing the Importance of Subgroup Allocations in Training Data

**Authors**: *Esther Rolf, Theodora T. Worledge, Benjamin Recht, Michael I. Jordan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rolf21a.html](http://proceedings.mlr.press/v139/rolf21a.html)

**Abstract**:

Collecting more diverse and representative training data is often touted as a remedy for the disparate performance of machine learning predictors across subpopulations. However, a precise framework for understanding how dataset properties like diversity affect learning outcomes is largely lacking. By casting data collection as part of the learning process, we demonstrate that diverse representation in training data is key not only to increasing subgroup performances, but also to achieving population-level objectives. Our analysis and experiments describe how dataset compositions influence performance and provide constructive results for using trends in existing data, alongside domain knowledge, to help guide intentional, objective-aware dataset design

----

## [823] TeachMyAgent: a Benchmark for Automatic Curriculum Learning in Deep RL

**Authors**: *Clément Romac, Rémy Portelas, Katja Hofmann, Pierre-Yves Oudeyer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/romac21a.html](http://proceedings.mlr.press/v139/romac21a.html)

**Abstract**:

Training autonomous agents able to generalize to multiple tasks is a key target of Deep Reinforcement Learning (DRL) research. In parallel to improving DRL algorithms themselves, Automatic Curriculum Learning (ACL) study how teacher algorithms can train DRL agents more efficiently by adapting task selection to their evolving abilities. While multiple standard benchmarks exist to compare DRL agents, there is currently no such thing for ACL algorithms. Thus, comparing existing approaches is difficult, as too many experimental parameters differ from paper to paper. In this work, we identify several key challenges faced by ACL algorithms. Based on these, we present TeachMyAgent (TA), a benchmark of current ACL algorithms leveraging procedural task generation. It includes 1) challenge-specific unit-tests using variants of a procedural Box2D bipedal walker environment, and 2) a new procedural Parkour environment combining most ACL challenges, making it ideal for global performance assessment. We then use TeachMyAgent to conduct a comparative study of representative existing approaches, showcasing the competitiveness of some ACL algorithms that do not use expert knowledge. We also show that the Parkour environment remains an open problem. We open-source our environments, all studied ACL algorithms (collected from open-source code or re-implemented), and DRL students in a Python package available at https://github.com/flowersteam/TeachMyAgent.

----

## [824] Discretization Drift in Two-Player Games

**Authors**: *Mihaela Rosca, Yan Wu, Benoit Dherin, David Barrett*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rosca21a.html](http://proceedings.mlr.press/v139/rosca21a.html)

**Abstract**:

Gradient-based methods for two-player games produce rich dynamics that can solve challenging problems, yet can be difficult to stabilize and understand. Part of this complexity originates from the discrete update steps given by simultaneous or alternating gradient descent, which causes each player to drift away from the continuous gradient flow – a phenomenon we call discretization drift. Using backward error analysis, we derive modified continuous dynamical systems that closely follow the discrete dynamics. These modified dynamics provide an insight into the notorious challenges associated with zero-sum games, including Generative Adversarial Networks. In particular, we identify distinct components of the discretization drift that can alter performance and in some cases destabilize the game. Finally, quantifying discretization drift allows us to identify regularizers that explicitly cancel harmful forms of drift or strengthen beneficial forms of drift, and thus improve performance of GAN training.

----

## [825] On the Predictability of Pruning Across Scales

**Authors**: *Jonathan S. Rosenfeld, Jonathan Frankle, Michael Carbin, Nir Shavit*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rosenfeld21a.html](http://proceedings.mlr.press/v139/rosenfeld21a.html)

**Abstract**:

We show that the error of iteratively magnitude-pruned networks empirically follows a scaling law with interpretable coefficients that depend on the architecture and task. We functionally approximate the error of the pruned networks, showing it is predictable in terms of an invariant tying width, depth, and pruning level, such that networks of vastly different pruned densities are interchangeable. We demonstrate the accuracy of this approximation over orders of magnitude in depth, width, dataset size, and density. We show that the functional form holds (generalizes) for large scale data (e.g., ImageNet) and architectures (e.g., ResNets). As neural networks become ever larger and costlier to train, our findings suggest a framework for reasoning conceptually and analytically about a standard method for unstructured pruning.

----

## [826] Benchmarks, Algorithms, and Metrics for Hierarchical Disentanglement

**Authors**: *Andrew Slavin Ross, Finale Doshi-Velez*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ross21a.html](http://proceedings.mlr.press/v139/ross21a.html)

**Abstract**:

In representation learning, there has been recent interest in developing algorithms to disentangle the ground-truth generative factors behind a dataset, and metrics to quantify how fully this occurs. However, these algorithms and metrics often assume that both representations and ground-truth factors are flat, continuous, and factorized, whereas many real-world generative processes involve rich hierarchical structure, mixtures of discrete and continuous variables with dependence between them, and even varying intrinsic dimensionality. In this work, we develop benchmarks, algorithms, and metrics for learning such hierarchical representations.

----

## [827] Simultaneous Similarity-based Self-Distillation for Deep Metric Learning

**Authors**: *Karsten Roth, Timo Milbich, Björn Ommer, Joseph Paul Cohen, Marzyeh Ghassemi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/roth21a.html](http://proceedings.mlr.press/v139/roth21a.html)

**Abstract**:

Deep Metric Learning (DML) provides a crucial tool for visual similarity and zero-shot retrieval applications by learning generalizing embedding spaces, although recent work in DML has shown strong performance saturation across training objectives. However, generalization capacity is known to scale with the embedding space dimensionality. Unfortunately, high dimensional embeddings also create higher retrieval cost for downstream applications. To remedy this, we propose S2SD - Simultaneous Similarity-based Self-distillation. S2SD extends DML with knowledge distillation from auxiliary, high-dimensional embedding and feature spaces to leverage complementary context during training while retaining test-time cost and with negligible changes to the training time. Experiments and ablations across different objectives and standard benchmarks show S2SD offering highly significant improvements of up to 7% in Recall@1, while also setting a new state-of-the-art.

----

## [828] Multi-group Agnostic PAC Learnability

**Authors**: *Guy N. Rothblum, Gal Yona*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rothblum21a.html](http://proceedings.mlr.press/v139/rothblum21a.html)

**Abstract**:

An agnostic PAC learning algorithm finds a predictor that is competitive with the best predictor in a benchmark hypothesis class, where competitiveness is measured with respect to a given loss function. However, its predictions might be quite sub-optimal for structured subgroups of individuals, such as protected demographic groups. Motivated by such fairness concerns, we study “multi-group agnostic PAC learnability”: fixing a measure of loss, a benchmark class $\H$ and a (potentially) rich collection of subgroups $\G$, the objective is to learn a single predictor such that the loss experienced by every group $g \in \G$ is not much larger than the best possible loss for this group within $\H$. Under natural conditions, we provide a characterization of the loss functions for which such a predictor is guaranteed to exist. For any such loss function we construct a learning algorithm whose sample complexity is logarithmic in the size of the collection $\G$. Our results unify and extend previous positive and negative results from the multi-group fairness literature, which applied for specific loss functions.

----

## [829] PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees

**Authors**: *Jonas Rothfuss, Vincent Fortuin, Martin Josifoski, Andreas Krause*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rothfuss21a.html](http://proceedings.mlr.press/v139/rothfuss21a.html)

**Abstract**:

Meta-learning can successfully acquire useful inductive biases from data. Yet, its generalization properties to unseen learning tasks are poorly understood. Particularly if the number of meta-training tasks is small, this raises concerns about overfitting. We provide a theoretical analysis using the PAC-Bayesian framework and derive novel generalization bounds for meta-learning. Using these bounds, we develop a class of PAC-optimal meta-learning algorithms with performance guarantees and a principled meta-level regularization. Unlike previous PAC-Bayesian meta-learners, our method results in a standard stochastic optimization problem which can be solved efficiently and scales well.When instantiating our PAC-optimal hyper-posterior (PACOH) with Gaussian processes and Bayesian Neural Networks as base learners, the resulting methods yield state-of-the-art performance, both in terms of predictive accuracy and the quality of uncertainty estimates. Thanks to their principled treatment of uncertainty, our meta-learners can also be successfully employed for sequential decision problems.

----

## [830] An Algorithm for Stochastic and Adversarial Bandits with Switching Costs

**Authors**: *Chloé Rouyer, Yevgeny Seldin, Nicolò Cesa-Bianchi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rouyer21a.html](http://proceedings.mlr.press/v139/rouyer21a.html)

**Abstract**:

We propose an algorithm for stochastic and adversarial multiarmed bandits with switching costs, where the algorithm pays a price $\lambda$ every time it switches the arm being played. Our algorithm is based on adaptation of the Tsallis-INF algorithm of Zimmert and Seldin (2021) and requires no prior knowledge of the regime or time horizon. In the oblivious adversarial setting it achieves the minimax optimal regret bound of $ O( (\lambda K)^{1/3}T^{2/3} + \sqrt{KT})$, where $T$ is the time horizon and $K$ is the number of arms. In the stochastically constrained adversarial regime, which includes the stochastic regime as a special case, it achieves a regret bound of $O((\lambda K)^{2/3} T^{1/3} + \ln T)\sum_{i \neq i^*} \Delta_i^{-1})$, where $\Delta_i$ are suboptimality gaps and $i^*$ is the unique optimal arm. In the special case of $\lambda = 0$ (no switching costs), both bounds are minimax optimal within constants. We also explore variants of the problem, where switching cost is allowed to change over time. We provide experimental evaluation showing competitiveness of our algorithm with the relevant baselines in the stochastic, stochastically constrained adversarial, and adversarial regimes with fixed switching cost.

----

## [831] Improving Lossless Compression Rates via Monte Carlo Bits-Back Coding

**Authors**: *Yangjun Ruan, Karen Ullrich, Daniel Severo, James Townsend, Ashish Khisti, Arnaud Doucet, Alireza Makhzani, Chris J. Maddison*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ruan21a.html](http://proceedings.mlr.press/v139/ruan21a.html)

**Abstract**:

Latent variable models have been successfully applied in lossless compression with the bits-back coding algorithm. However, bits-back suffers from an increase in the bitrate equal to the KL divergence between the approximate posterior and the true posterior. In this paper, we show how to remove this gap asymptotically by deriving bits-back coding algorithms from tighter variational bounds. The key idea is to exploit extended space representations of Monte Carlo estimators of the marginal likelihood. Naively applied, our schemes would require more initial bits than the standard bits-back coder, but we show how to drastically reduce this additional cost with couplings in the latent space. When parallel architectures can be exploited, our coders can achieve better rates than bits-back with little additional cost. We demonstrate improved lossless compression rates in a variety of settings, especially in out-of-distribution or sequential data compression.

----

## [832] On Signal-to-Noise Ratio Issues in Variational Inference for Deep Gaussian Processes

**Authors**: *Tim G. J. Rudner, Oscar Key, Yarin Gal, Tom Rainforth*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rudner21a.html](http://proceedings.mlr.press/v139/rudner21a.html)

**Abstract**:

We show that the gradient estimates used in training Deep Gaussian Processes (DGPs) with importance-weighted variational inference are susceptible to signal-to-noise ratio (SNR) issues. Specifically, we show both theoretically and via an extensive empirical evaluation that the SNR of the gradient estimates for the latent variable’s variational parameters decreases as the number of importance samples increases. As a result, these gradient estimates degrade to pure noise if the number of importance samples is too large. To address this pathology, we show how doubly-reparameterized gradient estimators, originally proposed for training variational autoencoders, can be adapted to the DGP setting and that the resultant estimators completely remedy the SNR issue, thereby providing more reliable training. Finally, we demonstrate that our fix can lead to consistent improvements in the predictive performance of DGP models.

----

## [833] Tilting the playing field: Dynamical loss functions for machine learning

**Authors**: *Miguel Ruiz-Garcia, Ge Zhang, Samuel S. Schoenholz, Andrea J. Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ruiz-garcia21a.html](http://proceedings.mlr.press/v139/ruiz-garcia21a.html)

**Abstract**:

We show that learning can be improved by using loss functions that evolve cyclically during training to emphasize one class at a time. In underparameterized networks, such dynamical loss functions can lead to successful training for networks that fail to find deep minima of the standard cross-entropy loss. In overparameterized networks, dynamical loss functions can lead to better generalization. Improvement arises from the interplay of the changing loss landscape with the dynamics of the system as it evolves to minimize the loss. In particular, as the loss function oscillates, instabilities develop in the form of bifurcation cascades, which we study using the Hessian and Neural Tangent Kernel. Valleys in the landscape widen and deepen, and then narrow and rise as the loss landscape changes during a cycle. As the landscape narrows, the learning rate becomes too large and the network becomes unstable and bounces around the valley. This process ultimately pushes the system into deeper and wider regions of the loss landscape and is characterized by decreasing eigenvalues of the Hessian. This results in better regularized models with improved generalization performance.

----

## [834] UnICORNN: A recurrent model for learning very long time dependencies

**Authors**: *T. Konstantin Rusch, Siddhartha Mishra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rusch21a.html](http://proceedings.mlr.press/v139/rusch21a.html)

**Abstract**:

The design of recurrent neural networks (RNNs) to accurately process sequential inputs with long-time dependencies is very challenging on account of the exploding and vanishing gradient problem. To overcome this, we propose a novel RNN architecture which is based on a structure preserving discretization of a Hamiltonian system of second-order ordinary differential equations that models networks of oscillators. The resulting RNN is fast, invertible (in time), memory efficient and we derive rigorous bounds on the hidden state gradients to prove the mitigation of the exploding and vanishing gradient problem. A suite of experiments are presented to demonstrate that the proposed RNN provides state of the art performance on a variety of learning tasks with (very) long-time dependencies.

----

## [835] Simple and Effective VAE Training with Calibrated Decoders

**Authors**: *Oleh Rybkin, Kostas Daniilidis, Sergey Levine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rybkin21a.html](http://proceedings.mlr.press/v139/rybkin21a.html)

**Abstract**:

Variational autoencoders (VAEs) provide an effective and simple method for modeling complex distributions. However, training VAEs often requires considerable hyperparameter tuning to determine the optimal amount of information retained by the latent variable. We study the impact of calibrated decoders, which learn the uncertainty of the decoding distribution and can determine this amount of information automatically, on the VAE performance. While many methods for learning calibrated decoders have been proposed, many of the recent papers that employ VAEs rely on heuristic hyperparameters and ad-hoc modifications instead. We perform the first comprehensive comparative analysis of calibrated decoder and provide recommendations for simple and effective VAE training. Our analysis covers a range of datasets and several single-image and sequential VAE models. We further propose a simple but novel modification to the commonly used Gaussian decoder, which computes the prediction variance analytically. We observe empirically that using heuristic modifications is not necessary with our method.

----

## [836] Model-Based Reinforcement Learning via Latent-Space Collocation

**Authors**: *Oleh Rybkin, Chuning Zhu, Anusha Nagabandi, Kostas Daniilidis, Igor Mordatch, Sergey Levine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/rybkin21b.html](http://proceedings.mlr.press/v139/rybkin21b.html)

**Abstract**:

The ability to plan into the future while utilizing only raw high-dimensional observations, such as images, can provide autonomous agents with broad and general capabilities. However, realistic tasks require performing temporally extended reasoning, and cannot be solved with only myopic, short-sighted planning. Recent work in model-based reinforcement learning (RL) has shown impressive results on tasks that require only short-horizon reasoning. In this work, we study how the long-horizon planning abilities can be improved with an algorithm that optimizes over sequences of states, rather than actions, which allows better credit assignment. To achieve this, we draw on the idea of collocation and adapt it to the image-based setting by leveraging probabilistic latent variable models, resulting in an algorithm that optimizes trajectories over latent variables. Our latent collocation method (LatCo) provides a general and effective visual planning approach, and significantly outperforms prior model-based approaches on challenging visual control tasks with sparse rewards and long-term goals. See the videos on the supplementary website \url{https://sites.google.com/view/latco-mbrl/.}

----

## [837] Training Data Subset Selection for Regression with Controlled Generalization Error

**Authors**: *Durga Sivasubramanian, Rishabh K. Iyer, Ganesh Ramakrishnan, Abir De*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/s21a.html](http://proceedings.mlr.press/v139/s21a.html)

**Abstract**:

Data subset selection from a large number of training instances has been a successful approach toward efficient and cost-effective machine learning. However, models trained on a smaller subset may show poor generalization ability. In this paper, our goal is to design an algorithm for selecting a subset of the training data, so that the model can be trained quickly, without significantly sacrificing on accuracy. More specifically, we focus on data subset selection for $L_2$ regularized regression problems and provide a novel problem formulation which seeks to minimize the training loss with respect to both the trainable parameters and the subset of training data, subject to error bounds on the validation set. We tackle this problem using several technical innovations. First, we represent this problem with simplified constraints using the dual of the original training problem and show that the objective of this new representation is a monotone and $\alpha$-submodular function, for a wide variety of modeling choices. Such properties lead us to develop SELCON, an efficient majorization-minimization algorithm for data subset selection, that admits an approximation guarantee even when the training provides an imperfect estimate of the trained model. Finally, our experiments on several datasets show that SELCON trades off accuracy and efficiency more effectively than the current state-of-the-art.

----

## [838] Unsupervised Part Representation by Flow Capsules

**Authors**: *Sara Sabour, Andrea Tagliasacchi, Soroosh Yazdani, Geoffrey E. Hinton, David J. Fleet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sabour21a.html](http://proceedings.mlr.press/v139/sabour21a.html)

**Abstract**:

Capsule networks aim to parse images into a hierarchy of objects, parts and relations. While promising, they remain limited by an inability to learn effective low level part descriptions. To address this issue we propose a way to learn primary capsule encoders that detect atomic parts from a single image. During training we exploit motion as a powerful perceptual cue for part definition, with an expressive decoder for part generation within a layered image model with occlusion. Experiments demonstrate robust part discovery in the presence of multiple objects, cluttered backgrounds, and occlusion. The learned part decoder is shown to infer the underlying shape masks, effectively filling in occluded regions of the detected shapes. We evaluate FlowCapsules on unsupervised part segmentation and unsupervised image classification.

----

## [839] Stochastic Sign Descent Methods: New Algorithms and Better Theory

**Authors**: *Mher Safaryan, Peter Richtárik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/safaryan21a.html](http://proceedings.mlr.press/v139/safaryan21a.html)

**Abstract**:

Various gradient compression schemes have been proposed to mitigate the communication cost in distributed training of large scale machine learning models. Sign-based methods, such as signSGD (Bernstein et al., 2018), have recently been gaining popularity because of their simple compression rule and connection to adaptive gradient methods, like ADAM. In this paper, we analyze sign-based methods for non-convex optimization in three key settings: (i) standard single node, (ii) parallel with shared data and (iii) distributed with partitioned data. For single machine case, we generalize the previous analysis of signSGD relying on intuitive bounds on success probabilities and allowing even biased estimators. Furthermore, we extend the analysis to parallel setting within a parameter server framework, where exponentially fast noise reduction is guaranteed with respect to number of nodes, maintaining $1$-bit compression in both directions and using small mini-batch sizes. Next, we identify a fundamental issue with signSGD to converge in distributed environment. To resolve this issue, we propose a new sign-based method, {\em Stochastic Sign Descent with Momentum (SSDM)}, which converges under standard bounded variance assumption with the optimal asymptotic rate. We validate several aspects of our theoretical findings with numerical experiments.

----

## [840] Adversarial Dueling Bandits

**Authors**: *Aadirupa Saha, Tomer Koren, Yishay Mansour*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/saha21a.html](http://proceedings.mlr.press/v139/saha21a.html)

**Abstract**:

We introduce the problem of regret minimization in Adversarial Dueling Bandits. As in classic Dueling Bandits, the learner has to repeatedly choose a pair of items and observe only a relative binary ‘win-loss’ feedback for this pair, but here this feedback is generated from an arbitrary preference matrix, possibly chosen adversarially. Our main result is an algorithm whose $T$-round regret compared to the \emph{Borda-winner} from a set of $K$ items is $\tilde{O}(K^{1/3}T^{2/3})$, as well as a matching $\Omega(K^{1/3}T^{2/3})$ lower bound. We also prove a similar high probability regret bound. We further consider a simpler \emph{fixed-gap} adversarial setup, which bridges between two extreme preference feedback models for dueling bandits: stationary preferences and an arbitrary sequence of preferences. For the fixed-gap adversarial setup we give an $\smash{ \tilde{O}((K/\Delta^2)\log{T}) }$ regret algorithm, where $\Delta$ is the gap in Borda scores between the best item and all other items, and show a lower bound of $\Omega(K/\Delta^2)$ indicating that our dependence on the main problem parameters $K$ and $\Delta$ is tight (up to logarithmic factors). Finally, we corroborate the theoretical results with empirical evaluations.

----

## [841] Dueling Convex Optimization

**Authors**: *Aadirupa Saha, Tomer Koren, Yishay Mansour*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/saha21b.html](http://proceedings.mlr.press/v139/saha21b.html)

**Abstract**:

We address the problem of convex optimization with preference (dueling) feedback. Like the traditional optimization objective, the goal is to find the optimal point with the least possible query complexity, however, without the luxury of even a zeroth order feedback. Instead, the learner can only observe a single noisy bit which is win-loss feedback for a pair of queried points based on their function values. % The problem is certainly of great practical relevance as in many real-world scenarios, such as recommender systems or learning from customer preferences, where the system feedback is often restricted to just one binary-bit preference information. % We consider the problem of online convex optimization (OCO) solely by actively querying $\{0,1\}$ noisy-comparison feedback of decision point pairs, with the objective of finding a near-optimal point (function minimizer) with the least possible number of queries. %a very general class of monotonic, non-decreasing transfer functions, and analyze the problem for any $d$-dimensional smooth convex function. % For the non-stationary OCO setup, where the underlying convex function may change over time, we prove an impossibility result towards achieving the above objective. We next focus only on the stationary OCO problem, and our main contribution lies in designing a normalized gradient descent based algorithm towards finding a $\epsilon$-best optimal point. Towards this, our algorithm is shown to yield a convergence rate of $\tilde O(\nicefrac{d\beta}{\epsilon \nu^2})$ ($\nu$ being the noise parameter) when the underlying function is $\beta$-smooth. Further we show an improved convergence rate of just $\tilde O(\nicefrac{d\beta}{\alpha \nu^2} \log \frac{1}{\epsilon})$ when the function is additionally also $\alpha$-strongly convex.

----

## [842] Optimal regret algorithm for Pseudo-1d Bandit Convex Optimization

**Authors**: *Aadirupa Saha, Nagarajan Natarajan, Praneeth Netrapalli, Prateek Jain*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/saha21c.html](http://proceedings.mlr.press/v139/saha21c.html)

**Abstract**:

We study online learning with bandit feedback (i.e. learner has access to only zeroth-order oracle) where cost/reward functions $\f_t$ admit a "pseudo-1d" structure, i.e. $\f_t(\w) = \loss_t(\pred_t(\w))$ where the output of $\pred_t$ is one-dimensional. At each round, the learner observes context $\x_t$, plays prediction $\pred_t(\w_t; \x_t)$ (e.g. $\pred_t(\cdot)=⟨\x_t, \cdot⟩$) for some $\w_t \in \mathbb{R}^d$ and observes loss $\loss_t(\pred_t(\w_t))$ where $\loss_t$ is a convex Lipschitz-continuous function. The goal is to minimize the standard regret metric. This pseudo-1d bandit convex optimization problem (\SBCO) arises frequently in domains such as online decision-making or parameter-tuning in large systems. For this problem, we first show a regret lower bound of $\min(\sqrt{dT}, T^{3/4})$ for any algorithm, where $T$ is the number of rounds. We propose a new algorithm \sbcalg that combines randomized online gradient descent with a kernelized exponential weights method to exploit the pseudo-1d structure effectively, guaranteeing the {\em optimal} regret bound mentioned above, up to additional logarithmic factors. In contrast, applying state-of-the-art online convex optimization methods leads to $\tilde{O}\left(\min\left(d^{9.5}\sqrt{T},\sqrt{d}T^{3/4}\right)\right)$ regret, that is significantly suboptimal in terms of $d$.

----

## [843] Asymptotics of Ridge Regression in Convolutional Models

**Authors**: *Mojtaba Sahraee-Ardakan, Tung Mai, Anup B. Rao, Ryan A. Rossi, Sundeep Rangan, Alyson K. Fletcher*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sahraee-ardakan21a.html](http://proceedings.mlr.press/v139/sahraee-ardakan21a.html)

**Abstract**:

Understanding generalization and estimation error of estimators for simple models such as linear and generalized linear models has attracted a lot of attention recently. This is in part due to an interesting observation made in machine learning community that highly over-parameterized neural networks achieve zero training error, and yet they are able to generalize well over the test samples. This phenomenon is captured by the so called double descent curve, where the generalization error starts decreasing again after the interpolation threshold. A series of recent works tried to explain such phenomenon for simple models. In this work, we analyze the asymptotics of estimation error in ridge estimators for convolutional linear models. These convolutional inverse problems, also known as deconvolution, naturally arise in different fields such as seismology, imaging, and acoustics among others. Our results hold for a large class of input distributions that include i.i.d. features as a special case. We derive exact formulae for estimation error of ridge estimators that hold in a certain high-dimensional regime. We show the double descent phenomenon in our experiments for convolutional models and show that our theoretical results match the experiments.

----

## [844] Momentum Residual Neural Networks

**Authors**: *Michael E. Sander, Pierre Ablin, Mathieu Blondel, Gabriel Peyré*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sander21a.html](http://proceedings.mlr.press/v139/sander21a.html)

**Abstract**:

The training of deep residual neural networks (ResNets) with backpropagation has a memory cost that increases linearly with respect to the depth of the network. A simple way to circumvent this issue is to use reversible architectures. In this paper, we propose to change the forward rule of a ResNet by adding a momentum term. The resulting networks, momentum residual neural networks (MomentumNets), are invertible. Unlike previous invertible architectures, they can be used as a drop-in replacement for any existing ResNet block. We show that MomentumNets can be interpreted in the infinitesimal step size regime as second-order ordinary differential equations (ODEs) and exactly characterize how adding momentum progressively increases the representation capabilities of MomentumNets: they can learn any linear mapping up to a multiplicative factor, while ResNets cannot. In a learning to optimize setting, where convergence to a fixed point is required, we show theoretically and empirically that our method succeeds while existing invertible architectures fail. We show on CIFAR and ImageNet that MomentumNets have the same accuracy as ResNets, while having a much smaller memory footprint, and show that pre-trained MomentumNets are promising for fine-tuning models.

----

## [845] Meta-Learning Bidirectional Update Rules

**Authors**: *Mark Sandler, Max Vladymyrov, Andrey Zhmoginov, Nolan Miller, Tom Madams, Andrew Jackson, Blaise Agüera y Arcas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sandler21a.html](http://proceedings.mlr.press/v139/sandler21a.html)

**Abstract**:

In this paper, we introduce a new type of generalized neural network where neurons and synapses maintain multiple states. We show that classical gradient-based backpropagation in neural networks can be seen as a special case of a two-state network where one state is used for activations and another for gradients, with update rules derived from the chain rule. In our generalized framework, networks have neither explicit notion of nor ever receive gradients. The synapses and neurons are updated using a bidirectional Hebb-style update rule parameterized by a shared low-dimensional "genome". We show that such genomes can be meta-learned from scratch, using either conventional optimization techniques, or evolutionary strategies, such as CMA-ES. Resulting update rules generalize to unseen tasks and train faster than gradient descent based optimizers for several standard computer vision and synthetic tasks.

----

## [846] Recomposing the Reinforcement Learning Building Blocks with Hypernetworks

**Authors**: *Elad Sarafian, Shai Keynan, Sarit Kraus*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sarafian21a.html](http://proceedings.mlr.press/v139/sarafian21a.html)

**Abstract**:

The Reinforcement Learning (RL) building blocks, i.e. $Q$-functions and policy networks, usually take elements from the cartesian product of two domains as input. In particular, the input of the $Q$-function is both the state and the action, and in multi-task problems (Meta-RL) the policy can take a state and a context. Standard architectures tend to ignore these variables’ underlying interpretations and simply concatenate their features into a single vector. In this work, we argue that this choice may lead to poor gradient estimation in actor-critic algorithms and high variance learning steps in Meta-RL algorithms. To consider the interaction between the input variables, we suggest using a Hypernetwork architecture where a primary network determines the weights of a conditional dynamic network. We show that this approach improves the gradient approximation and reduces the learning step variance, which both accelerates learning and improves the final performance. We demonstrate a consistent improvement across different locomotion tasks and different algorithms both in RL (TD3 and SAC) and in Meta-RL (MAML and PEARL).

----

## [847] Towards Understanding Learning in Neural Networks with Linear Teachers

**Authors**: *Roei Sarussi, Alon Brutzkus, Amir Globerson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sarussi21a.html](http://proceedings.mlr.press/v139/sarussi21a.html)

**Abstract**:

Can a neural network minimizing cross-entropy learn linearly separable data? Despite progress in the theory of deep learning, this question remains unsolved. Here we prove that SGD globally optimizes this learning problem for a two-layer network with Leaky ReLU activations. The learned network can in principle be very complex. However, empirical evidence suggests that it often turns out to be approximately linear. We provide theoretical support for this phenomenon by proving that if network weights converge to two weight clusters, this will imply an approximately linear decision boundary. Finally, we show a condition on the optimization that leads to weight clustering. We provide empirical results that validate our theoretical analysis.

----

## [848] E(n) Equivariant Graph Neural Networks

**Authors**: *Victor Garcia Satorras, Emiel Hoogeboom, Max Welling*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/satorras21a.html](http://proceedings.mlr.press/v139/satorras21a.html)

**Abstract**:

This paper introduces a new model to learn graph neural networks equivariant to rotations, translations, reflections and permutations called E(n)-Equivariant Graph Neural Networks (EGNNs). In contrast with existing methods, our work does not require computationally expensive higher-order representations in intermediate layers while it still achieves competitive or better performance. In addition, whereas existing methods are limited to equivariance on 3 dimensional spaces, our model is easily scaled to higher-dimensional spaces. We demonstrate the effectiveness of our method on dynamical systems modelling, representation learning in graph autoencoders and predicting molecular properties.

----

## [849] A Representation Learning Perspective on the Importance of Train-Validation Splitting in Meta-Learning

**Authors**: *Nikunj Saunshi, Arushi Gupta, Wei Hu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/saunshi21a.html](http://proceedings.mlr.press/v139/saunshi21a.html)

**Abstract**:

An effective approach in meta-learning is to utilize multiple “train tasks” to learn a good initialization for model parameters that can help solve unseen “test tasks” with very few samples by fine-tuning from this initialization. Although successful in practice, theoretical understanding of such methods is limited. This work studies an important aspect of these methods: splitting the data from each task into train (support) and validation (query) sets during meta-training. Inspired by recent work (Raghu et al., 2020), we view such meta-learning methods through the lens of representation learning and argue that the train-validation split encourages the learned representation to be {\em low-rank} without compromising on expressivity, as opposed to the non-splitting variant that encourages high-rank representations. Since sample efficiency benefits from low-rankness, the splitting strategy will require very few samples to solve unseen test tasks. We present theoretical results that formalize this idea for linear representation learning on a subspace meta-learning instance, and experimentally verify this practical benefit of splitting in simulations and on standard meta-learning benchmarks.

----

## [850] Low-Rank Sinkhorn Factorization

**Authors**: *Meyer Scetbon, Marco Cuturi, Gabriel Peyré*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/scetbon21a.html](http://proceedings.mlr.press/v139/scetbon21a.html)

**Abstract**:

Several recent applications of optimal transport (OT) theory to machine learning have relied on regularization, notably entropy and the Sinkhorn algorithm. Because matrix-vector products are pervasive in the Sinkhorn algorithm, several works have proposed to \textit{approximate} kernel matrices appearing in its iterations using low-rank factors. Another route lies instead in imposing low-nonnegative rank constraints on the feasible set of couplings considered in OT problems, with no approximations on cost nor kernel matrices. This route was first explored by \citet{forrow2018statistical}, who proposed an algorithm tailored for the squared Euclidean ground cost, using a proxy objective that can be solved through the machinery of regularized 2-Wasserstein barycenters. Building on this, we introduce in this work a generic approach that aims at solving, in full generality, the OT problem under low-nonnegative rank constraints with arbitrary costs. Our algorithm relies on an explicit factorization of low-rank couplings as a product of \textit{sub-coupling} factors linked by a common marginal; similar to an NMF approach, we alternatively updates these factors. We prove the non-asymptotic stationary convergence of this algorithm and illustrate its efficiency on benchmark experiments.

----

## [851] Linear Transformers Are Secretly Fast Weight Programmers

**Authors**: *Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/schlag21a.html](http://proceedings.mlr.press/v139/schlag21a.html)

**Abstract**:

We show the formal equivalence of linearised self-attention mechanisms and fast weight controllers from the early ’90s, where a slow neural net learns by gradient descent to program the fast weights of another net through sequences of elementary programming instructions which are additive outer products of self-invented activation patterns (today called keys and values). Such Fast Weight Programmers (FWPs) learn to manipulate the contents of a finite memory and dynamically interact with it. We infer a memory capacity limitation of recent linearised softmax attention variants, and replace the purely additive outer products by a delta rule-like programming instruction, such that the FWP can more easily learn to correct the current mapping from keys to values. The FWP also learns to compute dynamically changing learning rates. We also propose a new kernel function to linearise attention which balances simplicity and effectiveness. We conduct experiments on synthetic retrieval problems as well as standard machine translation and language modelling tasks which demonstrate the benefits of our methods.

----

## [852] Descending through a Crowded Valley - Benchmarking Deep Learning Optimizers

**Authors**: *Robin M. Schmidt, Frank Schneider, Philipp Hennig*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/schmidt21a.html](http://proceedings.mlr.press/v139/schmidt21a.html)

**Abstract**:

Choosing the optimizer is considered to be among the most crucial design decisions in deep learning, and it is not an easy one. The growing literature now lists hundreds of optimization methods. In the absence of clear theoretical guidance and conclusive empirical evidence, the decision is often made based on anecdotes. In this work, we aim to replace these anecdotes, if not with a conclusive ranking, then at least with evidence-backed heuristics. To do so, we perform an extensive, standardized benchmark of fifteen particularly popular deep learning optimizers while giving a concise overview of the wide range of possible choices. Analyzing more than 50,000 individual runs, we contribute the following three points: (i) Optimizer performance varies greatly across tasks. (ii) We observe that evaluating multiple optimizers with default parameters works approximately as well as tuning the hyperparameters of a single, fixed optimizer. (iii) While we cannot discern an optimization method clearly dominating across all tested tasks, we identify a significantly reduced subset of specific optimizers and parameter choices that generally lead to competitive results in our experiments: Adam remains a strong contender, with newer methods failing to significantly and consistently outperform it. Our open-sourced results are available as challenging and well-tuned baselines for more meaningful evaluations of novel optimization methods without requiring any further computational efforts.

----

## [853] Equivariant message passing for the prediction of tensorial properties and molecular spectra

**Authors**: *Kristof Schütt, Oliver T. Unke, Michael Gastegger*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/schutt21a.html](http://proceedings.mlr.press/v139/schutt21a.html)

**Abstract**:

Message passing neural networks have become a method of choice for learning on graphs, in particular the prediction of chemical properties and the acceleration of molecular dynamics studies. While they readily scale to large training data sets, previous approaches have proven to be less data efficient than kernel methods. We identify limitations of invariant representations as a major reason and extend the message passing formulation to rotationally equivariant representations. On this basis, we propose the polarizable atom interaction neural network (PaiNN) and improve on common molecule benchmarks over previous networks, while reducing model size and inference time. We leverage the equivariant atomwise representations obtained by PaiNN for the prediction of tensorial properties. Finally, we apply this to the simulation of molecular spectra, achieving speedups of 4-5 orders of magnitude compared to the electronic structure reference.

----

## [854] Just How Toxic is Data Poisoning? A Unified Benchmark for Backdoor and Data Poisoning Attacks

**Authors**: *Avi Schwarzschild, Micah Goldblum, Arjun Gupta, John P. Dickerson, Tom Goldstein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/schwarzschild21a.html](http://proceedings.mlr.press/v139/schwarzschild21a.html)

**Abstract**:

Data poisoning and backdoor attacks manipulate training data in order to cause models to fail during inference. A recent survey of industry practitioners found that data poisoning is the number one concern among threats ranging from model stealing to adversarial attacks. However, it remains unclear exactly how dangerous poisoning methods are and which ones are more effective considering that these methods, even ones with identical objectives, have not been tested in consistent or realistic settings. We observe that data poisoning and backdoor attacks are highly sensitive to variations in the testing setup. Moreover, we find that existing methods may not generalize to realistic settings. While these existing works serve as valuable prototypes for data poisoning, we apply rigorous tests to determine the extent to which we should fear them. In order to promote fair comparison in future work, we develop standardized benchmarks for data poisoning and backdoor attacks.

----

## [855] Connecting Sphere Manifolds Hierarchically for Regularization

**Authors**: *Damien Scieur, Youngsung Kim*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/scieur21a.html](http://proceedings.mlr.press/v139/scieur21a.html)

**Abstract**:

This paper considers classification problems with hierarchically organized classes. We force the classifier (hyperplane) of each class to belong to a sphere manifold, whose center is the classifier of its super-class. Then, individual sphere manifolds are connected based on their hierarchical relations. Our technique replaces the last layer of a neural network by combining a spherical fully-connected layer with a hierarchical layer. This regularization is shown to improve the performance of widely used deep neural network architectures (ResNet and DenseNet) on publicly available datasets (CIFAR100, CUB200, Stanford dogs, Stanford cars, and Tiny-ImageNet).

----

## [856] Learning Intra-Batch Connections for Deep Metric Learning

**Authors**: *Jenny Denise Seidenschwarz, Ismail Elezi, Laura Leal-Taixé*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/seidenschwarz21a.html](http://proceedings.mlr.press/v139/seidenschwarz21a.html)

**Abstract**:

The goal of metric learning is to learn a function that maps samples to a lower-dimensional space where similar samples lie closer than dissimilar ones. Particularly, deep metric learning utilizes neural networks to learn such a mapping. Most approaches rely on losses that only take the relations between pairs or triplets of samples into account, which either belong to the same class or two different classes. However, these methods do not explore the embedding space in its entirety. To this end, we propose an approach based on message passing networks that takes all the relations in a mini-batch into account. We refine embedding vectors by exchanging messages among all samples in a given batch allowing the training process to be aware of its overall structure. Since not all samples are equally important to predict a decision boundary, we use an attention mechanism during message passing to allow samples to weigh the importance of each neighbor accordingly. We achieve state-of-the-art results on clustering and image retrieval on the CUB-200-2011, Cars196, Stanford Online Products, and In-Shop Clothes datasets. To facilitate further research, we make available the code and the models at https://github.com/dvl-tum/intra_batch_connections.

----

## [857] Top-k eXtreme Contextual Bandits with Arm Hierarchy

**Authors**: *Rajat Sen, Alexander Rakhlin, Lexing Ying, Rahul Kidambi, Dean Foster, Daniel N. Hill, Inderjit S. Dhillon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sen21a.html](http://proceedings.mlr.press/v139/sen21a.html)

**Abstract**:

Motivated by modern applications, such as online advertisement and recommender systems, we study the top-$k$ extreme contextual bandits problem, where the total number of arms can be enormous, and the learner is allowed to select $k$ arms and observe all or some of the rewards for the chosen arms. We first propose an algorithm for the non-extreme realizable setting, utilizing the Inverse Gap Weighting strategy for selecting multiple arms. We show that our algorithm has a regret guarantee of $O(k\sqrt{(A-k+1)T \log (|F|T)})$, where $A$ is the total number of arms and $F$ is the class containing the regression function, while only requiring $\tilde{O}(A)$ computation per time step. In the extreme setting, where the total number of arms can be in the millions, we propose a practically-motivated arm hierarchy model that induces a certain structure in mean rewards to ensure statistical and computational efficiency. The hierarchical structure allows for an exponential reduction in the number of relevant arms for each context, thus resulting in a regret guarantee of $O(k\sqrt{(\log A-k+1)T \log (|F|T)})$. Finally, we implement our algorithm using a hierarchical linear function class and show superior performance with respect to well-known benchmarks on simulated bandit feedback experiments using extreme multi-label classification datasets. On a dataset with three million arms, our reduction scheme has an average inference time of only 7.9 milliseconds, which is a 100x improvement.

----

## [858] Pure Exploration and Regret Minimization in Matching Bandits

**Authors**: *Flore Sentenac, Jialin Yi, Clément Calauzènes, Vianney Perchet, Milan Vojnovic*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sentenac21a.html](http://proceedings.mlr.press/v139/sentenac21a.html)

**Abstract**:

Finding an optimal matching in a weighted graph is a standard combinatorial problem. We consider its semi-bandit version where either a pair or a full matching is sampled sequentially. We prove that it is possible to leverage a rank-1 assumption on the adjacency matrix to reduce the sample complexity and the regret of off-the-shelf algorithms up to reaching a linear dependency in the number of vertices (up to to poly-log terms).

----

## [859] State Entropy Maximization with Random Encoders for Efficient Exploration

**Authors**: *Younggyo Seo, Lili Chen, Jinwoo Shin, Honglak Lee, Pieter Abbeel, Kimin Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/seo21a.html](http://proceedings.mlr.press/v139/seo21a.html)

**Abstract**:

Recent exploration methods have proven to be a recipe for improving sample-efficiency in deep reinforcement learning (RL). However, efficient exploration in high-dimensional observation spaces still remains a challenge. This paper presents Random Encoders for Efficient Exploration (RE3), an exploration method that utilizes state entropy as an intrinsic reward. In order to estimate state entropy in environments with high-dimensional observations, we utilize a k-nearest neighbor entropy estimator in the low-dimensional representation space of a convolutional encoder. In particular, we find that the state entropy can be estimated in a stable and compute-efficient manner by utilizing a randomly initialized encoder, which is fixed throughout training. Our experiments show that RE3 significantly improves the sample-efficiency of both model-free and model-based RL methods on locomotion and navigation tasks from DeepMind Control Suite and MiniGrid benchmarks. We also show that RE3 allows learning diverse behaviors without extrinsic rewards, effectively improving sample-efficiency in downstream tasks.

----

## [860] Online Submodular Resource Allocation with Applications to Rebalancing Shared Mobility Systems

**Authors**: *Pier Giuseppe Sessa, Ilija Bogunovic, Andreas Krause, Maryam Kamgarpour*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sessa21a.html](http://proceedings.mlr.press/v139/sessa21a.html)

**Abstract**:

Motivated by applications in shared mobility, we address the problem of allocating a group of agents to a set of resources to maximize a cumulative welfare objective. We model the welfare obtainable from each resource as a monotone DR-submodular function which is a-priori unknown and can only be learned by observing the welfare of selected allocations. Moreover, these functions can depend on time-varying contextual information. We propose a distributed scheme to maximize the cumulative welfare by designing a repeated game among the agents, who learn to act via regret minimization. We propose two design choices for the game rewards based on upper confidence bounds built around the unknown welfare functions. We analyze them theoretically, bounding the gap between the cumulative welfare of the game and the highest cumulative welfare obtainable in hindsight. Finally, we evaluate our approach in a realistic case study of rebalancing a shared mobility system (i.e., positioning vehicles in strategic areas). From observed trip data, our algorithm gradually learns the users’ demand pattern and improves the overall system operation.

----

## [861] RRL: Resnet as representation for Reinforcement Learning

**Authors**: *Rutav M. Shah, Vikash Kumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shah21a.html](http://proceedings.mlr.press/v139/shah21a.html)

**Abstract**:

The ability to autonomously learn behaviors via direct interactions in uninstrumented environments can lead to generalist robots capable of enhancing productivity or providing care in unstructured settings like homes. Such uninstrumented settings warrant operations only using the robot’s proprioceptive sensor such as onboard cameras, joint encoders, etc which can be challenging for policy learning owing to the high dimensionality and partial observability issues. We propose RRL: Resnet as representation for Reinforcement Learning {–} a straightforward yet effective approach that can learn complex behaviors directly from proprioceptive inputs. RRL fuses features extracted from pre-trained Resnet into the standard reinforcement learning pipeline and delivers results comparable to learning directly from the state. In a simulated dexterous manipulation benchmark, where the state of the art methods fails to make significant progress, RRL delivers contact rich behaviors. The appeal of RRL lies in its simplicity in bringing together progress from the fields of Representation Learning, Imitation Learning, and Reinforcement Learning. Its effectiveness in learning behaviors directly from visual inputs with performance and sample efficiency matching learning directly from the state, even in complex high dimensional domains, is far from obvious.

----

## [862] Equivariant Networks for Pixelized Spheres

**Authors**: *Mehran Shakerinava, Siamak Ravanbakhsh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shakerinava21a.html](http://proceedings.mlr.press/v139/shakerinava21a.html)

**Abstract**:

Pixelizations of Platonic solids such as the cube and icosahedron have been widely used to represent spherical data, from climate records to Cosmic Microwave Background maps. Platonic solids have well-known global symmetries. Once we pixelize each face of the solid, each face also possesses its own local symmetries in the form of Euclidean isometries. One way to combine these symmetries is through a hierarchy. However, this approach does not adequately model the interplay between the two levels of symmetry transformations. We show how to model this interplay using ideas from group theory, identify the equivariant linear maps, and introduce equivariant padding that respects these symmetries. Deep networks that use these maps as their building blocks generalize gauge equivariant CNNs on pixelized spheres. These deep networks achieve state-of-the-art results on semantic segmentation for climate data and omnidirectional image processing. Code is available at https://git.io/JGiZA.

----

## [863] Personalized Federated Learning using Hypernetworks

**Authors**: *Aviv Shamsian, Aviv Navon, Ethan Fetaya, Gal Chechik*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shamsian21a.html](http://proceedings.mlr.press/v139/shamsian21a.html)

**Abstract**:

Personalized federated learning is tasked with training machine learning models for multiple clients, each with its own data distribution. The goal is to train personalized models collaboratively while accounting for data disparities across clients and reducing communication costs. We propose a novel approach to this problem using hypernetworks, termed pFedHN for personalized Federated HyperNetworks. In this approach, a central hypernetwork model is trained to generate a set of models, one model for each client. This architecture provides effective parameter sharing across clients while maintaining the capacity to generate unique and diverse personal models. Furthermore, since hypernetwork parameters are never transmitted, this approach decouples the communication cost from the trainable model size. We test pFedHN empirically in several personalized federated learning challenges and find that it outperforms previous methods. Finally, since hypernetworks share information across clients, we show that pFedHN can generalize better to new clients whose distributions differ from any client observed during training.

----

## [864] On the Power of Localized Perceptron for Label-Optimal Learning of Halfspaces with Adversarial Noise

**Authors**: *Jie Shen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shen21a.html](http://proceedings.mlr.press/v139/shen21a.html)

**Abstract**:

We study {\em online} active learning of homogeneous halfspaces in $\mathbb{R}^d$ with adversarial noise where the overall probability of a noisy label is constrained to be at most $\nu$. Our main contribution is a Perceptron-like online active learning algorithm that runs in polynomial time, and under the conditions that the marginal distribution is isotropic log-concave and $\nu = \Omega(\epsilon)$, where $\epsilon \in (0, 1)$ is the target error rate, our algorithm PAC learns the underlying halfspace with near-optimal label complexity of $\tilde{O}\big(d \cdot \polylog(\frac{1}{\epsilon})\big)$ and sample complexity of $\tilde{O}\big(\frac{d}{\epsilon} \big)$. Prior to this work, existing online algorithms designed for tolerating the adversarial noise are subject to either label complexity polynomial in $\frac{1}{\epsilon}$, or suboptimal noise tolerance, or restrictive marginal distributions. With the additional prior knowledge that the underlying halfspace is $s$-sparse, we obtain attribute-efficient label complexity of $\tilde{O}\big( s \cdot \polylog(d, \frac{1}{\epsilon}) \big)$ and sample complexity of $\tilde{O}\big(\frac{s}{\epsilon} \cdot \polylog(d) \big)$. As an immediate corollary, we show that under the agnostic model where no assumption is made on the noise rate $\nu$, our active learner achieves an error rate of $O(OPT) + \epsilon$ with the same running time and label and sample complexity, where $OPT$ is the best possible error rate achievable by any homogeneous halfspace.

----

## [865] Sample-Optimal PAC Learning of Halfspaces with Malicious Noise

**Authors**: *Jie Shen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shen21b.html](http://proceedings.mlr.press/v139/shen21b.html)

**Abstract**:

We study efficient PAC learning of homogeneous halfspaces in $\mathbb{R}^d$ in the presence of malicious noise of Valiant (1985). This is a challenging noise model and only until recently has near-optimal noise tolerance bound been established under the mild condition that the unlabeled data distribution is isotropic log-concave. However, it remains unsettled how to obtain the optimal sample complexity simultaneously. In this work, we present a new analysis for the algorithm of Awasthi et al. (2017) and show that it essentially achieves the near-optimal sample complexity bound of $\tilde{O}(d)$, improving the best known result of $\tilde{O}(d^2)$. Our main ingredient is a novel incorporation of a matrix Chernoff-type inequality to bound the spectrum of an empirical covariance matrix for well-behaved distributions, in conjunction with a careful exploration of the localization schemes of Awasthi et al. (2017). We further extend the algorithm and analysis to the more general and stronger nasty noise model of Bshouty et al. (2002), showing that it is still possible to achieve near-optimal noise tolerance and sample complexity in polynomial time.

----

## [866] Backdoor Scanning for Deep Neural Networks through K-Arm Optimization

**Authors**: *Guangyu Shen, Yingqi Liu, Guanhong Tao, Shengwei An, Qiuling Xu, Siyuan Cheng, Shiqing Ma, Xiangyu Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shen21c.html](http://proceedings.mlr.press/v139/shen21c.html)

**Abstract**:

Back-door attack poses a severe threat to deep learning systems. It injects hidden malicious behaviors to a model such that any input stamped with a special pattern can trigger such behaviors. Detecting back-door is hence of pressing need. Many existing defense techniques use optimization to generate the smallest input pattern that forces the model to misclassify a set of benign inputs injected with the pattern to a target label. However, the complexity is quadratic to the number of class labels such that they can hardly handle models with many classes. Inspired by Multi-Arm Bandit in Reinforcement Learning, we propose a K-Arm optimization method for backdoor detection. By iteratively and stochastically selecting the most promising labels for optimization with the guidance of an objective function, we substantially reduce the complexity, allowing to handle models with many classes. Moreover, by iteratively refining the selection of labels to optimize, it substantially mitigates the uncertainty in choosing the right labels, improving detection accuracy. At the time of submission, the evaluation of our method on over 4000 models in the IARPA TrojAI competition from round 1 to the latest round 4 achieves top performance on the leaderboard. Our technique also supersedes five state-of-the-art techniques in terms of accuracy and the scanning time needed. The code of our work is available at https://github.com/PurduePAML/K-ARM_Backdoor_Optimization

----

## [867] State Relevance for Off-Policy Evaluation

**Authors**: *Simon P. Shen, Yecheng Jason Ma, Omer Gottesman, Finale Doshi-Velez*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shen21d.html](http://proceedings.mlr.press/v139/shen21d.html)

**Abstract**:

Importance sampling-based estimators for off-policy evaluation (OPE) are valued for their simplicity, unbiasedness, and reliance on relatively few assumptions. However, the variance of these estimators is often high, especially when trajectories are of different lengths. In this work, we introduce Omitting-States-Irrelevant-to-Return Importance Sampling (OSIRIS), an estimator which reduces variance by strategically omitting likelihood ratios associated with certain states. We formalize the conditions under which OSIRIS is unbiased and has lower variance than ordinary importance sampling, and we demonstrate these properties empirically.

----

## [868] SparseBERT: Rethinking the Importance Analysis in Self-attention

**Authors**: *Han Shi, Jiahui Gao, Xiaozhe Ren, Hang Xu, Xiaodan Liang, Zhenguo Li, James Tin-Yau Kwok*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shi21a.html](http://proceedings.mlr.press/v139/shi21a.html)

**Abstract**:

Transformer-based models are popularly used in natural language processing (NLP). Its core component, self-attention, has aroused widespread interest. To understand the self-attention mechanism, a direct method is to visualize the attention map of a pre-trained model. Based on the patterns observed, a series of efficient Transformers with different sparse attention masks have been proposed. From a theoretical perspective, universal approximability of Transformer-based models is also recently proved. However, the above understanding and analysis of self-attention is based on a pre-trained model. To rethink the importance analysis in self-attention, we study the significance of different positions in attention matrix during pre-training. A surprising result is that diagonal elements in the attention map are the least important compared with other attention positions. We provide a proof showing that these diagonal elements can indeed be removed without deteriorating model performance. Furthermore, we propose a Differentiable Attention Mask (DAM) algorithm, which further guides the design of the SparseBERT. Extensive experiments verify our interesting findings and illustrate the effect of the proposed algorithm.

----

## [869] Learning Gradient Fields for Molecular Conformation Generation

**Authors**: *Chence Shi, Shitong Luo, Minkai Xu, Jian Tang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shi21b.html](http://proceedings.mlr.press/v139/shi21b.html)

**Abstract**:

We study a fundamental problem in computational chemistry known as molecular conformation generation, trying to predict stable 3D structures from 2D molecular graphs. Existing machine learning approaches usually first predict distances between atoms and then generate a 3D structure satisfying the distances, where noise in predicted distances may induce extra errors during 3D coordinate generation. Inspired by the traditional force field methods for molecular dynamics simulation, in this paper, we propose a novel approach called ConfGF by directly estimating the gradient fields of the log density of atomic coordinates. The estimated gradient fields allow directly generating stable conformations via Langevin dynamics. However, the problem is very challenging as the gradient fields are roto-translation equivariant. We notice that estimating the gradient fields of atomic coordinates can be translated to estimating the gradient fields of interatomic distances, and hence develop a novel algorithm based on recent score-based generative models to effectively estimate these gradients. Experimental results across multiple tasks show that ConfGF outperforms previous state-of-the-art baselines by a significant margin.

----

## [870] Segmenting Hybrid Trajectories using Latent ODEs

**Authors**: *Ruian Shi, Quaid Morris*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shi21c.html](http://proceedings.mlr.press/v139/shi21c.html)

**Abstract**:

Smooth dynamics interrupted by discontinuities are known as hybrid systems and arise commonly in nature. Latent ODEs allow for powerful representation of irregularly sampled time series but are not designed to capture trajectories arising from hybrid systems. Here, we propose the Latent Segmented ODE (LatSegODE), which uses Latent ODEs to perform reconstruction and changepoint detection within hybrid trajectories featuring jump discontinuities and switching dynamical modes. Where it is possible to train a Latent ODE on the smooth dynamical flows between discontinuities, we apply the pruned exact linear time (PELT) algorithm to detect changepoints where latent dynamics restart, thereby maximizing the joint probability of a piece-wise continuous latent dynamical representation. We propose usage of the marginal likelihood as a score function for PELT, circumventing the need for model-complexity-based penalization. The LatSegODE outperforms baselines in reconstructive and segmentation tasks including synthetic data sets of sine waves, Lotka Volterra dynamics, and UCI Character Trajectories.

----

## [871] Deeply-Debiased Off-Policy Interval Estimation

**Authors**: *Chengchun Shi, Runzhe Wan, Victor Chernozhukov, Rui Song*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shi21d.html](http://proceedings.mlr.press/v139/shi21d.html)

**Abstract**:

Off-policy evaluation learns a target policy’s value with a historical dataset generated by a different behavior policy. In addition to a point estimate, many applications would benefit significantly from having a confidence interval (CI) that quantifies the uncertainty of the point estimate. In this paper, we propose a novel procedure to construct an efficient, robust, and flexible CI on a target policy’s value. Our method is justified by theoretical results and numerical experiments. A Python implementation of the proposed procedure is available at https://github.com/ RunzheStat/D2OPE.

----

## [872] GANMEX: One-vs-One Attributions using GAN-based Model Explainability

**Authors**: *Sheng-Min Shih, Pin-Ju Tien, Zohar Karnin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shih21a.html](http://proceedings.mlr.press/v139/shih21a.html)

**Abstract**:

Attribution methods have been shown as promising approaches for identifying key features that led to learned model predictions. While most existing attribution methods rely on a baseline input for performing feature perturbations, limited research has been conducted to address the baseline selection issues. Poor choices of baselines limit the ability of one-vs-one explanations for multi-class classifiers, which means the attribution methods were not able to explain why an input belongs to its original class but not the other specified target class. Achieving one-vs-one explanation is crucial when certain classes are more similar than others, e.g. two bird types among multiple animals, by focusing on key differentiating features rather than shared features across classes. In this paper, we present GANMEX, a novel approach applying Generative Adversarial Networks (GAN) by incorporating the to-be-explained classifier as part of the adversarial networks. Our approach effectively selects the baseline as the closest realistic sample belong to the target class, which allows attribution methods to provide true one-vs-one explanations. We showed that GANMEX baselines improved the saliency maps and led to stronger performance on multiple evaluation metrics over the existing baselines. Existing attribution results are known for being insensitive to model randomization, and we demonstrated that GANMEX baselines led to better outcome under the cascading randomization of the model.

----

## [873] Large-Scale Meta-Learning with Continual Trajectory Shifting

**Authors**: *Jaewoong Shin, Haebeom Lee, Boqing Gong, Sung Ju Hwang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shin21a.html](http://proceedings.mlr.press/v139/shin21a.html)

**Abstract**:

Meta-learning of shared initialization parameters has shown to be highly effective in solving few-shot learning tasks. However, extending the framework to many-shot scenarios, which may further enhance its practicality, has been relatively overlooked due to the technical difficulties of meta-learning over long chains of inner-gradient steps. In this paper, we first show that allowing the meta-learners to take a larger number of inner gradient steps better captures the structure of heterogeneous and large-scale task distributions, thus results in obtaining better initialization points. Further, in order to increase the frequency of meta-updates even with the excessively long inner-optimization trajectories, we propose to estimate the required shift of the task-specific parameters with respect to the change of the initialization parameters. By doing so, we can arbitrarily increase the frequency of meta-updates and thus greatly improve the meta-level convergence as well as the quality of the learned initializations. We validate our method on a heterogeneous set of large-scale tasks, and show that the algorithm largely outperforms the previous first-order meta-learning methods in terms of both generalization performance and convergence, as well as multi-task learning and fine-tuning baselines.

----

## [874] AGENT: A Benchmark for Core Psychological Reasoning

**Authors**: *Tianmin Shu, Abhishek Bhandwaldar, Chuang Gan, Kevin A. Smith, Shari Liu, Dan Gutfreund, Elizabeth S. Spelke, Joshua B. Tenenbaum, Tomer D. Ullman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shu21a.html](http://proceedings.mlr.press/v139/shu21a.html)

**Abstract**:

For machine agents to successfully interact with humans in real-world settings, they will need to develop an understanding of human mental life. Intuitive psychology, the ability to reason about hidden mental variables that drive observable actions, comes naturally to people: even pre-verbal infants can tell agents from objects, expecting agents to act efficiently to achieve goals given constraints. Despite recent interest in machine agents that reason about other agents, it is not clear if such agents learn or hold the core psychology principles that drive human reasoning. Inspired by cognitive development studies on intuitive psychology, we present a benchmark consisting of a large dataset of procedurally generated 3D animations, AGENT (Action, Goal, Efficiency, coNstraint, uTility), structured around four scenarios (goal preferences, action efficiency, unobserved constraints, and cost-reward trade-offs) that probe key concepts of core intuitive psychology. We validate AGENT with human-ratings, propose an evaluation protocol emphasizing generalization, and compare two strong baselines built on Bayesian inverse planning and a Theory of Mind neural network. Our results suggest that to pass the designed tests of core intuitive psychology at human levels, a model must acquire or have built-in representations of how agents plan, combining utility computations and core knowledge of objects and physics.

----

## [875] Zoo-Tuning: Adaptive Transfer from A Zoo of Models

**Authors**: *Yang Shu, Zhi Kou, Zhangjie Cao, Jianmin Wang, Mingsheng Long*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shu21b.html](http://proceedings.mlr.press/v139/shu21b.html)

**Abstract**:

With the development of deep networks on various large-scale datasets, a large zoo of pretrained models are available. When transferring from a model zoo, applying classic single-model-based transfer learning methods to each source model suffers from high computational cost and cannot fully utilize the rich knowledge in the zoo. We propose \emph{Zoo-Tuning} to address these challenges, which learns to adaptively transfer the parameters of pretrained models to the target task. With the learnable channel alignment layer and adaptive aggregation layer, Zoo-Tuning \emph{adaptively aggregates channel aligned pretrained parameters to derive the target model}, which simultaneously promotes knowledge transfer and adapts source models to downstream tasks. The adaptive aggregation substantially reduces the computation cost at both training and inference. We further propose lite Zoo-Tuning with the temporal ensemble of batch average gating values to reduce the storage cost at the inference time. We evaluate our approach on a variety of tasks, including reinforcement learning, image classification, and facial landmark detection. Experiment results demonstrate that the proposed adaptive transfer learning approach can more effectively and efficiently transfer knowledge from a zoo of models.

----

## [876] Aggregating From Multiple Target-Shifted Sources

**Authors**: *Changjian Shui, Zijian Li, Jiaqi Li, Christian Gagné, Charles X. Ling, Boyu Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/shui21a.html](http://proceedings.mlr.press/v139/shui21a.html)

**Abstract**:

Multi-source domain adaptation aims at leveraging the knowledge from multiple tasks for predicting a related target domain. Hence, a crucial aspect is to properly combine different sources based on their relations. In this paper, we analyzed the problem for aggregating source domains with different label distributions, where most recent source selection approaches fail. Our proposed algorithm differs from previous approaches in two key ways: the model aggregates multiple sources mainly through the similarity of semantic conditional distribution rather than marginal distribution; the model proposes a unified framework to select relevant sources for three popular scenarios, i.e., domain adaptation with limited label on target domain, unsupervised domain adaptation and label partial unsupervised domain adaption. We evaluate the proposed method through extensive experiments. The empirical results significantly outperform the baselines.

----

## [877] Testing Group Fairness via Optimal Transport Projections

**Authors**: *Nian Si, Karthyek Murthy, Jose H. Blanchet, Viet Anh Nguyen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/si21a.html](http://proceedings.mlr.press/v139/si21a.html)

**Abstract**:

We have developed a statistical testing framework to detect if a given machine learning classifier fails to satisfy a wide range of group fairness notions. Our test is a flexible, interpretable, and statistically rigorous tool for auditing whether exhibited biases are intrinsic to the algorithm or simply due to the randomness in the data. The statistical challenges, which may arise from multiple impact criteria that define group fairness and which are discontinuous on model parameters, are conveniently tackled by projecting the empirical measure to the set of group-fair probability models using optimal transport. This statistic is efficiently computed using linear programming, and its asymptotic distribution is explicitly obtained. The proposed framework can also be used to test for composite fairness hypotheses and fairness with multiple sensitive attributes. The optimal transport testing formulation improves interpretability by characterizing the minimal covariate perturbations that eliminate the bias observed in the audit.

----

## [878] On Characterizing GAN Convergence Through Proximal Duality Gap

**Authors**: *Sahil Sidheekh, Aroof Aimen, Narayanan C. Krishnan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sidheekh21a.html](http://proceedings.mlr.press/v139/sidheekh21a.html)

**Abstract**:

Despite the accomplishments of Generative Adversarial Networks (GANs) in modeling data distributions, training them remains a challenging task. A contributing factor to this difficulty is the non-intuitive nature of the GAN loss curves, which necessitates a subjective evaluation of the generated output to infer training progress. Recently, motivated by game theory, Duality Gap has been proposed as a domain agnostic measure to monitor GAN training. However, it is restricted to the setting when the GAN converges to a Nash equilibrium. But GANs need not always converge to a Nash equilibrium to model the data distribution. In this work, we extend the notion of duality gap to proximal duality gap that is applicable to the general context of training GANs where Nash equilibria may not exist. We show theoretically that the proximal duality gap can monitor the convergence of GANs to a broader spectrum of equilibria that subsumes Nash equilibria. We also theoretically establish the relationship between the proximal duality gap and the divergence between the real and generated data distributions for different GAN formulations. Our results provide new insights into the nature of GAN convergence. Finally, we validate experimentally the usefulness of proximal duality gap for monitoring and influencing GAN training.

----

## [879] A Precise Performance Analysis of Support Vector Regression

**Authors**: *Houssem Sifaou, Abla Kammoun, Mohamed-Slim Alouini*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sifaou21a.html](http://proceedings.mlr.press/v139/sifaou21a.html)

**Abstract**:

In this paper, we study the hard and soft support vector regression techniques applied to a set of $n$ linear measurements of the form $y_i=\boldsymbol{\beta}_\star^{T}{\bf x}_i +n_i$ where $\boldsymbol{\beta}_\star$ is an unknown vector, $\left\{{\bf x}_i\right\}_{i=1}^n$ are the feature vectors and $\left\{{n}_i\right\}_{i=1}^n$ model the noise. Particularly, under some plausible assumptions on the statistical distribution of the data, we characterize the feasibility condition for the hard support vector regression in the regime of high dimensions and, when feasible, derive an asymptotic approximation for its risk. Similarly, we study the test risk for the soft support vector regression as a function of its parameters. Our results are then used to optimally tune the parameters intervening in the design of hard and soft support vector regression algorithms. Based on our analysis, we illustrate that adding more samples may be harmful to the test performance of support vector regression, while it is always beneficial when the parameters are optimally selected. Such a result reminds a similar phenomenon observed in modern learning architectures according to which optimally tuned architectures present a decreasing test performance curve with respect to the number of samples.

----

## [880] Directed Graph Embeddings in Pseudo-Riemannian Manifolds

**Authors**: *Aaron Sim, Maciej Wiatrak, Angus Brayne, Páidí Creed, Saee Paliwal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sim21a.html](http://proceedings.mlr.press/v139/sim21a.html)

**Abstract**:

The inductive biases of graph representation learning algorithms are often encoded in the background geometry of their embedding space. In this paper, we show that general directed graphs can be effectively represented by an embedding model that combines three components: a pseudo-Riemannian metric structure, a non-trivial global topology, and a unique likelihood function that explicitly incorporates a preferred direction in embedding space. We demonstrate the representational capabilities of this method by applying it to the task of link prediction on a series of synthetic and real directed graphs from natural language applications and biology. In particular, we show that low-dimensional cylindrical Minkowski and anti-de Sitter spacetimes can produce equal or better graph representations than curved Riemannian manifolds of higher dimensions.

----

## [881] Collaborative Bayesian Optimization with Fair Regret

**Authors**: *Rachael Hwee Ling Sim, Yehong Zhang, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sim21b.html](http://proceedings.mlr.press/v139/sim21b.html)

**Abstract**:

Bayesian optimization (BO) is a popular tool for optimizing complex and costly-to-evaluate black-box objective functions. To further reduce the number of function evaluations, any party performing BO may be interested to collaborate with others to optimize the same objective function concurrently. To do this, existing BO algorithms have considered optimizing a batch of input queries in parallel and provided theoretical bounds on their cumulative regret reflecting inefficiency. However, when the objective function values are correlated with real-world rewards (e.g., money), parties may be hesitant to collaborate if they risk incurring larger cumulative regret (i.e., smaller real-world reward) than others. This paper shows that fairness and efficiency are both necessary for the collaborative BO setting. Inspired by social welfare concepts from economics, we propose a new notion of regret capturing these properties and a collaborative BO algorithm whose convergence rate can be theoretically guaranteed by bounding the new regret, both of which share an adjustable parameter for trading off between fairness vs. efficiency. We empirically demonstrate the benefits (e.g., increased fairness) of our algorithm using synthetic and real-world datasets.

----

## [882] Dynamic Planning and Learning under Recovering Rewards

**Authors**: *David Simchi-Levi, Zeyu Zheng, Feng Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/simchi-levi21a.html](http://proceedings.mlr.press/v139/simchi-levi21a.html)

**Abstract**:

Motivated by emerging applications such as live-streaming e-commerce, promotions and recommendations, we introduce a general class of multi-armed bandit problems that have the following two features: (i) the decision maker can pull and collect rewards from at most $K$ out of $N$ different arms in each time period; (ii) the expected reward of an arm immediately drops after it is pulled, and then non-parametrically recovers as the idle time increases. With the objective of maximizing expected cumulative rewards over $T$ time periods, we propose, construct and prove performance guarantees for a class of “Purely Periodic Policies”. For the offline problem when all model parameters are known, our proposed policy obtains an approximation ratio that is at the order of $1-\mathcal O(1/\sqrt{K})$, which is asymptotically optimal when $K$ grows to infinity. For the online problem when the model parameters are unknown and need to be learned, we design an Upper Confidence Bound (UCB) based policy that approximately has $\widetilde{\mathcal O}(N\sqrt{T})$ regret against the offline benchmark. Our framework and policy design may have the potential to be adapted into other offline planning and online learning applications with non-stationary and recovering rewards.

----

## [883] PopSkipJump: Decision-Based Attack for Probabilistic Classifiers

**Authors**: *Carl-Johann Simon-Gabriel, Noman Ahmed Sheikh, Andreas Krause*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/simon-gabriel21a.html](http://proceedings.mlr.press/v139/simon-gabriel21a.html)

**Abstract**:

Most current classifiers are vulnerable to adversarial examples, small input perturbations that change the classification output. Many existing attack algorithms cover various settings, from white-box to black-box classifiers, but usually assume that the answers are deterministic and often fail when they are not. We therefore propose a new adversarial decision-based attack specifically designed for classifiers with probabilistic outputs. It is based on the HopSkipJump attack by Chen et al. (2019), a strong and query efficient decision-based attack originally designed for deterministic classifiers. Our P(robabilisticH)opSkipJump attack adapts its amount of queries to maintain HopSkipJump’s original output quality across various noise levels, while converging to its query efficiency as the noise level decreases. We test our attack on various noise models, including state-of-the-art off-the-shelf randomized defenses, and show that they offer almost no extra robustness to decision-based attacks. Code is available at https://github.com/cjsg/PopSkipJump.

----

## [884] Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances

**Authors**: *Berfin Simsek, François Ged, Arthur Jacot, Francesco Spadaro, Clément Hongler, Wulfram Gerstner, Johanni Brea*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/simsek21a.html](http://proceedings.mlr.press/v139/simsek21a.html)

**Abstract**:

We study how permutation symmetries in overparameterized multi-layer neural networks generate ‘symmetry-induced’ critical points. Assuming a network with $ L $ layers of minimal widths $ r_1^*, \ldots, r_{L-1}^* $ reaches a zero-loss minimum at $ r_1^*! \cdots r_{L-1}^*! $ isolated points that are permutations of one another, we show that adding one extra neuron to each layer is sufficient to connect all these previously discrete minima into a single manifold. For a two-layer overparameterized network of width $ r^*+ h =: m $ we explicitly describe the manifold of global minima: it consists of $ T(r^*, m) $ affine subspaces of dimension at least $ h $ that are connected to one another. For a network of width $m$, we identify the number $G(r,m)$ of affine subspaces containing only symmetry-induced critical points that are related to the critical points of a smaller network of width $r
Cite this Paper

 BibTeX 

@InProceedings{pmlr-v139-simsek21a,
  title = 	 {Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances},
  author =       {Simsek, Berfin and Ged, Fran{\c{c}}ois and Jacot, Arthur and Spadaro, Francesco and Hongler, Clement and Gerstner, Wulfram and Brea, Johanni},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9722--9732},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/simsek21a/simsek21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/simsek21a.html},
  abstract = 	 {We study how permutation symmetries in overparameterized multi-layer neural networks generate ‘symmetry-induced’ critical points. Assuming a network with $ L $ layers of minimal widths $ r_1^*, \ldots, r_{L-1}^* $ reaches a zero-loss minimum at $ r_1^*! \cdots r_{L-1}^*! $ isolated points that are permutations of one another, we show that adding one extra neuron to each layer is sufficient to connect all these previously discrete minima into a single manifold. For a two-layer overparameterized network of width $ r^*+ h =: m $ we explicitly describe the manifold of global minima: it consists of $ T(r^*, m) $ affine subspaces of dimension at least $ h $ that are connected to one another. For a network of width $m$, we identify the number $G(r,m)$ of affine subspaces containing only symmetry-induced critical points that are related to the critical points of a smaller network of width $r
Copy to ClipboardDownload
 Endnote 
%0 Conference Paper
%T Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances
%A Berfin Simsek
%A François Ged
%A Arthur Jacot
%A Francesco Spadaro
%A Clement Hongler
%A Wulfram Gerstner
%A Johanni Brea
%B Proceedings of the 38th International Conference on Machine Learning
%C Proceedings of Machine Learning Research
%D 2021
%E Marina Meila
%E Tong Zhang	
%F pmlr-v139-simsek21a
%I PMLR
%P 9722--9732
%U https://proceedings.mlr.press/v139/simsek21a.html
%V 139
%X We study how permutation symmetries in overparameterized multi-layer neural networks generate ‘symmetry-induced’ critical points. Assuming a network with $ L $ layers of minimal widths $ r_1^*, \ldots, r_{L-1}^* $ reaches a zero-loss minimum at $ r_1^*! \cdots r_{L-1}^*! $ isolated points that are permutations of one another, we show that adding one extra neuron to each layer is sufficient to connect all these previously discrete minima into a single manifold. For a two-layer overparameterized network of width $ r^*+ h =: m $ we explicitly describe the manifold of global minima: it consists of $ T(r^*, m) $ affine subspaces of dimension at least $ h $ that are connected to one another. For a network of width $m$, we identify the number $G(r,m)$ of affine subspaces containing only symmetry-induced critical points that are related to the critical points of a smaller network of width $r
Copy to ClipboardDownload
 APA 

Simsek, B., Ged, F., Jacot, A., Spadaro, F., Hongler, C., Gerstner, W. & Brea, J.. (2021). Geometry of the Loss Landscape in Overparameterized Neural Networks: Symmetries and Invariances. Proceedings of the 38th International Conference on Machine Learning, in Proceedings of Machine Learning Research 139:9722-9732 Available from https://proceedings.mlr.press/v139/simsek21a.html.


Copy to ClipboardDownload

Related Material


Download PDF
Supplementary PDF

----

## [885] Flow-based Attribution in Graphical Models: A Recursive Shapley Approach

**Authors**: *Raghav Singal, George Michailidis, Hoiyi Ng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/singal21a.html](http://proceedings.mlr.press/v139/singal21a.html)

**Abstract**:

We study the attribution problem in a graphical model, wherein the objective is to quantify how the effect of changes at the source nodes propagates through the graph. We develop a model-agnostic flow-based attribution method, called recursive Shapley value (RSV). RSV generalizes a number of existing node-based methods and uniquely satisfies a set of flow-based axioms. In addition to admitting a natural characterization for linear models and facilitating mediation analysis for non-linear models, RSV satisfies a mix of desirable properties discussed in the recent literature, including implementation invariance, sensitivity, monotonicity, and affine scale invariance.

----

## [886] Structured World Belief for Reinforcement Learning in POMDP

**Authors**: *Gautam Singh, Skand Vishwanath Peri, Junghyun Kim, Hyunseok Kim, Sungjin Ahn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/singh21a.html](http://proceedings.mlr.press/v139/singh21a.html)

**Abstract**:

Object-centric world models provide structured representation of the scene and can be an important backbone in reinforcement learning and planning. However, existing approaches suffer in partially-observable environments due to the lack of belief states. In this paper, we propose Structured World Belief, a model for learning and inference of object-centric belief states. Inferred by Sequential Monte Carlo (SMC), our belief states provide multiple object-centric scene hypotheses. To synergize the benefits of SMC particles with object representations, we also propose a new object-centric dynamics model that considers the inductive bias of object permanence. This enables tracking of object states even when they are invisible for a long time. To further facilitate object tracking in this regime, we allow our model to attend flexibly to any spatial location in the image which was restricted in previous models. In experiments, we show that object-centric belief provides a more accurate and robust performance for filtering and generation. Furthermore, we show the efficacy of structured world belief in improving the performance of reinforcement learning, planning and supervised reasoning.

----

## [887] Skew Orthogonal Convolutions

**Authors**: *Sahil Singla, Soheil Feizi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/singla21a.html](http://proceedings.mlr.press/v139/singla21a.html)

**Abstract**:

Training convolutional neural networks with a Lipschitz constraint under the $l_{2}$ norm is useful for provable adversarial robustness, interpretable gradients, stable training, etc. While 1-Lipschitz networks can be designed by imposing a 1-Lipschitz constraint on each layer, training such networks requires each layer to be gradient norm preserving (GNP) to prevent gradients from vanishing. However, existing GNP convolutions suffer from slow training, lead to significant reduction in accuracy and provide no guarantees on their approximations. In this work, we propose a GNP convolution layer called \textbf{S}kew \textbf{O}rthogonal \textbf{C}onvolution (SOC) that uses the following mathematical property: when a matrix is {\it Skew-Symmetric}, its exponential function is an {\it orthogonal} matrix. To use this property, we first construct a convolution filter whose Jacobian is Skew-Symmetric. Then, we use the Taylor series expansion of the Jacobian exponential to construct the SOC layer that is orthogonal. To efficiently implement SOC, we keep a finite number of terms from the Taylor series and provide a provable guarantee on the approximation error. Our experiments on CIFAR-10 and CIFAR-100 show that SOC allows us to train provably Lipschitz, large convolutional neural networks significantly faster than prior works while achieving significant improvements for both standard and certified robust accuracies.

----

## [888] Multi-Task Reinforcement Learning with Context-based Representations

**Authors**: *Shagun Sodhani, Amy Zhang, Joelle Pineau*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sodhani21a.html](http://proceedings.mlr.press/v139/sodhani21a.html)

**Abstract**:

https://drive.google.com/file/d/1lRV72XaKoxZjgQrLXBJhsM82x54_1Vc4/view?usp=sharing

----

## [889] Shortest-Path Constrained Reinforcement Learning for Sparse Reward Tasks

**Authors**: *Sungryull Sohn, Sungtae Lee, Jongwook Choi, Harm van Seijen, Mehdi Fatemi, Honglak Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sohn21a.html](http://proceedings.mlr.press/v139/sohn21a.html)

**Abstract**:

We propose the k-Shortest-Path (k-SP) constraint: a novel constraint on the agent’s trajectory that improves the sample efficiency in sparse-reward MDPs. We show that any optimal policy necessarily satisfies the k-SP constraint. Notably, the k-SP constraint prevents the policy from exploring state-action pairs along the non-k-SP trajectories (e.g., going back and forth). However, in practice, excluding state-action pairs may hinder the convergence of RL algorithms. To overcome this, we propose a novel cost function that penalizes the policy violating SP constraint, instead of completely excluding it. Our numerical experiment in a tabular RL setting demonstrates that the SP-constraint can significantly reduce the trajectory space of policy. As a result, our constraint enables more sample efficient learning by suppressing redundant exploration and exploitation. Our experiments on MiniGrid, DeepMind Lab, Atari, and Fetch show that the proposed method significantly improves proximal policy optimization (PPO) and outperforms existing novelty-seeking exploration methods including count-based exploration even in continuous control tasks, indicating that it improves the sample efficiency by preventing the agent from taking redundant actions.

----

## [890] Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving

**Authors**: *Yang Song, Chenlin Meng, Renjie Liao, Stefano Ermon*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/song21a.html](http://proceedings.mlr.press/v139/song21a.html)

**Abstract**:

Feedforward computation, such as evaluating a neural network or sampling from an autoregressive model, is ubiquitous in machine learning. The sequential nature of feedforward computation, however, requires a strict order of execution and cannot be easily accelerated with parallel computing. To enable parallelization, we frame the task of feedforward computation as solving a system of nonlinear equations. We then propose to find the solution using a Jacobi or Gauss-Seidel fixed-point iteration method, as well as hybrid methods of both. Crucially, Jacobi updates operate independently on each equation and can be executed in parallel. Our method is guaranteed to give exactly the same values as the original feedforward computation with a reduced (or equal) number of parallelizable iterations, and hence reduced time given sufficient parallel computing power. Experimentally, we demonstrate the effectiveness of our approach in accelerating (i) backpropagation of RNNs, (ii) evaluation of DenseNets, and (iii) autoregressive sampling of MADE and PixelCNN++, with speedup factors between 2.1 and 26 under various settings.

----

## [891] PC-MLP: Model-based Reinforcement Learning with Policy Cover Guided Exploration

**Authors**: *Yuda Song, Wen Sun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/song21b.html](http://proceedings.mlr.press/v139/song21b.html)

**Abstract**:

Model-based Reinforcement Learning (RL) is a popular learning paradigm due to its potential sample efficiency compared to model-free RL. However, existing empirical model-based RL approaches lack the ability to explore. This work studies a computationally and statistically efficient model-based algorithm for both Kernelized Nonlinear Regulators (KNR) and linear Markov Decision Processes (MDPs). For both models, our algorithm guarantees polynomial sample complexity and only uses access to a planning oracle. Experimentally, we first demonstrate the flexibility and the efficacy of our algorithm on a set of exploration challenging control tasks where existing empirical model-based RL approaches completely fail. We then show that our approach retains excellent performance even in common dense reward control benchmarks that do not require heavy exploration.

----

## [892] Fast Sketching of Polynomial Kernels of Polynomial Degree

**Authors**: *Zhao Song, David P. Woodruff, Zheng Yu, Lichen Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/song21c.html](http://proceedings.mlr.press/v139/song21c.html)

**Abstract**:

Kernel methods are fundamental in machine learning, and faster algorithms for kernel approximation provide direct speedups for many core tasks in machine learning. The polynomial kernel is especially important as other kernels can often be approximated by the polynomial kernel via a Taylor series expansion. Recent techniques in oblivious sketching reduce the dependence in the running time on the degree $q$ of the polynomial kernel from exponential to polynomial, which is useful for the Gaussian kernel, for which $q$ can be chosen to be polylogarithmic. However, for more slowly growing kernels, such as the neural tangent and arc cosine kernels, $q$ needs to be polynomial, and previous work incurs a polynomial factor slowdown in the running time. We give a new oblivious sketch which greatly improves upon this running time, by removing the dependence on $q$ in the leading order term. Combined with a novel sampling scheme, we give the fastest algorithms for approximating a large family of slow-growing kernels.

----

## [893] Variance Reduction via Primal-Dual Accelerated Dual Averaging for Nonsmooth Convex Finite-Sums

**Authors**: *Chaobing Song, Stephen J. Wright, Jelena Diakonikolas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/song21d.html](http://proceedings.mlr.press/v139/song21d.html)

**Abstract**:

Structured nonsmooth convex finite-sum optimization appears in many machine learning applications, including support vector machines and least absolute deviation. For the primal-dual formulation of this problem, we propose a novel algorithm called \emph{Variance Reduction via Primal-Dual Accelerated Dual Averaging (\vrpda)}. In the nonsmooth and general convex setting, \vrpda has the overall complexity $O(nd\log\min \{1/\epsilon, n\} + d/\epsilon )$ in terms of the primal-dual gap, where $n$ denotes the number of samples, $d$ the dimension of the primal variables, and $\epsilon$ the desired accuracy. In the nonsmooth and strongly convex setting, the overall complexity of \vrpda becomes $O(nd\log\min\{1/\epsilon, n\} + d/\sqrt{\epsilon})$ in terms of both the primal-dual gap and the distance between iterate and optimal solution. Both these results for \vrpda improve significantly on state-of-the-art complexity estimates—which are $O(nd\log \min\{1/\epsilon, n\} + \sqrt{n}d/\epsilon)$ for the nonsmooth and general convex setting and $O(nd\log \min\{1/\epsilon, n\} + \sqrt{n}d/\sqrt{\epsilon})$ for the nonsmooth and strongly convex setting—with a simpler and more straightforward algorithm and analysis. Moreover, both complexities are better than \emph{lower} bounds for general convex finite-sum optimization, because our approach makes use of additional, commonly occurring structure. Numerical experiments reveal competitive performance of \vrpda compared to state-of-the-art approaches.

----

## [894] Oblivious Sketching-based Central Path Method for Linear Programming

**Authors**: *Zhao Song, Zheng Yu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/song21e.html](http://proceedings.mlr.press/v139/song21e.html)

**Abstract**:

In this work, we propose a sketching-based central path method for solving linear programmings, whose running time matches the state of the art results [Cohen, Lee, Song STOC 19; Lee, Song, Zhang COLT 19]. Our method opens up the iterations of the central path method and deploys an "iterate and sketch" approach towards the problem by introducing a new coordinate-wise embedding technique, which may be of independent interest. Compare to previous methods, the work [Cohen, Lee, Song STOC 19] enjoys feasibility while being non-oblivious, and [Lee, Song, Zhang COLT 19] is oblivious but infeasible, and relies on $\mathit{dense}$ sketching matrices such as subsampled randomized Hadamard/Fourier transform matrices. Our method enjoys the benefits of being both oblivious and feasible, and can use $\mathit{sparse}$ sketching matrix [Nelson, Nguyen FOCS 13] to speed up the online matrix-vector multiplication. Our framework for solving LP naturally generalizes to a broader class of convex optimization problems including empirical risk minimization.

----

## [895] Causal Curiosity: RL Agents Discovering Self-supervised Experiments for Causal Representation Learning

**Authors**: *Sumedh A. Sontakke, Arash Mehrjou, Laurent Itti, Bernhard Schölkopf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sontakke21a.html](http://proceedings.mlr.press/v139/sontakke21a.html)

**Abstract**:

Humans show an innate ability to learn the regularities of the world through interaction. By performing experiments in our environment, we are able to discern the causal factors of variation and infer how they affect the dynamics of our world. Analogously, here we attempt to equip reinforcement learning agents with the ability to perform experiments that facilitate a categorization of the rolled-out trajectories, and to subsequently infer the causal factors of the environment in a hierarchical manner. We introduce a novel intrinsic reward, called causal curiosity, and show that it allows our agents to learn optimal sequences of actions, and to discover causal factors in the dynamics. The learned behavior allows the agent to infer a binary quantized representation for the ground-truth causal factors in every environment. Additionally, we find that these experimental behaviors are semantically meaningful (e.g., to differentiate between heavy and light blocks, our agents learn to lift them), and are learnt in a self-supervised manner with approximately 2.5 times less data than conventional supervised planners. We show that these behaviors can be re-purposed and fine-tuned (e.g., from lifting to pushing or other downstream tasks). Finally, we show that the knowledge of causal factor representations aids zero-shot learning for more complex tasks.

----

## [896] Decomposed Mutual Information Estimation for Contrastive Representation Learning

**Authors**: *Alessandro Sordoni, Nouha Dziri, Hannes Schulz, Geoffrey J. Gordon, Philip Bachman, Remi Tachet des Combes*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sordoni21a.html](http://proceedings.mlr.press/v139/sordoni21a.html)

**Abstract**:

Recent contrastive representation learning methods rely on estimating mutual information (MI) between multiple views of an underlying context. E.g., we can derive multiple views of a given image by applying data augmentation, or we can split a sequence into views comprising the past and future of some step in the sequence. Contrastive lower bounds on MI are easy to optimize, but have a strong underestimation bias when estimating large amounts of MI. We propose decomposing the full MI estimation problem into a sum of smaller estimation problems by splitting one of the views into progressively more informed subviews and by applying the chain rule on MI between the decomposed views. This expression contains a sum of unconditional and conditional MI terms, each measuring modest chunks of the total MI, which facilitates approximation via contrastive bounds. To maximize the sum, we formulate a contrastive lower bound on the conditional MI which can be approximated efficiently. We refer to our general approach as Decomposed Estimation of Mutual Information (DEMI). We show that DEMI can capture a larger amount of MI than standard non-decomposed contrastive bounds in a synthetic setting, and learns better representations in a vision domain and for dialogue generation.

----

## [897] Decoupling Representation Learning from Reinforcement Learning

**Authors**: *Adam Stooke, Kimin Lee, Pieter Abbeel, Michael Laskin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/stooke21a.html](http://proceedings.mlr.press/v139/stooke21a.html)

**Abstract**:

In an effort to overcome limitations of reward-driven feature learning in deep reinforcement learning (RL) from images, we propose decoupling representation learning from policy learning. To this end, we introduce a new unsupervised learning (UL) task, called Augmented Temporal Contrast (ATC), which trains a convolutional encoder to associate pairs of observations separated by a short time difference, under image augmentations and using a contrastive loss. In online RL experiments, we show that training the encoder exclusively using ATC matches or outperforms end-to-end RL in most environments. Additionally, we benchmark several leading UL algorithms by pre-training encoders on expert demonstrations and using them, with weights frozen, in RL agents; we find that agents using ATC-trained encoders outperform all others. We also train multi-task encoders on data from multiple environments and show generalization to different downstream RL tasks. Finally, we ablate components of ATC, and introduce a new data augmentation to enable replay of (compressed) latent images from pre-trained encoders when RL requires augmentation. Our experiments span visually diverse RL benchmarks in DeepMind Control, DeepMind Lab, and Atari, and our complete code is available at \url{https://github.com/astooke/rlpyt/tree/master/rlpyt/ul}.

----

## [898] K-shot NAS: Learnable Weight-Sharing for NAS with K-shot Supernets

**Authors**: *Xiu Su, Shan You, Mingkai Zheng, Fei Wang, Chen Qian, Changshui Zhang, Chang Xu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/su21a.html](http://proceedings.mlr.press/v139/su21a.html)

**Abstract**:

In one-shot weight sharing for NAS, the weights of each operation (at each layer) are supposed to be identical for all architectures (paths) in the supernet. However, this rules out the possibility of adjusting operation weights to cater for different paths, which limits the reliability of the evaluation results. In this paper, instead of counting on a single supernet, we introduce $K$-shot supernets and take their weights for each operation as a dictionary. The operation weight for each path is represented as a convex combination of items in a dictionary with a simplex code. This enables a matrix approximation of the stand-alone weight matrix with a higher rank ($K>1$). A \textit{simplex-net} is introduced to produce architecture-customized code for each path. As a result, all paths can adaptively learn how to share weights in the $K$-shot supernets and acquire corresponding weights for better evaluation. $K$-shot supernets and simplex-net can be iteratively trained, and we further extend the search to the channel dimension. Extensive experiments on benchmark datasets validate that K-shot NAS significantly improves the evaluation accuracy of paths and thus brings in impressive performance improvements.

----

## [899] More Powerful and General Selective Inference for Stepwise Feature Selection using Homotopy Method

**Authors**: *Kazuya Sugiyama, Vo Nguyen Le Duy, Ichiro Takeuchi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sugiyama21a.html](http://proceedings.mlr.press/v139/sugiyama21a.html)

**Abstract**:

Conditional selective inference (SI) has been actively studied as a new statistical inference framework for data-driven hypotheses. The basic idea of conditional SI is to make inferences conditional on the selection event characterized by a set of linear and/or quadratic inequalities. Conditional SI has been mainly studied in the context of feature selection such as stepwise feature selection (SFS). The main limitation of the existing conditional SI methods is the loss of power due to over-conditioning, which is required for computational tractability. In this study, we develop a more powerful and general conditional SI method for SFS using the homotopy method which enables us to overcome this limitation. The homotopy-based SI is especially effective for more complicated feature selection algorithms. As an example, we develop a conditional SI method for forward-backward SFS with AIC-based stopping criteria and show that it is not adversely affected by the increased complexity of the algorithm. We conduct several experiments to demonstrate the effectiveness and efficiency of the proposed method.

----

## [900] Not All Memories are Created Equal: Learning to Forget by Expiring

**Authors**: *Sainbayar Sukhbaatar, Da Ju, Spencer Poff, Stephen Roller, Arthur Szlam, Jason Weston, Angela Fan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sukhbaatar21a.html](http://proceedings.mlr.press/v139/sukhbaatar21a.html)

**Abstract**:

Attention mechanisms have shown promising results in sequence modeling tasks that require long-term memory. Recent work investigated mechanisms to reduce the computational cost of preserving and storing memories. However, not all content in the past is equally important to remember. We propose Expire-Span, a method that learns to retain the most important information and expire the irrelevant information. This forgetting of memories enables Transformers to scale to attend over tens of thousands of previous timesteps efficiently, as not all states from previous timesteps are preserved. We demonstrate that Expire-Span can help models identify and retain critical information and show it can achieve strong performance on reinforcement learning tasks specifically designed to challenge this functionality. Next, we show that Expire-Span can scale to memories that are tens of thousands in size, setting a new state of the art on incredibly long context tasks such as character-level language modeling and a frame-by-frame moving objects task. Finally, we analyze the efficiency of Expire-Span compared to existing approaches and demonstrate that it trains faster and uses less memory.

----

## [901] Nondeterminism and Instability in Neural Network Optimization

**Authors**: *Cecilia Summers, Michael J. Dinneen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/summers21a.html](http://proceedings.mlr.press/v139/summers21a.html)

**Abstract**:

Nondeterminism in neural network optimization produces uncertainty in performance, making small improvements difficult to discern from run-to-run variability. While uncertainty can be reduced by training multiple model copies, doing so is time-consuming, costly, and harms reproducibility. In this work, we establish an experimental protocol for understanding the effect of optimization nondeterminism on model diversity, allowing us to isolate the effects of a variety of sources of nondeterminism. Surprisingly, we find that all sources of nondeterminism have similar effects on measures of model diversity. To explain this intriguing fact, we identify the instability of model training, taken as an end-to-end procedure, as the key determinant. We show that even one-bit changes in initial parameters result in models converging to vastly different values. Last, we propose two approaches for reducing the effects of instability on run-to-run variability.

----

## [902] AutoSampling: Search for Effective Data Sampling Schedules

**Authors**: *Ming Sun, Haoxuan Dou, Baopu Li, Junjie Yan, Wanli Ouyang, Lei Cui*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sun21a.html](http://proceedings.mlr.press/v139/sun21a.html)

**Abstract**:

Data sampling acts as a pivotal role in training deep learning models. However, an effective sampling schedule is difficult to learn due to its inherent high-dimension as a hyper-parameter. In this paper, we propose an AutoSampling method to automatically learn sampling schedules for model training, which consists of the multi-exploitation step aiming for optimal local sampling schedules and the exploration step for the ideal sampling distribution. More specifically, we achieve sampling schedule search with shortened exploitation cycle to provide enough supervision. In addition, we periodically estimate the sampling distribution from the learned sampling schedules and perturb it to search in the distribution space. The combination of two searches allows us to learn a robust sampling schedule. We apply our AutoSampling method to a variety of image classification tasks illustrating the effectiveness of the proposed method.

----

## [903] What Makes for End-to-End Object Detection?

**Authors**: *Peize Sun, Yi Jiang, Enze Xie, Wenqi Shao, Zehuan Yuan, Changhu Wang, Ping Luo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sun21b.html](http://proceedings.mlr.press/v139/sun21b.html)

**Abstract**:

Object detection has recently achieved a breakthrough for removing the last one non-differentiable component in the pipeline, Non-Maximum Suppression (NMS), and building up an end-to-end system. However, what makes for its one-to-one prediction has not been well understood. In this paper, we first point out that one-to-one positive sample assignment is the key factor, while, one-to-many assignment in previous detectors causes redundant predictions in inference. Second, we surprisingly find that even training with one-to-one assignment, previous detectors still produce redundant predictions. We identify that classification cost in matching cost is the main ingredient: (1) previous detectors only consider location cost, (2) by additionally introducing classification cost, previous detectors immediately produce one-to-one prediction during inference. We introduce the concept of score gap to explore the effect of matching cost. Classification cost enlarges the score gap by choosing positive samples as those of highest score in the training iteration and reducing noisy positive samples brought by only location cost. Finally, we demonstrate the advantages of end-to-end object detection on crowded scenes.

----

## [904] DFAC Framework: Factorizing the Value Function via Quantile Mixture for Multi-Agent Distributional Q-Learning

**Authors**: *Wei-Fang Sun, Cheng-Kuang Lee, Chun-Yi Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sun21c.html](http://proceedings.mlr.press/v139/sun21c.html)

**Abstract**:

In fully cooperative multi-agent reinforcement learning (MARL) settings, the environments are highly stochastic due to the partial observability of each agent and the continuously changing policies of the other agents. To address the above issues, we integrate distributional RL and value function factorization methods by proposing a Distributional Value Function Factorization (DFAC) framework to generalize expected value function factorization methods to their distributional variants. DFAC extends the individual utility functions from deterministic variables to random variables, and models the quantile function of the total return as a quantile mixture. To validate DFAC, we demonstrate DFAC’s ability to factorize a simple two-step matrix game with stochastic rewards and perform experiments on all Super Hard tasks of StarCraft Multi-Agent Challenge, showing that DFAC is able to outperform expected value function factorization baselines.

----

## [905] Scalable Variational Gaussian Processes via Harmonic Kernel Decomposition

**Authors**: *Shengyang Sun, Jiaxin Shi, Andrew Gordon Wilson, Roger B. Grosse*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sun21d.html](http://proceedings.mlr.press/v139/sun21d.html)

**Abstract**:

We introduce a new scalable variational Gaussian process approximation which provides a high fidelity approximation while retaining general applicability. We propose the harmonic kernel decomposition (HKD), which uses Fourier series to decompose a kernel as a sum of orthogonal kernels. Our variational approximation exploits this orthogonality to enable a large number of inducing points at a low computational cost. We demonstrate that, on a range of regression and classification problems, our approach can exploit input space symmetries such as translations and reflections, and it significantly outperforms standard variational methods in scalability and accuracy. Notably, our approach achieves state-of-the-art results on CIFAR-10 among pure GP models.

----

## [906] Reasoning Over Virtual Knowledge Bases With Open Predicate Relations

**Authors**: *Haitian Sun, Patrick Verga, Bhuwan Dhingra, Ruslan Salakhutdinov, William W. Cohen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sun21e.html](http://proceedings.mlr.press/v139/sun21e.html)

**Abstract**:

We present the Open Predicate Query Language (OPQL); a method for constructing a virtual KB (VKB) trained entirely from text. Large Knowledge Bases (KBs) are indispensable for a wide-range of industry applications such as question answering and recommendation. Typically, KBs encode world knowledge in a structured, readily accessible form derived from laborious human annotation efforts. Unfortunately, while they are extremely high precision, KBs are inevitably highly incomplete and automated methods for enriching them are far too inaccurate. Instead, OPQL constructs a VKB by encoding and indexing a set of relation mentions in a way that naturally enables reasoning and can be trained without any structured supervision. We demonstrate that OPQL outperforms prior VKB methods on two different KB reasoning tasks and, additionally, can be used as an external memory integrated into a language model (OPQL-LM) leading to improvements on two open-domain question answering tasks.

----

## [907] PAC-Learning for Strategic Classification

**Authors**: *Ravi Sundaram, Anil Vullikanti, Haifeng Xu, Fan Yao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/sundaram21a.html](http://proceedings.mlr.press/v139/sundaram21a.html)

**Abstract**:

The study of strategic or adversarial manipulation of testing data to fool a classifier has attracted much recent attention. Most previous works have focused on two extreme situations where any testing data point either is completely adversarial or always equally prefers the positive label. In this paper, we generalize both of these through a unified framework for strategic classification and introduce the notion of strategic VC-dimension (SVC) to capture the PAC-learnability in our general strategic setup. SVC provably generalizes the recent concept of adversarial VC-dimension (AVC) introduced by Cullina et al. (2018). We instantiate our framework for the fundamental strategic linear classification problem. We fully characterize: (1) the statistical learnability of linear classifiers by pinning down its SVC; (2) it’s computational tractability by pinning down the complexity of the empirical risk minimization problem. Interestingly, the SVC of linear classifiers is always upper bounded by its standard VC-dimension. This characterization also strictly generalizes the AVC bound for linear classifiers in (Cullina et al., 2018).

----

## [908] Reinforcement Learning for Cost-Aware Markov Decision Processes

**Authors**: *Wesley Suttle, Kaiqing Zhang, Zhuoran Yang, Ji Liu, David N. Kraemer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/suttle21a.html](http://proceedings.mlr.press/v139/suttle21a.html)

**Abstract**:

Ratio maximization has applications in areas as diverse as finance, reward shaping for reinforcement learning (RL), and the development of safe artificial intelligence, yet there has been very little exploration of RL algorithms for ratio maximization. This paper addresses this deficiency by introducing two new, model-free RL algorithms for solving cost-aware Markov decision processes, where the goal is to maximize the ratio of long-run average reward to long-run average cost. The first algorithm is a two-timescale scheme based on relative value iteration (RVI) Q-learning and the second is an actor-critic scheme. The paper proves almost sure convergence of the former to the globally optimal solution in the tabular case and almost sure convergence of the latter under linear function approximation for the critic. Unlike previous methods, the two algorithms provably converge for general reward and cost functions under suitable conditions. The paper also provides empirical results demonstrating promising performance and lending strong support to the theoretical results.

----

## [909] Model-Targeted Poisoning Attacks with Provable Convergence

**Authors**: *Fnu Suya, Saeed Mahloujifar, Anshuman Suri, David Evans, Yuan Tian*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/suya21a.html](http://proceedings.mlr.press/v139/suya21a.html)

**Abstract**:

In a poisoning attack, an adversary who controls a small fraction of the training data attempts to select that data, so a model is induced that misbehaves in a particular way. We consider poisoning attacks against convex machine learning models and propose an efficient poisoning attack designed to induce a model specified by the adversary. Unlike previous model-targeted poisoning attacks, our attack comes with provable convergence to any attainable target model. We also provide a lower bound on the minimum number of poisoning points needed to achieve a given target model. Our method uses online convex optimization and finds poisoning points incrementally. This provides more flexibility than previous attacks which require an a priori assumption about the number of poisoning points. Our attack is the first model-targeted poisoning attack that provides provable convergence for convex models. In our experiments, it either exceeds or matches state-of-the-art attacks in terms of attack success rate and distance to the target model.

----

## [910] Generalization Error Bound for Hyperbolic Ordinal Embedding

**Authors**: *Atsushi Suzuki, Atsushi Nitanda, Jing Wang, Linchuan Xu, Kenji Yamanishi, Marc Cavazza*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/suzuki21a.html](http://proceedings.mlr.press/v139/suzuki21a.html)

**Abstract**:

Hyperbolic ordinal embedding (HOE) represents entities as points in hyperbolic space so that they agree as well as possible with given constraints in the form of entity $i$ is more similar to entity $j$ than to entity $k$. It has been experimentally shown that HOE can obtain representations of hierarchical data such as a knowledge base and a citation network effectively, owing to hyperbolic space’s exponential growth property. However, its theoretical analysis has been limited to ideal noiseless settings, and its generalization error in compensation for hyperbolic space’s exponential representation ability has not been guaranteed. The difficulty is that existing generalization error bound derivations for ordinal embedding based on the Gramian matrix are not applicable in HOE, since hyperbolic space is not inner-product space. In this paper, through our novel characterization of HOE with decomposed Lorentz Gramian matrices, we provide a generalization error bound of HOE for the first time, which is at most exponential with respect to the embedding space’s radius. Our comparison between the bounds of HOE and Euclidean ordinal embedding shows that HOE’s generalization error comes at a reasonable cost considering its exponential representation ability.

----

## [911] Of Moments and Matching: A Game-Theoretic Framework for Closing the Imitation Gap

**Authors**: *Gokul Swamy, Sanjiban Choudhury, J. Andrew Bagnell, Steven Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/swamy21a.html](http://proceedings.mlr.press/v139/swamy21a.html)

**Abstract**:

We provide a unifying view of a large family of previous imitation learning algorithms through the lens of moment matching. At its core, our classification scheme is based on whether the learner attempts to match (1) reward or (2) action-value moments of the expert’s behavior, with each option leading to differing algorithmic approaches. By considering adversarially chosen divergences between learner and expert behavior, we are able to derive bounds on policy performance that apply for all algorithms in each of these classes, the first to our knowledge. We also introduce the notion of moment recoverability, implicit in many previous analyses of imitation learning, which allows us to cleanly delineate how well each algorithmic family is able to mitigate compounding errors. We derive three novel algorithm templates (AdVIL, AdRIL, and DAeQuIL) with strong guarantees, simple implementation, and competitive empirical performance.

----

## [912] Parallel tempering on optimized paths

**Authors**: *Saifuddin Syed, Vittorio Romaniello, Trevor Campbell, Alexandre Bouchard-Côté*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/syed21a.html](http://proceedings.mlr.press/v139/syed21a.html)

**Abstract**:

Parallel tempering (PT) is a class of Markov chain Monte Carlo algorithms that constructs a path of distributions annealing between a tractable reference and an intractable target, and then interchanges states along the path to improve mixing in the target. The performance of PT depends on how quickly a sample from the reference distribution makes its way to the target, which in turn depends on the particular path of annealing distributions. However, past work on PT has used only simple paths constructed from convex combinations of the reference and target log-densities. This paper begins by demonstrating that this path performs poorly in the setting where the reference and target are nearly mutually singular. To address this issue, we expand the framework of PT to general families of paths, formulate the choice of path as an optimization problem that admits tractable gradient estimates, and propose a flexible new family of spline interpolation paths for use in practice. Theoretical and empirical results both demonstrate that our proposed methodology breaks previously-established upper performance limits for traditional paths.

----

## [913] Robust Representation Learning via Perceptual Similarity Metrics

**Authors**: *Saeid Asgari Taghanaki, Kristy Choi, Amir Hosein Khasahmadi, Anirudh Goyal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/taghanaki21a.html](http://proceedings.mlr.press/v139/taghanaki21a.html)

**Abstract**:

A fundamental challenge in artificial intelligence is learning useful representations of data that yield good performance on a downstream classification task, without overfitting to spurious input features. Extracting such task-relevant predictive information becomes particularly difficult for noisy and high-dimensional real-world data. In this work, we propose Contrastive Input Morphing (CIM), a representation learning framework that learns input-space transformations of the data to mitigate the effect of irrelevant input features on downstream performance. Our method leverages a perceptual similarity metric via a triplet loss to ensure that the transformation preserves task-relevant information. Empirically, we demonstrate the efficacy of our approach on various tasks which typically suffer from the presence of spurious correlations: classification with nuisance information, out-of-distribution generalization, and preservation of subgroup accuracies. We additionally show that CIM is complementary to other mutual information-based representation learning techniques, and demonstrate that it improves the performance of variational information bottleneck (VIB) when used in conjunction.

----

## [914] DriftSurf: Stable-State / Reactive-State Learning under Concept Drift

**Authors**: *Ashraf Tahmasbi, Ellango Jothimurugesan, Srikanta Tirthapura, Phillip B. Gibbons*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tahmasbi21a.html](http://proceedings.mlr.press/v139/tahmasbi21a.html)

**Abstract**:

When learning from streaming data, a change in the data distribution, also known as concept drift, can render a previously-learned model inaccurate and require training a new model. We present an adaptive learning algorithm that extends previous drift-detection-based methods by incorporating drift detection into a broader stable-state/reactive-state process. The advantage of our approach is that we can use aggressive drift detection in the stable state to achieve a high detection rate, but mitigate the false positive rate of standalone drift detection via a reactive state that reacts quickly to true drifts while eliminating most false positives. The algorithm is generic in its base learner and can be applied across a variety of supervised learning problems. Our theoretical analysis shows that the risk of the algorithm is (i) statistically better than standalone drift detection and (ii) competitive to an algorithm with oracle knowledge of when (abrupt) drifts occur. Experiments on synthetic and real datasets with concept drifts confirm our theoretical analysis.

----

## [915] Sinkhorn Label Allocation: Semi-Supervised Classification via Annealed Self-Training

**Authors**: *Kai Sheng Tai, Peter Bailis, Gregory Valiant*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tai21a.html](http://proceedings.mlr.press/v139/tai21a.html)

**Abstract**:

Self-training is a standard approach to semi-supervised learning where the learner’s own predictions on unlabeled data are used as supervision during training. In this paper, we reinterpret this label assignment process as an optimal transportation problem between examples and classes, wherein the cost of assigning an example to a class is mediated by the current predictions of the classifier. This formulation facilitates a practical annealing strategy for label assignment and allows for the inclusion of prior knowledge on class proportions via flexible upper bound constraints. The solutions to these assignment problems can be efficiently approximated using Sinkhorn iteration, thus enabling their use in the inner loop of standard stochastic optimization algorithms. We demonstrate the effectiveness of our algorithm on the CIFAR-10, CIFAR-100, and SVHN datasets in comparison with FixMatch, a state-of-the-art self-training algorithm.

----

## [916] Approximation Theory Based Methods for RKHS Bandits

**Authors**: *Sho Takemori, Masahiro Sato*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/takemori21a.html](http://proceedings.mlr.press/v139/takemori21a.html)

**Abstract**:

The RKHS bandit problem (also called kernelized multi-armed bandit problem) is an online optimization problem of non-linear functions with noisy feedback. Although the problem has been extensively studied, there are unsatisfactory results for some problems compared to the well-studied linear bandit case. Specifically, there is no general algorithm for the adversarial RKHS bandit problem. In addition, high computational complexity of existing algorithms hinders practical application. We address these issues by considering a novel amalgamation of approximation theory and the misspecified linear bandit problem. Using an approximation method, we propose efficient algorithms for the stochastic RKHS bandit problem and the first general algorithm for the adversarial RKHS bandit problem. Furthermore, we empirically show that one of our proposed methods has comparable cumulative regret to IGP-UCB and its running time is much shorter.

----

## [917] Supervised Tree-Wasserstein Distance

**Authors**: *Yuki Takezawa, Ryoma Sato, Makoto Yamada*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/takezawa21a.html](http://proceedings.mlr.press/v139/takezawa21a.html)

**Abstract**:

To measure the similarity of documents, the Wasserstein distance is a powerful tool, but it requires a high computational cost. Recently, for fast computation of the Wasserstein distance, methods for approximating the Wasserstein distance using a tree metric have been proposed. These tree-based methods allow fast comparisons of a large number of documents; however, they are unsupervised and do not learn task-specific distances. In this work, we propose the Supervised Tree-Wasserstein (STW) distance, a fast, supervised metric learning method based on the tree metric. Specifically, we rewrite the Wasserstein distance on the tree metric by the parent-child relationships of a tree, and formulate it as a continuous optimization problem using a contrastive loss. Experimentally, we show that the STW distance can be computed fast, and improves the accuracy of document classification tasks. Furthermore, the STW distance is formulated by matrix multiplications, runs on a GPU, and is suitable for batch processing. Therefore, we show that the STW distance is extremely efficient when comparing a large number of documents.

----

## [918] EfficientNetV2: Smaller Models and Faster Training

**Authors**: *Mingxing Tan, Quoc V. Le*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tan21a.html](http://proceedings.mlr.press/v139/tan21a.html)

**Abstract**:

This paper introduces EfficientNetV2, a new family of convolutional networks that have faster training speed and better parameter efficiency than previous models. To develop these models, we use a combination of training-aware neural architecture search and scaling, to jointly optimize training speed and parameter efficiency. The models were searched from the search space enriched with new ops such as Fused-MBConv. Our experiments show that EfficientNetV2 models train much faster than state-of-the-art models while being up to 6.8x smaller. Our training can be further sped up by progressively increasing the image size during training, but it often causes a drop in accuracy. To compensate for this accuracy drop, we propose an improved method of progressive learning, which adaptively adjusts regularization (e.g. data augmentation) along with image size. With progressive learning, our EfficientNetV2 significantly outperforms previous models on ImageNet and CIFAR/Cars/Flowers datasets. By pretraining on the same ImageNet21k, our EfficientNetV2 achieves 87.3% top-1 accuracy on ImageNet ILSVRC2012, outperforming the recent ViT by 2.0% accuracy while training 5x-11x faster using the same computing resources.

----

## [919] SGA: A Robust Algorithm for Partial Recovery of Tree-Structured Graphical Models with Noisy Samples

**Authors**: *Anshoo Tandon, Aldric H. J. Han, Vincent Y. F. Tan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tandon21a.html](http://proceedings.mlr.press/v139/tandon21a.html)

**Abstract**:

We consider learning Ising tree models when the observations from the nodes are corrupted by independent but non-identically distributed noise with unknown statistics. Katiyar et al. (2020) showed that although the exact tree structure cannot be recovered, one can recover a partial tree structure; that is, a structure belonging to the equivalence class containing the true tree. This paper presents a systematic improvement of Katiyar et al. (2020). First, we present a novel impossibility result by deriving a bound on the necessary number of samples for partial recovery. Second, we derive a significantly improved sample complexity result in which the dependence on the minimum correlation $\rho_{\min}$ is $\rho_{\min}^{-8}$ instead of $\rho_{\min}^{-24}$. Finally, we propose Symmetrized Geometric Averaging (SGA), a more statistically robust algorithm for partial tree recovery. We provide error exponent analyses and extensive numerical results on a variety of trees to show that the sample complexity of SGA is significantly better than the algorithm of Katiyar et al. (2020). SGA can be readily extended to Gaussian models and is shown via numerical experiments to be similarly superior.

----

## [920] 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed

**Authors**: *Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, Samyam Rajbhandari, Conglong Li, Xiangru Lian, Ji Liu, Ce Zhang, Yuxiong He*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tang21a.html](http://proceedings.mlr.press/v139/tang21a.html)

**Abstract**:

Scalable training of large models (like BERT and GPT-3) requires careful optimization rooted in model design, architecture, and system capabilities. From a system standpoint, communication has become a major bottleneck, especially on commodity systems with standard TCP interconnects that offer limited network bandwidth. Communication compression is an important technique to reduce training time on such systems. One of the most effective ways to compress communication is via error compensation compression, which offers robust convergence speed, even under 1-bit compression. However, state-of-the-art error compensation techniques only work with basic optimizers like SGD and momentum SGD, which are linearly dependent on the gradients. They do not work with non-linear gradient-based optimizers like Adam, which offer state-of-the-art convergence efficiency and accuracy for models like BERT. In this paper, we propose 1-bit Adam that reduces the communication volume by up to 5x, offers much better scalability, and provides the same convergence speed as uncompressed Adam. Our key finding is that Adam’s variance becomes stable (after a warmup phase) and can be used as a fixed precondition for the rest of the training (compression phase). We performed experiments on up to 256 GPUs and show that 1-bit Adam enables up to 3.3x higher throughput for BERT-Large pre-training and up to 2.9x higher throughput for SQuAD fine-tuning. In addition, we provide theoretical analysis for 1-bit Adam.

----

## [921] Taylor Expansion of Discount Factors

**Authors**: *Yunhao Tang, Mark Rowland, Rémi Munos, Michal Valko*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tang21b.html](http://proceedings.mlr.press/v139/tang21b.html)

**Abstract**:

In practical reinforcement learning (RL), the discount factor used for estimating value functions often differs from that used for defining the evaluation objective. In this work, we study the effect that this discrepancy of discount factors has during learning, and discover a family of objectives that interpolate value functions of two distinct discount factors. Our analysis suggests new ways for estimating value functions and performing policy optimization updates, which demonstrate empirical performance gains. This framework also leads to new insights on commonly-used deep RL heuristic modifications to policy optimization algorithms.

----

## [922] REPAINT: Knowledge Transfer in Deep Reinforcement Learning

**Authors**: *Yunzhe Tao, Sahika Genc, Jonathan Chung, Tao Sun, Sunil Mallya*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tao21a.html](http://proceedings.mlr.press/v139/tao21a.html)

**Abstract**:

Accelerating learning processes for complex tasks by leveraging previously learned tasks has been one of the most challenging problems in reinforcement learning, especially when the similarity between source and target tasks is low. This work proposes REPresentation And INstance Transfer (REPAINT) algorithm for knowledge transfer in deep reinforcement learning. REPAINT not only transfers the representation of a pre-trained teacher policy in the on-policy learning, but also uses an advantage-based experience selection approach to transfer useful samples collected following the teacher policy in the off-policy learning. Our experimental results on several benchmark tasks show that REPAINT significantly reduces the total training time in generic cases of task similarity. In particular, when the source tasks are dissimilar to, or sub-tasks of, the target tasks, REPAINT outperforms other baselines in both training-time reduction and asymptotic performance of return scores.

----

## [923] Understanding the Dynamics of Gradient Flow in Overparameterized Linear models

**Authors**: *Salma Tarmoun, Guilherme França, Benjamin D. Haeffele, René Vidal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tarmoun21a.html](http://proceedings.mlr.press/v139/tarmoun21a.html)

**Abstract**:

We provide a detailed analysis of the dynamics ofthe gradient flow in overparameterized two-layerlinear models. A particularly interesting featureof this model is that its nonlinear dynamics can beexactly solved as a consequence of a large num-ber of conservation laws that constrain the systemto follow particular trajectories. More precisely,the gradient flow preserves the difference of theGramian matrices of the input and output weights,and its convergence to equilibrium depends onboth the magnitude of that difference (which isfixed at initialization) and the spectrum of the data.In addition, and generalizing prior work, we proveour results without assuming small, balanced orspectral initialization for the weights. Moreover,we establish interesting mathematical connectionsbetween matrix factorization problems and differ-ential equations of the Riccati type.

----

## [924] Sequential Domain Adaptation by Synthesizing Distributionally Robust Experts

**Authors**: *Bahar Taskesen, Man-Chung Yue, Jose H. Blanchet, Daniel Kuhn, Viet Anh Nguyen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/taskesen21a.html](http://proceedings.mlr.press/v139/taskesen21a.html)

**Abstract**:

Least squares estimators, when trained on few target domain samples, may predict poorly. Supervised domain adaptation aims to improve the predictive accuracy by exploiting additional labeled training samples from a source distribution that is close to the target distribution. Given available data, we investigate novel strategies to synthesize a family of least squares estimator experts that are robust with regard to moment conditions. When these moment conditions are specified using Kullback-Leibler or Wasserstein-type divergences, we can find the robust estimators efficiently using convex optimization. We use the Bernstein online aggregation algorithm on the proposed family of robust experts to generate predictions for the sequential stream of target test samples. Numerical experiments on real data show that the robust strategies systematically outperform non-robust interpolations of the empirical least squares estimators.

----

## [925] A Language for Counterfactual Generative Models

**Authors**: *Zenna Tavares, James Koppel, Xin Zhang, Ria Das, Armando Solar-Lezama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tavares21a.html](http://proceedings.mlr.press/v139/tavares21a.html)

**Abstract**:

We present Omega, a probabilistic programming language with support for counterfactual inference. Counterfactual inference means to observe some fact in the present, and infer what would have happened had some past intervention been taken, e.g. “given that medication was not effective at dose x, what is the probability that it would have been effective at dose 2x?.” We accomplish this by introducing a new operator to probabilistic programming akin to Pearl’s do, define its formal semantics, provide an implementation, and demonstrate its utility through examples in a variety of simulation models.

----

## [926] Synthesizer: Rethinking Self-Attention for Transformer Models

**Authors**: *Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, Che Zheng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tay21a.html](http://proceedings.mlr.press/v139/tay21a.html)

**Abstract**:

The dot product self-attention is known to be central and indispensable to state-of-the-art Transformer models. But is it really required? This paper investigates the true importance and contribution of the dot product-based self-attention mechanism on the performance of Transformer models. Via extensive experiments, we find that (1) random alignment matrices surprisingly perform quite competitively and (2) learning attention weights from token-token (query-key) interactions is useful but not that important after all. To this end, we propose \textsc{Synthesizer}, a model that learns synthetic attention weights without token-token interactions. In our experiments, we first show that simple Synthesizers achieve highly competitive performance when compared against vanilla Transformer models across a range of tasks, including machine translation, language modeling, text generation and GLUE/SuperGLUE benchmarks. When composed with dot product attention, we find that Synthesizers consistently outperform Transformers. Moreover, we conduct additional comparisons of Synthesizers against Dynamic Convolutions, showing that simple Random Synthesizer is not only $60%$ faster but also improves perplexity by a relative $3.5%$. Finally, we show that simple factorized Synthesizers can outperform Linformers on encoding only tasks.

----

## [927] OmniNet: Omnidirectional Representations from Transformers

**Authors**: *Yi Tay, Mostafa Dehghani, Vamsi Aribandi, Jai Prakash Gupta, Philip Pham, Zhen Qin, Dara Bahri, Da-Cheng Juan, Donald Metzler*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tay21b.html](http://proceedings.mlr.press/v139/tay21b.html)

**Abstract**:

This paper proposes Omnidirectional Representations from Transformers (OMNINET). In OmniNet, instead of maintaining a strictly horizon-tal receptive field, each token is allowed to attend to all tokens in the entire network. This process can also be interpreted as a form of extreme or intensive attention mechanism that has the receptive field of the entire width and depth of the network. To this end, the omnidirectional attention is learned via a meta-learner, which is essentially another self-attention based model. In order to mitigate the computationally expensive costs of full receptive field attention, we leverage efficient self-attention models such as kernel-based, low-rank attention and/or Big Bird as the meta-learner. Extensive experiments are conducted on autoregressive language modeling(LM1B, C4), Machine Translation, Long Range Arena (LRA), and Image Recognition.The experiments show that OmniNet achieves considerable improvements across these tasks, including achieving state-of-the-art performance on LM1B,WMT’14 En-De/En-Fr, and Long Range Arena.Moreover, using omnidirectional representation in Vision Transformers leads to significant improvements on image recognition tasks on both few-shot learning and fine-tuning setups.

----

## [928] T-SCI: A Two-Stage Conformal Inference Algorithm with Guaranteed Coverage for Cox-MLP

**Authors**: *Jiaye Teng, Zeren Tan, Yang Yuan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/teng21a.html](http://proceedings.mlr.press/v139/teng21a.html)

**Abstract**:

It is challenging to deal with censored data, where we only have access to the incomplete information of survival time instead of its exact value. Fortunately, under linear predictor assumption, people can obtain guaranteed coverage for the confidence interval of survival time using methods like Cox Regression. However, when relaxing the linear assumption with neural networks (e.g., Cox-MLP \citep{katzman2018deepsurv,kvamme2019time}), we lose the guaranteed coverage. To recover the guaranteed coverage without linear assumption, we propose two algorithms based on conformal inference. In the first algorithm \emph{WCCI}, we revisit weighted conformal inference and introduce a new non-conformity score based on partial likelihood. We then propose a two-stage algorithm \emph{T-SCI}, where we run WCCI in the first stage and apply quantile conformal inference to calibrate the results in the second stage. Theoretical analysis shows that T-SCI returns guaranteed coverage under milder assumptions than WCCI. We conduct extensive experiments on synthetic data and real data using different methods, which validate our analysis.

----

## [929] Moreau-Yosida f-divergences

**Authors**: *Dávid Terjék*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/terjek21a.html](http://proceedings.mlr.press/v139/terjek21a.html)

**Abstract**:

Variational representations of $f$-divergences are central to many machine learning algorithms, with Lipschitz constrained variants recently gaining attention. Inspired by this, we define the Moreau-Yosida approximation of $f$-divergences with respect to the Wasserstein-$1$ metric. The corresponding variational formulas provide a generalization of a number of recent results, novel special cases of interest and a relaxation of the hard Lipschitz constraint. Additionally, we prove that the so-called tight variational representation of $f$-divergences can be to be taken over the quotient space of Lipschitz functions, and give a characterization of functions achieving the supremum in the variational representation. On the practical side, we propose an algorithm to calculate the tight convex conjugate of $f$-divergences compatible with automatic differentiation frameworks. As an application of our results, we propose the Moreau-Yosida $f$-GAN, providing an implementation of the variational formulas for the Kullback-Leibler, reverse Kullback-Leibler, $\chi^2$, reverse $\chi^2$, squared Hellinger, Jensen-Shannon, Jeffreys, triangular discrimination and total variation divergences as GANs trained on CIFAR-10, leading to competitive results and a simple solution to the problem of uniqueness of the optimal critic.

----

## [930] Understanding Invariance via Feedforward Inversion of Discriminatively Trained Classifiers

**Authors**: *Piotr Teterwak, Chiyuan Zhang, Dilip Krishnan, Michael C. Mozer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/teterwak21a.html](http://proceedings.mlr.press/v139/teterwak21a.html)

**Abstract**:

A discriminatively trained neural net classifier can fit the training data perfectly if all information about its input other than class membership has been discarded prior to the output layer. Surprisingly, past research has discovered that some extraneous visual detail remains in the unnormalized logits. This finding is based on inversion techniques that map deep embeddings back to images. We explore this phenomenon further using a novel synthesis of methods, yielding a feedforward inversion model that produces remarkably high fidelity reconstructions, qualitatively superior to those of past efforts. When applied to an adversarially robust classifier model, the reconstructions contain sufficient local detail and global structure that they might be confused with the original image in a quick glance, and the object category can clearly be gleaned from the reconstruction. Our approach is based on BigGAN (Brock, 2019), with conditioning on logits instead of one-hot class labels. We use our reconstruction model as a tool for exploring the nature of representations, including: the influence of model architecture and training objectives (specifically robust losses), the forms of invariance that networks achieve, representational differences between correctly and incorrectly classified images, and the effects of manipulating logits and images. We believe that our method can inspire future investigations into the nature of information flow in a neural net and can provide diagnostics for improving discriminative models. We provide pre-trained models and visualizations at \url{https://sites.google.com/view/understanding-invariance/home}.

----

## [931] Resource Allocation in Multi-armed Bandit Exploration: Overcoming Sublinear Scaling with Adaptive Parallelism

**Authors**: *Brijen Thananjeyan, Kirthevasan Kandasamy, Ion Stoica, Michael I. Jordan, Ken Goldberg, Joseph Gonzalez*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/thananjeyan21a.html](http://proceedings.mlr.press/v139/thananjeyan21a.html)

**Abstract**:

We study exploration in stochastic multi-armed bandits when we have access to a divisible resource that can be allocated in varying amounts to arm pulls. We focus in particular on the allocation of distributed computing resources, where we may obtain results faster by allocating more resources per pull, but might have reduced throughput due to nonlinear scaling. For example, in simulation-based scientific studies, an expensive simulation can be sped up by running it on multiple cores. This speed-up however, is partly offset by the communication among cores, which results in lower throughput than if fewer cores were allocated to run more trials in parallel. In this paper, we explore these trade-offs in two settings. First, in a fixed confidence setting, we need to find the best arm with a given target success probability as quickly as possible. We propose an algorithm which trades off between information accumulation and throughput and show that the time taken can be upper bounded by the solution of a dynamic program whose inputs are the gaps between the sub-optimal and optimal arms. We also prove a matching hardness result. Second, we present an algorithm for a fixed deadline setting, where we are given a time deadline and need to maximize the probability of finding the best arm. We corroborate our theoretical insights with simulation experiments that show that the algorithms consistently match or outperform baseline algorithms on a variety of problem instances.

----

## [932] Monte Carlo Variational Auto-Encoders

**Authors**: *Achille Thin, Nikita Kotelevskii, Arnaud Doucet, Alain Durmus, Eric Moulines, Maxim Panov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/thin21a.html](http://proceedings.mlr.press/v139/thin21a.html)

**Abstract**:

Variational auto-encoders (VAE) are popular deep latent variable models which are trained by maximizing an Evidence Lower Bound (ELBO). To obtain tighter ELBO and hence better variational approximations, it has been proposed to use importance sampling to get a lower variance estimate of the evidence. However, importance sampling is known to perform poorly in high dimensions. While it has been suggested many times in the literature to use more sophisticated algorithms such as Annealed Importance Sampling (AIS) and its Sequential Importance Sampling (SIS) extensions, the potential benefits brought by these advanced techniques have never been realized for VAE: the AIS estimate cannot be easily differentiated, while SIS requires the specification of carefully chosen backward Markov kernels. In this paper, we address both issues and demonstrate the performance of the resulting Monte Carlo VAEs on a variety of applications.

----

## [933] Efficient Generative Modelling of Protein Structure Fragments using a Deep Markov Model

**Authors**: *Christian B. Thygesen, Christian Skjødt Steenmans, Ahmad Salim Al-Sibahi, Lys Sanz Moreta, Anders Bundgård Sørensen, Thomas Hamelryck*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/thygesen21a.html](http://proceedings.mlr.press/v139/thygesen21a.html)

**Abstract**:

Fragment libraries are often used in protein structure prediction, simulation and design as a means to significantly reduce the vast conformational search space. Current state-of-the-art methods for fragment library generation do not properly account for aleatory and epistemic uncertainty, respectively due to the dynamic nature of proteins and experimental errors in protein structures. Additionally, they typically rely on information that is not generally or readily available, such as homologous sequences, related protein structures and other complementary information. To address these issues, we developed BIFROST, a novel take on the fragment library problem based on a Deep Markov Model architecture combined with directional statistics for angular degrees of freedom, implemented in the deep probabilistic programming language Pyro. BIFROST is a probabilistic, generative model of the protein backbone dihedral angles conditioned solely on the amino acid sequence. BIFROST generates fragment libraries with a quality on par with current state-of-the-art methods at a fraction of the run-time, while requiring considerably less information and allowing efficient evaluation of probabilities.

----

## [934] Understanding self-supervised learning dynamics without contrastive pairs

**Authors**: *Yuandong Tian, Xinlei Chen, Surya Ganguli*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tian21a.html](http://proceedings.mlr.press/v139/tian21a.html)

**Abstract**:

While contrastive approaches of self-supervised learning (SSL) learn representations by minimizing the distance between two augmented views of the same data point (positive pairs) and maximizing views from different data points (negative pairs), recent \emph{non-contrastive} SSL (e.g., BYOL and SimSiam) show remarkable performance {\it without} negative pairs, with an extra learnable predictor and a stop-gradient operation. A fundamental question rises: why they do not collapse into trivial representation? In this paper, we answer this question via a simple theoretical study and propose a novel approach, \ourmethod{}, that \emph{directly} sets the linear predictor based on the statistics of its inputs, rather than trained with gradient update. On ImageNet, it performs comparably with more complex two-layer non-linear predictors that employ BatchNorm and outperforms linear predictor by $2.5%$ in 300-epoch training (and $5%$ in 60-epoch). \ourmethod{} is motivated by our theoretical study of the nonlinear learning dynamics of non-contrastive SSL in simple linear networks. Our study yields conceptual insights into how non-contrastive SSL methods learn, how they avoid representational collapse, and how multiple factors, like predictor networks, stop-gradients, exponential moving averages, and weight decay all come into play. Our simple theory recapitulates the results of real-world ablation studies in both STL-10 and ImageNet. Code is released\footnote{\url{https://github.com/facebookresearch/luckmatters/tree/master/ssl}}.

----

## [935] Online Learning in Unknown Markov Games

**Authors**: *Yi Tian, Yuanhao Wang, Tiancheng Yu, Suvrit Sra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tian21b.html](http://proceedings.mlr.press/v139/tian21b.html)

**Abstract**:

We study online learning in unknown Markov games, a problem that arises in episodic multi-agent reinforcement learning where the actions of the opponents are unobservable. We show that in this challenging setting, achieving sublinear regret against the best response in hindsight is statistically hard. We then consider a weaker notion of regret by competing with the \emph{minimax value} of the game, and present an algorithm that achieves a sublinear $\tilde{\mathcal{O}}(K^{2/3})$ regret after $K$ episodes. This is the first sublinear regret bound (to our knowledge) for online learning in unknown Markov games. Importantly, our regret bound is independent of the size of the opponents’ action spaces. As a result, even when the opponents’ actions are fully observable, our regret bound improves upon existing analysis (e.g., (Xie et al., 2020)) by an exponential factor in the number of opponents.

----

## [936] BORE: Bayesian Optimization by Density-Ratio Estimation

**Authors**: *Louis C. Tiao, Aaron Klein, Matthias W. Seeger, Edwin V. Bonilla, Cédric Archambeau, Fabio Ramos*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tiao21a.html](http://proceedings.mlr.press/v139/tiao21a.html)

**Abstract**:

Bayesian optimization (BO) is among the most effective and widely-used blackbox optimization methods. BO proposes solutions according to an explore-exploit trade-off criterion encoded in an acquisition function, many of which are computed from the posterior predictive of a probabilistic surrogate model. Prevalent among these is the expected improvement (EI). The need to ensure analytical tractability of the predictive often poses limitations that can hinder the efficiency and applicability of BO. In this paper, we cast the computation of EI as a binary classification problem, building on the link between class-probability estimation and density-ratio estimation, and the lesser-known link between density-ratios and EI. By circumventing the tractability constraints, this reformulation provides numerous advantages, not least in terms of expressiveness, versatility, and scalability.

----

## [937] Nonparametric Decomposition of Sparse Tensors

**Authors**: *Conor Tillinghast, Shandian Zhe*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tillinghast21a.html](http://proceedings.mlr.press/v139/tillinghast21a.html)

**Abstract**:

Tensor decomposition is a powerful framework for multiway data analysis. Despite the success of existing approaches, they ignore the sparse nature of the tensor data in many real-world applications, explicitly or implicitly assuming dense tensors. To address this model misspecification and to exploit the sparse tensor structures, we propose Nonparametric dEcomposition of Sparse Tensors (\ours), which can capture both the sparse structure properties and complex relationships between the tensor nodes to enhance the embedding estimation. Specifically, we first use completely random measures to construct tensor-valued random processes. We prove that the entry growth is much slower than that of the corresponding tensor size, which implies sparsity. Given finite observations (\ie projections), we then propose two nonparametric decomposition models that couple Dirichlet processes and Gaussian processes to jointly sample the sparse entry indices and the entry values (the latter as a nonlinear mapping of the embeddings), so as to encode both the structure properties and nonlinear relationships of the tensor nodes into the embeddings. Finally, we use the stick-breaking construction and random Fourier features to develop a scalable, stochastic variational learning algorithm. We show the advantage of our approach in sparse tensor generation, and entry index and value prediction in several real-world applications.

----

## [938] Probabilistic Programs with Stochastic Conditioning

**Authors**: *David Tolpin, Yuan Zhou, Tom Rainforth, Hongseok Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tolpin21a.html](http://proceedings.mlr.press/v139/tolpin21a.html)

**Abstract**:

We tackle the problem of conditioning probabilistic programs on distributions of observable variables. Probabilistic programs are usually conditioned on samples from the joint data distribution, which we refer to as deterministic conditioning. However, in many real-life scenarios, the observations are given as marginal distributions, summary statistics, or samplers. Conventional probabilistic programming systems lack adequate means for modeling and inference in such scenarios. We propose a generalization of deterministic conditioning to stochastic conditioning, that is, conditioning on the marginal distribution of a variable taking a particular form. To this end, we first define the formal notion of stochastic conditioning and discuss its key properties. We then show how to perform inference in the presence of stochastic conditioning. We demonstrate potential usage of stochastic conditioning on several case studies which involve various kinds of stochastic conditioning and are difficult to solve otherwise. Although we present stochastic conditioning in the context of probabilistic programming, our formalization is general and applicable to other settings.

----

## [939] Deep Continuous Networks

**Authors**: *Nergis Tomen, Silvia-Laura Pintea, Jan van Gemert*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tomen21a.html](http://proceedings.mlr.press/v139/tomen21a.html)

**Abstract**:

CNNs and computational models of biological vision share some fundamental principles, which opened new avenues of research. However, fruitful cross-field research is hampered by conventional CNN architectures being based on spatially and depthwise discrete representations, which cannot accommodate certain aspects of biological complexity such as continuously varying receptive field sizes and dynamics of neuronal responses. Here we propose deep continuous networks (DCNs), which combine spatially continuous filters, with the continuous depth framework of neural ODEs. This allows us to learn the spatial support of the filters during training, as well as model the continuous evolution of feature maps, linking DCNs closely to biological models. We show that DCNs are versatile and highly applicable to standard image classification and reconstruction problems, where they improve parameter and data efficiency, and allow for meta-parametrization. We illustrate the biological plausibility of the scale distributions learned by DCNs and explore their performance in a neuroscientifically inspired pattern completion task. Finally, we investigate an efficient implementation of DCNs by changing input contrast.

----

## [940] Diffusion Earth Mover's Distance and Distribution Embeddings

**Authors**: *Alexander Tong, Guillaume Huguet, Amine Natik, Kincaid MacDonald, Manik Kuchroo, Ronald R. Coifman, Guy Wolf, Smita Krishnaswamy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tong21a.html](http://proceedings.mlr.press/v139/tong21a.html)

**Abstract**:

We propose a new fast method of measuring distances between large numbers of related high dimensional datasets called the Diffusion Earth Mover’s Distance (EMD). We model the datasets as distributions supported on common data graph that is derived from the affinity matrix computed on the combined data. In such cases where the graph is a discretization of an underlying Riemannian closed manifold, we prove that Diffusion EMD is topologically equivalent to the standard EMD with a geodesic ground distance. Diffusion EMD can be computed in {Õ}(n) time and is more accurate than similarly fast algorithms such as tree-based EMDs. We also show Diffusion EMD is fully differentiable, making it amenable to future uses in gradient-descent frameworks such as deep neural networks. Finally, we demonstrate an application of Diffusion EMD to single cell data collected from 210 COVID-19 patient samples at Yale New Haven Hospital. Here, Diffusion EMD can derive distances between patients on the manifold of cells at least two orders of magnitude faster than equally accurate methods. This distance matrix between patients can be embedded into a higher level patient manifold which uncovers structure and heterogeneity in patients. More generally, Diffusion EMD is applicable to all datasets that are massively collected in parallel in many medical and biological systems.

----

## [941] Training data-efficient image transformers & distillation through attention

**Authors**: *Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/touvron21a.html](http://proceedings.mlr.press/v139/touvron21a.html)

**Abstract**:

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. These high-performing vision transformers are pre-trained with hundreds of millions of images using a large infrastructure, thereby limiting their adoption. In this work, we produce competitive convolution-free transformers trained on ImageNet only using a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop) on ImageNet with no external data. We also introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention, typically from a convnet teacher. The learned transformers are competitive (85.2% top-1 acc.) with the state of the art on ImageNet, and similarly when transferred to other tasks. We will share our code and models.

----

## [942] Conservative Objective Models for Effective Offline Model-Based Optimization

**Authors**: *Brandon Trabucco, Aviral Kumar, Xinyang Geng, Sergey Levine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/trabucco21a.html](http://proceedings.mlr.press/v139/trabucco21a.html)

**Abstract**:

In this paper, we aim to solve data-driven model-based optimization (MBO) problems, where the goal is to find a design input that maximizes an unknown objective function provided access to only a static dataset of inputs and their corresponding objective values. Such data-driven optimization procedures are the only practical methods in many real-world domains where active data collection is expensive (e.g., when optimizing over proteins) or dangerous (e.g., when optimizing over aircraft designs, actively evaluating malformed aircraft designs is unsafe). Typical methods for MBO that optimize the input against a learned model of the unknown score function are affected by erroneous overestimation in the learned model caused due to distributional shift, that drives the optimizer to low-scoring or invalid inputs. To overcome this, we propose conservative objective models (COMs), a method that learns a model of the objective function which lower bounds the actual value of the ground-truth objective on out-of-distribution inputs and uses it for optimization. In practice, COMs outperform a number existing methods on a wide range of MBO problems, including optimizing controller parameters, robot morphologies, and superconducting materials.

----

## [943] Sparse within Sparse Gaussian Processes using Neighbor Information

**Authors**: *Gia-Lac Tran, Dimitrios Milios, Pietro Michiardi, Maurizio Filippone*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tran21a.html](http://proceedings.mlr.press/v139/tran21a.html)

**Abstract**:

Approximations to Gaussian processes (GPs) based on inducing variables, combined with variational inference techniques, enable state-of-the-art sparse approaches to infer GPs at scale through mini-batch based learning. In this work, we further push the limits of scalability of sparse GPs by allowing large number of inducing variables without imposing a special structure on the inducing inputs. In particular, we introduce a novel hierarchical prior, which imposes sparsity on the set of inducing variables. We treat our model variationally, and we experimentally show considerable computational gains compared to standard sparse GPs when sparsity on the inducing variables is realized considering the nearest inducing inputs of a random mini-batch of the data. We perform an extensive experimental validation that demonstrates the effectiveness of our approach compared to the state-of-the-art. Our approach enables the possibility to use sparse GPs using a large number of inducing points without incurring a prohibitive computational cost.

----

## [944] SMG: A Shuffling Gradient-Based Method with Momentum

**Authors**: *Trang H. Tran, Lam M. Nguyen, Quoc Tran-Dinh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tran21b.html](http://proceedings.mlr.press/v139/tran21b.html)

**Abstract**:

We combine two advanced ideas widely used in optimization for machine learning: \textit{shuffling} strategy and \textit{momentum} technique to develop a novel shuffling gradient-based method with momentum, coined \textbf{S}huffling \textbf{M}omentum \textbf{G}radient (SMG), for non-convex finite-sum optimization problems. While our method is inspired by momentum techniques, its update is fundamentally different from existing momentum-based methods. We establish state-of-the-art convergence rates of SMG for any shuffling strategy using either constant or diminishing learning rate under standard assumptions (i.e. \textit{$L$-smoothness} and \textit{bounded variance}). When the shuffling strategy is fixed, we develop another new algorithm that is similar to existing momentum methods, and prove the same convergence rates for this algorithm under the $L$-smoothness and bounded gradient assumptions. We demonstrate our algorithms via numerical simulations on standard datasets and compare them with existing shuffling methods. Our tests have shown encouraging performance of the new algorithms.

----

## [945] Bayesian Optimistic Optimisation with Exponentially Decaying Regret

**Authors**: *Hung Tran-The, Sunil Gupta, Santu Rana, Svetha Venkatesh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tran-the21a.html](http://proceedings.mlr.press/v139/tran-the21a.html)

**Abstract**:

Bayesian optimisation (BO) is a well known algorithm for finding the global optimum of expensive, black-box functions. The current practical BO algorithms have regret bounds ranging from $\mathcal{O}(\frac{logN}{\sqrt{N}})$ to $\mathcal O(e^{-\sqrt{N}})$, where $N$ is the number of evaluations. This paper explores the possibility of improving the regret bound in the noise-free setting by intertwining concepts from BO and optimistic optimisation methods which are based on partitioning the search space. We propose the BOO algorithm, a first practical approach which can achieve an exponential regret bound with order $\mathcal O(N^{-\sqrt{N}})$ under the assumption that the objective function is sampled from a Gaussian process with a Matérn kernel with smoothness parameter $\nu > 4 +\frac{D}{2}$, where $D$ is the number of dimensions. We perform experiments on optimisation of various synthetic functions and machine learning hyperparameter tuning tasks and show that our algorithm outperforms baselines.

----

## [946] On Disentangled Representations Learned from Correlated Data

**Authors**: *Frederik Träuble, Elliot Creager, Niki Kilbertus, Francesco Locatello, Andrea Dittadi, Anirudh Goyal, Bernhard Schölkopf, Stefan Bauer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/trauble21a.html](http://proceedings.mlr.press/v139/trauble21a.html)

**Abstract**:

The focus of disentanglement approaches has been on identifying independent factors of variation in data. However, the causal variables underlying real-world observations are often not statistically independent. In this work, we bridge the gap to real-world scenarios by analyzing the behavior of the most prominent disentanglement approaches on correlated data in a large-scale empirical study (including 4260 models). We show and quantify that systematically induced correlations in the dataset are being learned and reflected in the latent representations, which has implications for downstream applications of disentanglement such as fairness. We also demonstrate how to resolve these latent correlations, either using weak supervision during training or by post-hoc correcting a pre-trained model with a small number of labels.

----

## [947] A New Formalism, Method and Open Issues for Zero-Shot Coordination

**Authors**: *Johannes Treutlein, Michael Dennis, Caspar Oesterheld, Jakob N. Foerster*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/treutlein21a.html](http://proceedings.mlr.press/v139/treutlein21a.html)

**Abstract**:

In many coordination problems, independently reasoning humans are able to discover mutually compatible policies. In contrast, independently trained self-play policies are often mutually incompatible. Zero-shot coordination (ZSC) has recently been proposed as a new frontier in multi-agent reinforcement learning to address this fundamental issue. Prior work approaches the ZSC problem by assuming players can agree on a shared learning algorithm but not on labels for actions and observations, and proposes other-play as an optimal solution. However, until now, this “label-free” problem has only been informally defined. We formalize this setting as the label-free coordination (LFC) problem by defining the label-free coordination game. We show that other-play is not an optimal solution to the LFC problem as it fails to consistently break ties between incompatible maximizers of the other-play objective. We introduce an extension of the algorithm, other-play with tie-breaking, and prove that it is optimal in the LFC problem and an equilibrium in the LFC game. Since arbitrary tie-breaking is precisely what the ZSC setting aims to prevent, we conclude that the LFC problem does not reflect the aims of ZSC. To address this, we introduce an alternative informal operationalization of ZSC as a starting point for future work.

----

## [948] Learning a Universal Template for Few-shot Dataset Generalization

**Authors**: *Eleni Triantafillou, Hugo Larochelle, Richard S. Zemel, Vincent Dumoulin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/triantafillou21a.html](http://proceedings.mlr.press/v139/triantafillou21a.html)

**Abstract**:

Few-shot dataset generalization is a challenging variant of the well-studied few-shot classification problem where a diverse training set of several datasets is given, for the purpose of training an adaptable model that can then learn classes from \emph{new datasets} using only a few examples. To this end, we propose to utilize the diverse training set to construct a \emph{universal template}: a partial model that can define a wide array of dataset-specialized models, by plugging in appropriate components. For each new few-shot classification problem, our approach therefore only requires inferring a small number of parameters to insert into the universal template. We design a separate network that produces an initialization of those parameters for each given task, and we then fine-tune its proposed initialization via a few steps of gradient descent. Our approach is more parameter-efficient, scalable and adaptable compared to previous methods, and achieves the state-of-the-art on the challenging Meta-Dataset benchmark.

----

## [949] Provable Meta-Learning of Linear Representations

**Authors**: *Nilesh Tripuraneni, Chi Jin, Michael I. Jordan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tripuraneni21a.html](http://proceedings.mlr.press/v139/tripuraneni21a.html)

**Abstract**:

Meta-learning, or learning-to-learn, seeks to design algorithms that can utilize previous experience to rapidly learn new skills or adapt to new environments. Representation learning—a key tool for performing meta-learning—learns a data representation that can transfer knowledge across multiple tasks, which is essential in regimes where data is scarce. Despite a recent surge of interest in the practice of meta-learning, the theoretical underpinnings of meta-learning algorithms are lacking, especially in the context of learning transferable representations. In this paper, we focus on the problem of multi-task linear regression—in which multiple linear regression models share a common, low-dimensional linear representation. Here, we provide provably fast, sample-efficient algorithms to address the dual challenges of (1) learning a common set of features from multiple, related tasks, and (2) transferring this knowledge to new, unseen tasks. Both are central to the general problem of meta-learning. Finally, we complement these results by providing information-theoretic lower bounds on the sample complexity of learning these linear features.

----

## [950] Cumulants of Hawkes Processes are Robust to Observation Noise

**Authors**: *William Trouleau, Jalal Etesami, Matthias Grossglauser, Negar Kiyavash, Patrick Thiran*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/trouleau21a.html](http://proceedings.mlr.press/v139/trouleau21a.html)

**Abstract**:

Multivariate Hawkes processes (MHPs) are widely used in a variety of fields to model the occurrence of causally related discrete events in continuous time. Most state-of-the-art approaches address the problem of learning MHPs from perfect traces without noise. In practice, the process through which events are collected might introduce noise in the timestamps. In this work, we address the problem of learning the causal structure of MHPs when the observed timestamps of events are subject to random and unknown shifts, also known as random translations. We prove that the cumulants of MHPs are invariant to random translations, and therefore can be used to learn their underlying causal structure. Furthermore, we empirically characterize the effect of random translations on state-of-the-art learning methods. We show that maximum likelihood-based estimators are brittle, while cumulant-based estimators remain stable even in the presence of significant time shifts.

----

## [951] PixelTransformer: Sample Conditioned Signal Generation

**Authors**: *Shubham Tulsiani, Abhinav Gupta*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/tulsiani21a.html](http://proceedings.mlr.press/v139/tulsiani21a.html)

**Abstract**:

We propose a generative model that can infer a distribution for the underlying spatial signal conditioned on sparse samples e.g. plausible images given a few observed pixels. In contrast to sequential autoregressive generative models, our model allows conditioning on arbitrary samples and can answer distributional queries for any location. We empirically validate our approach across three image datasets and show that we learn to generate diverse and meaningful samples, with the distribution variance reducing given more observed pixels. We also show that our approach is applicable beyond images and can allow generating other types of spatial outputs e.g. polynomials, 3D shapes, and videos.

----

## [952] A Framework for Private Matrix Analysis in Sliding Window Model

**Authors**: *Jalaj Upadhyay, Sarvagya Upadhyay*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/upadhyay21a.html](http://proceedings.mlr.press/v139/upadhyay21a.html)

**Abstract**:

We perform a rigorous study of private matrix analysis when only the last $W$ updates to matrices are considered useful for analysis. We show the existing framework in the non-private setting is not robust to noise required for privacy. We then propose a framework robust to noise and use it to give first efficient $o(W)$ space differentially private algorithms for spectral approximation, principal component analysis (PCA), multi-response linear regression, sparse PCA, and non-negative PCA. Prior to our work, no such result was known for sparse and non-negative differentially private PCA even in the static data setting. We also give a lower bound to demonstrate the cost of privacy in the sliding window model.

----

## [953] Fast Projection Onto Convex Smooth Constraints

**Authors**: *Ilnura Usmanova, Maryam Kamgarpour, Andreas Krause, Kfir Y. Levy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/usmanova21a.html](http://proceedings.mlr.press/v139/usmanova21a.html)

**Abstract**:

The Euclidean projection onto a convex set is an important problem that arises in numerous constrained optimization tasks. Unfortunately, in many cases, computing projections is computationally demanding. In this work, we focus on projection problems where the constraints are smooth and the number of constraints is significantly smaller than the dimension. The runtime of existing approaches to solving such problems is either cubic in the dimension or polynomial in the inverse of the target accuracy. Conversely, we propose a simple and efficient primal-dual approach, with a runtime that scales only linearly with the dimension, and only logarithmically in the inverse of the target accuracy. We empirically demonstrate its performance, and compare it with standard baselines.

----

## [954] SGLB: Stochastic Gradient Langevin Boosting

**Authors**: *Aleksei Ustimenko, Liudmila Prokhorenkova*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/ustimenko21a.html](http://proceedings.mlr.press/v139/ustimenko21a.html)

**Abstract**:

This paper introduces Stochastic Gradient Langevin Boosting (SGLB) - a powerful and efficient machine learning framework that may deal with a wide range of loss functions and has provable generalization guarantees. The method is based on a special form of the Langevin diffusion equation specifically designed for gradient boosting. This allows us to theoretically guarantee the global convergence even for multimodal loss functions, while standard gradient boosting algorithms can guarantee only local optimum. We also empirically show that SGLB outperforms classic gradient boosting when applied to classification tasks with 0-1 loss function, which is known to be multimodal.

----

## [955] LTL2Action: Generalizing LTL Instructions for Multi-Task RL

**Authors**: *Pashootan Vaezipoor, Andrew C. Li, Rodrigo Toro Icarte, Sheila A. McIlraith*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vaezipoor21a.html](http://proceedings.mlr.press/v139/vaezipoor21a.html)

**Abstract**:

We address the problem of teaching a deep reinforcement learning (RL) agent to follow instructions in multi-task environments. Instructions are expressed in a well-known formal language {–} linear temporal logic (LTL) {–} and can specify a diversity of complex, temporally extended behaviours, including conditionals and alternative realizations. Our proposed learning approach exploits the compositional syntax and the semantics of LTL, enabling our RL agent to learn task-conditioned policies that generalize to new instructions, not observed during training. To reduce the overhead of learning LTL semantics, we introduce an environment-agnostic LTL pretraining scheme which improves sample-efficiency in downstream environments. Experiments on discrete and continuous domains target combinatorial task sets of up to $\sim10^{39}$ unique tasks and demonstrate the strength of our approach in learning to solve (unseen) tasks, given LTL instructions.

----

## [956] Active Deep Probabilistic Subsampling

**Authors**: *Hans Van Gorp, Iris A. M. Huijben, Bastiaan S. Veeling, Nicola Pezzotti, Ruud J. G. van Sloun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/van-gorp21a.html](http://proceedings.mlr.press/v139/van-gorp21a.html)

**Abstract**:

Subsampling a signal of interest can reduce costly data transfer, battery drain, radiation exposure and acquisition time in a wide range of problems. The recently proposed Deep Probabilistic Subsampling (DPS) method effectively integrates subsampling in an end-to-end deep learning model, but learns a static pattern for all datapoints. We generalize DPS to a sequential method that actively picks the next sample based on the information acquired so far; dubbed Active-DPS (A-DPS). We validate that A-DPS improves over DPS for MNIST classification at high subsampling rates. Moreover, we demonstrate strong performance in active acquisition Magnetic Resonance Image (MRI) reconstruction, outperforming DPS and other deep learning methods.

----

## [957] CURI: A Benchmark for Productive Concept Learning Under Uncertainty

**Authors**: *Ramakrishna Vedantam, Arthur Szlam, Maximilian Nickel, Ari Morcos, Brenden M. Lake*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vedantam21a.html](http://proceedings.mlr.press/v139/vedantam21a.html)

**Abstract**:

Humans can learn and reason under substantial uncertainty in a space of infinitely many compositional, productive concepts. For example, if a scene with two blue spheres qualifies as “daxy,” one can reason that the underlying concept may require scenes to have “only blue spheres” or “only spheres” or “only two objects.” In contrast, standard benchmarks for compositional reasoning do not explicitly capture a notion of reasoning under uncertainty or evaluate compositional concept acquisition. We introduce a new benchmark, Compositional Reasoning Under Uncertainty (CURI) that instantiates a series of few-shot, meta-learning tasks in a productive concept space to evaluate different aspects of systematic generalization under uncertainty, including splits that test abstract understandings of disentangling, productive generalization, learning boolean operations, variable binding, etc. Importantly, we also contribute a model-independent “compositionality gap” to evaluate the difficulty of generalizing out-of-distribution along each of these axes, allowing objective comparison of the difficulty of each compositional split. Evaluations across a range of modeling choices and splits reveal substantial room for improvement on the proposed benchmark.

----

## [958] Towards Domain-Agnostic Contrastive Learning

**Authors**: *Vikas Verma, Thang Luong, Kenji Kawaguchi, Hieu Pham, Quoc V. Le*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/verma21a.html](http://proceedings.mlr.press/v139/verma21a.html)

**Abstract**:

Despite recent successes, most contrastive self-supervised learning methods are domain-specific, relying heavily on data augmentation techniques that require knowledge about a particular domain, such as image cropping and rotation. To overcome such limitation, we propose a domain-agnostic approach to contrastive learning, named DACL, that is applicable to problems where domain-specific data augmentations are not readily available. Key to our approach is the use of Mixup noise to create similar and dissimilar examples by mixing data samples differently either at the input or hidden-state levels. We theoretically analyze our method and show advantages over the Gaussian-noise based contrastive learning approach. To demonstrate the effectiveness of DACL, we conduct experiments across various domains such as tabular data, images, and graphs. Our results show that DACL not only outperforms other domain-agnostic noising methods, such as Gaussian-noise, but also combines well with domain-specific methods, such as SimCLR, to improve self-supervised visual representation learning.

----

## [959] Sparsifying Networks via Subdifferential Inclusion

**Authors**: *Sagar Verma, Jean-Christophe Pesquet*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/verma21b.html](http://proceedings.mlr.press/v139/verma21b.html)

**Abstract**:

Sparsifying deep neural networks is of paramount interest in many areas, especially when those networks have to be implemented on low-memory devices. In this article, we propose a new formulation of the problem of generating sparse weights for a pre-trained neural network. By leveraging the properties of standard nonlinear activation functions, we show that the problem is equivalent to an approximate subdifferential inclusion problem. The accuracy of the approximation controls the sparsity. We show that the proposed approach is valid for a broad class of activation functions (ReLU, sigmoid, softmax). We propose an iterative optimization algorithm to induce sparsity whose convergence is guaranteed. Because of the algorithm flexibility, the sparsity can be ensured from partial training data in a minibatch manner. To demonstrate the effectiveness of our method, we perform experiments on various networks in different applicative contexts: image classification, speech recognition, natural language processing, and time-series forecasting.

----

## [960] Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies

**Authors**: *Paul Vicol, Luke Metz, Jascha Sohl-Dickstein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vicol21a.html](http://proceedings.mlr.press/v139/vicol21a.html)

**Abstract**:

Unrolled computation graphs arise in many scenarios, including training RNNs, tuning hyperparameters through unrolled optimization, and training learned optimizers. Current approaches to optimizing parameters in such computation graphs suffer from high variance gradients, bias, slow updates, or large memory usage. We introduce a method called Persistent Evolution Strategies (PES), which divides the computation graph into a series of truncated unrolls, and performs an evolution strategies-based update step after each unroll. PES eliminates bias from these truncations by accumulating correction terms over the entire sequence of unrolls. PES allows for rapid parameter updates, has low memory usage, is unbiased, and has reasonable variance characteristics. We experimentally demonstrate the advantages of PES compared to several other methods for gradient estimation on synthetic tasks, and show its applicability to training learned optimizers and tuning hyperparameters.

----

## [961] Online Graph Dictionary Learning

**Authors**: *Cédric Vincent-Cuaz, Titouan Vayer, Rémi Flamary, Marco Corneli, Nicolas Courty*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vincent-cuaz21a.html](http://proceedings.mlr.press/v139/vincent-cuaz21a.html)

**Abstract**:

Dictionary learning is a key tool for representation learning, that explains the data as linear combination of few basic elements. Yet, this analysis is not amenable in the context of graph learning, as graphs usually belong to different metric spaces. We fill this gap by proposing a new online Graph Dictionary Learning approach, which uses the Gromov Wasserstein divergence for the data fitting term. In our work, graphs are encoded through their nodes’ pairwise relations and modeled as convex combination of graph atoms, i.e. dictionary elements, estimated thanks to an online stochastic algorithm, which operates on a dataset of unregistered graphs with potentially different number of nodes. Our approach naturally extends to labeled graphs, and is completed by a novel upper bound that can be used as a fast approximation of Gromov Wasserstein in the embedding space. We provide numerical evidences showing the interest of our approach for unsupervised embedding of graph datasets and for online graph subspace estimation and tracking.

----

## [962] Neuro-algorithmic Policies Enable Fast Combinatorial Generalization

**Authors**: *Marin Vlastelica P., Michal Rolínek, Georg Martius*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vlastelica21a.html](http://proceedings.mlr.press/v139/vlastelica21a.html)

**Abstract**:

Although model-based and model-free approaches to learning the control of systems have achieved impressive results on standard benchmarks, generalization to task variations is still lacking. Recent results suggest that generalization for standard architectures improves only after obtaining exhaustive amounts of data. We give evidence that generalization capabilities are in many cases bottlenecked by the inability to generalize on the combinatorial aspects of the problem. We show that, for a certain subclass of the MDP framework, this can be alleviated by a neuro-algorithmic policy architecture that embeds a time-dependent shortest path solver in a deep neural network. Trained end-to-end via blackbox-differentiation, this method leads to considerable improvement in generalization capabilities in the low-data regime.

----

## [963] Efficient Training of Robust Decision Trees Against Adversarial Examples

**Authors**: *Daniël Vos, Sicco Verwer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vos21a.html](http://proceedings.mlr.press/v139/vos21a.html)

**Abstract**:

Current state-of-the-art algorithms for training robust decision trees have high runtime costs and require hours to run. We present GROOT, an efficient algorithm for training robust decision trees and random forests that runs in a matter of seconds to minutes. Where before the worst-case Gini impurity was computed iteratively, we find that we can solve this function analytically to improve time complexity from O(n) to O(1) in terms of n samples. Our results on both single trees and ensembles on 14 structured datasets as well as on MNIST and Fashion-MNIST demonstrate that GROOT runs several orders of magnitude faster than the state-of-the-art works and also shows better performance in terms of adversarial accuracy on structured data.

----

## [964] Object Segmentation Without Labels with Large-Scale Generative Models

**Authors**: *Andrey Voynov, Stanislav Morozov, Artem Babenko*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/voynov21a.html](http://proceedings.mlr.press/v139/voynov21a.html)

**Abstract**:

The recent rise of unsupervised and self-supervised learning has dramatically reduced the dependency on labeled data, providing high-quality representations for transfer on downstream tasks. Furthermore, recent works also employed these representations in a fully unsupervised setup for image classification, reducing the need for human labels on the fine-tuning stage as well. This work demonstrates that large-scale unsupervised models can also perform a more challenging object segmentation task, requiring neither pixel-level nor image-level labeling. Namely, we show that recent unsupervised GANs allow to differentiate between foreground/background pixels, providing high-quality saliency masks. By extensive comparison on common benchmarks, we outperform existing unsupervised alternatives for object segmentation, achieving new state-of-the-art.

----

## [965] Principal Component Hierarchy for Sparse Quadratic Programs

**Authors**: *Robbie Vreugdenhil, Viet Anh Nguyen, Armin Eftekhari, Peyman Mohajerin Esfahani*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/vreugdenhil21a.html](http://proceedings.mlr.press/v139/vreugdenhil21a.html)

**Abstract**:

We propose a novel approximation hierarchy for cardinality-constrained, convex quadratic programs that exploits the rank-dominating eigenvectors of the quadratic matrix. Each level of approximation admits a min-max characterization whose objective function can be optimized over the binary variables analytically, while preserving convexity in the continuous variables. Exploiting this property, we propose two scalable optimization algorithms, coined as the “best response" and the “dual program", that can efficiently screen the potential indices of the nonzero elements of the original program. We show that the proposed methods are competitive with the existing screening methods in the current sparse regression literature, and it is particularly fast on instances with high number of measurements in experiments with both synthetic and real datasets.

----

## [966] Whitening and Second Order Optimization Both Make Information in the Dataset Unusable During Training, and Can Reduce or Prevent Generalization

**Authors**: *Neha S. Wadia, Daniel Duckworth, Samuel S. Schoenholz, Ethan Dyer, Jascha Sohl-Dickstein*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wadia21a.html](http://proceedings.mlr.press/v139/wadia21a.html)

**Abstract**:

Machine learning is predicated on the concept of generalization: a model achieving low error on a sufficiently large training set should also perform well on novel samples from the same distribution. We show that both data whitening and second order optimization can harm or entirely prevent generalization. In general, model training harnesses information contained in the sample-sample second moment matrix of a dataset. For a general class of models, namely models with a fully connected first layer, we prove that the information contained in this matrix is the only information which can be used to generalize. Models trained using whitened data, or with certain second order optimization schemes, have less access to this information, resulting in reduced or nonexistent generalization ability. We experimentally verify these predictions for several architectures, and further demonstrate that generalization continues to be harmed even when theoretical requirements are relaxed. However, we also show experimentally that regularized second order optimization can provide a practical tradeoff, where training is accelerated but less information is lost, and generalization can in some circumstances even improve.

----

## [967] Safe Reinforcement Learning Using Advantage-Based Intervention

**Authors**: *Nolan Wagener, Byron Boots, Ching-An Cheng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wagener21a.html](http://proceedings.mlr.press/v139/wagener21a.html)

**Abstract**:

Many sequential decision problems involve finding a policy that maximizes total reward while obeying safety constraints. Although much recent research has focused on the development of safe reinforcement learning (RL) algorithms that produce a safe policy after training, ensuring safety during training as well remains an open problem. A fundamental challenge is performing exploration while still satisfying constraints in an unknown Markov decision process (MDP). In this work, we address this problem for the chance-constrained setting.We propose a new algorithm, SAILR, that uses an intervention mechanism based on advantage functions to keep the agent safe throughout training and optimizes the agent’s policy using off-the-shelf RL algorithms designed for unconstrained MDPs. Our method comes with strong guarantees on safety during "both" training and deployment (i.e., after training and without the intervention mechanism) and policy performance compared to the optimal safety-constrained policy. In our experiments, we show that SAILR violates constraints far less during training than standard safe RL and constrained MDP approaches and converges to a well-performing policy that can be deployed safely without intervention. Our code is available at https://github.com/nolanwagener/safe_rl.

----

## [968] Task-Optimal Exploration in Linear Dynamical Systems

**Authors**: *Andrew J. Wagenmaker, Max Simchowitz, Kevin G. Jamieson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wagenmaker21a.html](http://proceedings.mlr.press/v139/wagenmaker21a.html)

**Abstract**:

Exploration in unknown environments is a fundamental problem in reinforcement learning and control. In this work, we study task-guided exploration and determine what precisely an agent must learn about their environment in order to complete a particular task. Formally, we study a broad class of decision-making problems in the setting of linear dynamical systems, a class that includes the linear quadratic regulator problem. We provide instance- and task-dependent lower bounds which explicitly quantify the difficulty of completing a task of interest. Motivated by our lower bound, we propose a computationally efficient experiment-design based exploration algorithm. We show that it optimally explores the environment, collecting precisely the information needed to complete the task, and provide finite-time bounds guaranteeing that it achieves the instance- and task-optimal sample complexity, up to constant factors. Through several examples of the linear quadratic regulator problem, we show that performing task-guided exploration provably improves on exploration schemes which do not take into account the task of interest. Along the way, we establish that certainty equivalence decision making is instance- and task-optimal, and obtain the first algorithm for the linear quadratic regulator problem which is instance-optimal. We conclude with several experiments illustrating the effectiveness of our approach in practice.

----

## [969] Learning and Planning in Average-Reward Markov Decision Processes

**Authors**: *Yi Wan, Abhishek Naik, Richard S. Sutton*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wan21a.html](http://proceedings.mlr.press/v139/wan21a.html)

**Abstract**:

We introduce learning and planning algorithms for average-reward MDPs, including 1) the first general proven-convergent off-policy model-free control algorithm without reference states, 2) the first proven-convergent off-policy model-free prediction algorithm, and 3) the first off-policy learning algorithm that converges to the actual value function rather than to the value function plus an offset. All of our algorithms are based on using the temporal-difference error rather than the conventional error when updating the estimate of the average reward. Our proof techniques are a slight generalization of those by Abounadi, Bertsekas, and Borkar (2001). In experiments with an Access-Control Queuing Task, we show some of the difficulties that can arise when using methods that rely on reference states and argue that our new algorithms are significantly easier to use.

----

## [970] Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces

**Authors**: *Xingchen Wan, Vu Nguyen, Huong Ha, Bin Xin Ru, Cong Lu, Michael A. Osborne*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wan21b.html](http://proceedings.mlr.press/v139/wan21b.html)

**Abstract**:

High-dimensional black-box optimisation remains an important yet notoriously challenging problem. Despite the success of Bayesian optimisation methods on continuous domains, domains that are categorical, or that mix continuous and categorical variables, remain challenging. We propose a novel solution—we combine local optimisation with a tailored kernel design, effectively handling high-dimensional categorical and mixed search spaces, whilst retaining sample efficiency. We further derive convergence guarantee for the proposed approach. Finally, we demonstrate empirically that our method outperforms the current baselines on a variety of synthetic and real-world tasks in terms of performance, computational costs, or both.

----

## [971] Zero-Shot Knowledge Distillation from a Decision-Based Black-Box Model

**Authors**: *Zi Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21a.html](http://proceedings.mlr.press/v139/wang21a.html)

**Abstract**:

Knowledge distillation (KD) is a successful approach for deep neural network acceleration, with which a compact network (student) is trained by mimicking the softmax output of a pre-trained high-capacity network (teacher). In tradition, KD usually relies on access to the training samples and the parameters of the white-box teacher to acquire the transferred knowledge. However, these prerequisites are not always realistic due to storage costs or privacy issues in real-world applications. Here we propose the concept of decision-based black-box (DB3) knowledge distillation, with which the student is trained by distilling the knowledge from a black-box teacher (parameters are not accessible) that only returns classes rather than softmax outputs. We start with the scenario when the training set is accessible. We represent a sample’s robustness against other classes by computing its distances to the teacher’s decision boundaries and use it to construct the soft label for each training sample. After that, the student can be trained via standard KD. We then extend this approach to a more challenging scenario in which even accessing the training data is not feasible. We propose to generate pseudo samples that are distinguished by the decision boundaries of the DB3 teacher to the largest extent and construct soft labels for these samples, which are used as the transfer set. We evaluate our approaches on various benchmark networks and datasets and experiment results demonstrate their effectiveness.

----

## [972] Fairness of Exposure in Stochastic Bandits

**Authors**: *Lequn Wang, Yiwei Bai, Wen Sun, Thorsten Joachims*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21b.html](http://proceedings.mlr.press/v139/wang21b.html)

**Abstract**:

Contextual bandit algorithms have become widely used for recommendation in online systems (e.g. marketplaces, music streaming, news), where they now wield substantial influence on which items get shown to users. This raises questions of fairness to the items — and to the sellers, artists, and writers that benefit from this exposure. We argue that the conventional bandit formulation can lead to an undesirable and unfair winner-takes-all allocation of exposure. To remedy this problem, we propose a new bandit objective that guarantees merit-based fairness of exposure to the items while optimizing utility to the users. We formulate fairness regret and reward regret in this setting and present algorithms for both stochastic multi-armed bandits and stochastic linear bandits. We prove that the algorithms achieve sublinear fairness regret and reward regret. Beyond the theoretical analysis, we also provide empirical evidence that these algorithms can allocate exposure to different arms effectively.

----

## [973] A Proxy Variable View of Shared Confounding

**Authors**: *Yixin Wang, David M. Blei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21c.html](http://proceedings.mlr.press/v139/wang21c.html)

**Abstract**:

Causal inference from observational data can be biased by unobserved confounders. Confounders{—}the variables that affect both the treatments and the outcome{—}induce spurious non-causal correlations between the two. Without additional conditions, unobserved confounders generally make causal quantities hard to identify. In this paper, we focus on the setting where there are many treatments with shared confounding, and we study under what conditions is causal identification possible. The key observation is that we can view subsets of treatments as proxies of the unobserved confounder and identify the intervention distributions of the rest. Moreover, while existing identification formulas for proxy variables involve solving integral equations, we show that one can circumvent the need for such solutions by directly modeling the data. Finally, we extend these results to an expanded class of causal graphs, those with other confounders and selection variables.

----

## [974] Fast Algorithms for Stackelberg Prediction Game with Least Squares Loss

**Authors**: *Jiali Wang, He Chen, Rujun Jiang, Xudong Li, Zihao Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21d.html](http://proceedings.mlr.press/v139/wang21d.html)

**Abstract**:

The Stackelberg prediction game (SPG) has been extensively used to model the interactions between the learner and data provider in the training process of various machine learning algorithms. Particularly, SPGs played prominent roles in cybersecurity applications, such as intrusion detection, banking fraud detection, spam filtering, and malware detection. Often formulated as NP-hard bi-level optimization problems, it is generally computationally intractable to find global solutions to SPGs. As an interesting progress in this area, a special class of SPGs with the least squares loss (SPG-LS) have recently been shown polynomially solvable by a bisection method. However, in each iteration of this method, a semidefinite program (SDP) needs to be solved. The resulted high computational costs prevent its applications for large-scale problems. In contrast, we propose a novel approach that reformulates a SPG-LS as a single SDP of a similar form and the same dimension as those solved in the bisection method. Our SDP reformulation is, evidenced by our numerical experiments, orders of magnitude faster than the existing bisection method. We further show that the obtained SDP can be reduced to a second order cone program (SOCP). This allows us to provide real-time response to large-scale SPG-LS problems. Numerical results on both synthetic and real world datasets indicate that the proposed SOCP method is up to 20,000+ times faster than the state of the art.

----

## [975] Accelerate CNNs from Three Dimensions: A Comprehensive Pruning Framework

**Authors**: *Wenxiao Wang, Minghao Chen, Shuai Zhao, Long Chen, Jinming Hu, Haifeng Liu, Deng Cai, Xiaofei He, Wei Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21e.html](http://proceedings.mlr.press/v139/wang21e.html)

**Abstract**:

Most neural network pruning methods, such as filter-level and layer-level prunings, prune the network model along one dimension (depth, width, or resolution) solely to meet a computational budget. However, such a pruning policy often leads to excessive reduction of that dimension, thus inducing a huge accuracy loss. To alleviate this issue, we argue that pruning should be conducted along three dimensions comprehensively. For this purpose, our pruning framework formulates pruning as an optimization problem. Specifically, it first casts the relationships between a certain model’s accuracy and depth/width/resolution into a polynomial regression and then maximizes the polynomial to acquire the optimal values for the three dimensions. Finally, the model is pruned along the three optimal dimensions accordingly. In this framework, since collecting too much data for training the regression is very time-costly, we propose two approaches to lower the cost: 1) specializing the polynomial to ensure an accurate regression even with less training data; 2) employing iterative pruning and fine-tuning to collect the data faster. Extensive experiments show that our proposed algorithm surpasses state-of-the-art pruning algorithms and even neural architecture search-based algorithms.

----

## [976] Explainable Automated Graph Representation Learning with Hyperparameter Importance

**Authors**: *Xin Wang, Shuyi Fan, Kun Kuang, Wenwu Zhu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21f.html](http://proceedings.mlr.press/v139/wang21f.html)

**Abstract**:

Current graph representation (GR) algorithms require huge demand of human experts in hyperparameter tuning, which significantly limits their practical applications, leading to an urge for automated graph representation without human intervention. Although automated machine learning (AutoML) serves as a good candidate for automatic hyperparameter tuning, little literature has been reported on automated graph presentation learning and the only existing work employs a black-box strategy, lacking insights into explaining the relative importance of different hyperparameters. To address this issue, we study explainable automated graph representation with hyperparameter importance in this paper. We propose an explainable AutoML approach for graph representation (e-AutoGR) which utilizes explainable graph features during performance estimation and learns decorrelated importance weights for different hyperparameters in affecting the model performance through a non-linear decorrelated weighting regression. These learned importance weights can in turn help to provide more insights in hyperparameter search procedure. We theoretically prove the soundness of the decorrelated weighting algorithm. Extensive experiments on real-world datasets demonstrate the superiority of our proposed e-AutoGR model against state-of-the-art methods in terms of both model performance and hyperparameter importance explainability.

----

## [977] Self-Tuning for Data-Efficient Deep Learning

**Authors**: *Ximei Wang, Jinghan Gao, Mingsheng Long, Jianmin Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21g.html](http://proceedings.mlr.press/v139/wang21g.html)

**Abstract**:

Deep learning has made revolutionary advances to diverse applications in the presence of large-scale labeled datasets. However, it is prohibitively time-costly and labor-expensive to collect sufficient labeled data in most realistic scenarios. To mitigate the requirement for labeled data, semi-supervised learning (SSL) focuses on simultaneously exploring both labeled and unlabeled data, while transfer learning (TL) popularizes a favorable practice of fine-tuning a pre-trained model to the target data. A dilemma is thus encountered: Without a decent pre-trained model to provide an implicit regularization, SSL through self-training from scratch will be easily misled by inaccurate pseudo-labels, especially in large-sized label space; Without exploring the intrinsic structure of unlabeled data, TL through fine-tuning from limited labeled data is at risk of under-transfer caused by model shift. To escape from this dilemma, we present Self-Tuning to enable data-efficient deep learning by unifying the exploration of labeled and unlabeled data and the transfer of a pre-trained model, as well as a Pseudo Group Contrast (PGC) mechanism to mitigate the reliance on pseudo-labels and boost the tolerance to false labels. Self-Tuning outperforms its SSL and TL counterparts on five tasks by sharp margins, e.g. it doubles the accuracy of fine-tuning on Cars with $15%$ labels.

----

## [978] Label Distribution Learning Machine

**Authors**: *Jing Wang, Xin Geng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21h.html](http://proceedings.mlr.press/v139/wang21h.html)

**Abstract**:

Although Label Distribution Learning (LDL) has witnessed extensive classification applications, it faces the challenge of objective mismatch – the objective of LDL mismatches that of classification, which has seldom been noticed in existing studies. Our goal is to solve the objective mismatch and improve the classification performance of LDL. Specifically, we extend the margin theory to LDL and propose a new LDL method called \textbf{L}abel \textbf{D}istribution \textbf{L}earning \textbf{M}achine (LDLM). First, we define the label distribution margin and propose the \textbf{S}upport \textbf{V}ector \textbf{R}egression \textbf{M}achine (SVRM) to learn the optimal label. Second, we propose the adaptive margin loss to learn label description degrees. In theoretical analysis, we develop a generalization theory for the SVRM and analyze the generalization of LDLM. Experimental results validate the better classification performance of LDLM.

----

## [979] AlphaNet: Improved Training of Supernets with Alpha-Divergence

**Authors**: *Dilin Wang, Chengyue Gong, Meng Li, Qiang Liu, Vikas Chandra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21i.html](http://proceedings.mlr.press/v139/wang21i.html)

**Abstract**:

Weight-sharing neural architecture search (NAS) is an effective technique for automating efficient neural architecture design. Weight-sharing NAS builds a supernet that assembles all the architectures as its sub-networks and jointly trains the supernet with the sub-networks. The success of weight-sharing NAS heavily relies on distilling the knowledge of the supernet to the sub-networks. However, we find that the widely used distillation divergence, i.e., KL divergence, may lead to student sub-networks that over-estimate or under-estimate the uncertainty of the teacher supernet, leading to inferior performance of the sub-networks. In this work, we propose to improve the supernet training with a more generalized alpha-divergence. By adaptively selecting the alpha-divergence, we simultaneously prevent the over-estimation or under-estimation of the uncertainty of the teacher model. We apply the proposed alpha-divergence based supernets training to both slimmable neural networks and weight-sharing NAS, and demonstrate significant improvements. Specifically, our discovered model family, AlphaNet, outperforms prior-art models on a wide range of FLOPs regimes, including BigNAS, Once-for-All networks, and AttentiveNAS. We achieve ImageNet top-1 accuracy of 80.0% with only 444M FLOPs. Our code and pretrained models are available at https://github.com/facebookresearch/AlphaNet.

----

## [980] Global Convergence of Policy Gradient for Linear-Quadratic Mean-Field Control/Game in Continuous Time

**Authors**: *Weichen Wang, Jiequn Han, Zhuoran Yang, Zhaoran Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21j.html](http://proceedings.mlr.press/v139/wang21j.html)

**Abstract**:

Recent years have witnessed the success of multi-agent reinforcement learning, which has motivated new research directions for mean-field control (MFC) and mean-field game (MFG), as the multi-agent system can be well approximated by a mean-field problem when the number of agents grows to be very large. In this paper, we study the policy gradient (PG) method for the linear-quadratic mean-field control and game, where we assume each agent has identical linear state transitions and quadratic cost functions. While most recent works on policy gradient for MFC and MFG are based on discrete-time models, we focus on a continuous-time model where some of our analyzing techniques could be valuable to the interested readers. For both the MFC and the MFG, we provide PG update and show that it converges to the optimal solution at a linear rate, which is verified by a synthetic simulation. For the MFG, we also provide sufficient conditions for the existence and uniqueness of the Nash equilibrium.

----

## [981] SG-PALM: a Fast Physically Interpretable Tensor Graphical Model

**Authors**: *Yu Wang, Alfred Olivier Hero*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21k.html](http://proceedings.mlr.press/v139/wang21k.html)

**Abstract**:

We propose a new graphical model inference procedure, called SG-PALM, for learning conditional dependency structure of high-dimensional tensor-variate data. Unlike most other tensor graphical models the proposed model is interpretable and computationally scalable to high dimension. Physical interpretability follows from the Sylvester generative (SG) model on which SG-PALM is based: the model is exact for any observation process that is a solution of a partial differential equation of Poisson type. Scalability follows from the fast proximal alternating linearized minimization (PALM) procedure that SG-PALM uses during training. We establish that SG-PALM converges linearly (i.e., geometric convergence rate) to a global optimum of its objective function. We demonstrate scalability and accuracy of SG-PALM for an important but challenging climate prediction problem: spatio-temporal forecasting of solar flares from multimodal imaging data.

----

## [982] Deep Generative Learning via Schrödinger Bridge

**Authors**: *Gefei Wang, Yuling Jiao, Qian Xu, Yang Wang, Can Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21l.html](http://proceedings.mlr.press/v139/wang21l.html)

**Abstract**:

We propose to learn a generative model via entropy interpolation with a Schr{ö}dinger Bridge. The generative learning task can be formulated as interpolating between a reference distribution and a target distribution based on the Kullback-Leibler divergence. At the population level, this entropy interpolation is characterized via an SDE on [0,1] with a time-varying drift term. At the sample level, we derive our Schr{ö}dinger Bridge algorithm by plugging the drift term estimated by a deep score estimator and a deep density ratio estimator into the Euler-Maruyama method. Under some mild smoothness assumptions of the target distribution, we prove the consistency of both the score estimator and the density ratio estimator, and then establish the consistency of the proposed Schr{ö}dinger Bridge approach. Our theoretical results guarantee that the distribution learned by our approach converges to the target distribution. Experimental results on multimodal synthetic data and benchmark data support our theoretical findings and indicate that the generative model via Schr{ö}dinger Bridge is comparable with state-of-the-art GANs, suggesting a new formulation of generative learning. We demonstrate its usefulness in image interpolation and image inpainting.

----

## [983] Robust Inference for High-Dimensional Linear Models via Residual Randomization

**Authors**: *Y. Samuel Wang, Si Kai Lee, Panos Toulis, Mladen Kolar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21m.html](http://proceedings.mlr.press/v139/wang21m.html)

**Abstract**:

We propose a residual randomization procedure designed for robust inference using Lasso estimates in the high-dimensional setting. Compared to earlier work that focuses on sub-Gaussian errors, the proposed procedure is designed to work robustly in settings that also include heavy-tailed covariates and errors. Moreover, our procedure can be valid under clustered errors, which is important in practice, but has been largely overlooked by earlier work. Through extensive simulations, we illustrate our method’s wider range of applicability as suggested by theory. In particular, we show that our method outperforms state-of-art methods in challenging, yet more realistic, settings where the distribution of covariates is heavy-tailed or the sample size is small, while it remains competitive in standard, “well behaved" settings previously studied in the literature.

----

## [984] A Modular Analysis of Provable Acceleration via Polyak's Momentum: Training a Wide ReLU Network and a Deep Linear Network

**Authors**: *Jun-Kun Wang, Chi-Heng Lin, Jacob D. Abernethy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21n.html](http://proceedings.mlr.press/v139/wang21n.html)

**Abstract**:

Incorporating a so-called “momentum” dynamic in gradient descent methods is widely used in neural net training as it has been broadly observed that, at least empirically, it often leads to significantly faster convergence. At the same time, there are very few theoretical guarantees in the literature to explain this apparent acceleration effect. Even for the classical strongly convex quadratic problems, several existing results only show Polyak’s momentum has an accelerated linear rate asymptotically. In this paper, we first revisit the quadratic problems and show a non-asymptotic accelerated linear rate of Polyak’s momentum. Then, we provably show that Polyak’s momentum achieves acceleration for training a one-layer wide ReLU network and a deep linear network, which are perhaps the two most popular canonical models for studying optimization and deep learning in the literature. Prior works (Du et al. 2019) and (Wu et al. 2019) showed that using vanilla gradient descent, and with an use of over-parameterization, the error decays as $(1- \Theta(\frac{1}{ \kappa’}))^t$ after $t$ iterations, where $\kappa’$ is the condition number of a Gram Matrix. Our result shows that with the appropriate choice of parameters Polyak’s momentum has a rate of $(1-\Theta(\frac{1}{\sqrt{\kappa’}}))^t$. For the deep linear network, prior work (Hu et al. 2020) showed that vanilla gradient descent has a rate of $(1-\Theta(\frac{1}{\kappa}))^t$, where $\kappa$ is the condition number of a data matrix. Our result shows an acceleration rate $(1- \Theta(\frac{1}{\sqrt{\kappa}}))^t$ is achievable by Polyak’s momentum. This work establishes that momentum does indeed speed up neural net training.

----

## [985] Optimal Non-Convex Exact Recovery in Stochastic Block Model via Projected Power Method

**Authors**: *Peng Wang, Huikang Liu, Zirui Zhou, Anthony Man-Cho So*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21o.html](http://proceedings.mlr.press/v139/wang21o.html)

**Abstract**:

In this paper, we study the problem of exact community recovery in the symmetric stochastic block model, where a graph of $n$ vertices is randomly generated by partitioning the vertices into $K \ge 2$ equal-sized communities and then connecting each pair of vertices with probability that depends on their community memberships. Although the maximum-likelihood formulation of this problem is discrete and non-convex, we propose to tackle it directly using projected power iterations with an initialization that satisfies a partial recovery condition. Such an initialization can be obtained by a host of existing methods. We show that in the logarithmic degree regime of the considered problem, the proposed method can exactly recover the underlying communities at the information-theoretic limit. Moreover, with a qualified initialization, it runs in $\mO(n\log^2n/\log\log n)$ time, which is competitive with existing state-of-the-art methods. We also present numerical results of the proposed method to support and complement our theoretical development.

----

## [986] ConvexVST: A Convex Optimization Approach to Variance-stabilizing Transformation

**Authors**: *Mengfan Wang, Boyu Lyu, Guoqiang Yu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21p.html](http://proceedings.mlr.press/v139/wang21p.html)

**Abstract**:

The variance-stabilizing transformation (VST) problem is to transform heteroscedastic data to homoscedastic data so that they are more tractable for subsequent analysis. However, most of the existing approaches focus on finding an analytical solution for a certain parametric distribution, which severely limits the applications, because simple distributions cannot faithfully describe the real data while more complicated distributions cannot be analytically solved. In this paper, we converted the VST problem into a convex optimization problem, which can always be efficiently solved, identified the specific structure of the convex problem, which further improved the efficiency of the proposed algorithm, and showed that any finite discrete distributions and the discretized version of any continuous distributions from real data can be variance-stabilized in an easy and nonparametric way. We demonstrated the new approach on bioimaging data and achieved superior performance compared to peer algorithms in terms of not only the variance homoscedasticity but also the impact on subsequent analysis such as denoising. Source codes are available at https://github.com/yu-lab-vt/ConvexVST.

----

## [987] The Implicit Bias for Adaptive Optimization Algorithms on Homogeneous Neural Networks

**Authors**: *Bohan Wang, Qi Meng, Wei Chen, Tie-Yan Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21q.html](http://proceedings.mlr.press/v139/wang21q.html)

**Abstract**:

Despite their overwhelming capacity to overfit, deep neural networks trained by specific optimization algorithms tend to generalize relatively well to unseen data. Recently, researchers explained it by investigating the implicit bias of optimization algorithms. A remarkable progress is the work (Lyu & Li, 2019), which proves gradient descent (GD) maximizes the margin of homogeneous deep neural networks. Except the first-order optimization algorithms like GD, adaptive algorithms such as AdaGrad, RMSProp and Adam are popular owing to their rapid training process. Mean-while, numerous works have provided empirical evidence that adaptive methods may suffer from poor generalization performance. However, theoretical explanation for the generalization of adaptive optimization algorithms is still lacking. In this paper, we study the implicit bias of adaptive optimization algorithms on homogeneous neural networks. In particular, we study the convergent direction of parameters when they are optimizing the logistic loss. We prove that the convergent direction of Adam and RMSProp is the same as GD, while for AdaGrad, the convergent direction depends on the adaptive conditioner. Technically, we provide a unified framework to analyze convergent direction of adaptive optimization algorithms by constructing novel and nontrivial adaptive gradient flow and surrogate margin. The theoretical findings explain the superiority on generalization of exponential moving average strategy that is adopted by RMSProp and Adam. To the best of knowledge, it is the first work to study the convergent direction of adaptive optimizations on non-linear deep neural networks

----

## [988] Robust Learning for Data Poisoning Attacks

**Authors**: *Yunjuan Wang, Poorya Mianjy, Raman Arora*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21r.html](http://proceedings.mlr.press/v139/wang21r.html)

**Abstract**:

We investigate the robustness of stochastic approximation approaches against data poisoning attacks. We focus on two-layer neural networks with ReLU activation and show that under a specific notion of separability in the RKHS induced by the infinite-width network, training (finite-width) networks with stochastic gradient descent is robust against data poisoning attacks. Interestingly, we find that in addition to a lower bound on the width of the network, which is standard in the literature, we also require a distribution-dependent upper bound on the width for robust generalization. We provide extensive empirical evaluations that support and validate our theoretical results.

----

## [989] SketchEmbedNet: Learning Novel Concepts by Imitating Drawings

**Authors**: *Alexander Wang, Mengye Ren, Richard S. Zemel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21s.html](http://proceedings.mlr.press/v139/wang21s.html)

**Abstract**:

Sketch drawings capture the salient information of visual concepts. Previous work has shown that neural networks are capable of producing sketches of natural objects drawn from a small number of classes. While earlier approaches focus on generation quality or retrieval, we explore properties of image representations learned by training a model to produce sketches of images. We show that this generative, class-agnostic model produces informative embeddings of images from novel examples, classes, and even novel datasets in a few-shot setting. Additionally, we find that these learned representations exhibit interesting structure and compositionality.

----

## [990] Directional Bias Amplification

**Authors**: *Angelina Wang, Olga Russakovsky*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21t.html](http://proceedings.mlr.press/v139/wang21t.html)

**Abstract**:

Mitigating bias in machine learning systems requires refining our understanding of bias propagation pathways: from societal structures to large-scale data to trained models to impact on society. In this work, we focus on one aspect of the problem, namely bias amplification: the tendency of models to amplify the biases present in the data they are trained on. A metric for measuring bias amplification was introduced in the seminal work by Zhao et al. (2017); however, as we demonstrate, this metric suffers from a number of shortcomings including conflating different types of bias amplification and failing to account for varying base rates of protected attributes. We introduce and analyze a new, decoupled metric for measuring bias amplification, $BiasAmp_{\rightarrow}$ (Directional Bias Amplification). We thoroughly analyze and discuss both the technical assumptions and normative implications of this metric. We provide suggestions about its measurement by cautioning against predicting sensitive attributes, encouraging the use of confidence intervals due to fluctuations in the fairness of models across runs, and discussing the limitations of what this metric captures. Throughout this paper, we work to provide an interrogative look at the technical measurement of bias amplification, guided by our normative ideas of what we want it to encompass. Code is located at https://github.com/princetonvisualai/directional-bias-amp.

----

## [991] An exact solver for the Weston-Watkins SVM subproblem

**Authors**: *Yutong Wang, Clayton Scott*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21u.html](http://proceedings.mlr.press/v139/wang21u.html)

**Abstract**:

Recent empirical evidence suggests that the Weston-Watkins support vector machine is among the best performing multiclass extensions of the binary SVM. Current state-of-the-art solvers repeatedly solve a particular subproblem approximately using an iterative strategy. In this work, we propose an algorithm that solves the subproblem exactly using a novel reparametrization of the Weston-Watkins dual problem. For linear WW-SVMs, our solver shows significant speed-up over the state-of-the-art solver when the number of classes is large. Our exact subproblem solver also allows us to prove linear convergence of the overall solver.

----

## [992] SCC: an efficient deep reinforcement learning agent mastering the game of StarCraft II

**Authors**: *Xiangjun Wang, Junxiao Song, Penghui Qi, Peng Peng, Zhenkun Tang, Wei Zhang, Weimin Li, Xiongjun Pi, Jujie He, Chao Gao, Haitao Long, Quan Yuan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21v.html](http://proceedings.mlr.press/v139/wang21v.html)

**Abstract**:

AlphaStar, the AI that reaches GrandMaster level in StarCraft II, is a remarkable milestone demonstrating what deep reinforcement learning can achieve in complex Real-Time Strategy (RTS) games. However, the complexities of the game, algorithms and systems, and especially the tremendous amount of computation needed are big obstacles for the community to conduct further research in this direction. We propose a deep reinforcement learning agent, StarCraft Commander (SCC). With order of magnitude less computation, it demonstrates top human performance defeating GrandMaster players in test matches and top professional players in a live event. Moreover, it shows strong robustness to various human strategies and discovers novel strategies unseen from human plays. In this paper, we’ll share the key insights and optimizations on efficient imitation learning and reinforcement learning for StarCraft II full game.

----

## [993] Quantum algorithms for reinforcement learning with a generative model

**Authors**: *Daochen Wang, Aarthi Sundaram, Robin Kothari, Ashish Kapoor, Martin Roetteler*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21w.html](http://proceedings.mlr.press/v139/wang21w.html)

**Abstract**:

Reinforcement learning studies how an agent should interact with an environment to maximize its cumulative reward. A standard way to study this question abstractly is to ask how many samples an agent needs from the environment to learn an optimal policy for a $\gamma$-discounted Markov decision process (MDP). For such an MDP, we design quantum algorithms that approximate an optimal policy ($\pi^*$), the optimal value function ($v^*$), and the optimal $Q$-function ($q^*$), assuming the algorithms can access samples from the environment in quantum superposition. This assumption is justified whenever there exists a simulator for the environment; for example, if the environment is a video game or some other program. Our quantum algorithms, inspired by value iteration, achieve quadratic speedups over the best-possible classical sample complexities in the approximation accuracy ($\epsilon$) and two main parameters of the MDP: the effective time horizon ($\frac{1}{1-\gamma}$) and the size of the action space ($A$). Moreover, we show that our quantum algorithm for computing $q^*$ is optimal by proving a matching quantum lower bound.

----

## [994] Matrix Completion with Model-free Weighting

**Authors**: *Jiayi Wang, Raymond K. W. Wong, Xiaojun Mao, Kwun Chuen Gary Chan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21x.html](http://proceedings.mlr.press/v139/wang21x.html)

**Abstract**:

In this paper, we propose a novel method for matrix completion under general non-uniform missing structures. By controlling an upper bound of a novel balancing error, we construct weights that can actively adjust for the non-uniformity in the empirical risk without explicitly modeling the observation probabilities, and can be computed efficiently via convex optimization. The recovered matrix based on the proposed weighted empirical risk enjoys appealing theoretical guarantees. In particular, the proposed method achieves stronger guarantee than existing work in terms of the scaling with respect to the observation probabilities, under asymptotically heterogeneous missing settings (where entry-wise observation probabilities can be of different orders). These settings can be regarded as a better theoretical model of missing patterns with highly varying probabilities. We also provide a new minimax lower bound under a class of heterogeneous settings. Numerical experiments are also provided to demonstrate the effectiveness of the proposed method.

----

## [995] UniSpeech: Unified Speech Representation Learning with Labeled and Unlabeled Data

**Authors**: *Chengyi Wang, Yu Wu, Yao Qian, Ken'ichi Kumatani, Shujie Liu, Furu Wei, Michael Zeng, Xuedong Huang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21y.html](http://proceedings.mlr.press/v139/wang21y.html)

**Abstract**:

In this paper, we propose a unified pre-training approach called UniSpeech to learn speech representations with both labeled and unlabeled data, in which supervised phonetic CTC learning and phonetically-aware contrastive self-supervised learning are conducted in a multi-task learning manner. The resultant representations can capture information more correlated with phonetic structures and improve the generalization across languages and domains. We evaluate the effectiveness of UniSpeech for cross-lingual representation learning on public CommonVoice corpus. The results show that UniSpeech outperforms self-supervised pretraining and supervised transfer learning for speech recognition by a maximum of 13.4% and 26.9% relative phone error rate reductions respectively (averaged over all testing languages). The transferability of UniSpeech is also verified on a domain-shift speech recognition task, i.e., a relative word error rate reduction of 6% against the previous approach.

----

## [996] Instabilities of Offline RL with Pre-Trained Neural Representation

**Authors**: *Ruosong Wang, Yifan Wu, Ruslan Salakhutdinov, Sham M. Kakade*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21z.html](http://proceedings.mlr.press/v139/wang21z.html)

**Abstract**:

In offline reinforcement learning (RL), we seek to utilize offline data to evaluate (or learn) policies in scenarios where the data are collected from a distribution that substantially differs from that of the target policy to be evaluated. Recent theoretical advances have shown that such sample-efficient offline RL is indeed possible provided certain strong representational conditions hold, else there are lower bounds exhibiting exponential error amplification (in the problem horizon) unless the data collection distribution has only a mild distribution shift relative to the target policy. This work studies these issues from an empirical perspective to gauge how stable offline RL methods are. In particular, our methodology explores these ideas when using features from pre-trained neural networks, in the hope that these representations are powerful enough to permit sample efficient offline RL. Through extensive experiments on a range of tasks, we see that substantial error amplification does occur even when using such pre-trained representations (trained on the same task itself); we find offline RL is stable only under extremely mild distribution shift. The implications of these results, both from a theoretical and an empirical perspective, are that successful offline RL (where we seek to go beyond the low distribution shift regime) requires substantially stronger conditions beyond those which suffice for successful supervised learning.

----

## [997] Learning to Weight Imperfect Demonstrations

**Authors**: *Yunke Wang, Chang Xu, Bo Du, Honglak Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21aa.html](http://proceedings.mlr.press/v139/wang21aa.html)

**Abstract**:

This paper investigates how to weight imperfect expert demonstrations for generative adversarial imitation learning (GAIL). The agent is expected to perform behaviors demonstrated by experts. But in many applications, experts could also make mistakes and their demonstrations would mislead or slow the learning process of the agent. Recently, existing methods for imitation learning from imperfect demonstrations mostly focus on using the preference or confidence scores to distinguish imperfect demonstrations. However, these auxiliary information needs to be collected with the help of an oracle, which is usually hard and expensive to afford in practice. In contrast, this paper proposes a method of learning to weight imperfect demonstrations in GAIL without imposing extensive prior information. We provide a rigorous mathematical analysis, presenting that the weights of demonstrations can be exactly determined by combining the discriminator and agent policy in GAIL. Theoretical analysis suggests that with the estimated weights the agent can learn a better policy beyond those plain expert demonstrations. Experiments in the Mujoco and Atari environments demonstrate that the proposed algorithm outperforms baseline methods in handling imperfect expert demonstrations.

----

## [998] Evolving Attention with Residual Convolutions

**Authors**: *Yujing Wang, Yaming Yang, Jiangang Bai, Mingliang Zhang, Jing Bai, Jing Yu, Ce Zhang, Gao Huang, Yunhai Tong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21ab.html](http://proceedings.mlr.press/v139/wang21ab.html)

**Abstract**:

Transformer is a ubiquitous model for natural language processing and has attracted wide attentions in computer vision. The attention maps are indispensable for a transformer model to encode the dependencies among input tokens. However, they are learned independently in each layer and sometimes fail to capture precise patterns. In this paper, we propose a novel and generic mechanism based on evolving attention to improve the performance of transformers. On one hand, the attention maps in different layers share common knowledge, thus the ones in preceding layers can instruct the attention in succeeding layers through residual connections. On the other hand, low-level and high-level attentions vary in the level of abstraction, so we adopt convolutional layers to model the evolutionary process of attention maps. The proposed evolving attention mechanism achieves significant performance improvement over various state-of-the-art models for multiple tasks, including image classification, natural language understanding and machine translation.

----

## [999] Guarantees for Tuning the Step Size using a Learning-to-Learn Approach

**Authors**: *Xiang Wang, Shuai Yuan, Chenwei Wu, Rong Ge*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21ac.html](http://proceedings.mlr.press/v139/wang21ac.html)

**Abstract**:

Choosing the right parameters for optimization algorithms is often the key to their success in practice. Solving this problem using a learning-to-learn approach—using meta-gradient descent on a meta-objective based on the trajectory that the optimizer generates—was recently shown to be effective. However, the meta-optimization problem is difficult. In particular, the meta-gradient can often explode/vanish, and the learned optimizer may not have good generalization performance if the meta-objective is not chosen carefully. In this paper we give meta-optimization guarantees for the learning-to-learn approach on a simple problem of tuning the step size for quadratic loss. Our results show that the naïve objective suffers from meta-gradient explosion/vanishing problem. Although there is a way to design the meta-objective so that the meta-gradient remains polynomially bounded, computing the meta-gradient directly using backpropagation leads to numerical issues. We also characterize when it is necessary to compute the meta-objective on a separate validation set to ensure the generalization performance of the learned optimizer. Finally, we verify our results empirically and show that a similar phenomenon appears even for more complicated learned optimizers parametrized by neural networks.

----



[Go to the previous page](ICML-2021-list04.md)

[Go to the next page](ICML-2021-list06.md)

[Go to the catalog section](README.md)