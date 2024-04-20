## [2200] Stateful Strategic Regression

**Authors**: *Keegan Harris, Hoda Heidari, Zhiwei Steven Wu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1404c2624fa7f2507ba04fd9dfc5fb1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1404c2624fa7f2507ba04fd9dfc5fb1-Abstract.html)

**Abstract**:

Automated decision-making tools increasingly assess individuals to determine if they qualify for high-stakes opportunities. A recent line of research investigates how strategic agents may respond to such scoring tools to receive favorable assessments. While prior work has focused on the short-term strategic interactions between a decision-making institution (modeled as a principal) and individual decision-subjects (modeled as agents), we investigate interactions spanning multiple time-steps. In particular, we consider settings in which the agent's effort investment today can accumulate over time in the form of an internal state - impacting both his future rewards and that of the principal. We characterize the Stackelberg equilibrium of the resulting game and provide novel algorithms for computing it. Our analysis reveals several intriguing insights about the role of multiple interactions in shaping the game's outcome: First, we establish that in our stateful setting, the class of all linear assessment policies remains as powerful as the larger class of all monotonic assessment policies. While recovering the principal's optimal policy requires solving a non-convex optimization problem, we provide polynomial-time algorithms for recovering both the principal and agent's optimal policies under common assumptions about the process by which effort investments convert to observable features. Most importantly, we show that with multiple rounds of interaction at her disposal, the principal is more effective at incentivizing the agent to accumulate effort in her desired direction. Our work addresses several critical gaps in the growing literature on the societal impacts of automated decision-making - by focusing on longer time horizons and accounting for the compounding nature of decisions individuals receive over time.

----

## [2201] Self-Attention Between Datapoints: Going Beyond Individual Input-Output Pairs in Deep Learning

**Authors**: *Jannik Kossen, Neil Band, Clare Lyle, Aidan N. Gomez, Thomas Rainforth, Yarin Gal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1507aba9fc82ffa7cc7373c58f8a613-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1507aba9fc82ffa7cc7373c58f8a613-Abstract.html)

**Abstract**:

We challenge a common assumption underlying most supervised deep learning: that a model makes a prediction depending only on its parameters and the features of a single input. To this end, we introduce a general-purpose deep learning architecture that takes as input the entire dataset instead of processing one datapoint at a time. Our approach uses self-attention to reason about relationships between datapoints explicitly, which can be seen as realizing non-parametric models using parametric attention mechanisms. However, unlike conventional non-parametric models, we let the model learn end-to-end from the data how to make use of other datapoints for prediction. Empirically, our models solve cross-datapoint lookup and complex reasoning tasks unsolvable by traditional deep learning models. We show highly competitive results on tabular data, early results on CIFAR-10, and give insight into how the model makes use of the interactions between points.

----

## [2202] Your head is there to move you around: Goal-driven models of the primate dorsal pathway

**Authors**: *Patrick J. Mineault, Shahab Bakhtiari, Blake A. Richards, Christopher C. Pack*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1676935f9304b97d59b0738289d2e22-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1676935f9304b97d59b0738289d2e22-Abstract.html)

**Abstract**:

Neurons in the dorsal visual pathway of the mammalian brain are selective for motion stimuli, with the complexity of stimulus representations increasing along the hierarchy. This progression is similar to that of the ventral visual pathway, which is well characterized by artificial neural networks (ANNs) optimized for object recognition. In contrast, there are no image-computable models of the dorsal stream with comparable explanatory power. We hypothesized that the properties of dorsal stream neurons could be explained by a simple learning objective: the need for an organism to orient itself during self-motion. To test this hypothesis, we trained a 3D ResNet to predict an agent's self-motion parameters from visual stimuli in a simulated environment. We found that the responses in this network accounted well for the selectivity of neurons in a large database of single-neuron recordings from the dorsal visual stream of non-human primates. In contrast, ANNs trained on an action recognition dataset through supervised or self-supervised learning  could not explain responses in the dorsal stream, despite also being trained on naturalistic videos with moving objects. These results demonstrate that an ecologically relevant cost function can account for dorsal stream properties in the primate brain.

----

## [2203] Achieving Rotational Invariance with Bessel-Convolutional Neural Networks

**Authors**: *Valentin Delchevalerie, Adrien Bibal, Benoît Frénay, Alexandre Mayer*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f18224a1adfb7b3dbff668c9b655a35a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f18224a1adfb7b3dbff668c9b655a35a-Abstract.html)

**Abstract**:

For many applications in image analysis, learning models that are invariant to translations and rotations is paramount. This is the case, for example, in medical imaging where the objects of interest can appear at arbitrary positions, with arbitrary orientations. As of today, Convolutional Neural Networks (CNN) are one of the most powerful tools for image analysis. They achieve, thanks to convolutions, an invariance with respect to translations. In this work, we present a new type of convolutional layer that takes advantage of Bessel functions, well known in physics, to build Bessel-CNNs (B-CNNs) that are invariant to all the continuous set of possible rotation angles by design.

----

## [2204] Unsupervised Domain Adaptation with Dynamics-Aware Rewards in Reinforcement Learning

**Authors**: *Jinxin Liu, Hao Shen, Donglin Wang, Yachen Kang, Qiangxing Tian*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f187a23c3ee681ef6913f31fd6d6446b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f187a23c3ee681ef6913f31fd6d6446b-Abstract.html)

**Abstract**:

Unsupervised reinforcement learning aims to acquire skills without prior goal representations, where an agent automatically explores an open-ended environment to represent goals and learn the goal-conditioned policy. However, this procedure is often time-consuming, limiting the rollout in some potentially expensive target environments. The intuitive approach of training in another interaction-rich environment disrupts the reproducibility of trained skills in the target environment due to the dynamics shifts and thus inhibits direct transferring. Assuming free access to a source environment, we propose an unsupervised domain adaptation method to identify and acquire skills across dynamics. Particularly, we introduce a KL regularized objective to encourage emergence of skills, rewarding the agent for both discovering skills and aligning its behaviors respecting dynamics shifts. This suggests that both dynamics (source and target) shape the reward to facilitate the learning of adaptive skills. We also conduct empirical experiments to demonstrate that our method can effectively learn skills that can be smoothly deployed in target.

----

## [2205] GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph

**Authors**: *Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, Xing Xie*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f18a6d1cde4b205199de8729a6637b42-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f18a6d1cde4b205199de8729a6637b42-Abstract.html)

**Abstract**:

The representation learning on textual graph is to generate low-dimensional embeddings for the nodes based on the individual textual features and the neighbourhood information. Recent breakthroughs on pretrained language models and graph neural networks push forward the development of corresponding techniques. The existing works mainly rely on the cascaded model architecture: the textual features of nodes are independently encoded by language models at first; the textual embeddings are aggregated by graph neural networks afterwards. However, the above architecture is limited due to the independent modeling of textual features. In this work, we propose GraphFormers, where layerwise GNN components are nested alongside the transformer blocks of language models. With the proposed architecture, the text encoding and the graph aggregation are fused into an iterative workflow, making each node's semantic accurately comprehended from the global perspective. In addition, a progressive learning strategy is introduced, where the model is successively trained on manipulated data and original data to reinforce its capability of integrating information on graph. Extensive evaluations are conducted on three large-scale benchmark datasets, where GraphFormers outperform the SOTA baselines with comparable running efficiency. The source code is released at https://github.com/microsoft/GraphFormers .

----

## [2206] A Universal Law of Robustness via Isoperimetry

**Authors**: *Sébastien Bubeck, Mark Sellke*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f197002b9a0853eca5e046d9ca4663d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f197002b9a0853eca5e046d9ca4663d5-Abstract.html)

**Abstract**:

Classically, data interpolation with a parametrized model class is possible as long as the number of parameters is larger than the number of equations to be satisfied. A puzzling phenomenon in the current practice of deep learning is that models are trained with many more parameters than what this classical theory would suggest. We propose a theoretical explanation for this phenomenon. We prove that for a broad class of data distributions and model classes, overparametrization is {\em necessary} if one wants to interpolate the data {\em smoothly}. Namely we show that {\em smooth} interpolation requires $d$ times more parameters than mere interpolation, where $d$ is the ambient data dimension. We prove this universal law of robustness for any smoothly parametrized function class with polynomial size weights, and any covariate distribution verifying isoperimetry. In the case of two-layers neural networks and Gaussian covariates, this law was conjectured in prior work by Bubeck, Li and Nagaraj. We also give an interpretation of our result as an improved generalization bound for model classes consisting of smooth functions.

----

## [2207] On Contrastive Representations of Stochastic Processes

**Authors**: *Emile Mathieu, Adam Foster, Yee Whye Teh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f19c44d068fecac1d6d13a80df4f8e96-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f19c44d068fecac1d6d13a80df4f8e96-Abstract.html)

**Abstract**:

Learning representations of stochastic processes is an emerging problem in machine learning with applications from meta-learning to physical object models to time series. Typical methods rely on exact reconstruction of observations, but this approach breaks down as observations become high-dimensional or noise distributions become complex. To address this, we propose a unifying framework for learning contrastive representations of stochastic processes (CReSP) that does away with exact reconstruction. We dissect potential use cases for stochastic process representations, and propose methods that accommodate each. Empirically, we show that our methods are effective for learning representations of periodic functions, 3D objects and dynamical processes. Our methods tolerate noisy high-dimensional observations better than traditional approaches, and the learned representations transfer to a range of downstream tasks.

----

## [2208] A Domain-Shrinking based Bayesian Optimization Algorithm with Order-Optimal Regret Performance

**Authors**: *Sudeep Salgia, Sattar Vakili, Qing Zhao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f19fec2f129fbdba76493451275c883a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f19fec2f129fbdba76493451275c883a-Abstract.html)

**Abstract**:

We consider sequential optimization of an unknown function in a reproducing kernel Hilbert space. We propose a Gaussian process-based algorithm and establish its order-optimal regret performance (up to a poly-logarithmic factor). This is the first GP-based algorithm with an order-optimal regret guarantee. The proposed algorithm is rooted in the methodology of domain shrinking realized through a sequence of tree-based region pruning and refining to concentrate queries in increasingly smaller high-performing regions of the function domain. The search for high-performing regions is localized and guided by an iterative estimation of the optimal function value to ensure both learning efficiency and computational efficiency. Compared with the prevailing GP-UCB family of algorithms, the proposed algorithm reduces computational complexity by a factor of $O(T^{2d-1})$ (where $T$ is the time horizon and $d$ the dimension of the function domain).

----

## [2209] Scalars are universal: Equivariant machine learning, structured like classical physics

**Authors**: *Soledad Villar, David W. Hogg, Kate Storey-Fisher, Weichi Yao, Ben Blum-Smith*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1b0775946bc0329b35b823b86eeb5f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1b0775946bc0329b35b823b86eeb5f5-Abstract.html)

**Abstract**:

There has been enormous progress in the last few years in designing  neural networks that respect the fundamental symmetries and coordinate freedoms of physical law. Some of these frameworks make use of irreducible representations, some make use of high-order tensor objects, and some apply symmetry-enforcing constraints. Different physical laws obey different combinations of fundamental symmetries, but a large fraction (possibly all) of classical physics is equivariant to translation, rotation, reflection (parity), boost (relativity), and permutations. Here we show that it is simple to parameterize universally approximating polynomial functions that are equivariant under these symmetries, or under the Euclidean, Lorentz, and Poincar√© groups, at any dimensionality $d$. The key observation is that nonlinear O($d$)-equivariant (and related-group-equivariant) functions can be universally expressed in terms of a lightweight collection of scalars---scalar products and scalar contractions of the scalar, vector, and tensor inputs. We complement our theory with numerical examples that show that the scalar-based method is simple, efficient, and scalable.

----

## [2210] Unsupervised Object-Level Representation Learning from Scene Images

**Authors**: *Jiahao Xie, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, Chen Change Loy*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1b6f2857fb6d44dd73c7041e0aa0f19-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1b6f2857fb6d44dd73c7041e0aa0f19-Abstract.html)

**Abstract**:

Contrastive self-supervised learning has largely narrowed the gap to supervised pre-training on ImageNet. However, its success highly relies on the object-centric priors of ImageNet, i.e., different augmented views of the same image correspond to the same object. Such a heavily curated constraint becomes immediately infeasible when pre-trained on more complex scene images with many objects. To overcome this limitation, we introduce Object-level Representation Learning (ORL), a new self-supervised learning framework towards scene images. Our key insight is to leverage image-level self-supervised pre-training as the prior to discover object-level semantic correspondence, thus realizing object-level representation learning from scene images. Extensive experiments on COCO show that ORL significantly improves the performance of self-supervised learning on scene images, even surpassing supervised ImageNet pre-training on several downstream tasks. Furthermore, ORL improves the downstream performance when more unlabeled scene images are available, demonstrating its great potential of harnessing unlabeled data in the wild. We hope our approach can motivate future research on more general-purpose unsupervised representation learning from scene data.

----

## [2211] Do Transformers Really Perform Badly for Graph Representation?

**Authors**: *Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html)

**Abstract**:

The Transformer architecture has become a dominant choice in many domains, such as natural language processing and computer vision. Yet, it has not achieved competitive performance on popular leaderboards of graph-level prediction compared to mainstream GNN variants. Therefore, it remains a mystery how Transformers could perform well for graph representation learning. In this paper, we solve this mystery by presenting Graphormer, which is built upon the standard Transformer architecture, and could attain excellent results on a broad range of graph representation learning tasks, especially on the recent OGB Large-Scale Challenge. Our key insight to utilizing Transformer in the graph is the necessity of effectively encoding the structural information of a graph into the model. To this end, we propose several simple yet effective structural encoding methods to help Graphormer better model graph-structured data. Besides, we mathematically characterize the expressive power of Graphormer and exhibit that with our ways of encoding the structural information of graphs, many popular GNN variants could be covered as the special cases of Graphormer. The code and models of Graphormer will be made publicly available at \url{https://github.com/Microsoft/Graphormer}.

----

## [2212] Powerpropagation: A sparsity inducing weight reparameterisation

**Authors**: *Jonathan Schwarz, Siddhant M. Jayakumar, Razvan Pascanu, Peter E. Latham, Yee Whye Teh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f1e709e6aef16ba2f0cd6c7e4f52b9b6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f1e709e6aef16ba2f0cd6c7e4f52b9b6-Abstract.html)

**Abstract**:

The training of sparse neural networks is becoming an increasingly important tool for reducing the computational footprint of models at training and evaluation, as well enabling the effective scaling up of models. Whereas much work over the years has been dedicated to specialised pruning techniques, little attention has been paid to the inherent effect of gradient based training on model sparsity. Inthis work, we introduce Powerpropagation, a new weight-parameterisation for neural networks that leads to inherently sparse models. Exploiting the behaviour of gradient descent, our method gives rise to weight updates exhibiting a “rich get richer” dynamic, leaving low-magnitude parameters largely unaffected by learning. Models trained in this manner exhibit similar performance, but have a distributionwith markedly higher density at zero, allowing more parameters to be pruned safely. Powerpropagation is general, intuitive, cheap and straight-forward to implement and can readily be combined with various other techniques. To highlight its versatility, we explore it in two very different settings: Firstly, following a recent line of work, we investigate its effect on sparse training for resource-constrained settings. Here, we combine Powerpropagation with a traditional weight-pruning technique as well as recent state-of-the-art sparse-to-sparse algorithms, showing superior performance on the ImageNet benchmark. Secondly, we advocate the useof sparsity in overcoming catastrophic forgetting, where compressed representations allow accommodating a large number of tasks at fixed model capacity. In all cases our reparameterisation considerably increases the efficacy of the off-the-shelf methods.

----

## [2213] Stronger NAS with Weaker Predictors

**Authors**: *Junru Wu, Xiyang Dai, Dongdong Chen, Yinpeng Chen, Mengchen Liu, Ye Yu, Zhangyang Wang, Zicheng Liu, Mei Chen, Lu Yuan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)

**Abstract**:

Neural Architecture Search (NAS) often trains and evaluates a large number of architectures. Recent predictor-based NAS approaches attempt to alleviate such heavy computation costs with two key steps: sampling some architecture-performance pairs and fitting a proxy accuracy predictor. Given limited samples, these predictors, however, are far from accurate to locate top architectures due to the difficulty of fitting the huge search space. This paper reflects on a simple yet crucial question: if our final goal is to find the best architecture, do we really need to model the whole space well?. We propose a paradigm shift from fitting the whole architecture space using one strong predictor, to progressively fitting a search path towards the high-performance sub-space through a set of weaker predictors. As a key property of the weak predictors, their probabilities of sampling better architectures keep increasing. Hence we only sample a few well-performed architectures guided by the previously learned predictor and estimate a new better weak predictor. This embarrassingly easy framework, dubbed WeakNAS, produces coarse-to-fine iteration to gradually refine the ranking of sampling space. Extensive experiments demonstrate that WeakNAS costs fewer samples to find top-performance architectures on NAS-Bench-101 and NAS-Bench-201. Compared to state-of-the-art (SOTA) predictor-based NAS methods, WeakNAS outperforms all with notable margins, e.g., requiring at least 7.5x less samples to find global optimal on NAS-Bench-101. WeakNAS can also absorb their ideas to boost performance more. Further, WeakNAS strikes the new SOTA result of 81.3% in the ImageNet MobileNet Search Space. The code is available at: https://github.com/VITA-Group/WeakNAS.

----

## [2214] Convolutional Normalization: Improving Deep Convolutional Network Robustness and Training

**Authors**: *Sheng Liu, Xiao Li, Yuexiang Zhai, Chong You, Zhihui Zhu, Carlos Fernandez-Granda, Qing Qu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f23d125da1e29e34c552f448610ff25f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f23d125da1e29e34c552f448610ff25f-Abstract.html)

**Abstract**:

Normalization techniques have become a basic component in modern convolutional neural networks (ConvNets). In particular, many recent works demonstrate that promoting the orthogonality of the weights helps train deep models and improve robustness. For ConvNets, most existing methods are based on penalizing or normalizing weight matrices derived from concatenating or flattening the convolutional kernels. These methods often destroy or ignore the benign convolutional structure of the kernels; therefore, they are often expensive or impractical for deep ConvNets. In contrast, we introduce a simple and efficient ``Convolutional Normalization'' (ConvNorm) method that can fully exploit the convolutional structure in the Fourier domain and serve as a simple plug-and-play module to be conveniently incorporated into any ConvNets. Our method is inspired by recent work on preconditioning methods for convolutional sparse coding and can effectively promote each layer's channel-wise isometry. Furthermore, we show that our ConvNorm can reduce the layerwise spectral norm of the weight matrices and hence improve the Lipschitzness of the network, leading to easier training and improved robustness for deep ConvNets. Applied to classification under noise corruptions and generative adversarial network (GAN), we show that the ConvNorm improves the robustness of common ConvNets such as ResNet and the performance of GAN. We verify our findings via numerical experiments on CIFAR and ImageNet. Our implementation is available online at \url{https://github.com/shengliu66/ConvNorm}.

----

## [2215] Nearly-Tight and Oblivious Algorithms for Explainable Clustering

**Authors**: *Buddhima Gamlath, Xinrui Jia, Adam Polak, Ola Svensson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f24ad6f72d6cc4cb51464f2b29ab69d3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f24ad6f72d6cc4cb51464f2b29ab69d3-Abstract.html)

**Abstract**:

We study the problem of explainable clustering in the setting first formalized by Dasgupta, Frost, Moshkovitz, and Rashtchian (ICML 2020). A $k$-clustering is said to be explainable if it is given by a decision tree where each internal node splits data points with a threshold cut in a single dimension (feature), and each of the $k$ leaves corresponds to a cluster. We give an algorithm that outputs an explainable clustering that loses at most a factor of $O(\log^2 k)$ compared to an optimal (not necessarily explainable) clustering for the $k$-medians objective, and a factor of $O(k \log^2 k)$ for the $k$-means objective. This improves over the previous best upper bounds of $O(k)$ and $O(k^2)$, respectively, and nearly matches the previous $\Omega(\log k)$ lower bound for $k$-medians and our new $\Omega(k)$ lower bound for $k$-means. The algorithm is remarkably simple. In particular, given an initial not necessarily explainable clustering in $\mathbb{R}^d$, it is oblivious to the data points and runs in time $O(dk \log^2 k)$, independent of the number of data points $n$. Our upper and lower bounds also generalize to objectives given by higher $\ell_p$-norms.

----

## [2216] Deep Networks Provably Classify Data on Curves

**Authors**: *Tingran Wang, Sam Buchanan, Dar Gilboa, John Wright*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f26df67e8110ee2b44923db775e3e47f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f26df67e8110ee2b44923db775e3e47f-Abstract.html)

**Abstract**:

Data with low-dimensional nonlinear structure are ubiquitous in engineering and scientific problems. We study a model problem with such structure---a binary classification task that uses a deep fully-connected neural network to classify data drawn from two disjoint smooth curves on the unit sphere. Aside from mild regularity conditions, we place no restrictions on the configuration of the curves. We prove that when (i) the network depth is large relative to certain geometric properties that set the difficulty of the problem and (ii) the network width and number of samples is polynomial in the depth, randomly-initialized gradient descent quickly learns to correctly classify all points on the two curves with high probability. To our knowledge, this is the first generalization guarantee for deep networks with nonlinear data that depends only on intrinsic data properties. Our analysis proceeds by a reduction to dynamics in the neural tangent kernel (NTK) regime, where the network depth plays the role of a fitting resource in solving the classification problem. In particular, via fine-grained control of the decay properties of the NTK, we demonstrate that when the network is sufficiently deep, the NTK can be locally approximated by a translationally invariant operator on the manifolds and stably inverted over smooth functions, which guarantees convergence and generalization.

----

## [2217] COMBO: Conservative Offline Model-Based Policy Optimization

**Authors**: *Tianhe Yu, Aviral Kumar, Rafael Rafailov, Aravind Rajeswaran, Sergey Levine, Chelsea Finn*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html)

**Abstract**:

Model-based reinforcement learning (RL) algorithms, which learn a dynamics model from logged experience and perform conservative planning under the learned model, have emerged as a promising paradigm for offline reinforcement learning (offline RL). However, practical variants of such model-based algorithms rely on explicit uncertainty quantification for incorporating conservatism. Uncertainty estimation with complex models, such as deep neural networks, can be difficult and unreliable. We empirically find that uncertainty estimation is not accurate and leads to poor performance in certain scenarios in offline model-based RL. We overcome this limitation by developing a new model-based offline RL algorithm, COMBO, that trains a value function using both the offline dataset and data generated using rollouts under the model while also additionally regularizing the value function on out-of-support state-action tuples generated via model rollouts. This results in a conservative estimate of the value function for out-of-support state-action tuples, without requiring explicit uncertainty estimation. Theoretically, we show that COMBO satisfies a policy improvement guarantee in the offline setting. Through extensive experiments, we find that COMBO attains greater performance compared to prior offline RL on problems that demand generalization to related but previously unseen tasks, and also consistently matches or outperforms prior offline RL methods on widely studied offline RL benchmarks, including image-based tasks.

----

## [2218] Time-series Generation by Contrastive Imitation

**Authors**: *Daniel Jarrett, Ioana Bica, Mihaela van der Schaar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f2b4053221961416d47d497814a8064f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f2b4053221961416d47d497814a8064f-Abstract.html)

**Abstract**:

Consider learning a generative model for time-series data. The sequential setting poses a unique challenge: Not only should the generator capture the conditional dynamics of (stepwise) transitions, but its open-loop rollouts should also preserve the joint distribution of (multi-step) trajectories. On one hand, autoregressive models trained by MLE allow learning and computing explicit transition distributions, but suffer from compounding error during rollouts. On the other hand, adversarial models based on GAN training alleviate such exposure bias, but transitions are implicit and hard to assess. In this work, we study a generative framework that seeks to combine the strengths of both: Motivated by a moment-matching objective to mitigate compounding error, we optimize a local (but forward-looking) transition policy, where the reinforcement signal is provided by a global (but stepwise-decomposable) energy model trained by contrastive estimation. At training, the two components are learned cooperatively, avoiding the instabilities typical of adversarial objectives. At inference, the learned policy serves as the generator for iterative sampling, and the learned energy serves as a trajectory-level measure for evaluating sample quality. By expressly training a policy to imitate sequential behavior of time-series features in a dataset, this approach embodies "generation by imitation". Theoretically, we illustrate the correctness of this formulation and the consistency of the algorithm. Empirically, we evaluate its ability to generate predictively useful samples from real-world datasets, verifying that it performs at the standard of existing benchmarks.

----

## [2219] Differentially Private Sampling from Distributions

**Authors**: *Sofya Raskhodnikova, Satchit Sivakumar, Adam D. Smith, Marika Swanberg*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f2b5e92f61b6de923b063588ee6e7c48-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f2b5e92f61b6de923b063588ee6e7c48-Abstract.html)

**Abstract**:

We initiate an investigation of  private sampling from distributions. Given a dataset with $n$ independent observations from an unknown distribution $P$, a sampling algorithm must output a single observation from a distribution that is close in total variation distance to $P$ while satisfying differential privacy. Sampling abstracts the goal of generating small amounts of realistic-looking data. We provide tight upper and lower bounds for the dataset size needed for this task for three natural families of distributions: arbitrary distributions on $\{1,\ldots ,k\}$, arbitrary product distributions on $\{0,1\}^d$, and product distributions on on $\{0,1\}^d$ with bias in each coordinate bounded away from 0 and 1. We demonstrate that, in some parameter regimes, private sampling requires asymptotically fewer observations than learning a description of $P$ nonprivately; in other regimes, however, private sampling proves to be as difficult as private learning. Notably, for some classes of distributions, the overhead in the number of observations needed for private learning compared to non-private learning is completely captured by the number of observations needed for private sampling.

----

## [2220] On the Expected Complexity of Maxout Networks

**Authors**: *Hanna Tseran, Guido Montúfar*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f2c3b258e9cd8ba16e18f319b3c88c66-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f2c3b258e9cd8ba16e18f319b3c88c66-Abstract.html)

**Abstract**:

Learning with neural networks relies on the complexity of their representable functions, but more importantly, their particular assignment of typical parameters to functions of different complexity. Taking the number of activation regions as a complexity measure, recent works have shown that the practical complexity of deep ReLU networks is often far from the theoretical maximum. In this work, we show that this phenomenon also occurs in networks with maxout (multi-argument) activation functions and when considering the decision boundaries in classification tasks. We also show that the parameter space has a multitude of full-dimensional regions with widely different complexity, and obtain nontrivial lower bounds on the expected complexity. Finally, we investigate different parameter initialization procedures and show that they can increase the speed of convergence in training.

----

## [2221] Cross-view Geo-localization with Layer-to-Layer Transformer

**Authors**: *Hongji Yang, Xiufan Lu, Yingying Zhu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f31b20466ae89669f9741e047487eb37-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f31b20466ae89669f9741e047487eb37-Abstract.html)

**Abstract**:

In this work, we address the problem of cross-view geo-localization, which estimates the geospatial location of a street view image by matching it with a database of geo-tagged aerial images. The cross-view matching task is extremely challenging due to drastic appearance and geometry differences across views. Unlike existing methods that predominantly fall back on CNN, here we devise a novel layer-to-layer Transformer (L2LTR) that utilizes the properties of self-attention in Transformer to model global dependencies, thus significantly decreasing visual ambiguities in cross-view geo-localization. We also exploit the positional encoding of the Transformer to help the L2LTR understand and correspond geometric configurations between ground and aerial images. Compared to state-of-the-art methods that impose strong assumptions on geometry knowledge, the L2LTR flexibly learns the positional embeddings through the training objective. It hence becomes more practical in many real-world scenarios. Although Transformer is well suited to our task, its vanilla self-attention mechanism independently interacts within image patches in each layer, which overlooks correlations between layers. Instead, this paper proposes a simple yet effective self-cross attention mechanism to improve the quality of learned representations. Self-cross attention models global dependencies between adjacent layers and creates short paths for effective information flow. As a result, the proposed self-cross attention leads to more stable training, improves the generalization ability, and prevents the learned intermediate features from being overly similar. Extensive experiments demonstrate that our L2LTR performs favorably against state-of-the-art methods on standard, fine-grained, and cross-dataset cross-view geo-localization tasks. The code is available online.

----

## [2222] TAAC: Temporally Abstract Actor-Critic for Continuous Control

**Authors**: *Haonan Yu, Wei Xu, Haichao Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f337d999d9ad116a7b4f3d409fcc6480-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f337d999d9ad116a7b4f3d409fcc6480-Abstract.html)

**Abstract**:

We present temporally abstract actor-critic (TAAC), a simple but effective off-policy RL algorithm that incorporates closed-loop temporal abstraction into the actor-critic framework. TAAC adds a second-stage binary policy to choose between the previous action and a new action output by an actor. Crucially, its "act-or-repeat" decision hinges on the actually sampled action instead of the expected behavior of the actor. This post-acting switching scheme let the overall policy make more informed decisions. TAAC has two important features: a) persistent exploration, and b) a new compare-through Q operator for multi-step TD backup, specially tailored to the action repetition scenario. We demonstrate TAAC's advantages over several strong baselines across 14 continuous control tasks. Our surprising finding reveals that while achieving top performance, TAAC is able to "mine" a significant number of repeated actions with the trained policy even on continuous tasks whose problem structures on the surface seem to repel action repetition. This suggests that aside from encouraging persistent exploration, action repetition can find its place in a good policy behavior. Code is available at https://github.com/hnyu/taac.

----

## [2223] Learning Robust Hierarchical Patterns of Human Brain across Many fMRI Studies

**Authors**: *Dushyant Sahoo, Christos Davatzikos*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f33ba15effa5c10e873bf3842afb46a6-Abstract.html)

**Abstract**:

Multi-site fMRI studies face the challenge that the pooling introduces systematic non-biological site-specific variance due to hardware, software, and environment. In this paper, we propose to reduce site-specific variance in the estimation of hierarchical Sparsity Connectivity Patterns (hSCPs) in fMRI data via a simple yet effective matrix factorization while preserving biologically relevant variations. Our method leverages unsupervised adversarial learning to improve the reproducibility of the components. Experiments on simulated datasets display that the proposed method can estimate components with higher accuracy and reproducibility, while preserving age-related variation on a multi-center clinical data set.

----

## [2224] Global Convergence to Local Minmax Equilibrium in Classes of Nonconvex Zero-Sum Games

**Authors**: *Tanner Fiez, Lillian J. Ratliff, Eric Mazumdar, Evan Faulkner, Adhyyan Narang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3507289cfdc8c9ae93f4098111a13f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3507289cfdc8c9ae93f4098111a13f9-Abstract.html)

**Abstract**:

We study gradient descent-ascent learning dynamics with timescale separation ($\tau$-GDA) in unconstrained continuous action zero-sum games where the minimizing player faces a nonconvex optimization problem and the maximizing player optimizes a Polyak-Lojasiewicz (PL) or strongly-concave (SC) objective. In contrast to past work on gradient-based learning in nonconvex-PL/SC zero-sum games, we assess convergence in relation to natural game-theoretic equilibria instead of only notions of stationarity. In pursuit of this goal, we prove that the only locally stable points of the $\tau$-GDA continuous-time limiting system correspond to strict local minmax equilibria in each class of games. For these classes of games, we exploit timescale separation to construct a potential function that when combined with the stability characterization and an asymptotic saddle avoidance result gives a global asymptotic almost-sure convergence guarantee for the discrete-time gradient descent-ascent update to a set of the strict local minmax equilibrium. Moreover, we provide convergence rates for the gradient descent-ascent dynamics with timescale separation to approximate stationary points.

----

## [2225] Bandit Quickest Changepoint Detection

**Authors**: *Aditya Gopalan, Braghadeesh Lakshminarayanan, Venkatesh Saligrama*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html)

**Abstract**:

Many industrial and security applications employ a suite of sensors for detecting abrupt changes in temporal behavior patterns. These abrupt changes typically manifest locally, rendering only a small subset of sensors informative. Continuous monitoring of every sensor can be expensive due to resource constraints, and serves as a motivation for the bandit quickest changepoint detection problem, where sensing actions (or sensors) are sequentially chosen, and only measurements corresponding to chosen actions are observed. We derive an information-theoretic lower bound on the detection delay for a general class of finitely parameterized probability distributions. We then propose a computationally efficient online sensing scheme, which seamlessly balances the need for exploration of different sensing options with exploitation of querying informative actions. We derive expected delay bounds for the proposed scheme and show that these bounds match our information-theoretic lower bounds at low false alarm rates, establishing optimality of the proposed method. We then perform a number of experiments on synthetic and real datasets demonstrating the effectiveness of our proposed method.

----

## [2226] Can multi-label classification networks know what they don't know?

**Authors**: *Haoran Wang, Weitang Liu, Alex Bocchieri, Yixuan Li*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3b7e5d3eb074cde5b76e26bc0fb5776-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3b7e5d3eb074cde5b76e26bc0fb5776-Abstract.html)

**Abstract**:

Estimating out-of-distribution (OOD) uncertainty is a major challenge for safely deploying machine learning models in the open-world environment. Improved methods for OOD detection in multi-class classification have emerged, while OOD detection methods for multi-label classification remain underexplored and use rudimentary techniques. We propose JointEnergy, a simple and effective method, which estimates the OOD indicator scores by aggregating label-wise energy scores from multiple labels. We show that JointEnergy can be mathematically interpreted from a joint likelihood perspective. Our results show consistent improvement over previous methods that are based on the maximum-valued scores, which fail to capture joint information from multiple labels. We demonstrate the effectiveness of our method on three common multi-label classification benchmarks, including MS-COCO, PASCAL-VOC, and NUS-WIDE. We show that JointEnergy can reduce the FPR95 by up to 10.05% compared to the previous best baseline, establishing state-of-the-art performance.

----

## [2227] Balanced Chamfer Distance as a Comprehensive Metric for Point Cloud Completion

**Authors**: *Tong Wu, Liang Pan, Junzhe Zhang, Tai Wang, Ziwei Liu, Dahua Lin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3bd5ad57c8389a8a1a541a76be463bf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3bd5ad57c8389a8a1a541a76be463bf-Abstract.html)

**Abstract**:

Chamfer Distance (CD) and Earth Moverâ€™s Distance (EMD) are two broadly adopted metrics for measuring the similarity between two point sets. However, CD is usually insensitive to mismatched local density, and EMD is usually dominated by global distribution while overlooks the fidelity of detailed structures. Besides, their unbounded value range induces a heavy influence from the outliers. These defects prevent them from providing a consistent evaluation. To tackle these problems, we propose a new similarity measure named Density-aware Chamfer Distance (DCD). It is derived from CD and benefits from several desirable properties: 1) it can detect disparity of density distributions and is thus a more intensive measure of similarity compared to CD; 2) it is stricter with detailed structures and significantly more computationally efficient than EMD; 3) the bounded value range encourages a more stable and reasonable evaluation over the whole test set. We adopt DCD to evaluate the point cloud completion task, where experimental results show that DCD pays attention to both the overall structure and local geometric details and provides a more reliable evaluation even when CD and EMD contradict each other. We can also use DCD as the training loss, which outperforms the same model trained with CD loss on all three metrics. In addition, we propose a novel point discriminator module that estimates the priority for another guided down-sampling step, and it achieves noticeable improvements under DCD together with competitive results for both CD and EMD. We hope our work could pave the way for a more comprehensive and practical point cloud similarity evaluation. Our code will be available at https://github.com/wutong16/DensityawareChamfer_Distance.

----

## [2228] Optimal Gradient-based Algorithms for Non-concave Bandit Optimization

**Authors**: *Baihe Huang, Kaixuan Huang, Sham M. Kakade, Jason D. Lee, Qi Lei, Runzhe Wang, Jiaqi Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3d9de86462c28781cbe5c47ef22c3e5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3d9de86462c28781cbe5c47ef22c3e5-Abstract.html)

**Abstract**:

Bandit problems with linear or concave reward have been extensively studied, but relatively few works have studied bandits with non-concave reward. This work considers a large family of bandit problems where the unknown underlying reward function is non-concave, including the low-rank generalized linear bandit problems and two-layer neural network with polynomial activation bandit problem.For the low-rank generalized linear bandit problem, we provide a minimax-optimal algorithm in the dimension, refuting both conjectures in \cite{lu2021low,jun2019bilinear}. Our algorithms are based on a unified zeroth-order optimization paradigm that applies in great generality and attains optimal rates in several structured polynomial settings (in the dimension). We further demonstrate the applicability of our algorithms in RL in the generative model setting, resulting in improved sample complexity over prior approaches.Finally, we show that the standard optimistic algorithms (e.g., UCB) are sub-optimal by dimension factors. In the neural net setting (with polynomial activation functions) with noiseless reward, we provide a bandit algorithm with sample complexity equal to the intrinsic algebraic dimension. Again, we show that optimistic approaches have worse sample complexity, polynomial in the extrinsic dimension (which could be exponentially worse in the polynomial degree).

----

## [2229] On Optimal Interpolation in Linear Regression

**Authors**: *Eduard Oravkin, Patrick Rebeschini*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3e0eb8f4ae5f3afd35b5e4b6e5a2d78-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3e0eb8f4ae5f3afd35b5e4b6e5a2d78-Abstract.html)

**Abstract**:

Understanding when and why interpolating methods generalize well has recently been a topic of interest in statistical learning theory. However, systematically connecting interpolating methods to achievable notions of optimality has only received partial attention. In this paper, we ask the question of what is the optimal way to interpolate in linear regression using functions that are linear in the response variable (as the case for the Bayes optimal estimator in ridge regression) and depend on the data, the population covariance of the data, the signal-to-noise ratio and the covariance of the prior for the signal, but do not depend on the value of the signal itself nor the noise vector in the training data. We provide a closed-form expression for the interpolator that achieves this notion of optimality and show that it can be derived as the limit of preconditioned gradient descent with a specific initialization. We identify a regime where the minimum-norm interpolator provably generalizes arbitrarily worse than the optimal response-linear achievable interpolator that we introduce, and validate with numerical experiments that the notion of optimality we consider can be achieved by interpolating methods that only use the training data as input in the case of an isotropic prior. Finally, we extend the notion of optimal response-linear interpolation to random features regression under a linear data-generating model.

----

## [2230] Differentiable Optimization of Generalized Nondecomposable Functions using Linear Programs

**Authors**: *Zihang Meng, Lopamudra Mukherjee, Yichao Wu, Vikas Singh, Sathya N. Ravi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3f1b7fc5a8779a9e618e1f23a7b7860-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3f1b7fc5a8779a9e618e1f23a7b7860-Abstract.html)

**Abstract**:

We propose a framework which makes it feasible to directly train deep neural networks with respect to popular families of task-specific non-decomposable performance measures such as AUC, multi-class AUC, $F$-measure and others. A common feature of the optimization model that emerges from these tasks is that it involves solving a Linear Programs (LP) during training where representations learned by upstream layers characterize the constraints or the feasible set. The constraint matrix is not only large but the constraints are also modified at each iteration. We show how adopting a set of ingenious ideas proposed by Mangasarian for 1-norm SVMs -- which advocates for solving LPs with a generalized Newton method -- provides a simple and effective solution that can be run on the GPU. In particular, this strategy needs little unrolling, which makes it more efficient during backward pass. Further, even when the constraint matrix is too large to fit on the GPU memory (say large minibatch settings), we show that running the Newton method in a lower dimensional space yields accurate gradients for training, by utilizing a statistical concept called {\em sufficient}  dimension reduction. While a number of specialized algorithms have been proposed for the models that we describe here, our module turns out to be applicable without any  specific adjustments or relaxations. We describe each use case, study its properties and demonstrate the efficacy of the approach over alternatives which use surrogate lower bounds and often, specialized optimization schemes. Frequently, we achieve superior computational behavior and performance improvements on common datasets used in the literature.

----

## [2231] Towards Understanding Cooperative Multi-Agent Q-Learning with Value Factorization

**Authors**: *Jianhao Wang, Zhizhou Ren, Beining Han, Jianing Ye, Chongjie Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f3f1fa1e4348bfbebdeee8c80a04c3b9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f3f1fa1e4348bfbebdeee8c80a04c3b9-Abstract.html)

**Abstract**:

Value factorization is a popular and promising approach to scaling up multi-agent reinforcement learning in cooperative settings, which balances the learning scalability and the representational capacity of value functions. However, the theoretical understanding of such methods is limited. In this paper, we formalize a multi-agent fitted Q-iteration framework for analyzing factorized multi-agent Q-learning. Based on this framework, we investigate linear value factorization and reveal that multi-agent Q-learning with this simple decomposition implicitly realizes a powerful counterfactual credit assignment, but may not converge in some settings. Through further analysis, we find that on-policy training or richer joint value function classes can improve its local or global convergence properties, respectively. Finally, to support our theoretical implications in practical realization, we conduct an empirical analysis of state-of-the-art deep multi-agent Q-learning algorithms on didactic examples and a broad set of StarCraft II unit micromanagement tasks.

----

## [2232] Margin-Independent Online Multiclass Learning via Convex Geometry

**Authors**: *Guru Guruganesh, Allen Liu, Jon Schneider, Joshua R. Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f40723ed94042ea9ea36bfb5ad4157b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f40723ed94042ea9ea36bfb5ad4157b2-Abstract.html)

**Abstract**:

We consider the problem of multi-class classification, where a stream of adversarially chosen queries arrive and must be assigned a label online. Unlike traditional bounds which seek to minimize the misclassification rate, we minimize the total distance from each query to the region corresponding to its assigned label. When the true labels are determined via a nearest neighbor partition -- i.e. the label of a point is given by which of $k$ centers it is closest to in Euclidean distance -- we show that one can achieve a loss that is independent of the total number of queries. We complement this result by showing that learning general convex sets requires an almost linear loss per query. Our results build off of regret guarantees for the problem of contextual search. In addition, we develop a novel reduction technique from multiclass classification to binary classification which may be of independent interest.

----

## [2233] STEP: Out-of-Distribution Detection in the Presence of Limited In-Distribution Labeled Data

**Authors**: *Zhi Zhou, Lan-Zhe Guo, Zhanzhan Cheng, Yu-Feng Li, Shiliang Pu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f4334c131c781e2a6f0a5e34814c8147-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f4334c131c781e2a6f0a5e34814c8147-Abstract.html)

**Abstract**:

Existing semi-supervised learning (SSL) studies typically assume that unlabeled and test data are drawn from the same distribution as labeled data. However, in many real-world applications, it is desirable to have SSL algorithms that not only classify the samples drawn from the same distribution of labeled data but also detect out-of-distribution (OOD) samples drawn from an unknown distribution. In this paper, we study a setting called semi-supervised OOD detection. Two main challenges compared with previous OOD detection settings are i) the lack of labeled data and in-distribution data; ii) OOD samples could be unseen during training. Efforts on this direction remain limited. In this paper, we present an approach STEP significantly improving OOD detection performance by introducing a new technique: Structure-Keep Unzipping. It learns a new representation space in which OOD samples could be separated well. An efficient optimization algorithm is derived to solve the objective. Comprehensive experiments across various OOD detection benchmarks clearly show that our STEP approach outperforms other methods by a large margin and achieves remarkable detection performance on several benchmarks.

----

## [2234] Renyi Differential Privacy of The Subsampled Shuffle Model In Distributed Learning

**Authors**: *Antonious M. Girgis, Deepesh Data, Suhas N. Diggavi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f44ec26e2ac3f1ab8c2472d4b1c2ea86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f44ec26e2ac3f1ab8c2472d4b1c2ea86-Abstract.html)

**Abstract**:

We study privacy in a distributed learning framework, where clients collaboratively build a learning model iteratively throughinteractions with a server from whom we need privacy. Motivated by stochastic optimization and the federated learning (FL) paradigm, we focus on the case where a small fraction of data samples are randomly sub-sampled in each round to participate in the learning process, which also enables privacy amplification.  To obtain even stronger local privacy guarantees, we study this in the shuffle privacy model, where each client randomizes its response using a local differentially private (LDP) mechanism and the server only receives a random permutation (shuffle) of the clients' responses without theirassociation to each client. The principal result of this paper is a privacy-optimization performance trade-off for discrete randomization mechanisms in this sub-sampled shuffle privacy model. This is enabledthrough a new theoretical technique to analyze the Renyi Differential Privacy (RDP) of the sub-sampled shuffle model.  We numerically demonstrate that, for important regimes, with composition our boundyields significant improvement in privacy guarantee over the state-of-the-art approximate Differential Privacy (DP) guarantee (with strong composition) for sub-sampled shuffled models. We also demonstrate numerically significant improvement in privacy-learning performance operating point using real data sets. Despite these advances, an open question is to bridge the gap between lower and upper privacy bounds in our RDP analysis.

----

## [2235] Gradient-based Editing of Memory Examples for Online Task-free Continual Learning

**Authors**: *Xisen Jin, Arka Sadhu, Junyi Du, Xiang Ren*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f45a1078feb35de77d26b3f7a52ef502-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f45a1078feb35de77d26b3f7a52ef502-Abstract.html)

**Abstract**:

We explore task-free continual learning (CL), in which a model is trained to avoid catastrophic forgetting in the absence of explicit task boundaries or identities. Among many efforts on task-free CL, a notable family of approaches are memory-based that store and replay a subset of training examples. However, the utility of stored seen examples may diminish over time since CL models are continually updated. Here, we propose Gradient based Memory EDiting (GMED), a framework for editing stored examples in continuous input space via gradient updates, in order to create more "challenging" examples for replay. GMED-edited examples remain similar to their unedited forms, but can yield increased loss in the upcoming model updates, thereby making the future replays more effective in overcoming catastrophic forgetting. By construction, GMED can be seamlessly applied in conjunction with other memory-based CL algorithms to bring further improvement. Experiments validate the effectiveness of GMED, and our best method significantly outperforms baselines and previous state-of-the-art on five out of six datasets.

----

## [2236] Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time

**Authors**: *Ferran Alet, Maria Bauzá, Kenji Kawaguchi, Nurullah Giray Kuru, Tomás Lozano-Pérez, Leslie Pack Kaelbling*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f45cc474bff52cb1b2268a2f94a2abcf-Abstract.html)

**Abstract**:

From CNNs to attention mechanisms, encoding inductive biases into neural networks has been a fruitful source of improvement in machine learning. Adding auxiliary losses to the main objective function is a general way of encoding biases that can help networks learn better representations. However, since auxiliary losses are minimized only on training data, they suffer from the same generalization gap as regular task losses. Moreover, by adding a term to the loss function, the model optimizes a different objective than the one we care about. In this work we address both problems: first, we take inspiration from transductive learning and note that after receiving an input but before making a prediction, we can fine-tune our networks on any unsupervised loss. We call this process tailoring, because we customize the model to each input to ensure our prediction satisfies the inductive bias. Second, we formulate meta-tailoring, a nested optimization similar to that in meta-learning, and train our models to perform well on the task objective after adapting them using an unsupervised loss. The advantages of tailoring and meta-tailoring are discussed theoretically and demonstrated empirically on a diverse set of examples.

----

## [2237] Implicit Bias of SGD for Diagonal Linear Networks: a Provable Benefit of Stochasticity

**Authors**: *Scott Pesme, Loucas Pillaud-Vivien, Nicolas Flammarion*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f4661398cb1a3abd3ffe58600bf11322-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f4661398cb1a3abd3ffe58600bf11322-Abstract.html)

**Abstract**:

Understanding the implicit bias of training algorithms is of crucial importance in order to explain the success of overparametrised neural networks. In this paper, we study the dynamics of stochastic gradient descent over diagonal linear networks through its continuous time version, namely stochastic gradient flow. We explicitly characterise the solution chosen by the stochastic flow and prove that it always enjoys better generalisation properties than that of gradient flow.Quite surprisingly, we show that the convergence speed of the training loss controls the magnitude of the biasing effect: the slower the convergence, the better the bias. To fully complete our analysis, we provide convergence guarantees for the dynamics. We also give experimental results which support our theoretical claims. Our findings highlight the fact that structured noise can induce better generalisation and they help explain the greater performances of stochastic gradient  descent over gradient descent observed in practice.

----

## [2238] Iterative Teacher-Aware Learning

**Authors**: *Luyao Yuan, Dongruo Zhou, Junhong Shen, Jingdong Gao, Jeffrey L. Chen, Quanquan Gu, Ying Nian Wu, Song-Chun Zhu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f48c04ffab49ff0e5d1176244fdfb65c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f48c04ffab49ff0e5d1176244fdfb65c-Abstract.html)

**Abstract**:

In human pedagogy, teachers and students can interact adaptively to maximize communication efficiency. The teacher adjusts her teaching method for different students, and the student, after getting familiar with the teacher’s instruction mechanism, can infer the teacher’s intention to learn faster. Recently, the benefits of integrating this cooperative pedagogy into machine concept learning in discrete spaces have been proved by multiple works. However, how cooperative pedagogy can facilitate machine parameter learning hasn’t been thoroughly studied. In this paper, we propose a gradient optimization based teacher-aware learner who can incorporate teacher’s cooperative intention into the likelihood function and learn provably faster compared with the naive learning algorithms used in previous machine teaching works. We give theoretical proof that the iterative teacher-aware learning (ITAL) process leads to local and global improvements. We then validate our algorithms with extensive experiments on various tasks including regression, classification, and inverse reinforcement learning using synthetic and real data. We also show the advantage of modeling teacher-awareness when agents are learning from human teachers.

----

## [2239] Clockwork Variational Autoencoders

**Authors**: *Vaibhav Saxena, Jimmy Ba, Danijar Hafner*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f490d0af974fedf90cb0f1edce8e3dd5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f490d0af974fedf90cb0f1edce8e3dd5-Abstract.html)

**Abstract**:

Deep learning has enabled algorithms to generate realistic images. However, accurately predicting long video sequences requires understanding long-term dependencies and remains an open challenge. While existing video prediction models succeed at generating sharp images, they tend to fail at accurately predicting far into the future. We introduce the Clockwork VAE (CW-VAE), a video prediction model that leverages a hierarchy of latent sequences, where higher levels tick at slower intervals. We demonstrate the benefits of both hierarchical latents and temporal abstraction on 4 diverse video prediction datasets with sequences of up to 1000 frames, where CW-VAE outperforms top video prediction models. Additionally, we propose a Minecraft benchmark for long-term video prediction. We conduct several experiments to gain insights into CW-VAE and confirm that slower levels learn to represent objects that change more slowly in the video, and faster levels learn to represent faster objects.

----

## [2240] How Does it Sound?

**Authors**: *Kun Su, Xiulong Liu, Eli Shlizerman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f4e369c0a468d3aeeda0593ba90b5e55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f4e369c0a468d3aeeda0593ba90b5e55-Abstract.html)

**Abstract**:

One of the primary purposes of video is to capture people and their unique activities. It is often the case that the experience of watching the video can be enhanced by adding a musical soundtrack that is in-sync with the rhythmic features of these activities. How would this soundtrack sound? Such a problem is challenging since little is known about capturing the rhythmic nature of free body movements. In this work, we explore this problem and propose a novel system, called `RhythmicNet', which takes as an input a video which includes human movements and generates a soundtrack for it. RhythmicNet works directly with human movements by extracting skeleton keypoints and implements a sequence of models which translate the keypoints to rhythmic sounds.RhythmicNet follows the natural process of music improvisation which includes the prescription of streams of the beat, the rhythm and the melody. In particular, RhythmicNet first infers the music beat and the style pattern from body keypoints per each frame to produce rhythm. Next, it implements a transformer-based model to generate the hits of drum instruments and implements a U-net based model to generate the velocity and the offsets of the instruments. Additional types of instruments are added to the soundtrack by further conditioning on the generated drum sounds. We evaluate RhythmicNet on large scale datasets of videos that include body movements with inherit sound association, such as dance, as well as 'in the wild' internet videos of various movements and actions. We show that the method can generate plausible music that aligns well with different types of human movements.

----

## [2241] Stabilizing Dynamical Systems via Policy Gradient Methods

**Authors**: *Juan C. Perdomo, Jack Umenberger, Max Simchowitz*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f4f6dce2f3a0f9dada0c2b5b66452017-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f4f6dce2f3a0f9dada0c2b5b66452017-Abstract.html)

**Abstract**:

Stabilizing an unknown control system is one of the most fundamental problems in control systems engineering.  In this paper, we provide a simple, model-free algorithm for stabilizing fully observed dynamical systems.  While model-free methods have become increasingly popular in practice due to their simplicity and flexibility, stabilization via direct policy search has received surprisingly little attention. Our algorithm proceeds by solving a series of discounted LQR problems, where the discount factor is gradually increased. We prove that this method efficiently recovers a stabilizing controller for linear systems, and for smooth, nonlinear systems within a neighborhood of their equilibria. Our approach overcomes a significant limitation of prior work, namely the need for a pre-given stabilizing control policy. We empirically evaluate the effectiveness of our approach on common control benchmarks.

----

## [2242] Language models enable zero-shot prediction of the effects of mutations on protein function

**Authors**: *Joshua Meier, Roshan Rao, Robert Verkuil, Jason Liu, Tom Sercu, Alexander Rives*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html)

**Abstract**:

Modeling the effect of sequence variation on function is a fundamental problem for understanding and designing proteins. Since evolution encodes information about function into patterns in protein sequences, unsupervised models of variant effects can be learned from sequence data. The approach to date has been to fit a model to a family of related sequences. The conventional setting is limited, since a new model must be trained for each prediction task. We show that using only zero-shot inference, without any supervision from experimental data or additional training, protein language models capture the functional effects of sequence variation, performing at state-of-the-art.

----

## [2243] Deep Reinforcement Learning at the Edge of the Statistical Precipice

**Authors**: *Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, Marc G. Bellemare*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)

**Abstract**:

Deep reinforcement learning (RL) algorithms are predominantly evaluated by comparing their relative performance on a large suite of tasks. Most published results on deep RL benchmarks compare point estimates of aggregate performance such as mean and median scores across tasks, ignoring the statistical uncertainty implied by the use of a finite number of training runs. Beginning with the Arcade Learning Environment (ALE), the shift towards computationally-demanding benchmarks has led to the practice of evaluating only a small number of runs per task, exacerbating the statistical uncertainty in point estimates. In this paper, we argue that reliable evaluation in the few run deep RL regime cannot ignore the uncertainty in results without running the risk of slowing down progress in the field. We illustrate this point using a case study on the Atari 100k benchmark, where we find substantial discrepancies between conclusions drawn from point estimates alone versus a more thorough statistical analysis. With the aim of increasing the field's confidence in reported results with a handful of runs, we advocate for reporting interval estimates of aggregate performance and propose performance profiles to account for the variability in results, as well as present more robust and efficient aggregate metrics, such as interquartile mean scores, to achieve small uncertainty in results. Using such statistical tools, we scrutinize performance evaluations of existing algorithms on other widely used RL benchmarks including the ALE, Procgen, and the DeepMind Control Suite, again revealing discrepancies in prior comparisons. Our findings call for a change in how we evaluate performance in deep RL, for which we present a more rigorous evaluation methodology, accompanied with an open-source library rliable, to prevent unreliable results from stagnating the field. This work received an outstanding paper award at NeurIPS 2021.

----

## [2244] DRONE: Data-aware Low-rank Compression for Large NLP Models

**Authors**: *Patrick H. Chen, Hsiang-Fu Yu, Inderjit S. Dhillon, Cho-Jui Hsieh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f56de5ef149cf0aedcc8f4797031e229-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f56de5ef149cf0aedcc8f4797031e229-Abstract.html)

**Abstract**:

The representations learned by large-scale NLP models such as BERT have been widely used in various tasks. However, the increasing model size of the pre-trained models also brings efficiency challenges, including inference speed and model size when deploying models on mobile devices. Specifically, most operations in BERT consist of matrix multiplications. These matrices are not low-rank and thus canonical matrix decomposition could not find an efficient approximation. In this paper, we observe that the learned representation of each layer lies in a low-dimensional space. Based on this observation, we propose DRONE (data-aware low-rank compression), a provably optimal low-rank decomposition of weight matrices, which has a simple closed form solution that can be efficiently computed. DRONE can be applied to both fully connected and self-attention layers appearing in the BERT model. In addition to compressing standard models, out method can also be used on distilled BERT models to further improve compression rate. Experimental results show that DRONE is able to improve both model size and inference speed with limited loss in accuracy. Specifically, DRONE alone achieves 1.92x speedup on the MRPC task with only 1.5% loss in accuracy, and when DRONE is combined with distillation, it further achieves over 12.3x speedup on various natural language inference tasks.

----

## [2245] DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning

**Authors**: *Hussein Hazimeh, Zhe Zhao, Aakanksha Chowdhery, Maheswaran Sathiamoorthy, Yihua Chen, Rahul Mazumder, Lichan Hong, Ed H. Chi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html)

**Abstract**:

The Mixture-of-Experts (MoE) architecture is showing promising results in improving parameter sharing in multi-task learning (MTL) and in scaling high-capacity neural networks. State-of-the-art MoE models use a trainable "sparse gate'" to select a subset of the experts for each input example. While conceptually appealing, existing sparse gates, such as Top-k, are not smooth. The lack of smoothness can lead to convergence and statistical performance issues when training with gradient-based methods. In this paper, we develop DSelect-k: a continuously differentiable and sparse gate for MoE, based on a novel binary encoding formulation. The gate can be trained using first-order methods, such as stochastic gradient descent, and offers explicit control over the number of experts to select. We demonstrate the effectiveness of DSelect-k on both synthetic and real MTL datasets with up to 128 tasks. Our experiments indicate that DSelect-k can achieve statistically significant improvements in prediction and expert selection over popular MoE gates. Notably, on a real-world, large-scale recommender system, DSelect-k achieves over 22% improvement in predictive performance compared to Top-k. We provide an open-source implementation of DSelect-k.

----

## [2246] Mind the Gap: Assessing Temporal Generalization in Neural Language Models

**Authors**: *Angeliki Lazaridou, Adhiguna Kuncoro, Elena Gribovskaya, Devang Agrawal, Adam Liska, Tayfun Terzi, Mai Gimenez, Cyprien de Masson d'Autume, Tomás Kociský, Sebastian Ruder, Dani Yogatama, Kris Cao, Susannah Young, Phil Blunsom*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5bf0ba0a17ef18f9607774722f5698c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5bf0ba0a17ef18f9607774722f5698c-Abstract.html)

**Abstract**:

Our world is open-ended, non-stationary, and constantly evolving; thus what we talk about and how we talk about it change over time. This inherent dynamic nature of language contrasts with the current static language modelling paradigm, which trains and evaluates models on utterances from overlapping time periods. Despite impressive recent progress, we demonstrate that Transformer-XL language models perform worse in the realistic setup of predicting future utterances from beyond their training period, and that model performance becomes increasingly worse with time. We find that, while increasing model size alone—a key driver behind recent progress—does not solve this problem, having models that continually update their knowledge with new information can indeed mitigate this performance degradation over time. Hence, given the compilation of ever-larger language modelling datasets, combined with the growing list of language-model-based NLP applications that require up-to-date factual knowledge about the world, we argue that now is the right time to rethink the static way in which we currently train and evaluate our language models, and develop adaptive language models that can remain up-to-date with respect to our ever-changing and non-stationary world. We publicly release our dynamic, streaming language modelling benchmarks for WMT and arXiv to facilitate language model evaluation that takes temporal dynamics into account.

----

## [2247] Heavy Tails in SGD and Compressibility of Overparametrized Neural Networks

**Authors**: *Melih Barsbey, Milad Sefidgaran, Murat A. Erdogdu, Gaël Richard, Umut Simsekli*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5c3dd7514bf620a1b85450d2ae374b1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5c3dd7514bf620a1b85450d2ae374b1-Abstract.html)

**Abstract**:

Neural network compression techniques have become increasingly popular as they can drastically reduce the storage and computation requirements for very large networks. Recent empirical studies have illustrated that even simple pruning strategies can be surprisingly effective, and several theoretical studies have shown that compressible networks (in specific senses) should achieve a low generalization error. Yet, a theoretical characterization of the underlying causes that make the networks amenable to such simple compression schemes is still missing. In this study, focusing our attention on stochastic gradient descent (SGD), our main contribution is to link compressibility to two recently established properties of SGD: (i) as the network size goes to infinity, the system can converge to a mean-field limit, where the network weights behave independently [DBDFŞ20], (ii) for a large step-size/batch-size ratio, the SGD iterates can converge to a heavy-tailed stationary distribution  [HM20, GŞZ21]. Assuming that both of these phenomena occur simultaneously, we prove that the networks are guaranteed to be '$\ell_p$-compressible', and the compression errors of different pruning techniques (magnitude, singular value, or node pruning) become arbitrarily small as the network size increases. We further prove generalization bounds adapted to our theoretical framework, which are consistent with the observation that the generalization error will be lower for more compressible networks. Our theory and numerical study on various neural networks show that large step-size/batch-size ratios introduce heavy tails, which, in combination with overparametrization, result in compressibility.

----

## [2248] Targeted Neural Dynamical Modeling

**Authors**: *Cole L. Hurwitz, Akash Srivastava, Kai Xu, Justin Jude, Matthew G. Perich, Lee E. Miller, Matthias H. Hennig*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5cfbc876972bd0d031c8abc37344c28-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5cfbc876972bd0d031c8abc37344c28-Abstract.html)

**Abstract**:

Latent dynamics models have emerged as powerful tools for modeling and interpreting neural population activity. Recently, there has been a focus on incorporating simultaneously measured behaviour into these models to further disentangle sources of neural variability in their latent space. These approaches, however, are limited in their ability to capture the underlying neural dynamics (e.g. linear) and in their ability to relate the learned dynamics back to the observed behaviour (e.g. no time lag). To this end, we introduce Targeted Neural Dynamical Modeling (TNDM), a nonlinear state-space model that jointly models the neural activity and external behavioural variables. TNDM decomposes neural dynamics into behaviourally relevant and behaviourally irrelevant dynamics; the relevant dynamics are used to reconstruct the behaviour through a flexible linear decoder and both sets of dynamics are used to reconstruct the neural activity through a linear decoder with no time lag. We implement TNDM as a sequential variational autoencoder and validate it on simulated recordings and recordings taken from the premotor and motor cortex of a monkey performing a center-out reaching task. We show that TNDM is able to learn low-dimensional latent dynamics that are highly predictive of behaviour without sacrificing its fit to the neural data.

----

## [2249] Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation

**Authors**: *Shiqi Yang, Yaxing Wang, Joost van de Weijer, Luis Herranz, Shangling Jui*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5deaeeae1538fb6c45901d524ee2f98-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5deaeeae1538fb6c45901d524ee2f98-Abstract.html)

**Abstract**:

Domain adaptation (DA) aims to alleviate the domain shift between source domain and target domain. Most DA methods require access to the source data, but often that is not possible (e.g. due to data privacy or intellectual property). In this paper, we address the challenging source-free domain adaptation (SFDA) problem, where the source pretrained model is adapted to the target domain in the absence of source data. Our method is based on the observation that target data, which might no longer align with the source domain classifier, still forms clear clusters. We capture this intrinsic structure by defining local affinity of the target data, and encourage label consistency among data with high local affinity. We observe that higher affinity should be assigned to reciprocal neighbors, and propose a self regularization loss to decrease the negative impact of noisy neighbors. Furthermore, to aggregate information with more context, we consider expanded neighborhoods with small affinity values. In the experimental results we verify that the inherent structure of the target features is an important source of information for domain adaptation. We demonstrate that this local structure can be efficiently captured by considering the local neighbors, the reciprocal neighbors, and the expanded neighborhood. Finally, we achieve state-of-the-art performance on several 2D image and 3D point cloud recognition datasets. Code is available in https://github.com/Albert0147/SFDA_neighbors.

----

## [2250] Learning with Noisy Correspondence for Cross-modal Matching

**Authors**: *Zhenyu Huang, Guocheng Niu, Xiao Liu, Wenbiao Ding, Xinyan Xiao, Hua Wu, Xi Peng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5e62af885293cf4d511ceef31e61c80-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5e62af885293cf4d511ceef31e61c80-Abstract.html)

**Abstract**:

Cross-modal matching, which aims to establish the correspondence between two different modalities, is fundamental to a variety of tasks such as cross-modal retrieval and vision-and-language understanding. Although a huge number of cross-modal matching methods have been proposed and achieved remarkable progress in recent years, almost all of these methods implicitly assume that the multimodal training data are correctly aligned. In practice, however, such an assumption is extremely expensive even impossible to satisfy. Based on this observation, we reveal and study a latent and challenging direction in cross-modal matching, named noisy correspondence, which could be regarded as a new paradigm of noisy labels. Different from the traditional noisy labels which mainly refer to the errors in category labels, our noisy correspondence refers to the mismatch paired samples. To solve this new problem, we propose a novel method for learning with noisy correspondence, named Noisy Correspondence Rectifier (NCR). In brief, NCR divides the data into clean and noisy partitions based on the memorization effect of neural networks and then rectifies the correspondence via an adaptive prediction model in a co-teaching manner. To verify the effectiveness of our method, we conduct experiments by using the image-text matching as a showcase. Extensive experiments on Flickr30K, MS-COCO, and Conceptual Captions verify the effectiveness of our method. The code could be accessed from www.pengxi.me .

----

## [2251] Offline Reinforcement Learning with Reverse Model-based Imagination

**Authors**: *Jianhao Wang, Wenzhe Li, Haozhe Jiang, Guangxiang Zhu, Siyuan Li, Chongjie Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f5e647292cc4e1064968ca62bebe7e47-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f5e647292cc4e1064968ca62bebe7e47-Abstract.html)

**Abstract**:

In offline reinforcement learning (offline RL), one of the main challenges is to deal with the distributional shift between the learning policy and the given dataset. To address this problem,  recent offline RL methods attempt to introduce conservatism bias to encourage learning in high-confidence areas. Model-free approaches directly encode such bias into policy or value function learning using conservative regularizations or special network structures, but their constrained policy search limits the generalization beyond the offline dataset. Model-based approaches learn forward dynamics models with conservatism quantifications and then generate imaginary trajectories to extend the offline datasets. However, due to limited samples in offline datasets, conservatism quantifications often suffer from overgeneralization in out-of-support regions. The unreliable conservative measures will mislead forward model-based imaginations to undesired areas, leading to overaggressive behaviors. To encourage more conservatism, we propose a novel model-based offline RL framework, called Reverse Offline Model-based Imagination (ROMI). We learn a reverse dynamics model in conjunction with a novel reverse policy,  which can generate rollouts leading to the target goal states within the offline dataset. These reverse imaginations provide informed data augmentation for model-free policy learning and enable conservative generalization beyond the offline dataset. ROMI can effectively combine with off-the-shelf model-free algorithms to enable model-based generalization with proper conservatism. Empirical results show that our method can generate more conservative behaviors and achieve state-of-the-art performance on offline RL benchmark tasks.

----

## [2252] Parameter Prediction for Unseen Deep Architectures

**Authors**: *Boris Knyazev, Michal Drozdzal, Graham W. Taylor, Adriana Romero-Soriano*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f6185f0ef02dcaec414a3171cd01c697-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f6185f0ef02dcaec414a3171cd01c697-Abstract.html)

**Abstract**:

Deep learning has been successful in automating the design of features in machine learning pipelines. However, the algorithms optimizing neural network parameters remain largely hand-designed and computationally inefficient. We study if we can use deep learning to directly predict these parameters by exploiting the past knowledge of training other networks. We introduce a large-scale dataset of diverse computational graphs of neural architectures - DeepNets-1M - and use it to explore parameter prediction on CIFAR-10 and ImageNet. By leveraging advances in graph neural networks, we propose a hypernetwork that can predict performant parameters in a single forward pass taking a fraction of a second, even on a CPU. The proposed model achieves surprisingly good performance on unseen and diverse networks. For example, it is able to predict all 24 million parameters of a ResNet-50 achieving a 60% accuracy on CIFAR-10. On ImageNet, top-5 accuracy of some of our networks approaches 50%. Our task along with the model and results can potentially lead to a new, more computationally efficient paradigm of training networks. Our model also learns a strong representation of neural architectures enabling their analysis.

----

## [2253] FMMformer: Efficient and Flexible Transformer via Decomposed Near-field and Far-field Attention

**Authors**: *Tan M. Nguyen, Vai Suliafu, Stanley J. Osher, Long Chen, Bao Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f621585df244e9596dc70a39b579efb1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f621585df244e9596dc70a39b579efb1-Abstract.html)

**Abstract**:

We propose FMMformers, a class of efficient and flexible transformers inspired by the celebrated fast multipole method (FMM) for accelerating interacting particle simulation. FMM decomposes particle-particle interaction into near-field and far-field components and then performs direct and coarse-grained computation, respectively. Similarly, FMMformers decompose the attention into near-field and far-field attention, modeling the near-field attention by a banded matrix and the far-field attention by a low-rank matrix. Computing the attention matrix for FMMformers requires linear complexity in computational time and memory footprint with respect to the sequence length. In contrast, standard transformers suffer from quadratic complexity. We analyze and validate the advantage of FMMformers over the standard transformer on the Long Range Arena and language modeling benchmarks. FMMformers can even outperform the standard transformer in terms of accuracy by a significant margin. For instance, FMMformers achieve an average classification accuracy of $60.74\%$ over the five Long Range Arena tasks, which is significantly better than the standard transformer's average accuracy of $58.70\%$.

----

## [2254] Square Root Principal Component Pursuit: Tuning-Free Noisy Robust Matrix Recovery

**Authors**: *Junhui Zhang, Jingkai Yan, John Wright*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f65854da4622c1f1ad4ffeb361d7703c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f65854da4622c1f1ad4ffeb361d7703c-Abstract.html)

**Abstract**:

We propose a new framework -- Square Root Principal Component Pursuit -- for low-rank matrix recovery from observations corrupted with noise and outliers. Inspired by the square root Lasso, this new formulation does not require prior knowledge of the noise level. We show that a single, universal choice of the regularization parameter suffices to achieve reconstruction error proportional to the (a priori unknown) noise level. In comparison, previous formulations such as stable PCP rely on noise-dependent parameters to achieve similar performance, and are therefore challenging to deploy in applications where the noise level is unknown. We validate the effectiveness of our new method through experiments on simulated and real datasets. Our simulations corroborate the claim that a universal choice of the regularization parameter yields near optimal performance across a range of noise levels, indicating that the proposed method outperforms the (somewhat loose) bound proved here.

----

## [2255] Neural Bellman-Ford Networks: A General Graph Neural Network Framework for Link Prediction

**Authors**: *Zhaocheng Zhu, Zuobai Zhang, Louis-Pascal A. C. Xhonneux, Jian Tang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html)

**Abstract**:

Link prediction is a very fundamental task on graphs. Inspired by traditional path-based methods, in this paper we propose a general and flexible representation learning framework based on paths for link prediction. Specifically, we define the representation of a pair of nodes as the generalized sum of all path representations, with each path representation as the generalized product of the edge representations in the path. Motivated by the Bellman-Ford algorithm for solving the shortest path problem, we show that the proposed path formulation can be efficiently solved by the generalized Bellman-Ford algorithm. To further improve the capacity of the path formulation, we propose the Neural Bellman-Ford Network (NBFNet), a general graph neural network framework that solves the path formulation with learned operators in the generalized Bellman-Ford algorithm. The NBFNet parameterizes the generalized Bellman-Ford algorithm with 3 neural components, namely Indicator, Message and Aggregate functions, which corresponds to the boundary condition, multiplication operator, and summation operator respectively. The NBFNet covers many traditional path-based methods, and can be applied to both homogeneous graphs and multi-relational graphs (e.g., knowledge graphs) in both transductive and inductive settings. Experiments on both homogeneous graphs and knowledge graphs show that the proposed NBFNet outperforms existing methods by a large margin in both transductive and inductive settings, achieving new state-of-the-art results.

----

## [2256] CorticalFlow: A Diffeomorphic Mesh Transformer Network for Cortical Surface Reconstruction

**Authors**: *Léo Lebrat, Rodrigo Santa Cruz, Frédéric de Gournay, Darren Fu, Pierrick Bourgeat, Jurgen Fripp, Clinton Fookes, Olivier Salvado*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f6b5f8c32c65fee991049a55dc97d1ce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f6b5f8c32c65fee991049a55dc97d1ce-Abstract.html)

**Abstract**:

In this paper, we introduce CorticalFlow, a new geometric deep-learning model that, given a 3-dimensional image, learns to deform a reference template towards a targeted object. To conserve the template meshâ€™s topological properties, we train our model over a set of diffeomorphic transformations. This new implementation of a flow Ordinary Differential Equation (ODE) framework benefits from a small GPU memory footprint, allowing the generation of surfaces with several hundred thousand vertices. To reduce topological errors introduced by its discrete resolution, we derive numeric conditions which improve the manifoldness of the predicted triangle mesh. To exhibit the utility of CorticalFlow, we demonstrate its performance for the challenging task of brain cortical surface reconstruction. In contrast to the current state-of-the-art, CorticalFlow produces superior surfaces while reducing the computation time from nine and a half minutes to one second. More significantly, CorticalFlow enforces the generation of anatomically plausible surfaces; the absence of which has been a major impediment restricting the clinical relevance of such surface reconstruction methods.

----

## [2257] Bridging the Gap Between Practice and PAC-Bayes Theory in Few-Shot Meta-Learning

**Authors**: *Nan Ding, Xi Chen, Tomer Levinboim, Sebastian Goodman, Radu Soricut*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f6b6d2a114a9644419dc8d2315f22401-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f6b6d2a114a9644419dc8d2315f22401-Abstract.html)

**Abstract**:

Despite recent advances in its theoretical understanding, there still remains a significant gap in the ability of existing PAC-Bayesian theories on meta-learning to explain performance improvements in the few-shot learning setting, where the number of training examples in the target tasks is severely limited. This gap originates from an assumption in the existing theories which supposes that the number of training examples in the observed tasks and the number of training examples in the target tasks follow the same distribution, an assumption that rarely holds in practice. By relaxing this assumption, we develop two PAC-Bayesian bounds tailored for the few-shot learning setting and show that two existing meta-learning algorithms (MAML and Reptile) can be derived from our bounds, thereby bridging the gap between practice and PAC-Bayesian theories. Furthermore, we derive a new computationally-efficient PACMAML algorithm, and show it outperforms existing meta-learning algorithms on several few-shot benchmark datasets.

----

## [2258] SLOE: A Faster Method for Statistical Inference in High-Dimensional Logistic Regression

**Authors**: *Steve Yadlowsky, Taedong Yun, Cory Y. McLean, Alexander D'Amour*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f6c2a0c4b566bc99d596e58638e342b0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f6c2a0c4b566bc99d596e58638e342b0-Abstract.html)

**Abstract**:

Logistic regression remains one of the most widely used tools in applied statistics, machine learning and data science. However, in moderately high-dimensional problems, where the number of features $d$ is a non-negligible fraction of the sample size $n$, the logistic regression maximum likelihood estimator (MLE), and statistical procedures based the large-sample approximation of its distribution, behave poorly. Recently, Sur and Candès (2019) showed that these issues can be corrected by applying a new approximation of the MLE's sampling distribution in this high-dimensional regime. Unfortunately, these corrections are difficult to implement in practice, because they require an estimate of the \emph{signal strength}, which is a function of the underlying parameters $\beta$ of the logistic regression. To address this issue, we propose SLOE, a fast and straightforward approach to estimate the signal strength in logistic regression. The key insight of SLOE is that the Sur and Candès (2019) correction can be reparameterized in terms of the corrupted signal strength, which is only a function of the estimated parameters $\widehat \beta$. We propose an estimator for this quantity, prove that it is consistent in the relevant high-dimensional regime, and show that dimensionality correction using SLOE is accurate in finite samples. Compared to the existing ProbeFrontier heuristic, SLOE is conceptually simpler and orders of magnitude faster, making it suitable for routine use. We demonstrate the importance of routine dimensionality correction in the Heart Disease dataset from the UCI repository, and a genomics application using data from the UK Biobank.

----

## [2259] ELLA: Exploration through Learned Language Abstraction

**Authors**: *Suvir Mirchandani, Siddharth Karamcheti, Dorsa Sadigh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f6f154417c4665861583f9b9c4afafa2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f6f154417c4665861583f9b9c4afafa2-Abstract.html)

**Abstract**:

Building agents capable of understanding language instructions is critical to effective and robust human-AI collaboration. Recent work focuses on training these agents via reinforcement learning in environments with synthetic language; however, instructions often define long-horizon, sparse-reward tasks, and learning policies requires many episodes of experience. We introduce ELLA: Exploration through Learned Language Abstraction, a reward shaping approach geared towards boosting sample efficiency in sparse reward environments by correlating high-level instructions with simpler low-level constituents. ELLA has two key elements: 1) A termination classifier that identifies when agents complete low-level instructions, and 2) A relevance classifier that correlates low-level instructions with success on high-level tasks. We learn the termination classifier offline from pairs of instructions and terminal states. Notably, in departure from prior work in language and abstraction, we learn the relevance classifier online, without relying on an explicit decomposition of high-level instructions to low-level instructions. On a suite of complex BabyAI environments with varying instruction complexities and reward sparsity, ELLA shows gains in sample efficiency relative to language-based shaping and traditional RL methods.

----

## [2260] Learning Distilled Collaboration Graph for Multi-Agent Perception

**Authors**: *Yiming Li, Shunli Ren, Pengxiang Wu, Siheng Chen, Chen Feng, Wenjun Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f702defbc67edb455949f46babab0c18-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f702defbc67edb455949f46babab0c18-Abstract.html)

**Abstract**:

To promote better performance-bandwidth trade-off for multi-agent perception, we propose a novel distilled collaboration graph (DiscoGraph) to model trainable, pose-aware, and adaptive collaboration among agents. Our key novelties lie in two aspects. First, we propose a teacher-student framework to train DiscoGraph via knowledge distillation. The teacher model employs an early collaboration with holistic-view inputs; the student model is based on intermediate collaboration with single-view inputs. Our framework trains DiscoGraph by constraining post-collaboration feature maps in the student model to match the correspondences in the teacher model. Second, we propose a matrix-valued edge weight in DiscoGraph. In such a matrix, each element reflects the inter-agent attention at a specific spatial region, allowing an agent to adaptively highlight the informative regions. During inference, we only need to use the student model named as the distilled collaboration network (DiscoNet). Attributed to the teacher-student framework, multiple agents with the shared DiscoNet could collaboratively approach the performance of a hypothetical teacher model with a holistic view. Our approach is validated on V2X-Sim 1.0, a large-scale multi-agent perception dataset that we synthesized using CARLA and SUMO co-simulation. Our quantitative and qualitative experiments in multi-agent 3D object detection show that DiscoNet could not only achieve a better performance-bandwidth trade-off than the state-of-the-art collaborative perception methods, but also bring more straightforward design rationale. Our code is available on https://github.com/ai4ce/DiscoNet.

----

## [2261] Federated-EM with heterogeneity mitigation and variance reduction

**Authors**: *Aymeric Dieuleveut, Gersende Fort, Eric Moulines, Geneviève Robin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f740c8d9c193f16d8a07d3a8a751d13f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f740c8d9c193f16d8a07d3a8a751d13f-Abstract.html)

**Abstract**:

The Expectation Maximization (EM) algorithm is the default algorithm for inference in latent variable models. As in any other field of machine learning, applications of latent variable models to very large datasets make the use of advanced parallel and distributed architecture mandatory. This paper introduces FedEM, which is the first extension of the EM algorithm to the federated learning context. FedEM is  a new communication efficient method, which handles partial participation of local devices, and is robust to  heterogeneous distribution of the datasets. To alleviate the communication bottleneck, FedEM compresses appropriately defined complete data sufficient statistics. We also develop and analyze an extension of FedEM to further incorporate a variance reduction scheme. In all cases, we derive finite-time complexity bounds for smooth non-convex problems.  Numerical results are presented to support our theoretical findings, as well as an application to federated missing values imputation for biodiversity monitoring.

----

## [2262] On the Role of Optimization in Double Descent: A Least Squares Study

**Authors**: *Ilja Kuzborskij, Csaba Szepesvári, Omar Rivasplata, Amal Rannen-Triki, Razvan Pascanu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f754186469a933256d7d64095e963594-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f754186469a933256d7d64095e963594-Abstract.html)

**Abstract**:

Empirically it has been observed that the performance of deep neural networks steadily improves with increased model size, contradicting the classical view on overfitting and generalization. Recently, the double descent phenomenon has been proposed to reconcile this observation with theory, suggesting that the test error has a second descent when the model becomes sufficiently overparameterized, as the model size itself acts as an implicit regularizer. In this paper we add to the growing body of work in this space, providing a careful study of learning dynamics as a function of model size for the least squares scenario. We show an excess risk bound for the gradient descent solution of the least squares objective. The bound depends on the smallest non-zero eigenvalue of the sample covariance matrix of the input features, via a functional form that has the double descent behaviour. This gives a new perspective on the double descent curves reported in the literature, as our analysis of the excess risk allows to decouple the effect of optimization and generalization error. In particular, we find that in the case of noiseless regression, double descent is explained solely by optimization-related quantities, which was missed in studies focusing on the Moore-Penrose pseudoinverse solution. We believe that our derivation provides an alternative view compared to existing works, shedding some light on a possible cause of this phenomenon, at least in the considered least squares setting. We empirically explore if our predictions hold for neural networks, in particular whether the spectrum of the sample covariance of features at intermediary hidden layers has a similar behaviour as the one predicted by our derivations in the least squares setting.

----

## [2263] Neural Architecture Dilation for Adversarial Robustness

**Authors**: *Yanxi Li, Zhaohui Yang, Yunhe Wang, Chang Xu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html)

**Abstract**:

With the tremendous advances in the architecture and scale of convolutional neural networks (CNNs) over the past few decades, they can easily reach or even exceed the performance of humans in certain tasks. However, a recently discovered shortcoming of CNNs is that they are vulnerable to adversarial attacks. Although the adversarial robustness of CNNs can be improved by adversarial training, there is a trade-off between standard accuracy and adversarial robustness. From the neural architecture perspective, this paper aims to improve the adversarial robustness of the backbone CNNs that have a satisfactory accuracy. Under a minimal computational overhead, the introduction of a dilation architecture is expected to be friendly with the standard performance of the backbone CNN while pursuing adversarial robustness. Theoretical analyses on the standard and adversarial error bounds naturally motivate the proposed neural architecture dilation algorithm. Experimental results on real-world datasets and benchmark neural networks demonstrate the effectiveness of the proposed algorithm to balance the accuracy and adversarial robustness.

----

## [2264] Clustering Effect of Adversarial Robust Models

**Authors**: *Yang Bai, Xin Yan, Yong Jiang, Shu-Tao Xia, Yisen Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f770b62bc8f42a0b66751fe636fc6eb0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f770b62bc8f42a0b66751fe636fc6eb0-Abstract.html)

**Abstract**:

Adversarial robustness has received increasing attention along with the study of adversarial examples. So far, existing works show that robust models not only obtain robustness against various adversarial attacks but also boost the performance in some downstream tasks. However, the underlying mechanism of adversarial robustness is still not clear. In this paper, we interpret adversarial robustness from the perspective of linear components, and find that there exist some statistical properties for comprehensively robust models. Specifically, robust models show obvious hierarchical clustering effect on their linearized sub-networks, when removing or replacing all non-linear components (e.g., batch normalization, maximum pooling, or activation layers). Based on these observations, we propose a novel understanding of adversarial robustness and apply it on more tasks including domain adaption and robustness boosting. Experimental evaluations demonstrate the rationality and superiority of our proposed clustering strategy. Our code is available at https://github.com/bymavis/AdvWeightNeurIPS2021.

----

## [2265] On the Cryptographic Hardness of Learning Single Periodic Neurons

**Authors**: *Min Jae Song, Ilias Zadik, Joan Bruna*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f78688fb6a5507413ade54a230355acd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f78688fb6a5507413ade54a230355acd-Abstract.html)

**Abstract**:

We show a simple reduction which demonstrates the cryptographic hardness of learning a single periodic neuron over isotropic Gaussian distributions in the presence of noise. More precisely, our reduction shows that any polynomial-time algorithm (not necessarily gradient-based) for learning such functions under small noise implies a polynomial-time quantum algorithm for solving worst-case lattice problems, whose hardness form the foundation of lattice-based cryptography. Our core hard family of functions, which are well-approximated by one-layer neural networks, take the general form of a univariate periodic function applied to an affine projection of the data. These functions have appeared in previous seminal works which demonstrate their hardness against gradient-based (Shamir'18), and Statistical Query (SQ) algorithms (Song et al.'17). We show that if (polynomially) small noise is added to the labels, the intractability of learning these functions applies to all polynomial-time algorithms, beyond gradient-based and SQ algorithms, under the aforementioned cryptographic assumptions. Moreover, we demonstrate the necessity of noise in the hardness result by designing a polynomial-time algorithm for learning certain families of such functions under exponentially small adversarial noise. Our proposed algorithm is not a gradient-based or an SQ algorithm, but is rather based on the celebrated Lenstra-Lenstra-Lov\'asz (LLL) lattice basis reduction algorithm. Furthermore, in the absence of noise, this algorithm can be directly applied to solve CLWE detection (Bruna et al.'21) and phase retrieval with an optimal sample complexity of $d+1$ samples. In the former case, this improves upon the quadratic-in-$d$ sample complexity required in (Bruna et al.'21).

----

## [2266] PCA Initialization for Approximate Message Passing in Rotationally Invariant Models

**Authors**: *Marco Mondelli, Ramji Venkataramanan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7ac67a9aa8d255282de7d11391e1b69-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7ac67a9aa8d255282de7d11391e1b69-Abstract.html)

**Abstract**:

We study the problem of estimating a rank-1 signal in the presence of rotationally invariant noise--a class of perturbations more general than Gaussian noise.  Principal Component Analysis (PCA) provides a natural estimator, and sharp results on its performance have been obtained in the high-dimensional regime. Recently, an Approximate Message Passing (AMP) algorithm has been proposed as an alternative estimator with the potential to improve the accuracy of PCA. However, the existing analysis of AMP requires an initialization that is both correlated with the signal and independent of the noise, which is often unrealistic in practice. In this work, we combine the two methods, and propose to initialize AMP with PCA. Our main result is a rigorous asymptotic characterization of the performance of this estimator. Both the AMP algorithm and its analysis differ from those previously derived in the Gaussian setting: at every iteration, our AMP algorithm requires a specific term to account for PCA initialization, while in the Gaussian case, PCA initialization affects only the first iteration of AMP. The proof is based on a two-phase artificial AMP that first approximates the PCA estimator and then mimics the true AMP. Our numerical simulations show an excellent agreement between AMP results and theoretical predictions, and suggest an interesting open direction on achieving Bayes-optimal performance.

----

## [2267] Automatic and Harmless Regularization with Constrained and Lexicographic Optimization: A Dynamic Barrier Approach

**Authors**: *Chengyue Gong, Xingchao Liu, Qiang Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html)

**Abstract**:

Many machine learning tasks have to make a trade-off between two loss functions, typically the main data-fitness loss and an auxiliary loss. The most widely used approach is to optimize the linear combination of the objectives, which, however, requires manual tuning of the combination coefficient and is theoretically unsuitable for non-convex functions. In this work, we consider constrained optimization as a more principled approach for trading off two losses, with a special emphasis on lexicographic optimization, a degenerated limit of constrained optimization which optimizes a secondary loss inside the optimal set of the main loss. We propose a dynamic barrier gradient descent algorithm which provides a unified solution of both constrained and lexicographic optimization. We establish the convergence of the method for general non-convex functions.

----

## [2268] Corruption Robust Active Learning

**Authors**: *Yifang Chen, Simon S. Du, Kevin G. Jamieson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7b6bc883be91f56eb248d72de4d2847-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7b6bc883be91f56eb248d72de4d2847-Abstract.html)

**Abstract**:

We conduct theoretical studies on streaming-based active learning for binary classification under unknown adversarial label corruptions. In this setting, every time before the learner observes a sample, the adversary decides whether to corrupt the label ornot. First, we show that, in a benign corruption setting (which includes the misspecification setting as a special case),with a slight enlargement on the hypothesis elimination threshold, the classical RobustCAL framework can (surprisingly) achieve nearly the same label complexity guarantee as in the non-corrupted setting. However, this algorithm can fail in the general corruption setting. To resolve this drawback, we propose a new algorithm which is provably correct without any assumptions on the presence of corruptions. Furthermore, this algorithm enjoys the minimax label complexity in the non-corrupted setting (which is achieved by RobustCAL) and only requires $\tilde{\mathcal{O}}(C_{\mathrm{total}})$ additional labels in the corrupted setting to achieve $\mathcal{O}(\varepsilon + \frac{C_{\mathrm{total}}}{n})$, where $\varepsilon$ is the target accuracy, $C_{\mathrm{total}}$ is the total number of corruptions and $n$ is the total number of unlabeled samples.

----

## [2269] Metadata-based Multi-Task Bandits with Bayesian Hierarchical Models

**Authors**: *Runzhe Wan, Lin Ge, Rui Song*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7cfdde9db36af8e0d9a6d123d5c385e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7cfdde9db36af8e0d9a6d123d5c385e-Abstract.html)

**Abstract**:

How to explore efficiently is a central problem in multi-armed bandits. In this paper, we introduce the metadata-based multi-task bandit problem, where the agent needs to solve a large number of related multi-armed bandit tasks and can leverage some task-specific features (i.e., metadata) to share knowledge across tasks. As a general framework, we propose to capture task relations through the lens of Bayesian hierarchical models, upon which a Thompson sampling algorithm is designed to efficiently learn task relations, share information, and minimize the cumulative regrets. Two concrete examples for Gaussian bandits and Bernoulli bandits are carefully analyzed. The Bayes regret for Gaussian bandits clearly demonstrates the benefits of information sharing with our algorithm. The proposed method is further supported by extensive experiments.

----

## [2270] Program Synthesis Guided Reinforcement Learning for Partially Observed Environments

**Authors**: *Yichen Yang, Jeevana Priya Inala, Osbert Bastani, Yewen Pu, Armando Solar-Lezama, Martin C. Rinard*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7e2b2b75b04175610e5a00c1e221ebb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7e2b2b75b04175610e5a00c1e221ebb-Abstract.html)

**Abstract**:

A key challenge for reinforcement learning is solving long-horizon planning problems. Recent work has leveraged programs to guide reinforcement learning in these settings. However, these approaches impose a high manual burden on the user since they must provide a guiding program for every new task. Partially observed environments further complicate the programming task because the program must implement a strategy that correctly, and ideally optimally, handles every possible configuration of the hidden regions of the environment. We propose a new approach, model predictive program synthesis (MPPS), that uses program synthesis to automatically generate the guiding programs. It trains a generative model to predict the unobserved portions of the world, and then synthesizes a program based on samples from this model in a way that is robust to its uncertainty. In our experiments, we show that our approach significantly outperforms non-program-guided approaches on a set of challenging benchmarks, including a 2D Minecraft-inspired environment where the agent must complete a complex sequence of subtasks to achieve its goal, and achieves a similar performance as using handcrafted programs to guide the agent. Our results demonstrate that our approach can obtain the benefits of program-guided reinforcement learning without requiring the user to provide a new guiding program for every new task.

----

## [2271] Robust Allocations with Diversity Constraints

**Authors**: *Zeyu Shen, Lodewijk Gelauff, Ashish Goel, Aleksandra Korolova, Kamesh Munagala*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f7fbc4bafcc80cbf690acbef25f2ce1c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f7fbc4bafcc80cbf690acbef25f2ce1c-Abstract.html)

**Abstract**:

We consider the problem of allocating divisible items among multiple agents, and consider the setting where any agent is allowed to introduce {\emph diversity constraints} on the items they are allocated. We motivate this via settings where the items themselves correspond to user ad slots or task workers with attributes such as race and gender on which the principal seeks to achieve demographic parity. We consider the following question: When an agent expresses diversity constraints into an allocation rule, is the allocation of other agents hurt significantly? If this happens, the cost of introducing such constraints is disproportionately borne by agents who do not benefit from diversity. We codify this via two desiderata capturing {\em robustness}. These are {\emph no negative externality} -- other agents are not hurt -- and {\emph monotonicity} -- the agent enforcing the constraint does not see a large increase in value. We show in a formal sense that the Nash Welfare rule that maximizes product of agent values is {\emph uniquely} positioned to be robust when diversity constraints are introduced, while almost all other natural allocation rules fail this criterion. We also show that the guarantees achieved by Nash Welfare are nearly optimal within a widely studied class of allocation rules. We finally perform an empirical simulation on real-world data that models ad allocations to show that this gap between Nash Welfare and other rules persists in the wild.

----

## [2272] Activation Sharing with Asymmetric Paths Solves Weight Transport Problem without Bidirectional Connection

**Authors**: *Sunghyeon Woo, Jeongwoo Park, Jiwoo Hong, Dongsuk Jeon*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f80ebff16ccaa9b48a0224d7c489cef4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f80ebff16ccaa9b48a0224d7c489cef4-Abstract.html)

**Abstract**:

One of the reasons why it is difficult for the brain to perform backpropagation (BP) is the weight transport problem, which argues forward and feedback neurons cannot share the same synaptic weights during learning in biological neural networks. Recently proposed algorithms address the weight transport problem while providing good performance similar to BP in large-scale networks. However, they require bidirectional connections between the forward and feedback neurons to train their weights, which is observed to be rare in the biological brain. In this work, we propose an Activation Sharing algorithm that removes the need for bidirectional connections between the two types of neurons. In this algorithm, hidden layer outputs (activations) are shared across multiple layers during weight updates. By applying this learning rule to both forward and feedback networks, we solve the weight transport problem without the constraint of bidirectional connections, also achieving good performance even on deep convolutional neural networks for various datasets. In addition, our algorithm could significantly reduce memory access overhead when implemented in hardware.

----

## [2273] BlendGAN: Implicitly GAN Blending for Arbitrary Stylized Face Generation

**Authors**: *Mingcong Liu, Qiang Li, Zekui Qin, Guoxin Zhang, Pengfei Wan, Wen Zheng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f8417d04a0a2d5e1fb5c5253a365643c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f8417d04a0a2d5e1fb5c5253a365643c-Abstract.html)

**Abstract**:

Generative Adversarial Networks (GANs) have made a dramatic leap in high-fidelity image synthesis and stylized face generation. Recently, a layer-swapping mechanism has been developed to improve the stylization performance. However, this method is incapable of fitting arbitrary styles in a single model and requires hundreds of style-consistent training images for each style. To address the above issues, we propose BlendGAN for arbitrary stylized face generation by leveraging a flexible blending strategy and a generic artistic dataset. Specifically, we first train a self-supervised style encoder on the generic artistic dataset to extract the representations of arbitrary styles. In addition, a weighted blending module (WBM) is proposed to blend face and style representations implicitly and control the arbitrary stylization effect. By doing so, BlendGAN can gracefully fit arbitrary styles in a unified model while avoiding case-by-case preparation of style-consistent training images. To this end, we also present a novel large-scale artistic face dataset AAHQ. Extensive experiments demonstrate that BlendGAN outperforms state-of-the-art methods in terms of visual quality and style diversity for both latent-guided and reference-guided stylized face synthesis.

----

## [2274] Differentially Private Model Personalization

**Authors**: *Prateek Jain, John Rush, Adam D. Smith, Shuang Song, Abhradeep Guha Thakurta*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f8580959e35cb0934479bb007fb241c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f8580959e35cb0934479bb007fb241c2-Abstract.html)

**Abstract**:

We study personalization of supervised learning with user-level differential privacy. Consider a setting with many users, each of whom has a training data set drawn from their own distribution $P_i$. Assuming some shared structure among the problems $P_i$, can users collectively learn the shared structure---and solve their tasks better than they could individually---while preserving the privacy of their data? We formulate this question using joint, user-level differential privacy---that is, we control what is leaked about each user's entire data set. We provide algorithms that exploit popular non-private approaches in this domain like the Almost-No-Inner-Loop (ANIL) method, and give strong user-level privacy guarantees for our general approach. When the problems $P_i$ are linear regression problems with each user's regression vector lying in a common, unknown low-dimensional subspace, we show that our efficient algorithms satisfy nearly optimal estimation error guarantees. We also establish a general, information-theoretic upper bound via an exponential mechanism-based algorithm.

----

## [2275] Rates of Estimation of Optimal Transport Maps using Plug-in Estimators via Barycentric Projections

**Authors**: *Nabarun Deb, Promit Ghosal, Bodhisattva Sen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f862d13454fd267baa5fedfffb200567-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f862d13454fd267baa5fedfffb200567-Abstract.html)

**Abstract**:

Optimal transport maps between two probability distributions $\mu$ and $\nu$ on $\R^d$ have found extensive applications in both machine learning and statistics. In practice, these maps need to be estimated from data sampled according to $\mu$ and $\nu$. Plug-in estimators are perhaps most popular in estimating transport  maps in the field of computational optimal transport. In this paper, we provide a comprehensive analysis of the rates of convergences for general plug-in estimators defined via barycentric projections. Our main contribution is a new stability estimate for barycentric projections which proceeds under minimal smoothness assumptions and can be used to analyze general plug-in estimators. We illustrate the usefulness of this stability estimate by first providing rates of convergence for the natural discrete-discrete and semi-discrete estimators of  optimal transport maps. We then use the same stability estimate to show that, under additional smoothness assumptions of Besov type or Sobolev type, wavelet based or kernel smoothed plug-in estimators respectively speed up the rates of convergence and significantly mitigate the curse of dimensionality suffered by the natural discrete-discrete/semi-discrete estimators. As a by-product of our analysis, we also obtain faster rates of convergence for plug-in estimators of $W_2(\mu,\nu)$, the Wasserstein distance between $\mu$ and $\nu$, under the aforementioned smoothness assumptions, thereby complementing recent results in Chizat et al. (2020). Finally, we illustrate the applicability of our results in obtaining rates of convergence for Wasserstein barycenters between two probability distributions and obtaining asymptotic detection thresholds for some recent optimal-transport based tests of independence.

----

## [2276] Robust Generalization despite Distribution Shift via Minimum Discriminating Information

**Authors**: *Tobias Sutter, Andreas Krause, Daniel Kuhn*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f86890095c957e9b949d11d15f0d0cd5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f86890095c957e9b949d11d15f0d0cd5-Abstract.html)

**Abstract**:

Training models that perform well under distribution shifts is a central challenge in machine learning. In this paper, we introduce a modeling framework where, in addition to training data, we have partial structural knowledge of the shifted test distribution. We employ the principle of minimum discriminating information to embed the available prior knowledge, and use distributionally robust optimization to account for uncertainty due to the limited samples. By leveraging large deviation results, we obtain explicit generalization bounds with respect to the unknown shifted distribution. Lastly, we demonstrate the versatility of our framework by demonstrating it on two rather distinct applications: (1) training classifiers on systematically biased data and (2) off-policy evaluation in Markov Decision Processes.

----

## [2277] Soft Calibration Objectives for Neural Networks

**Authors**: *Archit Karandikar, Nicholas Cain, Dustin Tran, Balaji Lakshminarayanan, Jonathon Shlens, Michael C. Mozer, Becca Roelofs*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f8905bd3df64ace64a68e154ba72f24c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f8905bd3df64ace64a68e154ba72f24c-Abstract.html)

**Abstract**:

Optimal decision making requires that classifiers produce uncertainty estimates consistent with their empirical accuracy. However, deep neural networks are often under- or over-confident in their predictions. Consequently, methods have been developed to improve the calibration of their predictive uncertainty both during training and post-hoc. In this work, we propose differentiable losses to improve calibration based on a soft (continuous) version of the binning operation underlying popular calibration-error estimators. When incorporated into training, these soft calibration losses achieve state-of-the-art single-model ECE across multiple datasets with less than 1% decrease in accuracy. For instance, we observe an 82% reduction in ECE (70% relative to the post-hoc rescaled ECE) in exchange for a 0.7% relative decrease in accuracy relative to the cross entropy baseline on CIFAR-100.When incorporated post-training, the soft-binning-based calibration error objective improves upon temperature scaling, a popular recalibration method.  Overall, experiments across losses and datasets demonstrate that using calibration-sensitive procedures yield better uncertainty estimates under dataset shift than the standard practice of using a cross entropy loss and post-hoc recalibration methods.

----

## [2278] Distributional Gradient Matching for Learning Uncertain Neural Dynamics Models

**Authors**: *Lenart Treven, Philippe Wenk, Florian Dörfler, Andreas Krause*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f89394c979b34a25cc4ff8e11234fbfb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f89394c979b34a25cc4ff8e11234fbfb-Abstract.html)

**Abstract**:

Differential equations in general and neural ODEs in particular are an essential technique in continuous-time system identification. While many deterministic learning algorithms have been designed based on numerical integration via the adjoint method, many downstream tasks such as active learning, exploration in reinforcement learning, robust control, or filtering require accurate estimates of predictive uncertainties. In this work, we propose a novel approach towards estimating epistemically uncertain neural ODEs, avoiding the numerical integration bottleneck. Instead of modeling uncertainty in the ODE parameters, we directly model  uncertainties in the state space. Our algorithm distributional gradient matching (DGM) jointly trains a smoother and a dynamics model and matches their gradients via minimizing a Wasserstein loss. Our experiments show that, compared to traditional approximate inference methods based on numerical integration, our approach is faster to train, faster at predicting previously unseen trajectories, and in the context of neural ODEs, significantly more accurate.

----

## [2279] Shaping embodied agent behavior with activity-context priors from egocentric video

**Authors**: *Tushar Nagarajan, Kristen Grauman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f8b7aa3a0d349d9562b424160ad18612-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f8b7aa3a0d349d9562b424160ad18612-Abstract.html)

**Abstract**:

Complex physical tasks entail a sequence of object interactions, each with its own preconditions -- which can be difficult for robotic agents to learn efficiently solely through their own experience. We introduce an approach to discover activity-context priors from in-the-wild egocentric video captured with human worn cameras. For a given object, an activity-context prior represents the set of other compatible objects that are required for activities to succeed (e.g., a knife and cutting board brought together with a tomato are conducive to cutting). We encode our video-based prior as an auxiliary reward function that encourages an agent to bring compatible objects together before attempting an interaction. In this way, our model translates everyday human experience into embodied agent skills. We demonstrate our idea using egocentric EPIC-Kitchens video of people performing unscripted kitchen activities to benefit virtual household robotic agents performing various complex tasks in AI2-iTHOR, significantly accelerating agent learning.

----

## [2280] Adjusting for Autocorrelated Errors in Neural Networks for Time Series

**Authors**: *Fan-Keng Sun, Christopher I. Lang, Duane S. Boning*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f8e6ba1db0f3c4054afec1684ba8fb26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f8e6ba1db0f3c4054afec1684ba8fb26-Abstract.html)

**Abstract**:

An increasing body of research focuses on using neural networks to model time series. A common assumption in training neural networks via maximum likelihood estimation on time series is that the errors across time steps are uncorrelated. However, errors are actually autocorrelated in many cases due to the temporality of the data, which makes such maximum likelihood estimations inaccurate. In this paper, in order to adjust for autocorrelated errors, we propose to learn the autocorrelation coefficient jointly with the model parameters. In our experiments, we verify the effectiveness of our approach on time series forecasting. Results across a wide range of real-world datasets with various state-of-the-art models show that our method enhances performance in almost all cases. Based on these results, we suggest empirical critical values to determine the severity of autocorrelated errors. We also analyze several aspects of our method to demonstrate its advantages. Finally, other time series tasks are also considered to validate that our method is not restricted to only forecasting.

----

## [2281] A Geometric Analysis of Neural Collapse with Unconstrained Features

**Authors**: *Zhihui Zhu, Tianyu Ding, Jinxin Zhou, Xiao Li, Chong You, Jeremias Sulam, Qing Qu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html)

**Abstract**:

We provide the first global optimization landscape analysis of Neural Collapse -- an intriguing empirical phenomenon that arises in the last-layer classifiers and features of neural networks during the terminal phase of training. As recently reported by Papyan et al., this phenomenon implies that (i) the class means and the last-layer classifiers all collapse to the vertices of a Simplex Equiangular Tight Frame (ETF) up to scaling, and (ii) cross-example within-class variability of last-layer activations collapses to zero. We study the problem based on a simplified unconstrained feature model, which isolates the topmost layers from the classifier of the neural network. In this context, we show that the classical cross-entropy loss with weight decay has a benign global landscape, in the sense that the only global minimizers are the Simplex ETFs while all other critical points are strict saddles whose Hessian exhibit negative curvature directions. Our analysis of the simplified model not only explains what kind of features are learned in the last layer, but also shows why they can be efficiently optimized, matching the empirical observations in practical deep network architectures. These findings provide important practical implications. As an example, our experiments demonstrate that one may set the feature dimension equal to the number of classes and fix the last-layer classifier to be a Simplex ETF for network training, which reduces memory cost by over 20% on ResNet18 without sacrificing the generalization performance. The source code is available at https://github.com/tding1/Neural-Collapse.

----

## [2282] NeRS: Neural Reflectance Surfaces for Sparse-view 3D Reconstruction in the Wild

**Authors**: *Jason Y. Zhang, Gengshan Yang, Shubham Tulsiani, Deva Ramanan*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/f95ec3de395b4bce25b39ef6138da871-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/f95ec3de395b4bce25b39ef6138da871-Abstract.html)

**Abstract**:

Recent history has seen a tremendous growth of work exploring implicit representations of geometry and radiance, popularized through Neural Radiance Fields (NeRF).  Such works are fundamentally based on a (implicit) {\em volumetric} representation of occupancy, allowing them to model diverse scene structure including translucent objects and atmospheric obscurants. But because the vast majority of real-world scenes are composed of well-defined surfaces, we introduce a {\em surface} analog of such implicit models called Neural Reflectance Surfaces (NeRS). NeRS learns a neural shape representation of a closed surface that is diffeomorphic to a sphere, guaranteeing water-tight reconstructions. Even more importantly, surface parameterizations allow NeRS to learn (neural) bidirectional surface reflectance functions (BRDFs) that factorize view-dependent appearance into environmental illumination, diffuse color (albedo), and specular “shininess.” Finally, rather than illustrating our results on synthetic scenes or controlled in-the-lab capture, we assemble a novel dataset of multi-view images from online marketplaces for selling goods. Such “in-the-wild” multi-view image sets pose a number of challenges, including a small number of views with unknown/rough camera estimates. We demonstrate that surface-based neural reconstructions enable learning from such data, outperforming volumetric neural rendering-based reconstructions. We hope that NeRS serves as a first step toward building scalable, high-quality libraries of real-world shape, materials, and illumination.

----

## [2283] Unleashing the Power of Contrastive Self-Supervised Visual Models via Contrast-Regularized Fine-Tuning

**Authors**: *Yifan Zhang, Bryan Hooi, Dapeng Hu, Jian Liang, Jiashi Feng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fa14d4fe2f19414de3ebd9f63d5c0169-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fa14d4fe2f19414de3ebd9f63d5c0169-Abstract.html)

**Abstract**:

Contrastive self-supervised learning (CSL) has attracted increasing attention for model pre-training via unlabeled data. The resulted CSL models provide instance-discriminative visual features that are uniformly scattered in the feature space.  During deployment, the common practice is to directly fine-tune CSL models with cross-entropy, which however may not be the best strategy in practice. Although cross-entropy tends to separate inter-class features, the resulting models still have limited capability for reducing intra-class feature scattering that exists in CSL models. In this paper, we investigate whether applying contrastive learning to fine-tuning would bring further benefits, and analytically find that optimizing the contrastive loss benefits both discriminative representation learning and model optimization during fine-tuning. Inspired by these findings, we propose Contrast-regularized tuning (Core-tuning), a new approach for fine-tuning CSL models. Instead of simply adding the contrastive loss to the objective of fine-tuning, Core-tuning further applies a novel hard pair mining strategy for more effective contrastive fine-tuning, as well as smoothing the decision boundary to better exploit the learned discriminative feature space. Extensive experiments on image classification and semantic segmentation verify the effectiveness of Core-tuning.

----

## [2284] Discovery of Options via Meta-Learned Subgoals

**Authors**: *Vivek Veeriah, Tom Zahavy, Matteo Hessel, Zhongwen Xu, Junhyuk Oh, Iurii Kemaev, Hado van Hasselt, David Silver, Satinder Singh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fa246d0262c3925617b0c72bb20eeb1d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fa246d0262c3925617b0c72bb20eeb1d-Abstract.html)

**Abstract**:

Temporal abstractions in the form of options have been shown to help reinforcement learning (RL) agents learn faster. However, despite prior work on this topic, the problem of discovering options through interaction with an environment remains a challenge. In this paper, we introduce a novel meta-gradient approach for discovering useful options in multi-task RL environments. Our approach is based on a manager-worker decomposition of the RL agent, in which a manager maximises rewards from the environment by learning a task-dependent policy over both a set of task-independent discovered-options and primitive actions. The option-reward and termination functions that define a subgoal for each option are parameterised as neural networks and trained via meta-gradients to maximise their usefulness. Empirical analysis on gridworld and DeepMind Lab tasks show that: (1) our approach can discover meaningful and diverse temporally-extended options in multi-task RL domains, (2) the discovered options are frequently used by the agent while learning to solve the training tasks, and (3) that the discovered options help a randomly initialised manager learn faster in completely new tasks.

----

## [2285] Near-Optimal Lower Bounds For Convex Optimization For All Orders of Smoothness

**Authors**: *Ankit Garg, Robin Kothari, Praneeth Netrapalli, Suhail Sherif*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fa6c94460e902005a0b660266190c8ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fa6c94460e902005a0b660266190c8ba-Abstract.html)

**Abstract**:

We study the complexity of optimizing highly smooth convex functions. For a positive integer $p$, we want to find an $\epsilon$-approximate minimum of a convex function $f$, given oracle access to the function and its first $p$ derivatives, assuming that the $p$th derivative of $f$ is Lipschitz. Recently, three independent research groups (Jiang et al., PLMR 2019; Gasnikov et al., PLMR 2019; Bubeck et al., PLMR 2019) developed a new algorithm that solves this problem with $\widetilde{O}\left(1/\epsilon^{\frac{2}{3p+1}}\right)$ oracle calls for constant $p$. This is known to be optimal (up to log factors) for deterministic algorithms, but known lower bounds for randomized algorithms do not match this bound. We prove a new lower bound that matches this bound (up to log factors), and holds not only for randomized algorithms, but also for quantum algorithms.

----

## [2286] Topology-Imbalance Learning for Semi-Supervised Node Classification

**Authors**: *Deli Chen, Yankai Lin, Guangxiang Zhao, Xuancheng Ren, Peng Li, Jie Zhou, Xu Sun*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)

**Abstract**:

The class imbalance problem, as an important issue in learning node representations, has drawn increasing attention from the community. Although the imbalance considered by existing studies roots from the unequal quantity of labeled examples in different classes (quantity imbalance), we argue that graph data expose a unique source of imbalance from the asymmetric topological properties of the labeled nodes, i.e., labeled nodes are not equal in terms of their structural role in the graph (topology imbalance). In this work, we first probe the previously unknown topology-imbalance issue, including its characteristics, causes, and threats to semisupervised node classification learning. We then provide a unified view to jointly analyzing the quantity- and topology- imbalance issues by considering the node influence shift phenomenon with the Label Propagation algorithm. In light of our analysis, we devise an influence conflict detectionâ€“based metric Totoro to measure the degree of graph topology imbalance and propose a model-agnostic method ReNode to address the topology-imbalance issue by re-weighting the influence of labeled nodes adaptively based on their relative positions to class boundaries. Systematic experiments demonstrate the effectiveness and generalizability of our method in relieving topology-imbalance issue and promoting semi-supervised node classification. The further analysis unveils varied sensitivity of different graph neural networks (GNNs) to topology imbalance, which may serve as a new perspective in evaluating GNN architectures.

----

## [2287] Gradient Inversion with Generative Image Prior

**Authors**: *Jinwoo Jeon, Jaechang Kim, Kangwook Lee, Sewoong Oh, Jungseul Ok*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fa84632d742f2729dc32ce8cb5d49733-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fa84632d742f2729dc32ce8cb5d49733-Abstract.html)

**Abstract**:

Federated Learning (FL) is a distributed learning framework, in which the local data never leaves clients’ devices to preserve privacy, and the server trains models on the data via accessing only the gradients of those local data. Without further privacy mechanisms such as differential privacy, this leaves the system vulnerable against an attacker who inverts those gradients to reveal clients’ sensitive data. However, a gradient is often insufficient to reconstruct the user data without any prior knowledge. By exploiting a generative model pretrained on the data distribution, we demonstrate that data privacy can be easily breached. Further, when such prior knowledge is unavailable, we investigate the possibility of learning the prior from a sequence of gradients seen in the process of FL training. We experimentally show that the prior in a form of generative model is learnable from iterative interactions in FL. Our findings demonstrate that additional mechanisms are necessary to prevent privacy leakage in FL.

----

## [2288] Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Neural Network Robustness Verification

**Authors**: *Shiqi Wang, Huan Zhang, Kaidi Xu, Xue Lin, Suman Jana, Cho-Jui Hsieh, J. Zico Kolter*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fac7fead96dafceaf80c1daffeae82a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fac7fead96dafceaf80c1daffeae82a4-Abstract.html)

**Abstract**:

Bound propagation based incomplete neural network verifiers such as CROWN are very efficient and can significantly accelerate branch-and-bound (BaB) based complete verification of neural networks. However, bound propagation cannot fully handle the neuron split constraints introduced by BaB commonly handled by expensive linear programming (LP) solvers, leading to loose bounds and hurting verification efficiency. In this work, we develop $\beta$-CROWN, a new bound propagation based method that can fully encode neuron splits via optimizable parameters $\beta$ constructed from either primal or dual space. When jointly optimized in intermediate layers, $\beta$-CROWN generally produces better bounds than typical LP verifiers with neuron split constraints, while being as efficient and parallelizable as CROWN on GPUs. Applied to complete robustness verification benchmarks, $\beta$-CROWN with BaB is up to three orders of magnitude faster than LP-based BaB methods, and is notably faster than all existing approaches while producing lower timeout rates. By terminating BaB early, our method can also be used for efficient incomplete verification.  We consistently achieve higher verified accuracy in many settings compared to powerful incomplete verifiers, including those based on convex barrier breaking techniques. Compared to the typically tightest but very costly semidefinite programming (SDP) based incomplete verifiers, we obtain higher verified accuracy with three orders of magnitudes less verification time. Our algorithm empowered the $\alpha,\!\beta$-CROWN (alpha-beta-CROWN) verifier, the winning tool in VNN-COMP 2021. Our code is available at http://PaperCode.cc/BetaCROWN.

----

## [2289] Autobahn: Automorphism-based Graph Neural Nets

**Authors**: *Erik H. Thiede, Wenda Zhou, Risi Kondor*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/faf02b2358de8933f480a146f4d2d98e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/faf02b2358de8933f480a146f4d2d98e-Abstract.html)

**Abstract**:

We introduce Automorphism-based graph neural networks (Autobahn), a new family of graph neural networks. In an Autobahn, we decompose the graph into a collection of subgraphs and apply local convolutions that are equivariant to each subgraph's automorphism group. Specific choices of local neighborhoods and subgraphs recover existing architectures such as message passing neural networks. Our formalism also encompasses novel architectures: as an example, we introduce a graph neural network that decomposes the graph into paths and cycles. The resulting convolutions reflect the natural way that parts of the graph can transform, preserving the intuitive meaning of convolution without sacrificing global permutation equivariance. We validate our approach by applying Autobahn to molecular graphs, where it achieves results competitive with state-of-the-art message passing algorithms.

----

## [2290] Data Augmentation Can Improve Robustness

**Authors**: *Sylvestre-Alvise Rebuffi, Sven Gowal, Dan Andrei Calian, Florian Stimberg, Olivia Wiles, Timothy A. Mann*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fb4c48608ce8825b558ccf07169a3421-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fb4c48608ce8825b558ccf07169a3421-Abstract.html)

**Abstract**:

Adversarial training suffers from robust overfitting, a phenomenon where the robust test accuracy starts to decrease during training. In this paper, we focus on reducing robust overfitting by using common data augmentation schemes. We demonstrate that, contrary to previous findings, when combined with model weight averaging, data augmentation can significantly boost robust accuracy. Furthermore, we compare various augmentations techniques and observe that spatial composition techniques work the best for adversarial training. Finally, we evaluate our approach on CIFAR-10 against $\ell_\infty$ and $\ell_2$ norm-bounded perturbations of size $\epsilon = 8/255$ and $\epsilon = 128/255$, respectively. We show large absolute improvements of +2.93% and +2.16% in robust accuracy compared to previous state-of-the-art methods. In particular, against $\ell_\infty$ norm-bounded perturbations of size $\epsilon = 8/255$, our model reaches 60.07% robust accuracy without using any external data.  We also achieve a significant performance boost with this approach while using other architectures and datasets such as CIFAR-100, SVHN and TinyImageNet.

----

## [2291] Deep Explicit Duration Switching Models for Time Series

**Authors**: *Abdul Fatir Ansari, Konstantinos Benidis, Richard Kurle, Ali Caner Türkmen, Harold Soh, Alexander J. Smola, Bernie Wang, Tim Januschowski*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fb4c835feb0a65cc39739320d7a51c02-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fb4c835feb0a65cc39739320d7a51c02-Abstract.html)

**Abstract**:

Many complex time series can be effectively subdivided into distinct regimes that exhibit persistent dynamics. Discovering the switching behavior and the statistical patterns in these regimes is important for understanding the underlying dynamical system. We propose the Recurrent Explicit Duration Switching Dynamical System (RED-SDS), a flexible model that is capable of identifying both state- and time-dependent switching dynamics. State-dependent switching is enabled by a recurrent state-to-switch connection and an explicit duration count variable is used to improve the time-dependent switching behavior. We demonstrate how to perform efficient inference using a hybrid algorithm that approximates the posterior of the continuous states via an inference network and performs exact inference for the discrete switches and counts. The model is trained by maximizing a Monte Carlo lower bound of the marginal log-likelihood that can be computed efficiently as a byproduct of the inference routine. Empirical results on multiple datasets demonstrate that RED-SDS achieves  considerable improvement in time series segmentation and competitive forecasting performance against the state of the art.

----

## [2292] Shared Independent Component Analysis for Multi-Subject Neuroimaging

**Authors**: *Hugo Richard, Pierre Ablin, Bertrand Thirion, Alexandre Gramfort, Aapo Hyvärinen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fb508ef074ee78a0e58c68be06d8a2eb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fb508ef074ee78a0e58c68be06d8a2eb-Abstract.html)

**Abstract**:

We consider shared response modeling, a multi-view learning problem where one wants to identify common components from multiple datasets or views. We introduce Shared Independent Component Analysis (ShICA) that models eachview as a linear transform of shared independent components contaminated by additive Gaussian noise. We show that this model is identifiable if the components are either non-Gaussian or have enough diversity in noise variances. We then show that in some cases multi-set canonical correlation analysis can recover the correct unmixing matrices, but that even a small amount of sampling noise makes Multiset CCA fail. To solve this problem, we propose to use joint diagonalization after Multiset CCA, leading to a new approach called ShICA-J. We show via simulations that ShICA-J leads to improved results while being very fast to fit. While ShICA-J is based on second-order statistics, we further propose to leverage non-Gaussianity of the components using a maximum-likelihood method, ShICA-ML, that is both more accurate and more costly. Further, ShICA comes with a principled method for shared components estimation. Finally, we provide empirical evidence on fMRI and MEG datasets that ShICA yields more accurate estimation of the componentsthan alternatives.

----

## [2293] Shape from Blur: Recovering Textured 3D Shape and Motion of Fast Moving Objects

**Authors**: *Denys Rozumnyi, Martin R. Oswald, Vittorio Ferrari, Marc Pollefeys*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fb60d411a5c5b72b2e7d3527cfc84fd0-Abstract.html)

**Abstract**:

We address the novel task of jointly reconstructing the 3D shape, texture, and motion of an object from a single motion-blurred image. While previous approaches address the deblurring problem only in the 2D image domain, our proposed rigorous modeling of all object properties in the 3D domain enables the correct description of arbitrary object motion. This leads to significantly better image decomposition and sharper deblurring results. We model the observed appearance of a motion-blurred object as a combination of the background and a 3D object with constant translation and rotation. Our method minimizes a loss on reconstructing the input image via differentiable rendering with suitable regularizers. This enables estimating the textured 3D mesh of the blurred object with high fidelity. Our method substantially outperforms competing approaches on several benchmarks for fast moving objects deblurring. Qualitative results show that the reconstructed 3D mesh generates high-quality temporal super-resolution and novel views of the deblurred object.

----

## [2294] Batched Thompson Sampling

**Authors**: *Cem Kalkanli, Ayfer Özgür*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fb647ca6672b0930e9d00dc384d8b16f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fb647ca6672b0930e9d00dc384d8b16f-Abstract.html)

**Abstract**:

We introduce a novel anytime batched Thompson sampling policy for multi-armed bandits where the agent observes the rewards of her actions and adjusts her policy only at the end of a small number of batches. We show that this policy simultaneously achieves a problem dependent regret of order $O(\log(T))$ and a minimax regret of order $O(\sqrt{T\log(T)})$  while the number of batches can be bounded by $O(\log(T))$ independent of the problem instance over a time horizon $T$. We also prove that in expectation the instance dependent batch complexity of our policy is of order $O(\log\log(T))$. These results indicate that Thompson sampling performs competitively with recently proposed algorithms for the batched setting, which optimize the batch structure for a given  time horizon $T$ and prioritize exploration in the beginning of the experiment to eliminate suboptimal actions. Unlike these algorithms, the batched Thompson sampling algorithm we propose is an anytime policy, i.e. it operates without the knowledge of the time horizon $T$, and  as such it is the only anytime algorithm that achieves optimal regret with $O(\log\log(T))$ expected batch complexity. This is achieved through a dynamic batching strategy, which uses the agents estimates to adaptively increase the batch duration.

----

## [2295] Delayed Gradient Averaging: Tolerate the Communication Latency for Federated Learning

**Authors**: *Ligeng Zhu, Hongzhou Lin, Yao Lu, Yujun Lin, Song Han*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc03d48253286a798f5116ec00e99b2b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc03d48253286a798f5116ec00e99b2b-Abstract.html)

**Abstract**:

Federated Learning is an emerging direction in distributed machine learning that en-ables jointly training a model without sharing the data. Since the data is distributed across many edge devices through wireless / long-distance connections, federated learning suffers from inevitable high communication latency. However, the latency issues are undermined in the current literature [15] and existing approaches suchas FedAvg [27] become less efficient when the latency increases.  To over comethe problem, we propose \textbf{D}elayed \textbf{G}radient \textbf{A}veraging (DGA), which delays the averaging step to improve efficiency and allows local computation in parallel tocommunication. We theoretically prove that DGA attains a similar convergence rate as FedAvg, and empirically show that our algorithm can tolerate high network latency without compromising accuracy. Specifically, we benchmark the training speed on various vision (CIFAR, ImageNet) and language tasks (Shakespeare),with both IID and non-IID partitions, and show DGA can bring 2.55$\times$ to 4.07$\times$ speedup. Moreover, we built a 16-node Raspberry Pi cluster and show that DGA can consistently speed up real-world federated learning applications.

----

## [2296] Focal Attention for Long-Range Interactions in Vision Transformers

**Authors**: *Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Xiyang Dai, Bin Xiao, Lu Yuan, Jianfeng Gao*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc1a36821b02abbd2503fd949bfc9131-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc1a36821b02abbd2503fd949bfc9131-Abstract.html)

**Abstract**:

Recently, Vision Transformer and its variants have shown great promise on various computer vision tasks. The ability to capture local and global visual dependencies through self-attention is the key to its success. But it also brings challenges due to quadratic computational overhead, especially for the high-resolution vision tasks(e.g., object detection). Many recent works have attempted to reduce the cost and improve model performance by applying either coarse-grained global attention or fine-grained local attention. However, both approaches cripple the modeling power of the original self-attention mechanism of multi-layer Transformers, leading to sub-optimal solutions.  In this paper, we present focal attention, a new attention mechanism that incorporates both fine-grained local and coarse-grained global interactions.  In this new mechanism, each token attends its closest surrounding tokens at the fine granularity and the tokens far away at a coarse granularity and thus can capture both short- and long-range visual dependencies efficiently and effectively. With focal attention, we propose a new variant of Vision Transformer models, called Focal Transformers, which achieve superior performance over the state-of-the-art (SoTA) Vision Transformers on a range of public image classification and object detection benchmarks.  In particular, our Focal Transformer models with a moderate size of 51.1M and a large size of 89.8M achieve 83.6% and 84.0%Top-1 accuracy, respectively, on ImageNet classification at 224Ã—224.  When employed as the backbones, Focal Transformers achieve consistent and substantial improvements over the current SoTA Swin Transformers [44] across 6 different object detection methods.  Our largest Focal Transformer yields58.7/59.0boxmAPs and50.9/51.3mask mAPs on COCO mini-val/test-dev, and55.4mIoU onADE20K for semantic segmentation, creating new SoTA on three of the most challenging computer vision tasks.

----

## [2297] Scalable and Stable Surrogates for Flexible Classifiers with Fairness Constraints

**Authors**: *Harry Bendekgey, Erik B. Sudderth*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc2e6a440b94f64831840137698021e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc2e6a440b94f64831840137698021e1-Abstract.html)

**Abstract**:

We investigate how fairness relaxations scale to flexible classifiers like deep neural networks for images and text. We analyze an easy-to-use and robust way of imposing fairness constraints when training, and through this framework prove that some prior fairness surrogates exhibit degeneracies for non-convex models.  We resolve these problems via three new surrogates: an adaptive data re-weighting, and two smooth upper-bounds that are provably more robust than some previous methods. Our surrogates perform comparably to the state-of-the-art on low-dimensional fairness benchmarks, while achieving superior accuracy and stability for more complex computer vision and natural language processing tasks.

----

## [2298] Residual Pathway Priors for Soft Equivariance Constraints

**Authors**: *Marc Finzi, Greg Benton, Andrew Gordon Wilson*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc394e9935fbd62c8aedc372464e1965-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc394e9935fbd62c8aedc372464e1965-Abstract.html)

**Abstract**:

Models such as convolutional neural networks restrict the hypothesis space to a set of functions satisfying equivariance constraints, and improve generalization in problems by capturing relevant symmetries. However, symmetries are often only partially respected, preventing models with restriction biases from fitting the data. We introduce Residual Pathway Priors (RPPs) as a method for converting hard architectural constraints into soft priors, guiding models towards structured solutions while retaining the ability to capture additional complexity. RPPs are resilient to approximate or misspecified symmetries, and are as effective as fully constrained models even when symmetries are exact. We show that RPPs provide compelling performance on both model-free and model-based reinforcement learning problems, where contact forces and directional rewards violate the assumptions of equivariant networks. Finally, we demonstrate that RPPs have broad applicability, including dynamical systems, regression, and classification.

----

## [2299] Optimal Algorithms for Stochastic Contextual Preference Bandits

**Authors**: *Aadirupa Saha*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc3cf452d3da8402bebb765225ce8c0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc3cf452d3da8402bebb765225ce8c0e-Abstract.html)

**Abstract**:

We consider the problem of preference bandits in the contextual setting. At each round, the learner is presented with a context set of $K$ items, chosen randomly from a potentially infinite set of arms $\mathcal D \subseteq \mathbf R^d$. However, unlike classical contextual bandits, our framework only allows the learner to receive feedback in terms of item preferences: At each round, the learner is allowed to play a subset of size $q$ (any $q \in \{2,\ldots,K\}$) upon which only a (noisy) winner of the subset is revealed. Yet, same as the classical setup, the goal is still to compete against the best context arm at each round. The problem is relevant in various online decision-making scenarios, including recommender systems, information retrieval, tournament ranking--typically any application where it's easier to elicit the items' relative strength instead of their absolute scores. To the best of our knowledge, this work is the first to consider preference-based stochastic contextual bandits for potentially infinite decision spaces. We start with presenting two algorithms for the special case of pairwise preferences $(q=2)$: The first algorithm is simple and easy to implement with an $\tilde O(d\sqrt{T})$ regret guarantee, while the second algorithm is shown to achieve the optimal $\tilde O(\sqrt{dT})$ regret, as follows from our $\Omega(\sqrt {dT})$ matching lower bound analysis. We then proceed to analyze the problem for any general $q$-subsetwise preferences ($q \ge 2$), where surprisingly, our lower bound proves the fundamental performance limit to be $\Omega(\sqrt{d T})$ yet again, independent of the subsetsize $q$. Following this, we propose a matching upper bound algorithm justifying the tightness of our results. This implies having access to subsetwise preferences does not help in faster information aggregation for our feedback model. All the results are corroborated empirically against existing baselines.

----

## [2300] Tight High Probability Bounds for Linear Stochastic Approximation with Fixed Stepsize

**Authors**: *Alain Durmus, Eric Moulines, Alexey Naumov, Sergey Samsonov, Kevin Scaman, Hoi-To Wai*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc95fa5740ba01a870cfa52f671fe1e4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc95fa5740ba01a870cfa52f671fe1e4-Abstract.html)

**Abstract**:

This paper provides a non-asymptotic analysis of linear stochastic approximation (LSA) algorithms with fixed stepsize. This family of methods arises in many machine learning tasks and is used to obtain approximate solutions of a linear system $\bar{A}\theta = \bar{b}$ for which $\bar{A}$ and $\bar{b}$ can only be accessed through random estimates $\{({\bf A}_n, {\bf b}_n): n \in \mathbb{N}^*\}$.  Our analysis is based on new results regarding moments and high probability bounds for products of matrices which are shown to be tight. We derive high probability bounds on the performance of LSA under weaker conditions on the sequence $\{({\bf A}_n, {\bf b}_n): n \in \mathbb{N}^*\}$ than previous works. However, in contrast, we establish polynomial concentration bounds with order depending on the stepsize. We show that our conclusions cannot be improved  without additional assumptions on the sequence of random matrices $\{{\bf A}_n: n \in \mathbb{N}^*\}$, and in particular that no Gaussian or exponential high probability bounds can hold.  Finally, we pay a particular attention to establishing  bounds with sharp order with respect to the number of iterations and the stepsize and  whose leading terms contain the covariance matrices appearing in the central limit theorems.

----

## [2301] Learning Large Neighborhood Search Policy for Integer Programming

**Authors**: *Yaoxin Wu, Wen Song, Zhiguang Cao, Jie Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fc9e62695def29ccdb9eb3fed5b4c8c8-Abstract.html)

**Abstract**:

We propose a deep reinforcement learning (RL) method to learn large neighborhood search (LNS) policy for integer programming (IP). The RL policy is trained as the destroy operator to select a subset of variables at each step, which is reoptimized by an IP solver as the repair operator. However, the combinatorial number of variable subsets prevents direct application of typical RL algorithms. To tackle this challenge, we represent all subsets by factorizing them into binary decisions on each variable. We then design a neural network to learn policies for each variable in parallel, trained by a customized actor-critic algorithm. We evaluate the proposed method on four representative IP problems. Results show that it can find better solutions than SCIP in much less time, and significantly outperform other LNS baselines with the same runtime. Moreover, these advantages notably persist when the policies generalize to larger problems. Further experiments with Gurobi also reveal that our method can outperform this state-of-the-art commercial solver within the same time limit.

----

## [2302] Dynamic Trace Estimation

**Authors**: *Prathamesh Dharangutte, Christopher Musco*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fcdf698a5d673435e0a5a6f9ffea05ca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fcdf698a5d673435e0a5a6f9ffea05ca-Abstract.html)

**Abstract**:

We study a dynamic version of the implicit trace estimation problem. Given access to an oracle for computing matrix-vector multiplications with a dynamically changing matrix A, our goal is to maintain an accurate approximation to A's trace using as few multiplications as possible. We present a practical algorithm for solving this problem and prove that, in a natural setting, its complexity is quadratically better than the standard solution of repeatedly applying Hutchinson's stochastic trace estimator. We also provide an improved algorithm assuming additional common assumptions on A's dynamic updates. We support our theory with empirical results, showing significant computational improvements on three applications in machine learning and network science: tracking moments of the Hessian spectral density during neural network optimization, counting triangles and estimating natural connectivity in a dynamically changing graph.

----

## [2303] Provable Representation Learning for Imitation with Contrastive Fourier Features

**Authors**: *Ofir Nachum, Mengjiao Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fd00d3474e495e7b6d5f9f575b2d7ec4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fd00d3474e495e7b6d5f9f575b2d7ec4-Abstract.html)

**Abstract**:

In imitation learning, it is common to learn a behavior policy to match an unknown target policy via max-likelihood training on a collected set of target demonstrations. In this work, we consider using offline experience datasets -- potentially far from the target distribution -- to learn low-dimensional state representations that provably accelerate the sample-efficiency of downstream imitation learning. A central challenge in this setting is that the unknown target policy itself may not exhibit low-dimensional behavior, and so there is a potential for the representation learning objective to alias states in which the target policy acts differently. Circumventing this challenge, we derive a representation learning objective that provides an upper bound on the performance difference between the target policy and a low-dimensional policy trained with max-likelihood, and this bound is tight regardless of whether the target policy itself exhibits low-dimensional structure. Moving to the practicality of our method, we show that our objective can be implemented as contrastive learning, in which the transition dynamics are approximated by either an implicit energy-based model or, in some special cases, an implicit linear model with representations given by random Fourier features. Experiments on both tabular environments and high-dimensional Atari games provide quantitative evidence for the practical benefits of our proposed objective.

----

## [2304] MICo: Improved representations via sampling-based state similarity for Markov decision processes

**Authors**: *Pablo Samuel Castro, Tyler Kastner, Prakash Panangaden, Mark Rowland*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fd06b8ea02fe5b1c2496fe1700e9d16c-Abstract.html)

**Abstract**:

We present a new behavioural distance over the state space of a Markov decision process, and demonstrate the use of this distance as an effective means of shaping the learnt representations of deep reinforcement learning agents. While existing notions of state similarity are typically difficult to learn at scale due to high computational cost and lack of sample-based algorithms, our newly-proposed distance addresses both of these issues. In addition to providing detailed theoretical analyses, we provide empirical evidence that learning this distance alongside the value function yields structured and informative representations, including strong results on the Arcade Learning Environment benchmark.

----

## [2305] Counterfactual Explanations in Sequential Decision Making Under Uncertainty

**Authors**: *Stratis Tsirtsis, Abir De, Manuel Rodriguez*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fd0a5a5e367a0955d81278062ef37429-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fd0a5a5e367a0955d81278062ef37429-Abstract.html)

**Abstract**:

Methods to find counterfactual explanations have predominantly focused on one-step decision making processes. In this work, we initiate the development of methods to find counterfactual explanations for decision making processes in which multiple, dependent actions are taken sequentially over time. We start by formally characterizing a sequence of actions and states using finite horizon Markov decision processes and the Gumbel-Max structural causal model. Building upon this characterization, we formally state the problem of finding counterfactual explanations for sequential decision making processes. In our problem formulation, the counterfactual explanation specifies an alternative sequence of actions differing in at most k actions from the observed sequence that could have led the observed process realization to a better outcome. Then, we introduce a polynomial time algorithm based on dynamic programming to build a counterfactual policy that is guaranteed to always provide the optimal counterfactual explanation on every possible realization of the counterfactual environment dynamics. We validate our algorithm using both synthetic and real data from cognitive behavioral therapy and show that the counterfactual explanations our algorithm finds can provide valuable insights to enhance sequential decision making under uncertainty.

----

## [2306] Streaming Linear System Identification with Reverse Experience Replay

**Authors**: *Prateek Jain, Suhas S. Kowshik, Dheeraj Nagaraj, Praneeth Netrapalli*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fd2c5e4680d9a01dba3aada5ece22270-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fd2c5e4680d9a01dba3aada5ece22270-Abstract.html)

**Abstract**:

We consider the problem of estimating a linear time-invariant (LTI) dynamical system from a single trajectory via streaming algorithms, which is encountered in several applications including reinforcement learning (RL) and time-series analysis. While the LTI system estimation problem is well-studied in the {\em offline} setting, the practically important streaming/online setting has received little attention. Standard streaming methods like stochastic gradient descent (SGD) are unlikely to work since streaming points can be highly correlated. In this work, we propose a novel streaming algorithm, SGD with Reverse Experience Replay (SGD-RER), that is inspired by the experience replay (ER)  technique popular in the RL literature. SGD-RER divides data into small buffers and runs SGD backwards on the data stored in the individual buffers. We show that this algorithm exactly deconstructs the dependency structure and obtains information theoretically optimal guarantees for both parameter error and prediction error. Thus, we provide the first -- to the best of our knowledge -- optimal SGD-style algorithm for the classical problem of linear system identification with a first order oracle. Furthermore, SGD-RER can be applied to more general settings like sparse LTI identification with known sparsity pattern, and  non-linear dynamical systems. Our work demonstrates that the knowledge of data dependency structure can aid us in designing statistically and computationally efficient algorithms which can ``decorrelate'' streaming samples.

----

## [2307] SmoothMix: Training Confidence-calibrated Smoothed Classifiers for Certified Robustness

**Authors**: *Jongheon Jeong, Sejun Park, Minkyu Kim, Heung-Chang Lee, Do-Guk Kim, Jinwoo Shin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fd45ebc1e1d76bc1fe0ba933e60e9957-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fd45ebc1e1d76bc1fe0ba933e60e9957-Abstract.html)

**Abstract**:

Randomized smoothing is currently a state-of-the-art method to construct a certifiably robust classifier from neural networks against $\ell_2$-adversarial perturbations. Under the paradigm, the robustness of a classifier is aligned with the prediction confidence, i.e., the higher confidence from a smoothed classifier implies the better robustness. This motivates us to rethink the fundamental trade-off between accuracy and robustness in terms of calibrating confidences of a smoothed classifier. In this paper, we propose a simple training scheme, coined SmoothMix, to control the robustness of smoothed classifiers via self-mixup: it trains on convex combinations of samples along the direction of adversarial perturbation for each input. The proposed procedure effectively identifies over-confident, near off-class samples as a cause of limited robustness in case of smoothed classifiers, and offers an intuitive way to adaptively set a new decision boundary between these samples for better robustness. Our experimental results demonstrate that the proposed method can significantly improve the certified $\ell_2$-robustness of smoothed classifiers compared to existing state-of-the-art robust training methods.

----

## [2308] Action-guided 3D Human Motion Prediction

**Authors**: *Jiangxin Sun, Zihang Lin, Xintong Han, Jian-Fang Hu, Jia Xu, Wei-Shi Zheng*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fd9dd764a6f1d73f4340d570804eacc4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fd9dd764a6f1d73f4340d570804eacc4-Abstract.html)

**Abstract**:

The ability of forecasting future human motion is important for human-machine interaction systems to understand human behaviors and make interaction. In this work, we focus on developing models to predict future human motion from past observed video frames. Motivated by the observation that human motion is closely related to the action being performed, we propose to explore action context to guide motion prediction. Specifically, we construct an action-specific memory bank to store representative motion dynamics for each action category, and design a query-read process to retrieve some motion dynamics from the memory bank. The retrieved dynamics are consistent with the action depicted in the observed video frames and serve as a strong prior knowledge to guide motion prediction. We further formulate an action constraint loss to ensure the global semantic consistency of the predicted motion. Extensive experiments demonstrate the effectiveness of the proposed approach, and we achieve state-of-the-art performance on 3D human motion prediction.

----

## [2309] Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks

**Authors**: *Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fdb55ce855129e05da8374059cc82728-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fdb55ce855129e05da8374059cc82728-Abstract.html)

**Abstract**:

Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.

----

## [2310] Validating the Lottery Ticket Hypothesis with Inertial Manifold Theory

**Authors**: *Zeru Zhang, Jiayin Jin, Zijie Zhang, Yang Zhou, Xin Zhao, Jiaxiang Ren, Ji Liu, Lingfei Wu, Ruoming Jin, Dejing Dou*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fdc42b6b0ee16a2f866281508ef56730-Abstract.html)

**Abstract**:

Despite achieving remarkable efficiency, traditional network pruning techniques often follow manually-crafted heuristics to generate pruned sparse networks. Such heuristic pruning strategies are hard to guarantee that the pruned networks achieve test accuracy comparable to the original dense ones. Recent works have empirically identified and verified the Lottery Ticket Hypothesis (LTH): a randomly-initialized dense neural network contains an extremely sparse subnetwork, which can be trained to achieve similar accuracy to the former. Due to the lack of theoretical evidence, they often need to run multiple rounds of expensive training and pruning over the original large networks to discover the sparse subnetworks with low accuracy loss. By leveraging dynamical systems theory and inertial manifold theory, this work theoretically verifies the validity of the LTH. We explore the possibility of theoretically lossless pruning as well as one-time pruning, compared with existing neural network pruning and LTH techniques. We reformulate the neural network optimization problem as a gradient dynamical system and reduce this high-dimensional system onto inertial manifolds to obtain a low-dimensional system regarding pruned subnetworks. We demonstrate the precondition and existence of pruned subnetworks and prune the original networks in terms of the gap in their spectrum that make the subnetworks have the smallest dimensions.

----

## [2311] Are My Deep Learning Systems Fair? An Empirical Study of Fixed-Seed Training

**Authors**: *Shangshu Qian, Hung Viet Pham, Thibaud Lutellier, Zeou Hu, Jungwon Kim, Lin Tan, Yaoliang Yu, Jiahao Chen, Sameena Shah*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fdda6e957f1e5ee2f3b311fe4f145ae1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fdda6e957f1e5ee2f3b311fe4f145ae1-Abstract.html)

**Abstract**:

Deep learning (DL) systems have been gaining popularity in critical tasks such as credit evaluation and crime prediction. Such systems demand fairness. Recent work shows that DL software implementations introduce variance: identical DL training runs (i.e., identical network, data, configuration, software, and hardware) with a fixed seed produce different models. Such variance could make DL models and networks violate fairness compliance laws, resulting in negative social impact. In this paper, we conduct the first empirical study to quantify the impact of software implementation on the fairness and its variance of DL systems. Our study of 22 mitigation techniques and five baselines reveals up to 12.6% fairness variance across identical training runs with identical seeds. In addition, most debiasing algorithms have a negative impact on the model such as reducing model accuracy, increasing fairness variance, or increasing accuracy variance. Our literature survey shows that while fairness is gaining popularity in artificial intelligence (AI) related conferences, only 34.4% of the papers use multiple identical training runs to evaluate their approach, raising concerns about their resultsâ€™ validity. We call for better fairness evaluation and testing protocols to improve fairness and fairness variance of DL systems as well as DL research validity and reproducibility at large.

----

## [2312] Rectangular Flows for Manifold Learning

**Authors**: *Anthony L. Caterini, Gabriel Loaiza-Ganem, Geoff Pleiss, John P. Cunningham*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fde9264cf376fffe2ee4ddf4a988880d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fde9264cf376fffe2ee4ddf4a988880d-Abstract.html)

**Abstract**:

Normalizing flows are invertible neural networks with tractable change-of-volume terms, which allow optimization of their parameters to be efficiently performed via maximum likelihood. However, data of interest are typically assumed to live in some (often unknown) low-dimensional manifold embedded in a high-dimensional ambient space. The result is a modelling mismatch since -- by construction -- the invertibility requirement implies high-dimensional support of the learned distribution. Injective flows, mappings from low- to high-dimensional spaces, aim to fix this discrepancy by learning distributions on manifolds, but the resulting volume-change term becomes more challenging to evaluate. Current approaches either avoid computing this term entirely using various heuristics, or assume the manifold is known beforehand and therefore are not widely applicable. Instead, we propose two methods to tractably calculate the gradient of this term with respect to the parameters of the model, relying on careful use of automatic differentiation and techniques from numerical linear algebra. Both approaches perform end-to-end nonlinear manifold learning and density estimation for data projected onto this manifold. We study the trade-offs between our proposed methods, empirically verify that we outperform approaches ignoring the volume-change term by more accurately learning manifolds and the corresponding distributions on them, and show promising results on out-of-distribution detection. Our code is available at https://github.com/layer6ai-labs/rectangular-flows.

----

## [2313] On the Generative Utility of Cyclic Conditionals

**Authors**: *Chang Liu, Haoyue Tang, Tao Qin, Jintao Wang, Tie-Yan Liu*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe04e05fbe48920b8ba90bea2ddfe60b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe04e05fbe48920b8ba90bea2ddfe60b-Abstract.html)

**Abstract**:

We study whether and how can we model a joint distribution $p(x,z)$ using two conditional models $p(x|z)$ and $q(z|x)$ that form a cycle. This is motivated by the observation that deep generative models, in addition to a likelihood model $p(x|z)$, often also use an inference model $q(z|x)$ for extracting representation, but they rely on a usually uninformative prior distribution $p(z)$ to define a joint distribution, which may render problems like posterior collapse and manifold mismatch. To explore the possibility to model a joint distribution using only $p(x|z)$ and $q(z|x)$, we study their compatibility and determinacy, corresponding to the existence and uniqueness of a joint distribution whose conditional distributions coincide with them. We develop a general theory for operable equivalence criteria for compatibility, and sufficient conditions for determinacy. Based on the theory, we propose a novel generative modeling framework CyGen that only uses the two cyclic conditional models. We develop methods to achieve compatibility and determinacy, and to use the conditional models to fit and generate data. With the prior constraint removed, CyGen better fits data and captures more representative features, supported by both synthetic and real-world experiments.

----

## [2314] Structural Credit Assignment in Neural Networks using Reinforcement Learning

**Authors**: *Dhawal Gupta, Gabor Mihucz, Matthew Schlegel, James E. Kostas, Philip S. Thomas, Martha White*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe1f9c70bdf347497e1a01b6c486bdb9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe1f9c70bdf347497e1a01b6c486bdb9-Abstract.html)

**Abstract**:

Structural credit assignment in neural networks is a long-standing problem, with a variety of alternatives to backpropagation proposed to allow for local training of nodes. One of the early strategies was to treat each node as an agent and use a reinforcement learning method called REINFORCE to update each node locally with only a global reward signal. In this work, we revisit this approach and investigate if we can leverage other reinforcement learning approaches to improve learning. We first formalize training a neural network as a finite-horizon reinforcement learning problem and discuss how this facilitates using ideas from reinforcement learning like off-policy learning. We show that the standard on-policy REINFORCE approach, even with a variety of variance reduction approaches, learns suboptimal solutions. We introduce an off-policy approach, to facilitate reasoning about the greedy action for other agents and help overcome stochasticity in other agents. We conclude by showing that these networks of agents can be more robust to correlated samples when learning online.

----

## [2315] A Near-Optimal Algorithm for Stochastic Bilevel Optimization via Double-Momentum

**Authors**: *Prashant Khanduri, Siliang Zeng, Mingyi Hong, Hoi-To Wai, Zhaoran Wang, Zhuoran Yang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe2b421b8b5f0e7c355ace66a9fe0206-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe2b421b8b5f0e7c355ace66a9fe0206-Abstract.html)

**Abstract**:

This paper proposes a new algorithm -- the  \underline{S}ingle-timescale Do\underline{u}ble-momentum \underline{St}ochastic \underline{A}pprox\underline{i}matio\underline{n} (SUSTAIN) -- for tackling stochastic unconstrained bilevel optimization problems. We focus on bilevel problems where the lower level subproblem is strongly-convex and the upper level objective function is smooth. Unlike prior works which rely on \emph{two-timescale} or \emph{double loop} techniques, we design a stochastic momentum-assisted gradient estimator for both the upper and lower level updates. The latter allows us to control the error in the stochastic gradient updates due to inaccurate solution to both subproblems. If the upper objective function is smooth but possibly non-convex, we show that {SUSTAIN}~requires $O(\epsilon^{-3/2})$  iterations (each using $O(1)$ samples) to find an $\epsilon$-stationary solution. The $\epsilon$-stationary solution is defined as the point whose squared norm of the gradient of the outer function is less than or equal to $\epsilon$.  The total number of stochastic gradient samples required for the upper and lower level objective functions matches the best-known complexity for single-level stochastic gradient algorithms. We also analyze the case when the upper level objective function is strongly-convex.

----

## [2316] Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels

**Authors**: *Erik Englesson, Hossein Azizpour*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe2d010308a6b3799a3d9c728ee74244-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe2d010308a6b3799a3d9c728ee74244-Abstract.html)

**Abstract**:

Prior works have found it beneficial to combine provably noise-robust loss functions e.g., mean absolute error (MAE) with standard categorical loss function e.g. cross entropy (CE) to improve their learnability. Here, we propose to use Jensen-Shannon divergence as a noise-robust loss function and show that it interestingly interpolate between CE and MAE with a controllable mixing parameter. Furthermore, we make a crucial observation that CE exhibit lower consistency around noisy data points. Based on this observation, we adopt a generalized version of the Jensen-Shannon divergence for multiple distributions to encourage consistency around data points. Using this loss function, we show state-of-the-art results on both synthetic (CIFAR), and real-world (e.g., WebVision) noise with varying noise rates.

----

## [2317] Continual Learning via Local Module Composition

**Authors**: *Oleksiy Ostapenko, Pau Rodríguez, Massimo Caccia, Laurent Charlin*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe5e7cb609bdbe6d62449d61849c38b0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe5e7cb609bdbe6d62449d61849c38b0-Abstract.html)

**Abstract**:

Modularity is a compelling solution to continual learning (CL), the problem of modeling sequences of related tasks. Learning and then composing modules to solve different tasks provides an abstraction to address the principal challenges of CL including catastrophic forgetting, backward and forward transfer across tasks, and sub-linear model growth. We introduce local module composition (LMC), an approach to modular CL where each module is provided a local structural component that estimates a moduleâ€™s relevance to the input. Dynamic module composition is performed layer-wise based on local relevance scores. We demonstrate that agnosticity to task identities (IDs) arises from (local) structural learning that is module-specific as opposed to the task- and/or model-specific as in previous works, making LMC applicable to more CL settings compared to previous works. In addition, LMC also tracks statistics about the input distribution and adds new modules when outlier samples are detected. In the first set of experiments, LMC performs favorably compared to existing methods on the recent Continual Transfer-learning Benchmark without requiring task identities. In another study, we show that the locality of structural learning allows LMC to interpolate to related but unseen tasks (OOD), as well as to compose modular networks trained independently on different task sequences into a third modular network without any fine-tuning. Finally, in search for limitations of LMC we study it on more challenging sequences of 30 and 100 tasks, demonstrating that local module selection becomes much more challenging in presence of a large number of candidate modules. In this setting best performing LMC spawns much fewer modules compared to an oracle based baseline, however, it reaches a lower overall accuracy. The codebase is available under https://github.com/oleksost/LMC.

----

## [2318] Model-Based Episodic Memory Induces Dynamic Hybrid Controls

**Authors**: *Hung Le, Thommen George Karimpanal, Majid Abdolshah, Truyen Tran, Svetha Venkatesh*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe73f687e5bc5280214e0486b273a5f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe73f687e5bc5280214e0486b273a5f9-Abstract.html)

**Abstract**:

Episodic control enables sample efficiency in reinforcement learning by recalling past experiences from an episodic memory. We propose a new model-based episodic memory of trajectories addressing current limitations of episodic control. Our memory estimates trajectory values, guiding the agent towards good policies. Built upon the memory, we construct a complementary learning model via a dynamic hybrid control unifying model-based, episodic and habitual learning into a single architecture. Experiments demonstrate that our model allows significantly faster and better learning than other strong reinforcement learning agents across a variety of environments including stochastic and non-Markovian settings.

----

## [2319] FedDR - Randomized Douglas-Rachford Splitting Algorithms for Nonconvex Federated Composite Optimization

**Authors**: *Quoc Tran-Dinh, Nhan H. Pham, Dzung T. Phan, Lam M. Nguyen*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe7ee8fc1959cc7214fa21c4840dff0a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe7ee8fc1959cc7214fa21c4840dff0a-Abstract.html)

**Abstract**:

We develop two new algorithms, called, FedDR and asyncFedDR, for solving a fundamental nonconvex composite optimization problem in federated learning. Our algorithms rely on a novel combination between a nonconvex Douglas-Rachford splitting method, randomized block-coordinate strategies, and asynchronous im- plementation. They can also handle convex regularizers. Unlike recent methods in the literature, e.g., FedSplit and FedPD, our algorithms update only a subset of users at each communication round, and possibly in an asynchronous manner, making them more practical. These new algorithms can handle statistical and sys- tem heterogeneity, which are the two main challenges in federated learning, while achieving the best known communication complexity. In fact, our new algorithms match the communication complexity lower bound up to a constant factor under standard assumptions. Our numerical experiments illustrate the advantages of our methods over existing algorithms on synthetic and real datasets.

----

## [2320] Adversarial Examples Make Strong Poisons

**Authors**: *Liam Fowl, Micah Goldblum, Ping-yeh Chiang, Jonas Geiping, Wojciech Czaja, Tom Goldstein*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fe87435d12ef7642af67d9bc82a8b3cd-Abstract.html)

**Abstract**:

The adversarial machine learning literature is largely partitioned into evasion attacks on testing data and poisoning attacks on training data.  In this work, we show that adversarial examples, originally intended for attacking pre-trained models, are even more effective for data poisoning than recent methods designed specifically for poisoning. In fact, adversarial examples with labels re-assigned by the crafting network remain effective for training, suggesting that adversarial examples contain useful semantic content, just with the "wrong" labels (according to a network, but not a human). Our method, adversarial poisoning, is substantially more effective than existing poisoning methods for secure dataset release, and we release a poisoned version of ImageNet, ImageNet-P, to encourage research into the strength of this form of data obfuscation.

----

## [2321] Coresets for Decision Trees of Signals

**Authors**: *Ibrahim Jubran, Ernesto Evgeniy Sanches Shayda, Ilan Newman, Dan Feldman*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fea9c11c4ad9a395a636ed944a28b51a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fea9c11c4ad9a395a636ed944a28b51a-Abstract.html)

**Abstract**:

A $k$-decision tree $t$ (or $k$-tree) is a recursive partition of a matrix (2D-signal) into $k\geq 1$ block matrices (axis-parallel rectangles, leaves) where each rectangle is assigned a real label. Its regression or classification loss to a given matrix $D$ of $N$ entries (labels) is the sum of squared differences over every label in $D$ and its assigned label by $t$.Given an error parameter $\varepsilon\in(0,1)$, a $(k,\varepsilon)$-coreset $C$ of $D$ is a small summarization that provably approximates this loss to \emph{every} such tree, up to a multiplicative factor of $1\pm\varepsilon$. In particular, the optimal $k$-tree of $C$ is a $(1+\varepsilon)$-approximation to the optimal $k$-tree of $D$.We provide the first algorithm that outputs such a $(k,\varepsilon)$-coreset for \emph{every} such matrix $D$. The size $|C|$ of the coreset is polynomial in $k\log(N)/\varepsilon$, and its construction takes $O(Nk)$ time.This is by forging a link between decision trees from machine learning -- to partition trees in computational geometry. Experimental results on \texttt{sklearn} and \texttt{lightGBM} show that applying our coresets on real-world data-sets boosts the computation time of random forests and their parameter tuning by up to x$10$, while keeping similar accuracy. Full open source code is provided.

----

## [2322] Local plasticity rules can learn deep representations using self-supervised contrastive predictions

**Authors**: *Bernd Illing, Jean Ventura, Guillaume Bellec, Wulfram Gerstner*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/feade1d2047977cd0cefdafc40175a99-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/feade1d2047977cd0cefdafc40175a99-Abstract.html)

**Abstract**:

Learning in the brain is poorly understood and learning rules that respect biological constraints, yet yield deep hierarchical representations, are still unknown. Here, we propose a learning rule that takes inspiration from neuroscience and recent advances in self-supervised deep learning. Learning minimizes a simple layer-specific loss function and does not need to back-propagate error signals within or between layers. Instead, weight updates follow a local, Hebbian, learning rule that only depends on pre- and post-synaptic neuronal activity, predictive dendritic input and widely broadcasted modulation factors which are identical for large groups of neurons. The learning rule applies contrastive predictive learning to a causal, biological setting using saccades (i.e. rapid shifts in gaze direction). We find that networks trained with this self-supervised and local rule build deep hierarchical representations of images, speech and video.

----

## [2323] MobTCast: Leveraging Auxiliary Trajectory Forecasting for Human Mobility Prediction

**Authors**: *Hao Xue, Flora D. Salim, Yongli Ren, Nuria Oliver*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/fecf2c550171d3195c879d115440ae45-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/fecf2c550171d3195c879d115440ae45-Abstract.html)

**Abstract**:

Human mobility prediction is a core functionality in many location-based services and applications. However, due to the sparsity of mobility data, it is not an easy task to predict future POIs (place-of-interests) that are going to be visited. In this paper, we propose MobTCast, a Transformer-based context-aware network for mobility prediction. Specifically, we explore the influence of four types of context in mobility prediction: temporal, semantic, social, and geographical contexts. We first design a base mobility feature extractor using the Transformer architecture, which takes both the history POI sequence and the semantic information as input. It handles both the temporal and semantic contexts. Based on the base extractor and the social connections of a user, we employ a self-attention module to model the influence of the social context. Furthermore, unlike existing methods, we introduce a location prediction branch in MobTCast as an auxiliary task to model the geographical context and predict the next location. Intuitively, the geographical distance between the location of the predicted POI and the predicted location from the auxiliary branch should be as close as possible. To reflect this relation, we design a consistency loss to further improve the POI prediction performance. In our experimental results, MobTCast outperforms other state-of-the-art next POI prediction methods. Our approach illustrates the value of including different types of context in next POI prediction.

----

## [2324] Early Convolutions Help Transformers See Better

**Authors**: *Tete Xiao, Mannat Singh, Eric Mintun, Trevor Darrell, Piotr Dollár, Ross B. Girshick*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ff1418e8cc993fe8abcfe3ce2003e5c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ff1418e8cc993fe8abcfe3ce2003e5c5-Abstract.html)

**Abstract**:

Vision transformer (ViT) models exhibit substandard optimizability. In particular, they are sensitive to the choice of optimizer (AdamW vs. SGD), optimizer hyperparameters, and training schedule length. In comparison, modern convolutional neural networks are easier to optimize. Why is this the case? In this work, we conjecture that the issue lies with the patchify stem of ViT models, which is implemented by a stride-p p×p convolution (p = 16 by default) applied to the input image. This large-kernel plus large-stride convolution runs counter to typical design choices of convolutional layers in neural networks. To test whether this atypical design choice causes an issue, we analyze the optimization behavior of ViT models with their original patchify stem versus a simple counterpart where we replace the ViT stem by a small number of stacked stride-two 3×3 convolutions. While the vast majority of computation in the two ViT designs is identical, we find that this small change in early visual processing results in markedly different training behavior in terms of the sensitivity to optimization settings as well as the final model accuracy. Using a convolutional stem in ViT dramatically increases optimization stability and also improves peak performance (by ∼1-2% top-1 accuracy on ImageNet-1k), while maintaining flops and runtime. The improvement can be observed across the wide spectrum of model complexities (from 1G to 36G flops) and dataset scales (from ImageNet-1k to ImageNet-21k). These findings lead us to recommend using a standard, lightweight convolutional stem for ViT models in this regime as a more robust architectural choice compared to the original ViT model design.

----

## [2325] Error Compensated Distributed SGD Can Be Accelerated

**Authors**: *Xun Qian, Peter Richtárik, Tong Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ff1ced3097ccf17c1e67506cdad9ac95-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ff1ced3097ccf17c1e67506cdad9ac95-Abstract.html)

**Abstract**:

Gradient compression is a recent and increasingly popular technique for reducing the communication cost in distributed training of large-scale machine learning models. In this work we focus on developing efficient distributed methods that can work for any compressor satisfying a certain contraction property, which includes both unbiased (after appropriate scaling) and biased compressors such as RandK and TopK. Applied naively, gradient compression introduces errors that either slow down convergence or lead to divergence. A popular technique designed to tackle this issue is error compensation/error feedback. Due to the difficulties associated with analyzing biased compressors, it is not known whether gradient compression with error compensation can be combined with acceleration. In this work, we show for the first time that error compensated gradient compression methods can be accelerated. In particular, we propose and study the error compensated loopless Katyusha method, and establish an accelerated linear convergence rate under standard assumptions. We show through numerical experiments that the proposed method converges with substantially fewer communication rounds than previous error compensated algorithms.

----

## [2326] InfoGCL: Information-Aware Graph Contrastive Learning

**Authors**: *Dongkuan Xu, Wei Cheng, Dongsheng Luo, Haifeng Chen, Xiang Zhang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ff1e68e74c6b16a1a7b5d958b95e120c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ff1e68e74c6b16a1a7b5d958b95e120c-Abstract.html)

**Abstract**:

Various graph contrastive learning models have been proposed to improve the performance of tasks on graph datasets in recent years. While effective and prevalent, these models are usually carefully customized. In particular, despite all recent work create two contrastive views, they differ in a variety of view augmentations, architectures, and objectives. It remains an open question how to build your graph contrastive learning model from scratch for particular graph tasks and datasets. In this work, we aim to fill this gap by studying how graph information is transformed and transferred during the contrastive learning process, and proposing an information-aware graph contrastive learning framework called InfoGCL. The key to the success of the proposed framework is to follow the Information Bottleneck principle to reduce the mutual information between contrastive parts while keeping task-relevant information intact at both the levels of the individual module and the entire framework so that the information loss during graph representation learning can be minimized. We show for the first time that all recent graph contrastive learning methods can be unified by our framework. Based on theoretical and empirical analysis on benchmark graph datasets, we show that InfoGCL achieves state-of-the-art performance in the settings of both graph classification and node classification tasks.

----

## [2327] Meta-Learning for Relative Density-Ratio Estimation

**Authors**: *Atsutoshi Kumagai, Tomoharu Iwata, Yasuhiro Fujiwara*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ff49cc40a8890e6a60f40ff3026d2730-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ff49cc40a8890e6a60f40ff3026d2730-Abstract.html)

**Abstract**:

The ratio of two probability densities, called a density-ratio, is a vital quantity in machine learning. In particular, a relative density-ratio, which is a bounded extension of the density-ratio, has received much attention due to its stability and has been used in various applications such as outlier detection and dataset comparison. Existing methods for (relative) density-ratio estimation (DRE) require many instances from both densities. However, sufficient instances are often unavailable in practice. In this paper, we propose a meta-learning method for relative DRE, which estimates the relative density-ratio from a few instances by using knowledge in related datasets. Specifically, given two datasets that consist of a few instances, our model extracts the datasets' information by using neural networks and uses it to obtain instance embeddings appropriate for the relative DRE. We model the relative density-ratio by a linear model on the embedded space, whose global optimum solution can be obtained as a closed-form solution. The closed-form solution enables fast and effective adaptation to a few instances, and its differentiability enables us to train our model such that the expected test error for relative DRE can be explicitly minimized after adapting to a few instances. We empirically demonstrate the effectiveness of the proposed method by using three problems: relative DRE, dataset comparison, and outlier detection.

----

## [2328] Overcoming the curse of dimensionality with Laplacian regularization in semi-supervised learning

**Authors**: *Vivien Cabannes, Loucas Pillaud-Vivien, Francis R. Bach, Alessandro Rudi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ff4d5fbbafdf976cfdc032e3bde78de5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ff4d5fbbafdf976cfdc032e3bde78de5-Abstract.html)

**Abstract**:

As annotations of data can be scarce in large-scale practical problems, leveraging unlabelled examples is one of the most important aspects of machine learning. This is the aim of semi-supervised learning. To benefit from the access to unlabelled data, it is natural to diffuse smoothly knowledge of labelled data to unlabelled one. This induces to the use of Laplacian regularization. Yet, current implementations of Laplacian regularization suffer from several drawbacks, notably the well-known curse of dimensionality. In this paper, we design a new class of algorithms overcoming this issue, unveiling a large body of spectral filtering methods. Additionally, we provide a statistical analysis showing that our estimators exhibit desirable behaviors. They are implemented through (reproducing) kernel methods, for which we provide realistic computational guidelines in order to make our method usable with large amounts of data.

----

## [2329] Unlabeled Principal Component Analysis

**Authors**: *Yunzhen Yao, Liangzu Peng, Manolis C. Tsakiris*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ff8c1a3bd0c441439a0a081e560c85fc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ff8c1a3bd0c441439a0a081e560c85fc-Abstract.html)

**Abstract**:

We introduce robust principal component analysis from a data matrix in which the entries of its columns have been corrupted by permutations, termed Unlabeled Principal Component Analysis (UPCA). Using algebraic geometry, we establish that UPCA is a well-defined algebraic problem in the sense that the only matrices of minimal rank that agree with the given data are row-permutations of the ground-truth matrix, arising as the unique solutions of a polynomial system of equations. Further, we propose an efficient two-stage algorithmic pipeline for UPCA suitable for the practically relevant case where only a fraction of the data have been permuted. Stage-I employs outlier-robust PCA methods to estimate the ground-truth column-space. Equipped with the column-space, Stage-II applies recent methods for unlabeled sensing to restore the permuted data. Experiments on synthetic data, face images, educational and medical records reveal the potential of UPCA for applications such as data privatization and record linkage.

----

## [2330] Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data

**Authors**: *Andrew Jesson, Panagiotis Tigas, Joost van Amersfoort, Andreas Kirsch, Uri Shalit, Yarin Gal*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ffa4eb0e32349ae57f7a0ee8c7cd7c11-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ffa4eb0e32349ae57f7a0ee8c7cd7c11-Abstract.html)

**Abstract**:

Estimating personalized treatment effects from high-dimensional observational data is essential in situations where experimental designs are infeasible, unethical, or expensive. Existing approaches rely on fitting deep models on outcomes observed for treated and control populations. However, when measuring individual outcomes is costly, as is the case of a tumor biopsy, a sample-efficient strategy for acquiring each result is required. Deep Bayesian active learning provides a framework for efficient data acquisition by selecting points with high uncertainty. However, existing methods bias training data acquisition towards regions of non-overlapping support between the treated and control populations. These are not sample-efficient because the treatment effect is not identifiable in such regions. We introduce causal, Bayesian acquisition functions grounded in information theory that bias data acquisition towards regions with overlapping support to maximize sample efficiency for learning personalized treatment effects. We demonstrate the performance of the proposed acquisition strategies on synthetic and semi-synthetic datasets IHDP and CMNIST and their extensions, which aim to simulate common dataset biases and pathologies.

----

## [2331] Scalable Rule-Based Representation Learning for Interpretable Classification

**Authors**: *Zhuo Wang, Wei Zhang, Ning Liu, Jianyong Wang*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ffbd6cbb019a1413183c8d08f2929307-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ffbd6cbb019a1413183c8d08f2929307-Abstract.html)

**Abstract**:

Rule-based models, e.g., decision trees, are widely used in scenarios demanding high model interpretability for their transparent inner structures and good model expressivity. However, rule-based models are hard to optimize, especially on large data sets, due to their discrete parameters and structures. Ensemble methods and fuzzy/soft rules are commonly used to improve performance, but they sacrifice the model interpretability. To obtain both good scalability and interpretability, we propose a new classifier, named Rule-based Representation Learner (RRL), that automatically learns interpretable non-fuzzy rules for data representation and classification. To train the non-differentiable RRL effectively, we project it to a continuous space and propose a novel training method, called Gradient Grafting, that can directly optimize the discrete model using gradient descent. An improved design of logical activation functions is also devised to increase the scalability of RRL and enable it to discretize the continuous features end-to-end. Exhaustive experiments on nine small and four large data sets show that RRL outperforms the competitive interpretable approaches and can be easily adjusted to obtain a trade-off between classification accuracy and model complexity for different scenarios. Our code is available at: https://github.com/12wang3/rrl.

----

## [2332] Bridging Non Co-occurrence with Unlabeled In-the-wild Data for Incremental Object Detection

**Authors**: *Na Dong, Yongqiang Zhang, Mingli Ding, Gim Hee Lee*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ffc58105bf6f8a91aba0fa2d99e6f106-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ffc58105bf6f8a91aba0fa2d99e6f106-Abstract.html)

**Abstract**:

Deep networks have shown remarkable results in the task of object detection. However, their performance suffers critical drops when they are subsequently trained on novel classes without any sample from the base classes originally used to train the model. This phenomenon is known as catastrophic forgetting. Recently, several incremental learning methods are proposed to mitigate catastrophic forgetting for object detection. Despite the effectiveness, these methods require co-occurrence of the unlabeled base classes in the training data of the novel classes. This requirement is impractical in many real-world settings since the base classes do not necessarily co-occur with the novel classes. In view of this limitation, we consider a more practical setting of complete absence of co-occurrence of the base and novel classes for the object detection task. We propose the use of unlabeled in-the-wild data to bridge the non co-occurrence caused by the missing base classes during the training of additional novel classes. To this end, we introduce a blind sampling strategy based on the responses of the base-class model and pre-trained novel-class model to select a smaller relevant dataset from the large in-the-wild dataset for incremental learning. We then design a dual-teacher distillation framework to transfer the knowledge distilled from the base- and novel-class teacher models to the student model using the sampled in-the-wild data. Experimental results on the PASCAL VOC and MS COCO datasets show that our proposed method significantly outperforms other state-of-the-art class-incremental object detection methods when there is no co-occurrence between the base and novel classes during training.

----

## [2333] A Regression Approach to Learning-Augmented Online Algorithms

**Authors**: *Keerti Anand, Rong Ge, Amit Kumar, Debmalya Panigrahi*

**Conference**: *nips 2021*

**URL**: [https://proceedings.neurips.cc/paper/2021/hash/ffeed84c7cb1ae7bf4ec4bd78275bb98-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ffeed84c7cb1ae7bf4ec4bd78275bb98-Abstract.html)

**Abstract**:

The emerging field of learning-augmented online algorithms uses ML techniques to predict future input parameters and thereby improve the performance of online algorithms. Since these parameters are, in general, real-valued functions, a natural approach is to use regression techniques to make these predictions. We introduce this approach in this paper, and explore it in the context of a general online search framework that captures classic problems like (generalized) ski rental, bin packing, minimum makespan scheduling, etc. We show nearly tight bounds on the sample complexity of this regression problem, and extend our results to the agnostic setting. From a technical standpoint, we show that the key is to incorporate online optimization benchmarks in the design of the loss function for the regression problem, thereby diverging from the use of off-the-shelf regression tools with standard bounds on statistical error.

----



[Go to the previous page](NIPS-2021-list11.md)

[Go to the catalog section](README.md)